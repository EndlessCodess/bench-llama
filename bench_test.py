#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# bench_test.py —— 单用例编排器（benchmark 三层结构的中间层）
# =============================================================================
# 职责：把一个"用例"（如 LC-001）从配置翻译成一次可执行的基准会话。
#
# 三层调用链（自上而下）：
#   run_suite.py ──每 job 调用一次──▶ 本脚本（每 case 一个 bench_run 进程）
#     bench_test.py ────────────────▶ build/bench_run（C++ 执行内核）
#
# 单个用例的执行流程（run_session）：
#   ① 读 case 配置（configs/*.json）+ run-manifest.json（105 轮选择）+
#      metadata.jsonl（prompt 元数据），生成 plan.tsv（每轮一行）；
#   ② 启动一次 bench_run 子进程（模型只加载一次，105 轮 = 5 预热 + 100 正式
#      = 50 条池按序循环 2 遍，全部单轮、每轮间 KV 清空）；
#      --rounds N 时上述完整会话重复 N 轮（每轮重启进程、模型重载），
#      所有轮的正式样本合并统计（N×100 个/用例）—— 多次重复测量降低硬件噪声；
#   ③ 双线程实时解析子进程 stdout 的 JSONL 事件流（run_start/result/...），
#      同时启动监控线程定时采样进程 RSS 与 GPU 显存（NVML）；
#   ④ 每轮：run_start 事件到达时重置显存采样窗口并读"本轮基线"，
#      result 事件配对实测耗时与墙钟时间；
#   ⑤ 落盘：每轮一条记录（runs.jsonl，边跑边写，中断不丢），
#      汇总均值/标准差/极值（summary.json/csv），终端打印双语汇总表。
#
# 与 SGLang/vLLM 侧的对齐语义：
#   - 常驻进程：一个用例 = 一次进程、多次请求（引擎形态对齐）；
#   - 请求式：每轮独立单轮、KV 清空（请求形态对齐）；
#   - 同输入：同一份 prompt 文件、同一 chat 模板口径（token 数对齐）；
#   - warmup 吸收冷启动（模型加载/CUDA kernel 初始化），正式轮测稳态。
#
# 典型用法（单个用例）：
#   CUDA_VISIBLE_DEVICES=4 python3 bench_test.py \
#     --bench-run build/bench_run --model <模型> --config configs/fair-baseline.json \
#     --platform cuda --device-index 4 --case LC-001
# 常用开关：--warmup-runs/--formal-runs（覆盖轮数）、--dry-run（只校验不跑）、
#           --list-cases（列用例）、--timeout（每轮超时）、--fail-fast（失败即止）。
# =============================================================================

import argparse
import csv
import ctypes
import ctypes.util
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform as host_platform
import re
import shlex
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "qwen3.5-9b"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_BENCHMARK_CONFIG = PROJECT_ROOT / "configs" / "fair-baseline.json"
DEFAULT_PLATFORM_CONFIG = PROJECT_ROOT / "configs" / "platforms.json"

# One row per run: measured values (_measured) come from the bench_run JSONL
# result events (or from the external monitor), derived values (_est) are
# computed here. All fields are kept so runs.csv is self-describing.
# 每轮一条记录的全部字段（runs.jsonl / runs.csv 的列）：
#   *_measured  = 实测值（bench_run result 事件的耗时，或监控线程采到的 RSS/显存）；
#   *_est       = 由实测值在本脚本推导的估计值（如端到端吞吐率）；
#   保留全部字段，使 runs.csv 自描述、无需查代码即知每列含义。
RUN_FIELDS = [
    "case_id",
    "dataset",
    "status",
    "phase",
    "phase_run",
    "manifest_run",
    "round",
    "platform",
    "physical_device_index",
    "backend_device",
    "source_id",
    "pool_index",
    "domain",
    "difficulty",
    "prompt_file",
    "prompt_sha256",
    "target_prompt_tokens",
    "actual_prompt_tokens",
    "output_tokens_requested",
    "prompt_eval_ms_measured",
    "generation_eval_ms_measured",
    "ttft_ms_measured",
    "tpot_ms_measured",
    "itl_ms_measured",
    "e2e_ms_measured",
    "prompt_tps_measured",
    "generation_tps_measured",
    "tokens_generated",
    "draft_tokens_measured",
    "draft_accepted_measured",
    "draft_acceptance_rate_measured",
    "finish_reason",
    "output_sha256",
    "wall_time_s_measured",
    "throughput_tok_s_est",
    "peak_process_rss_mib_measured",
    "device_memory_baseline_mib_measured",
    "peak_device_memory_mib_measured",
    "device_memory_delta_mib_measured",
    "peak_gpu_utilization_pct_measured",
    "average_gpu_utilization_pct_measured",
    "returncode",
    "timed_out",
    "monitor_error",
    "error",
    "command",
]

# Summary table columns: the five headline metrics, one row per case,
# with bilingual "Chinese/English" headers.
# 汇总表列：一用例一行，五个核心指标（均值），表头为双语"中文/English"。
SUMMARY_FIELDS = [
    "case_id",
    "dataset",
    "status",
    "platform",
    "physical_device_index",
    "backend_device",
    "target_prompt_tokens",
    "output_tokens_requested",
    "平均输入长度(tokens)/Average Input Length(tokens)",
    "平均实际输出(tokens)/Average Generated Tokens",
    "warmup_runs",
    "formal_runs_requested",
    "formal_runs_ok",
    "formal_runs_failed",
    "首令牌延迟(ms)/TTFT(ms)",
    "每令牌时间(ms/token)/TPOT(ms/token)",
    "令牌间延迟(ms/token)/ITL(ms/token)",
    "端到端耗时(s)/E2E(s)",
    "吞吐率(token/s)/Throughput(token/s)",
    "峰值显存(GB)/Peak Memory(GB)",
    "GPU计算单元峰值利用率(%)/Peak GPU Compute Utilization(%)",
    "GPU计算单元平均利用率(%)/Average GPU Compute Utilization(%)",
    "草稿接受率(%)/Draft Acceptance Rate(%)",
]


# 备选显存读取器（NVML 不可用的平台，如 CANN/昇腾）：执行外部命令，
# {device} 占位符替换为卡号，从 stdout 提取第一个数字作为"当前已用显存（MiB）"。
# 只用于资源监控，不参与计时。
def read_memory_command(command_text, device_index):
    command = [part.format(device=device_index) for part in shlex.split(command_text)]
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    match = re.search(r"[-+]?[0-9]*\.?[0-9]+", result.stdout)
    if result.returncode != 0 or not match:
        raise RuntimeError("memory command did not return a numeric MiB value")
    return float(match.group(0))


# NVML 显存读取器（ctypes 直调 libnvidia-ml，不依赖 pynvml 包）：
# 读取"整块物理卡"的已用显存 —— 是整个设备的使用量，不区分进程，
# 因此结果侧用"每轮窗口峰值 − run_start 基线"的差值估计单轮 KV + 计算缓冲。
class NvmlMemoryReader:
    class Memory(ctypes.Structure):
        _fields_ = [
            ("total", ctypes.c_ulonglong),
            ("free", ctypes.c_ulonglong),
            ("used", ctypes.c_ulonglong),
        ]

    class Utilization(ctypes.Structure):
        _fields_ = [
            ("gpu", ctypes.c_uint),
            ("memory", ctypes.c_uint),
        ]

    def __init__(self, device_index):
        lib_name = ctypes.util.find_library("nvidia-ml") or "libnvidia-ml.so.1"
        self.lib = ctypes.CDLL(lib_name)
        self.lib.nvmlInit_v2.restype = ctypes.c_int
        self.lib.nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
        self.lib.nvmlDeviceGetMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(self.Memory)]
        self.lib.nvmlDeviceGetMemoryInfo.restype = ctypes.c_int
        self.lib.nvmlDeviceGetUtilizationRates.argtypes = [ctypes.c_void_p, ctypes.POINTER(self.Utilization)]
        self.lib.nvmlDeviceGetUtilizationRates.restype = ctypes.c_int
        self.lib.nvmlShutdown.restype = ctypes.c_int

        self._check(self.lib.nvmlInit_v2(), "nvmlInit_v2")
        self.handle = ctypes.c_void_p()
        self._check(
            self.lib.nvmlDeviceGetHandleByIndex_v2(device_index, ctypes.byref(self.handle)),
            "nvmlDeviceGetHandleByIndex_v2",
        )

    @staticmethod
    def _check(status, name):
        if status != 0:
            raise RuntimeError("{} failed with NVML status {}".format(name, status))

    def read_mib(self):
        memory = self.Memory()
        self._check(self.lib.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(memory)), "nvmlDeviceGetMemoryInfo")
        return memory.used / (1024.0 * 1024.0)

    def read_gpu_utilization_pct(self):
        utilization = self.Utilization()
        self._check(
            self.lib.nvmlDeviceGetUtilizationRates(self.handle, ctypes.byref(utilization)),
            "nvmlDeviceGetUtilizationRates",
        )
        return float(utilization.gpu)

    def close(self):
        self.lib.nvmlShutdown()


# 进程资源监视器：后台线程每 interval 秒采样一次
#   - RSS：/proc/<pid>/status 的 VmRSS（读取失败退回 ps -o rss=）；
#   - 显存：NVML（NvmlMemoryReader）或外部命令（read_memory_command）；
# 只保留窗口内峰值（peak_rss_mib / peak_device_mib）。
# reset_window() 在每轮 run_start 事件时由主线程调用，使采样窗口恰好覆盖一轮推理，
# 窗口峰值 = 本轮资源占用；本轮基线（窗口起点显存）由主线程 reset 之后立即读取
# （见 run_session ④），二者之差 = 本轮 KV 缓存 + 计算缓冲的增量。
class ProcessMonitor:
    def __init__(self, proc, interval, memory_reader=None, memory_command=None, device_index=0):
        self.proc = proc
        self.interval = interval
        self.memory_reader = memory_reader
        self.memory_command = memory_command
        self.device_index = device_index
        self.peak_rss_mib = 0.0
        self.peak_device_mib = 0.0
        self.peak_gpu_utilization_pct = None
        self.gpu_utilization_sum_pct = 0.0
        self.gpu_utilization_samples = 0
        self.error = ""
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval * 4))

    def reset_window(self):
        """Reset the peak trackers; called at every run_start event so each
        window covers exactly one run (baseline for the delta is read by the
        main thread right after this, before the run starts decoding)."""
        self.peak_rss_mib = 0.0
        self.peak_device_mib = 0.0
        self.peak_gpu_utilization_pct = None
        self.gpu_utilization_sum_pct = 0.0
        self.gpu_utilization_samples = 0

    def _read_rss_mib(self):
        status_path = Path("/proc") / str(self.proc.pid) / "status"
        if status_path.exists():
            for line in status_path.read_text(errors="replace").splitlines():
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
            return 0.0

        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(self.proc.pid)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        value = result.stdout.strip()
        return float(value) / 1024.0 if value else 0.0

    def _read_device_mib(self):
        if self.memory_reader is not None:
            return self.memory_reader.read_mib()
        if self.memory_command:
            return read_memory_command(self.memory_command, self.device_index)
        return 0.0

    def _read_gpu_utilization_pct(self):
        if self.memory_reader is not None and hasattr(self.memory_reader, "read_gpu_utilization_pct"):
            return self.memory_reader.read_gpu_utilization_pct()
        return None

    def read_device_mib_now(self):
        """Baseline read from the main thread (used at run_start)."""
        return self._read_device_mib()

    def _sample(self):
        self.peak_rss_mib = max(self.peak_rss_mib, self._read_rss_mib())
        self.peak_device_mib = max(self.peak_device_mib, self._read_device_mib())
        utilization = self._read_gpu_utilization_pct()
        if utilization is not None:
            if self.peak_gpu_utilization_pct is None:
                self.peak_gpu_utilization_pct = utilization
            else:
                self.peak_gpu_utilization_pct = max(self.peak_gpu_utilization_pct, utilization)
            self.gpu_utilization_sum_pct += utilization
            self.gpu_utilization_samples += 1

    def average_gpu_utilization_pct(self):
        if self.gpu_utilization_samples == 0:
            return None
        return self.gpu_utilization_sum_pct / self.gpu_utilization_samples

    def _run(self):
        try:
            while not self._stop.is_set() and self.proc.poll() is None:
                self._sample()
                self._stop.wait(self.interval)
            self._sample()
        except Exception as exc:
            self.error = str(exc)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the formal bench_run single-turn benchmark")
    parser.add_argument("--bench-run", default=os.environ.get("BENCH_RUN"), help="path to the bench_run executable (or BENCH_RUN)")  # bench_run 可执行文件路径（默认取环境变量 BENCH_RUN）
    parser.add_argument("--llama-server", help="llama-server executable used by the server benchmark adapter")
    parser.add_argument("--llama-cli", default=os.environ.get("LLAMA_CLI"), help=argparse.SUPPRESS)  # 已废弃别名：--llama-cli → --bench-run
    parser.add_argument("--model", default=os.environ.get("LLAMA_MODEL"), help="path to the GGUF model")  # GGUF 模型文件路径（默认取环境变量 LLAMA_MODEL）
    parser.add_argument("--draft-model", help="path to a DFlash draft GGUF")
    parser.add_argument("--config", default=str(DEFAULT_BENCHMARK_CONFIG), help="benchmark case configuration")  # 用例配置文件（默认 configs/fair-baseline.json）
    parser.add_argument("--platform-config", default=str(DEFAULT_PLATFORM_CONFIG), help="platform profiles")  # 平台档案文件（默认 configs/platforms.json）
    parser.add_argument("--processed-dir", default=str(DEFAULT_PROCESSED_DIR))  # 预处理数据目录（prompts、metadata.jsonl 等）
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))  # 结果根目录（每次运行在其下建时间戳子目录）
    parser.add_argument("--output-dir", help="exact output directory instead of a timestamped directory")  # 精确指定结果目录，替代自动生成的时间戳目录
    parser.add_argument("--platform", default="auto", help="platform profile name or auto")  # 平台档案名（cuda/cann/metal/cpu），auto 为自动检测
    parser.add_argument("--device-index", type=int, default=0, help="physical accelerator index")  # 物理加速卡索引
    parser.add_argument("--backend-device", help="llama.cpp device after visibility filtering")  # 环境变量过滤后传给 llama.cpp 的设备名（如 CUDA0）
    parser.add_argument("--case", action="append", default=[], help="case id or comma-separated case ids")  # 只跑指定用例（如 LC-001，可重复或逗号分隔）
    parser.add_argument("--warmup-runs", type=int, help="override manifest warmup count")  # 覆盖 manifest 中的预热轮数（冒烟测试时可设 0）
    parser.add_argument("--formal-runs", type=int, help="override manifest formal count")  # 覆盖 manifest 中的正式轮数
    parser.add_argument("--rounds", type=int, default=1, help="number of full rounds per case (default 1)")  # 每 case 完整跑几轮：每轮 = 一次全新 bench_run 进程（模型重载 + 105 轮）；10 轮 = 全套重复 10 次 = 29,400 次总推理，正式样本合并统计（1000 个/用例），多次重复测量降低硬件噪声
    parser.add_argument("--n-ctx", type=int)  # 上下文长度覆盖（默认 262144）
    parser.add_argument("--n-batch", type=int)  # batch 大小覆盖（默认 2048）
    parser.add_argument("--n-ubatch", type=int)  # micro-batch 大小覆盖（默认 512）
    parser.add_argument("--n-gpu-layers", type=int)  # 卸载到 GPU 的层数覆盖（默认 99）
    parser.add_argument("--parallel", type=int)  # 并发序列数覆盖（必须为 1，单轮单进程设计）
    parser.add_argument("--threads", type=int)  # CPU 线程数（默认交给 llama.cpp 自动选择）
    parser.add_argument("--type-k")  # K 缓存精度覆盖（f16/q8_0/q4_0）
    parser.add_argument("--type-v")  # V 缓存精度覆盖（f16/q8_0/q4_0）
    parser.add_argument("--flash-attn", choices=["on", "off", "auto"])  # Flash Attention 开关（LC-021..024 关掉）
    parser.add_argument("--poll", type=int)  # poll 间隔覆盖（LC-025..028 设为 0）
    parser.add_argument("--temperature", type=float)  # 采样温度覆盖（默认 0，确定性输出）
    parser.add_argument("--seed", type=int)  # 随机种子覆盖（默认 42，配合 temp 0 保证可复现）
    parser.add_argument("--draft-n-max", type=int)
    parser.add_argument("--draft-n-gpu-layers", type=int)
    parser.add_argument("--ignore-eos", dest="ignore_eos", action="store_true")
    parser.add_argument("--no-ignore-eos", dest="ignore_eos", action="store_false")
    parser.set_defaults(ignore_eos=None)
    parser.add_argument("--monitor-interval", type=float, default=0.1)  # 显存/RSS 采样间隔（秒）
    parser.add_argument(
        "--memory-command",
        help="command printing used device memory in MiB; {device} is replaced",  # 自定义显存查询命令，输出 MiB 数值；{device} 会被替换为卡号
    )
    parser.add_argument("--timeout", type=float, default=0.0, help="timeout per run in seconds (0 = no timeout)")  # 每轮超时秒数（0 = 不超时；超时则该轮标记失败并终止进程）
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")  # 附加环境变量（可重复）
    parser.add_argument("--extra-arg", action="append", default=[], help="extra bench_run argument")  # 附加 bench_run 参数（透传）
    parser.add_argument("--model-sha256", help="precomputed model SHA-256")  # 预计算的模型 SHA-256，直接写入 manifest 不再重复计算
    parser.add_argument("--draft-model-sha256", help="precomputed draft model SHA-256")
    parser.add_argument("--skip-model-hash", action="store_true")  # 跳过模型 SHA-256 计算（大文件哈希耗时，套件模式下由 run_suite.py 统一算一次）
    parser.add_argument("--skip-prompt-hash-check", action="store_true")  # 跳过 prompt 文件哈希校验
    parser.add_argument("--fail-fast", action="store_true")  # 用例内出现失败轮立即终止该用例（run_suite.py 默认开启）
    parser.add_argument("--dry-run", action="store_true", help="validate and print commands without running")  # 只校验配置并打印命令，不启动推理
    parser.add_argument("--list-cases", action="store_true")  # 列出配置中的全部用例后退出
    return parser.parse_args()


# 解析 KEY=VALUE 形式的重复参数（--env / --extra-arg），返回 dict；
# 不含 "=" 或键为空的项直接报错（宁可失败也不静默忽略）。
def parse_assignments(values, label):
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("{} must use KEY=VALUE: {}".format(label, value))
        key, item = value.split("=", 1)
        if not key:
            raise ValueError("{} has an empty key: {}".format(label, value))
        result[key] = item
    return result


# 加载用例配置文件（configs/fair-baseline.json）：
#   schema_version 必须为 1；用例 id 非空且唯一；
#   prompt_tokens/output_tokens 为正整数（与 manifest 的长度池对应）；
#   runtime 必须包含全部推理参数（n_ctx/n_batch/n_ubatch/n_gpu_layers/parallel/
#   type_k/type_v/flash_attn/poll/temperature/seed），缺任一即报错。
# 返回 (配置文件路径, 配置原文, {case_id: 用例定义})。
def load_benchmark_config(path):
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise ValueError("benchmark config does not exist: {}".format(config_path))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported benchmark config schema: {}".format(config.get("schema_version")))
    cases = {}
    for item in config.get("cases", []):
        case_id = item.get("id")
        if not case_id or case_id in cases:
            raise ValueError("benchmark case ids must be non-empty and unique")
        dataset = item.get("dataset")
        prompt_tokens = None if dataset else int(item["prompt_tokens"])
        output_tokens = int(item["output_tokens"])
        if (prompt_tokens is not None and prompt_tokens <= 0) or output_tokens <= 0:
            raise ValueError("case {} has non-positive token counts".format(case_id))
        if dataset is not None and (not isinstance(dataset, str) or not dataset):
            raise ValueError("case {} has an invalid dataset".format(case_id))
        cases[case_id] = {**item, "prompt_tokens": prompt_tokens, "output_tokens": output_tokens}
    if not cases:
        raise ValueError("benchmark config has no cases")
    runtime = config.get("runtime", {})
    required = (
        "n_ctx", "n_batch", "n_ubatch", "n_gpu_layers", "parallel", "type_k", "type_v",
        "flash_attn", "poll", "temperature", "seed",
    )
    missing = [name for name in required if name not in runtime]
    if missing:
        raise ValueError("benchmark runtime is missing: {}".format(", ".join(missing)))
    return config_path, config, cases


# 加载平台档案文件（configs/platforms.json）：每个平台（cuda/cann/metal/cpu）一份档案，
# 描述 backend_device、环境变量（如 CUDA_VISIBLE_DEVICES）、显存监控方式等。
# 校验：schema_version 为 1、档案非空、每份档案有 backend_device、environment 为对象。
def load_platform_profiles(path):
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise ValueError("platform config does not exist: {}".format(config_path))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported platform config schema: {}".format(config.get("schema_version")))
    profiles = config.get("profiles", {})
    if not profiles:
        raise ValueError("platform config has no profiles")
    for name, profile in profiles.items():
        if not profile.get("backend_device"):
            raise ValueError("platform profile {} has no backend_device".format(name))
        if not isinstance(profile.get("environment", {}), dict):
            raise ValueError("platform profile {} environment must be an object".format(name))
    return config_path, config, profiles


# 运行时参数默认值填充：命令行未指定的推理参数（--n-ctx/--type-k/...）
# 取用例配置 runtime 中的默认值；命令行显式指定的保持优先。
def apply_runtime_defaults(args, runtime):
    mapping = {
        "n_ctx": "n_ctx",
        "n_batch": "n_batch",
        "n_ubatch": "n_ubatch",
        "n_gpu_layers": "n_gpu_layers",
        "parallel": "parallel",
        "type_k": "type_k",
        "type_v": "type_v",
        "flash_attn": "flash_attn",
        "poll": "poll",
        "temperature": "temperature",
        "seed": "seed",
    }
    for argument, key in mapping.items():
        if getattr(args, argument) is None:
            setattr(args, argument, runtime[key])
    if args.draft_n_max is None:
        args.draft_n_max = int(runtime.get("draft_n_max", 15))
    if args.draft_n_gpu_layers is None:
        args.draft_n_gpu_layers = int(runtime.get("draft_n_gpu_layers", 99))
    if args.ignore_eos is None:
        args.ignore_eos = bool(runtime.get("ignore_eos", True))


# 平台解析：--platform auto 时按顺序探测 —— macOS 且档案含 metal → metal；
# 系统有 NVML 库且档案含 cuda → cuda；发现昇腾 SDK 目录 → cann；否则 cpu。
# 显式指定时校验该档案存在，不存在直接报错。
def detect_platform(requested, profiles):
    if requested != "auto":
        if requested not in profiles:
            raise ValueError("unknown platform profile: {}".format(requested))
        return requested
    if sys.platform == "darwin" and "metal" in profiles:
        return "metal"
    if ctypes.util.find_library("nvidia-ml") and "cuda" in profiles:
        return "cuda"
    if "cann" in profiles and any(
        Path(path).exists() for path in ("/usr/local/Ascend", "/usr/local/bin/npu-smi", "/usr/bin/npu-smi")
    ):
        return "cann"
    if "cpu" in profiles:
        return "cpu"
    raise ValueError("auto platform detection found no matching profile")


# 组装平台运行设置：
#   environment：档案环境变量，{device} 占位符替换为实际卡号（如 CUDA_VISIBLE_DEVICES=4）；
#   backend_device：可见性过滤后传给 llama.cpp 的设备名（如 CUDA0）；
#   n_gpu_layers：档案可覆盖默认 -ngl；memory_monitor：nvml 或 none。
def platform_settings(profile, device_index, backend_device, default_n_gpu_layers):
    environment = {
        key: str(value).format(device_index=device_index)
        for key, value in profile.get("environment", {}).items()
    }
    backend_device = backend_device or str(profile["backend_device"]).format(device_index=device_index)
    n_gpu_layers = int(profile.get("n_gpu_layers", default_n_gpu_layers))
    memory_monitor = profile.get("memory_monitor", "none")
    if memory_monitor not in ("none", "nvml"):
        raise ValueError("unsupported memory monitor: {}".format(memory_monitor))
    return environment, backend_device, n_gpu_layers, memory_monitor


# 用例过滤：--case 支持重复与逗号分隔（如 --case LC-001,LC-002）；未指定则全部。
# 按配置定义顺序去重返回；未知用例 id 直接报错。
def select_cases(values, cases):
    if not values:
        return list(cases)
    selected = []
    for value in values:
        for case_id in value.split(","):
            case_id = case_id.strip().upper()
            if not case_id:
                continue
            if case_id not in cases:
                raise ValueError("unknown case: {}".format(case_id))
            if case_id not in selected:
                selected.append(case_id)
    if not selected:
        raise ValueError("no cases selected")
    return selected


# 分块计算文件 SHA-256（8 MiB 块），大文件（模型 ~19 GB）不会一次性读入内存；
# 用于校验模型文件与 prompt 文件未被改动。
def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# 加载三个输入文件：
#   run-manifest.json — 105 轮运行清单（每轮选哪条 prompt、哪个池子、phase 与遍次）；
#   metadata.jsonl    — 每条 prompt 文件的元数据（模板后实际 token 数、SHA-256、
#                       源样本、领域、难度），按 prompt_file 建索引，重复键报错；
#   preparation.json  — 数据切分参数（seed、池大小、抽样方式），原样存入 manifest 供追溯。
def load_inputs(processed_dir):
    manifest_path = processed_dir / "run-manifest.json"
    metadata_path = processed_dir / "metadata.jsonl"
    preparation_path = processed_dir / "preparation.json"
    for path in (manifest_path, metadata_path, preparation_path):
        if not path.is_file():
            raise ValueError("required input does not exist: {}".format(path))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    metadata = {}
    for line_number, line in enumerate(metadata_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        key = item["prompt_file"]
        if key in metadata:
            raise ValueError("duplicate prompt_file in metadata line {}: {}".format(line_number, key))
        metadata[key] = item
    return manifest, preparation, metadata


# 从 manifest 选出本轮实际要跑的轮次：按 phase 分组后各取前 N 条
# （默认 5 预热 + 100 正式；冒烟测试可覆盖更小值）。请求数不能为负、
# 不能超过 manifest 该 phase 的总数。返回 [(phase, phase_run, selection)]，
# phase_run 从 1 重新编号（对应 bench_run 事件里的 phase_run 字段）。
def choose_selections(manifest, warmup_runs, formal_runs, dataset=None):
    selections = manifest["selections"]
    if dataset is not None:
        selections = [item for item in selections if item.get("dataset") == dataset]
    grouped = {
        "warmup": [item for item in selections if item.get("phase") == "warmup"],
        "formal": [item for item in selections if item.get("phase") == "formal"],
    }
    requested = {"warmup": warmup_runs, "formal": formal_runs}
    result = []
    for phase in ("warmup", "formal"):
        count = requested[phase]
        if count < 0:
            raise ValueError("{} run count cannot be negative".format(phase))
        if count > len(grouped[phase]):
            raise ValueError(
                "requested {} {} runs, but manifest only has {}".format(count, phase, len(grouped[phase]))
            )
        for phase_run, item in enumerate(grouped[phase][:count], 1):
            result.append((phase, phase_run, item))
    return result


# 校验每个用例将用到的全部 prompt：文件存在、metadata 有记录、source_id 与 manifest
# 一致、目标长度与用例配置匹配、SHA-256 与 metadata 一致（check_hashes 为 False 时跳过哈希）。
# 保证输入侧与 SGLang/vLLM 完全一致：同一份文件、同一批 token 数。
def prompt_for_case(case, selection):
    if case.get("dataset"):
        if selection.get("dataset") != case["dataset"]:
            raise ValueError("selection dataset does not match case dataset")
        return selection["prompt_file"]
    return selection["prompt_files"][str(case["prompt_tokens"])]


def validate_prompts(processed_dir, cases, selected_cases, selections_by_case, metadata, check_hashes):
    validated = {}
    for case_id in selected_cases:
        case = cases[case_id]
        target = case["prompt_tokens"]
        for _, _, selection in selections_by_case[case_id]:
            if case.get("dataset"):
                if "prompt_file" not in selection:
                    raise ValueError("manifest run {} has no dataset prompt".format(selection.get("run")))
            elif str(target) not in selection.get("prompt_files", {}):
                raise ValueError("manifest run {} has no {} prompt".format(selection.get("run"), target))
            relative = prompt_for_case(case, selection)
            path = processed_dir / relative
            if not path.is_file():
                raise ValueError("prompt does not exist: {}".format(path))
            if relative not in metadata:
                raise ValueError("prompt is missing from metadata: {}".format(relative))
            item = metadata[relative]
            if item["source_id"] != selection["source_id"]:
                raise ValueError("source_id mismatch for {}".format(relative))
            if target is not None and int(item["target_tokens"]) != target:
                raise ValueError("target token mismatch for {}".format(relative))
            if check_hashes and relative not in validated:
                actual = sha256_file(path)
                if actual != item["prompt_sha256"]:
                    raise ValueError("SHA-256 mismatch for {}".format(path))
                validated[relative] = actual


def build_run_plan(case_id, case, selections, metadata, processed_dir, run_dir):
    """Write plan.tsv for one case: one row per run with every metadata column
    bench_run needs. Prompt paths are absolute so bench_run works from any CWD.
    Returns the plan path."""
    # 为一个用例生成 plan.tsv（每轮一行，TAB 分隔）：
    #   列：prompt_file(绝对路径) | case_id | phase | phase_run | manifest_run |
    #       source_id | n_predict(输出 token 数) | expected_tokens(模板后实际 token 数)
    # bench_run 逐行读取执行；prompt 用绝对路径，与 bench_run 的工作目录无关。
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for phase, phase_run, selection in selections:
        relative = prompt_for_case(case, selection)
        prompt_meta = metadata[relative]
        lines.append("\t".join((
            str((processed_dir / relative).resolve()),  # prompt_file
            case_id,
            phase,
            str(phase_run),
            str(selection["run"]),
            selection["source_id"],
            str(case["output_tokens"]),
            str(int(prompt_meta["actual_tokens"])),  # expected_tokens (post-template count)
        )))
    plan_path = run_dir / "plan.tsv"
    plan_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return plan_path


def build_bench_run_command(args, model_path, plan_path, n_gpu_layers):
    """One bench_run process per case; the plan file carries all per-run data
    (prompt, phase, output length, expected tokens)."""
    # 组装 bench_run 启动命令（一个用例 = 一次进程调用）：
    # 推理参数（n_ctx/batch/ubatch/层数/缓存精度/FA/温度/种子/poll）全部来自配置与
    # 命令行覆盖；每轮数据（prompt、阶段、输出长度、期望 token 数）都在 plan.tsv 里，
    # 命令行不需要逐轮参数；--extra-arg 原样透传。
    command = [
        str(Path(args.bench_run)),
        "--model", str(model_path),
        "--plan", str(plan_path),
        "--n-ctx", str(args.n_ctx),
        "--n-batch", str(args.n_batch),
        "--n-ubatch", str(args.n_ubatch),
        "--n-gpu-layers", str(n_gpu_layers),
        "--cache-type-k", args.type_k,
        "--cache-type-v", args.type_v,
        "--flash-attn", args.flash_attn,
        "--temp", str(args.temperature),
        "--seed", str(args.seed),
        "--poll", str(args.poll),
    ]
    if args.threads is not None:
        command.extend(["--threads", str(args.threads)])
    if args.llama_server:
        command.extend(["--llama-server", str(Path(args.llama_server).resolve())])
        if args.draft_model:
            command.extend([
                "--draft-model", str(Path(args.draft_model)),
                "--draft-n-max", str(args.draft_n_max),
                "--draft-ngl", str(args.draft_n_gpu_layers),
            ])
        command.append("--ignore-eos" if args.ignore_eos else "--no-ignore-eos")
    command.extend(args.extra_arg)
    return command


class StreamReader(threading.Thread):
    """Reads one bench_run output stream line by line.

    JSONL event lines are parsed into self.events as (arrival_time, event)
    tuples; wait_event() lets the main thread consume events in order. Every
    raw line is also echoed to the session log.
    """
    # 后台线程逐行读取 bench_run 的一个输出流（stdout=JSONL 事件流，stderr=人类可读进度）：
    #   - 解析为 JSON 且有 event 字段的行，按到达顺序存入 self.events 供主线程消费；
    #   - 非 JSON 行（纯日志/进度）不入列表，但每一行都回显到会话日志（echo），原始输出完整可查；
    #   - 流结束时置 eof 标记并唤醒等待者。
    # wait_event 的语义见该方法注释。

    def __init__(self, stream, condition, echo=None):
        super().__init__(daemon=True)
        self.stream = stream
        self.condition = condition
        self.echo = echo
        self.events = []      # (arrival_time, event_dict) in arrival order
        self.cursor = 0       # index of the first unconsumed event
        self.consumed = []    # every consumed event, for diagnostics
        self.eof = False

    def run(self):
        try:
            for raw in self.stream:
                line = raw.decode("utf-8", errors="replace")
                if self.echo is not None:
                    self.echo.write(line)
                with self.condition:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        event = None
                    if isinstance(event, dict) and event.get("event"):
                        self.events.append((time.monotonic(), event))
                    self.condition.notify_all()
        finally:
            with self.condition:
                self.eof = True
                self.condition.notify_all()
            if self.echo is not None:
                self.echo.flush()

    def wait_event(self, proc, timeout, predicate):
        """Wait until an unconsumed event satisfies predicate.

        Non-matching events are consumed anyway and warn events are relayed to
        stderr so nothing is silently dropped. Returns (arrival_time, event,
        "ok"), or (None, None, "timeout") after `timeout` seconds, or
        (None, None, "exited") when the process ends without the event.
        """
        # 等待"满足谓词的事件"到达（如 run_start / result / error）：
        #   期间到达的不匹配事件照样消费掉（不匹配也返回前的扫描会推进 cursor），
        #   其中 warn 事件转发到 stderr —— 任何信息都不会被静默吞掉；
        #   返回 (事件到达时刻, 事件, 状态)，状态为 "ok" / "timeout"(超时) /
        #   "exited"(进程先退出且未等到该事件)。
        deadline = None if timeout <= 0 else time.monotonic() + timeout
        with self.condition:
            while True:
                while self.cursor < len(self.events):
                    arrival, event = self.events[self.cursor]
                    self.cursor += 1
                    self.consumed.append(event)
                    if predicate(event):
                        return arrival, event, "ok"
                    if event.get("event") == "warn":
                        print("  [warn] {}".format(event.get("message", "")), file=sys.stderr)
                if proc.poll() is not None and self.eof:
                    return None, None, "exited"
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None, None, "timeout"
                    self.condition.wait(timeout=remaining)
                else:
                    self.condition.wait(timeout=0.5)


# 终止 bench_run 进程组：先 SIGTERM 优雅退出（等 10 秒），超时再 SIGKILL。
# 进程组（Popen start_new_session=True 创建）保证连同其子进程一起清理，不留孤儿。
# 注意：绝不使用 pkill 类命令 —— 会误杀外层 shell/监控进程。
def terminate_process(proc):
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def new_run_record(args, case_id, case, phase, phase_run, round_num, selection, prompt_meta, relative,
                   platform_name, backend_device, command_text):
    """The per-run record skeleton; measured fields are filled as events arrive."""
    # 每轮记录骨架（对应 RUN_FIELDS 的全部列）：
    # 身份/来源/长度等静态字段立即填好，status 默认 "failed"；
    # 实测字段（四项耗时、RSS/显存、墙钟）由 run_session 在事件到达后填充；
    # command 全文冗余存档，runs.csv 自描述、无需查代码即知每列含义。
    return {
        "case_id": case_id,
        "dataset": case.get("dataset", ""),
        "status": "failed",
        "phase": phase,
        "phase_run": phase_run,
        "manifest_run": selection["run"],
        "round": round_num,
        "platform": platform_name,
        "physical_device_index": args.device_index,
        "backend_device": backend_device,
        "source_id": selection["source_id"],
        "pool_index": selection["pool_index"],
        "domain": prompt_meta.get("domain", ""),
        "difficulty": prompt_meta.get("difficulty", ""),
        "prompt_file": relative,
        "prompt_sha256": prompt_meta["prompt_sha256"],
        "target_prompt_tokens": case["prompt_tokens"],
        "actual_prompt_tokens": int(prompt_meta["actual_tokens"]),
        "output_tokens_requested": case["output_tokens"],
        "finish_reason": None,
        "output_sha256": None,
        "wall_time_s_measured": None,
        "peak_process_rss_mib_measured": None,
        "device_memory_baseline_mib_measured": None,
        "peak_device_memory_mib_measured": None,
        "device_memory_delta_mib_measured": None,
        "peak_gpu_utilization_pct_measured": None,
        "average_gpu_utilization_pct_measured": None,
        "returncode": None,
        "timed_out": False,
        "monitor_error": "",
        "error": "",
        "command": command_text,
    }


def run_session(
    args,  # 命令行参数（含 n_ctx/batch/超时等所有覆盖值）
    round_num,  # 本轮次号（1..args.rounds）；每轮 = 一次全新 bench_run 进程（模型重载，warmup 重新吸收冷启动）
    case_id,  # 用例 ID，如 LC-001
    case,  # 该用例的配置定义（输入长度、输出 token 数等）
    selections,  # 本轮要跑的 prompt 清单（105 条 = 5 预热 + 100 正式，50 条池按序循环 2 遍，来自 run-manifest）
    metadata,  # prompt 元数据档案（实际 token 数、SHA-256、来源领域等）
    processed_dir,  # 预处理数据目录（prompt 文件所在地）
    output_dir,  # 结果输出目录（本用例的日志与记录写到这里）
    environment,  # 子进程环境（含 CUDA_VISIBLE_DEVICES=4 等平台变量）
    platform_name,  # 平台名（cuda）
    backend_device,  # 传给 llama.cpp 的设备名（CUDA0）
    n_gpu_layers,  # 卸载到 GPU 的层数（-ngl）
    memory_monitor,  # 显存监控方式（"nvml" 或自定义命令）
    model_path,  # GGUF 模型文件路径
):
    """Run all selections of one case inside a single bench_run process.

    The model is loaded once; every run is an independent single-turn request
    in the same llama_context with the KV cache cleared between runs
    (llama_memory_clear inside bench_run), so context never accumulates and
    the warmup runs absorb model load and CUDA kernel initialization. This
    mirrors the request-based persistent-engine semantics of the SGLang/vLLM
    benchmarks. Bench_run emits one JSONL result event per run; this function
    pairs each with its run_start event for wall time and resets the memory
    window so the per-run delta covers one run's KV plus compute buffers.
    """
    # ① 准备阶段：建本用例的子目录，生成 plan.tsv（每条运行一行）与完整命令并存档（session.command.txt，保证每次运行的命令可审计）
    run_dir = output_dir / case_id
    run_dir.mkdir(parents=True, exist_ok=True)
    plan_path = build_run_plan(case_id, case, selections, metadata, processed_dir, run_dir)
    command = build_bench_run_command(args, model_path, plan_path, n_gpu_layers)
    command_text = shlex.join(command)
    (run_dir / "session.command.txt").write_text(command_text + "\n", encoding="utf-8")

    # ② 显存监控初始化：NVML 打开目标卡；每轮 run_start 事件时由主线程读取"本轮基线显存"，
    #    采样线程负责窗口内峰值；NVML 不可用时降级到自定义命令
    nvml = None
    device_monitor_enabled = False
    if memory_monitor == "nvml":
        try:
            nvml = NvmlMemoryReader(args.device_index)
            device_monitor_enabled = True
        except Exception as exc:
            print("warning: NVML monitoring unavailable: {}".format(exc), file=sys.stderr)
    if nvml is None and args.memory_command:
        try:
            read_memory_command(args.memory_command, args.device_index)
            device_monitor_enabled = True
        except Exception as exc:
            print("warning: device memory monitoring unavailable: {}".format(exc), file=sys.stderr)

    records = []
    proc = None
    stdout_reader = None
    stderr_reader = None
    monitor = None
    try:
        # 轮次 >1 时会话日志加 .rNN 后缀，避免多轮互相覆盖；plan.tsv / session.command.txt 各轮内容相同，覆盖无害
        round_suffix = ".r{:02d}".format(round_num) if args.rounds > 1 else ""
        with (run_dir / ("session.stdout" + round_suffix + ".log")).open("w", encoding="utf-8") as session_stdout, \
             (run_dir / ("session.stderr" + round_suffix + ".log")).open("w", encoding="utf-8") as session_stderr:
            # ③ 启动 bench_run 子进程：stdout/stderr 两个 StreamReader 后台线程逐行读输出
            #    （stdout 解析 JSONL 事件并回显，stderr 回显人类可读进度），
            #    一个 ProcessMonitor 线程定时采样 RSS 与显存
            proc = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
                start_new_session=True,
            )
            condition = threading.Condition()
            stdout_reader = StreamReader(proc.stdout, condition, echo=session_stdout)
            stderr_reader = StreamReader(proc.stderr, condition, echo=session_stderr)
            stdout_reader.start()
            stderr_reader.start()
            monitor = ProcessMonitor(
                proc,
                args.monitor_interval,
                memory_reader=nvml,
                memory_command=args.memory_command,
                device_index=args.device_index,
            )
            monitor.start()

            # ④ 顺序执行每条运行：等 run_start（KV 已清空、计时未开始）→ 重置内存窗口并读基线 →
            #    等 result 或 error 事件；超时或进程提前退出则该轮失败并跳过后续轮次
            session_error = ""
            for phase, phase_run, selection in selections:
                manifest_run = selection["run"]
                relative = prompt_for_case(case, selection)
                record = new_run_record(
                    args, case_id, case, phase, phase_run, round_num, selection, metadata[relative], relative,
                    platform_name, backend_device, command_text,
                )
                if session_error:
                    # 前置轮已失败：本轮不发送，直接标记跳过
                    record["error"] = "skipped: {}".format(session_error)
                    records.append(record)
                    continue

                start_time, event, status = stdout_reader.wait_event(
                    proc, args.timeout, lambda ev: ev.get("event") in ("run_start", "error")
                )
                if status == "timeout":
                    record["timed_out"] = True
                    record["error"] = "timeout"
                    session_error = "timeout"
                    terminate_process(proc)
                    records.append(record)
                    continue
                if status == "exited":
                    record["error"] = "bench_run exited with status {} before run_start".format(proc.returncode)
                    session_error = record["error"]
                    records.append(record)
                    continue
                if event["event"] == "error":
                    # run 失败发生在 run_start 之前（prompt 文件缺失、模板失败、超 n_ctx 等）
                    record["error"] = event.get("message", "run failed")
                    records.append(record)
                    continue
                # run_start：校验身份（防御性），重置内存窗口，读取本轮基线
                if event.get("manifest_run") != manifest_run or event.get("phase") != phase \
                        or event.get("phase_run") != phase_run:
                    record["error"] = "run_start identity mismatch (expected {} {}/{} run {}, got manifest_run={})".format(
                        phase, phase_run, manifest_run, event.get("phase"), event.get("phase_run"), event.get("manifest_run")
                    )
                    records.append(record)
                    continue
                record["actual_prompt_tokens"] = event["prompt_tokens"]
                monitor.reset_window()
                baseline_mib = None
                if device_monitor_enabled:
                    try:
                        baseline_mib = monitor.read_device_mib_now()
                    except Exception as exc:
                        print("warning: device memory baseline read failed: {}".format(exc), file=sys.stderr)

                result_time, event, status = stdout_reader.wait_event(
                    proc, args.timeout, lambda ev: ev.get("event") in ("result", "error")
                )
                if status == "timeout":
                    record["timed_out"] = True
                    record["error"] = "timeout"
                    session_error = "timeout"
                    terminate_process(proc)
                    records.append(record)
                    continue
                if status == "exited":
                    record["error"] = "bench_run exited with status {} before result".format(proc.returncode)
                    session_error = record["error"]
                    records.append(record)
                    continue
                if event["event"] == "error":
                    # run_start 之后失败（decode 失败等）：该轮失败，bench_run 继续下一轮
                    record["error"] = event.get("message", "run failed")
                    records.append(record)
                    continue

                # ⑤ result：填充实测指标；wall time = result 与 run_start 事件到达时间差（外部计时）
                record["status"] = "ok"
                for field in ("prompt_eval_ms_measured", "ttft_ms_measured", "generation_eval_ms_measured",
                              "e2e_ms_measured", "prompt_tps_measured", "generation_tps_measured"):
                    record[field] = event.get(field)
                record["tokens_generated"] = event.get("tokens_generated")
                record["finish_reason"] = event.get("finish_reason")
                record["output_sha256"] = event.get("output_sha256")
                draft_tokens = event.get("draft_tokens")
                record["draft_tokens_measured"] = draft_tokens if draft_tokens else None
                record["draft_accepted_measured"] = event.get("draft_accepted") if draft_tokens else None
                record["draft_acceptance_rate_measured"] = event.get("draft_acceptance_rate") if draft_tokens else None
                record["actual_prompt_tokens"] = event["prompt_tokens"]
                generation_tps = record.get("generation_tps_measured")
                if generation_tps is not None and generation_tps > 0.0:
                    tpot_ms = 1000.0 / generation_tps
                    record["tpot_ms_measured"] = tpot_ms
                    record["itl_ms_measured"] = tpot_ms
                wall_time = result_time - start_time
                record["wall_time_s_measured"] = wall_time
                if wall_time > 0.0:
                    record["throughput_tok_s_est"] = (record.get("tokens_generated") or 0) / wall_time
                record["peak_process_rss_mib_measured"] = monitor.peak_rss_mib
                record["device_memory_baseline_mib_measured"] = baseline_mib
                record["peak_device_memory_mib_measured"] = monitor.peak_device_mib if device_monitor_enabled else None
                record["device_memory_delta_mib_measured"] = (
                    max(0.0, monitor.peak_device_mib - baseline_mib)
                    if device_monitor_enabled and baseline_mib is not None else None
                )
                record["peak_gpu_utilization_pct_measured"] = (
                    monitor.peak_gpu_utilization_pct if device_monitor_enabled else None
                )
                record["average_gpu_utilization_pct_measured"] = (
                    monitor.average_gpu_utilization_pct() if device_monitor_enabled else None
                )
                record["monitor_error"] = monitor.error
                records.append(record)

            # ⑥ 收尾：等待进程正常退出（超时则强杀）
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                terminate_process(proc)
            monitor.stop()
    except KeyboardInterrupt:
        # 用户 Ctrl-C：先杀掉子进程再抛异常，不留孤儿进程
        if proc is not None and proc.poll() is None:
            terminate_process(proc)
        raise
    finally:
        # ⑦ 清理：停掉采样与读取线程、关闭 NVML 句柄
        if monitor is not None:
            monitor.stop()
        if stdout_reader is not None and stdout_reader.is_alive():
            stdout_reader.join(timeout=2)
        if stderr_reader is not None and stderr_reader.is_alive():
            stderr_reader.join(timeout=2)
        if nvml is not None:
            nvml.close()

    # ⑧ 补全每轮记录的最终退出码，返回全部记录交给 main 汇总
    returncode = proc.returncode if proc is not None else None
    for record in records:
        record["returncode"] = returncode
    return records


# 通用 CSV 落盘：表头 + 数据行；extrasaction="ignore" —— 记录 dict 里
# 未声明在 fieldnames 中的字段直接忽略，不报错。
def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# 单指标统计摘要（只统计成功轮）：均值/标准差/最小/最大；
# 无有效值（全为 None）返回 None（下游按缺失处理，不污染汇总表）；
# 样本数 <2 时标准差记为 0。
def metric_summary(records, field):
    values = [float(record[field]) for record in records if record.get(field) is not None]
    if not values:
        return None
    return {
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


# 单用例汇总：只统计正式轮（warmup 不计入结果）；status="ok" 仅当全部正式轮成功，
# 否则 "incomplete"；metrics 保存全部实测指标的统计摘要（mean/stdev/min/max）；
# 五大头条指标转成双语表头列：TTFT/TPOT/ITL 取均值，E2E 均值转秒，
# 吞吐率取均值，峰值显存取窗口最大并转 GB（注意：是整卡峰值，非进程独占）。
def summarize_case(case_id, case, records, warmup_runs, formal_runs, rounds, platform_name, device_index, backend_device):
    formal_records = [record for record in records if record["phase"] == "formal"]
    ok_records = [record for record in formal_records if record["status"] == "ok"]
    metrics = {}
    for field in (
        "actual_prompt_tokens",
        "tokens_generated",
        "prompt_eval_ms_measured",
        "generation_eval_ms_measured",
        "ttft_ms_measured",
        "tpot_ms_measured",
        "itl_ms_measured",
        "e2e_ms_measured",
        "prompt_tps_measured",
        "generation_tps_measured",
        "wall_time_s_measured",
        "throughput_tok_s_est",
        "peak_process_rss_mib_measured",
        "peak_device_memory_mib_measured",
        "device_memory_delta_mib_measured",
        "peak_gpu_utilization_pct_measured",
        "average_gpu_utilization_pct_measured",
        "draft_tokens_measured",
        "draft_accepted_measured",
        "draft_acceptance_rate_measured",
    ):
        metrics[field] = metric_summary(ok_records, field)

    def rounded(summary_dict):
        return None if not summary_dict else round(summary_dict["mean"], 2)

    summary = {
        "case_id": case_id,
        "dataset": case.get("dataset", ""),
        "status": "ok" if len(ok_records) == formal_runs else "incomplete",
        "platform": platform_name,
        "physical_device_index": device_index,
        "backend_device": backend_device,
        "target_prompt_tokens": case["prompt_tokens"],
        "output_tokens_requested": case["output_tokens"],
        "warmup_runs": warmup_runs,
        "formal_runs_requested": formal_runs,
        "rounds": rounds,
        "formal_runs_ok": len(ok_records),
        "formal_runs_failed": len(formal_records) - len(ok_records),
        "metrics": metrics,
    }
    # the five headline metrics, bilingual headers, one row per case
    peak = metrics.get("peak_device_memory_mib_measured")
    e2e = metrics.get("e2e_ms_measured")
    summary["首令牌延迟(ms)/TTFT(ms)"] = rounded(metrics.get("ttft_ms_measured"))
    summary["平均输入长度(tokens)/Average Input Length(tokens)"] = rounded(metrics.get("actual_prompt_tokens"))
    summary["平均实际输出(tokens)/Average Generated Tokens"] = rounded(metrics.get("tokens_generated"))
    summary["每令牌时间(ms/token)/TPOT(ms/token)"] = rounded(metrics.get("tpot_ms_measured"))
    summary["令牌间延迟(ms/token)/ITL(ms/token)"] = rounded(metrics.get("itl_ms_measured"))
    summary["端到端耗时(s)/E2E(s)"] = None if not e2e else round(e2e["mean"] / 1000.0, 2)
    summary["吞吐率(token/s)/Throughput(token/s)"] = rounded(metrics.get("throughput_tok_s_est"))
    summary["峰值显存(GB)/Peak Memory(GB)"] = (
        None if not peak else round(peak["max"] / 1024.0, 2)
    )
    peak_gpu_utilization = metrics.get("peak_gpu_utilization_pct_measured")
    summary["GPU计算单元峰值利用率(%)/Peak GPU Compute Utilization(%)"] = (
        None if not peak_gpu_utilization else round(peak_gpu_utilization["max"], 2)
    )
    summary["GPU计算单元平均利用率(%)/Average GPU Compute Utilization(%)"] = rounded(
        metrics.get("average_gpu_utilization_pct_measured")
    )
    draft_tokens = sum(int(record.get("draft_tokens_measured") or 0) for record in ok_records)
    draft_accepted = sum(int(record.get("draft_accepted_measured") or 0) for record in ok_records)
    summary["草稿接受率(%)/Draft Acceptance Rate(%)"] = (
        round(100.0 * draft_accepted / draft_tokens, 2) if draft_tokens > 0 else None
    )
    return summary


# 查询 bench_run 的 --version 输出（llama.cpp 版本号），写入 manifest，
# 保证结果可以回溯到具体二进制；不可用时记 "unavailable: ..."（不阻断）。
def command_version(path, llama_server=None):
    try:
        command = [str(path), "--version"]
        if llama_server:
            command.extend(["--llama-server", str(llama_server)])
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=20,
            check=False,
        )
        return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return "unavailable: {}".format(exc)


def main():
    # ① 解析命令行参数，加载两份配置：用例定义（fair-baseline.json）与平台档案（platforms.json）
    args = parse_args()
    if args.bench_run is None and args.llama_cli is not None:
        print("warning: --llama-cli is deprecated, use --bench-run", file=sys.stderr)
        args.bench_run = args.llama_cli

    # 读取lc配置
    benchmark_config_path, benchmark_config, cases = load_benchmark_config(args.config)
    platform_config_path, platform_config, profiles = load_platform_profiles(args.platform_config)
    apply_runtime_defaults(args, benchmark_config["runtime"])  # 命令行未指定的参数取配置中的默认值
    if args.parallel != 1:
        raise ValueError(
            "--parallel must be 1: every case runs as one bench_run process with sequential single-turn runs"
        )
    if args.list_cases:  # --list-cases：只打印用例清单（LC-001~004 及各自长度）就退出
        for case_id, case in cases.items():
            input_description = case.get("dataset") or "{} input tokens".format(case["prompt_tokens"])
            print("{}\t{}\t{} output tokens".format(case_id, input_description, case["output_tokens"]))
        return 0

    # ② 加载预处理数据：200 个 prompt 文件、105 轮运行清单（run-manifest.json）、每条 prompt 的元数据档案
    selected_cases = select_cases(args.case, cases)  # --case 过滤后实际要测的用例
    processed_dir = Path(args.processed_dir).resolve()
    manifest, preparation, metadata = load_inputs(processed_dir)
    warmup_runs = manifest["warmup_runs"] if args.warmup_runs is None else args.warmup_runs  # 5 预热轮
    formal_runs = manifest["formal_runs"] if args.formal_runs is None else args.formal_runs  # 100 正式轮（50 条 × 2 遍）
    selections_by_case = {
        case_id: choose_selections(manifest, warmup_runs, formal_runs, cases[case_id].get("dataset"))
        for case_id in selected_cases
    }
    # ③ 校验：prompt 文件存在、长度与用例配置匹配、SHA-256 与 metadata 一致（防数据被改动）
    validate_prompts(
        processed_dir,
        cases,
        selected_cases,
        selections_by_case,
        metadata,
        check_hashes=not args.skip_prompt_hash_check,
    )

    # ④ 参数合法性校验：bench_run / 模型路径必须存在且可执行、监控间隔为正、SHA-256 必须 64 位十六进制
    if not args.bench_run:
        raise ValueError("--bench-run or BENCH_RUN is required")
    if not args.model:
        raise ValueError("--model or LLAMA_MODEL is required")
    if benchmark_config.get("experiment", {}).get("value") == "dflash" and not args.draft_model:
        raise ValueError("the DFlash benchmark requires --draft-model")
    bench_run = Path(args.bench_run).resolve()
    model_path = Path(args.model).resolve()
    draft_model_path = Path(args.draft_model).resolve() if args.draft_model else None
    if not bench_run.is_file() or not os.access(bench_run, os.X_OK):
        raise ValueError("bench_run is not executable: {}".format(bench_run))
    if args.llama_server:
        llama_server = Path(args.llama_server).resolve()
        if not llama_server.is_file() or not os.access(llama_server, os.X_OK):
            raise ValueError("llama-server is not executable: {}".format(llama_server))
        args.llama_server = str(llama_server)
    if not model_path.is_file():
        raise ValueError("model does not exist: {}".format(model_path))
    if draft_model_path is not None and not draft_model_path.is_file():
        raise ValueError("draft model does not exist: {}".format(draft_model_path))
    if draft_model_path is not None and args.draft_n_max <= 0:
        raise ValueError("--draft-n-max must be positive")
    if args.monitor_interval <= 0:
        raise ValueError("--monitor-interval must be positive")
    if args.model_sha256 is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", args.model_sha256):
        raise ValueError("--model-sha256 must contain 64 hexadecimal characters")
    if args.draft_model_sha256 is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", args.draft_model_sha256):
        raise ValueError("--draft-model-sha256 must contain 64 hexadecimal characters")
    args.draft_model = str(draft_model_path) if draft_model_path is not None else None
    args.bench_run = str(bench_run)

    # ⑤ 平台解析：确定可见性环境变量（如 CUDA_VISIBLE_DEVICES=4）、传给 llama.cpp 的设备名（CUDA0）、
    #    GPU 层数、显存监控方式（NVML 或自定义命令）
    platform_name = detect_platform(args.platform, profiles)
    platform_profile = profiles[platform_name]
    platform_env, backend_device, n_gpu_layers, memory_monitor = platform_settings(
        platform_profile, args.device_index, args.backend_device, args.n_gpu_layers
    )
    if args.memory_command is None:
        args.memory_command = platform_profile.get("memory_command")
    # 组装子进程环境：继承当前环境 + 平台变量（限制只用指定卡）+ 用户附加变量
    extra_env = parse_assignments(args.env, "--env")
    environment = os.environ.copy()
    environment.pop("LLAMA_DUMP_GRAPH", None)
    environment.update(platform_env)
    environment.update(extra_env)

    # ⑥ --dry-run：只校验并打印每个用例将执行的完整命令（含环境变量前缀）与 plan.tsv 内容，不启动推理
    if args.dry_run:
        env_text = " ".join("{}={}".format(key, shlex.quote(value)) for key, value in {**platform_env, **extra_env}.items())
        print(
            "validated {} case(s), {} run(s); each case runs as one bench_run process (single-turn, KV cleared per run)".format(
                len(selected_cases), sum(len(items) for items in selections_by_case.values())
            )
        )
        with tempfile.TemporaryDirectory(prefix="bench-run-dry-") as tmp:
            for case_id in selected_cases:
                case = cases[case_id]
                selections = selections_by_case[case_id]
                plan_path = build_run_plan(case_id, case, selections, metadata, processed_dir, Path(tmp) / case_id)
                command = build_bench_run_command(args, model_path, plan_path, n_gpu_layers)
                print(
                    "\n[{}] one process, {} runs ({} warmup + {} formal)".format(
                        case_id, len(selections), warmup_runs, formal_runs
                    )
                )
                print("plan ({}):".format(plan_path))
                for line in plan_path.read_text(encoding="utf-8").splitlines():
                    print("  " + line)
                prefix = (env_text + " ") if env_text else ""
                print("{}{}".format(prefix, shlex.join(command)))
        return 0

    # ⑦ 建立结果目录：results/<时间戳>-bench-run-<平台>-device<卡号>/；已存在则拒绝（防覆盖旧结果）
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = Path(args.results_dir).resolve() / "{}-bench-run-{}-device{}".format(
            timestamp, platform_name, args.device_index
        )
    if output_dir.exists():
        raise ValueError("output directory already exists: {}".format(output_dir))
    output_dir.mkdir(parents=True)

    # ⑧ 模型 SHA-256：大文件哈希耗时，套件模式下由 run_suite.py 统一算一次、经 --model-sha256 传入，这里直接复用
    model_sha256 = args.model_sha256
    if model_sha256 is not None:
        model_sha256 = model_sha256.lower()
    elif not args.skip_model_hash:
        print("calculating model SHA-256...", flush=True)
        model_sha256 = sha256_file(model_path)
    draft_model_sha256 = args.draft_model_sha256.lower() if args.draft_model_sha256 else None
    if draft_model_path is not None and draft_model_sha256 is None and not args.skip_model_hash:
        print("calculating draft model SHA-256...", flush=True)
        draft_model_sha256 = sha256_file(draft_model_path)
    run_manifest = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": host_platform.node(),
        "host_platform": host_platform.platform(),
        "python": sys.version,
        "bench_run": str(bench_run),
        "bench_run_version": command_version(bench_run, args.llama_server),
        "llama_server": args.llama_server,
        "model": str(model_path),
        "model_size": model_path.stat().st_size,
        "model_sha256": model_sha256,
        "draft_model": str(draft_model_path) if draft_model_path is not None else None,
        "draft_model_size": draft_model_path.stat().st_size if draft_model_path is not None else None,
        "draft_model_sha256": draft_model_sha256,
        "benchmark_config": str(benchmark_config_path),
        "benchmark_definition": benchmark_config,
        "platform": platform_name,
        "platform_config": str(platform_config_path),
        "platform_definition": platform_config,
        "platform_profile": platform_profile,
        "physical_device_index": args.device_index,
        "backend_device": backend_device,
        "environment_overrides": {**platform_env, **extra_env},
        "processed_dir": str(processed_dir),
        "dataset_preparation": preparation,
        "source_run_manifest": manifest,
        "selected_cases": selected_cases,
        "settings": {
            "n_ctx": args.n_ctx,
            "n_batch": args.n_batch,
            "n_ubatch": args.n_ubatch,
            "n_gpu_layers": n_gpu_layers,
            "parallel": args.parallel,
            "type_k": args.type_k,
            "type_v": args.type_v,
            "flash_attn": args.flash_attn,
            "poll": args.poll,
            "temperature": args.temperature,
            "seed": args.seed,
            "speculative_type": "draft-dflash" if draft_model_path is not None else "none",
            "draft_n_max": args.draft_n_max if draft_model_path is not None else None,
            "draft_n_gpu_layers": args.draft_n_gpu_layers if draft_model_path is not None else None,
            "threads": args.threads,
            "ignore_eos": args.ignore_eos,
            "prefix_cache_hit_rate_pct": 0,
            "warmup_runs": warmup_runs,
            "formal_runs": formal_runs,
            "rounds": args.rounds,
            "extra_args": args.extra_arg,
        },
        "metric_notes": {
            "execution_mode": "one bench_run process per case; the model is loaded once and every run is an independent single-turn request in the same llama_context with the KV cache cleared between runs (llama_memory_clear), matching request-based persistent-engine benchmarks (SGLang/vLLM)",
            "warmup_semantics": "the warmup runs absorb one-time costs: model loading, CUDA context and kernel initialization; formal runs measure steady-state single-turn latency",
            "prompt_artifact": "the Qwen chat template is applied inside bench_run through the same common path as llama-server /apply-template (including reasoning-template workarounds), so token counts match the prepared metadata exactly; there is no stdin protocol and no trailing newline token",
            "summary_columns": "bilingual 中文/English; TTFT(ms)=首令牌延迟, TPOT(ms/token)=每令牌时间, ITL(ms/token)=令牌间延迟, E2E(s)=端到端耗时, Throughput(token/s)=吞吐率, Peak Memory(GB)=峰值显存",
            "ttft_ms_measured": "measured time from run start (KV cleared) to the first output token becoming available (prefill + sampling; the first token's logits come from the prefill forward pass itself, so no extra decode is included — matches the serving-engine (vLLM/SGLang) definition of TTFT); measured inside bench_run with ggml_time_us and llama_synchronize",
            "prompt_eval_ms_measured": "measured prompt evaluation (prefill) time inside bench_run",
            "generation_eval_ms_measured": "measured decode time of all generated tokens inside bench_run",
            "e2e_ms_measured": "measured end-to-end run time inside bench_run, converted to seconds in the summary",
            "tpot_ms_measured": "time per output token: 1000 / generation_tps_measured, identical to itl_ms_measured by definition",
            "itl_ms_measured": "inter-token latency, same value as tpot_ms_measured (single-stream greedy generation)",
            "wall_time_s_measured": "Python-side wall time between the run_start and result events; includes process/pipe overhead on top of e2e_ms_measured",
            "throughput_tok_s_est": "end-to-end throughput estimate: actual generated tokens divided by Python wall time; differs from generation_tps_measured which covers the generation phase only",
            "peak_device_memory_mib_measured": "whole physical device usage sampled per run window (peaks reset at each run_start), not process-specific; reported as 峰值显存(GB)/Peak Memory(GB) using the maximum",
            "device_memory_delta_mib_measured": "per-run window peak minus the baseline read at run_start (after the KV clear), so it reflects one run's KV plus compute buffers",
            "peak_gpu_utilization_pct_measured": "maximum NVML GPU compute utilization sample in each run window; the summary reports the maximum across formal runs",
            "average_gpu_utilization_pct_measured": "arithmetic mean of NVML GPU compute utilization samples in each run window; the summary reports the mean across formal runs",
            "prompt_tps_measured": "prompt tokens per second during prefill, measured inside bench_run",
            "generation_tps_measured": "generated tokens per second during the generation phase, measured inside bench_run",
            "draft_acceptance_rate_measured": "accepted DFlash draft tokens divided by proposed draft tokens; zero for non-speculative cases",
            "derived_fields": "fields ending in _est are estimates derived from measured values",
        },
    }
    # ⑨ 写 manifest.json：完整存档（软件版本、模型及 SHA-256、两份配置原文、平台档案、环境变量、数据准备参数、
    #    105 轮运行清单）—— 测试一结束就落盘，保证任何结果都可审计、可复现
    (output_dir / "manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # ⑩ 逐个用例执行：每用例 = 单进程多轮单轮会话（详见 run_session）；每轮记录实时追加写入
    #    runs.jsonl（边跑边落盘，中断也不丢已完成的轮次）
    records = []
    runs_jsonl = output_dir / "runs.jsonl"
    with runs_jsonl.open("w", encoding="utf-8") as jsonl:

        # 每个用例的多轮循环：每轮 = 一次全新 bench_run 进程（模型重载、warmup 重新吸收冷启动）
        # 例如lc01～lc04，每轮 105 条 prompt（5 预热 + 100 正式），每条 prompt 生成 1 个结果，最终 runs.jsonl 共 4×105=420 条记录
        for case_id in selected_cases:
            selections = selections_by_case[case_id]
            print(
                "[{}] {} round(s), each one bench_run process with {} runs".format(
                    case_id, args.rounds, len(selections)
                ),
                flush=True,
            )
            case_records = []
            failed_any = False

            # 单个测试lc01/02/03/04的多轮循环：每轮 = 一次全新 bench_run 进程（模型重载、warmup 重新吸收冷启动）
            for round_num in range(1, args.rounds + 1):  # 每轮 = 一次全新 bench_run 进程（模型重载、warmup 重新吸收冷启动）

                # 105次 run_session() → 105 条记录（5 预热 + 100 正式）写入 runs.jsonl，每次都是一个 bench_run 进程，保证每轮都是独立的单轮请求（KV 清空），符合 vLLM/SGLang 的持久引擎语义
                round_records = run_session(
                    args,
                    round_num,
                    case_id,
                    cases[case_id],
                    selections,
                    metadata,
                    processed_dir,
                    output_dir,
                    environment,
                    platform_name,
                    backend_device,
                    n_gpu_layers,
                    memory_monitor,
                    model_path,
                )
                case_records.extend(round_records)
                for record in round_records:
                    jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
                    jsonl.flush()
                    print(
                        "[{}] r{}/{} {}-{:02d} manifest-run={} source={}: {}".format(
                            case_id,
                            record["round"],
                            args.rounds,
                            record["phase"],
                            record["phase_run"],
                            record["manifest_run"],
                            record["source_id"],
                            record["status"],
                        ),
                        flush=True,
                    )
                if any(record["status"] != "ok" for record in round_records):
                    failed_any = True
                    if args.fail_fast:  # --fail-fast：该轮出现失败即中止本 case 的后续轮次
                        break
            records.extend(case_records)
            if args.fail_fast and failed_any:
                break

    # ⑪ 汇总：只统计正式轮（预热轮不算入结果），逐用例计算六指标均值 → runs.csv / summary.csv / summary.json
    summaries = []
    for case_id in selected_cases:
        case_records = [record for record in records if record["case_id"] == case_id]
        summaries.append(
            summarize_case(
                case_id,
                cases[case_id],
                case_records,
                warmup_runs,
                formal_runs * args.rounds,  # 多轮时正式样本总数 = 100 × rounds，状态判定与统计都按合并后的全量样本
                args.rounds,
                platform_name,
                args.device_index,
                backend_device,
            )
        )
    write_csv(output_dir / "runs.csv", RUN_FIELDS, records)
    write_csv(output_dir / "summary.csv", SUMMARY_FIELDS, summaries)
    (output_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # ⑫ 终端打印双语表头汇总表（TTFT / TPOT / ITL / E2E / 吞吐率 / 峰值显存），一用例一行
    print("\n== summary (中文/English) ==")
    print("\t".join(SUMMARY_FIELDS))
    for summary in summaries:
        print("\t".join(str(summary.get(field, "")) for field in SUMMARY_FIELDS))
    print("results: {}".format(output_dir))
    # 返回值：任一用例未完整跑完返回 1，供 run_suite.py 判断该任务成败
    return 1 if any(summary["status"] != "ok" for summary in summaries) else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        sys.exit(130)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        sys.exit(2)
