#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# run_suite.py —— 套件调度器（benchmark 三层结构的顶层）
# =============================================================================
# 职责：把一份套件配置翻译成一组 bench_test.py 子任务，串行执行并汇总。
#
# 三层调用链（自上而下）：
#   run_suite.py  ──每个 job（如 fair-baseline）调用一次──▶  bench_test.py
#     bench_test.py ──每个 case（如 LC-001）调用一次──────▶  bench_run（C++）
#
# 本文件只做"调度 + 汇总"，不碰 GPU、不做推理：
#   ① 读取套件配置，校验 job 的 model_key 与 case 配置声明一致；
#   ② 把 --model KEY=PATH 映射解析成 模型路径 → SHA-256（每个模型只哈希一次，
#      通过 --model-sha256 传给所有用到它的 job，避免 17GB 大文件重复哈希）；
#   ③ 按配置顺序逐个 job 串行执行 bench_test.py（每个 job 一个输出子目录），
#      --fail-fast 时任一 job 失败即中止后续 job；
#   ④ 收集各 job 的 summary.json，合并成套件级 summary.json / summary.csv，
#      连同全量命令、模型校验值写入 suite-manifest.json（可审计、可复现）。
#
# 典型用法（全套 28 case）：
#   CUDA_VISIBLE_DEVICES=4 python3 run_suite.py \
#     --bench-run build/bench_run \
#     --model f16=.../Qwen3.5-9B-F16.gguf \
#     --model q8_0=.../Qwen3.5-9B-Q8_0.gguf \
#     --model q4_k_m=.../Qwen3.5-9B-Q4_K_M.gguf \
#     --platform cuda --device-index 4
# 常用开关：--group N（只跑某类实验）、--job ID（只跑某 job）、
#           --smoke（每个 job 第一个 case 跑 1 轮冒烟）、--dry-run（只校验不跑）。
# =============================================================================

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SUITE = PROJECT_ROOT / "configs" / "lc-001-028-suite.json"


def parse_args():
    """解析命令行参数。绝大多数参数与 bench_test.py 同名同义，原样透传。"""
    parser = argparse.ArgumentParser(description="Run a configured group of bench_run benchmark jobs")
    parser.add_argument("--bench-run", default=os.environ.get("BENCH_RUN") or os.environ.get("LLAMA_CLI"),
                        help="path to the bench_run executable (or BENCH_RUN / LLAMA_CLI)")  # bench_run 可执行文件（C++ 推理内核）
    parser.add_argument("--suite", default=str(DEFAULT_SUITE), help="suite configuration")  # 套件配置（默认 lc-001-028-suite.json）
    parser.add_argument("--model", action="append", default=[], metavar="KEY=PATH")  # 模型映射，如 f16=/path/F16.gguf；KEY 必须覆盖套件用到的全部 model_key
    parser.add_argument("--job", action="append", default=[], help="job id or comma-separated job ids")  # 只跑指定 job（可重复/逗号分隔）
    parser.add_argument("--group", action="append", default=[], metavar="N",
                        help="worksheet category: 0=fair baseline, 1=weight quantization, 2=KV cache precision, 3=attention backend, 4=polling, 5=backend graph execution; repeatable or comma-separated")  # 只跑指定实验类别
    parser.add_argument("--platform", default="auto")  # 平台档案名（cuda/cann/metal/cpu），auto 自动检测
    parser.add_argument("--platform-config", default=str(PROJECT_ROOT / "configs" / "platforms.json"))  # 平台档案文件
    parser.add_argument("--processed-dir", default=str(PROJECT_ROOT / "data" / "processed" / "qwen3.5-9b"))  # 预处理数据目录（prompts、manifest 等）
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))  # 结果根目录（默认在其下建时间戳子目录）
    parser.add_argument("--output-dir", help="exact suite output directory")  # 精确指定套件输出目录（替代时间戳目录）
    parser.add_argument("--device-index", type=int, default=0)  # 物理加速卡索引
    parser.add_argument("--backend-device")  # 环境过滤后传给 llama.cpp 的设备名（如 CUDA0）
    parser.add_argument("--monitor-interval", type=float, default=0.1)  # 显存/RSS 采样间隔（秒）
    parser.add_argument("--memory-command")  # 自定义显存查询命令（输出 MiB 数值）
    parser.add_argument("--timeout", type=float, default=0.0)  # 每轮超时秒数（0=不超时）
    parser.add_argument("--rounds", type=int, default=1)  # 每 case 完整跑几轮：10 = 全套重复 10 次（10×105=1050 轮/case，共 29,400 次推理），正式样本合并统计降低噪声
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")  # 附加子进程环境变量
    parser.add_argument("--extra-arg", action="append", default=[])  # 附加 bench_run 参数（透传）
    parser.add_argument("--skip-model-hash", action="store_true")  # 跳过模型 SHA-256 计算（省时间，结果缺校验值）
    parser.add_argument("--skip-prompt-hash-check", action="store_true")  # 跳过 prompt 文件哈希校验
    parser.add_argument("--continue-on-error", action="store_true")  # 某 job 失败仍继续跑后续 job（默认 fail-fast）
    parser.add_argument("--smoke", action="store_true", help="run one formal invocation of each job's first case")  # 冒烟模式：每个 job 仅第一个 case 跑 1 轮
    parser.add_argument("--dry-run", action="store_true", help="validate all cases and print one command per case")  # 只校验不执行
    return parser.parse_args()


def parse_assignments(values, label):
    """解析 KEY=VALUE 形式的列表参数（如 --model f16=/path/a.gguf），返回 dict。
    label 仅用于报错信息（区分是 --model 还是 --env）。"""
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("{} must use KEY=VALUE: {}".format(label, value))
        key, item = value.split("=", 1)
        if not key or not item:
            raise ValueError("{} must use non-empty KEY=VALUE: {}".format(label, value))
        result[key] = item
    return result


def sha256_file(path):
    """分块（8 MiB）计算文件 SHA-256，避免一次性读入 17GB 大模型文件。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_job_ids(values):
    """展开 --job 的重复/逗号分隔写法为去重后的 job id 列表（保持出现顺序）。"""
    selected = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item and item not in selected:
                selected.append(item)
    return selected


def selected_groups(values):
    """同上，但用于 --group（实验类别 0-4）。"""
    selected = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item and item not in selected:
                selected.append(item)
    return selected


def load_suite(path, selected_ids, selected_groups):
    """加载并校验套件配置，返回 (suite_path, suite, jobs)。

    校验要点：
    - job id 唯一、job 指向的 case 配置文件存在且 schema 合法；
    - 关键一致性：case 配置文件里 experiment.model_key 声明的模型
      （如 lc-021-024-flash-attn-off.json 声明 q8_0）必须与套件 job 里
      的 model_key 一致，防止"配置说要 A 模型、套件却传 B 模型"这类
      静默错误（例如 F16 FA-off 会 OOM，配置改 Q8_0 后这里强制同步）；
    - 用 --job / --group 过滤后至少剩一个 job。
    """
    suite_path = Path(path).resolve()
    if not suite_path.is_file():
        raise ValueError("suite config does not exist: {}".format(suite_path))
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    if suite.get("schema_version") != 1:
        raise ValueError("unsupported suite schema: {}".format(suite.get("schema_version")))

    jobs = []
    known = set()
    known_groups = set()
    for item in suite.get("jobs", []):
        job_id = item.get("id")
        if not job_id or job_id in known:
            raise ValueError("suite job ids must be non-empty and unique")
        known.add(job_id)
        if "group" in item:
            known_groups.add(str(item["group"]))
        config_path = (suite_path.parent / item["config"]).resolve()
        if not config_path.is_file():
            raise ValueError("job config does not exist: {}".format(config_path))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("schema_version") != 1 or not config.get("cases"):
            raise ValueError("invalid benchmark config: {}".format(config_path))
        for field in ("environment",):
            if not isinstance(item.get(field, {}), dict):
                raise ValueError("job {} {} must be an object".format(job_id, field))
        if not isinstance(item.get("environment_by_platform", {}), dict):
            raise ValueError("job {} environment_by_platform must be an object".format(job_id))
        for environment in [item.get("environment", {}), *item.get("environment_by_platform", {}).values()]:
            if not isinstance(environment, dict):
                raise ValueError("job {} platform environment must be an object".format(job_id))
            for key, value in environment.items():
                if not key or not isinstance(value, str):
                    raise ValueError("job {} environment must contain non-empty string values".format(job_id))
        if not isinstance(item.get("unsupported_platforms", []), list):
            raise ValueError("job {} unsupported_platforms must be an array".format(job_id))
        declared_model_key = config.get("experiment", {}).get("model_key")
        if declared_model_key and declared_model_key != item.get("model_key"):
            raise ValueError("model key mismatch between suite and {}".format(config_path))
        jobs.append({**item, "config_path": config_path, "definition": config})

    unknown = set(selected_ids) - known
    if unknown:
        raise ValueError("unknown suite job(s): {}".format(", ".join(sorted(unknown))))
    if selected_ids:
        jobs = [job for job in jobs if job["id"] in selected_ids]
    unknown_groups = set(selected_groups) - known_groups
    if unknown_groups:
        raise ValueError("unknown suite group(s): {}".format(", ".join(sorted(unknown_groups))))
    if selected_groups:
        jobs = [job for job in jobs if str(job.get("group")) in selected_groups]
    if not jobs:
        raise ValueError("suite has no selected jobs")
    return suite_path, suite, jobs


def job_label(job):
    """job 的展示名，如 'g0/fair-baseline'（组号/作业名）。"""
    group = job.get("group")
    return "g{}/{}".format(group, job["id"]) if group is not None else job["id"]


def resolve_job_environment(job, platform):
    if platform in job.get("unsupported_platforms", []):
        raise ValueError("job {} is not supported on platform {}".format(job["id"], platform))
    environment = dict(job.get("environment", {}))
    by_platform = job.get("environment_by_platform", {})
    if by_platform:
        if platform == "auto":
            raise ValueError("job {} requires an explicit --platform".format(job["id"]))
        if platform not in by_platform:
            raise ValueError("job {} has no environment mapping for platform {}".format(job["id"], platform))
        environment.update(by_platform[platform])
    return environment


def append_common_args(command, args):
    """把与 job 无关的公共参数追加到 bench_test.py 命令尾部。
    每个 job 命令都以 bench_test.py 为基础再叠加 --config / --model / --output-dir。"""
    command.extend([
        "--platform", args.platform,
        "--platform-config", str(Path(args.platform_config).resolve()),
        "--processed-dir", str(Path(args.processed_dir).resolve()),
        "--device-index", str(args.device_index),
        "--monitor-interval", str(args.monitor_interval),
        "--timeout", str(args.timeout),
    ])
    if args.backend_device:
        command.extend(["--backend-device", args.backend_device])
    if args.memory_command:
        command.extend(["--memory-command", args.memory_command])
    if args.rounds != 1:
        command.extend(["--rounds", str(args.rounds)])  # 多轮重复测量：每 case 完整跑 N 轮（N>1 才传，保持单轮命令与历史一致）
    if args.skip_prompt_hash_check:
        command.append("--skip-prompt-hash-check")
    for value in args.env:
        command.extend(["--env", value])
    for value in args.extra_arg:
        command.extend(["--extra-arg", value])


def write_csv(path, rows):
    """按各行的并集字段写 CSV（无固定字段表，字段顺序按首次出现）。"""
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    # ① 校验 bench_run 可执行文件、加载套件配置、解析模型映射：
    #    套件用到的每个 model_key（f16/q8_0/q4_k_m）都必须有 --model KEY=PATH 提供
    args = parse_args()
    if not args.bench_run:
        raise ValueError("--bench-run or BENCH_RUN is required")
    bench_run = Path(args.bench_run).resolve()
    if not bench_run.is_file() or not os.access(bench_run, os.X_OK):
        raise ValueError("bench_run is not executable: {}".format(bench_run))

    selected_ids = selected_job_ids(args.job)
    selected_group_ids = selected_groups(args.group)
    suite_path, suite, jobs = load_suite(args.suite, selected_ids, selected_group_ids)
    model_values = parse_assignments(args.model, "--model")
    required_keys = {job["model_key"] for job in jobs}
    missing = required_keys - set(model_values)
    if missing:
        raise ValueError("missing model mapping(s): {}".format(", ".join(sorted(missing))))

    # ② 收集模型信息（路径、大小；SHA-256 稍后统一算一次）
    models = {}
    for key in sorted(required_keys):
        path = Path(model_values[key]).resolve()
        if not path.is_file():
            raise ValueError("model {} does not exist: {}".format(key, path))
        models[key] = {"path": str(path), "size": path.stat().st_size, "sha256": None}

    # ③ 组装每个 job 的基准命令：python3 bench_test.py --bench-run ... --model <模型> --config <case 配置>
    #    公共参数（平台、卡号、监控间隔等）统一追加；后续循环里再补 --output-dir / --model-sha256
    bench_script = PROJECT_ROOT / "bench_test.py"
    base_commands = []
    for job in jobs:
        job = {**job, "resolved_environment": resolve_job_environment(job, args.platform)}
        command = [
            sys.executable,
            str(bench_script),
            "--bench-run", str(bench_run),
            "--model", models[job["model_key"]]["path"],
            "--config", str(job["config_path"]),
        ]
        append_common_args(command, args)
        for key, value in job["resolved_environment"].items():
            command.extend(["--env", "{}={}".format(key, value)])
        base_commands.append((job, command))

    # ④ --dry-run：每个 job 只校验一条代表性命令（0 预热 + 1 正式），不启动推理
    if args.dry_run:
        print("suite: {}".format(suite["name"]))
        print("dry-run validates every case with one representative formal command")
        for job, command in base_commands:
            command.extend(["--warmup-runs", "0", "--formal-runs", "1", "--dry-run"])
            print("\n[{}] {}".format(job_label(job), shlex.join(command)), flush=True)
            result = subprocess.run(command, check=False)
            if result.returncode != 0 and not args.continue_on_error:
                return result.returncode
        return 0

    # ⑤ 建套件输出目录：results/<时间戳>-<套件名>-<平台>-device<卡号>/；已存在则拒绝（防覆盖）
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (
        Path(args.results_dir).resolve() / "{}-{}-{}-device{}".format(
            timestamp, suite["name"], args.platform, args.device_index
        )
    )
    if output_dir.exists():
        raise ValueError("output directory already exists: {}".format(output_dir))
    output_dir.mkdir(parents=True)

    # ⑥ 模型 SHA-256：每个模型只算一次（17GB 的 F16 文件哈希约需几十秒），
    #    算完后经 --model-sha256 传给所有用它跑 job 的 bench_test.py，避免重复计算
    if not args.skip_model_hash:
        for key, item in models.items():
            print("calculating {} model SHA-256...".format(key), flush=True)
            item["sha256"] = sha256_file(Path(item["path"]))

    # ⑦ 逐个 job 串行执行（同一时刻只跑一个 bench_test.py，GPU 独占）：
    #    每个 job 一个子目录，命令追加 --output-dir 与 --model-sha256（或 --skip-model-hash）
    run_jobs = []
    aggregate = []
    returncode = 0
    for job, command in base_commands:
        job_output = output_dir / job["id"]
        command.extend(["--output-dir", str(job_output), "--fail-fast"])  # --fail-fast：job 内出现失败轮即中止该 job
        model_info = models[job["model_key"]]
        if model_info["sha256"]:
            command.extend(["--model-sha256", model_info["sha256"]])
        else:
            command.append("--skip-model-hash")
        if args.smoke:
            # 冒烟模式：每个 job 只跑第一个 case 的 1 轮正式（0 预热）
            first_case = job["definition"]["cases"][0]["id"]
            command.extend(["--case", first_case, "--warmup-runs", "0", "--formal-runs", "1"])

        print("\n[{}] starting".format(job_label(job)), flush=True)
        result = subprocess.run(command, check=False)
        status = "ok" if result.returncode == 0 else "failed"
        # 记录 job 的完整命令（shlex.join 可原样回放）与退出码，写入 suite-manifest 供溯源
        run_jobs.append({
            "id": job["id"],
            "group": job.get("group"),
            "status": status,
            "returncode": result.returncode,
            "config": str(job["config_path"]),
            "model_key": job["model_key"],
            "environment": job["resolved_environment"],
            "output_dir": str(job_output),
            "command": shlex.join(command),
        })
        # ⑧ 聚合：从该 job 的 summary.json 收集每个 case 的汇总行，补充 job/组/实验变量信息；
        #    同时从 runs.jsonl 收集失败轮的错误信息（附加到对应 case 的 errors 字段）
        summary_path = job_output / "summary.json"
        if summary_path.is_file():
            run_errors = {}
            runs_path = job_output / "runs.jsonl"
            if runs_path.is_file():
                for line in runs_path.read_text(encoding="utf-8").splitlines():
                    run = json.loads(line)
                    if run.get("error"):
                        run_errors.setdefault(run["case_id"], []).append(run["error"])
            for row in json.loads(summary_path.read_text(encoding="utf-8")):
                aggregate.append({
                    "job_id": job["id"],
                    "group": job.get("group"),
                    "experiment_variable": job["definition"].get("experiment", {}).get("variable"),
                    "experiment_value": job["definition"].get("experiment", {}).get("value"),
                    "model_key": job["model_key"],
                    "errors": "; ".join(run_errors.get(row["case_id"], [])),
                    **row,
                })
        if result.returncode != 0:
            # 默认 fail-fast：job 失败即终止整个套件（--continue-on-error 可继续）
            returncode = result.returncode
            if not args.continue_on_error:
                break

    # ⑨ 落盘套件级产物：suite-manifest.json（溯源：套件定义/模型/每 job 命令）
    #    + summary.json（28 行合并结果）+ summary.csv（剔除 metrics 大字段）
    manifest = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "suite_config": str(suite_path),
        "suite_definition": suite,
        "mode": "smoke" if args.smoke else "formal",
        "platform": args.platform,
        "physical_device_index": args.device_index,
        "bench_run": str(bench_run),
        "models": models,
        "jobs": run_jobs,
    }
    (output_dir / "suite-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(output_dir / "summary.csv", [
        {key: value for key, value in row.items() if key != "metrics"} for row in aggregate
    ])
    print("\nsuite results: {}".format(output_dir))
    return returncode


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        sys.exit(130)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        sys.exit(2)
