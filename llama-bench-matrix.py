#!/usr/bin/env python3

import argparse
import csv
import ctypes
import ctypes.util
import datetime as dt
import fnmatch
import json
import os
from pathlib import Path
import platform as host_platform
import re
import shlex
import statistics
import subprocess
import sys
import threading
import time


SUMMARY_FIELDS = [
    "case_id",
    "status",
    "platform",
    "device",
    "gpu_info",
    "backends",
    "model_key",
    "model_filename",
    "model_type",
    "category",
    "priority",
    "prompt_tokens",
    "output_tokens",
    "prefix_hit_rate_pct",
    "state",
    "warmup_runs",
    "formal_runs",
    "n_batch",
    "n_ubatch",
    "type_k",
    "type_v",
    "flash_attn",
    "poll",
    "pp_ms",
    "pp_tokens_per_second",
    "ttft_est_ms",
    "tpot_ms",
    "itl_est_ms",
    "decode_tokens_per_second",
    "e2e_ms",
    "e2e_tokens_per_second",
    "peak_device_memory_mib",
    "device_memory_baseline_mib",
    "device_memory_delta_mib",
    "peak_process_rss_mib",
    "wall_time_s",
    "command",
    "error",
]


class NvmlMemoryReader:
    class Memory(ctypes.Structure):
        _fields_ = [
            ("total", ctypes.c_ulonglong),
            ("free", ctypes.c_ulonglong),
            ("used", ctypes.c_ulonglong),
        ]

    def __init__(self, device_index):
        lib_name = ctypes.util.find_library("nvidia-ml") or "libnvidia-ml.so.1"
        self.lib = ctypes.CDLL(lib_name)
        self.lib.nvmlInit_v2.restype = ctypes.c_int
        self.lib.nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
        self.lib.nvmlDeviceGetMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(self.Memory)]
        self.lib.nvmlDeviceGetMemoryInfo.restype = ctypes.c_int
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

    def close(self):
        self.lib.nvmlShutdown()


class ProcessMonitor:
    def __init__(self, proc, interval, memory_reader=None, memory_command=None, device_index=0):
        self.proc = proc
        self.interval = interval
        self.memory_reader = memory_reader
        self.memory_command = memory_command
        self.device_index = device_index
        self.peak_rss_mib = 0.0
        self.peak_device_mib = 0.0
        self.error = ""
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval * 4))

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
            command = [part.format(device=self.device_index) for part in shlex.split(self.memory_command)]
            result = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            match = re.search(r"[-+]?[0-9]*\.?[0-9]+", result.stdout)
            if result.returncode == 0 and match:
                return float(match.group(0))
        return 0.0

    def _sample(self):
        self.peak_rss_mib = max(self.peak_rss_mib, self._read_rss_mib())
        self.peak_device_mib = max(self.peak_device_mib, self._read_device_mib())

    def _run(self):
        try:
            while not self._stop.is_set() and self.proc.poll() is None:
                self._sample()
                self._stop.wait(self.interval)
            self._sample()
        except Exception as exc:
            self.error = str(exc)


def parse_args():
    parser = argparse.ArgumentParser(description="Run a portable llama-bench test matrix")
    parser.add_argument("--bench", default="./build/bin/llama-bench", help="path to llama-bench")
    parser.add_argument(
        "--matrix",
        default=str(Path(__file__).with_name("llama-bench-matrix.csv")),
        help="CSV test matrix",
    )
    parser.add_argument("--model", action="append", default=[], metavar="KEY=PATH", help="model mapping")
    parser.add_argument("--platform", choices=["auto", "cuda", "cann", "metal", "cpu"], default="auto")
    parser.add_argument("--device-index", type=int, default=0, help="physical device index")
    parser.add_argument("--backend-device", help="llama.cpp device name after visibility filtering")
    parser.add_argument("--case", action="append", default=[], help="case id, comma list, or glob")
    parser.add_argument("--results-dir", default="bench-results")
    parser.add_argument("--warmup", type=int, help="override matrix warmup count")
    parser.add_argument("--repetitions", type=int, help="override matrix formal run count")
    parser.add_argument("--batch-size", type=int, help="override n_batch")
    parser.add_argument("--ubatch-size", type=int, help="override n_ubatch")
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--monitor-interval", type=float, default=0.1)
    parser.add_argument(
        "--memory-command",
        help="optional command that prints used device memory in MiB; {device} is replaced",
    )
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--timeout", type=float, default=0.0, help="per-case timeout in seconds")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    return parser.parse_args()


def parse_assignments(values, label):
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("{} must use KEY=VALUE: {}".format(label, value))
        key, item = value.split("=", 1)
        result[key.strip()] = item.strip()
    return result


def load_matrix(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "case_id",
        "model_key",
        "prompt_tokens",
        "output_tokens",
        "warmup_runs",
        "formal_runs",
    }
    missing = required.difference(rows[0].keys() if rows else [])
    if missing:
        raise ValueError("matrix is missing columns: {}".format(", ".join(sorted(missing))))
    return rows


def select_cases(rows, selectors):
    enabled = [row for row in rows if row.get("enabled", "1").strip().lower() not in ("0", "false", "no")]
    if not selectors:
        return enabled
    patterns = []
    for selector in selectors:
        patterns.extend(part.strip() for part in selector.split(",") if part.strip())
    return [row for row in enabled if any(fnmatch.fnmatchcase(row["case_id"], pattern) for pattern in patterns)]


def detect_platform(requested):
    if requested != "auto":
        return requested
    if sys.platform == "darwin":
        return "metal"
    if ctypes.util.find_library("nvidia-ml"):
        return "cuda"
    if any(Path(path).exists() for path in ("/usr/local/Ascend", "/usr/local/bin/npu-smi", "/usr/bin/npu-smi")):
        return "cann"
    return "cpu"


def platform_settings(platform_name, device_index, backend_device):
    environment = {}
    if platform_name == "cuda":
        environment["CUDA_VISIBLE_DEVICES"] = str(device_index)
        backend_device = backend_device or "CUDA0"
    elif platform_name == "cann":
        environment["ASCEND_RT_VISIBLE_DEVICES"] = str(device_index)
        backend_device = backend_device or "CANN0"
    elif platform_name == "metal":
        backend_device = backend_device or "Metal"
    else:
        backend_device = backend_device or "CPU"
    return environment, backend_device


def numeric(row, name, override=None):
    if override is not None:
        return override
    return int(row[name])


def build_command(args, row, model_path, backend_device, warmup, formal):
    prompt = int(row["prompt_tokens"])
    output = int(row["output_tokens"])
    n_batch = numeric(row, "n_batch", args.batch_size)
    n_ubatch = numeric(row, "n_ubatch", args.ubatch_size)
    command = [
        str(Path(args.bench)),
        "-m",
        model_path,
        "-p",
        str(prompt),
        "-n",
        str(output),
        "-pg",
        "{},{}".format(prompt, output),
        "-b",
        str(n_batch),
        "-ub",
        str(n_ubatch),
        "-ctk",
        row.get("type_k", "f16") or "f16",
        "-ctv",
        row.get("type_v", "f16") or "f16",
        "-fa",
        row.get("flash_attn", "auto") or "auto",
        "--poll",
        row.get("poll", "50") or "50",
        "-ngl",
        str(0 if args.platform == "cpu" else args.n_gpu_layers),
        "-dev",
        backend_device,
        "-r",
        str(warmup + formal),
        "--no-warmup",
        "-o",
        "json",
    ]
    if args.threads is not None:
        command.extend(["-t", str(args.threads)])
    if row.get("extra_args", "").strip():
        command.extend(shlex.split(row["extra_args"]))
    command.extend(args.extra_arg)
    return command


def mean(values):
    return statistics.fmean(values)


def stdev(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def find_test(rows, prompt, output):
    for row in rows:
        if int(row["n_prompt"]) == prompt and int(row["n_gen"]) == output:
            return row
    raise ValueError("missing llama-bench result for n_prompt={}, n_gen={}".format(prompt, output))


def formal_samples(test, warmup, formal):
    samples = [int(value) for value in test["samples_ns"]]
    expected = warmup + formal
    if len(samples) != expected:
        raise ValueError("expected {} samples, got {}".format(expected, len(samples)))
    return samples[warmup:]


def summarize(rows, case, warmup, formal):
    prompt = int(case["prompt_tokens"])
    output = int(case["output_tokens"])
    pp_row = find_test(rows, prompt, 0)
    tg_row = find_test(rows, 0, output)
    pg_row = find_test(rows, prompt, output)

    pp_samples = formal_samples(pp_row, warmup, formal)
    tg_samples = formal_samples(tg_row, warmup, formal)
    pg_samples = formal_samples(pg_row, warmup, formal)
    pp_ns = mean(pp_samples)
    tg_ns = mean(tg_samples)
    pg_ns = mean(pg_samples)
    token_ns = tg_ns / output

    return {
        "pp_ms": pp_ns / 1e6,
        "pp_stddev_ms": stdev(pp_samples) / 1e6,
        "pp_tokens_per_second": prompt * 1e9 / pp_ns,
        "tg_ms": tg_ns / 1e6,
        "tg_stddev_ms": stdev(tg_samples) / 1e6,
        "decode_tokens_per_second": output * 1e9 / tg_ns,
        "tpot_ms": token_ns / 1e6,
        "itl_est_ms": token_ns / 1e6,
        "ttft_est_ms": (pp_ns + token_ns) / 1e6,
        "e2e_ms": pg_ns / 1e6,
        "e2e_stddev_ms": stdev(pg_samples) / 1e6,
        "e2e_tokens_per_second": (prompt + output) * 1e9 / pg_ns,
        "formal_samples_ns": {
            "pp": pp_samples,
            "tg": tg_samples,
            "pg": pg_samples,
        },
        "raw_tests": rows,
        "metadata": pp_row,
    }


def read_completed(path):
    completed = set()
    if not path.exists():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status") == "ok":
            completed.add(record["case_id"])
    return completed


def write_summary(path, records):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def run_case(args, row, model_path, environment, backend_device, platform_name, output_dir):
    warmup = numeric(row, "warmup_runs", args.warmup)
    formal = numeric(row, "formal_runs", args.repetitions)
    command = build_command(args, row, model_path, backend_device, warmup, formal)
    command_text = shlex.join(command)
    case_id = row["case_id"]
    print("[{}] {}".format(case_id, command_text), flush=True)

    if args.dry_run:
        return {"case_id": case_id, "status": "dry-run", "command": command_text}

    nvml = None
    baseline_device_mib = 0.0
    if platform_name == "cuda":
        try:
            nvml = NvmlMemoryReader(args.device_index)
            baseline_device_mib = nvml.read_mib()
        except Exception as exc:
            print("[{}] warning: NVML monitoring unavailable: {}".format(case_id, exc), file=sys.stderr)

    started = time.monotonic()
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    monitor = ProcessMonitor(
        proc,
        args.monitor_interval,
        memory_reader=nvml,
        memory_command=args.memory_command,
        device_index=args.device_index,
    )
    monitor.start()
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=args.timeout if args.timeout > 0 else None)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.terminate()
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
    finally:
        monitor.stop()
        if nvml is not None:
            nvml.close()
    wall_time = time.monotonic() - started

    (output_dir / "{}.stdout.json".format(case_id)).write_text(stdout, encoding="utf-8")
    (output_dir / "{}.stderr.log".format(case_id)).write_text(stderr, encoding="utf-8")
    (output_dir / "{}.command.txt".format(case_id)).write_text(command_text + "\n", encoding="utf-8")

    record = {
        "case_id": case_id,
        "status": "failed",
        "platform": platform_name,
        "device": backend_device,
        "model_key": row["model_key"],
        "model_filename": model_path,
        "category": row.get("category", ""),
        "priority": row.get("priority", ""),
        "prompt_tokens": int(row["prompt_tokens"]),
        "output_tokens": int(row["output_tokens"]),
        "prefix_hit_rate_pct": 0,
        "state": "hot",
        "warmup_runs": warmup,
        "formal_runs": formal,
        "n_batch": numeric(row, "n_batch", args.batch_size),
        "n_ubatch": numeric(row, "n_ubatch", args.ubatch_size),
        "type_k": row.get("type_k", "f16") or "f16",
        "type_v": row.get("type_v", "f16") or "f16",
        "flash_attn": row.get("flash_attn", "auto") or "auto",
        "poll": int(row.get("poll", "50") or 50),
        "peak_device_memory_mib": monitor.peak_device_mib,
        "device_memory_baseline_mib": baseline_device_mib,
        "device_memory_delta_mib": max(0.0, monitor.peak_device_mib - baseline_device_mib),
        "peak_process_rss_mib": monitor.peak_rss_mib,
        "wall_time_s": wall_time,
        "command": command_text,
        "monitor_error": monitor.error,
        "returncode": proc.returncode,
        "error": "",
    }

    if timed_out:
        record["error"] = "timeout"
        return record
    if proc.returncode != 0:
        record["error"] = "llama-bench exited with status {}".format(proc.returncode)
        return record

    try:
        tests = json.loads(stdout)
        metrics = summarize(tests, row, warmup, formal)
        metadata = metrics.pop("metadata")
        record.update(metrics)
        record["gpu_info"] = metadata.get("gpu_info", "")
        record["backends"] = metadata.get("backends", "")
        record["model_type"] = metadata.get("model_type", "")
        record["status"] = "ok"
    except Exception as exc:
        record["error"] = "result parsing failed: {}".format(exc)
    return record


def main():
    args = parse_args()
    matrix = load_matrix(args.matrix)
    selected = select_cases(matrix, args.case)

    if args.list_cases:
        for row in matrix:
            print("{}\t{}\t{}\t{}+{}".format(
                row["case_id"], row.get("category", ""), row["model_key"], row["prompt_tokens"], row["output_tokens"]
            ))
        return 0
    if not selected:
        raise ValueError("no matrix cases selected")

    args.platform = detect_platform(args.platform)
    platform_env, backend_device = platform_settings(args.platform, args.device_index, args.backend_device)
    model_map = parse_assignments(args.model, "--model")
    extra_env = parse_assignments(args.env, "--env")
    environment = os.environ.copy()
    environment.update(platform_env)
    environment.update(extra_env)

    bench_path = Path(args.bench)
    if not bench_path.is_file() or not os.access(bench_path, os.X_OK):
        raise ValueError("llama-bench is not executable: {}".format(bench_path))

    missing_models = sorted({row["model_key"] for row in selected}.difference(model_map))
    if missing_models:
        raise ValueError("missing --model mappings: {}".format(", ".join(missing_models)))
    for key in sorted({row["model_key"] for row in selected}):
        if not Path(model_map[key]).is_file():
            raise ValueError("model does not exist for {}: {}".format(key, model_map[key]))

    if args.dry_run:
        for row in selected:
            warmup = numeric(row, "warmup_runs", args.warmup)
            formal = numeric(row, "formal_runs", args.repetitions)
            command = build_command(args, row, model_map[row["model_key"]], backend_device, warmup, formal)
            print("[{}] {}".format(row["case_id"], shlex.join(command)))
        return 0

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.results_dir) / "{}-{}-gpu{}".format(timestamp, args.platform, args.device_index)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "results.jsonl"
    completed = read_completed(jsonl_path) if args.resume else set()

    manifest = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": host_platform.node(),
        "host_platform": host_platform.platform(),
        "python": sys.version,
        "platform": args.platform,
        "device_index": args.device_index,
        "backend_device": backend_device,
        "bench": str(bench_path.resolve()),
        "matrix": str(Path(args.matrix).resolve()),
        "models": model_map,
        "environment_overrides": {**platform_env, **extra_env},
        "selected_cases": [row["case_id"] for row in selected],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    records = []
    with jsonl_path.open("a", encoding="utf-8") as jsonl:
        for row in selected:
            if row["case_id"] in completed:
                print("[{}] skipped (already complete)".format(row["case_id"]))
                continue
            record = run_case(
                args,
                row,
                model_map[row["model_key"]],
                environment,
                backend_device,
                args.platform,
                output_dir,
            )
            jsonl.write(json.dumps(record, ensure_ascii=True) + "\n")
            jsonl.flush()
            records.append(record)
            print("[{}] {}".format(row["case_id"], record["status"]), flush=True)

    write_summary(output_dir / "summary.csv", records)
    print("results: {}".format(output_dir))
    return 1 if any(record["status"] == "failed" for record in records) else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        sys.exit(2)
