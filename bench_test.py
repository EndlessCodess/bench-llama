#!/usr/bin/env python3

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
import threading
import time


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "qwen3.5-9b"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_BENCHMARK_CONFIG = PROJECT_ROOT / "configs" / "fair-baseline.json"
DEFAULT_PLATFORM_CONFIG = PROJECT_ROOT / "configs" / "platforms.json"

RUN_FIELDS = [
    "case_id",
    "status",
    "phase",
    "phase_run",
    "manifest_run",
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
    "prompt_tps_measured",
    "generation_tps_measured",
    "prompt_eval_ms_est",
    "tpot_ms_est",
    "itl_ms_est",
    "generation_eval_ms_est",
    "ttft_ms_est",
    "compute_e2e_ms_est",
    "wall_time_s_measured",
    "peak_process_rss_mib_measured",
    "device_memory_baseline_mib_measured",
    "peak_device_memory_mib_measured",
    "device_memory_delta_mib_measured",
    "returncode",
    "timed_out",
    "monitor_error",
    "error",
    "command",
]

# Summary table columns: the five headline metrics, one row per case,
# with bilingual "Chinese/English" headers.
SUMMARY_FIELDS = [
    "case_id",
    "status",
    "platform",
    "physical_device_index",
    "backend_device",
    "target_prompt_tokens",
    "output_tokens_requested",
    "warmup_runs",
    "formal_runs_requested",
    "formal_runs_ok",
    "formal_runs_failed",
    "首令牌延迟(ms)/TTFT(ms)",
    "每令牌时间(ms/token)/TPOT(ms/token)",
    "令牌间延迟(ms/token)/ITL(ms/token)",
    "端到端耗时(s)/E2E(s)",
    "峰值显存(GB)/Peak Memory(GB)",
]

PERF_RE = re.compile(
    r"Prompt:\s*([0-9]+(?:\.[0-9]+)?)\s*t/s\s*\|\s*Generation:\s*([0-9]+(?:\.[0-9]+)?)\s*t/s",
    re.IGNORECASE,
)


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
            return read_memory_command(self.memory_command, self.device_index)
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
    parser = argparse.ArgumentParser(description="Run the formal llama-cli long-context baseline")
    parser.add_argument("--llama-cli", default=os.environ.get("LLAMA_CLI"), help="path to llama-cli")
    parser.add_argument("--model", default=os.environ.get("LLAMA_MODEL"), help="path to the GGUF model")
    parser.add_argument("--config", default=str(DEFAULT_BENCHMARK_CONFIG), help="benchmark case configuration")
    parser.add_argument("--platform-config", default=str(DEFAULT_PLATFORM_CONFIG), help="platform profiles")
    parser.add_argument("--processed-dir", default=str(DEFAULT_PROCESSED_DIR))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--output-dir", help="exact output directory instead of a timestamped directory")
    parser.add_argument("--platform", default="auto", help="platform profile name or auto")
    parser.add_argument("--device-index", type=int, default=0, help="physical accelerator index")
    parser.add_argument("--backend-device", help="llama.cpp device after visibility filtering")
    parser.add_argument("--case", action="append", default=[], help="case id or comma-separated case ids")
    parser.add_argument("--warmup-runs", type=int, help="override manifest warmup count")
    parser.add_argument("--formal-runs", type=int, help="override manifest formal count")
    parser.add_argument("--n-ctx", type=int)
    parser.add_argument("--n-batch", type=int)
    parser.add_argument("--n-ubatch", type=int)
    parser.add_argument("--n-gpu-layers", type=int)
    parser.add_argument("--parallel", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--type-k")
    parser.add_argument("--type-v")
    parser.add_argument("--flash-attn", choices=["on", "off", "auto"])
    parser.add_argument("--poll", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--monitor-interval", type=float, default=0.1)
    parser.add_argument(
        "--memory-command",
        help="command printing used device memory in MiB; {device} is replaced",
    )
    parser.add_argument("--timeout", type=float, default=0.0, help="timeout per turn in seconds (0 = no timeout)")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--extra-arg", action="append", default=[], help="extra llama-cli argument")
    parser.add_argument("--model-sha256", help="precomputed model SHA-256")
    parser.add_argument("--skip-model-hash", action="store_true")
    parser.add_argument("--skip-prompt-hash-check", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="validate and print commands without running")
    parser.add_argument("--list-cases", action="store_true")
    return parser.parse_args()


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
        prompt_tokens = int(item["prompt_tokens"])
        output_tokens = int(item["output_tokens"])
        if prompt_tokens <= 0 or output_tokens <= 0:
            raise ValueError("case {} has non-positive token counts".format(case_id))
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


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def choose_selections(manifest, warmup_runs, formal_runs):
    grouped = {
        "warmup": [item for item in manifest["selections"] if item.get("phase") == "warmup"],
        "formal": [item for item in manifest["selections"] if item.get("phase") == "formal"],
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


def validate_prompts(processed_dir, cases, selected_cases, selections, metadata, check_hashes):
    validated = {}
    for case_id in selected_cases:
        target = cases[case_id]["prompt_tokens"]
        for _, _, selection in selections:
            prompt_key = str(target)
            if prompt_key not in selection.get("prompt_files", {}):
                raise ValueError("manifest run {} has no {} prompt".format(selection.get("run"), target))
            relative = selection["prompt_files"][prompt_key]
            path = processed_dir / relative
            if not path.is_file():
                raise ValueError("prompt does not exist: {}".format(path))
            if relative not in metadata:
                raise ValueError("prompt is missing from metadata: {}".format(relative))
            item = metadata[relative]
            if item["source_id"] != selection["source_id"]:
                raise ValueError("source_id mismatch for {}".format(relative))
            if int(item["target_tokens"]) != target:
                raise ValueError("target token mismatch for {}".format(relative))
            if check_hashes and relative not in validated:
                actual = sha256_file(path)
                if actual != item["prompt_sha256"]:
                    raise ValueError("SHA-256 mismatch for {}".format(path))
                validated[relative] = actual


def build_session_command(args, case, model_path, backend_device, n_gpu_layers):
    """One llama-cli process per case; all prompts are fed through stdin.

    --single-turn is deliberately absent so the main loop stays alive across
    turns, and -mli enables multiline input so each stdin block is one prompt.
    The -f flag is not used because it would load a single prompt at startup.
    """
    command = [
        str(Path(args.llama_cli)),
        "-m", str(model_path),
        "-n", str(case["output_tokens"]),
        "-c", str(args.n_ctx),
        "-b", str(args.n_batch),
        "-ub", str(args.n_ubatch),
        "-ngl", str(n_gpu_layers),
        "-dev", backend_device,
        "--parallel", str(args.parallel),
        "--temp", str(args.temperature),
        "--seed", str(args.seed),
        "--ignore-eos",
        "--cache-type-k", args.type_k,
        "--cache-type-v", args.type_v,
        "--flash-attn", args.flash_attn,
        "--poll", str(args.poll),
        "-mli",
        "--perf",
        "--show-timings",
        "--simple-io",
        "--no-display-prompt",
        "--no-warmup",
        "--color", "off",
        "--log-colors", "off",
    ]
    if args.threads is not None:
        command.extend(["--threads", str(args.threads)])
    command.extend(args.extra_arg)
    return command


class StreamReader(threading.Thread):
    """Reads one llama-cli output stream line by line into shared lists.

    Every line is timestamped when it arrives and tagged with whether it is a
    per-turn '[ Prompt: ... t/s | Generation: ... t/s ]' line, so the main
    thread can wait for a specific turn and measure its wall time precisely.
    """

    def __init__(self, stream, condition, echo=None):
        super().__init__(daemon=True)
        self.stream = stream
        self.condition = condition
        self.echo = echo
        self.lines = []
        self.times = []
        self.timing = []
        self.eof = False

    def run(self):
        try:
            for raw in self.stream:
                line = raw.decode("utf-8", errors="replace")
                if self.echo is not None:
                    self.echo.write(line)
                with self.condition:
                    self.lines.append(line)
                    self.times.append(time.monotonic())
                    self.timing.append(PERF_RE.search(line) is not None)
                    self.condition.notify_all()
        finally:
            with self.condition:
                self.eof = True
                self.condition.notify_all()
            if self.echo is not None:
                self.echo.flush()


def wait_for_turn(stdout_reader, expected_matches, proc, timeout):
    """Wait until expected_matches timing lines are seen on stdout.

    Returns "ok", "timeout", or "exited" (the process ended before the lines).
    """
    deadline = None if timeout <= 0 else time.monotonic() + timeout
    with stdout_reader.condition:
        while True:
            if sum(stdout_reader.timing) >= expected_matches:
                return "ok"
            if proc.poll() is not None and stdout_reader.eof:
                return "exited"
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return "timeout"
                stdout_reader.condition.wait(timeout=remaining)
            else:
                stdout_reader.condition.wait(timeout=0.5)


def timing_line_at(stdout_reader, match_index):
    """Return (arrival_time, prompt_tps, generation_tps) of the match_index-th timing line."""
    with stdout_reader.condition:
        seen = 0
        for i, is_timing in enumerate(stdout_reader.timing):
            if not is_timing:
                continue
            if seen == match_index:
                match = PERF_RE.search(stdout_reader.lines[i])
                return stdout_reader.times[i], float(match.group(1)), float(match.group(2))
            seen += 1
    raise ValueError("timing line {} not found".format(match_index))


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


def run_session(
    args,
    case_id,
    case,
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
):
    """Run all selections of one case inside a single llama-cli process.

    Prompts are fed through stdin as consecutive chat turns in the same context
    window, so the KV cache persists across the runs and only the first warmup
    turn pays model loading. Each turn's '[ Prompt: ... | Generation: ... ]'
    line is parsed for per-turn throughput.
    """
    run_dir = output_dir / case_id
    run_dir.mkdir(parents=True, exist_ok=True)
    command = build_session_command(args, case, model_path, backend_device, n_gpu_layers)
    command_text = shlex.join(command)
    (run_dir / "session.command.txt").write_text(command_text + "\n", encoding="utf-8")

    nvml = None
    baseline_device_mib = None
    device_monitor_enabled = False
    if memory_monitor == "nvml":
        try:
            nvml = NvmlMemoryReader(args.device_index)
            baseline_device_mib = nvml.read_mib()
            device_monitor_enabled = True
        except Exception as exc:
            print("warning: NVML monitoring unavailable: {}".format(exc), file=sys.stderr)
    if nvml is None and args.memory_command:
        try:
            baseline_device_mib = read_memory_command(args.memory_command, args.device_index)
            device_monitor_enabled = True
        except Exception as exc:
            print("warning: device memory monitoring unavailable: {}".format(exc), file=sys.stderr)

    records = []
    proc = None
    stdout_reader = None
    stderr_reader = None
    monitor = None
    session_started = time.monotonic()
    try:
        with (run_dir / "session.stdout.log").open("w", encoding="utf-8") as session_stdout, \
             (run_dir / "session.stderr.log").open("w", encoding="utf-8") as session_stderr:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
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

            prev_turn_time = session_started
            prev_stdout_idx = 0
            prev_stderr_idx = 0
            turn_failed = False
            session_error = ""

            for phase, phase_run, selection in selections:
                relative = selection["prompt_files"][str(case["prompt_tokens"])]
                prompt_path = processed_dir / relative
                prompt_meta = metadata[relative]
                run_name = "{}-{:02d}".format(phase, phase_run)
                turn_stdout_path = run_dir / "{}.stdout.log".format(run_name)
                turn_stderr_path = run_dir / "{}.stderr.log".format(run_name)

                prompt_tps = None
                generation_tps = None
                wall_time = None
                timed_out = False
                error = ""

                if not turn_failed and proc.poll() is None:
                    prompt_text = prompt_path.read_text(encoding="utf-8")
                    if prompt_text.endswith("\n"):
                        prompt_text = prompt_text[:-1]
                    # A trailing backslash line toggles multiline input off and
                    # ends the turn; it contributes one trailing newline token.
                    proc.stdin.write(prompt_text.encode("utf-8") + b"\\\n")
                    proc.stdin.flush()
                    expected = sum(stdout_reader.timing) + 1
                    result = wait_for_turn(stdout_reader, expected, proc, args.timeout)
                    if result == "ok":
                        match_time, prompt_tps, generation_tps = timing_line_at(
                            stdout_reader, expected - 1
                        )
                        wall_time = match_time - prev_turn_time
                        prev_turn_time = match_time
                    elif result == "timeout":
                        timed_out = True
                        terminate_process(proc)
                        error = "timeout"
                        session_error = error
                        turn_failed = True
                        wall_time = time.monotonic() - prev_turn_time
                    else:
                        error = "llama-cli exited with status {}".format(proc.returncode)
                        session_error = error
                        turn_failed = True
                        wall_time = time.monotonic() - prev_turn_time
                else:
                    error = "skipped: {}".format(session_error) if session_error else "skipped: process terminated"
                    wall_time = 0.0

                # slice this turn's lines out of the session logs
                with stdout_reader.condition:
                    turn_stdout = "".join(stdout_reader.lines[prev_stdout_idx:])
                    turn_stderr = "".join(stderr_reader.lines[prev_stderr_idx:])
                    prev_stdout_idx = len(stdout_reader.lines)
                    prev_stderr_idx = len(stderr_reader.lines)
                turn_stdout_path.write_text(turn_stdout, encoding="utf-8")
                turn_stderr_path.write_text(turn_stderr, encoding="utf-8")

                record = {
                    "case_id": case_id,
                    "status": "ok" if (prompt_tps is not None and generation_tps is not None) else "failed",
                    "phase": phase,
                    "phase_run": phase_run,
                    "manifest_run": selection["run"],
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
                    # the stdin protocol adds one trailing newline token to every prompt
                    "actual_prompt_tokens": int(prompt_meta["actual_tokens"]) + 1,
                    "output_tokens_requested": case["output_tokens"],
                    "wall_time_s_measured": wall_time,
                    "peak_process_rss_mib_measured": monitor.peak_rss_mib,
                    "device_memory_baseline_mib_measured": baseline_device_mib,
                    "peak_device_memory_mib_measured": monitor.peak_device_mib if device_monitor_enabled else None,
                    "device_memory_delta_mib_measured": (
                        max(0.0, monitor.peak_device_mib - baseline_device_mib) if device_monitor_enabled else None
                    ),
                    "returncode": None,
                    "timed_out": timed_out,
                    "monitor_error": monitor.error,
                    "error": error,
                    "command": command_text,
                }
                if record["status"] == "ok":
                    prompt_eval_ms = record["actual_prompt_tokens"] / prompt_tps * 1000.0
                    tpot_ms = 1000.0 / generation_tps
                    generation_eval_ms = case["output_tokens"] / generation_tps * 1000.0
                    record.update({
                        "prompt_tps_measured": prompt_tps,
                        "generation_tps_measured": generation_tps,
                        "prompt_eval_ms_est": prompt_eval_ms,
                        "tpot_ms_est": tpot_ms,
                        "itl_ms_est": tpot_ms,
                        "generation_eval_ms_est": generation_eval_ms,
                        "ttft_ms_est": prompt_eval_ms + tpot_ms,
                        "compute_e2e_ms_est": prompt_eval_ms + generation_eval_ms,
                    })
                records.append(record)

            if proc.poll() is None and not turn_failed:
                try:
                    proc.stdin.write(b"/exit\n")
                    proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
            try:
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                terminate_process(proc)
            monitor.stop()
    except KeyboardInterrupt:
        if proc is not None and proc.poll() is None:
            terminate_process(proc)
        raise
    finally:
        if monitor is not None:
            monitor.stop()
        if stdout_reader is not None and stdout_reader.is_alive():
            stdout_reader.join(timeout=2)
        if stderr_reader is not None and stderr_reader.is_alive():
            stderr_reader.join(timeout=2)
        if nvml is not None:
            nvml.close()

    returncode = proc.returncode if proc is not None else None
    for record in records:
        record["returncode"] = returncode
    return records


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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


def summarize_case(case_id, case, records, warmup_runs, formal_runs, platform_name, device_index, backend_device):
    formal_records = [record for record in records if record["phase"] == "formal"]
    ok_records = [record for record in formal_records if record["status"] == "ok"]
    metrics = {}
    for field in (
        "prompt_tps_measured",
        "generation_tps_measured",
        "prompt_eval_ms_est",
        "tpot_ms_est",
        "itl_ms_est",
        "generation_eval_ms_est",
        "ttft_ms_est",
        "compute_e2e_ms_est",
        "wall_time_s_measured",
        "peak_process_rss_mib_measured",
        "peak_device_memory_mib_measured",
        "device_memory_delta_mib_measured",
    ):
        metrics[field] = metric_summary(ok_records, field)

    def rounded(summary_dict):
        return None if not summary_dict else round(summary_dict["mean"], 2)

    summary = {
        "case_id": case_id,
        "status": "ok" if len(ok_records) == formal_runs else "incomplete",
        "platform": platform_name,
        "physical_device_index": device_index,
        "backend_device": backend_device,
        "target_prompt_tokens": case["prompt_tokens"],
        "output_tokens_requested": case["output_tokens"],
        "warmup_runs": warmup_runs,
        "formal_runs_requested": formal_runs,
        "formal_runs_ok": len(ok_records),
        "formal_runs_failed": len(formal_records) - len(ok_records),
        "metrics": metrics,
    }
    # the five headline metrics, bilingual headers, one row per case
    peak = metrics.get("peak_device_memory_mib_measured")
    summary["首令牌延迟(ms)/TTFT(ms)"] = rounded(metrics.get("ttft_ms_est"))
    summary["每令牌时间(ms/token)/TPOT(ms/token)"] = rounded(metrics.get("tpot_ms_est"))
    summary["令牌间延迟(ms/token)/ITL(ms/token)"] = rounded(metrics.get("itl_ms_est"))
    summary["端到端耗时(s)/E2E(s)"] = rounded(metrics.get("wall_time_s_measured"))
    summary["峰值显存(GB)/Peak Memory(GB)"] = (
        None if not peak else round(peak["max"] / 1024.0, 2)
    )
    return summary


def command_version(path):
    try:
        result = subprocess.run(
            [str(path), "--version"],
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
    args = parse_args()
    benchmark_config_path, benchmark_config, cases = load_benchmark_config(args.config)
    platform_config_path, platform_config, profiles = load_platform_profiles(args.platform_config)
    apply_runtime_defaults(args, benchmark_config["runtime"])
    if args.list_cases:
        for case_id, case in cases.items():
            print("{}\t{} input tokens\t{} output tokens".format(case_id, case["prompt_tokens"], case["output_tokens"]))
        return 0

    selected_cases = select_cases(args.case, cases)
    processed_dir = Path(args.processed_dir).resolve()
    manifest, preparation, metadata = load_inputs(processed_dir)
    warmup_runs = manifest["warmup_runs"] if args.warmup_runs is None else args.warmup_runs
    formal_runs = manifest["formal_runs"] if args.formal_runs is None else args.formal_runs
    selections = choose_selections(manifest, warmup_runs, formal_runs)
    validate_prompts(
        processed_dir,
        cases,
        selected_cases,
        selections,
        metadata,
        check_hashes=not args.skip_prompt_hash_check,
    )

    if not args.llama_cli:
        raise ValueError("--llama-cli or LLAMA_CLI is required")
    if not args.model:
        raise ValueError("--model or LLAMA_MODEL is required")
    llama_cli = Path(args.llama_cli).resolve()
    model_path = Path(args.model).resolve()
    if not llama_cli.is_file() or not os.access(llama_cli, os.X_OK):
        raise ValueError("llama-cli is not executable: {}".format(llama_cli))
    if not model_path.is_file():
        raise ValueError("model does not exist: {}".format(model_path))
    if args.monitor_interval <= 0:
        raise ValueError("--monitor-interval must be positive")
    if args.model_sha256 is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", args.model_sha256):
        raise ValueError("--model-sha256 must contain 64 hexadecimal characters")
    args.llama_cli = str(llama_cli)

    platform_name = detect_platform(args.platform, profiles)
    platform_profile = profiles[platform_name]
    platform_env, backend_device, n_gpu_layers, memory_monitor = platform_settings(
        platform_profile, args.device_index, args.backend_device, args.n_gpu_layers
    )
    if args.memory_command is None:
        args.memory_command = platform_profile.get("memory_command")
    extra_env = parse_assignments(args.env, "--env")
    environment = os.environ.copy()
    environment.pop("LLAMA_DUMP_GRAPH", None)
    environment.update(platform_env)
    environment.update(extra_env)

    if args.dry_run:
        env_text = " ".join("{}={}".format(key, shlex.quote(value)) for key, value in {**platform_env, **extra_env}.items())
        print(
            "validated {} case(s), {} prompt file(s); each case runs as one llama-cli session".format(
                len(selected_cases), len(selected_cases) * len(selections)
            )
        )
        for case_id in selected_cases:
            case = cases[case_id]
            command = build_session_command(args, case, model_path, backend_device, n_gpu_layers)
            print(
                "\n[{}] one process, {} turns via stdin ({} warmup + {} formal)".format(
                    case_id, len(selections), warmup_runs, formal_runs
                )
            )
            prefix = (env_text + " ") if env_text else ""
            print("{}{}".format(prefix, shlex.join(command)))
        return 0

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = Path(args.results_dir).resolve() / "{}-llama-cli-{}-device{}".format(
            timestamp, platform_name, args.device_index
        )
    if output_dir.exists():
        raise ValueError("output directory already exists: {}".format(output_dir))
    output_dir.mkdir(parents=True)

    model_sha256 = args.model_sha256
    if model_sha256 is not None:
        model_sha256 = model_sha256.lower()
    elif not args.skip_model_hash:
        print("calculating model SHA-256...", flush=True)
        model_sha256 = sha256_file(model_path)
    run_manifest = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": host_platform.node(),
        "host_platform": host_platform.platform(),
        "python": sys.version,
        "llama_cli": str(llama_cli),
        "llama_cli_version": command_version(llama_cli),
        "model": str(model_path),
        "model_size": model_path.stat().st_size,
        "model_sha256": model_sha256,
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
            "ignore_eos": True,
            "prefix_cache_hit_rate_pct": 0,
            "warmup_runs": warmup_runs,
            "formal_runs": formal_runs,
            "extra_args": args.extra_arg,
        },
        "metric_notes": {
            "execution_mode": "one llama-cli process per case; prompts are fed through stdin as consecutive chat turns in the same context window, so the KV cache persists across runs",
            "prompt_artifact": "the stdin multiline protocol adds one trailing newline token to every prompt",
            "summary_columns": "bilingual 中文/English; TTFT(ms)=首令牌延迟, TPOT(ms/token)=每令牌时间, ITL(ms/token)=令牌间延迟, E2E(s)=端到端耗时, Peak Memory(GB)=峰值显存",
            "ttft_ms_est": "estimated time to first token: prompt_eval_ms_est + tpot_ms_est (llama-cli reports throughput, not a true TTFT)",
            "tpot_ms_est": "time per output token, 1000 / generation_tps",
            "itl_ms_est": "inter-token latency, 1000 / generation_tps (same value as tpot_ms_est by definition)",
            "wall_time_s_measured": "E2E: per-turn wall time between consecutive timing lines, measured; the first warmup turn includes model loading",
            "peak_device_memory_mib_measured": "whole physical device usage, not process-specific; reported as 峰值显存(GB)/Peak Memory(GB) using the maximum",
            "prompt_tps_measured": "parsed directly from llama-cli per-turn timing lines",
            "generation_tps_measured": "parsed directly from llama-cli per-turn timing lines",
            "derived_fields": "fields ending in _est are estimates derived from throughput",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    records = []
    runs_jsonl = output_dir / "runs.jsonl"
    with runs_jsonl.open("w", encoding="utf-8") as jsonl:
        for case_id in selected_cases:
            print(
                "[{}] one llama-cli session, {} turns".format(case_id, len(selections)),
                flush=True,
            )
            case_records = run_session(
                args,
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
            records.extend(case_records)
            for record in case_records:
                jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
                jsonl.flush()
                print(
                    "[{}] {}-{:02d} manifest-run={} source={}: {}".format(
                        case_id,
                        record["phase"],
                        record["phase_run"],
                        record["manifest_run"],
                        record["source_id"],
                        record["status"],
                    ),
                    flush=True,
                )
            if args.fail_fast and any(record["status"] != "ok" for record in case_records):
                break

    summaries = []
    for case_id in selected_cases:
        case_records = [record for record in records if record["case_id"] == case_id]
        summaries.append(
            summarize_case(
                case_id,
                cases[case_id],
                case_records,
                warmup_runs,
                formal_runs,
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
    print("\n== summary (中文/English) ==")
    print("\t".join(SUMMARY_FIELDS))
    for summary in summaries:
        print("\t".join(str(summary.get(field, "")) for field in SUMMARY_FIELDS))
    print("results: {}".format(output_dir))
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
