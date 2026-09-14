#!/usr/bin/env python3
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time
import urllib.request

from bench_server_run import free_port, server_command, stop_server, wait_for_server


CASES = {
    "baseline": {},
    "kv-q8": {"cache_type_k": "q8_0", "cache_type_v": "q8_0"},
    "kv-q4": {"cache_type_k": "q4_0", "cache_type_v": "q4_0"},
    "fa-off": {"flash_attn": "off"},
    "poll-0": {"poll": 0},
    "graph-off": {"_environment": {"GGML_CUDA_DISABLE_GRAPHS": "1"}},
    "threads-4": {"threads": 4},
    "threads-8": {"threads": 8},
    "threads-8-backend-sampling": {"threads": 8, "_server_args": ["--backend-sampling"]},
    "threads-4-backend-sampling": {"threads": 4, "_server_args": ["--backend-sampling"]},
    "threads-4-batch16-backend-sampling": {
        "threads": 4,
        "threads_batch": 16,
        "_server_args": ["--backend-sampling"],
    },
    "threads-4-batch16-backend-sampling-ubatch1024": {
        "threads": 4,
        "threads_batch": 16,
        "n_ubatch": 1024,
        "_server_args": ["--backend-sampling"],
    },
    "threads-4-batch16-backend-sampling-ubatch2048": {
        "threads": 4,
        "threads_batch": 16,
        "n_ubatch": 2048,
        "_server_args": ["--backend-sampling"],
    },
    "threads-4-batch16-backend-sampling-mtp-n2": {
        "threads": 4,
        "threads_batch": 16,
        "spec_type": "draft-mtp",
        "draft_n_max": 2,
        "_server_args": ["--backend-sampling"],
    },
    "threads-4-batch16-backend-sampling-mtp-n3": {
        "threads": 4,
        "threads_batch": 16,
        "spec_type": "draft-mtp",
        "draft_n_max": 3,
        "_server_args": ["--backend-sampling"],
    },
    "threads-4-batch16-backend-sampling-dflash-n4": {
        "threads": 4,
        "threads_batch": 16,
        "draft_n_max": 4,
        "_requires_draft_model": True,
        "_server_args": ["--backend-sampling"],
    },
    "threads-4-batch16-backend-sampling-dflash-n8": {
        "threads": 4,
        "threads_batch": 16,
        "draft_n_max": 8,
        "_requires_draft_model": True,
        "_server_args": ["--backend-sampling"],
    },
    "threads-4-batch16-backend-sampling-dflash-n15": {
        "threads": 4,
        "threads_batch": 16,
        "draft_n_max": 15,
        "_requires_draft_model": True,
        "_server_args": ["--backend-sampling"],
    },
    "threads-16-backend-sampling": {"threads": 16, "_server_args": ["--backend-sampling"]},
    "threads-8-no-cont-batching": {"threads": 8, "_server_args": ["--no-cont-batching"]},
    "threads-8-kv-q8": {"threads": 8, "cache_type_k": "q8_0", "cache_type_v": "q8_0"},
    "threads-8-backend-sampling-poll-0": {"threads": 8, "poll": 0, "_server_args": ["--backend-sampling"]},
    "threads-8-backend-sampling-kv-q8": {
        "threads": 8,
        "cache_type_k": "q8_0",
        "cache_type_v": "q8_0",
        "_server_args": ["--backend-sampling"],
    },
    "threads-8-backend-sampling-no-cont-batching": {
        "threads": 8,
        "_server_args": ["--backend-sampling", "--no-cont-batching"],
    },
    "threads-4-backend-sampling-no-cont-batching": {
        "threads": 4,
        "_server_args": ["--backend-sampling", "--no-cont-batching"],
    },
}

for draft_n_max in (1, 2, 3):
    CASES.setdefault(f"threads-4-batch16-backend-sampling-mtp-n{draft_n_max}", {
        "threads": 4,
        "threads_batch": 16,
        "spec_type": "draft-mtp",
        "draft_n_max": draft_n_max,
        "_server_args": ["--backend-sampling"],
    })

for draft_n_max in (*range(2, 11), 12, 15):
    CASES.setdefault(f"threads-4-batch16-backend-sampling-dflash-n{draft_n_max}", {
        "threads": 4,
        "threads_batch": 16,
        "draft_n_max": draft_n_max,
        "_requires_draft_model": True,
        "_server_args": ["--backend-sampling"],
    })


class Memory(ctypes.Structure):
    _fields_ = [(k, ctypes.c_ulonglong) for k in ("total", "free", "used")]


class Utilization(ctypes.Structure):
    _fields_ = [(k, ctypes.c_uint) for k in ("gpu", "memory")]


class Monitor:
    def __init__(self, device):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        self.handle = ctypes.c_void_p()
        if self.lib.nvmlInit_v2() or self.lib.nvmlDeviceGetHandleByIndex_v2(device, ctypes.byref(self.handle)):
            raise RuntimeError("NVML initialization failed")
        self.stop = threading.Event()
        self.samples = []

    def read(self):
        mem, util = Memory(), Utilization()
        if self.lib.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(mem)):
            raise RuntimeError("NVML memory query failed")
        rc = self.lib.nvmlDeviceGetUtilizationRates(self.handle, ctypes.byref(util))
        return {"time": time.time(), "used_mib": mem.used / 1048576, "free_mib": mem.free / 1048576,
                "gpu_util_pct": util.gpu if rc == 0 else None}

    def loop(self):
        while not self.stop.is_set():
            try:
                self.samples.append(self.read())
            except RuntimeError:
                pass
            self.stop.wait(0.2)


def request(base, row, args):
    body = {"model": "benchmark", "messages": row["messages"], "max_tokens": args.max_tokens,
            "temperature": args.temp, "top_p": 0.95, "seed": args.seed,
            "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False},
            "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    chunks, output, timings, usage = [], [], None, None
    finish = None
    with urllib.request.urlopen(req, timeout=1800) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            timings = event.get("timings") or timings
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                text = choice.get("delta", {}).get("content") or ""
                if text:
                    chunks.append((time.perf_counter() - start) * 1000)
                    output.append(text)
                finish = choice.get("finish_reason") or finish
    wall_ms = (time.perf_counter() - start) * 1000
    if not timings or not usage or not chunks:
        raise RuntimeError("Missing server timings, token usage, or output chunks")
    n = usage["completion_tokens"]
    return {"ttft_client_ms": chunks[0], "e2e_client_ms": wall_ms,
            "tpot_client_ms": (chunks[-1] - chunks[0]) / (n - 1) if n > 1 else None,
            "decode_ms_per_token": timings["predicted_per_token_ms"],
            "chunk_arrival_ms": chunks, "timings": timings, "usage": usage,
            "finish_reason": finish, "output": "".join(output)}


def select_stratified(rows, limit, output_tokens_by_id=None):
    groups = {}
    for row in rows:
        output_tokens = (output_tokens_by_id or {}).get(row["id"], row["recorded_output_tokens"])
        if output_tokens <= 64:
            output_band = "00064"
        elif output_tokens <= 256:
            output_band = "00256"
        elif output_tokens <= 1024:
            output_band = "01024"
        else:
            output_band = "large"
        key = (row["length_bucket"], row["stage"], output_band)
        groups.setdefault(key, []).append(row)

    selected = []
    keys = sorted(groups)
    while len(selected) < limit:
        added = False
        for key in keys:
            if groups[key] and len(selected) < limit:
                selected.append(groups[key].pop(0))
                added = True
        if not added:
            break
    return sorted(selected, key=lambda row: row["order"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=4)
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=18000)
    parser.add_argument("--measure-limit", type=int, default=0)
    parser.add_argument("--measure-selection", choices=("head", "stratified"), default="head")
    parser.add_argument("--selection-reference", type=Path,
                        help="Prior requests.jsonl used to stratify by actual completion length")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--threads-batch", type=int)
    parser.add_argument("--draft-threads", type=int)
    parser.add_argument("--draft-threads-batch", type=int)
    parser.add_argument("--draft-n-min", type=int, default=0)
    parser.add_argument("--draft-p-min", type=float, default=0.0)
    parser.add_argument("--backend-sampling", choices=("case", "on", "off"), default="case")
    parser.add_argument("--draft-backend-sampling", choices=("on", "off"), default="on")
    args = parser.parse_args()
    if not 0.0 <= args.draft_p_min <= 1.0:
        raise ValueError("draft-p-min must be between 0 and 1")
    if args.draft_n_min < 0:
        raise ValueError("draft-n-min must be non-negative")
    if args.output_dir.exists() and not args.resume:
        raise ValueError("Output directory exists; use a new directory")
    rows = [json.loads(s) for s in args.dataset.read_text().splitlines() if s.strip()]
    if len(rows) != 205 or sum(r["split"] == "warmup" for r in rows) != 5:
        raise ValueError("Expected 5 warmup and 200 measured requests")
    if len({r["messages_sha256"] for r in rows}) != 205:
        raise ValueError("Duplicate inputs")
    if args.measure_limit < 0 or args.measure_limit > 200:
        raise ValueError("measure-limit must be between 0 and 200")
    if args.measure_limit:
        measured = [r for r in rows if r["split"] == "measure"]
        if args.measure_selection == "stratified":
            output_tokens_by_id = None
            if args.selection_reference:
                reference_rows = [json.loads(s) for s in args.selection_reference.read_text().splitlines() if s.strip()]
                output_tokens_by_id = {
                    row["id"]: int(row["usage"]["completion_tokens"])
                    for row in reference_rows if row.get("split") == "measure"
                }
                missing = [row["id"] for row in measured if row["id"] not in output_tokens_by_id]
                if missing:
                    raise ValueError(f"Selection reference is missing {len(missing)} measured request IDs")
            measured = select_stratified(measured, args.measure_limit, output_tokens_by_id)
        else:
            measured = measured[:args.measure_limit]
        rows = [r for r in rows if r["split"] == "warmup"] + measured
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    if args.smoke:
        rows = [{**max(rows, key=lambda r: r["recorded_input_tokens"]), "split": "smoke"}]
        args.max_tokens = 8
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.device)
    env.pop("GGML_CUDA_DISABLE_GRAPHS", None)
    monitor = Monitor(args.device)
    if monitor.read()["free_mib"] < 28000:
        raise RuntimeError("Selected GPU has insufficient free memory for an isolated baseline")
    manifest = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                "selection_reference_sha256": hashlib.sha256(args.selection_reference.read_bytes()).hexdigest()
                    if args.selection_reference else None,
                "request_ids": [row["id"] for row in rows],
                "model_bytes": Path(args.model).stat().st_size,
                "version": subprocess.run([args.llama_server, "--version"], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout,
                "metrics": "Client TTFT: first nonempty content chunk; TPOT: first-to-last chunk / (completion_tokens-1). Chunk timestamps are not token-level ITL. NVML measurements are device-wide.",
                "runs": []}
    if args.resume:
        previous = json.loads((args.output_dir / "manifest.json").read_text())
        for key in ("dataset_sha256", "selection_reference_sha256", "request_ids", "model_bytes", "version"):
            if previous[key] != manifest[key]:
                raise ValueError("Resume mismatch: " + key)
        current_args = {k: v for k, v in manifest["arguments"].items() if k != "resume"}
        previous_args = {k: v for k, v in previous["arguments"].items() if k != "resume"}
        if current_args != previous_args:
            raise ValueError("Resume arguments differ")
        manifest = previous
        manifest.setdefault("resumptions", []).append(time.time())
    for case in args.cases:
        case_dir = args.output_dir / case
        case_dir.mkdir(exist_ok=args.resume)
        results = []
        record_path = case_dir / "requests.jsonl"
        if args.resume and record_path.exists():
            results = [json.loads(s) for s in record_path.read_text().splitlines() if s.strip()]
            if len(results) > len(rows):
                raise ValueError("Too many existing results")
            for old, row in zip(results, rows):
                if any(old[k] != row[k] for k in ("id", "order", "split", "messages_sha256")):
                    raise ValueError("Resume sample mismatch")
            summary_path = case_dir / "summary.json"
            if len(results) == len(rows) and summary_path.exists():
                if json.loads(summary_path.read_text())["status"] == "completed":
                    print(f"{case}: already completed", flush=True)
                    continue
        settings = dict(n_ctx=65536, n_batch=2048, n_ubatch=512, n_gpu_layers=99,
                        cache_type_k="f16", cache_type_v="f16", flash_attn="on", poll=50,
                        threads=16, threads_batch=None, draft_model=None, draft_ngl=99,
                        draft_n_max=3, spec_type="none", temp=args.temperature, seed=42)
        case_options = CASES[case]
        settings.update({key: value for key, value in case_options.items() if not key.startswith("_")})
        if args.threads is not None:
            settings["threads"] = args.threads
        if args.threads_batch is not None:
            settings["threads_batch"] = args.threads_batch
        if case_options.get("_requires_draft_model"):
            if not args.draft_model:
                raise ValueError(f"{case} requires --draft-model")
            settings["draft_model"] = args.draft_model
        cfg_values = vars(args).copy()
        cfg_values.update(settings)
        cfg = argparse.Namespace(**cfg_values)
        port = free_port()
        threads_batch = cfg.threads if cfg.threads_batch is None else cfg.threads_batch
        command = server_command(cfg, port) + ["--threads-batch", str(threads_batch), "--threads-http", "2",
                                              "--fit", "off", "--no-context-shift"]
        if not cfg.draft_model:
            command.extend(["--spec-type", cfg.spec_type])
            if cfg.spec_type != "none":
                command.extend(["--spec-draft-n-max", str(cfg.draft_n_max)])
        if cfg.spec_type != "none" or cfg.draft_model:
            command.extend(["--spec-draft-n-min", str(args.draft_n_min),
                            "--spec-draft-p-min", str(args.draft_p_min)])
            if args.draft_threads is not None:
                command.extend(["--spec-draft-threads", str(args.draft_threads)])
            if args.draft_threads_batch is not None:
                command.extend(["--spec-draft-threads-batch", str(args.draft_threads_batch)])
            command.append("--spec-draft-backend-sampling" if args.draft_backend_sampling == "on"
                           else "--no-spec-draft-backend-sampling")
        server_args = list(case_options.get("_server_args", []))
        if args.backend_sampling == "on" and "--backend-sampling" not in server_args:
            server_args.append("--backend-sampling")
        elif args.backend_sampling == "off":
            server_args = [arg for arg in server_args if arg != "--backend-sampling"]
        command.extend(server_args)
        case_env = env.copy()
        case_env.update(case_options.get("_environment", {}))
        manifest["runs"].append({"case": case, "command": command,
                                 "environment": {k: case_env.get(k) for k in ("CUDA_VISIBLE_DEVICES", "GGML_CUDA_DISABLE_GRAPHS")}})
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        status = "completed"
        proc = None
        with (case_dir / "server.log").open("a") as log, record_path.open("a") as records:
            try:
                proc = subprocess.Popen(command, env=case_env, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
                wait_for_server(f"http://127.0.0.1:{port}", proc)
                print(f"{case}: server ready pid={proc.pid}", flush=True)
                if results and len(results) < len(rows):
                    with (case_dir / "resume-warmup.jsonl").open("a") as rewarm:
                        for row in rows[:5]:
                            extra = request(f"http://127.0.0.1:{port}", row, cfg)
                            extra.update({"id": row["id"], "split": "resume_warmup"})
                            rewarm.write(json.dumps(extra, ensure_ascii=False) + "\n")
                            rewarm.flush()
                    print(f"{case}: resume warmup done; retained {len(results)} results", flush=True)
                for row in rows[len(results):]:
                    monitor.samples = []
                    monitor.stop.clear()
                    thread = threading.Thread(target=monitor.loop)
                    thread.start()
                    try:
                        result = request(f"http://127.0.0.1:{port}", row, cfg)
                    finally:
                        monitor.stop.set()
                        thread.join()
                    result.update({"id": row["id"], "order": row["order"], "split": row["split"],
                                   "messages_sha256": row["messages_sha256"], "device_samples": monitor.samples})
                    records.write(json.dumps(result, ensure_ascii=False) + "\n")
                    records.flush()
                    results.append(result)
                    print(f"{case}: {row['split']} {len(results)}/{len(rows)} output={result['usage']['completion_tokens']} TPOT={result['decode_ms_per_token']:.2f} ms/token E2E={result['e2e_client_ms']/1000:.1f}s", flush=True)
            except Exception as exc:
                status = "failed"
                (case_dir / "error.txt").write_text(str(exc))
                print(f"{case}: FAILED {exc}", flush=True)
            finally:
                if proc is not None:
                    stop_server(proc)
        formal = [r for r in results if r["split"] == "measure"]
        summary = {"case": case, "status": status, "completed": len(results), "formal": len(formal),
                   "length_capped": sum(r["finish_reason"] == "length" for r in formal)}
        draft_n = sum(int(r["timings"].get("draft_n") or 0) for r in formal)
        draft_accepted = sum(int(r["timings"].get("draft_n_accepted") or 0) for r in formal)
        summary["speculative"] = {
            "draft_n": draft_n,
            "draft_accepted": draft_accepted,
            "acceptance_rate": draft_accepted / draft_n if draft_n else None,
        }
        for key in ("ttft_client_ms", "tpot_client_ms", "decode_ms_per_token", "e2e_client_ms"):
            values = [r[key] for r in formal if r[key] is not None]
            summary[key] = {"mean": statistics.mean(values), "median": statistics.median(values)} if values else None
        decode_ms = sum(float(r["timings"]["predicted_ms"]) for r in formal)
        decode_intervals = sum(max(0, int(r["timings"]["predicted_n"]) - 1) for r in formal)
        summary["pooled_tpot_ms"] = decode_ms / decode_intervals if decode_intervals else None
        summary["pooled_decode_tokens_per_second"] = decode_intervals * 1000 / decode_ms if decode_ms else None
        device_samples = [sample for result in formal for sample in result["device_samples"]]
        summary["gpu_peak_used_mib"] = max((sample["used_mib"] for sample in device_samples), default=None)
        utilization = [sample["gpu_util_pct"] for sample in device_samples if sample["gpu_util_pct"] is not None]
        summary["gpu_utilization_mean_pct"] = statistics.mean(utilization) if utilization else None
        (case_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        if case == "baseline" and status == "failed":
            raise RuntimeError("Baseline failed; stopping suite")
    print("Suite finished", flush=True)


if __name__ == "__main__":
    main()
