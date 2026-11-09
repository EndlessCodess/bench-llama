#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


def parse_args():
    parser = argparse.ArgumentParser(description="Run benchmark plans through llama-server")
    parser.add_argument("--llama-server", default=os.environ.get("LLAMA_SERVER"))
    parser.add_argument("--model")
    parser.add_argument("--plan")
    parser.add_argument("--n-ctx", type=int, default=262144)
    parser.add_argument("--n-batch", type=int, default=2048)
    parser.add_argument("--n-ubatch", type=int, default=512)
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument("--cache-type-k", default="f16")
    parser.add_argument("--cache-type-v", default="f16")
    parser.add_argument("--flash-attn", default="auto")
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--poll", type=int, default=50)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--draft-model")
    parser.add_argument("--spec-type", choices=["draft-eagle3", "draft-dflash", "draft-dspark"], default="draft-dflash")
    parser.add_argument("--draft-n-max", type=int, default=15)
    parser.add_argument("--draft-ngl", type=int, default=99)
    parser.add_argument("--ignore-eos", dest="ignore_eos", action="store_true")
    parser.add_argument("--no-ignore-eos", dest="ignore_eos", action="store_false")
    parser.add_argument("--version", action="store_true")
    parser.set_defaults(ignore_eos=False)
    return parser.parse_args()


def emit(event):
    print(json.dumps(event, ensure_ascii=False, separators=(",", ":")), flush=True)


def request_json(url, body=None, timeout=30):
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def request_text(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def speculative_metrics(base_url):
    names = {
        "llamacpp:spec_decode_num_draft_tokens_total": "draft_tokens",
        "llamacpp:spec_decode_num_accepted_tokens_total": "draft_accepted",
        "llamacpp:spec_decode_num_drafts_total": "draft_verification_steps",
    }
    result = {value: 0 for value in names.values()}
    for line in request_text(base_url + "/metrics").splitlines():
        if not line or line.startswith("#"):
            continue
        metric, _, value = line.partition(" ")
        key = names.get(metric)
        if key is not None:
            result[key] = int(float(value))
    return result


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for_server(base_url, proc, timeout=600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("llama-server exited with status {}".format(proc.returncode))
        try:
            request_json(base_url + "/health", timeout=2)
            return
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            time.sleep(1)
    raise RuntimeError("timed out waiting for llama-server")


def stop_server(proc):
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def load_plan(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 8:
                raise ValueError("invalid plan row: expected 8 tab-separated fields")
            rows.append({
                "prompt_file": parts[0],
                "case_id": parts[1],
                "phase": parts[2],
                "phase_run": int(parts[3]),
                "manifest_run": int(parts[4]),
                "source_id": parts[5],
                "n_predict": int(parts[6]),
                "expected_tokens": int(parts[7]),
            })
    if not rows:
        raise ValueError("plan has no rows")
    return rows


def server_command(args, port):
    command = [
        str(Path(args.llama_server).resolve()),
        "--model", str(Path(args.model).resolve()),
        "--gpu-layers", str(args.n_gpu_layers),
        "--ctx-size", str(args.n_ctx),
        "--batch-size", str(args.n_batch),
        "--ubatch-size", str(args.n_ubatch),
        "--parallel", "1",
        "--cache-type-k", args.cache_type_k,
        "--cache-type-v", args.cache_type_v,
        "--flash-attn", args.flash_attn,
        "--poll", str(args.poll),
        "--jinja",
        "--reasoning", "off",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--no-ui",
        "--no-warmup",
        "--metrics",
    ]
    if args.threads is not None:
        command.extend(["--threads", str(args.threads)])
    if args.draft_model:
        command.extend([
            "--spec-draft-model", str(Path(args.draft_model).resolve()),
            "--spec-draft-ngl", str(args.draft_ngl),
            "--spec-type", args.spec_type,
            "--spec-draft-n-max", str(args.draft_n_max),
        ])
    return command


def run_request(base_url, row, args, metrics_before):
    prompt = Path(row["prompt_file"]).read_text(encoding="utf-8")
    body = {
        "model": "benchmark",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": row["n_predict"],
        "temperature": args.temp,
        "seed": args.seed,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": False,
    }
    if args.ignore_eos:
        body["ignore_eos"] = True
    response = request_json(base_url + "/v1/chat/completions", body, timeout=1800)
    timings = response["timings"]
    usage = response["usage"]
    content = response["choices"][0]["message"].get("content", "")
    prompt_tokens = int(usage["prompt_tokens"])
    generated = int(usage["completion_tokens"])
    prompt_ms = float(timings["prompt_ms"])
    generation_ms = float(timings["predicted_ms"])
    draft_tokens = int(timings.get("draft_n", 0))
    draft_accepted = int(timings.get("draft_n_accepted", 0))
    for _ in range(10):
        metrics_after = speculative_metrics(base_url)
        if draft_tokens == 0 or metrics_after["draft_tokens"] - metrics_before["draft_tokens"] >= draft_tokens:
            break
        time.sleep(0.01)
    draft_verification_steps = max(
        0,
        metrics_after["draft_verification_steps"] - metrics_before["draft_verification_steps"],
    )
    return {
        "prompt_tokens": prompt_tokens,
        "tokens_generated": generated,
        "draft_tokens": draft_tokens,
        "draft_accepted": draft_accepted,
        "draft_acceptance_rate": 100.0 * draft_accepted / draft_tokens if draft_tokens else 0.0,
        "draft_verification_steps": draft_verification_steps,
        "draft_mean_accepted_tokens": draft_accepted / draft_verification_steps if draft_verification_steps else 0.0,
        "draft_mean_advance_length": 1.0 + draft_accepted / draft_verification_steps if draft_verification_steps else 0.0,
        "prompt_eval_ms_measured": prompt_ms,
        "ttft_ms_measured": prompt_ms,
        "generation_eval_ms_measured": generation_ms,
        "e2e_ms_measured": prompt_ms + generation_ms,
        "prompt_tps_measured": float(timings["prompt_per_second"]),
        "generation_tps_measured": float(timings["predicted_per_second"]),
        "finish_reason": response["choices"][0].get("finish_reason"),
        "output_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "output_text": content,
    }, metrics_after


def main():
    args = parse_args()
    if args.version:
        if not args.llama_server:
            print("bench_server_run: LLAMA_SERVER is not set")
            return 0
        result = subprocess.run(
            [str(Path(args.llama_server).resolve()), "--version"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        print(result.stdout.strip())
        return result.returncode
    if not args.llama_server or not args.model or not args.plan:
        raise ValueError("--llama-server, --model, and --plan are required")

    rows = load_plan(args.plan)
    port = free_port()
    command = server_command(args, port)
    proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=sys.stderr, stderr=sys.stderr)
    try:
        wait_for_server("http://127.0.0.1:{}".format(port), proc)
        metrics = speculative_metrics("http://127.0.0.1:{}".format(port))
        emit({
            "event": "model_loaded",
            "model": str(Path(args.model).resolve()),
            "server_command": command,
            "speculative": args.spec_type if args.draft_model else "none",
            "ignore_eos": args.ignore_eos,
        })
        for row in rows:
            identity = {
                "case_id": row["case_id"],
                "phase": row["phase"],
                "phase_run": row["phase_run"],
                "manifest_run": row["manifest_run"],
            }
            emit({
                "event": "run_start",
                **identity,
                "source_id": row["source_id"],
                "prompt_file": row["prompt_file"],
                "prompt_tokens": row["expected_tokens"],
                "expected_tokens": row["expected_tokens"],
                "n_predict": row["n_predict"],
            })
            try:
                result, metrics = run_request("http://127.0.0.1:{}".format(port), row, args, metrics)
                emit({
                    "event": "result",
                    **identity,
                    "source_id": row["source_id"],
                    "prompt_file": row["prompt_file"],
                    "expected_tokens": row["expected_tokens"],
                    "n_predict": row["n_predict"],
                    **result,
                })
            except Exception as exc:
                emit({"event": "error", **identity, "fatal": False, "message": str(exc)})
        emit({"event": "all_done", "total_runs": len(rows)})
    finally:
        stop_server(proc)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print("bench_server_run: {}".format(exc), file=sys.stderr)
        sys.exit(2)
