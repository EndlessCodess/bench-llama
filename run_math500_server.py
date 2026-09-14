#!/usr/bin/env python3

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark llama-server on MATH-500")
    parser.add_argument("--llama-server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--draft-model", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cuda-visible-devices", default="6")
    parser.add_argument("--port", type=int, default=18086)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--draft-n-max", type=int, default=15)
    return parser.parse_args()


def request_json(url, body=None, timeout=30):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def wait_for_server(base_url, proc, timeout=300):
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


def make_prompt(problem):
    return (
        "Solve the following mathematics problem. Show your reasoning step by step, "
        "and put the final answer inside \\boxed{}.\n\nProblem:\n" + problem
    )


def load_rows(path, limit):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows[:limit] if limit > 0 else rows


def load_completed(path):
    completed = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                completed.add(json.loads(line)["index"])
    return completed


def write_status(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def summarize(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    predicted_n = sum(row["timings"]["predicted_n"] for row in rows)
    predicted_ms = sum(row["timings"]["predicted_ms"] for row in rows)
    wall_seconds = sum(row["wall_seconds"] for row in rows)
    draft_n = sum(row["timings"].get("draft_n", 0) for row in rows)
    accepted_n = sum(row["timings"].get("draft_n_accepted", 0) for row in rows)
    acceptance = accepted_n / draft_n if draft_n else None
    return {
        "requests": len(rows),
        "prompt_tokens": sum(row["usage"]["prompt_tokens"] for row in rows),
        "completion_tokens": predicted_n,
        "completion_seconds": predicted_ms / 1000.0,
        "wall_seconds": wall_seconds,
        "pooled_generation_tokens_per_second": 1000.0 * predicted_n / predicted_ms if predicted_ms else None,
        "pooled_end_to_end_tokens_per_second": predicted_n / wall_seconds if wall_seconds else None,
        "mean_generation_tokens_per_second": sum(row["timings"]["predicted_per_second"] for row in rows) / len(rows),
        "draft_tokens": draft_n,
        "accepted_draft_tokens": accepted_n,
        "draft_acceptance": acceptance,
        "draft_acceptance_percent": acceptance * 100.0 if acceptance is not None else None,
    }


def stop_server(proc):
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def run_phase(args, name, rows, draft_model):
    phase_dir = args.output_dir / name
    phase_dir.mkdir(parents=True, exist_ok=True)
    results_path = phase_dir / "results.jsonl"
    completed = load_completed(results_path)
    log_handle = (phase_dir / "server.log").open("a", encoding="utf-8")

    command = [
        str(args.llama_server),
        "-m", str(args.model),
        "-ngl", "99",
        "-fa", "auto",
        "-c", "40960",
        "-b", "2048",
        "-ub", "512",
        "-np", "1",
        "--jinja",
        "--reasoning", "off",
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--no-ui",
        "--no-warmup",
    ]
    if draft_model is not None:
        command += [
            "-md", str(draft_model),
            "-ngld", "99",
            "--spec-type", "draft-dflash",
            "--spec-draft-n-max", str(args.draft_n_max),
        ]

    (phase_dir / "command.json").write_text(
        json.dumps({"command": command, "cuda_visible_devices": args.cuda_visible_devices}, indent=2) + "\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    proc = subprocess.Popen(command, env=env, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
    base_url = "http://127.0.0.1:{}".format(args.port)

    try:
        wait_for_server(base_url, proc)
        for warmup_index, row in enumerate(rows[-args.warmups:]):
            body = {
                "model": "qwen3-8b",
                "messages": [{"role": "user", "content": make_prompt(row["problem"])}],
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "seed": 42,
                "chat_template_kwargs": {"enable_thinking": False},
                "stream": False,
            }
            request_json(base_url + "/v1/chat/completions", body, timeout=1800)

        with results_path.open("a", encoding="utf-8", buffering=1) as output:
            for index, row in enumerate(rows):
                if index in completed:
                    continue
                body = {
                    "model": "qwen3-8b",
                    "messages": [{"role": "user", "content": make_prompt(row["problem"])}],
                    "max_tokens": args.max_tokens,
                    "temperature": 0,
                    "seed": 42,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "stream": False,
                }
                started = time.time()
                response = request_json(base_url + "/v1/chat/completions", body, timeout=1800)
                result = {
                    "index": index,
                    "unique_id": row["unique_id"],
                    "subject": row["subject"],
                    "level": row["level"],
                    "expected_answer": row["answer"],
                    "content": response["choices"][0]["message"].get("content", ""),
                    "finish_reason": response["choices"][0]["finish_reason"],
                    "usage": response["usage"],
                    "timings": response["timings"],
                    "wall_seconds": time.time() - started,
                }
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                write_status(args.output_dir / "status.json", {
                    "phase": name,
                    "completed": index + 1,
                    "total": len(rows),
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                })
    finally:
        stop_server(proc)
        log_handle.close()
        time.sleep(5)

    summary = summarize(results_path)
    (phase_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.dataset, args.limit)
    manifest = {
        "dataset": str(args.dataset.resolve()),
        "requests": len(rows),
        "warmups_per_phase": args.warmups,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "thinking": False,
        "target_model": str(args.model.resolve()),
        "draft_model": str(args.draft_model.resolve()) if args.draft_model else None,
        "draft_n_max": args.draft_n_max if args.draft_model else None,
        "cuda_visible_devices": args.cuda_visible_devices,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    summaries = {"baseline": run_phase(args, "baseline", rows, None)}
    if args.draft_model:
        summaries["dflash"] = run_phase(args, "dflash", rows, args.draft_model)
        base_tps = summaries["baseline"]["pooled_generation_tokens_per_second"]
        draft_tps = summaries["dflash"]["pooled_generation_tokens_per_second"]
        base_e2e_tps = summaries["baseline"]["pooled_end_to_end_tokens_per_second"]
        draft_e2e_tps = summaries["dflash"]["pooled_end_to_end_tokens_per_second"]
        summaries["comparison"] = {
            "draft_acceptance_percent": summaries["dflash"]["draft_acceptance_percent"],
            "generation_speedup_x": draft_tps / base_tps,
            "end_to_end_speedup_x": draft_e2e_tps / base_e2e_tps,
        }
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    write_status(args.output_dir / "status.json", {
        "phase": "complete",
        "completed": len(rows),
        "total": len(rows),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })


if __name__ == "__main__":
    main()
