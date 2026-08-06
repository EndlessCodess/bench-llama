#!/usr/bin/env python3

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parent
DEFAULT_TARGETS = (2048, 8192, 16384, 28672)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare fixed-length LongBench-v2 prompt pools for llama.cpp benchmarks"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data" / "longbench-v2" / "data.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "qwen3.5-9b",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=os.environ.get("LLAMA_MODEL"),
        help="GGUF model used for its tokenizer (or LLAMA_MODEL)",
    )
    parser.add_argument(
        "--llama-server",
        type=Path,
        default=os.environ.get("LLAMA_SERVER"),
        help="llama-server executable (or LLAMA_SERVER)",
    )
    parser.add_argument(
        "--targets",
        default=",".join(str(value) for value in DEFAULT_TARGETS),
        help="comma-separated final prompt token counts",
    )
    parser.add_argument("--pool-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--formal-runs", type=int, default=10)
    parser.add_argument(
        "--max-token-gap",
        type=int,
        default=8,
        help="maximum allowed target minus actual token count",
    )
    parser.add_argument("--server-timeout", type=float, default=120.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if args.model is None:
        raise ValueError("--model or LLAMA_MODEL is required")
    if args.llama_server is None:
        raise ValueError("--llama-server or LLAMA_SERVER is required")

    args.dataset = args.dataset.resolve()
    args.output = args.output.resolve()
    args.model = args.model.resolve()
    args.llama_server = args.llama_server.resolve()

    for label, path in (
        ("dataset", args.dataset),
        ("model", args.model),
        ("llama-server", args.llama_server),
    ):
        if not path.is_file():
            raise ValueError("{} does not exist: {}".format(label, path))
    if not os.access(args.llama_server, os.X_OK):
        raise ValueError("llama-server is not executable: {}".format(args.llama_server))

    try:
        args.targets = tuple(sorted({int(value) for value in args.targets.split(",")}))
    except ValueError as exc:
        raise ValueError("--targets must be comma-separated integers") from exc
    if not args.targets or any(value <= 0 for value in args.targets):
        raise ValueError("all targets must be positive")
    if args.pool_size <= 0:
        raise ValueError("--pool-size must be positive")
    if args.warmup_runs < 0 or args.formal_runs <= 0:
        raise ValueError("run counts must be non-negative, with at least one formal run")
    if args.max_token_gap < 0:
        raise ValueError("--max-token-gap must be non-negative")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def request_json(base_url, method, path, payload=None, timeout=120.0):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        base_url + path,
        data=data,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


class TokenizerServer:
    def __init__(self, executable, model, log_path, timeout):
        self.executable = executable
        self.model = model
        self.log_path = log_path
        self.timeout = timeout
        self.port = free_port()
        self.base_url = "http://127.0.0.1:{}".format(self.port)
        self.process = None
        self.log_handle = None

    def __enter__(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = self.log_path.open("w", encoding="utf-8")
        command = [
            str(self.executable),
            "-m",
            str(self.model),
            "-c",
            "512",
            "-ngl",
            "0",
            "--no-warmup",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--no-ui",
            "-np",
            "1",
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["ASCEND_RT_VISIBLE_DEVICES"] = ""
        self.process = subprocess.Popen(
            command,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            env=environment,
            text=True,
        )

        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    "llama-server exited with status {}; see {}".format(
                        self.process.returncode, self.log_path
                    )
                )
            try:
                response = request_json(self.base_url, "GET", "/health", timeout=2.0)
                if response.get("status") == "ok":
                    return self
            except (OSError, urllib.error.URLError, json.JSONDecodeError):
                time.sleep(0.2)
        raise RuntimeError("timed out waiting for llama-server; see {}".format(self.log_path))

    def __exit__(self, exc_type, exc_value, traceback):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.log_handle is not None:
            self.log_handle.close()

    def apply_template(self, content):
        response = request_json(
            self.base_url,
            "POST",
            "/apply-template",
            {
                "messages": [{"role": "user", "content": content}],
                "add_generation_prompt": True,
            },
            timeout=self.timeout,
        )
        return response["prompt"]

    def count_prompt_tokens(self, content):
        formatted = self.apply_template(content)
        response = request_json(
            self.base_url,
            "POST",
            "/tokenize",
            {
                "content": formatted,
                "add_special": False,
                "parse_special": True,
            },
            timeout=self.timeout,
        )
        return len(response["tokens"])


def build_user_prompt(item, context):
    return (
        "Please read the following context and answer the multiple-choice question.\n\n"
        "Context:\n"
        + context
        + "\n\nQuestion:\n"
        + item["question"]
        + "\n\nA. "
        + item["choice_A"]
        + "\nB. "
        + item["choice_B"]
        + "\nC. "
        + item["choice_C"]
        + "\nD. "
        + item["choice_D"]
        + "\n\nSelect the correct answer."
    )


def fit_prompt(item, target, tokenizer):
    context = item["context"]
    empty_prompt = build_user_prompt(item, "")
    empty_count = tokenizer.count_prompt_tokens(empty_prompt)
    if empty_count >= target:
        raise ValueError(
            "sample {} needs {} tokens without context, target is {}".format(
                item["_id"], empty_count, target
            )
        )

    high = min(len(context), max(1024, target * 8))
    high_count = tokenizer.count_prompt_tokens(build_user_prompt(item, context[:high]))
    while high_count <= target and high < len(context):
        high = min(len(context), high * 2)
        high_count = tokenizer.count_prompt_tokens(build_user_prompt(item, context[:high]))
    if high_count < target and high == len(context):
        raise ValueError(
            "sample {} has only {} prompt tokens, target is {}".format(
                item["_id"], high_count, target
            )
        )

    low = 0
    best_chars = 0
    best_count = empty_count
    while low <= high:
        middle = (low + high) // 2
        prompt = build_user_prompt(item, context[:middle])
        count = tokenizer.count_prompt_tokens(prompt)
        if count <= target:
            if count > best_count or (count == best_count and middle > best_chars):
                best_chars = middle
                best_count = count
            low = middle + 1
        else:
            high = middle - 1

    for cut in range(best_chars + 1, min(len(context), best_chars + 16) + 1):
        prompt = build_user_prompt(item, context[:cut])
        count = tokenizer.count_prompt_tokens(prompt)
        if count <= target and count >= best_count:
            best_chars = cut
            best_count = count
        if count > target and cut > best_chars + 4:
            break

    return build_user_prompt(item, context[:best_chars]), best_chars, best_count


def choose_source_pool(dataset, pool_size, seed, minimum_words):
    eligible = [item for item in dataset if len(item["context"].split()) >= minimum_words]
    by_domain = {}
    for item in eligible:
        by_domain.setdefault(item["domain"], []).append(item)

    rng = random.Random(seed)
    for items in by_domain.values():
        rng.shuffle(items)

    domains = sorted(by_domain)
    selected = []
    cursor = {domain: 0 for domain in domains}
    while len(selected) < pool_size:
        made_progress = False
        for domain in domains:
            index = cursor[domain]
            if index < len(by_domain[domain]):
                selected.append(by_domain[domain][index])
                cursor[domain] += 1
                made_progress = True
                if len(selected) == pool_size:
                    break
        if not made_progress:
            break

    if len(selected) < pool_size:
        raise ValueError(
            "only {} eligible samples for requested pool size {}".format(
                len(selected), pool_size
            )
        )
    return selected


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare(args):
    if args.output.exists():
        if not args.force:
            raise ValueError("output already exists (use --force): {}".format(args.output))
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    print("loading dataset: {}".format(args.dataset), flush=True)
    with args.dataset.open(encoding="utf-8") as handle:
        dataset = json.load(handle)

    minimum_words = max(args.targets)
    source_pool = choose_source_pool(dataset, args.pool_size, args.seed, minimum_words)
    log_path = args.output / "tokenizer-server.log"
    metadata = []

    print("starting tokenizer server", flush=True)
    with TokenizerServer(args.llama_server, args.model, log_path, args.server_timeout) as tokenizer:
        for pool_index, item in enumerate(source_pool):
            for target in args.targets:
                print(
                    "[{}/{}] source={} target={}".format(
                        pool_index + 1, len(source_pool), item["_id"], target
                    ),
                    flush=True,
                )
                prompt, context_chars, actual_tokens = fit_prompt(item, target, tokenizer)
                gap = target - actual_tokens
                if gap > args.max_token_gap:
                    raise RuntimeError(
                        "sample {} target {} missed by {} tokens".format(
                            item["_id"], target, gap
                        )
                    )

                prompt_dir = args.output / "prompts" / str(target)
                prompt_dir.mkdir(parents=True, exist_ok=True)
                filename = "{:03d}-{}.txt".format(pool_index, item["_id"])
                prompt_path = prompt_dir / filename
                prompt_path.write_text(prompt, encoding="utf-8")

                metadata.append(
                    {
                        "pool_index": pool_index,
                        "source_id": item["_id"],
                        "domain": item["domain"],
                        "sub_domain": item["sub_domain"],
                        "difficulty": item["difficulty"],
                        "official_length": item["length"],
                        "target_tokens": target,
                        "actual_tokens": actual_tokens,
                        "token_gap": gap,
                        "context_chars": context_chars,
                        "context_words": len(item["context"][:context_chars].split()),
                        "prompt_file": str(prompt_path.relative_to(args.output)),
                        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                        "answer": item["answer"],
                    }
                )

    metadata_path = args.output / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for record in metadata:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    source_records = []
    for pool_index, item in enumerate(source_pool):
        source_records.append(
            {
                "pool_index": pool_index,
                "source_id": item["_id"],
                "domain": item["domain"],
                "sub_domain": item["sub_domain"],
                "difficulty": item["difficulty"],
                "official_length": item["length"],
                "answer": item["answer"],
            }
        )
    write_json(args.output / "source-pool.json", source_records)

    run_rng = random.Random(args.seed)
    run_count = args.warmup_runs + args.formal_runs
    selections = []
    for run_index in range(run_count):
        pool_index = run_rng.randrange(len(source_pool))
        item = source_pool[pool_index]
        selections.append(
            {
                "run": run_index + 1,
                "phase": "warmup" if run_index < args.warmup_runs else "formal",
                "pool_index": pool_index,
                "source_id": item["_id"],
                "prompt_files": {
                    str(target): "prompts/{}/{:03d}-{}.txt".format(
                        target, pool_index, item["_id"]
                    )
                    for target in args.targets
                },
            }
        )
    write_json(
        args.output / "run-manifest.json",
        {
            "seed": args.seed,
            "sampling": "with_replacement",
            "warmup_runs": args.warmup_runs,
            "formal_runs": args.formal_runs,
            "targets": list(args.targets),
            "selections": selections,
        },
    )

    write_json(
        args.output / "preparation.json",
        {
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "dataset": str(args.dataset),
            "dataset_sha256": sha256_file(args.dataset),
            "dataset_samples": len(dataset),
            "model": str(args.model),
            "model_size": args.model.stat().st_size,
            "llama_server": str(args.llama_server),
            "targets": list(args.targets),
            "pool_size": args.pool_size,
            "seed": args.seed,
            "max_token_gap": args.max_token_gap,
            "token_count_scope": "Qwen chat template including assistant generation prefix",
        },
    )

    print("prepared dataset: {}".format(args.output), flush=True)


def main():
    args = parse_args()
    validate_args(args)
    prepare(args)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, urllib.error.URLError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        sys.exit(1)
