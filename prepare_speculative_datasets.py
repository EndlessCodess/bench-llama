#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path
import random

from prepare_dataset import TokenizerServer, sha256_file


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Alpaca, MATH-500, and HumanEval for LC-033 through LC-038")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--llama-server", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "processed" / "qwen3-8b-speculative")
    parser.add_argument("--alpaca", type=Path, default=ROOT / "data" / "alpaca" / "data" / "train-00000-of-00001-a09b74b3ef9c3b56.parquet")
    parser.add_argument("--math500", type=Path, default=ROOT / "data" / "math-500" / "test.jsonl")
    parser.add_argument("--humaneval", type=Path, default=ROOT / "data" / "humaneval" / "openai_humaneval" / "test-00000-of-00001.parquet")
    parser.add_argument("--pool-size", type=int, default=50)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--formal-runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


def read_parquet(path):
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read the Alpaca and HumanEval parquet files") from exc
    return parquet.read_table(path).to_pylist()


def load_datasets(args):
    alpaca = []
    for index, row in enumerate(read_parquet(args.alpaca)):
        prompt = row["instruction"].strip()
        if row.get("input", "").strip():
            prompt += "\n\nInput:\n" + row["input"].strip()
        alpaca.append({"source_id": "alpaca-{}".format(index), "prompt": prompt})

    math500 = []
    for line in args.math500.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prompt = "Solve the following mathematics problem. Show your reasoning step by step, and put the final answer inside \\boxed{}.\n\nProblem:\n" + row["problem"]
        math500.append({"source_id": row["unique_id"], "prompt": prompt, "difficulty": row.get("level", "")})

    humaneval = []
    for row in read_parquet(args.humaneval):
        humaneval.append({"source_id": row["task_id"], "prompt": row["prompt"]})

    return {"alpaca": alpaca, "math-500": math500, "humaneval": humaneval}


def main():
    args = parse_args()
    if args.pool_size <= 0 or args.warmup_runs < 0 or args.formal_runs <= 0:
        raise ValueError("run counts and pool size must be positive")
    if args.formal_runs % args.pool_size != 0:
        raise ValueError("formal runs must be a multiple of pool size")
    for path in (args.model, args.llama_server, args.alpaca, args.math500, args.humaneval):
        if not path.is_file():
            raise ValueError("required input does not exist: {}".format(path))
    if args.output.exists():
        raise ValueError("output already exists: {}".format(args.output))

    datasets = load_datasets(args)
    selected = {}
    for offset, (name, rows) in enumerate(datasets.items()):
        if len(rows) < args.pool_size:
            raise ValueError("{} has only {} rows".format(name, len(rows)))
        selected[name] = random.Random(args.seed + offset).sample(rows, args.pool_size)

    args.output.mkdir(parents=True)
    metadata = []
    selections = []
    with TokenizerServer(args.llama_server.resolve(), args.model.resolve(), args.output / "tokenizer-server.log", 300.0, enable_thinking=False) as tokenizer:
        for dataset, rows in selected.items():
            for pool_index, row in enumerate(rows):
                relative = Path("prompts") / dataset / "{:03d}.txt".format(pool_index)
                path = args.output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(row["prompt"], encoding="utf-8")
                mode = "completion" if dataset == "humaneval" else "chat"
                actual_tokens = tokenizer.count_prompt_tokens(row["prompt"], chat=mode == "chat")
                metadata.append({
                    "dataset": dataset,
                    "pool_index": pool_index,
                    "source_id": row["source_id"],
                    "domain": dataset,
                    "difficulty": row.get("difficulty", ""),
                    "prompt_file": str(relative),
                    "prompt_sha256": hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest(),
                    "target_tokens": actual_tokens,
                    "actual_tokens": actual_tokens,
                    "request_mode": mode,
                })
                print("{} [{}/{}] tokens={}".format(dataset, pool_index + 1, args.pool_size, actual_tokens), flush=True)

            order = list(range(args.warmup_runs))
            order += [index for _ in range(args.formal_runs // args.pool_size) for index in range(args.pool_size)]
            for run_index, pool_index in enumerate(order):
                phase = "warmup" if run_index < args.warmup_runs else "formal"
                selections.append({
                    "dataset": dataset,
                    "run": run_index + 1,
                    "phase": phase,
                    "pass": None if phase == "warmup" else 1 + (run_index - args.warmup_runs) // args.pool_size,
                    "pool_index": pool_index,
                    "source_id": rows[pool_index]["source_id"],
                    "prompt_file": str(Path("prompts") / dataset / "{:03d}.txt".format(pool_index)),
                })

    (args.output / "metadata.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in metadata), encoding="utf-8"
    )
    (args.output / "run-manifest.json").write_text(json.dumps({
        "seed": args.seed,
        "sampling": {"warmup": "fixed_sequential", "formal": "cycle_sequential"},
        "warmup_runs": args.warmup_runs,
        "formal_runs": args.formal_runs,
        "pool_size": args.pool_size,
        "datasets": list(datasets),
        "request_modes": {"alpaca": "chat", "math-500": "chat", "humaneval": "completion"},
        "prompt_protocol_version": 2,
        "selections": selections,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "preparation.json").write_text(json.dumps({
        "model": str(args.model.resolve()),
        "model_size": args.model.stat().st_size,
        "llama_server": str(args.llama_server.resolve()),
        "pool_size_per_dataset": args.pool_size,
        "warmup_runs_per_round": args.warmup_runs,
        "formal_runs_per_round": args.formal_runs,
        "recommended_rounds": 10,
        "formal_runs_per_case_total": args.formal_runs * 10,
        "seed": args.seed,
        "datasets": {
            "alpaca": {"path": str(args.alpaca.resolve()), "sha256": sha256_file(args.alpaca)},
            "math-500": {"path": str(args.math500.resolve()), "sha256": sha256_file(args.math500)},
            "humaneval": {"path": str(args.humaneval.resolve()), "sha256": sha256_file(args.humaneval)},
        },
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("prepared: {}".format(args.output))


if __name__ == "__main__":
    main()
