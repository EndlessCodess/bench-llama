#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path
import random

from prepare_dataset import TokenizerServer


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a deterministic MATH-500 DFlash plan")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "math-500" / "test.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "processed" / "qwen3-8b-math500")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--llama-server", type=Path, required=True)
    parser.add_argument("--pool-size", type=int, default=50)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--formal-runs", type=int, default=100)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260826)
    return parser.parse_args()


def make_prompt(problem):
    return (
        "Solve the following mathematics problem. Show your reasoning step by step, "
        "and put the final answer inside \\boxed{}.\n\nProblem:\n" + problem
    )


def main():
    args = parse_args()
    rows = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line]
    selected = random.Random(args.seed).sample(rows, args.pool_size)
    if args.formal_runs % args.pool_size != 0:
        raise ValueError("formal runs must be a multiple of pool size")
    if args.output.exists():
        raise ValueError("output already exists: {}".format(args.output))

    prompt_dir = args.output / "prompts"
    prompt_dir.mkdir(parents=True)
    records = []
    with TokenizerServer(args.llama_server.resolve(), args.model.resolve(), args.output / "tokenizer-server.log", 120.0) as tokenizer:
        for index, row in enumerate(selected):
            prompt = make_prompt(row["problem"])
            actual_tokens = tokenizer.count_prompt_tokens(prompt)
            prompt_path = prompt_dir / "{:03d}.txt".format(index)
            prompt_path.write_text(prompt, encoding="utf-8")
            records.append({
                "pool_index": index,
                "source_id": row["unique_id"],
                "subject": row["subject"],
                "level": row["level"],
                "actual_tokens": actual_tokens,
                "prompt_file": str(prompt_path.relative_to(args.output)),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            })
            print("[{}/{}] {} tokens={}".format(index + 1, args.pool_size, row["unique_id"], actual_tokens), flush=True)

    selections = list(range(args.warmup_runs))
    selections += [index for _ in range(args.formal_runs // args.pool_size) for index in range(args.pool_size)]
    lines = []
    for run_index, pool_index in enumerate(selections):
        record = records[pool_index]
        phase = "warmup" if run_index < args.warmup_runs else "formal"
        phase_run = run_index + 1 if phase == "warmup" else run_index - args.warmup_runs + 1
        lines.append("\t".join((
            str((args.output / record["prompt_file"]).resolve()),
            "MATH-500",
            phase,
            str(phase_run),
            str(run_index + 1),
            record["source_id"],
            str(args.output_tokens),
            str(record["actual_tokens"]),
        )))

    (args.output / "plan.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output / "metadata.json").write_text(json.dumps({
        "dataset": "HuggingFaceH4/MATH-500",
        "dataset_file": str(args.dataset.resolve()),
        "seed": args.seed,
        "pool_size": args.pool_size,
        "warmup_runs": args.warmup_runs,
        "formal_runs": args.formal_runs,
        "output_tokens": args.output_tokens,
        "records": records,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("plan: {}".format(args.output / "plan.tsv"))


if __name__ == "__main__":
    main()
