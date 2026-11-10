#!/usr/bin/env python3
"""Reuse the exact historical prompt pool and request order for a new tokenizer."""

import argparse
import json
from pathlib import Path
import shutil

from prepare_dataset import TokenizerServer, sha256_file
from bench_server_run import validate_humaneval_prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--llama-server", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = [json.loads(line) for line in (args.source / "metadata.jsonl").read_text().splitlines() if line.strip()]
    with TokenizerServer(args.llama_server.resolve(), args.model.resolve(),
                         args.output / "tokenizer-server.log", 600.0,
                         enable_thinking=False) as tokenizer:
        for row in metadata:
            relative = Path(row["prompt_file"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Expected a relative prompt path")
            source = args.source / relative
            if sha256_file(source) != row["prompt_sha256"]:
                raise ValueError("Prompt hash mismatch: " + str(source))
            destination = args.output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            prompt = source.read_text(encoding="utf-8")
            mode = "completion" if row.get("dataset") == "humaneval" else "chat"
            if mode == "completion":
                validate_humaneval_prompt(prompt)
            tokens = tokenizer.count_prompt_tokens(prompt, chat=mode == "chat")
            row["target_tokens"] = tokens
            row["actual_tokens"] = tokens
            row["request_mode"] = mode
    (args.output / "metadata.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in metadata))
    manifest = json.loads((args.source / "run-manifest.json").read_text(encoding="utf-8"))
    manifest.update(prompt_protocol_version=2, request_modes={"alpaca": "chat", "math-500": "chat", "humaneval": "completion"})
    (args.output / "run-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "preparation.json").write_text(json.dumps({
        "model": str(args.model.resolve()), "llama_server": str(args.llama_server.resolve()),
        "source_processed_dir": str(args.source.resolve()), "prompt_pool_unchanged": True,
        "source_manifest_sha256": sha256_file(args.source / "run-manifest.json"),
        "enable_thinking": False, "prompts": len(metadata),
        "prompt_protocol_version": 2,
        "request_modes": {"alpaca": "chat", "math-500": "chat", "humaneval": "completion"},
    }, indent=2) + "\n")
    print("Retokenized {} unchanged prompts: {}".format(len(metadata), args.output), flush=True)


if __name__ == "__main__":
    main()
