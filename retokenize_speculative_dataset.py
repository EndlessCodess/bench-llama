#!/usr/bin/env python3
"""Reuse the exact historical prompt pool and request order for a new tokenizer."""

import argparse
import json
from pathlib import Path
import shutil

from prepare_dataset import TokenizerServer, sha256_file


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
            tokens = tokenizer.count_prompt_tokens(source.read_text())
            row["target_tokens"] = tokens
            row["actual_tokens"] = tokens
    (args.output / "metadata.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in metadata))
    shutil.copyfile(args.source / "run-manifest.json", args.output / "run-manifest.json")
    (args.output / "preparation.json").write_text(json.dumps({
        "model": str(args.model.resolve()), "llama_server": str(args.llama_server.resolve()),
        "source_processed_dir": str(args.source.resolve()), "prompt_pool_unchanged": True,
        "source_manifest_sha256": sha256_file(args.source / "run-manifest.json"),
        "enable_thinking": False, "prompts": len(metadata),
    }, indent=2) + "\n")
    print("Retokenized {} unchanged prompts: {}".format(len(metadata), args.output), flush=True)


if __name__ == "__main__":
    main()
