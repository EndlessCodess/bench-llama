import argparse
import collections
import hashlib
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Extract fixed system/user trace samples.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Output already exists; choose a new directory.")
    pools = collections.defaultdict(list)
    seen = set()
    for path in sorted(args.source.glob("*.json")):
        trace = json.loads(path.read_text())
        for obs in trace.get("observations", []):
            messages = obs.get("input")
            if obs.get("type") != "GENERATION" or not isinstance(messages, list):
                continue
            if not all(isinstance(m, dict) for m in messages):
                continue
            if [m.get("role") for m in messages] != ["system", "user"]:
                continue
            if not all(isinstance(m.get("content"), str) for m in messages):
                continue
            digest = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            tokens = obs.get("promptTokens", (obs.get("usage") or {}).get("input"))
            if not isinstance(tokens, int) or tokens <= 0:
                raise ValueError("Missing recorded input token count")
            bucket = next((str(n) for n in (2048, 8192, 16384, 32768) if tokens <= n), "over32768")
            stage = (obs.get("metadata") or {}).get("langgraph_node", "unknown")
            pools[(stage, bucket)].append({
                "id": obs["id"], "trace_id": trace.get("trace_id"),
                "source_file": path.name, "messages": messages,
                "messages_sha256": digest, "stage": stage, "length_bucket": bucket,
                "recorded_input_tokens": tokens,
                "recorded_output_tokens": obs.get("completionTokens"),
                "original_model_parameters": obs.get("modelParameters"),
            })
    total = sum(map(len, pools.values()))
    if total < 205:
        raise ValueError("Need at least 205 unique requests")
    rng = random.Random(args.seed)
    quotas = {k: 205 * len(v) // total for k, v in pools.items()}
    order = sorted(pools, key=lambda k: (-(205 * len(pools[k]) % total), k))
    for k in order[:205 - sum(quotas.values())]:
        quotas[k] += 1
    selected = []
    for k in sorted(pools):
        rng.shuffle(pools[k])
        selected.extend(pools[k][:quotas[k]])
    rng.shuffle(selected)
    for i, row in enumerate(selected):
        row["split"] = "warmup" if i < 5 else "measure"
        row["order"] = i
    args.output.mkdir(parents=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected)
    (args.output / "samples.jsonl").write_text(payload, encoding="utf-8")
    manifest = {
        "source": str(args.source.resolve()), "seed": args.seed,
        "selection": "Unique system/user inputs; proportional stage/recorded-length strata; fixed shuffle",
        "eligible_unique_requests": total, "warmup": 5, "measure": 200, "rounds": 1,
        "token_counts": "Historical trace counts, not measured with the current GGUF tokenizer",
        "samples_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "source_files_used": len({r["source_file"] for r in selected}),
        "strata": [{"stage": k[0], "length_bucket": k[1], "available": len(pools[k]), "selected": quotas[k]} for k in sorted(pools)],
        "samples": [{k: v for k, v in row.items() if k not in ("messages", "original_model_parameters")} for row in selected],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("eligible_unique_requests", "warmup", "measure", "source_files_used", "samples_sha256")}))


if __name__ == "__main__":
    main()
