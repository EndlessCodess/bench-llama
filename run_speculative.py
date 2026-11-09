#!/usr/bin/env python3
"""Run paired LC33-LC38 server benchmarks, completing each round in order."""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys

import bench_test
import run_suite


ROOT = Path(__file__).resolve().parent
PAIRS = {"alpaca": (33, 34), "math-500": (35, 36), "humaneval": (37, 38)}
CANN_FLAGS = (
    "GGML_CANN_310P_WQMM", "GGML_CANN_GDN310P_DECODE", "GGML_CANN_GDN310P_PREFILL",
    "GGML_CANN_GDN310P_SNAPSHOTS", "GGML_CANN_SSM_CONV_WEIGHT_PREPACK",
    "GGML_CANN_310P_NATIVE_MROPE", "GGML_CANN_310P_PARTIAL_MROPE",
)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def now():
    return dt.datetime.now().astimezone().isoformat()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="target GGUF, F16 or Q8_0")
    parser.add_argument("--draft-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed/qwen3.5-9b-speculative-nothink")
    parser.add_argument("--platform", default="cann_310p", choices=("cann_310p", "cann_910b", "cuda", "metal", "cpu"))
    parser.add_argument("--device-index", type=int, required=True, help="physical device index")
    parser.add_argument("--datasets", nargs="+", choices=tuple(PAIRS), default=list(PAIRS))
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--formal-runs", type=int, default=50)
    parser.add_argument("--allow-repeated-samples", action="store_true", help="allow repeated source IDs within a formal round")
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--n-batch", type=int, default=512)
    parser.add_argument("--n-ubatch", type=int, default=512)
    parser.add_argument("--draft-n-max", type=int, default=15)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="validate prompts and commands without loading a model")
    args = parser.parse_args()
    for name in ("rounds", "formal_runs", "n_ctx", "n_batch", "n_ubatch", "draft_n_max", "threads"):
        if getattr(args, name) <= 0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    if args.warmup_runs < 0 or args.device_index < 0 or args.n_ubatch > args.n_batch:
        parser.error("invalid warmup count, device index, or microbatch size")
    if args.n_ubatch < args.draft_n_max + 1:
        parser.error("--n-ubatch must fit the anchor and draft tokens")
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("--datasets must not repeat a dataset")
    for name in ("llama_server", "model", "draft_model", "processed_dir", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    for path in (args.llama_server, args.model, args.draft_model):
        if not path.is_file():
            parser.error("input file does not exist: {}".format(path))
    if not os.access(args.llama_server, os.X_OK):
        parser.error("llama-server is not executable")
    if args.output_dir.exists():
        parser.error("output directory already exists; use a new directory")
    return args


def sample_plan(args):
    manifest, _, _ = bench_test.load_inputs(args.processed_dir)
    result = {}
    for dataset in args.datasets:
        selections = bench_test.choose_selections(manifest, args.warmup_runs, args.formal_runs, dataset)
        formal = [item[2]["source_id"] for item in selections if item[0] == "formal"]
        if len(set(formal)) < len(formal) and not args.allow_repeated_samples:
            raise ValueError("{}: {} formal requests use {} unique samples; reduce --formal-runs or explicitly pass --allow-repeated-samples".format(dataset, len(formal), len(set(formal))))
        result[dataset] = {
            "unique_formal_samples": len(set(formal)),
            "formal_requests_per_round": len(formal),
            "selections": [{"phase": phase, "phase_run": index, **selection} for phase, index, selection in selections],
        }
    return result


def prepare_suite(args, folder, smoke=False):
    folder.mkdir()
    jobs = []
    runtime = {
        "n_ctx": args.n_ctx, "n_batch": args.n_batch, "n_ubatch": args.n_ubatch,
        "threads": args.threads, "draft_n_max": args.draft_n_max, "spec_type": "draft-dflash",
    }
    for dataset in args.datasets:
        for offset, template in enumerate(("lc-033-037-dflash-off.json", "lc-034-038-dflash-on.json")):
            config = json.loads((ROOT / "configs" / template).read_text(encoding="utf-8"))
            case_id = "LC-{:03d}".format(PAIRS[dataset][offset])
            config["cases"] = [case for case in config["cases"] if case["id"] == case_id]
            if smoke:
                config["cases"][0]["output_tokens"] = 64
            config["runtime"].update(runtime)
            config["experiment"]["model_key"] = "target"
            job = {"id": case_id.lower(), "group": 6, "config": case_id + ".json", "runner": "server", "model_key": "target"}
            if offset:
                config["experiment"]["draft_model_key"] = "draft"
                job["draft_model_key"] = "draft"
            save(folder / (case_id + ".json"), config)
            jobs.append(job)
    suite = folder / "suite.json"
    save(suite, {"schema_version": 1, "name": "paired-speculative-decoding", "jobs": jobs})
    return suite


def command(args, suite, output):
    return [sys.executable, "-u", str(ROOT / "run_suite.py"), "--suite", str(suite),
            "--llama-server", str(args.llama_server), "--model", "target=" + str(args.model),
            "--model", "draft=" + str(args.draft_model), "--processed-dir", str(args.processed_dir),
            "--platform", args.platform, "--device-index", str(args.device_index),
            "--rounds", "1", "--warmup-runs", str(args.warmup_runs), "--formal-runs", str(args.formal_runs),
            "--output-dir", str(output)]


def check_smoke(args):
    checks = []
    for dataset in args.datasets:
        outputs = []
        for number in PAIRS[dataset]:
            case_id = "LC-{:03d}".format(number)
            folder = args.output_dir / "smoke" / case_id.lower() / case_id
            events = [json.loads(line) for line in (folder / "session.stdout.log").read_text(encoding="utf-8").splitlines()]
            results = [event for event in events if event.get("event") == "result"]
            if len(results) != 1 or results[0]["tokens_generated"] <= 0 or not results[0]["output_text"].strip():
                raise ValueError("{}: smoke produced no valid output".format(case_id))
            result = results[0]
            if number % 2 == 0 and (result["draft_tokens"] <= 0 or result["draft_accepted"] <= 0):
                raise ValueError("{}: smoke did not propose and accept draft tokens".format(case_id))
            outputs.append(result["output_sha256"])
        checks.append({"dataset": dataset, "baseline_draft_text_match": outputs[0] == outputs[1]})
    save(args.output_dir / "smoke-check.json", {"checks": checks, "note": "Short output matches do not establish strict lossless parity."})


def export_results(args, completed):
    summaries = []
    for dataset in args.datasets:
        for number in PAIRS[dataset]:
            case_id = "LC-{:03d}".format(number)
            records = []
            for round_number in range(1, completed + 1):
                path = args.output_dir / "round-{:02d}".format(round_number) / case_id.lower() / "runs.jsonl"
                records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
            case = {"dataset": dataset, "prompt_tokens": None, "output_tokens": 512}
            summary = bench_test.summarize_case(case_id, case, records, args.warmup_runs, args.formal_runs * args.rounds,
                                                args.rounds, args.platform, args.device_index, records[0]["backend_device"])
            summary.update(rounds_completed=completed, formal_runs_per_round=args.formal_runs)
            generation = summary["metrics"]["generation_tps_measured"]
            summary["decode_tps"] = generation["mean"] if generation else None
            summaries.append(summary)
    run_suite.add_speculative_comparisons(summaries)
    by_case = {row["case_id"]: row for row in summaries}
    for row in summaries:
        if "baseline_case_id" in row:
            baseline = by_case[row["baseline_case_id"]]["metrics"]["tpot_ms_measured"]
            measured = row["metrics"]["tpot_ms_measured"]
            row["tpot_speedup_x"] = baseline["mean"] / measured["mean"] if baseline and measured and measured["mean"] > 0 else None
    save(args.output_dir / "summary.json", summaries)
    run_suite.write_csv(args.output_dir / "summary.csv", [{key: value for key, value in row.items() if key != "metrics"} for row in summaries])


def main():
    args = parse_args()
    samples = sample_plan(args)
    args.output_dir.mkdir(parents=True)
    save(args.output_dir / "sample-plan.json", samples)
    save(args.output_dir / "protocol.json", {**vars(args), "llama_server": str(args.llama_server), "model": str(args.model),
         "draft_model": str(args.draft_model), "processed_dir": str(args.processed_dir), "output_dir": str(args.output_dir),
         "max_output_tokens": 512, "cache_prompt": False, "thinking": False, "temperature": 0, "seed": 42,
         "execution_order": "round-major, baseline/draft paired per dataset"})
    suite = prepare_suite(args, args.output_dir / "configs")
    smoke_suite = prepare_suite(args, args.output_dir / "smoke-configs", smoke=True)
    environment = os.environ.copy()
    environment.update(NO_PROXY="*", no_proxy="*", LLAMA_ARG_FIT="off", GGML_SCHED_DEBUG="0")
    for key in ("PROFILING_MODE", "PROFILING_OPTIONS", "GGML_CANN_WQMM_DEBUG_CHECK"):
        environment.pop(key, None)
    if args.platform == "cann_310p":
        environment.update({key: "on" for key in CANN_FLAGS})
    state = {"started": now(), "stage": "preparing", "rounds_requested": args.rounds, "rounds_completed": 0}

    def execute(label, argv):
        state.update(stage=label, updated=now())
        save(args.output_dir / "status.json", state)
        save(args.output_dir / (label + "-command.json"), argv)
        print("{}: {}".format(label, args.output_dir / (label + ".log")), flush=True)
        with (args.output_dir / (label + ".log")).open("w", encoding="utf-8") as log:
            subprocess.run(argv, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        for dataset, plan in samples.items():
            print("{}: {} unique samples, {} formal requests per round".format(dataset, plan["unique_formal_samples"], plan["formal_requests_per_round"]), flush=True)
        execute("dry-run", command(args, suite, args.output_dir / "validation") + ["--dry-run", "--skip-model-hash"])
        if not args.dry_run:
            execute("smoke", command(args, smoke_suite, args.output_dir / "smoke") + ["--smoke"])
            check_smoke(args)
            if not args.smoke_only:
                for round_number in range(1, args.rounds + 1):
                    label = "round-{:02d}".format(round_number)
                    execute(label, command(args, suite, args.output_dir / label))
                    export_results(args, round_number)
                    state["rounds_completed"] = round_number
        state.update(stage="dry-run-complete" if args.dry_run else "smoke-complete" if args.smoke_only else "complete", finished=now())
        save(args.output_dir / "status.json", state)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        state.update(stage="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc), finished=now())
        save(args.output_dir / "status.json", state)
        raise


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print("run_speculative: {}".format(exc), file=sys.stderr)
        sys.exit(1)
