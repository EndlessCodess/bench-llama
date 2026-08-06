#!/usr/bin/env python3

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SUITE = PROJECT_ROOT / "configs" / "lc-001-028-suite.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Run a configured group of llama-cli benchmark jobs")
    parser.add_argument("--llama-cli", default=os.environ.get("LLAMA_CLI"), help="path to llama-cli")
    parser.add_argument("--suite", default=str(DEFAULT_SUITE), help="suite configuration")
    parser.add_argument("--model", action="append", default=[], metavar="KEY=PATH")
    parser.add_argument("--job", action="append", default=[], help="job id or comma-separated job ids")
    parser.add_argument("--group", action="append", default=[], metavar="N",
                        help="worksheet category: 0=fair baseline, 1=weight quantization, 2=KV cache precision, 3=attention backend, 4=polling; repeatable or comma-separated")
    parser.add_argument("--platform", default="auto")
    parser.add_argument("--platform-config", default=str(PROJECT_ROOT / "configs" / "platforms.json"))
    parser.add_argument("--processed-dir", default=str(PROJECT_ROOT / "data" / "processed" / "qwen3.5-9b"))
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--output-dir", help="exact suite output directory")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--backend-device")
    parser.add_argument("--monitor-interval", type=float, default=0.1)
    parser.add_argument("--memory-command")
    parser.add_argument("--timeout", type=float, default=0.0)
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--skip-model-hash", action="store_true")
    parser.add_argument("--skip-prompt-hash-check", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="run one formal invocation of each job's first case")
    parser.add_argument("--dry-run", action="store_true", help="validate all cases and print one command per case")
    return parser.parse_args()


def parse_assignments(values, label):
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("{} must use KEY=VALUE: {}".format(label, value))
        key, item = value.split("=", 1)
        if not key or not item:
            raise ValueError("{} must use non-empty KEY=VALUE: {}".format(label, value))
        result[key] = item
    return result


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_job_ids(values):
    selected = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item and item not in selected:
                selected.append(item)
    return selected


def selected_groups(values):
    selected = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item and item not in selected:
                selected.append(item)
    return selected


def load_suite(path, selected_ids, selected_groups):
    suite_path = Path(path).resolve()
    if not suite_path.is_file():
        raise ValueError("suite config does not exist: {}".format(suite_path))
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    if suite.get("schema_version") != 1:
        raise ValueError("unsupported suite schema: {}".format(suite.get("schema_version")))

    jobs = []
    known = set()
    known_groups = set()
    for item in suite.get("jobs", []):
        job_id = item.get("id")
        if not job_id or job_id in known:
            raise ValueError("suite job ids must be non-empty and unique")
        known.add(job_id)
        if "group" in item:
            known_groups.add(str(item["group"]))
        config_path = (suite_path.parent / item["config"]).resolve()
        if not config_path.is_file():
            raise ValueError("job config does not exist: {}".format(config_path))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("schema_version") != 1 or not config.get("cases"):
            raise ValueError("invalid benchmark config: {}".format(config_path))
        declared_model_key = config.get("experiment", {}).get("model_key")
        if declared_model_key and declared_model_key != item.get("model_key"):
            raise ValueError("model key mismatch between suite and {}".format(config_path))
        jobs.append({**item, "config_path": config_path, "definition": config})

    unknown = set(selected_ids) - known
    if unknown:
        raise ValueError("unknown suite job(s): {}".format(", ".join(sorted(unknown))))
    if selected_ids:
        jobs = [job for job in jobs if job["id"] in selected_ids]
    unknown_groups = set(selected_groups) - known_groups
    if unknown_groups:
        raise ValueError("unknown suite group(s): {}".format(", ".join(sorted(unknown_groups))))
    if selected_groups:
        jobs = [job for job in jobs if str(job.get("group")) in selected_groups]
    if not jobs:
        raise ValueError("suite has no selected jobs")
    return suite_path, suite, jobs


def job_label(job):
    group = job.get("group")
    return "g{}/{}".format(group, job["id"]) if group is not None else job["id"]


def append_common_args(command, args):
    command.extend([
        "--platform", args.platform,
        "--platform-config", str(Path(args.platform_config).resolve()),
        "--processed-dir", str(Path(args.processed_dir).resolve()),
        "--device-index", str(args.device_index),
        "--monitor-interval", str(args.monitor_interval),
        "--timeout", str(args.timeout),
    ])
    if args.backend_device:
        command.extend(["--backend-device", args.backend_device])
    if args.memory_command:
        command.extend(["--memory-command", args.memory_command])
    if args.skip_prompt_hash_check:
        command.append("--skip-prompt-hash-check")
    for value in args.env:
        command.extend(["--env", value])
    for value in args.extra_arg:
        command.extend(["--extra-arg", value])


def write_csv(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if not args.llama_cli:
        raise ValueError("--llama-cli or LLAMA_CLI is required")
    llama_cli = Path(args.llama_cli).resolve()
    if not llama_cli.is_file() or not os.access(llama_cli, os.X_OK):
        raise ValueError("llama-cli is not executable: {}".format(llama_cli))

    selected_ids = selected_job_ids(args.job)
    selected_group_ids = selected_groups(args.group)
    suite_path, suite, jobs = load_suite(args.suite, selected_ids, selected_group_ids)
    model_values = parse_assignments(args.model, "--model")
    required_keys = {job["model_key"] for job in jobs}
    missing = required_keys - set(model_values)
    if missing:
        raise ValueError("missing model mapping(s): {}".format(", ".join(sorted(missing))))

    models = {}
    for key in sorted(required_keys):
        path = Path(model_values[key]).resolve()
        if not path.is_file():
            raise ValueError("model {} does not exist: {}".format(key, path))
        models[key] = {"path": str(path), "size": path.stat().st_size, "sha256": None}

    bench_script = PROJECT_ROOT / "bench_test.py"
    base_commands = []
    for job in jobs:
        command = [
            sys.executable,
            str(bench_script),
            "--llama-cli", str(llama_cli),
            "--model", models[job["model_key"]]["path"],
            "--config", str(job["config_path"]),
        ]
        append_common_args(command, args)
        base_commands.append((job, command))

    if args.dry_run:
        print("suite: {}".format(suite["name"]))
        print("dry-run validates every case with one representative formal command")
        for job, command in base_commands:
            command.extend(["--warmup-runs", "0", "--formal-runs", "1", "--dry-run"])
            print("\n[{}] {}".format(job_label(job), shlex.join(command)), flush=True)
            result = subprocess.run(command, check=False)
            if result.returncode != 0 and not args.continue_on_error:
                return result.returncode
        return 0

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (
        Path(args.results_dir).resolve() / "{}-{}-{}-device{}".format(
            timestamp, suite["name"], args.platform, args.device_index
        )
    )
    if output_dir.exists():
        raise ValueError("output directory already exists: {}".format(output_dir))
    output_dir.mkdir(parents=True)

    if not args.skip_model_hash:
        for key, item in models.items():
            print("calculating {} model SHA-256...".format(key), flush=True)
            item["sha256"] = sha256_file(Path(item["path"]))

    run_jobs = []
    aggregate = []
    returncode = 0
    for job, command in base_commands:
        job_output = output_dir / job["id"]
        command.extend(["--output-dir", str(job_output), "--fail-fast"])
        model_info = models[job["model_key"]]
        if model_info["sha256"]:
            command.extend(["--model-sha256", model_info["sha256"]])
        else:
            command.append("--skip-model-hash")
        if args.smoke:
            first_case = job["definition"]["cases"][0]["id"]
            command.extend(["--case", first_case, "--warmup-runs", "0", "--formal-runs", "1"])

        print("\n[{}] starting".format(job_label(job)), flush=True)
        result = subprocess.run(command, check=False)
        status = "ok" if result.returncode == 0 else "failed"
        run_jobs.append({
            "id": job["id"],
            "group": job.get("group"),
            "status": status,
            "returncode": result.returncode,
            "config": str(job["config_path"]),
            "model_key": job["model_key"],
            "output_dir": str(job_output),
            "command": shlex.join(command),
        })
        summary_path = job_output / "summary.json"
        if summary_path.is_file():
            run_errors = {}
            runs_path = job_output / "runs.jsonl"
            if runs_path.is_file():
                for line in runs_path.read_text(encoding="utf-8").splitlines():
                    run = json.loads(line)
                    if run.get("error"):
                        run_errors.setdefault(run["case_id"], []).append(run["error"])
            for row in json.loads(summary_path.read_text(encoding="utf-8")):
                aggregate.append({
                    "job_id": job["id"],
                    "group": job.get("group"),
                    "experiment_variable": job["definition"].get("experiment", {}).get("variable"),
                    "experiment_value": job["definition"].get("experiment", {}).get("value"),
                    "model_key": job["model_key"],
                    "errors": "; ".join(run_errors.get(row["case_id"], [])),
                    **row,
                })
        if result.returncode != 0:
            returncode = result.returncode
            if not args.continue_on_error:
                break

    manifest = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "suite_config": str(suite_path),
        "suite_definition": suite,
        "mode": "smoke" if args.smoke else "formal",
        "platform": args.platform,
        "physical_device_index": args.device_index,
        "llama_cli": str(llama_cli),
        "models": models,
        "jobs": run_jobs,
    }
    (output_dir / "suite-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(output_dir / "summary.csv", [
        {key: value for key, value in row.items() if key != "metrics"} for row in aggregate
    ])
    print("\nsuite results: {}".format(output_dir))
    return returncode


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        sys.exit(130)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        sys.exit(2)
