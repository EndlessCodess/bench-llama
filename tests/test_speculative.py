import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bench_server_run
import prepare_dataset
import prepare_speculative_datasets
FAKE_SERVER = r'''
import argparse, json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
if "--version" in sys.argv:
    print("speculative HTTP fixture")
    sys.exit(0)
parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, required=True)
args, _ = parser.parse_known_args()
draft = "--spec-draft-model" in sys.argv and not os.environ.get("SPEC_FIXTURE_ZERO_DRAFT")
counts = {"draft_tokens": 0, "accepted_tokens": 0, "drafts": 0}
def record(value):
    with open(os.environ["SPEC_FIXTURE_TRACE"], "a") as handle:
        handle.write(json.dumps(value) + "\n")
record({"event": "start", "pid": os.getpid(), "draft": draft, "argv": sys.argv})
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def answer(self, value, code=200, content_type="application/json"):
        content = value.encode() if isinstance(value, str) else json.dumps(value).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.end_headers()
        self.wfile.write(content)
    def do_GET(self):
        if self.path == "/metrics":
            text = "\n".join("llamacpp:spec_decode_num_" + key + "_total " + str(value) for key, value in counts.items())
            self.answer(text, content_type="text/plain")
        else:
            self.answer({"status": "ok"})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        record({"event": "request", "pid": os.getpid(), "draft": draft, "path": self.path, "body": body})
        if os.environ.get("SPEC_FIXTURE_FAIL"):
            self.answer({"error": "fixture failure"}, code=500)
            return
        if draft:
            counts["draft_tokens"] += 15
            counts["accepted_tokens"] += 2
            counts["drafts"] += 1
        completion = "    return 42\n\ndef unrelated():\n    return 0\n"
        matches = [completion.find(stop) for stop in body.get("stop", []) if stop in completion]
        if matches:
            completion = completion[:min(matches)]
        choice = {"text": completion} if self.path == "/v1/completions" else {"message": {"content": "return 42"}}
        self.answer({"usage": {"prompt_tokens": 3, "completion_tokens": 3},
                     "choices": [{**choice, "finish_reason": "stop"}],
                     "timings": {"prompt_ms": 1.0, "predicted_ms": 2.0 if draft else 4.0,
                                 "prompt_per_second": 3000.0, "predicted_per_second": 1000.0 if draft else 500.0,
                                 "draft_n": 15 if draft else 0, "draft_n_accepted": 2 if draft else 0}})
HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
'''


class SpeculativeCampaignTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="speculative-regression-")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.server = self.folder / "llama-server"
        self.server.write_text("#!" + sys.executable + "\n" + textwrap.dedent(FAKE_SERVER))
        self.server.chmod(0o755)
        self.model = self.folder / "target.gguf"
        self.model.write_text("fixture")
        self.trace = self.folder / "trace.jsonl"
        self.processed = self.folder / "processed"
        self.processed.mkdir()
        self.prompts = {}
        metadata, selections = [], []
        for dataset in ("alpaca", "math-500", "humaneval"):
            for index in range(2):
                relative = "prompts/{}/{:03d}.txt".format(dataset, index)
                text = "{} sample {}".format(dataset, index)
                if dataset == "humaneval":
                    text = '\n\ndef sample_{}(x):\n    """Return x.\n    """\n'.format(index)
                self.prompts[(dataset, index)] = text
                path = self.processed / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
                metadata.append({"dataset": dataset, "source_id": "{}/{}".format(dataset, index),
                                 "pool_index": index, "prompt_file": relative, "actual_tokens": 3,
                                 "target_tokens": 3, "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                 "request_mode": "completion" if dataset == "humaneval" else "chat"})
            for number, index in enumerate((0, 0, 1, 0, 1), 1):
                selections.append({"dataset": dataset, "run": number, "phase": "warmup" if number == 1 else "formal",
                                   "pool_index": index, "source_id": "{}/{}".format(dataset, index),
                                   "prompt_file": "prompts/{}/{:03d}.txt".format(dataset, index)})
        (self.processed / "metadata.jsonl").write_text("".join(json.dumps(row) + "\n" for row in metadata))
        (self.processed / "preparation.json").write_text("{}")
        (self.processed / "run-manifest.json").write_text(json.dumps({"warmup_runs": 1, "formal_runs": 4, "selections": selections}))

    def run_campaign(self, name, *extra, fail=False, zero_draft=False):
        output = self.folder / name
        argv = [sys.executable, str(ROOT / "run_speculative.py"), "--llama-server", str(self.server),
                "--model", str(self.model), "--draft-model", str(self.model), "--processed-dir", str(self.processed),
                "--platform", "cpu", "--device-index", "0", "--warmup-runs", "1", "--formal-runs", "2",
                "--output-dir", str(output), *extra]
        environment = {**os.environ, "SPEC_FIXTURE_TRACE": str(self.trace)}
        if fail:
            environment["SPEC_FIXTURE_FAIL"] = "1"
        if zero_draft:
            environment["SPEC_FIXTURE_ZERO_DRAFT"] = "1"
        result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment, timeout=90)
        return output, result

    def test_round_order_persistent_requests_and_aggregate(self):
        output, result = self.run_campaign("success", "--rounds", "2")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(json.loads((output / "status.json").read_text())["stage"], "complete")
        rows = json.loads((output / "summary.json").read_text())
        self.assertEqual([row["case_id"] for row in rows], ["LC-{:03d}".format(i) for i in range(33, 39)])
        self.assertTrue(all(row["formal_runs_ok"] == 4 and row["formal_runs_failed"] == 0 for row in rows))
        self.assertEqual(rows[-1]["tpot_speedup_x"], 2.0)
        self.assertEqual(rows[-1]["decode_tps"], 1000.0)
        trace = [json.loads(line) for line in self.trace.read_text().splitlines()]
        starts = [row for row in trace if row["event"] == "start"]
        self.assertEqual(len(starts), 18)  # Six smoke cases, then six cases in each round.
        requests = [row for row in trace if row["event"] == "request"]
        self.assertEqual(len(requests), 6 + 12 * 3)
        for start in starts:
            self.assertEqual(start["argv"][start["argv"].index("--threads") + 1], "8")
            actual = [row for row in requests if row["pid"] == start["pid"]]
            self.assertIn(len(actual), (1, 3))
        expected = [self.prompts[(d, 0)] for d in ("alpaca", "alpaca", "math-500", "math-500", "humaneval", "humaneval")]
        def prompt(row):
            return row["body"]["prompt"] if row["path"] == "/v1/completions" else row["body"]["messages"][0]["content"]
        self.assertEqual([next(prompt(row) for row in requests if row["pid"] == start["pid"]) for start in starts], expected * 3)
        for row in requests:
            self.assertFalse(row["body"]["cache_prompt"])
            self.assertFalse(row["body"]["stream"])
            if row["path"] == "/v1/completions":
                self.assertIn(row["body"]["prompt"], [self.prompts[("humaneval", i)] for i in range(2)])
                self.assertNotIn("messages", row["body"])
                self.assertNotIn("chat_template_kwargs", row["body"])
                self.assertEqual(row["body"]["stop"], ["\nclass", "\ndef", "\n#", "\nif", "\nprint"])
            else:
                self.assertEqual(row["path"], "/v1/chat/completions")
                self.assertFalse(row["body"]["chat_template_kwargs"]["enable_thinking"])
        events = [json.loads(line) for line in (output / "round-01/lc-038/LC-038/session.stdout.log").read_text().splitlines()]
        self.assertEqual([row["output_text"] for row in events if row["event"] == "result"], ["    return 42\n"] * 3)
        self.assertTrue(all(row["request_mode"] == "completion" for row in events if row["event"] in ("run_start", "result")))
        records = [json.loads(line) for line in (output / "round-01/lc-038/runs.jsonl").read_text().splitlines()]
        self.assertTrue(all(row["request_mode"] == "completion" for row in records))

    def test_repeated_formal_samples_require_explicit_flag(self):
        output, result = self.run_campaign("rejected", "--formal-runs", "4", "--dry-run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--allow-repeated-samples", result.stdout)
        self.assertFalse(output.exists())
        output, result = self.run_campaign("allowed", "--formal-runs", "4", "--allow-repeated-samples", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stdout)
        plan = json.loads((output / "sample-plan.json").read_text())
        self.assertEqual(plan["humaneval"]["unique_formal_samples"], 2)
        self.assertEqual(plan["humaneval"]["formal_requests_per_round"], 4)
        self.assertFalse(self.trace.exists())

    def test_smoke_failure_stops_formal_execution(self):
        output, result = self.run_campaign("failed", fail=True)
        self.assertNotEqual(result.returncode, 0)
        state = json.loads((output / "status.json").read_text())
        self.assertEqual(state["stage"], "failed")
        self.assertEqual(state["rounds_completed"], 0)
        self.assertFalse((output / "round-01").exists())

    def test_smoke_without_drafting_stops_formal_execution(self):
        output, result = self.run_campaign("zero-draft", "--datasets", "humaneval", zero_draft=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(json.loads((output / "status.json").read_text())["stage"], "failed")
        self.assertIn("smoke did not propose and accept draft tokens", result.stdout)
        self.assertFalse((output / "round-01").exists())

    def test_humaneval_loader_preserves_original_prompt(self):
        original = self.prompts[("humaneval", 0)]
        math = self.folder / "math.jsonl"
        math.write_text(json.dumps({"unique_id": "math/0", "problem": "1+1"}) + "\n")
        args = SimpleNamespace(alpaca=self.folder / "alpaca.parquet", math500=math, humaneval=self.folder / "human.parquet")
        with mock.patch.object(prepare_speculative_datasets, "read_parquet", side_effect=[
            [{"instruction": "hi", "input": ""}], [{"task_id": "HumanEval/0", "prompt": original}],
        ]):
            row = prepare_speculative_datasets.load_datasets(args)["humaneval"][0]
        self.assertEqual(row["prompt"], original)
        self.assertEqual(row["source_id"], "HumanEval/0")

    def test_raw_token_count_bypasses_chat_template(self):
        tokenizer = prepare_dataset.TokenizerServer(self.server, self.model, self.folder / "tokenizer.log", 1)
        original = self.prompts[("humaneval", 0)]
        with mock.patch.object(prepare_dataset, "request_json", return_value={"tokens": [1, 2, 3]}) as request:
            self.assertEqual(tokenizer.count_prompt_tokens(original, chat=False), 3)
        self.assertEqual(request.call_count, 1)
        self.assertEqual(request.call_args.args[2], "/tokenize")
        self.assertEqual(request.call_args.args[3]["content"], original)
        self.assertTrue(request.call_args.args[3]["add_special"])

    def test_legacy_humaneval_metadata_is_rejected_before_loading_model(self):
        path = self.processed / "metadata.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            if row["dataset"] == "humaneval":
                row["request_mode"] = "chat"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        output, result = self.run_campaign("legacy", "--datasets", "humaneval", "--dry-run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("request_mode=completion", (output / "dry-run.log").read_text())
        self.assertFalse(self.trace.exists())

    def test_old_instruction_wrapper_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "original dataset prompt"):
            bench_server_run.validate_humaneval_prompt("Complete the following Python function. Return the completed code.\n")


if __name__ == "__main__":
    unittest.main()
