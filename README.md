# bench_test_llama

[简体中文](README_ZH.md)

`bench_test_llama` is a standalone benchmark project for comparing llama.cpp
backends on different hardware. It is kept outside the llama.cpp source tree so
that benchmark code, datasets, manifests, and results do not modify the upstream
repository.

The current target platforms are:

- NVIDIA CUDA, initially RTX 5090
- Huawei Ascend CANN, including 310P and 910B
- Apple Metal

## Methodology: single-turn requests on a persistent process

The formal benchmark runs each case (5 warmup + 100 formal runs) inside **one
`bench_run` process**: the model is loaded once, and every run is an
independent **single-turn** request in the same `llama_context` with the KV
cache cleared between runs (`llama_memory_clear`), so context never accumulates.
This mirrors the request-based persistent-engine semantics of the SGLang/vLLM
benchmarks:

- **Warmup is meaningful**: model loading, CUDA context, and kernel
  initialization happen exactly once per case; the warmup runs absorb these
  one-time costs and the formal runs measure steady-state single-turn latency.
- **Context never accumulates**: each run only occupies its own prompt's KV
  cache, so all four input lengths (2K..28K) fit on a 32 GiB GPU, and the
  per-run memory delta reflects one run's KV plus compute buffers.
- **No `llama-cli`**: a small C++ tool (`bench_run.cpp`) calls `llama_decode`
  directly and measures each run inside the process with `ggml_time_us` plus
  `llama_synchronize` (CUDA is asynchronous), emitting one JSONL result event
  per run on stdout.

The chat template is applied inside `bench_run` through the same
`common_chat_templates_apply` code path as `llama-server /apply-template`
(including reasoning-template workarounds), so token counts match the prepared
metadata exactly, and greedy sampling (`--temp 0` plus an EOG logit bias, i.e.
`--ignore-eos`) keeps generation deterministic at the exact requested length.

## Status

Completed:

- Downloaded and verified the official LongBench-v2 dataset
- Implemented reproducible prompt preprocessing
- Generated fixed 2K, 8K, 16K, and 28K prompt pools
- Generated a fixed 5-run warmup and 10-run formal manifest
- Implemented the `bench_run` C++ runner (single-turn, persistent process)
- Added CUDA, CANN, Metal, and CPU device selection
- Added per-run logs, memory monitoring, and unified summaries
- Converted and verified F16, Q8_0, and Q4_K_M model variants
- Added six one-variable configurations and a suite runner for LC-005 through LC-028

In progress:

- Formal RTX 5090 baseline runs
- Hardware validation on 310P, 910B, and Metal

The existing result under `results/20260805-120212-cli-mock-cuda-gpu1-LC-001`
is an old pipeline mock using Qwen3-0.6B. It is not a formal Qwen3.5-9B result.

## Fair baseline

The initial fair baseline is defined as follows:

| Parameter | Value |
| --- | --- |
| Model | Qwen3.5-9B BF16 GGUF |
| Input lengths | 2048, 8192, 16384, 28672 tokens |
| Output lengths | 64, 256, 256, 512 tokens |
| Context size | 262144 tokens |
| Concurrency | 1 |
| Parallel sequences | 1 |
| GPU layers | 99 |
| Batch size | 2048 |
| Micro-batch size | 512 |
| KV cache | F16 K and F16 V |
| Flash Attention | auto |
| Poll | 50 |
| Temperature | 0 |
| Prefix cache hit | 0 |
| Warmup runs | 5 per input length |
| Formal runs | 10 per input length |

All platforms must use the same model, prompt files, run order, and output
lengths. Platform-specific options are limited to backend and device selection.

## Project layout

```text
bench_test_llama/
|-- README.md
|-- README_ZH.md
|-- bench_run.cpp      <- C++ runner: one process runs the whole plan
|-- CMakeLists.txt
|-- bench_test.py      <- Python orchestrator (config, plan, monitoring, summary)
|-- run_suite.py
|-- prepare_dataset.py
|-- build/             <- bench_run build output
|-- configs/
|   |-- fair-baseline.json
|   `-- platforms.json
|-- data/
|   |-- longbench-v2/
|   |   `-- data.json
|   `-- processed/
|       `-- qwen3.5-9b/
|           |-- prompts/
|           |   |-- 2048/
|           |   |-- 8192/
|           |   |-- 16384/
|           |   `-- 28672/
|           |-- metadata.jsonl
|           |-- preparation.json
|           |-- run-manifest.json
|           |-- source-pool.json
|           `-- tokenizer-server.log
|-- results/
|-- llama-bench-matrix.py
|-- llama-bench-matrix.csv
`-- llama-bench-matrix.md
```

The `llama-bench-matrix.*` files are an earlier `llama-bench` prototype. They
will not be used as the formal runner.

Benchmark cases and hardware adaptation are both configuration-driven:

- `configs/fair-baseline.json` defines cases and common baseline parameters.
- `configs/platforms.json` defines visibility variables, llama.cpp device names,
  and memory monitoring methods.

`bench_test.py` only handles the fixed run manifest, plan generation, process
execution, metric collection, and common result format. It does not hard-code
an RTX 5090 or a physical device number. New backends should normally be added
as profiles without changing the benchmark flow.

## Build bench_run

`bench_run` links against an existing llama.cpp build's shared libraries, so
building llama.cpp itself is not required:

```bash
cd /path/to/bench_test_llama
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CPP_ROOT=/path/to/llama.cpp \
  -DLLAMA_CPP_BUILD=/path/to/llama.cpp/build-cuda13
cmake --build build -j
```

- `LLAMA_CPP_ROOT` is the llama.cpp source tree (for headers and `common/`).
- `LLAMA_CPP_BUILD` is the build directory containing the shared libraries in
  `bin/` (`.so` on Linux and `.dylib` on macOS). Point it at the CUDA, CANN, or
  Metal build made for the target platform.
- `--version` prints the llama.cpp system info; a bare run with a 2-line
  `plan.tsv` and `n_predict=8` is a quick smoke test.

### One-command macOS / Apple Metal test

Place `llama.cpp` and this repository next to each other, then run:

```bash
cd bench_test_llama
./macos_bench.sh --model /path/to/Qwen3.5-9B.gguf
```

This builds Release + Metal versions of llama.cpp and `bench_run`, then runs
one LC-001 warmup and one formal request as a smoke test. Run the formal
LC-001 through LC-004 baseline only after the smoke test succeeds:

```bash
./macos_bench.sh --model /path/to/Qwen3.5-9B.gguf --full
```

Use `--llama-root /path/to/llama.cpp` when the repositories are not siblings.
Apple Silicon uses unified memory, so the current result records process RSS
instead of reporting it as dedicated GPU memory. TTFT, TPOT, ITL, and E2E are
collected normally.

## LC-001 through LC-028

`configs/lc-001-028-suite.json` covers all 28 llama.cpp cases of the v6.1 test
matrix in one command: the fair baseline (`fair-baseline.json`, LC-001..004)
plus six one-variable groups (`configs/lc-005-028-suite.json`, LC-005..028):
Q8_0 weights, Q4_K_M weights, Q8_0 KV cache, Q4_0 KV cache, Flash Attention
off, and poll 0. The suite maps logical model keys (`f16`, `q8_0`, and
`q4_k_m`) to local model paths, so the same experiment definitions can be used
on CUDA, CANN, and Metal.

Validate all 28 cases without inference:

```bash
./run_suite.py --bench-run /path/to/bench_run \
  --model f16=/path/to/Qwen3.5-9B-F16.gguf \
  --model q8_0=/path/to/Qwen3.5-9B-Q8_0.gguf \
  --model q4_k_m=/path/to/Qwen3.5-9B-Q4_K_M.gguf \
  --platform cuda --device-index 4 --dry-run
```

Use `--smoke --continue-on-error` for one invocation of the first case in every group. Without
`--smoke`, the runner executes the configured 5 warmups and 10 formal runs for
all 28 cases. Conversion commands, sizes, hashes, and the absence of an imatrix
for the current Q4_K_M conversion are recorded in `configs/model-recipes.json`.

The worksheet test matrix is divided into seven categories; `--group` runs one
category at a time (repeatable or comma-separated, combined with `--job`):

```text
--group 0   fair baseline          (LC-001..004)
--group 1   weight quantization    (LC-005..012)
--group 2   KV cache precision     (LC-013..020)
--group 3   attention backend      (LC-021..024)
--group 4   polling                (LC-025..028)
--group 5   backend graph off      (LC-029..032)
--group 6   DFlash speculation     (LC-033..038)
```

For example, run only the weight-quantization block:

```bash
./run_suite.py --bench-run /path/to/bench_run \
  --model f16=/path/to/Qwen3.5-9B-F16.gguf \
  --model q8_0=/path/to/Qwen3.5-9B-Q8_0.gguf \
  --model q4_k_m=/path/to/Qwen3.5-9B-Q4_K_M.gguf \
  --platform cuda --device-index 4 --group 1
```

Without `--group` or `--job`, the whole suite runs.

### DFlash speculative decoding (LC-033 through LC-038)

The six cases form three controlled pairs: Alpaca (LC-033 off / LC-034 on), MATH-500 (LC-035 / LC-036), and HumanEval (LC-037 / LC-038). Each pair reuses the same prompts and runtime settings; only the DFlash draft is changed. Inputs retain their natural dataset lengths. Generation stops at EOS with a maximum of 512 tokens. One round contains 5 warmups and 100 formal requests, so `--rounds 10` produces 1000 formal requests per case. These cases use the official `llama-server` speculative path; the adapter records the same JSONL events consumed by the common benchmark orchestrator.

Prepare the deterministic 50-prompt pools once:

```bash
python3 prepare_speculative_datasets.py \
  --model /path/to/Qwen3.5-9B-F16.gguf \
  --output data/processed/qwen3.5-9b-speculative-nothink \
  --llama-server /path/to/llama-server
```

Run all six cases with the target and DFlash GGUF files:

```bash
./run_suite.py --bench-run /path/to/bench_run \
  --llama-server /path/to/llama-server \
  --suite configs/lc-001-038-suite.json \
  --model-profile qwen3.5-9b \
  --model f16=/path/to/Qwen3.5-9B-F16.gguf \
  --model dflash_f16=/path/to/Qwen3.5-9B-DFlash-F16.gguf \
  --platform cuda --device-index 6 --group 6 --rounds 10
```

The enabled rows record draft acceptance. Suite summaries also add generation and end-to-end speedup relative to the paired disabled row.

### Model profiles

Model profiles keep tokenizer-specific data and model runtime limits out of the benchmark logic. Select a profile by name and provide paths using the resolved model keys:

```bash
./run_suite.py --bench-run /path/to/bench_run \
  --suite configs/lc-001-038-suite.json \
  --model-profile qwen3-8b --job dflash-on \
  --processed-dir data/processed/qwen3-8b-speculative \
  --model bf16=/path/to/Qwen3-8B-BF16.gguf \
  --model dflash_bf16=/path/to/dflash-Qwen3-8B-BF16.gguf \
  --platform cuda --device-index 0
```

The Qwen3-8B profile selects `data/processed/qwen3-8b`, maps the suite's 16-bit target role to BF16, and uses its native 40960-token context limit. Prepare the tokenizer-specific prompts before the first run:

```bash
python3 prepare_dataset.py \
  --model /path/to/Qwen3-8B-BF16.gguf \
  --llama-server /path/to/llama-server \
  --output data/processed/qwen3-8b \
  --pool-size 50 --warmup-runs 5 --formal-runs 100
```

An explicit `--processed-dir` overrides the profile. Commands without `--model-profile` retain the previous Qwen3.5-9B defaults.

Because context never accumulates, every case fits on a 32 GiB single GPU even
in the F16 fair-baseline configuration. The exact Flash-Attention-off
configuration (LC-021..024) still needs more memory than the current RTX 5090
provides: F16 weights, a 256K context, and batch 2048 require an additional
compute buffer of approximately 8.54 GiB. The configuration is intentionally
unchanged for execution on larger-memory devices; silently reducing the context
or batch would invalidate the one-variable comparison.

## Requirements

- Python 3.8 or newer
- The llama.cpp source tree plus a build with shared libraries
  (`libllama`, `libllama-common`, `libggml`, `libggml-base`) for `bench_run`,
  and a build containing `llama-server` for dataset preprocessing
- A Qwen3.5-9B GGUF model
- No third-party Python packages are required for preprocessing

llama.cpp and the model are external dependencies. Their paths are passed on
the command line and are not copied into this project.

## Download LongBench-v2

The official dataset contains 503 samples and is approximately 444 MiB. A
domestic Hugging Face mirror can be selected with `HF_ENDPOINT`:

```bash
cd /path/to/bench_test_llama

mkdir -p data/longbench-v2
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

wget -c \
  "${HF_ENDPOINT}/datasets/zai-org/LongBench-v2/resolve/main/data.json?download=true" \
  -O data/longbench-v2/data.json
```

Verify the source file:

```bash
echo "15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2  data/longbench-v2/data.json" \
  | sha256sum -c -
```

The raw dataset is never modified by preprocessing.

## Prepare prompt pools

Set paths to the local llama.cpp build and GGUF model:

```bash
cd /path/to/bench_test_llama

LLAMA_CPP_ROOT=/path/to/llama.cpp
MODEL=/path/to/Qwen3.5-9B-BF16.gguf

./prepare_dataset.py \
  --model "$MODEL" \
  --llama-server "$LLAMA_CPP_ROOT/build/bin/llama-server"
```

Use `--force` to replace an existing processed directory:

```bash
./prepare_dataset.py \
  --model "$MODEL" \
  --llama-server "$LLAMA_CPP_ROOT/build/bin/llama-server" \
  --force
```

The defaults are:

```text
targets      = 2048,8192,16384,28672
pool size    = 30 source samples
seed         = 20260805
warmup runs  = 5
formal runs  = 10
sampling     = with replacement
```

Run `./prepare_dataset.py --help` for all options.

## Preprocessing method

Each LongBench-v2 item contains a long `context`, a question, four choices, and
a reference answer. Preprocessing performs the following steps:

1. Select 30 source samples, balanced across the six dataset domains.
2. Keep the question and all four choices intact.
3. Truncate only the source context.
4. Apply the Qwen chat template through llama.cpp.
5. Count tokens through llama.cpp's native tokenizer.
6. Search for the longest context prefix that does not exceed the target.
7. Store the prompt text, actual token count, source ID, and SHA-256.

The reference answer is stored in metadata but is never included in the model
prompt. No repeated filler text is added to reach a target length.

The preprocessor starts a temporary local `llama-server` and uses its
`/apply-template` and `/tokenize` endpoints. The server is started with
accelerators hidden because preprocessing does not perform model inference.
The model is loaded once, so hundreds of tokenizer calls do not repeatedly pay
GGUF startup cost.

## Generated data

The current processed pool contains 120 prompt files:

| Target | Files | Actual token counts |
| ---: | ---: | --- |
| 2048 | 30 | 27 x 2048, 3 x 2047 |
| 8192 | 30 | 28 x 8192, 2 x 8191 |
| 16384 | 30 | 27 x 16384, 3 x 16383 |
| 28672 | 30 | 30 x 28672 |

The one-token differences occur because a Unicode character boundary can move
the BPE count by more than one token. Actual counts are always recorded in
`metadata.jsonl`. The maximum observed difference is one token.

Important generated files:

- `prompts/<target>/*.txt`: raw user prompts consumed by `bench_run`
- `metadata.jsonl`: source, length, hash, and reference metadata per prompt
- `source-pool.json`: the fixed balanced source pool
- `run-manifest.json`: the fixed random run order shared by all targets
- `preparation.json`: source checksum, tokenizer model, and preparation options

The same source IDs and run order are used across all four target lengths. This
keeps content variation controlled when measuring context-length scaling.

## Run the formal baseline

### LC-003 length discrepancy

The v6.1 workbook is internally inconsistent. Its llama.cpp `LC-003` row uses
24576 input tokens, while the overall rule, the corresponding SGLang/vLLM tier,
and the previously confirmed plan use 16384. The current prompt pool and
`configs/fair-baseline.json` retain the confirmed 16384-token version:

```text
LC-001 = 2048  input / 64  output
LC-002 = 8192  input / 256 output
LC-003 = 16384 input / 256 output
LC-004 = 28672 input / 512 output
```

This is therefore the agreed 16K baseline, not an exact transcription of the
workbook's llama.cpp LC-003 cell. If 24576 is selected later, regenerate the
prompt pool with `prepare_dataset.py --targets 2048,8192,24576,28672` and update
the benchmark configuration; the runner does not need to change.

First validate the inputs and inspect every command (including the generated
`plan.tsv`) without starting inference:

```bash
cd /path/to/bench_test_llama

./bench_test.py \
  --bench-run /path/to/bench_run \
  --model /path/to/Qwen3.5-9B-BF16.gguf \
  --platform cuda \
  --device-index 4 \
  --dry-run
```

Remove `--dry-run` to execute all four cases. For CUDA, `--device-index 4`
selects physical GPU 4. The runner sets `CUDA_VISIBLE_DEVICES=4`, so the device
passed to llama.cpp is `CUDA0`.

Run one case as a smoke test:

```bash
./bench_test.py \
  --bench-run /path/to/bench_run \
  --model /path/to/Qwen3.5-9B-BF16.gguf \
  --platform cuda \
  --device-index 4 \
  --case LC-001 \
  --warmup-runs 0 \
  --formal-runs 1
```

`BENCH_RUN` and `LLAMA_MODEL` can be used instead of the corresponding path
arguments. Each case runs as one `bench_run` process: the model is loaded once,
and all 105 prompts (5 warmup + 100 formal) run sequentially as independent
single-turn requests with the KV cache cleared between runs. `--parallel` must
be 1. The plan file (`plan.tsv`) written under each case directory carries the
prompt path, phase, manifest run, source ID, output length, and expected token
count for every run.

Each run creates `results/<timestamp>-bench-run-<platform>-device<index>/` with:

- `manifest.json` for software, model, parameters, and the source run manifest
- per-case `plan.tsv`, `session.command.txt`, `session.stdout.log`, and
  `session.stderr.log`
- `runs.jsonl` and `runs.csv` for individual measurements
- `summary.json` and `summary.csv` aggregating successful formal runs only

`prompt_tps_measured`, `generation_tps_measured`, `prompt_eval_ms_measured`,
`ttft_ms_measured` (prompt evaluation plus sampling — the first token is
produced by the prefill forward pass itself, so no extra decode is included;
this matches the serving-engine (vLLM/SGLang) definition of TTFT),
`generation_eval_ms_measured`, and `e2e_ms_measured` are measured inside
`bench_run` with `ggml_time_us` and `llama_synchronize`. `tpot_ms_measured` and
`itl_ms_measured` are 1000 / `generation_tps_measured` (identical by definition
for single-stream greedy generation). The per-run wall time, RSS, and
whole-device peak memory are externally measured, and the memory window resets
at every `run_start` event so the device-memory delta reflects one run's KV
plus compute buffers. CUDA memory is total physical-device usage, so both its
baseline and delta are recorded.

`summary.csv` / `summary.json` report the five headline metrics with bilingual
"Chinese/English" headers, one row per case: 首令牌延迟(ms)/TTFT(ms),
每令牌时间(ms/token)/TPOT(ms/token), 令牌间延迟(ms/token)/ITL(ms/token),
端到端耗时(s)/E2E(s), and 峰值显存(GB)/Peak Memory(GB). TTFT/TPOT/ITL are
means over the successful formal runs, E2E is the mean `e2e_ms_measured`
(converted to seconds), and peak memory is the maximum sampled device usage.
Full per-turn detail stays in `runs.jsonl` and `runs.csv`.
For speculative runs, these files also record proposed and accepted draft tokens, target verification steps, mean accepted draft tokens per verification, and mean advance length. The last value includes the one token produced by the target model at each verification step.

Run `./bench_test.py --help` for all options or `./bench_test.py --list-cases`
for the four fixed cases.

## Multi-platform execution

Every platform reuses the same baseline config, prompt pool, and run manifest.
Only the platform-specific executable, model path, profile, and physical device
selection change:

```bash
# NVIDIA CUDA
./bench_test.py --platform cuda --device-index 4 \
  --bench-run /path/to/cuda/bench_run --model /path/to/model.gguf

# Ascend CANN
./bench_test.py --platform cann --device-index 0 \
  --bench-run /path/to/cann/bench_run --model /path/to/model.gguf

# Apple Metal
./bench_test.py --platform metal \
  --bench-run /path/to/metal/bench_run --model /path/to/model.gguf
```

The built-in profiles map CUDA to `CUDA_VISIBLE_DEVICES` plus `CUDA0`, CANN to
`ASCEND_RT_VISIBLE_DEVICES` plus `CANN0`, Metal to `Metal`, and CPU to `CPU` with
`-ngl 0`. CUDA memory is monitored through NVML. Other platforms can provide a
command that prints memory usage in MiB through `--memory-command` or their
profile. CANN and Metal have only passed command-generation dry runs so far;
their backend names and memory commands still require validation on the target
machines.

## Reproducibility rules

- Do not randomly sample again for each platform.
- Reuse the committed or archived `run-manifest.json` on every platform.
- Record the llama.cpp revision, build options, model checksum, and device name.
- Keep raw `bench_run` stdout, stderr, and the exact command for every run.
- Treat warmup runs separately and aggregate only the 10 formal runs.
- Report measured and derived metrics with different field names.
- Do not report old mock data as a formal benchmark result.

Large models, the raw dataset, processed prompt text, and raw result logs should
normally be excluded from Git. Download scripts, checksums, manifests, schemas,
and summary tables are sufficient for reproducibility.

## Speculative decoding only

For paired LC33-LC38 baseline/DFlash tests with an existing `llama-server`, see
[speculative benchmark instructions](docs/speculative-benchmark.md).
The server workflow does not require building `bench_run`.
