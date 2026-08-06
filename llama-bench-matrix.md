# Portable llama.cpp benchmark matrix

This local test driver runs the llama.cpp cases from `llama-bench-matrix.csv`.
It uses only the Python standard library and keeps backend-specific handling at
the process boundary.

The driver runs three llama-bench tests for each case:

- PP with the exact synthetic prompt token count.
- TG with the exact generated token count.
- PP+TG for combined execution time.

The first `warmup_runs` samples are discarded. The next `formal_runs` samples
are used for the summary. All samples execute in the same llama-bench process.

`llama-bench` excludes tokenization and sampling. Therefore `ttft_est_ms` and
`itl_est_ms` are estimates based on model execution, not client-observed
end-to-end latency. Raw JSON is retained for every case.

## RTX 5090

Run one baseline case on physical GPU 7:

```bash
python3 scripts/llama-bench-matrix.py \
  --bench ./build-cuda13/bin/llama-bench \
  --platform cuda \
  --device-index 7 \
  --model baseline=/path/to/Qwen3.5-9B-F16.gguf \
  --case LC-001
```

Run every case after all model variants are available:

```bash
python3 scripts/llama-bench-matrix.py \
  --bench ./build-cuda13/bin/llama-bench \
  --platform cuda \
  --device-index 7 \
  --model baseline=/path/to/Qwen3.5-9B-F16.gguf \
  --model q8_0=/path/to/Qwen3.5-9B-Q8_0.gguf \
  --model q4_k_m=/path/to/Qwen3.5-9B-Q4_K_M.gguf
```

CUDA visibility is restricted to the selected physical device. Inside
llama.cpp that device is named `CUDA0`. Device memory is sampled through NVML,
so the script does not require the `nvidia-smi` executable.

Use `--dry-run` to inspect commands. Use `--case 'LC-00*'` or repeat `--case`
to select multiple cases.

## CANN and Metal

The same matrix can be used with:

```bash
--platform cann --device-index 0 --backend-device CANN0
--platform metal --backend-device Metal
```

Process RSS is always sampled. CUDA device memory is built in. For another
accelerator, `--memory-command` may name a command that prints current used
device memory in MiB. The token `{device}` is replaced with `--device-index`.

## Results

Each run creates a timestamped directory under `bench-results` containing:

- `manifest.json`: host, backend, model mappings, and selected cases.
- `summary.csv`: normalized metrics suitable for spreadsheet import.
- `results.jsonl`: complete normalized records and formal samples.
- `<case>.stdout.json`: unmodified llama-bench output.
- `<case>.stderr.log`: backend and fallback logs.
- `<case>.command.txt`: exact reproduction command.

