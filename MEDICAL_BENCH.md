# Medical trace benchmark

`run_medical_suite.py` reuses the server lifecycle and command builder from `bench_server_run.py`. It sends complete system/user messages from the local Langfuse sample file.

Each configuration uses the same 5 warmup and 200 measured requests, in the same order, for one round. This is a natural-length medical workload, not the historical fixed-length LC matrix. The six configurations are baseline, Q8 KV, Q4 KV, Flash Attention off, polling off, and CUDA Graph off. All use the same AutoRound Q4_K_M weights. Speculative decoding and F16/Q8 weight comparisons are excluded.

Defaults: one GPU, one server slot, all model layers offloaded, context 65536, batch 2048, ubatch 512, 16 generation/batch CPU threads, temperature 0.1, top-p 0.95, seed 42, thinking disabled, no prompt cache reuse, and normal EOS stopping with an 18000-token ceiling. The server receives `--fit off` and `--no-context-shift` to prevent silent configuration changes or input truncation.

```sh
python3 run_medical_suite.py \
  --llama-server /path/to/llama-server \
  --model /path/to/model.gguf \
  --dataset data/medical-langfuse/qwen3.6-205-v1/samples.jsonl \
  --output-dir results/medical-run \
  --device 4
```

Use `--smoke` to check one longest input with only 8 output tokens. Smoke results are not formal measurements. Use `--cases baseline` to select individual configurations. The NVML monitor currently requires an NVIDIA GPU.

Each case writes `server.log`, `requests.jsonl`, and `summary.json`. The suite manifest records commands, environment overrides, binary version, dataset hash, and model path/size. Failed cases are explicitly marked; baseline failure stops the suite. Existing output directories are never overwritten.

Client TTFT measures request submission to the first nonempty content chunk. Client TPOT is the time between the first and last content chunks divided by completion tokens minus one. Chunk arrivals are recorded but are not labeled as token-level ITL. The server's `predicted_per_token_ms` is stored separately as `decode_ms_per_token`; use this field for server decode performance. Client E2E includes HTTP and response handling. NVML samples every 200 ms and measures the entire device, not individual CUDA core occupancy.

Historical token counts are sampling metadata. Actual token counts come from the current server. Output text is retained for inspection, not scored against a medical gold standard. Interpret performance together with output length and length-limit frequency.

Raw medical messages, source identifiers, and generated responses stay in ignored local data/results directories.
