# Qwen3.6 medical speculative tuning

Goal: minimize TPOT first, then compare TTFT. Use one GPU and one concurrent request.

## Fixed configuration

- Target: `Qwen3.6-35B-A3B-Q4_K_M.gguf`
- Draft: `Qwen3.6-35B-A3B-DFlash-BF16.gguf`
- Context: 65536
- Batch / micro-batch: 2048 / 512
- Target threads / batch threads: 4 / 16
- Flash Attention and CUDA Graph: enabled
- Target backend sampling: enabled
- Thinking: disabled

## Stage 1: draft length screening

Use greedy decoding and 50 stratified requests. The selection is based on the actual completion lengths from the prior baseline, not the source trace output lengths.

```bash
cd /data1/xuyang/projects/bench_test_llama

python3 run_medical_suite.py \
  --llama-server /data1/xuyang/projects/llama.cpp/build-cuda13/bin/llama-server \
  --model /data1/xuyang/models/Qwen3.6-35B-A3B-AutoRound-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --draft-model /data1/xuyang/models/Qwen3.6-35B-A3B-DFlash-GGUF/Qwen3.6-35B-A3B-DFlash-BF16.gguf \
  --dataset data/medical-langfuse/qwen3.6-205-v1/samples.jsonl \
  --selection-reference results/medical-qwen36-spec-final-gpu7-20260908/threads-4-batch16-backend-sampling/requests.jsonl \
  --output-dir results/medical-spec-tuning-stage1-gpu7 \
  --device 7 \
  --temperature 0 \
  --measure-limit 50 \
  --measure-selection stratified \
  --backend-sampling on \
  --cases \
    baseline \
    threads-4-batch16-backend-sampling-mtp-n1 \
    threads-4-batch16-backend-sampling-mtp-n2 \
    threads-4-batch16-backend-sampling-mtp-n3 \
    threads-4-batch16-backend-sampling-dflash-n2 \
    threads-4-batch16-backend-sampling-dflash-n3 \
    threads-4-batch16-backend-sampling-dflash-n4 \
    threads-4-batch16-backend-sampling-dflash-n5 \
    threads-4-batch16-backend-sampling-dflash-n6 \
    threads-4-batch16-backend-sampling-dflash-n7 \
    threads-4-batch16-backend-sampling-dflash-n8 \
    threads-4-batch16-backend-sampling-dflash-n9 \
    threads-4-batch16-backend-sampling-dflash-n10 \
    threads-4-batch16-backend-sampling-dflash-n12 \
    threads-4-batch16-backend-sampling-dflash-n15
```

Keep the best MTP setting and the two best neighboring DFlash settings. Reject a setting if it improves pooled TPOT only by relying on truncated requests.

## Stage 2: local search

For each retained draft length, test one variable at a time on the same 50 requests:

1. `--draft-p-min`: 0, 0.05, 0.10, 0.20, 0.30
2. `--draft-n-min`: 0, 1
3. `--draft-backend-sampling`: on, off
4. Draft threads / batch threads: 2/8, 4/16, 8/16

After each step, retain the lowest pooled TPOT setting and use it as the base for the next step. Compare acceptance rate, truncation count, TTFT, E2E, and peak memory as guardrails. Do not select by acceptance rate alone.

## Stage 3: full validation

Run baseline, best MTP, and best DFlash with temperature 0.1 on all 5 warmup + 200 measured requests. The final report must include pooled TPOT, speedup over the paired baseline, median TTFT, median E2E, acceptance rate, truncation count, and peak GPU memory.

