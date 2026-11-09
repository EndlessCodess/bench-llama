# LC33-LC38 投机解码测试

`run_speculative.py` 是本分支的投机测试入口，复用现有的
`run_suite.py -> bench_test.py -> bench_server_run.py -> llama-server`。
Python 只调度完整请求、采集结果；草稿生成、目标验证和状态回退由 llama.cpp 实现。
这条路径不调用本仓库的 C++ `bench_run`，也不要求编译它。

## 场景与参数

| 数据集 | baseline | DFlash |
| --- | --- | --- |
| Alpaca | LC33 | LC34 |
| MATH-500 | LC35 | LC36 |
| HumanEval | LC37 | LC38 |

- 每轮按照 LC33、34、35、36、37、38 顺序执行，完成后再进入下一轮。
- 单卡、单请求，模型在每个场景内加载一次，预热和正式请求共用 server。
- `n_ctx=8192`，`batch=512`，`ubatch=512`，CPU 线程数 8，F16 KV。
- DFlash 每轮最多提出 15 个草稿 token；每条请求最多生成 512 个 token，遇到 EOS 提前结束。
- 温度 0、seed 42，关闭 thinking 和 prompt cache。
- 正式运行之前，自动对所有选定场景各跑一条最多输出 64 token 的 smoke。
- smoke 检查有效输出，以及 DFlash 的草稿提出数和接受数；不满足时阻止正式运行。
  `smoke-check.json` 单独记录短输出是否与 baseline 一致，这不等于证明严格无损。

默认每轮 `5+50`：5 次预热、50 次正式请求。仓库内每个数据集有固定的 50 个样本，
默认每轮每个样本正式使用一次。**`5+100` 表示 100 次正式请求，不表示 100 道不同题目**；
要复现历史的两遍样本池，必须显式设置 `--formal-runs 100 --allow-repeated-samples`。
三个轮次会重复同一套样本和顺序，用于评估性能波动；预热不计入正式均值。

目标模型可以是 F16 或 Q8_0；下面的已验证组合是 Qwen3.5-9B Q8_0 target 与
Qwen3.5-9B DFlash F16 draft，llama.cpp 提交 `de3d776fb00cc051a2d7f9c5ba18ecffaa555a66`，
CANN 9.1.0、Ascend 310P3。Q8 是否走融合 W8A16 取决于 llama.cpp 的后端构建，
本项目不实现矩阵乘、GDN、RoPE 或自定义设备 kernel。

## 启动

先加载自己的 CANN 环境，再设置本机文件路径。只需要 Python 3 标准库和已有的
`llama-server` 可执行文件；不依赖 Conda、其他用户目录或某个旧结果目录。

```bash
source /path/to/cann-9.1.0/set_env.sh
export LLAMA_SERVER=/path/to/llama.cpp/build/bin/llama-server
export TARGET_MODEL=/path/to/Qwen3.5-9B-Q8_0.gguf
export DRAFT_MODEL=/path/to/Qwen3.5-9B-DFlash-F16.gguf

python3 run_speculative.py \
  --llama-server "$LLAMA_SERVER" \
  --model "$TARGET_MODEL" --draft-model "$DRAFT_MODEL" \
  --platform cann_310p --device-index 1 \
  --rounds 3 --warmup-runs 5 --formal-runs 50 \
  --output-dir results-local/speculative-NEW
```

`--device-index` 必须显式给出，表示物理卡号。CANN 可见性过滤后，进程内部使用 `CANN0`。
310P 路径会在子进程中打开已验证的 WQMM、GDN decode/prefill/snapshot、卷积权重预转换和 MRoPE 开关。
CPU、CUDA、Metal、910B 的平台选择沿用现有档案；本次整理只验证了 310P 历史推理结果及无设备的调度回归。

只检查文件、prompt 哈希和完整命令，不启动模型：在上面的命令追加 `--dry-run`。
只做 smoke：追加 `--smoke-only`。每次使用一个新的输出目录，避免覆盖历史结果。
只测 HumanEval：追加 `--datasets humaneval`。

要复现之前的三轮 `5+100`，将正式请求参数改为：

```bash
--rounds 3 --warmup-runs 5 --formal-runs 100 --allow-repeated-samples
```

输入样本位于 `data/processed/qwen3.5-9b-speculative-nothink`，包含原始 prompt、题号、哈希和请求顺序。
换目标 tokenizer 时，可使用 `retokenize_speculative_dataset.py --source ... --output ... --model ... --llama-server ...`
重新统计同一批文本的输入 token 数，然后通过 `--processed-dir` 指定新目录。
从原始数据重新采样使用已有的 `prepare_speculative_datasets.py`，其 parquet 输入需要额外安装 `pyarrow`。

## 结果和日志

```text
results-local/speculative-NEW/
  protocol.json              # 本次参数
  sample-plan.json           # 实际题号、顺序、唯一样本数及请求数
  status.json                # 当前阶段、完成轮数及失败原因
  dry-run.log                # prompt 和命令校验
  smoke/                     # 每个场景的一次短请求
  round-01/                  # 第一整轮的 LC33-LC38
    lc-038/
      runs.jsonl             # 该轮该场景完整运行后写入的请求记录
      summary.json
      LC-038/
        plan.tsv
        session.command.txt
        session.stdout.log   # 实时事件、每条生成文本和速度
        session.stderr.log   # llama-server 原始日志
  round-02/
  summary.json               # 每轮结束更新，依据正式原始记录合并
  summary.csv
```

每轮的 `suite-manifest.json` 记录模型 SHA-256、server 位置及实际命令；场景 manifest 还包含版本信息。
`decode_tps` 是逐请求 decode 速度的均值，对应 CLI 的 Generation；`吞吐率(token/s)/Throughput(token/s)`
沿用原表格定义，使用完整请求耗时，二者不能混用。TPOT 加速比使用 baseline 平均 TPOT 除以 DFlash 平均 TPOT。
TTFT 字段沿用当前适配器的服务端 `prompt_ms`；非流式请求不测量客户端收到首个 token 的真实网络延迟。

310P 的显存和 AI Core 数据来自 `npu-smi info` 定时采样，统计的是整卡，其他进程会影响读数。
AI Core 利用率不是 Cube/Vector 指令流水线忙时间。结果、profile、GGUF 和构建目录均被忽略，禁止作为本分支的提交内容。

## 不占用加速卡的回归检查

```bash
python3 -m unittest discover -s tests -p 'test_speculative.py' -v
```

使用本地 HTTP fixture 验证持久 server、多条独立请求、预热排除、轮次顺序、重复样本检查和失败停止。
这些检查验证测试框架，不评估模型精度或实际 NPU 性能。
