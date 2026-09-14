# Qwen3.6-35B-A3B 医疗场景 llama.cpp 单卡推理调优报告

## 1. 结论摘要

本次工作围绕 `Qwen3.6-35B-A3B-int4-mixed-AutoRound` 的 GGUF 版本，在单张 NVIDIA GeForce RTX 5090 上优化 llama.cpp 医疗问答推理性能。优化目标按任务要求设为 `TPOT > TTFT`，即优先降低连续生成阶段的单 token 延迟，再观察首 token 延迟、端到端延迟、显存和输出稳定性。

最终结论如下：

1. 普通推理的推荐配置为 `threads=4`、`threads-batch=16`、Flash Attention 开启、CUDA Graph 开启、后端采样开启、KV Cache 使用 F16。
2. 普通推理优化后，pooled TPOT 从 4.113 ms/token 降至 3.562 ms/token，降低 13.4%；解码吞吐从 243.1 token/s 提升至 280.8 token/s，提升 15.5%。
3. 在普通推理最优配置上继续加入投机解码后，最终最优方案为 DFlash `n_max=4`。
4. 最新 5+200 完整验证中，DFlash `n_max=4` 的 pooled TPOT 为 2.873 ms/token，解码吞吐为 348.1 token/s。相对同轮基线，TPOT 降低 23.1%，吞吐提高 30.0%。
5. DFlash 的代价是 TTFT 和短请求 E2E 增加：中位 TTFT 从 1012 ms 增至 1445 ms，中位 E2E 从 1801 ms 增至 2136 ms。因为医疗数据中包含较多短输出，请求越短，草稿模型初始化和验证开销越难摊薄。
6. 如果只追求最低 TPOT，选择 DFlash `n_max=4`；如果更看重 TTFT、E2E 和显存的均衡，内置 MTP `n_max=3` 更稳妥；如果请求普遍较短或首 token 最重要，则保留普通推理基线。

### 汇报核心总表

| 阶段 / 配置 | GPU | pooled TPOT (ms/token) | 解码吞吐 (token/s) | TTFT | E2E | 显存峰值 (MiB) | 阶段结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 原始普通推理：t=16，tb=16 | 4 | 4.113 | 243.1 | 均值 966.94 ms | 均值 2.132 s | 22483 | 初始基准 |
| 优化普通推理：t=4，tb=16，后端采样 | 4 | 3.562 | 280.8 | 均值 1007.97 ms | 均值 2.017 s | 22531 | TPOT 较原始基准降低 13.4% |
| 最新配对基线 | 7 | 3.736 | 267.7 | 中位数 1012 ms | 中位数 1801 ms | 22531 | 投机解码对照组 |
| MTP n=3 | 7 | 3.277 | 305.2 | 中位数 1355 ms | 中位数 2088 ms | 23541 | TPOT 较配对基线降低 12.3%，均衡候选 |
| DFlash n=6 | 7 | 2.919 | 342.6 | 中位数 1687 ms | 中位数 2313 ms | 25061 | 5+200 后不及 n=4 |
| **DFlash n=4** | **7** | **2.873** | **348.1** | **中位数 1445 ms** | **中位数 2136 ms** | **24943** | **TPOT 较配对基线降低 23.1%，最终最优** |

表中的普通推理 GPU 4 数据使用均值，最新投机验证 GPU 7 数据使用中位数。两组来自不同测试轮次，不能直接用 TTFT/E2E 做严格横向计算；DFlash 和 MTP 的加速结论均以 GPU 7 同轮配对基线为准。

## 2. 测试对象与环境

| 项目 | 配置 |
| --- | --- |
| 主模型 | `Qwen3.6-35B-A3B-Q4_K_M.gguf` |
| 主模型来源 | Qwen3.6-35B-A3B int4 mixed AutoRound 转换的 GGUF |
| 主模型文件大小 | 21,711,660,416 bytes，约 20.22 GiB |
| DFlash 草稿模型 | `Qwen3.6-35B-A3B-DFlash-BF16.gguf` |
| MTP | 主模型内置 MTP |
| 推理引擎 | llama.cpp `0.1.0-dev`，build 10460，commit `373336672` |
| llama.cpp 可执行文件 | `build-cuda13/bin/llama-server` |
| 硬件 | 单张 NVIDIA GeForce RTX 5090 |
| 并发 | 1 |
| 上下文长度 | 65536 |
| batch / ubatch | 2048 / 512 |
| GPU offload | `--gpu-layers 99`，主模型和草稿模型均尽量全量上卡 |
| 思考模式 | 关闭，`--reasoning off` |
| 停止条件 | 遇到 EOS 正常结束，不强制生成固定长度 |
| 最大输出 | 18000 token，仅作为安全上限 |
| 最终验证采样 | temperature=0.1，top-p=0.95，seed=42 |
| 正式测试规模 | 每个配置 5 条预热 + 200 条正式请求，1 轮 |

普通推理基础开关与 CPU 参数调优主要在物理 GPU 4 上完成，投机解码完整调优在物理 GPU 7 上完成。不同 GPU 编号代表同型号 5090，但不同测试轮次仍可能存在温度、频率和系统负载波动，因此跨轮次结果用于观察趋势，最终加速比只使用同一阶段的配对基线。

## 3. 医疗数据集准备

原始数据位于：

```text
/data2/sfx/test/test_qwen3.6-35B-A3B-int4-vllm-dflash/datasets/langfuse_traces
```

预处理脚本为 `prepare_medical_traces.py`。处理流程为：

1. 扫描 Langfuse trace JSON 文件，只保留 `GENERATION` 类型记录。
2. 只保留完整的 `[system, user]` 两段消息，确保输入能直接交给 chat completion 接口。
3. 按消息内容 SHA-256 去重，共得到 3054 条可用唯一请求。
4. 按业务 stage 和历史输入长度区间分层。长度区间为 2K、8K、16K、32K 和大于 32K。
5. 使用固定随机种子 `20260907` 按各层占比抽取 205 条，其中前 5 条用于预热，后 200 条用于正式测试。

最终数据文件为：

```text
data/medical-langfuse/qwen3.6-205-v1/samples.jsonl
```

数据集摘要：

| 项目 | 数值 |
| --- | ---: |
| 可用唯一请求 | 3054 |
| 最终样本 | 205 |
| 预热 / 正式 | 5 / 200 |
| 涉及原始文件 | 123 |
| 正式样本平均输入 | 5400.9 token |
| 正式样本输入中位数 | 4345.5 token |
| 正式样本输入 P95 | 13482 token |
| 基准平均输出 | 284.3 token |
| 基准输出中位数 | 123.5 token |
| 基准输出 P95 | 955 token |

历史 trace 中的 token 数只用于分层抽样，正式指标中的输入和输出 token 数以当前 GGUF 模型及 llama.cpp 服务端统计为准。

## 4. 指标口径

| 指标 | 定义与用途 |
| --- | --- |
| TTFT | 客户端发出请求到收到首个非空内容块的时间，反映首 token 响应速度 |
| pooled TPOT | 全部正式请求的服务端 decode 总时间除以全部 decode token 间隔，是本次主要优化指标 |
| 解码吞吐 | `1000 / pooled TPOT`，单位 token/s |
| E2E | 客户端从请求开始到接收完整响应的时间，包含 TTFT、生成、HTTP 和客户端处理 |
| 接受率 | 投机解码中被主模型接受的草稿 token 数占草稿 token 总数的比例 |
| 显存峰值 | 以 200 ms 间隔通过 NVML 采集的整卡已用显存峰值 |
| GPU 利用率 | NVML 整卡利用率采样，不等价于 CUDA Core 或 Tensor Core 的实际计算效率 |

选择 pooled TPOT 而不是逐请求 TPOT 均值，是为了避免只生成少量 token 的极短请求对平均值产生过大影响。接受率只作为诊断指标，不是最终优化目标；高接受率并不必然代表低 TPOT，因为过于保守的草稿策略可能减少每轮有效草稿 token 数。

## 5. 第一阶段：基础功能开关验证

首先使用相同的 5+200 请求检查 KV Cache、Flash Attention、轮询和 CUDA Graph。该阶段使用物理 GPU 4。

| 配置 | pooled TPOT (ms/token) | 相对基准 | TTFT 均值 (ms) | E2E 均值 (s) | 显存峰值 (MiB) | GPU 利用率均值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 基准 | 4.113 | 1.000x | 966.94 | 2.132 | 22483 | 78.9% |
| KV Q8 | 4.096 | 1.004x | 943.32 | 2.116 | 21995 | 80.6% |
| KV Q4 | 4.206 | 0.978x | 944.13 | 2.160 | 21675 | 79.3% |
| Flash Attention 关闭 | 4.253 | 0.967x | 1068.48 | 2.270 | 24973 | 78.3% |
| poll=0 | 4.090 | 1.006x | 964.61 | 2.124 | 22483 | 76.8% |
| CUDA Graph 关闭 | 5.893 | 0.698x | 948.14 | 2.618 | 22479 | 73.9% |

这一阶段得到的结论：

- CUDA Graph 必须保持开启。关闭后 TPOT 增加约 43.3%，是最明显的负优化。
- Flash Attention 应保持开启。关闭后 TPOT、TTFT 和显存均退化。
- KV Q8 可节省约 488 MiB 显存，性能接近 F16，但会改变部分输出，因此不作为追求最高性能和输出稳定性的默认配置。
- KV Q4 虽可节省约 808 MiB，但 TPOT 退化约 2.3%，且量化 KV 需要额外做医疗质量验证。
- `poll=0` 与基准差异约 0.6%，不足以证明有稳定收益，保留默认 `poll=50`。

六组配置均完成 200 条请求，无失败、无长度截断。

## 6. 第二阶段：普通推理参数调优

基础开关确定后，先使用 5+30 的小样本筛选 CPU 线程、后端采样、连续批处理和 ubatch，再对候选方案执行 5+200 完整验证。

主要搜索内容包括：

- `threads`：4、8、16
- `threads-batch`：4、8、16
- `--backend-sampling`：开启或关闭
- continuous batching：开启或关闭
- `ubatch-size`：512、1024、2048
- KV Cache：F16 与 Q8
- `poll`、CUDA Graph 优化和 CUDA PDL 开关

最终 5+200 结果如下：

| 配置 | pooled TPOT (ms/token) | 解码吞吐 (token/s) | TTFT 均值 (ms) | E2E 均值 (s) | 显存峰值 (MiB) | GPU 利用率均值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原始基准：t=16，tb=16 | 4.113 | 243.1 | 966.94 | 2.132 | 22483 | 78.9% |
| t=8，tb=8，后端采样，关闭连续批处理 | 3.620 | 276.2 | 1243.86 | 2.270 | 22527 | 74.4% |
| t=4，tb=4，后端采样 | 3.609 | 277.1 | 1176.78 | 2.200 | 22527 | 79.2% |
| **t=4，tb=16，后端采样** | **3.562** | **280.8** | **1007.97** | **2.017** | **22531** | **82.9%** |

普通推理最优配置相对原始基准：

- pooled TPOT 降低 13.4%。
- 解码吞吐提升 15.5%。
- TTFT 增加 4.2%。
- E2E 均值降低 5.4%。
- 显存只增加约 48 MiB。
- 200/200 条生成文本与原始基准逐字一致。

`threads=4` 适合生成阶段，`threads-batch=16` 则保留了长 prompt 处理所需的 CPU 并行度。`ubatch=1024/2048` 在短测中出现输出不完全一致，且 2048 有性能退化，因此保留 512。关闭连续批处理在单并发下没有超过最终方案。`GGML_CUDA_GRAPH_OPT=1` 未体现收益，关闭 CUDA PDL 则有明显退化。

## 7. 第三阶段：投机解码初步验证

在普通推理最优参数上加入 MTP 和 DFlash。第一轮使用 5+30 请求筛选草稿长度，结果如下：

| 候选 | pooled TPOT (ms/token) | 接受率 | 截断 |
| --- | ---: | ---: | ---: |
| MTP n=2 | 2.906 | 96.42% | 1/30 |
| MTP n=3 | 2.631 | 85.95% | 0/30 |
| DFlash n=4 | 2.636 | 72.36% | 0/30 |
| DFlash n=8 | 2.586 | 75.82% | 1/30 |
| DFlash n=15 | 3.502 | 32.41% | 0/30 |

这一轮表明 MTP 和 DFlash 都有降低 TPOT 的潜力，但也暴露出两个问题：

- 30 条样本太少，DFlash n=8 等结果容易受输出长度和偶发超长生成影响。
- 接受率与 TPOT 不是单调关系，不能只根据接受率或单次最小 TPOT 选参数。

因此后续增加到 50 条分层样本，并扩大 DFlash `n_max` 搜索范围。

## 8. 第四阶段：投机长度精细搜索

精细搜索使用 5 条预热 + 50 条正式请求、temperature=0 的固定贪心解码，并根据先前真实完成长度进行分层抽样，以减少随机输出和样本结构变化的干扰。测试在物理 GPU 7 上完成。

| 配置 | pooled TPOT (ms/token) | 解码吞吐 (token/s) | 接受率 | 显存峰值 (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 基线 | 3.824 | 261.5 | - | 22527 |
| MTP n=1 | 3.787 | 264.1 | 96.69% | 23381 |
| MTP n=2 | 3.386 | 295.3 | 89.73% | 23445 |
| MTP n=3 | 3.152 | 317.2 | 84.86% | 23541 |
| DFlash n=2 | 3.123 | 320.2 | 90.68% | 24803 |
| DFlash n=3 | 2.952 | 338.8 | 76.44% | 24865 |
| DFlash n=4 | 2.620 | 381.6 | 76.25% | 24923 |
| DFlash n=5 | 2.753 | 363.3 | 63.64% | 24985 |
| **DFlash n=6** | **2.571** | **389.0** | 63.31% | 25053 |
| DFlash n=7 | 2.640 | 378.9 | 53.85% | 25131 |
| DFlash n=8 | 2.867 | 348.8 | 67.62% | 25211 |
| DFlash n=9 | 2.795 | 357.8 | 63.24% | 25287 |
| DFlash n=10 | 3.263 | 306.5 | 48.18% | 25331 |
| DFlash n=12 | 3.283 | 304.6 | 43.25% | 25463 |
| DFlash n=15 | 2.746 | 364.1 | 46.68% | 25657 |

50 条筛选中 DFlash n=6 的 TPOT 最低，DFlash n=4 次之。草稿长度继续增加并未持续提升性能：更长的草稿会增加草稿模型计算量和主模型验证成本，接受率下降后还会产生更多回滚，因而存在中间最优点。

## 9. 第五阶段：DFlash 局部参数搜索

以筛选阶段暂时最优的 DFlash n=6 为中心，每次只改变一个变量。下表仍为 5+50 贪心筛选结果。

| n_max=6 的变体 | pooled TPOT (ms/token) | 解码吞吐 (token/s) | 接受率 | 结论 |
| --- | ---: | ---: | ---: | --- |
| 默认：p_min=0，n_min=0 | 2.571 | 389.0 | 63.31% | 保留 |
| p_min=0.05 | 2.621 | 381.5 | 63.31% | 变慢 |
| p_min=0.10 | 2.574 | 388.6 | 63.31% | 无稳定收益 |
| p_min=0.20 | 2.960 | 337.8 | 69.97% | 接受率升高但 TPOT 退化 |
| p_min=0.30 | 4.093 | 244.3 | 71.54% | 明显退化 |
| n_min=1 | 2.728 | 366.6 | 63.31% | 变慢 |
| 草稿后端采样关闭 | 3.280 | 304.9 | 63.31% | 明显退化 |
| 草稿线程 2 / batch 线程 8 | 2.656 | 376.6 | 63.31% | 变慢 |
| 草稿线程 8 / batch 线程 16 | 2.653 | 376.9 | 63.31% | 变慢 |

局部搜索确定：

- `spec-draft-p-min=0` 最好。
- `spec-draft-n-min=0` 最好。
- 主模型和草稿模型都应开启后端采样。
- 不显式指定草稿模型线程参数，使用 llama-server 默认值更快。
- 提高 p_min 虽可提高统计接受率，但减少了有用草稿，实际 TPOT 反而更差。

## 10. 第六阶段：5+200 最终验证

最终恢复实际测试采样口径 temperature=0.1，关闭思考，对基线、MTP n=3、DFlash n=6 和 DFlash n=4 分别执行 5 条预热 + 200 条正式请求。所有配置均无请求失败、无 18000 token 长度截断。

| 配置 | pooled TPOT (ms/token) | 解码吞吐 (token/s) | 相对基线 | TTFT 中位数 (ms) | E2E 中位数 (ms) | 接受率 | 显存峰值 (MiB) | 截断 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 基线 | 3.736 | 267.7 | 1.00x | 1012 | 1801 | - | 22531 | 0/200 |
| MTP n=3 | 3.277 | 305.2 | 1.14x | 1355 | 2088 | 80.89% | 23541 | 0/200 |
| DFlash n=6 | 2.919 | 342.6 | 1.28x | 1687 | 2313 | 55.22% | 25061 | 0/200 |
| **DFlash n=4** | **2.873** | **348.1** | **1.30x** | **1445** | **2136** | **66.16%** | **24943** | **0/200** |

最终完整验证推翻了 50 条筛选中 n=6 略优的暂时结论。DFlash n=4 在 200 条样本上同时获得更低 TPOT、更高吞吐、更低 TTFT、更高接受率和更低显存，因此确定为最终最优参数。

DFlash n=4 相对本轮基线：

- pooled TPOT 从 3.736 降至 2.873 ms/token，降低 23.1%。
- 解码吞吐从 267.7 提升至 348.1 token/s，提高 30.0%。
- TTFT 中位数从 1012 增至 1445 ms，增加 42.7%。
- E2E 中位数从 1801 增至 2136 ms，增加 18.6%。
- 显存峰值从 22531 增至 24943 MiB，额外占用约 2.36 GiB。

MTP n=3 相对 DFlash n=4 的 TPOT 更高，但只额外占用约 1.0 GiB 显存，TTFT 也更低，适合作为均衡方案。

## 11. 最终推荐参数

### 11.1 TPOT 优先：DFlash n=4

```bash
CUDA_VISIBLE_DEVICES=7 \
/data1/xuyang/projects/llama.cpp/build-cuda13/bin/llama-server \
  --model /data1/xuyang/models/Qwen3.6-35B-A3B-AutoRound-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --spec-draft-model /data1/xuyang/models/Qwen3.6-35B-A3B-DFlash-GGUF/Qwen3.6-35B-A3B-DFlash-BF16.gguf \
  --gpu-layers 99 \
  --spec-draft-ngl 99 \
  --ctx-size 65536 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --parallel 1 \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --flash-attn on \
  --poll 50 \
  --threads 4 \
  --threads-batch 16 \
  --threads-http 2 \
  --backend-sampling \
  --spec-type draft-dflash \
  --spec-draft-n-max 4 \
  --spec-draft-n-min 0 \
  --spec-draft-p-min 0 \
  --spec-draft-backend-sampling \
  --fit off \
  --no-context-shift \
  --jinja \
  --reasoning off
```

CUDA Graph 和 CUDA PDL 保持默认开启，不设置 `GGML_CUDA_DISABLE_GRAPHS=1`，也不设置 `GGML_CUDA_PDL=0`。

### 11.2 均衡方案：MTP n=3

在普通推理最优参数上使用：

```text
--spec-type mtp
--spec-draft-n-max 3
--spec-draft-n-min 0
--spec-draft-p-min 0
```

MTP 不需要额外的外部草稿模型，显存开销和 TTFT 增量均低于 DFlash。

### 11.3 TTFT 优先：关闭投机解码

如果线上请求输出普遍较短，或者首 token 延迟比解码吞吐更重要，则使用普通推理最优配置：

```text
--threads 4
--threads-batch 16
--backend-sampling
--spec-type none
```

## 12. 如何理解最终结果

本次数据清楚体现了投机解码的适用边界：

- DFlash 确实显著降低了生成阶段 TPOT，说明草稿模型对该医疗数据具有可利用的预测能力。
- 投机解码不是无条件降低整条请求耗时。它需要额外加载草稿模型，并在每轮执行草稿、验证、接受或回滚，因此 TTFT 和短请求 E2E 可能变差。
- 医疗数据的输出中位数只有约 124 token，许多请求不足以充分摊薄投机开销。因此即使 DFlash TPOT 最优，整体 E2E 中位数仍慢于基线。
- DFlash n=4 的接受率为 66.16%，低于 MTP n=3 的 80.89%，但 TPOT 更低。这再次说明接受率不能单独决定性能，草稿长度、草稿计算成本和单轮接受 token 数需要共同考虑。
- 30 条和 50 条小样本适合快速淘汰明显较差参数，但不足以确定 n=4 与 n=6 这类接近候选，最终必须以 200 条完整结果为准。

## 13. 限制与后续工作

1. 本轮重点是性能调优，没有使用医疗准确率、事实性或专家评分作为质量指标。
2. temperature=0.1 时，投机路径会改变随机数消耗顺序，输出与基线不一定逐字一致；逐字一致率不能代替医疗质量评测。
3. `--backend-sampling` 在当前 llama.cpp 版本中仍为 experimental，升级 llama.cpp 后应重新回归。
4. GPU 利用率为 NVML 整卡采样，不能解释到 CUDA Core、Tensor Core 或内存带宽瓶颈。若需要硬件层结论，应补充 Nsight Systems/Compute 分析。
5. 当前结论针对单卡、单并发和自然输出长度。多并发、固定长输出、不同上下文长度或其他 GPU/NPU 平台需要单独验证。
6. 后续医疗质量评测应至少对比基线、MTP n=3、DFlash n=4 三组输出，并报告准确率、拒答率、事实错误和超长生成比例。
7. 如果部署流量能够按预期输出长度分流，可对短输出请求关闭投机，对长输出请求启用 DFlash，以同时控制 TTFT 和 TPOT。

## 14. 原始数据与复现文件

| 内容 | 路径 |
| --- | --- |
| 医疗数据集 | `data/medical-langfuse/qwen3.6-205-v1/` |
| 数据预处理脚本 | `prepare_medical_traces.py` |
| 测试驱动 | `run_medical_suite.py` |
| 测试方法说明 | `MEDICAL_BENCH.md` |
| 六项基础开关 5+200 | `results/medical-qwen36-6cases-5w200-20260907/` |
| 普通推理第一轮筛选 | `results/medical-qwen36-tuning-stage1-20260907/` |
| 普通推理第二轮筛选 | `results/medical-qwen36-tuning-stage2-20260907/` |
| 普通推理补充筛选 | `results/medical-qwen36-tuning-stage2b-20260907/` |
| ubatch 筛选 | `results/medical-qwen36-tuning-stage3-20260907/` |
| 普通推理完整候选 | `results/medical-qwen36-tuning-final-20260907/` |
| 普通推理最终最优 | `results/medical-qwen36-tuning-final-batch16-20260907/` |
| 初步投机解码 30 条筛选 | `results/medical-qwen36-spec-screen30-gpu7-20260908/` |
| 初步投机解码 5+200 | `results/medical-qwen36-spec-final-gpu7-20260908/` |
| 精细草稿长度 50 条筛选 | `results/medical-spec-tuning-stage1-gpu7-20260908/` |
| DFlash 局部参数筛选 | `results/medical-spec-tuning-stage2-*gpu7-20260909/` |
| 最终基线、MTP n=3、DFlash n=6 | `results/medical-spec-tuning-final-gpu7-20260909/` |
| 最终 DFlash n=4 确认 | `results/medical-spec-tuning-final-n4-gpu7-20260909/` |

每个正式结果目录中：

- `manifest.json` 保存模型、数据集哈希、llama.cpp 版本、启动命令和环境变量。
- `summary.json` 保存配置级汇总指标。
- `requests.jsonl` 保存逐请求输入输出 token、TTFT、TPOT、E2E 和投机接受统计。
- `server.log` 保存 llama-server 启动信息与服务端运行日志。
