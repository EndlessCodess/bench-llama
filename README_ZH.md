# bench_test_llama

[English](README.md)

`bench_test_llama` 是一个独立的 llama.cpp 后端测试项目，用于比较不同硬件上的推理性能。项目与 llama.cpp 源码仓库分离，避免测试代码、数据集、运行清单和测试结果污染上游仓库。

当前计划支持的平台包括：

- NVIDIA CUDA，首个测试设备为 RTX 5090
- 华为昇腾 CANN，包括 310P 和 910B
- Apple Metal

正式端到端测试使用 `llama-cli`。官方 `llama-bench` 仅作为可选的算子与吞吐微基准工具，不作为正式测试入口。

## 当前状态

已完成：

- 下载并校验 LongBench-v2 官方数据集
- 实现可复现的数据预处理脚本
- 生成固定的 2K、8K、16K 和 28K prompt 池
- 生成固定的 5 次预热和 10 次正式测试运行清单
- 实现基于 `llama-cli` 的正式基线脚本（单进程会话模式：同一上下文窗口累积 15 轮，KV 缓存跨轮驻留复用）
- 实现 CUDA、CANN、Metal 和 CPU 的设备选择
- 实现逐次日志、内存监控和统一结果汇总
- 输出双语（中文/英文）五指标汇总表：TTFT、TPOT、ITL、E2E、峰值显存
- 完成 Qwen3.5-9B F16、Q8_0 和 Q4_K_M 模型转换与校验
- 完成 LC-001 至 LC-028 全部 28 个用例的套件配置，支持按 v6.1 表格类别分块（`--group 0~4`）运行

正在进行：

- RTX 5090 正式基线测试
- 310P、910B 和 Metal 实机验证

`results/20260805-120212-cli-mock-cuda-gpu1-LC-001` 是早期流程验证结果，使用的是 Qwen3-0.6B，不属于 Qwen3.5-9B 正式测试数据。

## 公平基线

第一阶段公平基线参数如下：

| 参数 | 配置 |
| --- | --- |
| 模型 | Qwen3.5-9B BF16 GGUF |
| 输入长度 | 2048、8192、16384、28672 tokens |
| 输出长度 | 64、256、256、512 tokens |
| 上下文窗口 | 262144 tokens |
| 并发数 | 1 |
| 并行序列数 | 1 |
| GPU 层数 | 99 |
| Batch size | 2048 |
| Micro-batch size | 512 |
| KV Cache | K 为 F16，V 为 F16 |
| Flash Attention | auto |
| Poll | 50 |
| Temperature | 0 |
| Prefix cache 命中率 | 0 |
| 预热次数 | 每种输入长度 5 次 |
| 正式次数 | 每种输入长度 10 次 |

所有平台必须使用相同的模型、prompt 文件、运行顺序和输出长度。不同平台之间只能调整后端与设备相关参数。

## 项目结构

```text
bench_test_llama/
|-- README.md
|-- README_ZH.md
|-- bench_test.py
|-- run_suite.py
|-- prepare_dataset.py
|-- configs/
|   |-- fair-baseline.json
|   |-- lc-001-028-suite.json
|   |-- lc-005-028-suite.json
|   |-- lc-005-008-weight-q8-0.json
|   |-- lc-009-012-weight-q4-k-m.json
|   |-- lc-013-016-kv-q8-0.json
|   |-- lc-017-020-kv-q4-0.json
|   |-- lc-021-024-flash-attn-off.json
|   |-- lc-025-028-poll-0.json
|   |-- model-recipes.json
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

`llama-bench-matrix.*` 是早期基于 `llama-bench` 的原型，不会作为正式 `llama-cli` 测试脚本使用。

测试场景与硬件适配均位于配置文件中：

- `configs/fair-baseline.json`：用例、输入输出长度和公平基线公共参数
- `configs/platforms.json`：设备可见性环境变量、llama.cpp 后端设备名和显存监控方式

`bench_test.py` 本身负责读取固定清单、启动 `llama-cli`、采集指标和输出统一结果，不写死某张 RTX 5090 或某个物理卡号。增加新的后端时优先增加平台 profile，不修改测试主流程。

## 环境要求

- Python 3.8 或更高版本
- 本地 llama.cpp 构建，且包含 `llama-server` 和 `llama-cli`
- Qwen3.5-9B GGUF 模型
- 数据预处理不依赖第三方 Python 包

llama.cpp 和模型属于外部依赖，通过命令行参数传入路径，不会复制进本项目。

## 下载 LongBench-v2

LongBench-v2 官方数据集包含 503 条样本，文件大小约为 444 MiB。可以通过 `HF_ENDPOINT` 选择国内 Hugging Face 镜像：

```bash
cd /path/to/bench_test_llama

mkdir -p data/longbench-v2
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

wget -c \
  "${HF_ENDPOINT}/datasets/zai-org/LongBench-v2/resolve/main/data.json?download=true" \
  -O data/longbench-v2/data.json
```

校验原始文件：

```bash
echo "15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2  data/longbench-v2/data.json" \
  | sha256sum -c -
```

预处理过程中不会修改原始数据集。

## 生成 prompt 池

设置本地 llama.cpp 和 GGUF 模型路径：

```bash
cd /path/to/bench_test_llama

LLAMA_CPP_ROOT=/path/to/llama.cpp
MODEL=/path/to/Qwen3.5-9B-BF16.gguf

./prepare_dataset.py \
  --model "$MODEL" \
  --llama-server "$LLAMA_CPP_ROOT/build/bin/llama-server"
```

如果需要覆盖已有的处理结果，添加 `--force`：

```bash
./prepare_dataset.py \
  --model "$MODEL" \
  --llama-server "$LLAMA_CPP_ROOT/build/bin/llama-server" \
  --force
```

默认参数：

```text
目标长度       = 2048,8192,16384,28672
来源池大小     = 30 条
随机种子       = 20260805
预热次数       = 5
正式测试次数   = 10
抽样方式       = 有放回抽样
```

运行 `./prepare_dataset.py --help` 可以查看全部选项。

## 预处理方法

每条 LongBench-v2 数据包含长 `context`、一个问题、四个选项和一个标准答案。预处理流程如下：

1. 从六个领域中均衡选取 30 条来源样本。
2. 完整保留问题和四个选项。
3. 只截取来源样本的 `context`。
4. 通过 llama.cpp 应用 Qwen chat template。
5. 使用 llama.cpp 原生 tokenizer 统计最终 token 数。
6. 搜索不超过目标 token 数的最长 context 前缀。
7. 保存 prompt、实际 token 数、来源 ID 和 SHA-256。

标准答案只写入 metadata，不会放进模型输入。预处理不会通过重复 `hello` 或其他无意义文本强行补齐长度。

预处理脚本会临时启动一个本地 `llama-server`，调用 `/apply-template` 和 `/tokenize` 接口。由于预处理不执行模型前向，脚本启动服务时会隐藏加速设备。模型只加载一次，避免数百次 tokenizer 调用重复承担 GGUF 启动开销。

## 生成结果

当前数据池共包含 120 个 prompt 文件：

| 目标长度 | 文件数 | 实际 token 数 |
| ---: | ---: | --- |
| 2048 | 30 | 27 条为 2048，3 条为 2047 |
| 8192 | 30 | 28 条为 8192，2 条为 8191 |
| 16384 | 30 | 27 条为 16384，3 条为 16383 |
| 28672 | 30 | 30 条为 28672 |

少 1 token 的原因是 Unicode 字符边界有时会让 BPE token 数一次变化超过 1。每条数据的真实 token 数都记录在 `metadata.jsonl` 中，当前最大差距只有 1 token。

重要生成文件：

- `prompts/<target>/*.txt`：通过 stdin 按会话轮次输入模型的原始用户 prompt
- `metadata.jsonl`：每条 prompt 的来源、长度、哈希与标准答案信息
- `source-pool.json`：固定且领域均衡的来源池
- `run-manifest.json`：所有长度和平台共用的固定随机运行顺序
- `preparation.json`：原始数据校验值、tokenizer 模型和预处理参数

四种目标长度使用相同的来源 ID 和运行顺序，从而控制文本内容差异，便于分析上下文长度变化带来的性能影响。

## 运行正式基线

### 关于 LC-003 长度

v6.1 表格存在一处不一致：llama.cpp 的 `LC-003` 行写的是 24576 tokens，但表格的总体规则、SGLang/vLLM 对应第三档以及此前确认的方案使用 16384 tokens。当前 prompt 池和 `configs/fair-baseline.json` 保留已经确认的 16384 版本：

```text
LC-001 = 2048  输入 / 64  输出
LC-002 = 8192  输入 / 256 输出
LC-003 = 16384 输入 / 256 输出
LC-004 = 28672 输入 / 512 输出
```

因此当前方案是“我们确认的 16K 公平基线”，并非与表格中 llama.cpp 的 `LC-003=24576` 单元格逐字一致。若最终决定按 24576 执行，需要用 `prepare_dataset.py --targets 2048,8192,24576,28672` 重新生成对应 prompt 池，并修改基准配置，runner 本身无需修改。

先用 dry-run 校验数据、模型路径和最终命令。该操作不会启动推理：

```bash
cd /path/to/bench_test_llama

./bench_test.py \
  --llama-cli /path/to/llama.cpp/build/bin/llama-cli \
  --model /path/to/Qwen3.5-9B-BF16.gguf \
  --platform cuda \
  --device-index 6 \
  --dry-run
```

确认后删除 `--dry-run` 即可执行四组正式基线。CUDA 的 `--device-index 6` 指物理 6 号卡；脚本会设置 `CUDA_VISIBLE_DEVICES=6`，因此传给 llama.cpp 的设备名是 `CUDA0`。

只运行一组或做一次冒烟测试：

```bash
./bench_test.py \
  --llama-cli /path/to/llama.cpp/build/bin/llama-cli \
  --model /path/to/Qwen3.5-9B-BF16.gguf \
  --platform cuda \
  --device-index 6 \
  --case LC-001 \
  --warmup-runs 0 \
  --formal-runs 1
```

也可以通过环境变量指定两个主要路径：

```bash
LLAMA_CLI=/path/to/llama-cli \
LLAMA_MODEL=/path/to/model.gguf \
./bench_test.py --platform cuda --device-index 6
```

每个 case 只启动一个 `llama-cli` 进程，15 条 prompt（5 次预热 + 10 次正式）通过 stdin 作为同一上下文窗口中的连续对话轮次输入，KV 缓存跨轮驻留复用，只有第一轮预热承担模型加载。默认启用 `--ignore-eos`，确保生成指定数量的 token；同时禁用 `llama-cli` 自带的隐藏 warmup，因为测试清单已经明确给出了 5 次预热。stdin 多行协议会给每条 prompt 追加一个换行 token（已记录在 manifest 的 `metric_notes` 中）。

结果保存在 `results/<时间>-llama-cli-<平台>-device<编号>/`：

- `manifest.json`：模型校验值、软件版本、测试参数和固定运行清单
- `LC-xxx/session.stdout.log`、`session.stderr.log`、`session.command.txt`：整个会话的原始记录，另有每轮的 `{phase}-{run:02d}.stdout.log` / `.stderr.log` 切片
- `runs.jsonl` 和 `runs.csv`：逐次结构化结果
- `summary.json` 和 `summary.csv`：只聚合正式运行的结果

`prompt_tps_measured` 和 `generation_tps_measured` 直接解析自 `llama-cli` 每轮打印的 timing 行；进程 wall time、RSS 和整卡显存峰值由脚本外部采集。`ttft_ms_est`、`tpot_ms_est`、`itl_ms_est` 等带 `_est` 的字段是根据吞吐推导的估算值，不是真正逐 token 打点得到的 TTFT 或 ITL。CUDA 显存值是物理卡的总占用，不是该进程独占值，因此同时记录了启动前基线和增量。

`summary.csv` / `summary.json` 输出五项核心指标，表头为双语“中文/英文”，每个 case 一行：首令牌延迟(ms)/TTFT(ms)、每令牌时间(ms/token)/TPOT(ms/token)、令牌间延迟(ms/token)/ITL(ms/token)、端到端耗时(s)/E2E(s)、峰值显存(GB)/Peak Memory(GB)。TTFT/TPOT/ITL 取正式轮次均值（TTFT = prompt 处理时间 + 一个输出 token；TPOT 与 ITL 均为 1000 / generation_tps），E2E 为相邻 timing 行之间的每轮实测墙钟时间均值，峰值显存取采样最大值。逐轮明细仍保留在 `runs.jsonl` 和 `runs.csv` 中。

运行 `./bench_test.py --help` 查看完整参数，运行 `./bench_test.py --list-cases` 查看四个固定测试项。

## LC-001 至 LC-028

`configs/lc-001-028-suite.json` 一条命令覆盖 v6.1 测试矩阵的全部 28 个 llama.cpp 用例：公平基线（LC-001 至 LC-004）加上 LC-005 至 LC-028 的六组单变量实验。除表中变量外，其余参数与 F16 公平基线一致：

| 用例 | 唯一实验变量 | 模型键 |
| --- | --- | --- |
| LC-001 至 LC-004 | 公平基线 | `f16` |
| LC-005 至 LC-008 | 权重量化 Q8_0 | `q8_0` |
| LC-009 至 LC-012 | 权重量化 Q4_K_M | `q4_k_m` |
| LC-013 至 LC-016 | K/V Cache 均为 Q8_0 | `f16` |
| LC-017 至 LC-020 | K/V Cache 均为 Q4_0 | `f16` |
| LC-021 至 LC-024 | Flash Attention 关闭 | `f16` |
| LC-025 至 LC-028 | poll 为 0 | `f16` |

所有组仍使用 2K、8K、16K、28K 四档输入和同一批 prompt。Q4_K_M 当前没有匹配的 importance matrix，因此模型是无 imatrix 转换；该条件已写入 `configs/model-recipes.json`，不能与使用 imatrix 的 Q4_K_M 结果混为一谈。

量化模型由 F16 GGUF 转换，BF16 和 F16 原文件均保留：

```bash
LLAMA_QUANTIZE=/path/to/llama-quantize
MODEL_DIR=/path/to/Qwen3.5-9B-GGUF

"$LLAMA_QUANTIZE" "$MODEL_DIR/Qwen3.5-9B-F16.gguf" \
  "$MODEL_DIR/Qwen3.5-9B-Q8_0.gguf" Q8_0 16

"$LLAMA_QUANTIZE" "$MODEL_DIR/Qwen3.5-9B-F16.gguf" \
  "$MODEL_DIR/Qwen3.5-9B-Q4_K_M.gguf" Q4_K_M 16
```

先校验全部 28 个场景并打印代表命令，不启动推理：

```bash
./run_suite.py \
  --llama-cli /path/to/llama-cli \
  --model f16=/path/to/Qwen3.5-9B-F16.gguf \
  --model q8_0=/path/to/Qwen3.5-9B-Q8_0.gguf \
  --model q4_k_m=/path/to/Qwen3.5-9B-Q4_K_M.gguf \
  --platform cuda --device-index 6 --dry-run
```

每组只运行第一个 2K 场景一次：

```bash
./run_suite.py \
  --llama-cli /path/to/llama-cli \
  --model f16=/path/to/Qwen3.5-9B-F16.gguf \
  --model q8_0=/path/to/Qwen3.5-9B-Q8_0.gguf \
  --model q4_k_m=/path/to/Qwen3.5-9B-Q4_K_M.gguf \
  --platform cuda --device-index 6 --smoke --skip-model-hash \
  --continue-on-error
```

删除 `--smoke --skip-model-hash --continue-on-error` 即执行全部 28 个场景，每个场景 5 次预热和 10 次正式运行。正式运行默认只对三个唯一模型各计算一次 SHA-256，再把结果传给各子任务，不会重复扫描 F16 文件。

v6.1 表格按测试类别分成五块，`--group` 可以一块一块跑（可重复使用或用逗号分隔，也可与 `--job` 叠加）：

```text
--group 0   公平基线        (LC-001..004)
--group 1   权重量化        (LC-005..012)
--group 2   KV Cache 精度   (LC-013..020)
--group 3   Attention 后端  (LC-021..024)
--group 4   Polling         (LC-025..028)
```

例如只跑权重量化块：

```bash
./run_suite.py \
  --llama-cli /path/to/llama-cli \
  --model f16=/path/to/Qwen3.5-9B-F16.gguf \
  --model q8_0=/path/to/Qwen3.5-9B-Q8_0.gguf \
  --model q4_k_m=/path/to/Qwen3.5-9B-Q4_K_M.gguf \
  --platform cuda --device-index 6 --group 1
```

不指定 `--group` 或 `--job` 时执行整份套件。

会话累积模式在 32 GiB 单卡上有物理限制：15 轮会话结束时上下文累计到 15 倍（输入 + 输出），28K 档整体超过 `n_ctx` 262144 上限，8K/16K 档只有在量化权重或量化 KV 的组合下才放得下，F16 公平基线配置只能覆盖 2K 档。这些用例刻意保留在套件中，硬件达到要求（例如更大显存）即可原样执行。

在当前 RTX 5090 上，严格的 LC-021 至 LC-024 配置会在初始化时显存不足：F16 权重、256K context、batch 2048 且关闭 Flash Attention 时，仅 prompt-processing 计算缓冲区还需约 8.54 GiB。配置仍保留原规范，方便在大显存设备上直接执行。不要静默降低 context/batch、关闭 KV offload 或换成 Q8 权重，否则结果不再属于同一单变量实验。

套件结果目录包含 `suite-manifest.json` 和跨任务的 `summary.json/csv`；每个实验组的子目录仍保留 `bench_test.py` 生成的完整 manifest、命令、原始日志和逐次结果。

## 多平台运行

所有平台复用同一份 `fair-baseline.json`、prompt 池和 `run-manifest.json`，只替换各平台自己的 `llama-cli`、模型路径与平台 profile：

```bash
# NVIDIA CUDA
./bench_test.py --platform cuda --device-index 6 \
  --llama-cli /path/to/cuda/llama-cli --model /path/to/model.gguf

# Ascend CANN
./bench_test.py --platform cann --device-index 0 \
  --llama-cli /path/to/cann/llama-cli --model /path/to/model.gguf

# Apple Metal
./bench_test.py --platform metal \
  --llama-cli /path/to/metal/llama-cli --model /path/to/model.gguf
```

内置 profile 的设备映射如下：

| Profile | 可见性设置 | llama.cpp 设备 |
| --- | --- | --- |
| CUDA | `CUDA_VISIBLE_DEVICES=<物理编号>` | `CUDA0` |
| CANN | `ASCEND_RT_VISIBLE_DEVICES=<物理编号>` | `CANN0` |
| Metal | 不设置可见性变量 | `Metal` |
| CPU | 不设置可见性变量 | `CPU`，并强制 `-ngl 0` |

CUDA 默认通过 NVML 采集整卡显存。其他平台可以用 `--memory-command` 传入一个输出 MiB 数值的采集命令；也可以直接在 `platforms.json` 的 profile 中配置。CANN 和 Metal 目前只通过 dry-run 验证了命令生成，仍需到对应机器上验证实际后端名称和显存采集命令。

## 可复现性要求

- 不要在不同平台上重新随机抽样。
- 所有平台必须复用同一份 `run-manifest.json`。
- 记录 llama.cpp 版本、构建选项、模型校验值和设备名称。
- 保存每次运行的完整 `llama-cli` 命令、stdout 和 stderr。
- 预热结果与正式结果分开，只聚合 10 次正式测试。
- 实测指标和推导指标必须使用不同的字段名称。
- 不要把旧 mock 数据当作正式测试结果。

模型文件、原始数据集、处理后的 prompt 和原始运行日志通常不应提交到 Git。下载脚本、校验值、manifest、结果格式和汇总表已经足够用于复现。
