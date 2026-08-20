# 共享基准数据集使用说明（SGLang / vLLM 侧）

本包是与 llama.cpp 侧共用的统一基准数据集。核心目的：**双方跑不同的测试案例，但使用完全相同的 prompt 输入**，保证输入侧公平。拿到本包后按下面的说明直接用即可，无需任何额外处理。

## 1. 数据集是什么

- 来源：LongBench-v2 官方数据集（503 条），预处理脚本固定生成，生成后不再改动
- 内容：**200 条 prompt**，按目标 token 长度切分为 4 个池子

```
prompts/
├── 2048/   （50 条，模板后约 2048 tokens）
├── 8192/   （50 条，模板后约 8192 tokens）
├── 16384/  （50 条，模板后约 16384 tokens）
└── 28672/  （50 条，模板后约 28672 tokens）
```

- 每个池子大小都是 **50 条，全部来自不同的源样本，池内无重复**（跨池子内容可控）
- 每条 prompt = 1 个源样本的 question + 4 个 choices + 截断后的 context，**无重复填充文本**
- 4 个池子共用同一批源样本的**同一顺序**（见 `run-manifest.json`），长度对比时内容差异受控

## 2. 数据如何切分

切分流程（已固化在 `preparation.json`，供追溯）：

1. 从 LongBench-v2 的 6 个 domain 均衡选取 50 个源样本（固定 seed = 20260805）
2. 对每个源样本：保留 question 和全部 4 个 choices，**只截断 context**
3. 套 Qwen chat 模板（`add_generation_prompt=true`）后统计 token 数
4. 二分查找最长的 context 前缀，使模板后 token 数 ≤ 目标长度
5. 每个目标长度（2048/8192/16384/28672）生成一条 prompt 文件，记录实际 token 数

所以 `prompts/2048/xxx.txt` 和 `prompts/8192/xxx.txt` 中同名的源样本，是同一个 question 配不同长度的 context 截断。

## 3. 使用方式（直接照做即可）

### 3.1 取 prompt 文本

`prompts/<target>/*.txt` 是**原始 user 内容**（未套模板）。发请求时用 chat 格式：

```json
{"messages": [{"role": "user", "content": "<prompt 文件全文>"}], "add_generation_prompt": true}
```

让引擎自己的模板逻辑应用 Qwen 模板。模板后 token 数 ≈ 对应池子的目标长度（见 `metadata.jsonl` 的 `actual_tokens`）。

### 3.2 选哪个池子

按你的测试案例需要的输入长度选目录：

| 需要输入长度（模板后） | 用目录 |
| --- | --- |
| ~2048 tokens | `prompts/2048/` |
| ~8192 tokens | `prompts/8192/` |
| ~16384 tokens | `prompts/16384/` |
| ~28672 tokens | `prompts/28672/` |

池子内 50 条如何抽样、每条用几次，由你的测试案例自行决定（比如固定随机选 N 条、或全部 50 条各跑一次）。

### 3.3 校验（可选但推荐）

- `metadata.jsonl` 每条记录字段：

| 字段 | 含义 |
| --- | --- |
| `pool_index` | 池内编号 0-49（文件名前缀 000-049 与此对应） |
| `source_id` | LongBench-v2 源样本 ID |
| `target_tokens` | 目标长度（2048/8192/16384/28672） |
| `actual_tokens` | 模板后实际 token 数（Qwen tokenizer 统计） |
| `token_gap` | target - actual（0 或 1） |
| `prompt_file` | 相对路径 |
| `prompt_sha256` | 文件哈希 |
| `answer` / `domain` 等 | 样本元数据 |

- 若你的引擎 tokenize 结果与 `actual_tokens` 相差 ±1（BPE 边界差异），**以实际值为准**，记录实际值即可
- 文件哈希可在取用后复核（`sha256sum`），确保输入未被改动

## 4. 可选的固定运行顺序（`run-manifest.json`）

llama.cpp 侧的测试按固定顺序跑：每个长度池子固定 5 轮 warmup + 100 轮 formal = 105 轮。**全部无随机，固定 seed 可完整复现**：warmup 固定跑池内前 5 条（pool 0-4，顺序固定，不计入结果）；formal 阶段按池内顺序（pool 0-49）顺序循环 2 遍：每条 prompt 恰好被测 2 次（同一 prompt 的两次结果可直接对比稳定性）。

```json
{"seed": 20260805, "warmup_runs": 5, "formal_runs": 100, "passes": 2,
 "sampling": {"warmup": "fixed_sequential", "formal": "cycle_sequential_2x"},
 "selections": [{"run": 1, "phase": "warmup", "pool_index": 0, "prompt_files": {"2048": "...", "8192": "...", ...}}, ...]}
```

- 如果你们需要与 llama.cpp 侧**逐轮对齐**（同一轮用同一条 prompt），直接按此文件顺序发请求即可
- 如果你们有自己的案例定义，**忽略此文件**，只用 `prompts/` 池子自行抽样

## 5. 注意事项

- prompt 文件只含文本，**不含模型、不含模板、不含任何引擎相关配置**
- 模型建议与 llama.cpp 侧一致：Qwen3.5-9B（BF16），模型文件 SHA-256 见 `configs/model-recipes.json`（由对方另行提供/确认）
- 任何切分逻辑的疑问以 `preparation.json`（seed、目标、池大小、抽样方式）为准
