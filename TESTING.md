# 测试快速上手

本机（8 卡共享机器）跑 llama.cpp 侧基准的速查手册。**所有碰 GPU 的命令一律 `CUDA_VISIBLE_DEVICES=4`，只许用物理 GPU 4。**

## 0. 跑之前（必做）

```bash
nvidia-smi   # 确认 GPU4（第 5 块，编号 4）空闲，显存 ≈0.5 GiB 基线
cd /data1/xuyang/projects/bench_test_llama
```

模型（Qwen3.5-9B GGUF 三档，路径固定）：
- F16：`/data1/xuyang/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-F16.gguf`
- Q8_0：`/data1/xuyang/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q8_0.gguf`
- Q4_K_M：`/data1/xuyang/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf`

## 1. 三层结构（先看懂，再测试）

| 层       | 文件                     | 职责                                                              |
| -------- | ------------------------ | ----------------------------------------------------------------- |
| 套件调度 | `run_suite.py`           | 按套件配置逐 job 起 bench_test.py                                 |
| 用例编排 | `bench_test.py`          | 一个 case = 一次 bench_run 进程（105 轮 = 5 warmup + 100 formal） |
| 推理内核 | `build/bench_run`（C++） | 单进程加载模型，逐轮 KV 清空 + 单轮推理 + JSONL 事件输出          |

## 2. 常用命令

### 2.1 冒烟（最快验证，几分钟）
```bash
CUDA_VISIBLE_DEVICES=4 python3 run_suite.py --bench-run build/bench_run \
  --model f16=/data1/xuyang/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-F16.gguf \
  --model q8_0=/data1/xuyang/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q8_0.gguf \
  --model q4_k_m=/data1/xuyang/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf \
  --suite configs/lc-001-028-suite.json --platform cuda --device-index 4 \
  --smoke
```
`--smoke` = 每个 job 第一个 case 只跑 1 轮，用于验证环境/代码没坏。

### 2.2 只测一个 case（单 case 单轮）
```bash
CUDA_VISIBLE_DEVICES=4 python3 bench_test.py --bench-run build/bench_run \
  --model /data1/xuyang/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-F16.gguf \
  --config configs/fair-baseline.json --platform cuda --device-index 4 \
  --case LC-001 --warmup-runs 1 --formal-runs 1
```
加 `--warmup-runs 0 --formal-runs 1` 更快，只验证能跑通。

### 2.3 完整套件 1 轮（28 case × 105 轮 = 2940 次推理，≈3.3h）
```bash
CUDA_VISIBLE_DEVICES=4 python3 run_suite.py --bench-run build/bench_run \
  --model f16=... --model q8_0=... --model q4_k_m=... \
  --suite configs/lc-001-028-suite.json --platform cuda --device-index 4
```

### 2.4 完整套件 10 轮（29,400 次推理，≈33-36h，正式跑分用）
```bash
CUDA_VISIBLE_DEVICES=4 nohup python3 run_suite.py --bench-run build/bench_run \
  --model f16=/data1/xuyang/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-F16.gguf \
  --model q8_0=/data1/xuyang/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q8_0.gguf \
  --model q4_k_m=/data1/xuyang/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf \
  --suite configs/lc-001-028-suite.json --platform cuda --device-index 4 \
  --rounds 10 \
  > rounds10-run.log 2>&1 &
```
10 轮 = 全套重复 10 次（每次都是全新进程、模型重载），1050 轮/case，正式样本合并统计（1000 个/case），10 次重复测量降误差。

### 2.5 只跑部分
```bash
# 只跑某几个 case（可逗号分隔/重复）
--case LC-001,LC-002
# 只跑某几个 job
--job fair-baseline,kv-q8-0
# 只跑某个组（0=fair基线 1=权重量化 2=KV精度 3=注意力后端 4=polling）
--group 0
```

### 2.6 只验证不跑
```bash
... --dry-run          # 校验配置并打印命令，不启动推理
... --list-cases       # 列出全部用例
```

## 3. 结果在哪看

```
results/<时间戳>-lc-001-028-v1-cuda-device4/
├── suite-manifest.json        # 全套配置存档
├── summary.json / summary.csv # 全套 28 case 汇总（套件级）
└── <job>/                     # 如 fair-baseline/
    ├── runs.jsonl / runs.csv  # job 内逐轮明细（1050 行/case，10 轮时）
    ├── summary.json / csv     # job 级汇总
    └── LC-001/
        ├── plan.tsv           # 该轮次的执行计划
        ├── session.command.txt
        ├── session.stdout.log # bench_run 事件流（runs>1 时带 .rNN 后缀）
        └── session.stderr.log
```

**summary.csv 关键列**（均值，正式样本 100 个/轮）：
首令牌延迟 TTFT(ms) | 每令牌 TPOT(ms/token) | 令牌间延迟 ITL(ms/token) | 端到端 E2E(s) | 吞吐(ttok/s) | 峰值显存(GB)

**事件流速查**（session.stdout.log 里）：`model_loaded` → 105×（`run_start` → `result`）→ `all_done`。
数结果轮数：`grep -c '"event":"result"' session.stdout.log` 应为 105（×轮数）。

## 4. 注意事项（踩过的坑）

1. **只用 GPU4**：不设 `CUDA_VISIBLE_DEVICES=4` 会默认抢 0 号卡；`--device-index 4` 必须与之一致。
2. `--parallel` 必须为 1（单轮单进程设计，套件配置里已固定）。
3. 停任务：**不要 pkill**（会连外层 shell 一起杀）——用任务 ID 停止（TaskStop / kill 具体 PID）。
4. 10 轮跑完前别开其他占显存的东西（峰值 ~24GB，卡上别的东西会 OOM）。
5. `--skip-model-hash`：不想每次等模型 SHA-256（大文件哈希耗时）可加，结果缺校验值。
6. 失败排查顺序：`session.stdout.log` 里 error/warn 事件 → `session.stderr.log` → `session.command.txt` 核对参数。
7. 结果目录想固定名字：`--output-dir <路径>`（默认按时间戳自动建）。

## 5. 测试矩阵速览（32 case）

| 组  | 变量           | case                                 | 模型          | 关键参数                                            |
| --- | -------------- | ------------------------------------ | ------------- | --------------------------------------------------- |
| 0   | fair 基线      | LC-001..004                          | f16           | 2048/8192/16384/28672 tokens，输出 64/256/512/1024  |
| 1   | 权重量化       | LC-005..008 q8_0、LC-009..012 q4_k_m | q8_0 / q4_k_m | 同上长度×输出                                       |
| 2   | KV 精度        | LC-013..016 q8_0、LC-017..020 q4_0   | f16           | type-k/v 覆盖                                       |
| 3   | FlashAttention | LC-021..024 关 FA                    | q8_0          | `--flash-attn off`（FA-off 的 KQ 矩阵固定 ~8.6GiB） |
| 4   | polling        | LC-025..028 poll=0                   | f16           | `--poll 0`                                          |
| 5   | Graph execution | LC-029..032 graph off               | f16           | 关闭平台图执行，其他参数同 LC-001..004               |

只补测 CUDA Graph 关闭组：

```bash
CUDA_VISIBLE_DEVICES=6 python3 run_suite.py --bench-run build/bench_run \
  --model f16=/data1/xuyang/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-F16.gguf \
  --suite configs/lc-001-032-suite.json --job graph-off \
  --platform cuda --device-index 6
```

图执行平台口径：

| 平台 | 基线构建 | 默认运行 | LC-029..032 |
| --- | --- | --- | --- |
| CUDA | `-DGGML_CUDA_GRAPHS=ON` | 开启 | `GGML_CUDA_DISABLE_GRAPHS=1` |
| 910B | `-DUSE_ACL_GRAPH=ON` | 开启 | `GGML_CANN_ACL_GRAPH=off` |
| 310P | `-DUSE_ACL_GRAPH=OFF` | 不支持 | 不适用，不运行 |

910B 补测时使用 `--platform cann_910b`。310P 使用 `--platform cann_310p` 和通用的 `lc-001-028-suite.json`，不包含 LC-029..032。

910B 机器上先构建开启 ACL Graph 的 llama.cpp 和对应的 `bench_run`：

```bash
cmake -S /path/to/llama.cpp -B /path/to/llama.cpp/build-cann-910b \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DGGML_CANN=ON -DUSE_ACL_GRAPH=ON
cmake --build /path/to/llama.cpp/build-cann-910b -j

cmake -S . -B build-cann-910b -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CPP_ROOT=/path/to/llama.cpp \
  -DLLAMA_CPP_BUILD=/path/to/llama.cpp/build-cann-910b
cmake --build build-cann-910b -j
```

先使用一个 case 冒烟验证：

```bash
python3 run_suite.py --bench-run build-cann-910b/bench_run \
  --model f16=/path/to/Qwen3.5-9B-F16.gguf \
  --suite configs/lc-001-032-suite.json --job fair-baseline \
  --platform cann_910b --device-index 0 --smoke
```

冒烟通过后去掉 `--smoke` 即可按套件配置正式测试。
