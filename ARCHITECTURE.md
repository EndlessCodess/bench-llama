# 基准测试架构讲解：三个执行文件

> 讲解提纲：**为什么是三层 → 每一层干什么 → 数据怎么流 → 指标怎么算 → 10×28×105 怎么组织**
> 涉及文件：`run_suite.py`（套件调度器）、`bench_test.py`（用例编排器）、`bench_run.cpp`（推理内核）

---

## 1. 背景：为什么要自研 C++ 执行器

本项目的对比对象是 **SGLang / vLLM**——常驻服务引擎，基准语义是"**一个进程常驻，逐条发单轮请求**"。

llama.cpp 原来的做法（`llama-cli`）是**每条样本重启一次进程**，测的是"冷启动"吞吐，和对方口径不对等，而且 warmup 完全没有意义（进程间不共享状态）。

**结论**：用 llama-cli 无法做到"常驻进程 + 逐轮 KV 清空 + 逐轮计时"，于是自研 `bench_run`（C++，基于 `examples/simple/simple.cpp` 改造），并保持三层分工：

```
┌─────────────────────────────────────────────────────────────────┐
│  run_suite.py   套件调度器：读套件配置，逐个 job 起 bench_test   │
│        │  （每 job = 一个 bench_test.py 进程，串行执行，GPU 独占）│
│        ▼                                                          │
│  bench_test.py  用例编排器：配置/校验/监控/汇总，一个 case 起一次 │
│        │  bench_run（每 case 10 轮 = 10 个 bench_run 进程）       │
│        ▼                                                          │
│  bench_run.cpp  推理内核：进程内加载模型一次，逐轮 KV 清空 +      │
│                 单轮推理 + 逐轮计时，stdout 输出 JSONL 事件流      │
└─────────────────────────────────────────────────────────────────┘
```

**分工原则**：Python 层负责"配置、校验、内存监控、汇总输出"；C++ 层只做一件事——"一个进程把计划里所有轮跑完"，不碰任何配置逻辑。

---

## 2. run_suite.py —— 套件调度器（约 360 行）

**输入**：套件配置文件 `configs/lc-001-028-suite.json`（7 个 job × 4 个 case = 28 个 LC）

**流程**（main 主循环，约 :300-314）：

```
① 解析命令行（--suite / --model KEY=PATH / --rounds / --smoke / --dry-run ...）
② 读套件配置 → 每个 job 确定：case 配置文件 + model_key（f16/q8_0/q4_k_m）
③ 模型 SHA-256：每个模型只算一次（17GB 哈希约几十秒），经 --model-sha256 传给
   所有用它跑 job 的 bench_test.py，避免重复计算
④ 逐个 job 串行执行：for job, command in base_commands:
      subprocess.run(command)      ← 一个 job = 一个 bench_test.py 进程
⑤ 每个 job 结束后：从 job 的 summary.json 收集逐 case 汇总行 + 从 runs.jsonl
   收集失败轮的 error 信息 → 聚合出套件级 summary.json / summary.csv
⑥ 写 suite-manifest.json（全套配置 + 每个 job 的完整命令存档，可回放）
```

**关键参数**：
| 参数 | 作用 |
| --- | --- |
| `--suite` | 套件配置路径 |
| `--model KEY=PATH` | 模型映射（可重复，KEY 覆盖套件用到的全部 model_key） |
| `--rounds N` | 每 case 完整跑 N 轮（透传给 bench_test.py） |
| `--job / --group` | 只跑部分 job / 部分组 |
| `--smoke` | 每个 job 第一个 case 只跑 1 轮（快速验证） |
| `--dry-run` | 只校验配置、打印命令，不启动推理 |

---

## 3. bench_test.py —— 用例编排器（约 1350 行）

### 3.1 配置从哪里读（main() 启动时只读一次）

| 配置来源 | 函数 / 行号 | 内容 |
| --- | --- | --- |
| 命令行参数 | `parse_args` :276 | --config / --platform-config / --processed-dir / 各种覆盖参数 |
| 用例配置文件 | `load_benchmark_config` :342（调用点 :1075） | case 定义（prompt_tokens / output_tokens）+ runtime 段（n_ctx/n_batch/type_k/flash_attn/poll/temp/seed…） |
| 平台档案 | `load_platform_profiles` :375（调用点 :1076） | cuda/cann/metal/cpu 各自的环境变量、backend_device、显存监控方式 |
| 预处理数据 | `load_inputs` :487（调用点 :1090） | run-manifest.json（105 轮执行清单）+ metadata.jsonl（每条 prompt 的档案）+ preparation.json |

配置读完后全部揉进 `args` 对象：`:1077` `apply_runtime_defaults` 把命令行没给的运行时参数从 config 的 `runtime` 段补默认值；`:1118` `platform_settings` 生成 `CUDA_VISIBLE_DEVICES=4`、`CUDA0`、n_gpu_layers、nvml 监控。

### 3.2 一个 case 的执行流程（main 主循环 :1255-1304）

```
for case_id in selected_cases:            # 本 job 的 4 个 case（如 LC-001..004）
    for round_num in range(1, rounds+1):  # --rounds 10 → 每 case 起 10 次 bench_run
        round_records = run_session(...)  # ← 核心：一轮 = 一次 bench_run 进程
        # 每条记录实时追加写 runs.jsonl（边跑边落盘，中断不丢已完成轮次）
        # 每轮结束打印 105 行进度（"一轮中间 rounds10-run.log 不动"的原因）
```

### 3.3 run_session —— 一轮怎么跑（:755）

```
① 生成 plan.tsv（build_run_plan :562）：
   prompt_file / case_id / phase / phase_run / manifest_run / source_id / n_predict / expected_tokens
   ← 由 run-manifest.json + metadata.jsonl 拼出，每 case 一份
② 拼 bench_run 命令行（build_bench_run_command :590）：runtime 参数全部来自 config
③ Popen 启动 bench_run（start_new_session=True，环境含 CUDA_VISIBLE_DEVICES=4）
   stdout → session.stdout.rNN.log，stderr → session.stderr.rNN.log（StreamReader 双线程实时读）
④ 逐事件消费 JSONL：model_loaded → 105 ×（run_start → result）→ all_done
   run_start 事件时刻重置内存窗口基线（ProcessMonitor 0.1s 采样 NVML）
   result 事件 → 填该轮 record（含墙钟、显存、tpot/itl 派生）
⑤ 105 条记录返回 → 写 runs.jsonl + 打印进度行
```

### 3.4 汇总（summarize_case）

对每个 case 的**正式样本**（100 轮 × 10 轮 = 1000 个/case）做 mean / stdev / min / max，输出：

- job 级 `summary.json` / `summary.csv`（6 个指标列 + 状态）
- 套件级汇总由 run_suite.py 聚合

---

## 4. bench_run.cpp —— 推理内核（约 1100 行）

### 4.1 职责

单个进程内：加载模型一次 → 按 plan.tsv 顺序逐轮执行 → 每轮输出一行 JSONL 事件 → 结束时汇总事件。**不做任何配置逻辑。**

### 4.2 单轮语义 run_one（:543）

```
┌─ KV 清空：llama_memory_clear（llama_kv_cache_clear 已在 llama.cpp 9923 移除）
│  → 上下文永不累积，每轮只占单条 prompt 的 KV → 显存回到单轮量级
├─ 读 prompt 文件 → common_chat_templates_apply 套 Qwen 模板
│  （与 llama-server /apply-template 同一路径，含 <think> workaround；token 数须与
│   metadata 一致，不一致发 warn）
├─ tokenize（两段式）→ n_prompt 个 token
├─ PREFILL 阶段（:603-637）：
│   llama_batch_init(n_prompt, 0, 1) 一次分配 + 指针 offset view 零拷贝切片
│   每 n_batch=2048 一片 → 一次 llama_decode（28K prompt = 14 次 decode）
│   只给最后一个 token 生成 logits（省 ~17GB 写带宽）
├─ 生成阶段（:771-834）：
│   首 token：采样（prefill forward 已产出其 logits，采样完即可得，TTFT 计时点
│   t_first_avail 在此）
│   后续逐 token：CPU argmax 采样（temp≤0 = 确定性，seed 仅记录）
│   → llama_batch_get_one（NULL pos = 从 KV 自动继续）
│   → llama_decode → llama_synchronize（CUDA 异步必须）
│   EOG 以 -INFINITY bias 手动注入（common_sampler_init 不会自动应用 ignore_eos）
└─ 计时（:836-847）：四项耗时算进 res 结构体（这里不输出，只计算）
```

### 4.3 时序定义（全部 ggml_time_us + llama_synchronize）

| 指标 | 定义 | 说明 |
| --- | --- | --- |
| `prompt_eval_ms` | t_pp_end − t0 | 纯 prefill 耗时（t0 = KV 清空后） |
| `ttft_ms` | t_first_avail − t0 | prefill + 采样（首 token 由 prefill forward 直接产出，无额外 decode；vLLM/SGLang serving 口径） |
| `generation_eval_ms` | t_tg_end − t_tg_start | 生成期，从第 1 个输出 token 起算 |
| `e2e_ms` | t_end − t0 | 整轮墙钟 |
| `prompt_tps` | prompt_tokens ÷ (prompt_eval_ms ÷ 1000) | prefill 吞吐 |
| `generation_tps` | tokens_generated ÷ (generation_eval_ms ÷ 1000) | 生成吞吐 |
| `tpot / itl` | 1000 ÷ generation_tps | 由 Python 层派生（定义同值） |

### 4.4 输出：JSONL 事件协议（stdout，每行 fflush 实时刷出）

```
{"event":"model_loaded", ...}                    ← 进程启动，模型加载完成
{"event":"run_start", "prompt_tokens":N, ...}    ← 每轮一条（KV 清空后、计时前；Python 以此为内存基线 + 墙钟起点）
{"event":"result", "prompt_eval_ms_measured":..., "ttft_ms_measured":...,
                "generation_tps_measured":..., ...}  ← 每轮一条（最终结果输出点，见 :1048-1100）
{"event":"warn", "message":...}                  ← 非致命（如 token 数漂移）
{"event":"error", "message":..., "fatal":...}    ← 失败轮（仍继续跑剩余轮）
{"event":"all_done", "total_runs":105, ...}      ← 进程结束信号（配合退出码判成败）
```

**result 事件是"所有统计数据的源头"**：身份信息（case/phase/run）→ 规模（prompt/expected/n_predict/generated）→ 四项耗时 → 两个 TPS，全在这一行 JSON 里。下游 Python 只做解析 + 派生 + 聚合。

---

## 5. 数据流全貌：从 C++ 计时到 summary.csv

```
bench_run 内部（纯内存）                 bench_run 输出                 bench_test.py                    summarize_case
┌─────────────────────┐   ggml_time_us   ┌──────────────┐  JSONL 管道  ┌───────────────┐  1000 样本   ┌──────────────────┐
│ run_one 算四项耗时 ─┼───────────────▶ │ result 事件 ─┼────────────▶ │ 每轮 record  ─┼───────────▶ │ mean/stdev/min/max│
│ 存进 res 结构体     │                 │ emit_object  │  实时 fflush  │ runs.jsonl    │             │ summary.json/csv │
└─────────────────────┘                 └──────────────┘              │ 派生 tpot/itl │             └──────────────────┘
                                                                      │ 墙钟/显存     │
                                                                      └───────────────┘
```

---

## 6. 10 轮 × 28 case × 105 轮怎么组织

**层级计数**（10 轮完整套件）：

| 层 | 进程数 | 说明 |
| --- | --- | --- |
| run_suite.py | 1 | 整套只有一个 |
| bench_test.py | 7 | 7 个 job 串行，同一时刻只有 1 个 |
| bench_run | 28 × 10 = 280 | 每 case 每轮一个新进程（模型重载） |
| 单轮推理（run_one） | 29,400 次 | 280 进程 × 105 轮 |

**每 case 的轮次构成**：105 轮 = 5 warmup（固定 pool 0-4，不计入统计）+ 100 formal（50 条 prompt 顺序循环 2 遍，每条恰好测 2 次，无随机、固定 seed 可完整复现）。10 轮后正式样本合并 = 1000 个/case。

**每 case 目录里有什么**（10 轮时）：

```
LC-001/
├── plan.tsv                  # 每轮次的执行计划（105 行）
├── session.stdout.r01.log    # 第 1 轮的 JSONL 事件流（105 result 事件，可 grep -c 核对）
├── ...                       # r02 ~ r10（每轮一个独立文件）
├── session.stderr.rNN.log    # 人类可读进度 + warn 转发
└── session.command.txt       # 该轮 bench_run 完整命令行
```

**进度怎么看**：
- `rounds10-run.log`：每轮结束才打印 105 行（一轮中间不动是正常的）
- `session.stdout.rNN.log`：实时逐条刷新（C++ fflush），`grep -c '"event":"result"'` 数已完成条数
- 中途断电/中断不丢已完成轮次（runs.jsonl 边跑边落盘）

---

## 7. 关键设计决策（Q&A）

**Q1 为什么不用 llama-cli？**
llama-cli 无"常驻进程 + 逐请求 KV 清空 + 单轮计时"模式；只有自研工具能做到与 SGLang/vLLM 请求式引擎同口径。

**Q2 为什么 105 轮而不是 10 轮？**
5 warmup 吸收模型加载 / CUDA kernel 初始化（每轮进程内只发生一次）；100 formal 给出极紧的统计（LC-001 TTFT p5-p95 跨度仅 ~2-3ms；同 prompt 两次测量差值中位数 ~1.7ms）。

**Q3 --rounds 10 和 105 轮是什么关系？**
105 轮 = 一个 bench_run 进程内的轮次（模型只加载一次）；`--rounds 10` = 该 105 轮完整重复 10 次（每次全新进程、模型重载）。目的：10 次独立重复测量降低硬件噪声。

**Q4 为什么 --parallel 必须为 1？**
单轮单进程设计：一个 bench_run 进程串行跑 105 轮，不支持并发序列；并行度由外部（多 case 多进程）而非内部提供。

**Q5 为什么每轮 KV 清空？**
对比对象 vLLM/SGLang 的请求式引擎每请求独立 KV；累积上下文会随轮次线性增长显存（f16 KV = 128KiB/token，524K 窗口不量化直接爆 32GB），且语义不对等。

**Q6 为什么模板必须走 common_chat_templates_apply？**
直接调 llama_chat_apply_template 会漏掉 chat-diff-analyzer workaround（如旧 Qwen reasoning 模板注入 `<think>`），导致 token 数与 prepare_dataset.py 的 metadata 不一致（2048 vs 2046）。

**Q7 EOG 为什么手动注入？**
common_sampler_init 不会自动应用 ignore_eos（那在 common_init_sampler_from_model 里）——必须自己遍历 vocab 把 EOG token 的 logit 置 -INFINITY，否则生成长度不可控。

**Q8 计时为什么每段都要 llama_synchronize？**
CUDA 是异步的：decode 返回不代表 GPU 算完；llama_synchronize 确保取点时刻 = 实际完成时刻，否则耗时被低估。

---

## 附：测试矩阵（28 case）

| 组 | 变量 | case | 模型 | 关键参数 |
| --- | --- | --- | --- | --- |
| 0 | fair 基线 | LC-001..004 | f16 | 2048/8192/16384/28672 tokens，输出 64/256/512/1024 |
| 1 | 权重量化 | LC-005..008 q8_0、LC-009..012 q4_k_m | q8_0 / q4_k_m | 同上长度×输出 |
| 2 | KV 精度 | LC-013..016 q8_0、LC-017..020 q4_0 | f16 | type-k/v 覆盖 |
| 3 | FlashAttention | LC-021..024 关 FA | q8_0 | `--flash-attn off`（KQ 矩阵固定 ~8.6GiB） |
| 4 | polling | LC-025..028 poll=0 | f16 | `--poll 0` |
