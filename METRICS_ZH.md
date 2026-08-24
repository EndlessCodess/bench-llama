# 主要观察指标测试说明

本文说明 `bench_test_llama` 如何测量以下指标：TTFT、TPOT、ITL、E2E、占用显存峰值、占用显存均值、GPU 计算单元平均利用率、GPU 计算单元峰值利用率和执行状态。

## 1. 当前实现状态

| 指标 | 当前状态 | 数据来源 |
| --- | --- | --- |
| TTFT | 已实现 | `bench_run.cpp` 内部计时 |
| TPOT | 已实现 | C++ 生成阶段计时，Python 派生 |
| ITL | 已实现，但当前等同于 TPOT | Python 派生 |
| E2E | 已实现 | `bench_run.cpp` 内部计时 |
| 占用显存峰值 | 已实现 | CUDA 平台通过 NVML 定时采样整张卡已用显存 |
| 占用显存均值 | 尚未实现 | 应复用显存采样序列计算平均值 |
| GPU 计算单元平均利用率 | CUDA 已实现 | NVML `nvmlDeviceGetUtilizationRates().gpu` |
| GPU 计算单元峰值利用率 | CUDA 已实现 | NVML `nvmlDeviceGetUtilizationRates().gpu` |
| 执行状态 | 已实现 | 事件、进程退出码、超时及正式请求完成数 |

当前 `summary.csv` 中的显存单位是 GB，而目标表格使用 MB。两者只需换算：

```text
显存(GB) = 显存(MB) / 1024
```

## 2. 单次请求的测量窗口

每个测试 case 的每个 round 启动一个常驻 `bench_run` 进程。模型在该进程中只加载一次，随后按执行计划连续处理预热请求和正式请求。每条请求执行前清空 KV Cache，因此各请求之间不继承上下文。

单次请求的主要时序如下：

```text
清空 KV Cache
    |
    | run_start 事件
    v
t0 -> Prefill -> t_pp_end -> 首 token 采样完成 -> t_first_avail
                                                |
                                                v
                         逐 token Decode -> t_tg_end -> t_end
                                                          |
                                                          | result 事件
                                                          v
```

其中：

- `t0`：C++ 内部推理计时起点，位于 KV 清空、Prompt 读取、模板处理和 Tokenize 之后。
- `t_pp_end`：全部输入 token 完成 Prefill 的时刻。
- `t_first_avail`：根据 Prefill 输出的 logits 完成首 token 采样、首 token 已可返回的时刻。
- `t_tg_start`：生成阶段 Decode 计时起点。
- `t_tg_end`：最后一个输出 token 对应的 Decode 完成时刻。
- `t_end`：该请求全部推理工作完成的时刻。

CUDA 调用是异步的。每次需要记录 GPU 完成时刻前，`bench_run` 都调用 `llama_synchronize()`，避免把 CUDA Kernel 的提交时间误当成实际完成时间。

## 3. 延迟指标

### 3.1 TTFT

TTFT（Time To First Token）表示从推理开始到首个输出 token 可用的时间。

```text
TTFT(ms) = (t_first_avail - t0) / 1000
```

该指标包括：

- Prompt Prefill；
- 首 token 的采样。

该指标不包括：

- 进程启动和模型加载；
- GGUF 文件读取；
- Prompt 文件读取、聊天模板处理和 Tokenize；
- 首 token 可用后的后续 Decode。

首 token 的 logits 已由 Prefill 的最后一次 forward 产生，因此 TTFT 不额外包含一次 Decode。

实现位置：`bench_run.cpp` 的 `run_one()`，由 `t_first_avail - t0` 得到 `ttft_ms_measured`。

### 3.2 TPOT

TPOT（Time Per Output Token）表示生成阶段每个输出 token 的平均耗时。

首先在 C++ 中计算纯生成吞吐：

```text
generation_eval_ms = (t_tg_end - t_tg_start) / 1000
generation_tps = tokens_generated / (generation_eval_ms / 1000)
```

随后由 Python 计算 TPOT：

```text
TPOT(ms/token) = 1000 / generation_tps
               = generation_eval_ms / tokens_generated
```

TPOT 的分母不包含 Prefill 和 TTFT，因此 `1000 / TPOT` 对应纯 Decode 阶段的 `generation_tps`，不等于端到端吞吐率。

实现位置：

- `bench_run.cpp` 输出 `generation_eval_ms_measured` 和 `generation_tps_measured`；
- `bench_test.py` 派生 `tpot_ms_measured`。

### 3.3 ITL

ITL（Inter-Token Latency）通常表示相邻两个输出 token 的到达时间差。

当前项目没有记录每个 token 的独立到达时间序列，而是在单并发、逐 token Decode 场景下直接定义：

```text
ITL(ms/token) = TPOT(ms/token)
```

因此当前 ITL 是平均令牌间隔，不是 P50、P90、P99 等逐 token 延迟分布。如果后续需要尾延迟，应记录每个 token 的可用时间戳，再独立计算 ITL 分布。

### 3.4 E2E

E2E（End-to-End Latency）表示 C++ 推理执行器内部从推理开始到全部输出 token 处理完成的总耗时。

```text
E2E(ms) = (t_end - t0) / 1000
E2E(s)  = E2E(ms) / 1000
```

E2E 包含 Prefill、首 token 采样和全部 Decode，但不包含模型加载、Prompt 文件读取、模板处理和 Tokenize。

项目还记录了 Python 外部墙钟 `wall_time_s_measured`。它由 `run_start` 和 `result` 两个 JSON 事件的到达时间之差得到，包含少量 batch 创建、JSON 管道和进程通信开销。表格中的 E2E 使用 C++ 内部的 `e2e_ms_measured`，不使用 Python 外部墙钟。

## 4. 显存指标

### 4.1 采样方法

CUDA 平台通过 NVML 的 `nvmlDeviceGetMemoryInfo()` 读取物理 GPU 的 `memory.used`。监控线程按照 `--monitor-interval` 定时采样，当前正式测试使用 0.1 秒采样间隔。

每次收到 `run_start` 事件时，Python 重置当前请求的监控窗口；收到 `result` 后，将该窗口的统计值写入 `runs.csv` 和 `runs.jsonl`。

当前采集的是整张物理 GPU 的已用显存，不是严格的单进程显存。因此测试时必须保证目标卡没有其他任务，否则其他进程的显存也会进入结果。

### 4.2 占用显存峰值

单次请求的峰值：

```text
peak_memory_run(MB) = max(memory_sample_1, ..., memory_sample_n)
```

单个 case 的汇总峰值：

```text
peak_memory_case(MB) = max(所有成功正式请求的 peak_memory_run)
```

预热请求不参与最终汇总。

当前原始字段为 `peak_device_memory_mib_measured`。现有汇总表将其除以 1024 输出为 GB；若目标表格要求 MB，应直接输出 MiB 数值或将列名明确写为 MiB。

### 4.3 占用显存均值

该指标目前尚未写入代码。建议使用与峰值相同的请求窗口和采样序列：

```text
average_memory_run(MB) = sum(memory_samples) / sample_count
average_memory_case(MB) = mean(所有成功正式请求的 average_memory_run)
```

为了跨平台结果可比，所有后端应使用相同或尽可能接近的采样周期，并明确这是“设备总已用内存均值”还是“当前进程设备内存均值”。当前峰值指标属于前者，因此均值也应先采用前者，保持口径一致。

## 5. GPU 计算单元利用率

### 5.1 CUDA 采样方法

CUDA 平台调用：

```text
nvmlDeviceGetUtilizationRates(device).gpu
```

监控线程按照 `--monitor-interval` 定时读取，当前正式测试的采样间隔为 0.1 秒。采样窗口与显存相同：从当前请求的 `run_start` 到 `result`。

NVML `gpu` utilization 的含义是：在 NVML 内部采样周期内，GPU 上至少有一个 Kernel 正在执行的时间比例。它是 GPU 忙碌率，不是 CUDA Core、Tensor Core 或 SM 实际达到理论峰值算力的比例。

因此该指标适合判断 GPU 是否持续繁忙，但不能单独用于判断 Kernel 优化质量。深入分析需要 Nsight Compute 的 SM Throughput、Achieved Occupancy、Tensor Core Utilization 和 DRAM Throughput 等指标。

### 5.2 GPU 计算单元峰值利用率

单次请求：

```text
peak_gpu_util_run(%) = max(util_sample_1, ..., util_sample_n)
```

单个 case 汇总：

```text
peak_gpu_util_case(%) = max(所有成功正式请求的 peak_gpu_util_run)
```

原始字段为 `peak_gpu_utilization_pct_measured`。

### 5.3 GPU 计算单元平均利用率

单次请求：

```text
average_gpu_util_run(%) = sum(util_samples) / sample_count
```

单个 case 汇总：

```text
average_gpu_util_case(%) = mean(所有成功正式请求的 average_gpu_util_run)
```

原始字段为 `average_gpu_utilization_pct_measured`。

当前利用率采集只在 CUDA/NVIDIA 平台实现。昇腾 310P/910B 和 Metal 需要分别接入对应的设备监控接口，并在结果清单中记录采样工具和采样周期。

## 6. 执行状态

每条请求默认状态为 `failed`，只有收到合法的 `result` 事件后才改为 `ok`。以下情况会保持失败状态并记录原因：

- 等待 `run_start` 或 `result` 超时；
- `bench_run` 提前退出；
- Prompt、模板、Tokenize、Prefill 或 Decode 失败；
- 收到 `error` 事件；
- 进程返回非零退出码。

单个 case 的汇总状态：

```text
ok         = 请求的所有正式样本全部成功
incomplete = 至少一个正式样本失败或缺失
```

预热请求用于稳定运行状态，不进入性能指标汇总；但预热失败仍应视为测试异常并检查日志。

## 7. 正式样本的汇总规则

`bench_test.py` 只汇总成功的正式请求，预热请求不参与结果。延迟和平均利用率默认取算术平均，详细 JSON 中同时保留标准差、最小值和最大值。

| 表格指标 | case 级汇总规则 |
| --- | --- |
| TTFT | 成功正式请求的均值 |
| TPOT | 成功正式请求的均值 |
| ITL | 成功正式请求的均值，当前与 TPOT 相同 |
| E2E | 成功正式请求的均值，换算为秒 |
| 占用显存峰值 | 成功正式请求峰值中的最大值 |
| 占用显存均值 | 尚未实现；建议取各正式请求显存均值的均值 |
| GPU 计算单元平均利用率 | 各正式请求平均利用率的均值 |
| GPU 计算单元峰值利用率 | 各正式请求峰值中的最大值 |
| 执行状态 | 全部正式请求成功为 `ok`，否则为 `incomplete` |

## 8. 数据文件

- `runs.csv`：每条预热和正式请求的原始指标。
- `runs.jsonl`：与 `runs.csv` 对应的逐请求 JSON 记录。
- `summary.json`：包含每项指标的 mean、stdev、min 和 max。
- `summary.csv`：每个 case 一行的简洁汇总。
- `session.stdout*.log`：`bench_run` 输出的 JSON 事件流。
- `session.stderr*.log`：llama.cpp 运行日志。
- `suite-manifest.json`：硬件、模型、配置和实际执行命令。

查看最新 CUDA 测试汇总：

```text
results/20260821-141218-lc-001-028-v1-cuda-device6/summary.csv
```

## 9. 结果解释注意事项

1. TPOT 的倒数对应纯 Decode 吞吐，不对应包含 TTFT 的端到端吞吐。
2. 当前 ITL 是 TPOT 的同义派生值，不能表示逐 token 尾延迟。
3. 当前显存值是整张卡的已用显存，测试期间必须独占目标设备。
4. NVML GPU 利用率表示 Kernel 忙碌率，不表示实际达到的峰值计算效率。
5. 0.1 秒采样可能漏掉极短暂的显存或利用率尖峰；需要捕获更短瞬态时应缩短采样周期或使用平台 Profiler。
6. 不同平台必须在结果中记录监控工具、采样周期和指标语义，不能只比较同名百分比。
