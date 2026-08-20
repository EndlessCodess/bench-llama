# Context Shift（循环窗口）对 Qwen3.5-9B 失效的原因报告

日期：2026-08-06 · 对象：llama.cpp（本地构建，CUDA 后端）· 模型：Qwen3.5-9B BF16 GGUF

## 一句话结论

Qwen3.5 使用 **IM-RoPE**（interleaved multi-scale RoPE，每 token 4 个位置值），与 llama.cpp 的 **K-shift**（`--context-shift` 循环窗口机制，假设每 token 1 个位置值）**架构级不兼容**。llama.cpp 检测到该模型后**静默禁用** K-shift：开不开 `--context-shift` 效果完全一样，超窗输入一律拒绝（HTTP 400），生成中溢出直接报错终止。这是**上游未实现**，不是配置问题。

## 背景：为什么需要循环窗口

基准测试采用"会话累积模式"：单进程、同一上下文窗口内连续 15 轮对话（5 预热 + 10 正式），KV 缓存跨轮驻留。15 轮结束时上下文累计到 15 ×（输入 + 输出）token：

| 档位 | 每轮（输入+输出） | 15 轮累计 | n_ctx 262144 | 结论 |
| ---: | ---: | ---: | ---: | --- |
| LC-001 (2K) | 2048 + 64 | 31.7K | 放得下 | ✓ |
| LC-002 (8K) | 8192 + 256 | 126.7K | 放得下 | 显存不够（F16 KV 35.4GB） |
| LC-003 (16K) | 16384 + 256 | 249.6K | 放得下 | 显存不够（50.4GB） |
| LC-004 (28K) | 28672 + 512 | 437.8K | **超出** | ✗ 永远不可能 |

长档位唯一的出路是窗口滚动（K-shift）：窗口写满后丢弃最旧 token、把剩余 token 的位置整体前移，继续生成。实测这条路对 Qwen3.5 走不通，原因如下。

## 根因：IM-RoPE 与 K-shift 架构不兼容

### 经典 RoPE（Llama / Qwen2 系）下 K-shift 为什么可行

- 每个 token **1 个位置值**，位置张量形状 `1 × n_tokens`。
- 位置 p 的 K 向量旋转：每对维度旋转角度 = `p × θᵢ`（θᵢ 是各维度的频率，单一全局 n_rot）。
- 窗口滚动丢 2 个旧 token 后，token 从位置 p 变成 p−2：所有维度对的修正角**统一**为 `−2 × θᵢ`，等价于在 K 上做一次"反向旋转"，是一个单一、高效的图操作。

### Qwen3.5 的 IM-RoPE

- 每层每个 token 有 **4 个位置值**（多尺度，如 (p, 2p, 4p, 8p)），位置张量形状 `4 × n_tokens`，每 token 实际 token 数翻 4 倍。
- 位置 5 的 K 旋转是 4 个角度的复合：`[5θ₀, 10θ₁, 20θ₂, 40θ₃]`。
- 同样丢 2 个 token（token 从位置 5 → 3），需要的修正角是 `[−2θ₀, −4θ₁, −8θ₂, −16θ₃]`——**每个维度对的修正角 = Δp × 各自的尺度系数，各不相同**。

K-shift 的图是按"单一 n_rot、统一修正角"设计的，无法表达这种按尺度分别修正的旋转，llama.cpp **未实现** IM-RoPE 的 K-shift。

## 附：显存账本（为什么 8K/16K/28K 档放不下）

### 模型结构参数（GGUF 元数据实测，非推算）

| 参数 | 值 |
| --- | --- |
| 层数 n_layer | 32 |
| KV head 数 n_kv_heads | 4（GQA） |
| head 维度 | 256 |
| 上下文上限 n_ctx（元数据） | 262144 |
| 权重 F16 文件 | 17.92 GB |
| 权重 Q8_0 文件 | 9.53 GB |
| 权重 Q4_K_M 文件 | 5.63 GB |

### 计算公式

```
总显存 = 权重文件大小 + KV 缓存 + 计算缓冲（约 1.5~2 GB）

KV 缓存 = 累积上下文长度 × 每 token KV 字节
每 token KV 字节 = 层数 × KV head 数 × head_dim × 2(K+V) × 每元素字节
                = 32 × 4 × 256 × 2 × 2B = 131072 B ≈ 128 KiB/token (f16)
```

量化 KV 时每元素字节：Q8_0 = 36/32 = 1.125 B（4B scale + 32×1B 数据，每 block 32 元素）、Q4_0 = 20/32 = 0.625 B（4B scale + 16B 数据）→ 每 token 分别 ≈ 72 KiB、40 KiB。

累积上下文 = 15 轮 ×（输入 + 输出），f16 权重 + f16 KV（32 GiB 卡可用约 31.7 GB）：

| 档位 | 15 轮累积上下文 | KV 缓存 | 总显存 | 结论 |
| ---: | ---: | ---: | ---: | --- |
| LC-001 (2K) | 31,680 | 4.15 GB | 17.92+4.15+2 ≈ 24.1 GB | ✓ 实测峰值 24.25 GB |
| LC-002 (8K) | 126,720 | 16.61 GB | ≈ 36.5 GB | ✗ 超 31.7 GB |
| LC-003 (16K) | 249,600 | 32.71 GB | ≈ 52.6 GB | ✗ |
| LC-004 (28K) | 437,760 | 57.4 GB | ≈ 77 GB | ✗ 且超 n_ctx 262144，永远不可行 |

### 8K/16K 档只有量化组合才放得下

| 组合 | 8K 档总显存 | 16K 档总显存 |
| --- | ---: | ---: |
| F16 权重 + F16 KV | 36.5 GB ✗ | 52.6 GB ✗ |
| F16 权重 + Q8_0 KV | 29.3 GB ✓ | 38.3 GB ✗ |
| Q8_0 权重 + Q8_0 KV | 20.9 GB ✓ | 29.9 GB ✓ |
| Q4_K_M 权重 + Q4_0 KV | 12.8 GB ✓ | 17.9 GB ✓ |

这就是套件里 LC-002/LC-003 在 F16 公平基线配置下需要更大显存设备、且量化组合配置被刻意保留的原因。

## 代码证据（llama.cpp 源码，均已核对）

| 位置 | 内容 | 含义 |
| --- | --- | --- |
| `src/llama-model.cpp:2696-2700` | `QWEN35 / QWEN35MOE / QWEN3VL → LLAMA_ROPE_TYPE_IMROPE` | Qwen3.5 架构使用 IM-RoPE 类型 |
| `src/llama-hparams.cpp:232-234` | `n_pos_per_embd() = (MROPE\|IMROPE) ? 4 : 1` | IM-RoPE 每 token 4 个位置值 |
| `src/llama-kv-cache.cpp:1171-1181` | `get_can_shift()`：`n_pos_per_embd() > 1 → return false` | **禁用 K-shift 的核心判断** |
| `src/llama-kv-cache.cpp:852-856` | `if (!get_can_shift()) GGML_ABORT("...does not support K-shift")` | 强行 shift 会直接崩溃 |
| `tools/server/server-context.cpp:1227-1232` | `llama_memory_can_shift()==false → ctx_shift 置 false` + 日志 "ctx_shift is not supported by this context, it will be disabled" | 服务端启动时静默禁用开关 |
| `tools/server/server-context.cpp:2809` | `send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER)` | 生成中溢出、但 shift 被禁用时直接报错 |
| `tools/server/server-context.cpp:3090-3098` | `"input (X tokens) is larger than the max context size (Y tokens). skipping"` → HTTP 400 | 超窗 prompt 拒绝，不滚动 |
| `tools/server/server-context.cpp:3124` | `can_cache_reuse = llama_memory_can_shift(...) && ...` | `--cache-reuse`（前缀缓存 KV 平移）同样依赖 K-shift，一并失效 |

## 实测验证（GPU6）

- 测试：`llama-server -c 2048 --context-shift` + 发送 8192 token 的 prompt
- 结果：HTTP 400，错误消息 `input (8192 tokens) is larger than the max context size (2048 tokens). skipping`；服务端日志出现 `ctx_shift is not supported by this context, it will be disabled`
- 对照：经典 RoPE 模型（Llama/Qwen2 系）同样的配置可以正常滚动

## 影响与结论

1. `--context-shift` 和 `--cache-reuse` 对 Qwen3.5-9B **完全无效**，开与不开行为一致。
2. 会话累积 15 轮模式在 32 GiB 单卡 + Qwen3.5-9B 上只有 **2K 档（LC-001）** 可行；8K/16K 档需要量化权重/量化 KV 才放得下；28K 档超出 n_ctx 262144 上限，永远无法执行。
3. 这是 llama.cpp 上游未实现（IM-RoPE 每 token 多位置值与单一 n_rot 的 K-shift 假设冲突），非配置问题；后续可关注上游对 `n_pos_per_embd() > 1` 模型的 K-shift 支持。
