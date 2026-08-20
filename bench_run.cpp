// bench_run.cpp — single-turn request benchmark runner for llama.cpp.
//
// Loads the model once, then executes every run in a plan file as an
// independent single-turn generation: the KV cache is cleared before each run
// so context never accumulates across runs. This mirrors the request-based
// persistent-engine semantics of SGLang / vLLM benchmarks.
//
// Emits machine-readable JSONL events on stdout; human progress goes to stderr.
//
// Plan format (tab-separated):
//   prompt_file\tcase_id\tphase\tphase_run\tmanifest_run\tsource_id\tn_predict\texpected_tokens
//
// Events:
//   {"event":"model_loaded","system_info":...}
//   {"event":"run_start",   "case_id":...,"phase":...,"phase_run":...,"manifest_run":...,
//                           "source_id":...,"prompt_file":...,"prompt_tokens":N,
//                           "expected_tokens":M,"n_predict":K,"t0_us":...}
//   {"event":"result",      ... ,"prompt_eval_ms_measured":...,"ttft_ms_measured":...,
//                           "generation_eval_ms_measured":...,"e2e_ms_measured":...,
//                           "prompt_tps_measured":...,"generation_tps_measured":...,
//                           "tokens_generated":...}
//   {"event":"warn",        "message":...}
//   {"event":"error",       "message":...,"fatal":false,"case_id":...,"phase":...,"phase_run":...,"manifest_run":...}
//   {"event":"all_done",    "total_runs":N,"failed_runs":M}
//
// Exit codes: 0 = all runs ok, 1 = at least one run failed, 2 = usage/plan error.
//
// =============================================================================
// 中文总览
// =============================================================================
// 为什么存在：对比对象 SGLang/vLLM 是"常驻引擎 + 请求式基准"（引擎启动一次、
// 逐请求发、逐请求测）。llama-cli 每次运行都重启进程（冷启动、warmup 无意义、
// 口径不对等），所以本项目自研本工具：单进程加载模型一次，按 plan 顺序执行
// 所有轮次，每轮都是相互独立的单轮请求。
//
// 单轮语义（run_one）：
//   1. llama_memory_clear 清空 KV —— 上下文永不累积，每轮是独立请求；
//   2. 读原始 prompt 文件（未套模板）→ 套 Qwen chat 模板（走 common 路径，
//      与 llama-server /apply-template 及 metadata 的 token 计数完全一致）→ tokenize；
//   3. prefill 按 n_batch 分片 decode，每片后 llama_synchronize（CUDA 异步必须）；
//   4. 采样 + decode 循环生成 n_predict 个 token（EOG 被 -INFINITY bias 抑制，
//      必然生成足量 token，不会提前结束）。
//
// 计时口径（全部 ggml_time_us + llama_synchronize 取实际完成时刻）：
//   prompt_eval_ms     = prefill 结束 − 开始            （纯 prefill）
//   ttft_ms            = 首 token 采样完 − 开始         （prefill + 采样：首 token 的
//                         logits 由 prefill forward 直接产出，无额外 decode；与
//                         vLLM/SGLang 的 serving 口径一致，不再 +1 次 decode）
//   generation_eval_ms = 最后一个 token decode 完 − 第 1 个输出 token decode 开始
//   e2e_ms             = 全部结束 − 开始
//   prompt_tps / generation_tps 由上述换算，generation_tps 倒数即 TPOT/ITL。
//
// 事件协议：stdout 每行一个 JSON 事件（fflush 实时刷新），
//   run_start 事件在 KV 清空后、计时开始前发出 —— Python 层以它为
//   内存采样窗口的基线；result 事件携带全部实测耗时。error 不 fatal，
//   该轮失败后继续跑完剩余轮次，最后 all_done + 退出码 1。
//
// 调用方：bench_test.py（每 case 一次进程），plan.tsv 由 bench_test.py 从
// run-manifest.json 生成。构建：cmake 链接 llama.cpp 构建产物（build-cuda13）。
// =============================================================================

#include "llama.h"
#include "sampling.h" // common_sampler_init / common_sampler_sample / common_sampler_free
#include "chat.h"     // common_chat_templates_* (matches the llama-server /apply-template path)

#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// JSONL emission（stdout 事件输出层）
// ---------------------------------------------------------------------------
// bench_run 与 Python 层的唯一通信通道：每行一个 JSON 事件。
// 约定：stdout 只放机器可读事件（每行一条），人类可读进度一律走 stderr。
// 所有事件必须 puts 后立即 fflush —— 否则 Python 的流式读取会卡在管道缓冲里。

// 对字符串做 JSON 转义（引号/反斜杠/换行等控制字符），保证事件流永远是合法 JSON
static void json_escape(const std::string &s, std::string &out)
{
    for (unsigned char c : s)
    {
        switch (c)
        {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (c < 0x20)
            {
                // 其余控制字符转成 \uXXXX
                char buf[8];
                snprintf(buf, sizeof(buf), "\\u%04x", c);
                out += buf;
            }
            else
            {
                out += (char)c;
            }
        }
    }
}

// 事件字段：有序 (键, 已转义的值) 列表。用 pair 保证输出字段顺序稳定可预期
typedef std::vector<std::pair<std::string, std::string>> Fields;

// 组装一行 JSON 并输出（puts + fflush，事件实时可见）
static void emit_object(Fields fields)
{
    std::string line = "{";
    for (size_t i = 0; i < fields.size(); ++i)
    {
        if (i > 0)
        {
            line += ",";
        }
        line += "\"" + fields[i].first + "\":" + fields[i].second;
    }
    line += "}";
    puts(line.c_str());
    fflush(stdout);
}

// 字符串字段：加引号并转义
static std::string jstr(const std::string &s)
{
    std::string out = "\"";
    json_escape(s, out);
    out += "\"";
    return out;
}

// 数值字段（浮点，固定 6 位小数，保证可复现）
static std::string jnum(double v)
{
    char buf[64];
    snprintf(buf, sizeof(buf), "%.6f", v);
    return buf;
}

// 数值字段（整数）
static std::string jint(int64_t v)
{
    char buf[64];
    snprintf(buf, sizeof(buf), "%lld", (long long)v);
    return buf;
}

// 所有事件公共的标识字段：event 类型 + case/phase/轮次身份，
// Python 层靠这些字段把事件与计划中的某轮一一配对
static Fields common_fields(const char *event, const std::string &case_id,
                            const std::string &phase, int phase_run, int manifest_run)
{
    Fields f;
    f.push_back({"event", jstr(event)});
    f.push_back({"case_id", jstr(case_id)});
    f.push_back({"phase", jstr(phase)});
    f.push_back({"phase_run", jint(phase_run)});
    f.push_back({"manifest_run", jint(manifest_run)});
    return f;
}

// ---------------------------------------------------------------------------
// Command line（命令行参数）
// ---------------------------------------------------------------------------
// 所有运行时参数都由 Python 层从 case 配置（configs/*.json）透传过来，
// 本工具不做任何配置逻辑，只负责执行。

// 运行参数集合：默认值即 fair-baseline 的常规设置
struct BenchArgs
{
    std::string model_path;           // GGUF 模型文件（必需）
    std::string plan_path;            // plan.tsv 路径（必需，见 read_plan）
    uint32_t n_ctx = 262144;          // 上下文窗口长度（KV 容量）
    uint32_t n_batch = 2048;          // 逻辑 batch：prefill 按此切片
    uint32_t n_ubatch = 512;          // 物理 batch：llama_decode 实际一次处理的 token 数上限
    int32_t n_gpu_layers = 99;        // 卸载到 GPU 的层数（-ngl）
    std::string cache_type_k = "f16"; // K 缓存精度（f16/q8_0/q4_0 等）
    std::string cache_type_v = "f16"; // V 缓存精度
    std::string flash_attn = "auto";  // Flash Attention 模式（auto/disabled/enabled）
    float temp = 0.0f;                // 采样温度；<=0 走纯 greedy（argmax，确定性）
    uint32_t seed = 42;               // 采样 RNG 种子（temp<=0 时仅记录，不影响输出）
    int32_t n_threads = 0;            // CPU 线程数；0 = auto（取硬件并发数）
    int32_t poll = 50;                // 线程池轮询级别；-1 = 用库默认（LC-025..028 测 0）
    bool show_version = false;        // --version：打印系统信息后退出
    std::string dump_template;        // 调试：把 chat 模板应用到该 prompt 文件并打印结果
};

static void print_usage(const char *argv0)
{
    fprintf(stderr, "usage: %s --model model.gguf --plan plan.tsv [options]\n", argv0);
    fprintf(stderr, "  -m, --model <path>        GGUF model file (required)\n");
    fprintf(stderr, "      --plan <path>         TSV plan file (required)\n");
    fprintf(stderr, "  -c, --n-ctx <n>           context size (default 262144)\n");
    fprintf(stderr, "  -b, --n-batch <n>         logical batch size (default 2048)\n");
    fprintf(stderr, "      --n-ubatch <n>        physical batch size (default 512)\n");
    fprintf(stderr, "      --n-gpu-layers <n>    layers to offload to GPU (default 99)\n");
    fprintf(stderr, "      --cache-type-k <t>    K cache type: f32 f16 bf16 q8_0 q4_0 q4_1 iq4_nl q5_0 q5_1 (default f16)\n");
    fprintf(stderr, "      --cache-type-v <t>    V cache type (default f16)\n");
    fprintf(stderr, "      --flash-attn <m>      auto|disabled|enabled (default auto)\n");
    fprintf(stderr, "      --temp <t>            temperature, <= 0 selects greedily (default 0)\n");
    fprintf(stderr, "      --seed <n>            RNG seed (default 42)\n");
    fprintf(stderr, "  -t, --threads <n>         CPU threads, 0 = auto (default 0)\n");
    fprintf(stderr, "      --poll <n>            threadpool polling level, -1 = default (default 50)\n");
    fprintf(stderr, "      --version             print system info and exit\n");
}

// 命令行解析：同时支持 "--key value" 与 "--key=value" 两种写法
// （--key=value 形式会把 key 截到等号前；短参 -m/-c/-b/-t/-ngl 也接受）。
// 解析失败或未知参数 → 打印用法并 exit(2)（用法错误码）
static int parse_args(int argc, char **argv, BenchArgs &args)
{
    const char *argv0 = argc > 0 ? argv[0] : "bench_run";
    // 取参数值：--key=value 直接内联，否则消费下一个 argv；缺失即报错退出
    auto need_value = [&](int &i, const char *name, const char *inline_value) -> const char *
    {
        if (inline_value != nullptr)
        {
            return inline_value;
        }
        if (i + 1 < argc)
        {
            return argv[++i];
        }
        fprintf(stderr, "bench_run: missing value for %s\n", name);
        print_usage(argv0);
        exit(2);
    };

    for (int i = 1; i < argc; ++i)
    {
        const char *arg = argv[i];
        const char *inline_value = nullptr;
        const char *eq = strchr(arg, '=');
        if (eq != nullptr && arg[0] == '-' && arg[1] == '-')
        {
            // --key=value
            inline_value = eq + 1;
        }
        const char *key = arg;
        char key_buf[64] = {0};
        if (inline_value != nullptr)
        {
            size_t n = eq - arg;
            if (n >= sizeof(key_buf))
            {
                n = sizeof(key_buf) - 1;
            }
            memcpy(key_buf, arg, n);
            key = key_buf;
        }

        if (strcmp(key, "--model") == 0 || strcmp(arg, "-m") == 0)
        {
            args.model_path = need_value(i, arg, inline_value);
        }
        else if (strcmp(key, "--plan") == 0)
        {
            args.plan_path = need_value(i, arg, inline_value);
        }
        else if (strcmp(key, "--n-ctx") == 0 || strcmp(arg, "-c") == 0)
        {
            args.n_ctx = (uint32_t)atoi(need_value(i, arg, inline_value));
        }
        else if (strcmp(key, "--n-batch") == 0 || strcmp(arg, "-b") == 0)
        {
            args.n_batch = (uint32_t)atoi(need_value(i, arg, inline_value));
        }
        else if (strcmp(key, "--n-ubatch") == 0)
        {
            args.n_ubatch = (uint32_t)atoi(need_value(i, arg, inline_value));
        }
        else if (strcmp(key, "--n-gpu-layers") == 0 || strcmp(arg, "-ngl") == 0)
        {
            args.n_gpu_layers = atoi(need_value(i, arg, inline_value));
        }
        else if (strcmp(key, "--cache-type-k") == 0)
        {
            args.cache_type_k = need_value(i, arg, inline_value);
        }
        else if (strcmp(key, "--cache-type-v") == 0)
        {
            args.cache_type_v = need_value(i, arg, inline_value);
        }
        else if (strcmp(key, "--flash-attn") == 0)
        {
            args.flash_attn = need_value(i, arg, inline_value);
        }
        else if (strcmp(key, "--temp") == 0)
        {
            args.temp = (float)atof(need_value(i, arg, inline_value));
        }
        else if (strcmp(key, "--seed") == 0)
        {
            args.seed = (uint32_t)strtoul(need_value(i, arg, inline_value), nullptr, 10);
        }
        else if (strcmp(key, "--threads") == 0 || strcmp(arg, "-t") == 0)
        {
            args.n_threads = atoi(need_value(i, arg, inline_value));
        }
        else if (strcmp(key, "--poll") == 0)
        {
            args.poll = atoi(need_value(i, arg, inline_value));
        }
        else if (strcmp(key, "--version") == 0)
        {
            args.show_version = true;
        }
        else if (strcmp(key, "--dump-template") == 0)
        {
            args.dump_template = need_value(i, arg, inline_value);
        }
        else
        {
            fprintf(stderr, "bench_run: unknown option: %s\n", arg);
            print_usage(argv0);
            return 2;
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Plan file（运行计划：plan.tsv）
// ---------------------------------------------------------------------------
// 由 bench_test.py 从 run-manifest.json 生成，一行一条运行，纯路径无内嵌文本，
// 因此没有转义问题。列顺序（tab 分隔，至少 7 列）：
//   prompt_file \t case_id \t phase \t phase_run \t manifest_run \t source_id \t n_predict \t expected_tokens(可选)

// 一行计划展开后的结构
struct RunPlan
{
    std::string prompt_file;  // prompt 文件绝对路径（未套模板的原始 user 内容）
    std::string case_id;      // 用例 ID（LC-001 等）
    std::string phase;        // "warmup"（预热，不计入结果）| "formal"（正式）
    int phase_run = 0;        // 阶段内序号（warmup 1-5 / formal 1-100）
    int manifest_run = 0;     // run-manifest.json 中的全局轮次（1-105）
    std::string source_id;    // LongBench-v2 源样本 ID（用于追溯与身份校验）
    int n_predict = 0;        // 本次要生成的 token 数（64/256/512）
    int expected_tokens = -1; // metadata 中模板后实际 token 数；-1 = 未知
};

// 读入 plan 文件：跳过空行与 # 注释行，容忍 \r（Windows 换行）；
// 格式非法（列数不足/phase 非法/计数非正）→ 返回 2（用法错误）
static int read_plan(const std::string &path, std::vector<RunPlan> &runs)
{
    std::ifstream in(path);
    if (!in)
    {
        fprintf(stderr, "bench_run: cannot open plan file: %s\n", path.c_str());
        return 2;
    }
    std::string line;
    while (std::getline(in, line))
    {
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }
        if (line.empty() || line[0] == '#')
        {
            continue;
        }
        std::vector<std::string> cols;
        size_t start = 0;
        for (size_t i = 0; i <= line.size(); ++i)
        {
            if (i == line.size() || line[i] == '\t')
            {
                cols.push_back(line.substr(start, i - start));
                start = i + 1;
            }
        }
        if (cols.size() < 7)
        {
            fprintf(stderr, "bench_run: plan line has %zu fields, expected at least 7:\n  %s\n", cols.size(), line.c_str());
            return 2;
        }
        RunPlan r;
        r.prompt_file = cols[0];
        r.case_id = cols[1];
        r.phase = cols[2];
        r.phase_run = atoi(cols[3].c_str());
        r.manifest_run = atoi(cols[4].c_str());
        r.source_id = cols[5];
        r.n_predict = atoi(cols[6].c_str());
        if (cols.size() > 7 && !cols[7].empty() && cols[7] != "-")
        {
            r.expected_tokens = atoi(cols[7].c_str());
        }
        if (r.phase != "warmup" && r.phase != "formal")
        {
            fprintf(stderr, "bench_run: invalid phase '%s' (expected warmup|formal)\n", r.phase.c_str());
            return 2;
        }
        if (r.n_predict <= 0 || r.phase_run <= 0 || r.manifest_run <= 0)
        {
            fprintf(stderr, "bench_run: invalid plan line (n_predict/phase_run/manifest_run must be > 0):\n  %s\n", line.c_str());
            return 2;
        }
        runs.push_back(std::move(r));
    }
    if (runs.empty())
    {
        fprintf(stderr, "bench_run: plan file is empty: %s\n", path.c_str());
        return 2;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Helpers（工具函数）
// ---------------------------------------------------------------------------

// 校验 UTF-8 合法性：非法字节序列直接拒绝该轮（prompt 文件必须干净 UTF-8，
// 否则 tokenize 结果不可信）
static bool valid_utf8(const std::string &s)
{
    size_t i = 0;
    while (i < s.size())
    {
        unsigned char c = (unsigned char)s[i];
        if (c < 0x80)
        {
            ++i;
            continue;
        }
        int n = 0;
        if ((c & 0xE0) == 0xC0)
        {
            n = 1;
        }
        else if ((c & 0xF0) == 0xE0)
        {
            n = 2;
        }
        else if ((c & 0xF8) == 0xF0)
        {
            n = 3;
        }
        else
        {
            return false;
        }
        if (i + n >= s.size())
        {
            return false;
        }
        for (int k = 1; k <= n; ++k)
        {
            if (((unsigned char)s[i + k] & 0xC0) != 0x80)
            {
                return false;
            }
        }
        i += n + 1;
    }
    return true;
}

// 把 --cache-type-k/v 的文本值映射为 ggml 类型；未知值回退 f16
// （参数合法性在 main 中先校验过，这里只负责映射）
static enum ggml_type parse_cache_type(const std::string &t)
{
    if (t == "f32")
        return GGML_TYPE_F32;
    if (t == "f16")
        return GGML_TYPE_F16;
    if (t == "bf16")
        return GGML_TYPE_BF16;
    if (t == "q8_0")
        return GGML_TYPE_Q8_0;
    if (t == "q4_0")
        return GGML_TYPE_Q4_0;
    if (t == "q4_1")
        return GGML_TYPE_Q4_1;
    if (t == "iq4_nl")
        return GGML_TYPE_IQ4_NL;
    if (t == "q5_0")
        return GGML_TYPE_Q5_0;
    if (t == "q5_1")
        return GGML_TYPE_Q5_1;
    return GGML_TYPE_F16; // unknown -> f16 (arg validation emits an error first)
}

// 把 --flash-attn 文本映射为 llama 枚举（接受 auto/disabled/off/enabled/on 五种写法）
static enum llama_flash_attn_type parse_flash_attn(const std::string &m)
{
    if (m == "disabled" || m == "off")
        return LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (m == "enabled" || m == "on")
        return LLAMA_FLASH_ATTN_TYPE_ENABLED;
    return LLAMA_FLASH_ATTN_TYPE_AUTO;
}

// Replicates llama-cli --ignore-eos: the EOG logit bias is normally injected by
// common_init_sampler_from_model (common.cpp), which common_sampler_init does
// not call, so bench_run must add it to sparams.logit_bias itself.
// 复刻 llama-cli 的 --ignore-eos：把词表中所有 EOG（End Of Generation）token 的
// logit 置为 -INFINITY，保证每轮必然生成恰好 n_predict 个 token、绝不提前终止
// （生成时长因此可比、TPOT 统计才干净）。common_sampler_init 不会像 llama-cli
// 那样自动注入该 bias，必须手动添加。
static void add_eog_bias(const llama_model *model, common_params_sampling &sparams)
{
    const llama_vocab *vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    for (llama_token t = 0; t < n_vocab; ++t)
    {
        if (llama_vocab_is_eog(vocab, t))
        {
            sparams.logit_bias.push_back({t, -INFINITY});
        }
    }
}

// Replicates --poll: the llama_context has no poll field, so the polling level
// is set on the CPU threadpool attached to the context.
// 复刻 llama-cli 的 --poll：llama_context_params 没有 poll 字段，只能通过
// 附加到 context 的 CPU 线程池设置轮询级别。做法：从 CPU backend 的
// proc_address 动态解析 "ggml_threadpool_new"（llama-cli 内部同样如此），
// 设置 tpp.poll 后创建线程池并 attach。
static ggml_threadpool_t create_threadpool(int32_t n_threads, int32_t poll)
{
    if (n_threads <= 0)
    {
        n_threads = (int32_t)std::thread::hardware_concurrency();
    }
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_dev == nullptr)
    {
        fprintf(stderr, "bench_run: no CPU backend found\n");
        return nullptr;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(cpu_dev);
    auto new_fn = (decltype(ggml_threadpool_new) *)ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
    if (new_fn == nullptr)
    {
        fprintf(stderr, "bench_run: cannot resolve ggml_threadpool_new\n");
        return nullptr;
    }
    ggml_threadpool_params tpp = ggml_threadpool_params_default(n_threads);
    if (poll >= 0)
    {
        tpp.poll = (uint32_t)poll;
    }
    ggml_threadpool_t tp = new_fn(&tpp);
    if (tp == nullptr)
    {
        fprintf(stderr, "bench_run: threadpool create failed (n_threads = %d)\n", n_threads);
    }
    return tp;
}

// Chat template with add_ass=true, matching the token-count scope recorded in
// preparation.json ("Qwen chat template including assistant generation prefix").
//
// This intentionally goes through common_chat_templates_apply (the same code
// path as llama-server /apply-template and llama-cli) rather than the raw
// llama_chat_apply_template: the common path applies template workarounds such
// as appending "<think>" after the assistant prefix for old Qwen reasoning
// templates, which is what the prepared metadata (and the SGLang/vLLM side)
// counts.
// 套 Qwen chat 模板（add_generation_prompt=true）。
// 关键点：必须走 common_chat_templates_apply（与 llama-server /apply-template、
// llama-cli 同一代码路径），而不是裸 llama_chat_apply_template ——
// common 路径会应用 chat-diff-analyzer workarounds（例如旧 Qwen reasoning
// 模板会在 assistant 前缀后注入 <think>），prepare_dataset.py 统计 metadata
// 的 actual_tokens 用的正是这个口径。两边不一致的话 token 数与
// SGLang/vLLM 侧就对不上。
static bool apply_chat_template(const llama_model *model, const std::string &user_content,
                                std::string &out, std::string &err)
{
    common_chat_templates_ptr tmpls = common_chat_templates_init(model, "", "", "");
    if (!tmpls)
    {
        err = "model has no chat template";
        return false;
    }

    common_chat_templates_inputs inputs;
    common_chat_msg msg;
    msg.role = "user";
    msg.content = user_content;
    inputs.messages.push_back(msg);
    inputs.add_generation_prompt = true;
    inputs.use_jinja = true;

    common_chat_params params = common_chat_templates_apply(tmpls.get(), inputs);
    if (params.prompt.empty())
    {
        err = "chat template produced an empty prompt";
        return false;
    }
    out = std::move(params.prompt);
    return true;
}

// 两段式 tokenize：第一遍只求所需 token 数（传 nullptr 缓冲区返回负数长度），
// 第二遍填充实际结果。special=true + parse_special=true（启用特殊 token，
// 与 llama-server /tokenize 的 add_special=false, parse_special=true 口径一致）
static bool tokenize_prompt(const llama_vocab *vocab, const std::string &text,
                            std::vector<llama_token> &tokens)
{
    const int n = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, true, true);
    if (n < 0)
    {
        return false;
    }
    tokens.resize(n);
    if (llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), (int32_t)tokens.size(), true, true) < 0)
    {
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Per-run execution（单轮执行，本工具的核心）
// ---------------------------------------------------------------------------

// 一轮的计时结果（全部毫秒，双精度保留小数）。口径对应 llama-cli 输出：
//   prompt_eval_ms     = 纯 prefill（预填充）耗时
//   ttft_ms            = prefill + 采样（首令牌延迟；首 token 由 prefill 产出，无额外 decode）
//   generation_eval_ms = 全部生成 token 的 decode 总耗时
//   e2e_ms             = 整轮墙钟时间
struct RunResult
{
    bool ok = false;
    double prompt_eval_ms = 0.0;     // prefill decode time
    double ttft_ms = 0.0;            // prefill + sampling; first token produced by the prefill pass itself
    double generation_eval_ms = 0.0; // decode time of all generated tokens
    double e2e_ms = 0.0;             // wall time of the whole run
    int prompt_tokens = 0;
    int tokens_generated = 0;
    int expected_tokens = -1;
};

// 执行一轮完整的单轮生成。返回 false 时 err 给出原因，该轮失败但不影响后续轮。
// 时序细节（重要）：
//   t0 → [KV 清空已完成] → prefill 分片 decode → t_pp_end → 采样第 1 个 token
//   → t_first_avail → 解码第 1 个 token（生成期起点 t_tg_start）→ t_first_end
//   → 循环解码剩余 token → t_end
//   TTFT = t_first_avail − t0（首 token 采样完即可得，无需额外 decode）
//   每个 decode 后必须 llama_synchronize(ctx)：CUDA 是异步的，不解同步
//   ggml_time_us 取到的只是"提交时刻"，不是 GPU 实际完成时刻。
static bool run_one(llama_context *ctx, const llama_vocab *vocab, common_sampler *smpl,
                    const RunPlan &plan, const BenchArgs &args, RunResult &res, std::string &err)
{
    // clear the KV cache: every run is an independent single-turn request
    // 清空 KV：每轮都是相互独立的单轮请求，上下文永不累积
    // （llama_kv_cache_clear 在新版 llama.cpp 已删除，llama_memory_clear 是替代）
    llama_memory_clear(llama_get_memory(ctx), false);

    // read the raw user prompt (the same files that llama-cli -f consumed)
    // 读原始 user 内容（未套模板的 prompt 文件）
    std::ifstream fin(plan.prompt_file, std::ios::binary);
    if (!fin)
    {
        err = "cannot open prompt file: " + plan.prompt_file;
        return false;
    }
    std::string raw((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
    if (!valid_utf8(raw))
    {
        err = "prompt file is not valid UTF-8: " + plan.prompt_file;
        return false;
    }

    // apply the chat template, then tokenize with special tokens enabled
    std::string templated;
    std::string tpl_err;
    if (!apply_chat_template(llama_get_model(ctx), raw, templated, tpl_err))
    {
        err = "chat template failed: " + tpl_err;
        return false;
    }
    std::vector<llama_token> tokens;
    if (!tokenize_prompt(vocab, templated, tokens))
    {
        err = "tokenize failed";
        return false;
    }

    const int n_prompt = (int)tokens.size();
    const int n_predict = plan.n_predict;

    if (n_prompt + n_predict > (int64_t)args.n_ctx)
    {
        err = "prompt (" + std::to_string(n_prompt) + " tokens) + n_predict (" + std::to_string(n_predict) + ") exceeds n_ctx (" + std::to_string(args.n_ctx) + ")";
        return false;
    }

    // run_start is emitted after the KV clear and before any timing; the Python
    // layer uses this event as the memory-window baseline and wall-clock start.
    // run_start 事件在 KV 清空后、任何计时开始前发出：
    // Python 层收到它时重置显存采样窗口并读取"本轮基线显存"（此后本轮 KV +
    // 计算缓冲的增量都归入这一轮的窗口），同时以事件到达时刻作为墙钟起点。
    Fields sf = common_fields("run_start", plan.case_id, plan.phase, plan.phase_run, plan.manifest_run);
    sf.push_back({"source_id", jstr(plan.source_id)});
    sf.push_back({"prompt_file", jstr(plan.prompt_file)});
    sf.push_back({"prompt_tokens", jint(n_prompt)});
    sf.push_back({"expected_tokens", jint(plan.expected_tokens)});
    sf.push_back({"n_predict", jint(n_predict)});
    sf.push_back({"t0_us", jint(ggml_time_us())});
    emit_object(std::move(sf));

    // prompt batch: only the last prompt token produces logits
    // 构造 prefill batch：n_prompt 个 token 全部参与计算（位置 0..n_prompt-1），
    // 但只有最后一个 token 需要 logits（后续采样只用它）——省掉前面
    // n_prompt-1 次无用的 logits 计算
    llama_batch batch = llama_batch_init(n_prompt, 0, 1);
    for (int i = 0; i < n_prompt; ++i)
    {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 0;
    }
    batch.logits[n_prompt - 1] = 1;

    // t0 = 本轮计时起点（KV 已清空、batch 已就绪、run_start 已发出）
    const int64_t t0 = ggml_time_us();

    // prefill, sliced into n_batch chunks (same pattern as batched-bench)
    // prefill 按 n_batch=2048 分片 decode（llama_decode 一次最多处理 n_batch
    // 个 token，n_ubatch=512 决定物理片大小）；每片后 synchronize 取实际完成点
    // prompt过长的话，会分多片 decode，直到全部 n_prompt 个 token 都 decode 完毕
    for (int i = 0; i < n_prompt; i += (int)args.n_batch)
    {
        const int32_t n_tokens = (int32_t)std::min(args.n_batch, (uint32_t)(n_prompt - i));
        llama_batch view = {
            n_tokens,
            batch.token + i,
            nullptr,
            batch.pos + i,
            batch.n_seq_id + i,
            batch.seq_id + i,
            batch.logits + i,
        };
        if (llama_decode(ctx, view) != 0)
        {
            llama_batch_free(batch);
            err = "prefill decode failed";
            return false;
        }
        llama_synchronize(ctx);
    }
    const int64_t t_pp_end = ggml_time_us();

    // first generated token: produced by the prefill pass + sampling
    // （serving 引擎口径）prefill 的 forward 已算出第 1 个输出 token 的 logits，
    // 采样后首 token 即可得，不需要额外的一次 decode；common_sampler_sample
    // 内部自带同步（temp<=0 时为纯 greedy/argmax，确定性输出）
    llama_token new_token = common_sampler_sample(smpl, ctx, -1);
    // TTFT 计时点：采样完成、首 token 可用的时刻（采样本身也是首 token 产出的
    // 组成部分，vLLM/SGLang 的 TTFT 同样止于采样后 token 发出）
    const int64_t t_first_avail = ggml_time_us();
    if (llama_vocab_is_eog(vocab, new_token))
    {
        // unreachable with the EOG bias; guarded for safety
        // 理论上不可达（EOG 已被 -INFINITY bias 抑制），仅防御：
        // 首 token 为 EOG 也算"收到首 token"，TTFT 口径不变
        llama_batch_free(batch);
        res.prompt_eval_ms = (t_pp_end - t0) / 1000.0;
        res.ttft_ms = (t_first_avail - t0) / 1000.0;
        res.e2e_ms = (ggml_time_us() - t0) / 1000.0;
        res.prompt_tokens = n_prompt;
        res.expected_tokens = plan.expected_tokens;
        res.tokens_generated = 0;
        res.ok = true;
        return true;
    }

    // t_tg_start = 生成阶段起点（第 1 个输出 token 开始 decode）
    const int64_t t_tg_start = ggml_time_us();
    llama_batch one = llama_batch_get_one(&new_token, 1);
    if (llama_decode(ctx, one) != 0)
    {
        llama_batch_free(batch);
        err = "decode failed (first generated token)";
        return false;
    }
    llama_synchronize(ctx);
    // end of the first generated token decode: this anchors t_tg_end (the loop
    // below keeps updating t_tg_end and it ends up at the last token)
    // 第 1 个输出 token decode 完成的时刻 —— 作为 t_tg_end 的初始锚点（循环里
    // 每次 decode 后都会更新，最终停在最后一个 token 上）。
    // 注意：这里不再是 TTFT 的计时点，TTFT 取 t_first_avail（采样完即可得）
    const int64_t t_first_end = ggml_time_us();
    int64_t t_tg_end = t_first_end;

    // 循环解码剩余 token：每轮"采样 → decode → synchronize → 记时刻"，
    // 直到生成满 n_predict 个（EOG 被抑制，正常情况下不会提前 break）
    int n_generated = 1;
    while (n_generated < n_predict)
    {
        new_token = common_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, new_token))
        {
            break;
        }
        llama_batch one2 = llama_batch_get_one(&new_token, 1);
        if (llama_decode(ctx, one2) != 0)
        {
            llama_batch_free(batch);
            err = "decode failed (generated token " + std::to_string(n_generated + 1) + ")";
            return false;
        }
        llama_synchronize(ctx);
        t_tg_end = ggml_time_us();
        ++n_generated;
    }
    const int64_t t_end = ggml_time_us();

    // 四项耗时：全部由 ggml_time_us 差值换算（μs → ms）。
    // TTFT = prefill + 采样（首 token 的 logits 由 prefill forward 直接产出，
    // 采样完即可得，不包含任何额外 decode —— vLLM/SGLang serving 口径）。
    // 【注意】这里只是把结果"算"进 res 结构体（纯内存，不输出）；
    // 真正的输出点是 main() 主循环里的 result 事件（emit_object，见下文）。
    res.prompt_eval_ms = (t_pp_end - t0) / 1000.0;
    res.ttft_ms = (t_first_avail - t0) / 1000.0;
    res.generation_eval_ms = (t_tg_end - t_tg_start) / 1000.0;
    res.e2e_ms = (t_end - t0) / 1000.0;
    res.prompt_tokens = n_prompt;
    res.expected_tokens = plan.expected_tokens;
    res.tokens_generated = n_generated;
    res.ok = true;

    llama_batch_free(batch);
    return true;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv)
{
    std::setlocale(LC_NUMERIC, "C"); // 数值格式固定为 C 风格（小数点，不受 locale 影响）

    BenchArgs args;
    if (parse_args(argc, argv, args) != 0)
    {
        return 2;
    }
    if (args.show_version)
    {
        // --version：打印 llama.cpp 编译/系统信息后退出（bench_test.py 用来记录版本）
        printf("%s", llama_print_system_info());
        return 0;
    }
    if (args.model_path.empty())
    {
        print_usage(argv[0]);
        return 2;
    }
    if (!args.dump_template.empty() && args.plan_path.empty())
    {
        // debug mode: apply the chat template to one prompt file and print the result
        // 调试模式：不推理，只把 chat 模板应用到指定 prompt 文件并打印
        // （套模板后的完整文本 + token 数），用于核对模板口径/分词结果
        ggml_backend_load_all();

        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = 0; // 调试模式只加载权重用于模板/分词，不卸载到 GPU
        llama_model *model = llama_model_load_from_file(args.model_path.c_str(), mparams);
        if (model == nullptr)
        {
            fprintf(stderr, "bench_run: failed to load model: %s\n", args.model_path.c_str());
            return 1;
        }
        std::ifstream fin(args.dump_template, std::ios::binary);
        std::string raw((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
        std::string out, err;
        if (!apply_chat_template(model, raw, out, err))
        {
            fprintf(stderr, "bench_run: %s\n", err.c_str());
            llama_model_free(model);
            return 1;
        }
        std::vector<llama_token> tokens;
        if (!tokenize_prompt(llama_model_get_vocab(model), out, tokens))
        {
            fprintf(stderr, "bench_run: tokenize failed\n");
            llama_model_free(model);
            return 1;
        }
        fwrite(out.data(), 1, out.size(), stdout);
        printf("\n--TOKENS: %zu--\n", tokens.size());
        llama_model_free(model);
        return 0;
    }
    if (args.plan_path.empty())
    {
        print_usage(argv[0]);
        return 2;
    }

    // cache type validation
    auto valid_cache_type = [](const std::string &t)
    {
        return t == "f32" || t == "f16" || t == "bf16" || t == "q8_0" || t == "q4_0" || t == "q4_1" || t == "iq4_nl" || t == "q5_0" || t == "q5_1";
    };
    if (!valid_cache_type(args.cache_type_k) || !valid_cache_type(args.cache_type_v))
    {
        fprintf(stderr, "bench_run: invalid cache type (k = '%s', v = '%s')\n",
                args.cache_type_k.c_str(), args.cache_type_v.c_str());
        return 2;
    }
    if (args.flash_attn != "auto" && args.flash_attn != "disabled" && args.flash_attn != "enabled" && args.flash_attn != "off" && args.flash_attn != "on")
    {
        fprintf(stderr, "bench_run: invalid --flash-attn value: '%s'\n", args.flash_attn.c_str());
        return 2;
    }

    std::vector<RunPlan> runs;
    if (read_plan(args.plan_path, runs) != 0)
    {
        return 2;
    }

    // 加载全部后端（CUDA 等），然后加载模型：
    // 整个过程只发生一次 —— 这就是"warmup 吸收冷启动"的前提：
    // 模型文件读取、CUDA 上下文创建、kernel 初始化全部只做一遍
    ggml_backend_load_all();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = args.n_gpu_layers; // 层卸载到 GPU（99 = 全部）

    llama_model *model = llama_model_load_from_file(args.model_path.c_str(), mparams);
    if (model == nullptr)
    {
        fprintf(stderr, "bench_run: failed to load model: %s\n", args.model_path.c_str());
        return 1;
    }

    const llama_vocab *vocab = llama_model_get_vocab(model);

    // 上下文参数：n_ctx=262144（KV 容量）、batch/ubatch、KV 缓存精度、
    // Flash Attention 开关 —— 全部来自 Python 层透传的 case 配置
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = args.n_ctx;
    cparams.n_batch = args.n_batch;
    cparams.n_ubatch = args.n_ubatch;
    cparams.type_k = parse_cache_type(args.cache_type_k);
    cparams.type_v = parse_cache_type(args.cache_type_v);
    cparams.flash_attn_type = parse_flash_attn(args.flash_attn);

    llama_context *ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr)
    {
        fprintf(stderr, "bench_run: failed to create the llama_context\n");
        llama_model_free(model);
        return 1;
    }

    // threadpool with the requested polling level (replicates llama-cli --poll)
    // 附加带指定 poll 级别的 CPU 线程池（复刻 --poll；失败仅告警，不阻止运行）
    ggml_threadpool_t tp = create_threadpool(args.n_threads, args.poll);
    if (tp != nullptr)
    {
        llama_attach_threadpool(ctx, tp, tp);
    }

    // sampler: llama-cli defaults (greedy at temp <= 0) + EOG suppression
    // 采样器：temp<=0 时纯 greedy（确定性输出）+ EOG 抑制（见 add_eog_bias）
    common_params_sampling sparams;
    sparams.temp = args.temp;
    sparams.seed = args.seed;
    add_eog_bias(model, sparams);
    common_sampler *smpl = common_sampler_init(model, sparams);
    if (smpl == nullptr)
    {
        fprintf(stderr, "bench_run: failed to initialize the sampler\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    // model_loaded 事件：模型与运行时参数快照（Python 层据此记录版本与配置）；
    // 这是事件流的第一行，之后才是逐轮的 run_start/result
    Fields mf;
    mf.push_back({"event", jstr("model_loaded")});
    mf.push_back({"model", jstr(args.model_path)});
    mf.push_back({"llama_version", jstr(llama_print_system_info())});
    mf.push_back({"n_ctx", jint(args.n_ctx)});
    mf.push_back({"n_batch", jint(args.n_batch)});
    mf.push_back({"n_ubatch", jint(args.n_ubatch)});
    mf.push_back({"cache_type_k", jstr(args.cache_type_k)});
    mf.push_back({"cache_type_v", jstr(args.cache_type_v)});
    mf.push_back({"flash_attn", jstr(args.flash_attn)});
    mf.push_back({"temp", jnum(args.temp)});
    mf.push_back({"seed", jint(args.seed)});
    mf.push_back({"poll", jint(args.poll)});
    emit_object(std::move(mf));

    int failed = 0;
    // 主循环：按 plan 顺序逐轮执行（同一进程、同一 context、不重启）。
    // 单轮失败只发 error 事件并计数，继续跑剩余轮次；全部结束后统一退出码
    for (size_t i = 0; i < runs.size(); ++i)
    {
        const RunPlan &r = runs[i];
        fprintf(stderr, "bench_run: [%zu/%zu] %s %d/%d (case %s, n_predict %d)\n",
                i + 1, runs.size(), r.phase.c_str(), r.phase_run, r.manifest_run,
                r.case_id.c_str(), r.n_predict);

        RunResult res;
        std::string run_err;
        if (!run_one(ctx, vocab, smpl, r, args, res, run_err))
        {
            // 运行前失败（prompt 文件缺失/模板失败/超 n_ctx 等）：发 error 事件
            ++failed;
            fprintf(stderr, "bench_run: run %d/%zu failed: %s\n", r.manifest_run, runs.size(),
                    run_err.empty() ? "unknown error" : run_err.c_str());
            Fields ef = common_fields("error", r.case_id, r.phase, r.phase_run, r.manifest_run);
            ef.push_back({"fatal", "false"});
            ef.push_back({"message", jstr(run_err.empty() ? "bench_run: run failed" : run_err)});
            emit_object(std::move(ef));
            continue;
        }

        // 两处一致性校验（不致命，只发 warn）：
        // ① 实际 prompt token 数与 metadata 的 expected_tokens 不一致
        //   （正常应完全相等；不等说明模板口径或分词结果漂移，结果可信度下降）；
        // ② 生成 token 数不足 n_predict（EOG 被抑制后理论上不会发生）
        if (res.expected_tokens >= 0 && res.prompt_tokens != res.expected_tokens)
        {
            Fields wf = common_fields("warn", r.case_id, r.phase, r.phase_run, r.manifest_run);
            wf.push_back({"message", jstr("prompt_tokens (" + std::to_string(res.prompt_tokens) + ") != expected_tokens (" + std::to_string(res.expected_tokens) + ")")});
            emit_object(std::move(wf));
        }
        if (res.tokens_generated != r.n_predict)
        {
            Fields wf = common_fields("warn", r.case_id, r.phase, r.phase_run, r.manifest_run);
            wf.push_back({"message", jstr("generation stopped early: tokens_generated (" + std::to_string(res.tokens_generated) + ") != n_predict (" + std::to_string(r.n_predict) + ")")});
            emit_object(std::move(wf));
        }

        // ═══════════════════════════════════════════════════════════════════
        // result 事件：本轮的【最终结果输出点】——所有统计数据的源头。
        //
        // 数据流全貌（C++ → Python 共四步）：
        //   ① run_one 内部只把四项耗时"算"进 res 结构体（纯内存，不输出）；
        //   ② 到这里才真正"输出"：把 res + 身份/计数信息拼成一行 JSON
        //      经 emit_object（puts + fflush 实时刷出）写到 stdout；
        //   ③ bench_test.py 的 StreamReader 逐行解析本事件 → 填每轮 record
        //      写 runs.jsonl；tpot_ms_measured = itl_ms_measured =
        //      1000/generation_tps 等派生指标由 Python 层算；
        //   ④ summarize_case 再对正式样本做 mean/stdev/min/max 聚合
        //      写入 summary.json / csv。
        //
        // 字段分四组：
        //   ① 身份信息（common_fields）：event / case_id / phase / phase_run / manifest_run
        //   ② 输入输出规模：模型用了多少输入 token、计划多少个输出、实际生了多少个
        //   ③ 四项实测耗时（单位 ms，来自 run_one 的 res）
        //   ④ 两个 TPS：由②③当场换算（除以 1000 把 ms 变成秒）
        // ═══════════════════════════════════════════════════════════════════
        Fields rf = common_fields("result", r.case_id, r.phase, r.phase_run, r.manifest_run);
        // ── ② 输入输出规模 ─────────────────────────────────────────────
        rf.push_back({"source_id", jstr(r.source_id)});              // LongBench-v2 源样本 ID
        rf.push_back({"prompt_file", jstr(r.prompt_file)});          // 该轮用的 prompt 文件
        rf.push_back({"prompt_tokens", jint(res.prompt_tokens)});    // 模板后实际输入 token 数
        rf.push_back({"expected_tokens", jint(res.expected_tokens)});// metadata 预期值（校验用）
        rf.push_back({"n_predict", jint(r.n_predict)});              // 计划生成的输出 token 数
        rf.push_back({"tokens_generated", jint(res.tokens_generated)}); // 实际生成的个数（正常 = n_predict）
        // ── ③ 四项实测耗时（ms）───────────────────────────────────────
        // prompt_eval_ms       = 纯 prefill（t_pp_end − t0）
        // ttft_ms              = prefill + 采样（t_first_avail − t0；首 token 由 prefill
        //                        forward 直接产出，无额外 decode —— vLLM/SGLang 口径）
        // generation_eval_ms   = 生成期，从第 1 个输出 token 起算（t_tg_end − t_tg_start）
        // e2e_ms               = 整轮墙钟（KV 清空后开始计时，t_end − t0）
        rf.push_back({"prompt_eval_ms_measured", jnum(res.prompt_eval_ms)});
        rf.push_back({"ttft_ms_measured", jnum(res.ttft_ms)});
        rf.push_back({"generation_eval_ms_measured", jnum(res.generation_eval_ms)});
        rf.push_back({"e2e_ms_measured", jnum(res.e2e_ms)});
        // ── ④ 两个 TPS：token 数 ÷ (耗时 ms ÷ 1000) ────────────────────
        // 除零保护：耗时为 0（异常轮）记 0.0 而不是 NaN/Inf 污染统计
        if (res.prompt_eval_ms > 0.0)
        {
            // prefill 吞吐 = 输入 token 数 ÷ 纯 prefill 耗时
            rf.push_back({"prompt_tps_measured", jnum(res.prompt_tokens / (res.prompt_eval_ms / 1000.0))});
        }
        else
        {
            rf.push_back({"prompt_tps_measured", jnum(0.0)});
        }
        if (res.generation_eval_ms > 0.0)
        {
            // 生成吞吐 = 输出 token 数 ÷ 生成期耗时（分母不含 prefill）。
            // Python 层用它的倒数派生 tpot_ms_measured = itl_ms_measured = 1000/generation_tps
            rf.push_back({"generation_tps_measured", jnum(res.tokens_generated / (res.generation_eval_ms / 1000.0))});
        }
        else
        {
            rf.push_back({"generation_tps_measured", jnum(0.0)});
        }
        // 最后一步：整行 JSON 写入 stdout 并 fflush，Python 侧 StreamReader 实时收到
        emit_object(std::move(rf));
    }

    // all_done：事件流的最后一行，报告总轮数与失败数。
    // Python 层以它为"进程正常结束"的信号（配合退出码判断成败）
    Fields df;
    df.push_back({"event", jstr("all_done")});
    df.push_back({"total_runs", jint((int64_t)runs.size())});
    df.push_back({"failed_runs", jint(failed)});
    emit_object(std::move(df));

    // 释放资源（进程即将退出，主要保证 CUDA 上下文干净释放、不污染下个进程）
    common_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    // 退出码：0 = 全部成功；1 = 至少一轮失败（事件流中 error 可查）；2 = 用法/计划错误
    return failed > 0 ? 1 : 0;
}
