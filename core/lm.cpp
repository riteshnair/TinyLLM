#include "lm/lm.h"

#include <chrono>
#include <cstring>
#include <cstdlib>

struct lm_runtime {
    lm_config config;
    lm_probe_sink sink;
    void *sink_user;
    uint64_t next_trace_id;
};

static uint64_t now_ns() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

static int text_eq(const char *a, const char *b) {
    return a != nullptr && b != nullptr && std::strcmp(a, b) == 0;
}

static uint64_t parse_u64(const char *text, int *ok) {
    char *end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 0);
    *ok = text != nullptr && end != text && *end == '\0';
    return static_cast<uint64_t>(value);
}

static lm_status parse_backend(const char *text, lm_backend_kind *out) {
    if (text_eq(text, "auto")) *out = LM_BACKEND_AUTO;
    else if (text_eq(text, "cpu")) *out = LM_BACKEND_CPU;
    else if (text_eq(text, "vulkan")) *out = LM_BACKEND_VULKAN;
    else if (text_eq(text, "rocr")) *out = LM_BACKEND_ROCR;
    else if (text_eq(text, "rocm")) *out = LM_BACKEND_ROCM;
    else if (text_eq(text, "cuda")) *out = LM_BACKEND_CUDA;
    else if (text_eq(text, "openvino")) *out = LM_BACKEND_OPENVINO;
    else if (text_eq(text, "directml")) *out = LM_BACKEND_DIRECTML;
    else return LM_ERR_PARSE;
    return LM_OK;
}

static lm_status parse_load(const char *text, lm_load_mode *out) {
    if (text_eq(text, "eager")) *out = LM_LOAD_EAGER;
    else if (text_eq(text, "mmap")) *out = LM_LOAD_MMAP;
    else if (text_eq(text, "lazy")) *out = LM_LOAD_LAZY;
    else if (text_eq(text, "stream")) *out = LM_LOAD_STREAM;
    else return LM_ERR_PARSE;
    return LM_OK;
}

static lm_status parse_weight_policy(const char *text, lm_weight_policy *out) {
    if (text_eq(text, "preserve")) *out = LM_WEIGHT_PRESERVE;
    else if (text_eq(text, "quantize-cache")) *out = LM_WEIGHT_QUANTIZE_CACHE;
    else return LM_ERR_PARSE;
    return LM_OK;
}

static lm_status parse_kv(const char *text, lm_kv_dtype *out) {
    if (text_eq(text, "f16")) *out = LM_KV_F16;
    else if (text_eq(text, "bf16")) *out = LM_KV_BF16;
    else if (text_eq(text, "q8")) *out = LM_KV_Q8;
    else if (text_eq(text, "q6")) *out = LM_KV_Q6;
    else if (text_eq(text, "q4")) *out = LM_KV_Q4;
    else return LM_ERR_PARSE;
    return LM_OK;
}

void lm_config_init(lm_config *config) {
    if (!config) return;
    std::memset(config, 0, sizeof(*config));
    config->backend = LM_BACKEND_AUTO;
    config->resolved_backend = LM_BACKEND_AUTO;
    config->load_mode = LM_LOAD_MMAP;
    config->weight_policy = LM_WEIGHT_PRESERVE;
    config->kv_dtype = LM_KV_F16;
    config->context_tokens = 4096u;
    config->threads = 1u;
    config->kv_page_tokens = 32u;
    config->prefetch = 1u;
    config->deterministic = 0u;
}

static lm_status take_u64(const char *arg, int *index, int argc, char **argv, uint64_t *out) {
    if (*index + 1 >= argc) return LM_ERR_ARGUMENT;
    int ok = 0;
    *out = parse_u64(argv[++(*index)], &ok);
    (void)arg;
    return ok ? LM_OK : LM_ERR_PARSE;
}

lm_status lm_config_parse_argv(lm_config *config, int argc, char **argv, const char **bad_argument) {
    if (!config || argc < 0 || (argc > 0 && !argv)) return LM_ERR_ARGUMENT;
    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        lm_status status = LM_OK;
        if (text_eq(arg, "--backend")) {
            if (++i >= argc) status = LM_ERR_ARGUMENT;
            else status = parse_backend(argv[i], &config->backend);
        } else if (text_eq(arg, "--load")) {
            if (++i >= argc) status = LM_ERR_ARGUMENT;
            else status = parse_load(argv[i], &config->load_mode);
        } else if (text_eq(arg, "--weights")) {
            if (++i >= argc) status = LM_ERR_ARGUMENT;
            else status = parse_weight_policy(argv[i], &config->weight_policy);
        } else if (text_eq(arg, "--kv-dtype")) {
            if (++i >= argc) status = LM_ERR_ARGUMENT;
            else status = parse_kv(argv[i], &config->kv_dtype);
        } else if (text_eq(arg, "--device")) {
            uint64_t value = 0; status = take_u64(arg, &i, argc, argv, &value);
            if (status == LM_OK && value > UINT32_MAX) status = LM_ERR_RANGE;
            if (status == LM_OK) config->device_index = static_cast<uint32_t>(value);
        } else if (text_eq(arg, "--context")) {
            uint64_t value = 0; status = take_u64(arg, &i, argc, argv, &value);
            if (status == LM_OK && (value == 0 || value > UINT32_MAX)) status = LM_ERR_RANGE;
            if (status == LM_OK) config->context_tokens = static_cast<uint32_t>(value);
        } else if (text_eq(arg, "--threads")) {
            uint64_t value = 0; status = take_u64(arg, &i, argc, argv, &value);
            if (status == LM_OK && (value == 0 || value > UINT32_MAX)) status = LM_ERR_RANGE;
            if (status == LM_OK) config->threads = static_cast<uint32_t>(value);
        } else if (text_eq(arg, "--kv-page-tokens")) {
            uint64_t value = 0; status = take_u64(arg, &i, argc, argv, &value);
            if (status == LM_OK && (value == 0 || value > UINT32_MAX)) status = LM_ERR_RANGE;
            if (status == LM_OK) config->kv_page_tokens = static_cast<uint32_t>(value);
        } else if (text_eq(arg, "--vram-limit")) {
            status = take_u64(arg, &i, argc, argv, &config->vram_limit_bytes);
        } else if (text_eq(arg, "--host-cache-bytes")) {
            status = take_u64(arg, &i, argc, argv, &config->host_cache_bytes);
        } else if (text_eq(arg, "--pinned-cache-bytes")) {
            status = take_u64(arg, &i, argc, argv, &config->pinned_cache_bytes);
        } else if (text_eq(arg, "--device-cache-bytes")) {
            status = take_u64(arg, &i, argc, argv, &config->device_cache_bytes);
        } else if (text_eq(arg, "--model")) {
            if (++i >= argc) status = LM_ERR_ARGUMENT;
            else if (std::strlen(argv[i]) >= LM_PATH_MAX) status = LM_ERR_RANGE;
            else std::strncpy(config->model_path, argv[i], LM_PATH_MAX - 1u);
        } else if (text_eq(arg, "--no-prefetch")) {
            config->prefetch = 0u;
        } else if (text_eq(arg, "--trace")) {
            config->trace = 1u;
        } else if (text_eq(arg, "--deterministic")) {
            config->deterministic = 1u;
        } else if (text_eq(arg, "--help") || text_eq(arg, "--dump-config") || text_eq(arg, "--list-devices")) {
            continue;
        } else {
            status = LM_ERR_PARSE;
        }
        if (status != LM_OK) {
            if (bad_argument) *bad_argument = arg;
            return status;
        }
    }
    return LM_OK;
}

const char *lm_status_name(lm_status status) {
    switch (status) {
        case LM_OK: return "ok";
        case LM_ERR_ARGUMENT: return "argument";
        case LM_ERR_PARSE: return "parse";
        case LM_ERR_RANGE: return "range";
        case LM_ERR_UNSUPPORTED: return "unsupported";
        case LM_ERR_CAPACITY: return "capacity";
        case LM_ERR_STATE: return "state";
        case LM_ERR_IO: return "io";
        default: return "unknown";
    }
}

const char *lm_backend_name(lm_backend_kind backend) {
    switch (backend) {
        case LM_BACKEND_AUTO: return "auto";
        case LM_BACKEND_CPU: return "cpu";
        case LM_BACKEND_VULKAN: return "vulkan";
        case LM_BACKEND_ROCR: return "rocr";
        case LM_BACKEND_ROCM: return "rocm";
        case LM_BACKEND_CUDA: return "cuda";
        case LM_BACKEND_OPENVINO: return "openvino";
        case LM_BACKEND_DIRECTML: return "directml";
        default: return "unknown";
    }
}

const char *lm_load_mode_name(lm_load_mode mode) {
    switch (mode) {
        case LM_LOAD_EAGER: return "eager";
        case LM_LOAD_MMAP: return "mmap";
        case LM_LOAD_LAZY: return "lazy";
        case LM_LOAD_STREAM: return "stream";
        default: return "unknown";
    }
}

const char *lm_weight_policy_name(lm_weight_policy policy) {
    switch (policy) {
        case LM_WEIGHT_PRESERVE: return "preserve";
        case LM_WEIGHT_QUANTIZE_CACHE: return "quantize-cache";
        default: return "unknown";
    }
}

const char *lm_kv_dtype_name(lm_kv_dtype dtype) {
    switch (dtype) {
        case LM_KV_F16: return "f16";
        case LM_KV_BF16: return "bf16";
        case LM_KV_Q8: return "q8";
        case LM_KV_Q6: return "q6";
        case LM_KV_Q4: return "q4";
        default: return "unknown";
    }
}

lm_status lm_runtime_create(const lm_config *config, lm_runtime **out_runtime) {
    if (!config || !out_runtime) return LM_ERR_ARGUMENT;
    *out_runtime = nullptr;
    lm_runtime *runtime = static_cast<lm_runtime *>(std::calloc(1u, sizeof(*runtime)));
    if (!runtime) return LM_ERR_CAPACITY;
    runtime->config = *config;
    runtime->config.resolved_backend = config->backend == LM_BACKEND_AUTO ? LM_BACKEND_CPU : config->backend;
    runtime->next_trace_id = 1u;
    if (config->weight_policy == LM_WEIGHT_QUANTIZE_CACHE) {
        std::free(runtime);
        return LM_ERR_UNSUPPORTED;
    }
    if (runtime->config.resolved_backend == LM_BACKEND_VULKAN) {
        uint32_t device_count = 0u;
        const lm_status discovered = lm_vulkan_device_count(&device_count);
        if (discovered != LM_OK || device_count == 0u || config->device_index >= device_count) {
            std::free(runtime);
            return discovered != LM_OK ? discovered : LM_ERR_UNSUPPORTED;
        }
    } else if (runtime->config.resolved_backend != LM_BACKEND_CPU) {
        std::free(runtime);
        return LM_ERR_UNSUPPORTED;
    }
    *out_runtime = runtime;
    return LM_OK;
}

void lm_runtime_destroy(lm_runtime *runtime) { std::free(runtime); }

void lm_runtime_set_probe_sink(lm_runtime *runtime, lm_probe_sink sink, void *user) {
    if (!runtime) return;
    runtime->sink = sink;
    runtime->sink_user = user;
}

lm_status lm_runtime_emit_probe(lm_runtime *runtime, const lm_probe *probe) {
    if (!runtime || !probe) return LM_ERR_ARGUMENT;
    if (runtime->sink) runtime->sink(runtime->sink_user, probe);
    return LM_OK;
}

lm_status lm_runtime_dump_config(const lm_runtime *runtime, FILE *out) {
    if (!runtime || !out) return LM_ERR_ARGUMENT;
    const lm_config &c = runtime->config;
    fprintf(out, "backend=%s\n", lm_backend_name(c.resolved_backend));
    fprintf(out, "requested_backend=%s\n", lm_backend_name(c.backend));
    fprintf(out, "device=%u\n", c.device_index);
    fprintf(out, "context=%u\n", c.context_tokens);
    fprintf(out, "threads=%u\n", c.threads);
    fprintf(out, "load=%s\n", lm_load_mode_name(c.load_mode));
    fprintf(out, "weight_policy=%s\n", lm_weight_policy_name(c.weight_policy));
    fprintf(out, "kv_dtype=%s\n", lm_kv_dtype_name(c.kv_dtype));
    fprintf(out, "kv_page_tokens=%u\n", c.kv_page_tokens);
    fprintf(out, "prefetch=%s\n", c.prefetch ? "on" : "off");
    fprintf(out, "trace=%s\n", c.trace ? "on" : "off");
    fprintf(out, "deterministic=%s\n", c.deterministic ? "on" : "off");
    fprintf(out, "model=%s\n", c.model_path[0] ? c.model_path : "(none)");
    fprintf(out, "control_plane_budget=41943040\n");
    return LM_OK;
}

void lm_probe_emit_runtime(lm_runtime *runtime, uint32_t stage, uint32_t kind, uint32_t bytes) {
    if (!runtime || !runtime->config.trace) return;
    lm_probe probe{};
    probe.trace_id = runtime->next_trace_id++;
    probe.kind = kind;
    probe.stage = stage;
    probe.bytes = bytes;
    probe.timestamp_ns = now_ns();
    (void)lm_runtime_emit_probe(runtime, &probe);
}
