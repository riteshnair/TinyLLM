#include "lm/lm.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <fstream>
#if defined(__unix__) || defined(__APPLE__)
#include <arpa/inet.h>
#include <cerrno>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

static void print_help() {
    std::puts("tiny-lm control-plane slice");
    std::puts("usage: tiny-lm [options]");
    std::puts("  --backend auto|cpu|vulkan|rocr|rocm|cuda|openvino|directml");
    std::puts("  --model PATH --context N --threads N --device N");
    std::puts("  --load eager|mmap|lazy|stream --kv-dtype f16|bf16|q8|q6|q4");
    std::puts("  --kv-page-tokens N --trace --deterministic --no-prefetch");
    std::puts("  --dump-config --dry-run --list-devices --help");
    std::puts("  --generate --prompt TEXT --max-new-tokens N");
    std::puts("  --sampling greedy|top-k|top-p|typical --top-k N --top-p P --min-p P");
    std::puts("  --attention-window N --rope-scale S --prefill-chunk-tokens N");
    std::puts("  --typical-p P --temperature T --repetition-penalty P --frequency-penalty P");
    std::puts("  --presence-penalty P --seed N");
    std::puts("  --tokenizer PATH --config PATH (required for standard HF SafeTensors)");
    std::puts("  --server --port N --model PATH");
}

static void text_probe(void *, const lm_probe *probe) {
    std::printf("probe trace=%llu stage=%u kind=%u bytes=%u\n",
                static_cast<unsigned long long>(probe->trace_id), probe->stage,
                probe->kind, probe->bytes);
}

static bool parse_u32_limit(const char *text, uint32_t *out, unsigned long limit) {
    if (!text || !out || text[0] == '\0') return false;
    char *end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (!end || *end != '\0' || value > limit) return false;
    *out = static_cast<uint32_t>(value);
    return true;
}

static bool parse_bounded_u32(const char *text, uint32_t *out) {
    return parse_u32_limit(text, out, 1024u);
}

static bool parse_finite_float(const char *text, float *out) {
    if (!text || !out || text[0] == '\0') return false;
    char *end = nullptr;
    const float value = std::strtof(text, &end);
    if (!end || *end != '\0' || !std::isfinite(value)) return false;
    *out = value;
    return true;
}

static bool matrix_shape_ok(const lm_model_tensor_info &tensor,
                            uint32_t rows, uint32_t columns, bool raw_gguf_axes) {
    if (tensor.rank != 2u) return false;
    if (tensor.dims[0] == rows && tensor.dims[1] == columns) return true;
    return raw_gguf_axes && tensor.dims[0] == columns && tensor.dims[1] == rows;
}

static std::string resolve_shader_path(const char *name) {
    if (!name) return std::string();
    std::ifstream direct(name, std::ios::binary);
    if (direct.good()) return std::string(name);
#if defined(__unix__) || defined(__APPLE__)
    char executable[4096] = {};
    const ssize_t length = readlink("/proc/self/exe", executable, sizeof(executable) - 1u);
    if (length > 0) {
        executable[length] = '\0';
        const std::string path(executable);
        const size_t slash = path.find_last_of('/');
        if (slash != std::string::npos) {
            const std::string candidate = path.substr(0u, slash + 1u) + name;
            std::ifstream beside_executable(candidate, std::ios::binary);
            if (beside_executable.good()) return candidate;
        }
    }
#endif
    return std::string(name);
}

static lm_status open_cli_model(const char *path, lm_model_file **out_model, char *error, size_t error_capacity) {
    if (!path || !out_model) return LM_ERR_ARGUMENT;
    const std::string source(path);
    const size_t of = source.find("-of-");
    const size_t dash = of == std::string::npos ? std::string::npos : source.rfind('-', of - 1u);
    if (of == std::string::npos || dash == std::string::npos) return lm_model_open(path, out_model, error, error_capacity);
    const std::string index_text = source.substr(dash + 1u, of - dash - 1u);
    const size_t count_begin = of + 4u;
    const size_t count_end = source.find_first_not_of("0123456789", count_begin);
    if (index_text.empty() || count_end == std::string::npos || count_end == count_begin) return LM_ERR_PARSE;
    char *index_end = nullptr;
    char *count_end_ptr = nullptr;
    const unsigned long index = std::strtoul(index_text.c_str(), &index_end, 10);
    const std::string count_text = source.substr(count_begin, count_end - count_begin);
    const unsigned long count = std::strtoul(count_text.c_str(), &count_end_ptr, 10);
    if (index_end != index_text.c_str() + index_text.size() || count_end_ptr != count_text.c_str() + count_text.size() ||
        index != 1u || count < 2u || count > 1024u) return LM_ERR_UNSUPPORTED;
    const std::string prefix = source.substr(0u, dash + 1u);
    const std::string suffix = source.substr(count_end);
    std::vector<std::string> paths;
    std::vector<const char *> path_ptrs;
    paths.reserve(count); path_ptrs.reserve(count);
    const size_t width = index_text.size() > count_text.size() ? index_text.size() : count_text.size();
    for (unsigned long part = 1u; part <= count; ++part) {
        char part_index[32] = {}; char part_count[32] = {};
        std::snprintf(part_index, sizeof(part_index), "%0*lu", static_cast<int>(width), part);
        std::snprintf(part_count, sizeof(part_count), "%0*lu", static_cast<int>(width), count);
        paths.emplace_back(prefix + part_index + "-of-" + part_count + suffix);
    }
    for (const std::string &part : paths) path_ptrs.push_back(part.c_str());
    return lm_model_open_sharded(path_ptrs.data(), path_ptrs.size(), out_model, error, error_capacity);
}

struct http_stream_state {
    int socket_fd;
    const lm_model_file *model;
};

struct generation_assets {
    const char *tokenizer_path;
    const char *config_path;
    lm_sampling_config sampling;
    uint32_t attention_window;
    float rope_scale;
    uint32_t prefill_chunk_tokens;
};

static lm_status run_native_generation(const lm_config &config, const generation_assets &assets,
                                       const char *prompt, uint32_t max_new_tokens, char *result,
                                       size_t result_capacity, size_t *result_bytes,
                                       lm_native_token_callback callback = nullptr,
                                       void *callback_user = nullptr) {
    if (!config.model_path[0] || !prompt || max_new_tokens == 0u ||
        (result == nullptr) != (result_bytes == nullptr)) return LM_ERR_ARGUMENT;
    lm_model_file *model = nullptr;
    char error[128] = {};
    lm_status status = open_cli_model(config.model_path, &model, error, sizeof(error));
    if (status != LM_OK) {
        std::fprintf(stderr, "model open failed: %s (%s)\\n", lm_status_name(status), error);
        return status;
    }
    if (callback && callback_user) static_cast<http_stream_state *>(callback_user)->model = model;
    lm_decoder_graph_binding graph{};
    lm_model_info model_info{};
    status = lm_model_get_info(model, &model_info);
    if (status != LM_OK) { lm_model_close(model); return status; }
    const bool safetensors = model_info.format == LM_MODEL_SAFETENSORS;
    if (safetensors) {
        if (!assets.tokenizer_path || !assets.config_path) {
            lm_model_close(model); return LM_ERR_UNSUPPORTED;
        }
        status = lm_model_set_tokenizer_json(model, assets.tokenizer_path, error, sizeof(error));
        if (status == LM_OK) status = lm_model_set_hf_config_json(model, assets.config_path, error, sizeof(error));
        if (status != LM_OK) { lm_model_close(model); return status; }
    }
    status = lm_model_build_llama_graph(model, &graph);
    if (status != LM_OK || graph.layer_count == 0u || graph.layer_count > LM_DECODER_PLAN_MAX_LAYERS) {
        lm_model_close(model);
        return status == LM_OK ? LM_ERR_UNSUPPORTED : status;
    }
    auto descriptor = [model](uint64_t index, lm_model_tensor_info *out) {
        return lm_model_tensor_info_at(model, index, out);
    };
    lm_model_tensor_info embedding{}, output{}, output_norm{};
    lm_model_architecture architecture{};
    if (lm_model_get_architecture(model, &architecture) != LM_OK || architecture.block_count != graph.layer_count ||
        architecture.embedding_length == 0u || architecture.intermediate_length == 0u ||
        architecture.head_count == 0u || architecture.head_count_kv == 0u ||
        architecture.head_count_kv > architecture.head_count ||
        architecture.head_count % architecture.head_count_kv != 0u ||
        architecture.embedding_length % architecture.head_count != 0u) {
        lm_model_close(model);
        return LM_ERR_UNSUPPORTED;
    }
    status = descriptor(graph.token_embedding, &embedding);
    if (status != LM_OK) { lm_model_close(model); return status; }
    status = descriptor(graph.output, &output);
    if (status != LM_OK) { lm_model_close(model); return status; }
    status = descriptor(graph.output_norm, &output_norm);
    if (status != LM_OK) { lm_model_close(model); return status; }
    uint32_t token_count = 0u;
    if (lm_model_token_count(model, &token_count) != LM_OK || token_count == 0u) {
        lm_model_close(model); return LM_ERR_UNSUPPORTED;
    }
    const bool raw_gguf_axes = !safetensors;
    const bool standard_embedding_shape = safetensors && embedding.rank == 2u &&
        embedding.dims[0] == token_count && embedding.dims[1] == architecture.embedding_length;
    const bool output_shape_ok = graph.output_tied ?
        (output.rank == 2u && output.dims[0] == embedding.dims[0] && output.dims[1] == embedding.dims[1]) :
        (safetensors ? (output.rank == 2u && output.dims[0] == token_count && output.dims[1] == architecture.embedding_length) :
         matrix_shape_ok(output, static_cast<uint32_t>(embedding.dims[1]),
                         static_cast<uint32_t>(embedding.dims[0]), raw_gguf_axes));
    if (embedding.rank != 2u || (!safetensors && !matrix_shape_ok(embedding,
        static_cast<uint32_t>(architecture.embedding_length),
        static_cast<uint32_t>(embedding.dims[1]), raw_gguf_axes)) ||
        (!safetensors && embedding.dims[0] != architecture.embedding_length) ||
        (!standard_embedding_shape && safetensors) || !output_shape_ok ||
        output_norm.rank != 1u || output_norm.dims[0] != architecture.embedding_length ||
        output.type != embedding.type ||
        (!safetensors && output_norm.type != LM_DTYPE_F32) ||
        (safetensors && output_norm.type != LM_DTYPE_F32 && output_norm.type != LM_DTYPE_F16 && output_norm.type != LM_DTYPE_BF16)) {
        lm_model_close(model);
        return LM_ERR_UNSUPPORTED;
    }
    const uint32_t matrix_type = embedding.type;
    const uint32_t hidden_size = safetensors ? architecture.embedding_length : static_cast<uint32_t>(embedding.dims[0]);
    const bool scalar16 = safetensors && (matrix_type == LM_DTYPE_F16 || matrix_type == LM_DTYPE_BF16);
    const bool scalar32 = safetensors && matrix_type == LM_DTYPE_F32;
    const bool packed = !safetensors && (matrix_type == 8u || matrix_type == 12u);
    if (!scalar16 && !scalar32 && !packed) {
        lm_model_close(model);
        return LM_ERR_UNSUPPORTED;
    }
    lm_backend_kind execution_backend = config.backend;
    if (execution_backend == LM_BACKEND_AUTO) {
        uint32_t device_count = 0u;
        execution_backend = (!scalar16 && lm_vulkan_device_count(&device_count) == LM_OK && device_count != 0u) ?
                            LM_BACKEND_VULKAN : LM_BACKEND_CPU;
    }
    if (scalar16 && execution_backend != LM_BACKEND_CPU) {
        lm_model_close(model);
        return LM_ERR_UNSUPPORTED;
    }
    std::vector<uint64_t> indices;
    try {
        indices.reserve(2u + static_cast<size_t>(graph.layer_count) * 7u);
        indices.push_back(graph.token_embedding);
        indices.push_back(graph.output);
        for (uint32_t layer_index = 0u; layer_index < graph.layer_count; ++layer_index) {
            const lm_decoder_layer_binding &layer = graph.layers[layer_index];
            const uint64_t layer_indices[] = {layer.attn_q, layer.attn_k, layer.attn_v, layer.attn_output,
                                               layer.ffn_gate, layer.ffn_down, layer.ffn_up};
            lm_model_tensor_info layer_infos[7] = {};
            for (uint32_t i = 0u; i < 7u; ++i) {
                status = descriptor(layer_indices[i], &layer_infos[i]);
                if (status != LM_OK) { lm_model_close(model); return status; }
                if (layer_infos[i].type != matrix_type || layer_infos[i].rank != 2u ||
                (safetensors && layer_infos[i].type != LM_DTYPE_F32 && layer_infos[i].type != LM_DTYPE_F16 && layer_infos[i].type != LM_DTYPE_BF16)) {
                    lm_model_close(model); return LM_ERR_UNSUPPORTED;
                }
            }
            const uint32_t head_dim = hidden_size / architecture.head_count;
            const uint32_t kv_width = head_dim * architecture.head_count_kv;
            if (!matrix_shape_ok(layer_infos[0], hidden_size, hidden_size, raw_gguf_axes) ||
                !matrix_shape_ok(layer_infos[1], kv_width, hidden_size, raw_gguf_axes) ||
                !matrix_shape_ok(layer_infos[2], kv_width, hidden_size, raw_gguf_axes) ||
                !matrix_shape_ok(layer_infos[3], hidden_size, hidden_size, raw_gguf_axes) ||
                !matrix_shape_ok(layer_infos[4], architecture.intermediate_length, hidden_size, raw_gguf_axes) ||
                !matrix_shape_ok(layer_infos[5], hidden_size, architecture.intermediate_length, raw_gguf_axes) ||
                !matrix_shape_ok(layer_infos[6], architecture.intermediate_length, hidden_size, raw_gguf_axes)) {
                lm_model_close(model); return LM_ERR_UNSUPPORTED;
            }
            lm_model_tensor_info attn_norm{};
            lm_model_tensor_info ffn_norm{};
            status = descriptor(layer.attn_norm, &attn_norm);
            if (status != LM_OK || attn_norm.rank != 1u || attn_norm.dims[0] != hidden_size ||
                ((!safetensors && attn_norm.type != LM_DTYPE_F32) ||
                 (safetensors && attn_norm.type != LM_DTYPE_F32 && attn_norm.type != LM_DTYPE_F16 && attn_norm.type != LM_DTYPE_BF16))) {
                lm_model_close(model); return status != LM_OK ? status : LM_ERR_UNSUPPORTED;
            }
            status = descriptor(layer.ffn_norm, &ffn_norm);
            if (status != LM_OK || ffn_norm.rank != 1u || ffn_norm.dims[0] != hidden_size ||
                ((!safetensors && ffn_norm.type != LM_DTYPE_F32) ||
                 (safetensors && ffn_norm.type != LM_DTYPE_F32 && ffn_norm.type != LM_DTYPE_F16 && ffn_norm.type != LM_DTYPE_BF16))) {
                lm_model_close(model); return status != LM_OK ? status : LM_ERR_UNSUPPORTED;
            }
            for (uint32_t i = 0u; i < 7u; ++i) indices.push_back(layer_indices[i]);
        }
    } catch (...) {
        lm_model_close(model);
        return LM_ERR_CAPACITY;
    }
    if ((!graph.output_tied && token_count != (output.dims[0] > output.dims[1] ? output.dims[0] : output.dims[1])) ||
        (!safetensors && token_count != embedding.dims[1]) || token_count > 65536u || hidden_size > 4096u || architecture.intermediate_length > 16384u) {
        lm_model_close(model);
        return LM_ERR_UNSUPPORTED;
    }
    uint64_t max_scratch = 0u;
    for (const uint64_t index : indices) {
        lm_model_tensor_info tensor{};
        status = descriptor(index, &tensor);
        if (status != LM_OK) { lm_model_close(model); return status; }
        if (safetensors) {
            if (tensor.rank != 2u || tensor.dims[0] > UINT64_MAX / tensor.dims[1] ||
                tensor.dims[0] * tensor.dims[1] > UINT64_MAX / sizeof(float)) {
                lm_model_close(model); return LM_ERR_CAPACITY;
            }
            const uint64_t bytes = tensor.dims[0] * tensor.dims[1] * sizeof(float);
            if (bytes > max_scratch) max_scratch = bytes;
        } else {
            lm_model_tensor_binding binding{};
            status = lm_model_tensor_bind_native(model, index, &binding);
            if (status != LM_OK) { lm_model_close(model); return status; }
            if (binding.span.bytes > max_scratch) max_scratch = binding.span.bytes;
        }
    }
    if (max_scratch == 0u || max_scratch > (64ull << 20u)) {
        lm_model_close(model);
        return LM_ERR_CAPACITY;
    }
    std::vector<unsigned char> scratch;
    std::vector<char> generated;
    try {
        scratch.resize(static_cast<size_t>(max_scratch));
        generated.resize(static_cast<size_t>(max_new_tokens) * 256u + 1u, '\0');
    } catch (...) {
        lm_model_close(model);
        return LM_ERR_CAPACITY;
    }
    const char *shader_name = safetensors ? "matvec_f32.comp.spv" :
                              (matrix_type == 8u ? "matvec_q8_0_f32.comp.spv" : "matvec_q4_k_f32.comp.spv");
    const std::string shader_path = resolve_shader_path(shader_name);
    lm_native_generation_config generation{};
    generation.step.matvec = {execution_backend, config.device_index, shader_path.c_str(), nullptr};
    generation.step.layer_index = 0u;
    generation.step.vocab_size = token_count;
    generation.step.hidden_size = hidden_size;
    generation.step.intermediate_size = architecture.intermediate_length;
    generation.step.head_count = architecture.head_count;
    generation.step.head_count_kv = architecture.head_count_kv;
    generation.step.use_rope = 1u;
    generation.step.rope_theta = architecture.rope_frequency_base;
    generation.step.rms_epsilon = architecture.rms_epsilon > 0.0f ? architecture.rms_epsilon : 1.0e-5f;
    generation.step.matrix_format = safetensors ? LM_QUANT_NONE :
                                    (matrix_type == 8u ? LM_QUANT_GGML_Q8_0 : LM_QUANT_GGML_Q4_K);
    generation.step.attention_window = assets.attention_window;
    generation.step.rope_scale = assets.rope_scale;
    generation.has_architecture = 1u;
    generation.architecture = architecture;
    generation.kv_dtype = config.kv_dtype;
    generation.use_typed_kv = execution_backend == LM_BACKEND_CPU ? 1u : 0u;
    generation.sampling = assets.sampling;
    generation.max_new_tokens = max_new_tokens;
    generation.prefill_chunk_tokens = assets.prefill_chunk_tokens;
    generation.trace_sink = config.trace ? text_probe : nullptr;
    generation.trace_user = nullptr;
    size_t generated_bytes = 0u;
    if (callback) {
        if (result || result_bytes) { lm_model_close(model); return LM_ERR_ARGUMENT; }
        std::vector<uint32_t> prompt_tokens;
        try {
            prompt_tokens.resize(std::strlen(prompt) + 1u, 0u);
        } catch (...) {
            lm_model_close(model);
            return LM_ERR_CAPACITY;
        }
        size_t prompt_count = 0u;
        status = lm_model_token_encode(model, prompt, std::strlen(prompt), prompt_tokens.data(),
                                       prompt_tokens.size(), &prompt_count);
        if (status == LM_OK)
            status = lm_model_generate_native_stream(model, &graph, &generation,
                                                     prompt_tokens.data(), prompt_count,
                                                     scratch.data(), scratch.size(), callback,
                                                     callback_user);
    } else {
        status = lm_model_generate_native_text(model, &graph, &generation, prompt, std::strlen(prompt),
                                               scratch.data(), scratch.size(), generated.data(), generated.size(),
                                               &generated_bytes);
    }
    if (status == LM_OK) {
        if (result) {
            if (generated_bytes > result_capacity) status = LM_ERR_CAPACITY;
            else {
                std::memcpy(result, generated.data(), generated_bytes);
                *result_bytes = generated_bytes;
            }
        } else {
            std::printf("generated=");
            std::fwrite(generated.data(), 1u, generated_bytes, stdout);
            std::puts("");
        }
    }
    lm_model_close(model);
    return status;
}

static bool json_string_field(const char *body, const char *key, std::string *out) {
    if (!body || !key || !out) return false;
    const std::string needle = std::string("\"") + key + "\"";
    const char *at = std::strstr(body, needle.c_str());
    if (!at) return false;
    at += needle.size();
    while (*at && std::isspace(static_cast<unsigned char>(*at))) ++at;
    if (*at++ != ':') return false;
    while (*at && std::isspace(static_cast<unsigned char>(*at))) ++at;
    if (*at++ != '\"') return false;
    out->clear();
    for (size_t count = 0u; *at && count < (1u << 16u); ++count, ++at) {
        if (*at == '\"') return true;
        if (*at == '\\') {
            ++at;
            if (!*at) return false;
            if (*at == 'n') out->push_back('\n');
            else if (*at == 'r') out->push_back('\r');
            else if (*at == 't') out->push_back('\t');
            else if (*at == '\"' || *at == '\\' || *at == '/') out->push_back(*at);
            else return false;
        } else {
            if (static_cast<unsigned char>(*at) < 0x20u) return false;
            out->push_back(*at);
        }
    }
    return false;
}

static bool json_chat_messages_field(const char *body, std::string *out) {
    if (!body || !out) return false;
    const char *at = std::strstr(body, "\"messages\"");
    if (!at) return false;
    at += 10u;
    while (*at && std::isspace(static_cast<unsigned char>(*at))) ++at;
    if (*at++ != ':') return false;
    while (*at && std::isspace(static_cast<unsigned char>(*at))) ++at;
    if (*at++ != '[') return false;
    out->clear();
    bool found = false;
    for (;;) {
        while (*at && std::isspace(static_cast<unsigned char>(*at))) ++at;
        if (*at == ']') return found;
        if (*at++ != '{') return false;
        const char *object = at - 1;
        const char *cursor = at;
        unsigned depth = 1u;
        bool quoted = false;
        bool escaped = false;
        while (*cursor && depth != 0u) {
            const char c = *cursor++;
            if (quoted) {
                if (escaped) escaped = false;
                else if (c == '\\') escaped = true;
                else if (c == '"') quoted = false;
            } else if (c == '"') quoted = true;
            else if (c == '{') ++depth;
            else if (c == '}') --depth;
        }
        if (quoted || depth != 0u) return false;
        const size_t object_bytes = static_cast<size_t>(cursor - object);
        if (object_bytes > (1u << 16u)) return false;
        std::string object_text(object, object_bytes);
        std::string role;
        std::string content;
        if (!json_string_field(object_text.c_str(), "role", &role) ||
            !json_string_field(object_text.c_str(), "content", &content) ||
            (role != "system" && role != "user" && role != "assistant")) return false;
        if (out->size() + role.size() + content.size() + 3u > (1u << 16u)) return false;
        if (found) out->push_back('\n');
        out->append(role);
        out->append(": ");
        out->append(content);
        found = true;
        at = cursor;
        while (*at && std::isspace(static_cast<unsigned char>(*at))) ++at;
        if (*at == ',') { ++at; continue; }
        if (*at == ']') return found;
        return false;
    }
}

static bool json_u32_field(const char *body, const char *key, uint32_t *out) {
    if (!body || !key || !out) return false;
    const std::string needle = std::string("\"") + key + "\"";
    const char *at = std::strstr(body, needle.c_str());
    if (!at) return false;
    at += needle.size();
    while (*at && std::isspace(static_cast<unsigned char>(*at))) ++at;
    if (*at++ != ':') return false;
    while (*at && std::isspace(static_cast<unsigned char>(*at))) ++at;
    if (*at < '0' || *at > '9') return false;
    uint64_t value = 0u;
    while (*at >= '0' && *at <= '9') {
        value = value * 10u + static_cast<uint64_t>(*at - '0');
        if (value > 1024u) return false;
        ++at;
    }
    *out = static_cast<uint32_t>(value);
    return true;
}

static bool json_bool_field(const char *body, const char *key, bool *out) {
    if (!body || !key || !out) return false;
    const std::string needle = std::string("\"") + key + "\"";
    const char *at = std::strstr(body, needle.c_str());
    if (!at) return false;
    at += needle.size();
    while (*at && std::isspace(static_cast<unsigned char>(*at))) ++at;
    if (*at++ != ':') return false;
    while (*at && std::isspace(static_cast<unsigned char>(*at))) ++at;
    if (std::strncmp(at, "true", 4u) == 0) { *out = true; return true; }
    if (std::strncmp(at, "false", 5u) == 0) { *out = false; return true; }
    return false;
}

static std::string json_escape(const char *text, size_t bytes) {
    std::string escaped;
    escaped.reserve(bytes + 16u);
    for (size_t i = 0u; i < bytes; ++i) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        if (c == '\\' || c == '\"') { escaped.push_back('\\'); escaped.push_back(static_cast<char>(c)); }
        else if (c == '\n') escaped += "\\n";
        else if (c == '\r') escaped += "\\r";
        else if (c == '\t') escaped += "\\t";
        else if (c >= 0x20u) escaped.push_back(static_cast<char>(c));
    }
    return escaped;
}

#if defined(__unix__) || defined(__APPLE__)
static bool send_all(int socket_fd, const char *data, size_t bytes) {
    while (bytes != 0u) {
        const ssize_t sent = send(socket_fd, data, bytes, 0);
        if (sent <= 0) return false;
        data += sent;
        bytes -= static_cast<size_t>(sent);
    }
    return true;
}

static void send_http(int socket_fd, int status, const char *content_type, const std::string &body) {
    char header[256] = {};
    const int length = std::snprintf(header, sizeof(header),
                                     "HTTP/1.1 %d\r\nContent-Type: %s\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n",
                                     status, content_type, body.size());
    if (length > 0) { send_all(socket_fd, header, static_cast<size_t>(length)); send_all(socket_fd, body.data(), body.size()); }
}

static lm_status http_stream_token(void *user, uint32_t token_id, float probability) {
    (void)probability;
    http_stream_state *state = static_cast<http_stream_state *>(user);
    if (!state || !state->model) return LM_ERR_ARGUMENT;
    size_t token_bytes = 0u;
    lm_status status = lm_model_token_at(state->model, token_id, nullptr, 0u, &token_bytes);
    if (status != LM_OK || token_bytes > (1u << 16u)) return status == LM_OK ? LM_ERR_CAPACITY : status;
    std::vector<char> token;
    try { token.assign(token_bytes + 1u, '\0'); }
    catch (const std::bad_alloc &) { return LM_ERR_CAPACITY; }
    status = lm_model_token_at(state->model, token_id, token.data(), token.size(), &token_bytes);
    if (status != LM_OK) return status;
    const std::string event = std::string(R"(data: {"choices":[{"text":")") +
                              json_escape(token.data(), token_bytes) +
                              R"(","index":0,"finish_reason":null}]}

)";
    return send_all(state->socket_fd, event.data(), event.size()) ? LM_OK : LM_ERR_IO;
}

static int run_http_server(const lm_config &config, const generation_assets &assets, uint32_t port) {
    const int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) return 4;
    int reuse = 1;
    (void)setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(static_cast<uint16_t>(port));
    if (bind(server_fd, reinterpret_cast<const sockaddr *>(&address), sizeof(address)) < 0 || listen(server_fd, 8) < 0) {
        close(server_fd);
        return 4;
    }
    std::printf("server=127.0.0.1:%u\n", port);
    for (;;) {
        const int client = accept(server_fd, nullptr, nullptr);
        if (client < 0) { if (errno == EINTR) continue; break; }
        char request[1u << 16u] = {};
        const ssize_t received = recv(client, request, sizeof(request) - 1u, 0);
        if (received <= 0) { close(client); continue; }
        request[received] = '\0';
        const char *body = std::strstr(request, "\r\n\r\n");
        if (!body) { send_http(client, 400, "application/json", "{\"error\":\"invalid_http\"}"); close(client); continue; }
        body += 4;
        if (std::strncmp(request, "GET /health ", 12u) == 0) {
            send_http(client, 200, "application/json", "{\"status\":\"ok\"}");
        } else if (std::strncmp(request, "POST /v1/completions ", 21u) == 0 ||
                   std::strncmp(request, "POST /v1/chat/completions ", 26u) == 0) {
            const bool chat_request = std::strncmp(request, "POST /v1/chat/completions ", 26u) == 0;
            std::string prompt;
            uint32_t max_tokens = 8u;
            bool stream = false;
            const bool prompt_valid = chat_request ? json_chat_messages_field(body, &prompt) :
                                                      json_string_field(body, "prompt", &prompt);
            if (!prompt_valid ||
                (std::strstr(body, "\"max_tokens\"") && !json_u32_field(body, "max_tokens", &max_tokens)) ||
                (std::strstr(body, "\"stream\"") && !json_bool_field(body, "stream", &stream)) ||
                max_tokens == 0u || max_tokens > 64u) {
                send_http(client, 400, "application/json", "{\"error\":\"invalid_request\"}");
            } else {
                if (stream) {
                    const std::string header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
                    if (!send_all(client, header.data(), header.size())) {
                        close(client);
                        continue;
                    }
                    http_stream_state stream_state{client, nullptr};
                    const lm_status status = run_native_generation(config, assets, prompt.c_str(), max_tokens,
                                                                   nullptr, 0u, nullptr,
                                                                   http_stream_token, &stream_state);
                    if (status == LM_OK) {
                        static const char done[] = R"(data: [DONE]

)";
                        (void)send_all(client, done, sizeof(done) - 1u);
                    } else {
                        const std::string event = std::string(R"(data: {"error":")") +
                                                  lm_status_name(status) + R"("}

)";
                        (void)send_all(client, event.data(), event.size());
                    }
                } else {
                    std::vector<char> generated(static_cast<size_t>(max_tokens) * 256u + 1u, '\0');
                    size_t generated_bytes = 0u;
                    const lm_status status = run_native_generation(config, assets, prompt.c_str(), max_tokens,
                                                                   generated.data(), generated.size(), &generated_bytes);
                    if (status != LM_OK) {
                        const std::string error = std::string("{\"error\":\"") + lm_status_name(status) + "\"}";
                        send_http(client, status == LM_ERR_UNSUPPORTED ? 501 : 500, "application/json", error);
                    } else {
                        const std::string response = std::string("{\"choices\":[{\"text\":\"") +
                                                      json_escape(generated.data(), generated_bytes) + "\",\"finish_reason\":\"stop\"}]}";
                        send_http(client, 200, "application/json", response);
                    }
                }
            }
        } else {
            send_http(client, 404, "application/json", "{\"error\":\"not_found\"}");
        }
        close(client);
    }
    close(server_fd);
    return 0;
}
#else
static int run_http_server(const lm_config &, const generation_assets &, uint32_t) { return 4; }
#endif

int main(int argc, char **argv) {
    lm_config config;
    lm_config_init(&config);
    bool generate = false;
    bool server = false;
    uint32_t server_port = 8080u;
    const char *prompt = nullptr;
    const char *tokenizer_path = nullptr;
    const char *config_path = nullptr;
    uint32_t max_new_tokens = 0u;
    uint32_t attention_window = 0u;
    uint32_t prefill_chunk_tokens = 0u;
    float rope_scale = 0.0f;
    lm_sampling_config sampling{};
    lm_sampling_config_init(&sampling);
    std::vector<char *> filtered;
    filtered.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--generate") == 0) generate = true;
        else if (std::strcmp(argv[i], "--server") == 0) server = true;
        else if (std::strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            const unsigned long value = std::strtoul(argv[++i], nullptr, 10);
            if (value == 0u || value > 65535u) { std::fprintf(stderr, "configuration error: invalid --port\\n"); return 2; }
            server_port = static_cast<uint32_t>(value);
        }         else if (std::strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (std::strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) tokenizer_path = argv[++i];
        else if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) config_path = argv[++i];
        else if (std::strcmp(argv[i], "--max-new-tokens") == 0 && i + 1 < argc) {
            if (!parse_bounded_u32(argv[++i], &max_new_tokens)) {
                std::fprintf(stderr, "configuration error: invalid --max-new-tokens\\n");
                return 2;
            }
        } else if (std::strcmp(argv[i], "--attention-window") == 0 && i + 1 < argc) {
            if (!parse_u32_limit(argv[++i], &attention_window, 1u << 20u)) {
                std::fprintf(stderr, "configuration error: invalid --attention-window\\n"); return 2;
            }
        } else if (std::strcmp(argv[i], "--prefill-chunk-tokens") == 0 && i + 1 < argc) {
            if (!parse_u32_limit(argv[++i], &prefill_chunk_tokens, 1u << 20u)) {
                std::fprintf(stderr, "configuration error: invalid --prefill-chunk-tokens\\n"); return 2;
            }
        } else if (std::strcmp(argv[i], "--rope-scale") == 0 && i + 1 < argc) {
            if (!parse_finite_float(argv[++i], &rope_scale) || rope_scale < 0.0f) {
                std::fprintf(stderr, "configuration error: invalid --rope-scale\\n"); return 2;
            }
        } else if (std::strcmp(argv[i], "--sampling") == 0 && i + 1 < argc) {
            const char *value = argv[++i];
            if (std::strcmp(value, "greedy") == 0) sampling.mode = LM_SAMPLING_GREEDY;
            else if (std::strcmp(value, "top-k") == 0) sampling.mode = LM_SAMPLING_TOP_K;
            else if (std::strcmp(value, "top-p") == 0) sampling.mode = LM_SAMPLING_TOP_P;
            else if (std::strcmp(value, "typical") == 0) sampling.mode = LM_SAMPLING_TYPICAL;
            else { std::fprintf(stderr, "configuration error: invalid --sampling\\n"); return 2; }
        } else if (std::strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            if (!parse_bounded_u32(argv[++i], &sampling.top_k)) {
                std::fprintf(stderr, "configuration error: invalid --top-k\\n"); return 2;
            }
        } else if ((std::strcmp(argv[i], "--top-p") == 0 || std::strcmp(argv[i], "--min-p") == 0 ||
                    std::strcmp(argv[i], "--typical-p") == 0 || std::strcmp(argv[i], "--temperature") == 0 ||
                    std::strcmp(argv[i], "--repetition-penalty") == 0 || std::strcmp(argv[i], "--frequency-penalty") == 0 ||
                    std::strcmp(argv[i], "--presence-penalty") == 0) && i + 1 < argc) {
            const char *option = argv[i];
            float value = 0.0f;
            if (!parse_finite_float(argv[++i], &value)) {
                std::fprintf(stderr, "configuration error: invalid decoding float\\n"); return 2;
            }
            if (std::strcmp(option, "--top-p") == 0) sampling.top_p = value;
            else if (std::strcmp(option, "--min-p") == 0) sampling.min_p = value;
            else if (std::strcmp(option, "--typical-p") == 0) sampling.typical_p = value;
            else if (std::strcmp(option, "--temperature") == 0) sampling.temperature = value;
            else if (std::strcmp(option, "--repetition-penalty") == 0) sampling.repetition_penalty = value;
            else if (std::strcmp(option, "--frequency-penalty") == 0) sampling.frequency_penalty = value;
            else sampling.presence_penalty = value;
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            char *end = nullptr;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            if (!end || *end != '\0') { std::fprintf(stderr, "configuration error: invalid --seed\\n"); return 2; }
            sampling.seed = static_cast<uint64_t>(value);
        } else filtered.push_back(argv[i]);
    }
    if (generate && max_new_tokens == 0u) {
        std::fprintf(stderr, "configuration error: --generate requires --max-new-tokens\\n");
        return 2;
    }
    const char *bad = nullptr;
    const lm_status parsed = lm_config_parse_argv(&config, static_cast<int>(filtered.size()), filtered.data(), &bad);
    if (parsed != LM_OK) {
        std::fprintf(stderr, "configuration error: %s (%s)\n",
                     lm_status_name(parsed), bad ? bad : "unknown");
        return 2;
    }
    if (server && !config.model_path[0]) {
        std::fprintf(stderr, "configuration error: --server requires --model\\n");
        return 2;
    }
    const generation_assets assets{tokenizer_path, config_path, sampling, attention_window,
                                   rope_scale, prefill_chunk_tokens};
    bool list_devices = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0) {
            print_help();
            return 0;
        }
        if (std::strcmp(argv[i], "--list-devices") == 0) list_devices = true;
    }
    if (server) return run_http_server(config, assets, server_port);
    if (generate) {
        const lm_status status = run_native_generation(config, assets, prompt, max_new_tokens, nullptr, 0u, nullptr);
        if (status != LM_OK) std::fprintf(stderr, "generation failed: %s\n", lm_status_name(status));
        return status == LM_OK ? 0 : 4;
    }
    if (list_devices) {
        uint32_t count = 0u;
        const lm_status status = lm_vulkan_device_count(&count);
        if (status != LM_OK) {
            std::fprintf(stderr, "device discovery failed: %s\n", lm_status_name(status));
            return 4;
        }
        for (uint32_t i = 0u; i < count; ++i) {
            lm_vulkan_device_info info{};
            if (lm_vulkan_device_info_get(i, &info) != LM_OK) return 4;
            std::printf("device=%u name=%s api=%u vendor=0x%08x shader_int_dot=%u subgroup=%u cpu=%u\n",
                        i, info.name, info.api_version, info.vendor_id,
                        info.shader_int_dot, info.subgroup, info.is_cpu);
        }
        if (config.backend == LM_BACKEND_AUTO && !config.model_path[0]) return 0;
    }

    lm_runtime *runtime = nullptr;
    const lm_status created = lm_runtime_create(&config, &runtime);
    if (created != LM_OK) {
        std::fprintf(stderr, "backend selection failed: %s; requested backend is not built in this slice\n",
                     lm_status_name(created));
        return 3;
    }
    if (config.trace) lm_runtime_set_probe_sink(runtime, text_probe, nullptr);
    (void)lm_runtime_dump_config(runtime, stdout);
    if (config.trace) {
        lm_probe probe{};
        probe.trace_id = 1u;
        probe.kind = 1u;
        probe.stage = 1u;
        probe.bytes = static_cast<uint32_t>(std::strlen(config.model_path));
        (void)lm_runtime_emit_probe(runtime, &probe);
    }
    std::puts("status=control-plane-ready");
    std::puts("inference=not-enabled-in-this-vertical-slice");
    lm_runtime_destroy(runtime);
    return 0;
}
