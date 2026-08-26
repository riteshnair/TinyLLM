#include "lm/lm.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <cctype>
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
    std::puts("  --server --port N --model PATH");
}

static void text_probe(void *, const lm_probe *probe) {
    std::printf("probe trace=%llu stage=%u kind=%u bytes=%u\n",
                static_cast<unsigned long long>(probe->trace_id), probe->stage,
                probe->kind, probe->bytes);
}

static bool parse_bounded_u32(const char *text, uint32_t *out) {
    if (!text || !out || text[0] == '\0') return false;
    char *end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (!end || *end != '\0' || value > 1024u) return false;
    *out = static_cast<uint32_t>(value);
    return true;
}

static lm_status run_native_generation(const lm_config &config, const char *prompt,
                                       uint32_t max_new_tokens, char *result,
                                       size_t result_capacity, size_t *result_bytes) {
    if (!config.model_path[0] || !prompt || max_new_tokens == 0u ||
        (result == nullptr) != (result_bytes == nullptr)) return LM_ERR_ARGUMENT;
    lm_model_file *model = nullptr;
    char error[128] = {};
    lm_status status = lm_model_open(config.model_path, &model, error, sizeof(error));
    if (status != LM_OK) {
        std::fprintf(stderr, "model open failed: %s (%s)\\n", lm_status_name(status), error);
        return status;
    }
    lm_decoder_graph_binding graph{};
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
    if (embedding.rank != 2u || output.rank != 2u || embedding.dims[1] != output.dims[0] ||
        embedding.dims[0] != output.dims[1] || embedding.dims[0] != architecture.embedding_length ||
        output_norm.rank != 1u || output_norm.dims[0] != embedding.dims[0] ||
        (embedding.type != 8u && embedding.type != 12u) || output.type != embedding.type ||
        output_norm.type != LM_DTYPE_F32) {
        lm_model_close(model);
        return LM_ERR_UNSUPPORTED;
    }
    const uint32_t matrix_type = embedding.type;
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
                if (layer_infos[i].type != matrix_type || layer_infos[i].rank != 2u) {
                    lm_model_close(model); return LM_ERR_UNSUPPORTED;
                }
            }
            const uint32_t head_dim = static_cast<uint32_t>(embedding.dims[0]) / architecture.head_count;
            const uint32_t kv_width = head_dim * architecture.head_count_kv;
            if (layer_infos[0].dims[0] != embedding.dims[0] || layer_infos[0].dims[1] != embedding.dims[0] ||
                layer_infos[1].dims[0] != kv_width || layer_infos[1].dims[1] != embedding.dims[0] ||
                layer_infos[2].dims[0] != kv_width || layer_infos[2].dims[1] != embedding.dims[0] ||
                layer_infos[3].dims[0] != embedding.dims[0] || layer_infos[3].dims[1] != embedding.dims[0] ||
                layer_infos[4].dims[0] != architecture.intermediate_length ||
                layer_infos[4].dims[1] != embedding.dims[0] ||
                layer_infos[5].dims[0] != embedding.dims[0] ||
                layer_infos[5].dims[1] != architecture.intermediate_length ||
                layer_infos[6].dims[0] != architecture.intermediate_length ||
                layer_infos[6].dims[1] != embedding.dims[0]) {
                lm_model_close(model); return LM_ERR_UNSUPPORTED;
            }
            lm_model_tensor_info attn_norm{};
            lm_model_tensor_info ffn_norm{};
            status = descriptor(layer.attn_norm, &attn_norm);
            if (status != LM_OK || attn_norm.type != LM_DTYPE_F32 || attn_norm.rank != 1u ||
                attn_norm.dims[0] != embedding.dims[0]) {
                lm_model_close(model); return status != LM_OK ? status : LM_ERR_UNSUPPORTED;
            }
            status = descriptor(layer.ffn_norm, &ffn_norm);
            if (status != LM_OK || ffn_norm.type != LM_DTYPE_F32 || ffn_norm.rank != 1u ||
                ffn_norm.dims[0] != embedding.dims[0]) {
                lm_model_close(model); return status != LM_OK ? status : LM_ERR_UNSUPPORTED;
            }
            for (uint32_t i = 0u; i < 7u; ++i) indices.push_back(layer_indices[i]);
        }
    } catch (...) {
        lm_model_close(model);
        return LM_ERR_CAPACITY;
    }
    uint32_t token_count = 0u;
    if (lm_model_token_count(model, &token_count) != LM_OK || token_count != output.dims[0] ||
        token_count != embedding.dims[1] || embedding.dims[0] > 4096u || architecture.intermediate_length > 16384u) {
        lm_model_close(model);
        return LM_ERR_UNSUPPORTED;
    }
    uint64_t max_scratch = 0u;
    for (const uint64_t index : indices) {
        lm_model_tensor_binding binding{};
        status = lm_model_tensor_bind_native(model, index, &binding);
        if (status != LM_OK) { lm_model_close(model); return status; }
        if (binding.span.bytes > max_scratch) max_scratch = binding.span.bytes;
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
    const char *shader = matrix_type == 8u ? "matvec_q8_0_f32.comp.spv" : "matvec_q4_k_f32.comp.spv";
    lm_native_generation_config generation{};
    generation.step.matvec = {config.backend, config.device_index, shader, nullptr};
    generation.step.layer_index = 0u;
    generation.step.vocab_size = token_count;
    generation.step.hidden_size = static_cast<uint32_t>(embedding.dims[0]);
    generation.step.intermediate_size = architecture.intermediate_length;
    generation.step.head_count = architecture.head_count;
    generation.step.head_count_kv = architecture.head_count_kv;
    generation.step.use_rope = 0u;
    generation.step.rms_epsilon = 1.0e-5f;
    generation.step.matrix_format = matrix_type == 8u ? LM_QUANT_GGML_Q8_0 : LM_QUANT_GGML_Q4_K;
    generation.sampling = {LM_SAMPLING_GREEDY, 0u, 1.0f, 1u};
    generation.max_new_tokens = max_new_tokens;
    size_t generated_bytes = 0u;
    status = lm_model_generate_native_text(model, &graph, &generation, prompt, std::strlen(prompt),
                                           scratch.data(), scratch.size(), generated.data(), generated.size(),
                                           &generated_bytes);
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

static int run_http_server(const lm_config &config, uint32_t port) {
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
            std::string prompt;
            uint32_t max_tokens = 8u;
            bool stream = false;
            if (!json_string_field(body, "prompt", &prompt) ||
                (std::strstr(body, "\"max_tokens\"") && !json_u32_field(body, "max_tokens", &max_tokens)) ||
                (std::strstr(body, "\"stream\"") && !json_bool_field(body, "stream", &stream)) ||
                max_tokens == 0u || max_tokens > 64u) {
                send_http(client, 400, "application/json", "{\"error\":\"invalid_request\"}");
            } else {
                std::vector<char> generated(static_cast<size_t>(max_tokens) * 256u + 1u, '\0');
                size_t generated_bytes = 0u;
                const lm_status status = run_native_generation(config, prompt.c_str(), max_tokens,
                                                               generated.data(), generated.size(), &generated_bytes);
                if (status != LM_OK) {
                    const std::string error = std::string("{\"error\":\"") + lm_status_name(status) + "\"}";
                    send_http(client, status == LM_ERR_UNSUPPORTED ? 501 : 500, "application/json", error);
                } else if (stream) {
                    const std::string event = std::string("data: {\"choices\":[{\"text\":\"") +
                                              json_escape(generated.data(), generated_bytes) +
                                              "\"}]}\n\ndata: [DONE]\n\n";
                    const std::string header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
                    send_all(client, header.data(), header.size());
                    send_all(client, event.data(), event.size());
                } else {
                    const std::string response = std::string("{\"choices\":[{\"text\":\"") +
                                                  json_escape(generated.data(), generated_bytes) + "\",\"finish_reason\":\"stop\"}]}";
                    send_http(client, 200, "application/json", response);
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
static int run_http_server(const lm_config &, uint32_t) { return 4; }
#endif

int main(int argc, char **argv) {
    lm_config config;
    lm_config_init(&config);
    bool generate = false;
    bool server = false;
    uint32_t server_port = 8080u;
    const char *prompt = nullptr;
    uint32_t max_new_tokens = 0u;
    std::vector<char *> filtered;
    filtered.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--generate") == 0) generate = true;
        else if (std::strcmp(argv[i], "--server") == 0) server = true;
        else if (std::strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            const unsigned long value = std::strtoul(argv[++i], nullptr, 10);
            if (value == 0u || value > 65535u) { std::fprintf(stderr, "configuration error: invalid --port\\n"); return 2; }
            server_port = static_cast<uint32_t>(value);
        } else if (std::strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (std::strcmp(argv[i], "--max-new-tokens") == 0 && i + 1 < argc) {
            if (!parse_bounded_u32(argv[++i], &max_new_tokens)) {
                std::fprintf(stderr, "configuration error: invalid --max-new-tokens\\n");
                return 2;
            }
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
    bool list_devices = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0) {
            print_help();
            return 0;
        }
        if (std::strcmp(argv[i], "--list-devices") == 0) list_devices = true;
    }
    if (server) return run_http_server(config, server_port);
    if (generate) {
        const lm_status status = run_native_generation(config, prompt, max_new_tokens, nullptr, 0u, nullptr);
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
