#include "lm/lm.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void print_help() {
    std::puts("tiny-lm control-plane slice");
    std::puts("usage: tiny-lm [options]");
    std::puts("  --backend auto|cpu|vulkan|rocr|rocm|cuda|openvino|directml");
    std::puts("  --model PATH --context N --threads N --device N");
    std::puts("  --load eager|mmap|lazy|stream --kv-dtype f16|bf16|q8|q6|q4");
    std::puts("  --kv-page-tokens N --trace --deterministic --no-prefetch");
    std::puts("  --dump-config --dry-run --list-devices --help");
    std::puts("  --generate --prompt TEXT --max-new-tokens N");
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
                                       uint32_t max_new_tokens) {
    if (!config.model_path[0] || !prompt || max_new_tokens == 0u) return LM_ERR_ARGUMENT;
    lm_model_file *model = nullptr;
    char error[128] = {};
    lm_status status = lm_model_open(config.model_path, &model, error, sizeof(error));
    if (status != LM_OK) {
        std::fprintf(stderr, "model open failed: %s (%s)\\n", lm_status_name(status), error);
        return status;
    }
    lm_decoder_graph_binding graph{};
    status = lm_model_build_llama_graph(model, &graph);
    if (status != LM_OK || graph.layer_count != 1u) {
        lm_model_close(model);
        return status == LM_OK ? LM_ERR_UNSUPPORTED : status;
    }
    auto descriptor = [model](uint64_t index, lm_model_tensor_info *out) {
        return lm_model_tensor_info_at(model, index, out);
    };
    lm_model_tensor_info embedding{}, output{}, gate{}, up{}, down{}, ffn_norm{}, output_norm{};
    lm_model_architecture architecture{};
    if (lm_model_get_architecture(model, &architecture) != LM_OK || architecture.block_count != graph.layer_count ||
        architecture.embedding_length == 0u || architecture.intermediate_length == 0u ||
        architecture.head_count == 0u || architecture.head_count_kv == 0u ||
        architecture.head_count_kv > architecture.head_count ||
        architecture.head_count % architecture.head_count_kv != 0u) {
        lm_model_close(model);
        return LM_ERR_UNSUPPORTED;
    }
    const lm_decoder_layer_binding &layer = graph.layers[0];
    const uint64_t indices[] = {graph.token_embedding, graph.output, layer.ffn_gate,
                                layer.ffn_up, layer.ffn_down};
    lm_model_tensor_info *descriptors[] = {&embedding, &output, &gate, &up, &down};
    for (uint32_t i = 0u; i < 5u; ++i) {
        status = descriptor(indices[i], descriptors[i]);
        if (status != LM_OK) { lm_model_close(model); return status; }
    }
    status = descriptor(layer.ffn_norm, &ffn_norm);
    if (status != LM_OK) { lm_model_close(model); return status; }
    status = descriptor(graph.output_norm, &output_norm);
    if (status != LM_OK) { lm_model_close(model); return status; }
    if (embedding.rank != 2u || output.rank != 2u || gate.rank != 2u || up.rank != 2u || down.rank != 2u ||
        embedding.dims[1] != output.dims[0] || embedding.dims[0] != output.dims[1] ||
        embedding.dims[0] != architecture.embedding_length || gate.dims[0] != architecture.intermediate_length ||
        gate.dims[1] != embedding.dims[0] || up.dims[1] != embedding.dims[0] ||
        down.dims[1] != gate.dims[0] || down.dims[0] != embedding.dims[0] ||
        ffn_norm.rank != 1u || output_norm.rank != 1u || ffn_norm.dims[0] != embedding.dims[0] ||
        output_norm.dims[0] != embedding.dims[0] ||
        (embedding.type != 8u && embedding.type != 12u)) {
        lm_model_close(model);
        return LM_ERR_UNSUPPORTED;
    }
    const uint32_t matrix_type = embedding.type;
    for (const lm_model_tensor_info *info : descriptors)
        if (info->type != matrix_type) { lm_model_close(model); return LM_ERR_UNSUPPORTED; }
    if (ffn_norm.type != LM_DTYPE_F32 || output_norm.type != LM_DTYPE_F32) {
        lm_model_close(model);
        return LM_ERR_UNSUPPORTED;
    }
    uint32_t token_count = 0u;
    if (lm_model_token_count(model, &token_count) != LM_OK || token_count != output.dims[0] ||
        token_count != embedding.dims[1] || embedding.dims[0] > 4096u || gate.dims[0] > 16384u) {
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
    generation.step.matvec = {config.backend, config.device_index, shader};
    generation.step.layer_index = 0u;
    generation.step.vocab_size = token_count;
    generation.step.hidden_size = static_cast<uint32_t>(embedding.dims[0]);
    generation.step.intermediate_size = static_cast<uint32_t>(gate.dims[0]);
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
        std::printf("generated=");
        std::fwrite(generated.data(), 1u, generated_bytes, stdout);
        std::puts("");
    }
    lm_model_close(model);
    return status;
}

int main(int argc, char **argv) {
    lm_config config;
    lm_config_init(&config);
    bool generate = false;
    const char *prompt = nullptr;
    uint32_t max_new_tokens = 0u;
    std::vector<char *> filtered;
    filtered.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--generate") == 0) generate = true;
        else if (std::strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) prompt = argv[++i];
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
    bool list_devices = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0) {
            print_help();
            return 0;
        }
        if (std::strcmp(argv[i], "--list-devices") == 0) list_devices = true;
    }
    if (generate) {
        const lm_status status = run_native_generation(config, prompt, max_new_tokens);
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
