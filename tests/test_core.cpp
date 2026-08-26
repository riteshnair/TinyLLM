#include "lm/lm.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>

struct ProbeState { unsigned count; uint32_t last_stage; };

static void capture_probe(void *user, const lm_probe *probe) {
    ProbeState *state = static_cast<ProbeState *>(user);
    ++state->count;
    state->last_stage = probe->stage;
}

static void test_defaults() {
    lm_config config;
    lm_config_init(&config);
    assert(config.backend == LM_BACKEND_AUTO);
    assert(config.load_mode == LM_LOAD_MMAP);
    assert(config.context_tokens == 4096u);
    assert(config.kv_page_tokens == 32u);
    assert(config.threads == 1u);
}

static void test_cli() {
    char a0[] = "tiny-lm";
    char a1[] = "--backend"; char a2[] = "cpu";
    char a3[] = "--load"; char a4[] = "lazy";
    char a5[] = "--weights"; char a6[] = "preserve";
    char a7[] = "--kv-dtype"; char a8[] = "q8";
    char a9[] = "--context"; char a10[] = "2048";
    char a11[] = "--threads"; char a12[] = "4";
    char a13[] = "--trace"; char a14[] = "--model"; char a15[] = "model.gguf";
    char *argv[] = {a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15};
    lm_config config;
    lm_config_init(&config);
    const char *bad = nullptr;
    assert(lm_config_parse_argv(&config, 16, argv, &bad) == LM_OK);
    assert(config.backend == LM_BACKEND_CPU);
    assert(config.load_mode == LM_LOAD_LAZY);
    assert(config.weight_policy == LM_WEIGHT_PRESERVE);
    assert(config.kv_dtype == LM_KV_Q8);
    assert(config.context_tokens == 2048u);
    assert(config.threads == 4u);
    assert(config.trace == 1u);
    assert(std::strcmp(config.model_path, "model.gguf") == 0);
}

static void test_quantize_policy_gate() {
    lm_config config;
    lm_config_init(&config);
    config.weight_policy = LM_WEIGHT_QUANTIZE_CACHE;
    lm_runtime *runtime = nullptr;
    assert(lm_runtime_create(&config, &runtime) == LM_ERR_UNSUPPORTED);
    assert(runtime == nullptr);
}

static void test_vulkan_backend_resolution() {
    char a0[] = "tiny-lm"; char a1[] = "--backend"; char a2[] = "vulkan";
    char *argv[] = {a0,a1,a2};
    lm_config config;
    lm_config_init(&config);
    const char *bad = nullptr;
    assert(lm_config_parse_argv(&config, 3, argv, &bad) == LM_OK);
    lm_runtime *runtime = nullptr;
    uint32_t device_count = 0u;
    const lm_status discovered = lm_vulkan_device_count(&device_count);
    const lm_status created = lm_runtime_create(&config, &runtime);
    if (discovered == LM_OK && device_count > 0u) {
        assert(created == LM_OK && runtime != nullptr);
        lm_runtime_destroy(runtime);
    } else {
        assert(created == LM_ERR_UNSUPPORTED && runtime == nullptr);
    }
    lm_config auto_config;
    lm_config_init(&auto_config);
    lm_runtime *auto_runtime = nullptr;
    assert(lm_runtime_create(&auto_config, &auto_runtime) == LM_OK);
    FILE *dump = std::tmpfile();
    assert(dump != nullptr);
    assert(lm_runtime_dump_config(auto_runtime, dump) == LM_OK);
    std::rewind(dump);
    char dump_text[128] = {};
    const size_t read = std::fread(dump_text, 1u, sizeof(dump_text) - 1u, dump);
    dump_text[read] = '\0';
    if (discovered == LM_OK && device_count > 0u)
        assert(std::strstr(dump_text, "backend=vulkan") != nullptr);
    else
        assert(std::strstr(dump_text, "backend=cpu") != nullptr);
    std::fclose(dump);
    lm_runtime_destroy(auto_runtime);
}

static void test_tensor_and_buffer() {
    float values[6] = {};
    const uint32_t dims[2] = {2u, 3u};
    lm_tensor tensor{};
    assert(lm_tensor_make_view(values, sizeof(values), LM_DTYPE_F32, 2u, dims, &tensor) == LM_OK);
    assert(tensor.strides[0] == 3u && tensor.strides[1] == 1u);
    lm_tensor bad = tensor;
    bad.bytes = 4u;
    assert(lm_tensor_validate(&bad) == LM_ERR_CAPACITY);
    lm_buffer *buffer = nullptr;
    assert(lm_buffer_alloc(64u, &buffer) == LM_OK);
    lm_tensor buffer_view{};
    assert(lm_buffer_view(buffer, &buffer_view) == LM_OK);
    assert(buffer_view.bytes == 64u && buffer_view.dtype == LM_DTYPE_U8);
    unsigned char quantized_bytes[20] = {};
    const uint32_t quant_dims[1] = {32u};
    lm_tensor quantized{};
    assert(lm_tensor_make_quant_view(quantized_bytes, sizeof(quantized_bytes), 1u, quant_dims, 32u, 20u, &quantized) == LM_OK);
    assert(quantized.quant_format == LM_QUANT_BLOCK_STORAGE && quantized.dtype == LM_DTYPE_U8);
    assert(lm_tensor_make_quant_view(quantized_bytes, 19u, 1u, quant_dims, 32u, 20u, &quantized) == LM_ERR_CAPACITY);
    unsigned char q8_bytes[34] = {};
    q8_bytes[0] = 0x00u;
    q8_bytes[1] = 0x3cu;
    for (unsigned i = 0u; i < 32u; ++i) q8_bytes[2u + i] = static_cast<unsigned char>(i + 1u);
    lm_tensor q8_weights{};
    assert(lm_tensor_make_q8_0_view(q8_bytes, sizeof(q8_bytes), 1u, quant_dims, &q8_weights) == LM_OK);
    const float q8_input[32] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float q8_result = 0.0f;
    assert(lm_cpu_dot_q8_0(&q8_weights, q8_input, 32u, &q8_result) == LM_OK);
    assert(q8_result == 528.0f);
    assert(lm_tensor_make_q8_0_view(q8_bytes, 33u, 1u, quant_dims, &q8_weights) == LM_ERR_CAPACITY);
    unsigned char q4_bytes[18] = {};
    q4_bytes[0] = 0x00u;
    q4_bytes[1] = 0x3cu;
    for (unsigned i = 0u; i < 16u; ++i) q4_bytes[2u + i] = 0x99u;
    lm_tensor q4_weights{};
    assert(lm_tensor_make_q4_0_view(q4_bytes, sizeof(q4_bytes), 1u, quant_dims, &q4_weights) == LM_OK);
    float q4_result = 0.0f;
    assert(lm_cpu_dot_q4_0(&q4_weights, q8_input, 32u, &q4_result) == LM_OK);
    assert(q4_result == 32.0f);
    assert(lm_tensor_make_q4_0_view(q4_bytes, 17u, 1u, quant_dims, &q4_weights) == LM_ERR_CAPACITY);
    unsigned char q4_k_bytes[144] = {};
    q4_k_bytes[0] = 0x00u;
    q4_k_bytes[1] = 0x3cu;
    for (unsigned i = 0u; i < 4u; ++i) q4_k_bytes[4u + i] = 1u;
    for (unsigned i = 0u; i < 4u; ++i) q4_k_bytes[12u + i] = 1u;
    for (unsigned i = 0u; i < 128u; ++i) q4_k_bytes[16u + i] = 0x11u;
    const uint32_t q4_k_dims[1] = {256u};
    lm_tensor q4_k_weights{};
    assert(lm_tensor_make_q4_k_view(q4_k_bytes, sizeof(q4_k_bytes), 1u, q4_k_dims, &q4_k_weights) == LM_OK);
    float q4_k_result = 0.0f;
    float q4_k_ones[256] = {};
    for (float &value : q4_k_ones) value = 1.0f;
    assert(lm_cpu_dot_q4_k(&q4_k_weights, q4_k_ones, 256u, &q4_k_result) == LM_OK);
    assert(q4_k_result == 256.0f);
    assert(lm_tensor_make_q4_k_view(q4_k_bytes, sizeof(q4_k_bytes) - 1u, 1u, q4_k_dims, &q4_k_weights) == LM_ERR_CAPACITY);
    lm_buffer_free(buffer);
}

static void test_file_access() {
    const char *path = "test-file.bin";
    {
        std::ofstream file(path, std::ios::binary);
        const char bytes[] = "0123456789";
        file.write(bytes, sizeof(bytes) - 1u);
    }
    lm_file *file = nullptr;
    assert(lm_file_open(path, &file) == LM_OK);
    uint64_t size = 0u;
    assert(lm_file_size(file, &size) == LM_OK && size == 10u);
    char readback[5] = {};
    assert(lm_file_read(file, 3u, readback, 4u) == LM_OK);
    assert(std::memcmp(readback, "3456", 4u) == 0);
    lm_file_span span{};
    assert(lm_file_span_make(file, 2u, 5u, &span) == LM_OK);
    std::memset(readback, 0, sizeof(readback));
    assert(lm_file_span_read(&span, 1u, readback, 4u) == LM_OK);
    assert(std::memcmp(readback, "3456", 4u) == 0);
    assert(lm_file_span_read(&span, 2u, readback, 4u) == LM_ERR_RANGE);
    assert(lm_file_read(file, size, nullptr, 0u) == LM_OK);
    assert(lm_file_read(file, 8u, readback, 4u) == LM_ERR_RANGE);
    lm_file_close(file);
    std::remove(path);
}

static void test_native_model_tensor_binding() {
    auto put_u32 = [](std::vector<unsigned char> *bytes, uint32_t value) {
        for (unsigned i = 0u; i < 4u; ++i) bytes->push_back(static_cast<unsigned char>((value >> (8u * i)) & 0xffu));
    };
    auto put_u64 = [](std::vector<unsigned char> *bytes, uint64_t value) {
        for (unsigned i = 0u; i < 8u; ++i) bytes->push_back(static_cast<unsigned char>((value >> (8u * i)) & 0xffu));
    };
    auto put_descriptor = [&put_u32, &put_u64](std::vector<unsigned char> *bytes, const char *name,
                                                uint32_t type, uint64_t offset, uint64_t dimension) {
        const size_t length = std::strlen(name);
        put_u64(bytes, length);
        bytes->insert(bytes->end(), name, name + length);
        put_u32(bytes, 1u);
        put_u64(bytes, dimension);
        put_u32(bytes, type);
        put_u64(bytes, offset);
    };
    auto put_matrix_descriptor = [&put_u32, &put_u64](std::vector<unsigned char> *bytes, const char *name,
                                                       uint32_t type, uint64_t offset,
                                                       uint64_t first_dimension, uint64_t second_dimension) {
        const size_t length = std::strlen(name);
        put_u64(bytes, length);
        bytes->insert(bytes->end(), name, name + length);
        put_u32(bytes, 2u);
        put_u64(bytes, first_dimension);
        put_u64(bytes, second_dimension);
        put_u32(bytes, type);
        put_u64(bytes, offset);
    };
    std::vector<unsigned char> gguf(24u, 0u);
    gguf[0] = 'G'; gguf[1] = 'G'; gguf[2] = 'U'; gguf[3] = 'F';
    gguf[4] = 3u;
    gguf[8] = 3u;
    put_descriptor(&gguf, "q4", 2u, 0u, 32u);
    put_matrix_descriptor(&gguf, "q8", 8u, 32u, 1u, 32u);
    put_descriptor(&gguf, "q4_k", 12u, 96u, 256u);
    const size_t header_size = (gguf.size() + 31u) & ~static_cast<size_t>(31u);
    gguf.resize(header_size + 96u + 144u, 0u);
    gguf[header_size] = 0x00u;
    gguf[header_size + 1u] = 0x3cu;
    for (unsigned i = 0u; i < 16u; ++i) gguf[header_size + 2u + i] = 0x99u;
    gguf[header_size + 32u] = 0x00u;
    gguf[header_size + 33u] = 0x3cu;
    for (unsigned i = 0u; i < 32u; ++i) gguf[header_size + 34u + i] = static_cast<unsigned char>(i + 1u);
    gguf[header_size + 96u] = 0x00u;
    gguf[header_size + 97u] = 0x3cu;
    for (unsigned i = 0u; i < 4u; ++i) gguf[header_size + 100u + i] = 1u;
    for (unsigned i = 0u; i < 4u; ++i) gguf[header_size + 108u + i] = 1u;
    for (unsigned i = 0u; i < 128u; ++i) gguf[header_size + 112u + i] = 0x11u;
    const char *path = "test-native-binding.gguf";
    {
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char *>(gguf.data()), static_cast<std::streamsize>(gguf.size()));
    }
    lm_model_file *model = nullptr;
    char error[128] = {};
    const lm_status opened = lm_model_open(path, &model, error, sizeof(error));
    if (opened != LM_OK) std::fprintf(stderr, "native_binding_open=%s error=%s\\n", lm_status_name(opened), error);
    assert(opened == LM_OK);
    lm_model_tensor_binding q4_binding{};
    assert(lm_model_tensor_bind_native(model, 0u, &q4_binding) == LM_OK);
    assert(q4_binding.elements == 32u && q4_binding.span.bytes == 18u &&
           q4_binding.quant_format == LM_QUANT_GGML_Q4_0);
    unsigned char q4_bytes[18] = {};
    lm_tensor q4_tensor{};
    assert(lm_model_tensor_binding_read(&q4_binding, q4_bytes, sizeof(q4_bytes), &q4_tensor) == LM_OK);
    const float input[32] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                             1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                             1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                             1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float result = 0.0f;
    assert(lm_cpu_dot_q4_0(&q4_tensor, input, 32u, &result) == LM_OK && result == 32.0f);
    lm_model_tensor_binding q8_binding{};
    assert(lm_model_tensor_bind_native(model, 1u, &q8_binding) == LM_OK);
    assert(q8_binding.elements == 32u && q8_binding.span.offset == q4_binding.span.offset + 32u &&
           q8_binding.span.bytes == 34u && q8_binding.quant_format == LM_QUANT_GGML_Q8_0);
    unsigned char q8_bytes[34] = {};
    lm_tensor q8_tensor{};
    assert(lm_model_tensor_binding_read(&q8_binding, q8_bytes, sizeof(q8_bytes), &q8_tensor) == LM_OK);
    assert(lm_cpu_dot_q8_0(&q8_tensor, input, 32u, &result) == LM_OK && result == 528.0f);
    unsigned char bound_q8_scratch[34] = {};
    float bound_q8_input[32] = {};
    for (float &value : bound_q8_input) value = 1.0f;
    float bound_q8_cpu = 0.0f;
    assert(lm_model_tensor_binding_dot_q8_0_cpu(&q8_binding, bound_q8_scratch,
                                                sizeof(bound_q8_scratch), bound_q8_input,
                                                32u, &bound_q8_cpu) == LM_OK);
    assert(bound_q8_cpu == 528.0f);
    uint32_t q8_vulkan_devices = 0u;
    if (lm_vulkan_device_count(&q8_vulkan_devices) == LM_OK && q8_vulkan_devices != 0u) {
        float bound_q8_vulkan = 0.0f;
        assert(lm_model_tensor_binding_dot_q8_0_vulkan(&q8_binding, bound_q8_scratch,
                                                       sizeof(bound_q8_scratch),
                                                       "dot_q8_0_f32.comp.spv", 0u,
                                                       bound_q8_input, 32u,
                                                       &bound_q8_vulkan) == LM_OK);
        assert(bound_q8_vulkan == bound_q8_cpu);
    }
    assert(lm_model_tensor_binding_dot_q8_0_cpu(&q8_binding, bound_q8_scratch,
                                                sizeof(bound_q8_scratch) - 1u,
                                                bound_q8_input, 32u, &bound_q8_cpu) == LM_ERR_CAPACITY);
    lm_model_tensor_binding q8_matrix_binding = q8_binding;
    q8_matrix_binding.descriptor.rank = 2u;
    q8_matrix_binding.descriptor.dims[0] = 1u;
    q8_matrix_binding.descriptor.dims[1] = 32u;
    float q8_matrix_cpu[1] = {};
    assert(lm_model_tensor_binding_matvec_q8_0_cpu(&q8_matrix_binding, bound_q8_scratch,
                                                   sizeof(bound_q8_scratch), 1u, 32u,
                                                   bound_q8_input, q8_matrix_cpu) == LM_OK);
    assert(q8_matrix_cpu[0] == bound_q8_cpu);
    lm_native_matvec_config native_cpu{LM_BACKEND_CPU, 0u, nullptr, nullptr};
    float native_cpu_output[1] = {};
    assert(lm_model_tensor_matvec_native(model, 1u, &native_cpu, bound_q8_scratch,
                                         sizeof(bound_q8_scratch), 1u, 32u,
                                         bound_q8_input, native_cpu_output) == LM_OK);
    assert(native_cpu_output[0] == q8_matrix_cpu[0]);
    if (q8_vulkan_devices != 0u) {
        float q8_matrix_vulkan[1] = {};
        assert(lm_model_tensor_binding_matvec_q8_0_vulkan(&q8_matrix_binding, bound_q8_scratch,
                                                          sizeof(bound_q8_scratch), 1u, 32u,
                                                          "matvec_q8_0_f32.comp.spv", 0u,
                                                          bound_q8_input, q8_matrix_vulkan) == LM_OK);
        assert(q8_matrix_vulkan[0] == q8_matrix_cpu[0]);
        lm_native_matvec_config native_vulkan{LM_BACKEND_VULKAN, 0u, "matvec_q8_0_f32.comp.spv", nullptr};
        float native_vulkan_output[1] = {};
        assert(lm_model_tensor_matvec_native(model, 1u, &native_vulkan, bound_q8_scratch,
                                             sizeof(bound_q8_scratch), 1u, 32u,
                                             bound_q8_input, native_vulkan_output) == LM_OK);
        assert(native_vulkan_output[0] == q8_matrix_cpu[0]);
#if LM_HAS_VULKAN
        lm_vulkan_packed_context *packed_context = nullptr;
        assert(lm_vulkan_packed_context_create("matvec_q8_0_f32.comp.spv", 0u, 34u, 32u, &packed_context) == LM_OK);
        native_vulkan.vulkan_context = packed_context;
        float reusable_output[1] = {};
        assert(lm_model_tensor_matvec_native(model, 1u, &native_vulkan, bound_q8_scratch,
                                             sizeof(bound_q8_scratch), 1u, 32u,
                                             bound_q8_input, reusable_output) == LM_OK);
        assert(reusable_output[0] == q8_matrix_cpu[0]);
        assert(lm_model_tensor_matvec_native(model, 1u, &native_vulkan, bound_q8_scratch,
                                             sizeof(bound_q8_scratch), 1u, 32u,
                                             bound_q8_input, reusable_output) == LM_OK);
        lm_vulkan_packed_context_destroy(packed_context);
#endif
    } else {
#if !LM_HAS_VULKAN
        lm_vulkan_packed_context *packed_context = nullptr;
        assert(lm_vulkan_packed_context_create("matvec_q8_0_f32.comp.spv", 0u, 34u, 32u, &packed_context) == LM_ERR_UNSUPPORTED);
#endif
    }
    lm_model_tensor_binding q4_k_binding{};
    assert(lm_model_tensor_bind_native(model, 2u, &q4_k_binding) == LM_OK);
    assert(q4_k_binding.elements == 256u && q4_k_binding.span.bytes == 144u &&
           q4_k_binding.quant_format == LM_QUANT_GGML_Q4_K);
    unsigned char bound_q4_k[144] = {};
    lm_tensor bound_q4_k_tensor{};
    assert(lm_model_tensor_binding_read(&q4_k_binding, bound_q4_k, sizeof(bound_q4_k), &bound_q4_k_tensor) == LM_OK);
    float bound_q4_k_result = 0.0f;
    float bound_q4_k_input[256] = {};
    for (float &value : bound_q4_k_input) value = 1.0f;
    assert(lm_cpu_dot_q4_k(&bound_q4_k_tensor, bound_q4_k_input, 256u, &bound_q4_k_result) == LM_OK);
    assert(bound_q4_k_result == 256.0f);
    lm_model_tensor_binding matrix_binding = q4_k_binding;
    matrix_binding.descriptor.rank = 2u;
    matrix_binding.descriptor.dims[0] = 1u;
    matrix_binding.descriptor.dims[1] = 256u;
    unsigned char matrix_scratch[144] = {};
    float matrix_input[256] = {};
    for (float &value : matrix_input) value = 1.0f;
    float matrix_cpu[1] = {};
    assert(lm_model_tensor_binding_matvec_q4_k_cpu(&matrix_binding, matrix_scratch,
                                                   sizeof(matrix_scratch), 1u, 256u,
                                                   matrix_input, matrix_cpu) == LM_OK);
    assert(matrix_cpu[0] == 256.0f);
    uint32_t model_vulkan_devices = 0u;
    if (lm_vulkan_device_count(&model_vulkan_devices) == LM_OK && model_vulkan_devices != 0u) {
        float matrix_vulkan[1] = {};
        assert(lm_model_tensor_binding_matvec_q4_k_vulkan(&matrix_binding, matrix_scratch,
                                                          sizeof(matrix_scratch), 1u, 256u,
                                                          "matvec_q4_k_f32.comp.spv", 0u,
                                                          matrix_input, matrix_vulkan) == LM_OK);
        assert(matrix_vulkan[0] == matrix_cpu[0]);
#if LM_HAS_VULKAN
        lm_vulkan_packed_context *packed_q4_context = nullptr;
        assert(lm_vulkan_packed_context_create("matvec_q4_k_f32.comp.spv", 0u, 144u, 256u, &packed_q4_context) == LM_OK);
        float reusable_q4[1] = {};
        assert(lm_vulkan_packed_context_matvec(packed_q4_context, bound_q4_k, 1u, 1u,
                                               matrix_input, reusable_q4) == LM_OK);
        assert(reusable_q4[0] == matrix_cpu[0]);
        assert(lm_vulkan_packed_context_matvec(packed_q4_context, bound_q4_k, 1u, 1u,
                                               matrix_input, reusable_q4) == LM_OK);
        lm_vulkan_packed_context_destroy(packed_q4_context);
#endif
    }
    assert(lm_model_tensor_binding_view(&q4_binding, q4_bytes, sizeof(q4_bytes) - 1u, &q4_tensor) == LM_ERR_CAPACITY);
    lm_model_close(model);
    std::remove(path);
    const char *graph_path = "test-llama-graph.gguf";
    const char *graph_names[] = {
        "token_embd.weight", "output.weight", "output_norm.weight",
        "blk.0.attn_norm.weight", "blk.0.attn_q.weight", "blk.0.attn_k.weight",
        "blk.0.attn_v.weight", "blk.0.attn_output.weight", "blk.0.ffn_norm.weight",
        "blk.0.ffn_gate.weight", "blk.0.ffn_down.weight", "blk.0.ffn_up.weight"};
    std::vector<unsigned char> graph_gguf(24u, 0u);
    graph_gguf[0] = 'G'; graph_gguf[1] = 'G'; graph_gguf[2] = 'U'; graph_gguf[3] = 'F';
    graph_gguf[4] = 3u;
    graph_gguf[8] = 12u;
    for (unsigned i = 0u; i < 12u; ++i)
        put_descriptor(&graph_gguf, graph_names[i], 0u, static_cast<uint64_t>(i) * 32u, 1u);
    const size_t graph_header = (graph_gguf.size() + 31u) & ~static_cast<size_t>(31u);
    graph_gguf.resize(graph_header + 356u, 0u);
    {
        std::ofstream file(graph_path, std::ios::binary);
        file.write(reinterpret_cast<const char *>(graph_gguf.data()), static_cast<std::streamsize>(graph_gguf.size()));
    }
    lm_model_file *graph_model = nullptr;
    assert(lm_model_open(graph_path, &graph_model, error, sizeof(error)) == LM_OK);
    lm_decoder_graph_binding graph_binding{};
    assert(lm_model_build_llama_graph(graph_model, &graph_binding) == LM_OK);
    assert(graph_binding.token_embedding == 0u && graph_binding.output == 1u &&
           graph_binding.output_norm == 2u && graph_binding.layer_count == 1u);
    assert(graph_binding.layers[0].attn_q == 4u && graph_binding.layers[0].ffn_up == 11u);
    lm_model_close(graph_model);
    std::remove(graph_path);

    std::vector<unsigned char> malformed(24u, 0u);
    malformed[0] = 'G'; malformed[1] = 'G'; malformed[2] = 'U'; malformed[3] = 'F'; malformed[4] = 3u;
    malformed[8] = 1u;
    put_descriptor(&malformed, "short", 2u, 0u, 32u);
    const size_t malformed_header = (malformed.size() + 31u) & ~static_cast<size_t>(31u);
    malformed.resize(malformed_header + 17u, 0u);
    const char *bad_path = "test-short-native-binding.gguf";
    {
        std::ofstream file(bad_path, std::ios::binary);
        file.write(reinterpret_cast<const char *>(malformed.data()), static_cast<std::streamsize>(malformed.size()));
    }
    lm_model_info info{};
    assert(lm_model_inspect(bad_path, &info, error, sizeof(error)) == LM_ERR_PARSE);
    std::remove(bad_path);
}

static void test_model_bound_moe_q4_k() {
    const uint32_t hidden = 256u;
    const uint32_t intermediate = 256u;
    const uint32_t experts = 2u;
    const uint64_t gate_expert_bytes = 512ull * 144u;
    const uint64_t down_expert_bytes = 256ull * 144u;
    const uint64_t gate_tensor_bytes = gate_expert_bytes * experts;
    const uint64_t down_tensor_bytes = down_expert_bytes * experts;
    const uint64_t router_tensor_bytes = static_cast<uint64_t>(experts) * hidden * sizeof(float);
    auto put_u32 = [](std::vector<unsigned char> *bytes, uint32_t value) {
        for (unsigned i = 0u; i < 4u; ++i) bytes->push_back(static_cast<unsigned char>((value >> (8u * i)) & 0xffu));
    };
    auto put_u64 = [](std::vector<unsigned char> *bytes, uint64_t value) {
        for (unsigned i = 0u; i < 8u; ++i) bytes->push_back(static_cast<unsigned char>((value >> (8u * i)) & 0xffu));
    };
    auto put_rank2 = [&put_u32, &put_u64](std::vector<unsigned char> *bytes, const char *name,
                                            uint64_t first_dimension, uint64_t second_dimension,
                                            uint32_t type, uint64_t offset) {
        const size_t length = std::strlen(name);
        put_u64(bytes, length);
        bytes->insert(bytes->end(), name, name + length);
        put_u32(bytes, 2u);
        put_u64(bytes, first_dimension);
        put_u64(bytes, second_dimension);
        put_u32(bytes, type);
        put_u64(bytes, offset);
    };
    auto put_rank3 = [&put_u32, &put_u64](std::vector<unsigned char> *bytes, const char *name,
                                            uint64_t first_dimension, uint64_t second_dimension,
                                            uint64_t offset) {
        const size_t length = std::strlen(name);
        put_u64(bytes, length);
        bytes->insert(bytes->end(), name, name + length);
        put_u32(bytes, 3u);
        put_u64(bytes, first_dimension);
        put_u64(bytes, second_dimension);
        put_u64(bytes, 2u);
        put_u32(bytes, 12u);
        put_u64(bytes, offset);
    };
    std::vector<unsigned char> gguf(24u, 0u);
    gguf[0] = 'G'; gguf[1] = 'G'; gguf[2] = 'U'; gguf[3] = 'F'; gguf[4] = 3u; gguf[8] = 3u;
    put_rank3(&gguf, "layers.0.feed_forward.experts.w1", hidden, intermediate * 2u, 0u);
    put_rank3(&gguf, "layers.0.feed_forward.experts.w2", intermediate, hidden, gate_tensor_bytes);
    put_rank2(&gguf, "blk.0.ffn_router.weight", experts, hidden, LM_DTYPE_F32, gate_tensor_bytes + down_tensor_bytes);
    const size_t header_size = (gguf.size() + 31u) & ~static_cast<size_t>(31u);
    gguf.resize(header_size + static_cast<size_t>(gate_tensor_bytes + down_tensor_bytes + router_tensor_bytes), 0u);
    for (uint32_t tensor = 0u; tensor < 2u; ++tensor) {
        const size_t tensor_base = header_size + static_cast<size_t>(tensor == 0u ? 0u : gate_tensor_bytes);
        const uint64_t tensor_expert_bytes = tensor == 0u ? gate_expert_bytes : down_expert_bytes;
        const uint32_t tensor_blocks = tensor == 0u ? 512u : 256u;
        for (uint32_t expert = 0u; expert < experts; ++expert) {
            const unsigned char nibble = static_cast<unsigned char>(expert == 0u ? 0x11u : 0x22u);
            const size_t expert_base = tensor_base + static_cast<size_t>(expert * tensor_expert_bytes);
            for (uint32_t block = 0u; block < tensor_blocks; ++block) {
                unsigned char *base = gguf.data() + expert_base + static_cast<size_t>(block) * 144u;
                base[1] = 0x3cu;
                for (uint32_t i = 0u; i < 4u; ++i) base[4u + i] = 1u;
                for (uint32_t i = 0u; i < 4u; ++i) base[12u + i] = 1u;
                for (uint32_t i = 0u; i < 128u; ++i) base[16u + i] = nibble;
            }
        }
    }
    float *router_data = reinterpret_cast<float *>(gguf.data() + header_size + gate_tensor_bytes + down_tensor_bytes);
    for (uint32_t i = 0u; i < hidden; ++i) router_data[hidden + i] = 1.0f;
    const char *path = "test-model-bound-moe.gguf";
    {
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char *>(gguf.data()), static_cast<std::streamsize>(gguf.size()));
    }
    lm_model_file *model = nullptr;
    char error[128] = {};
    assert(lm_model_open(path, &model, error, sizeof(error)) == LM_OK);
    lm_moe_route route{};
    route.expert_count = experts;
    route.experts_per_token = 1u;
    route.selected[0] = 1u;
    route.weights[0] = 1.0f;
    std::vector<unsigned char> expected_gate(static_cast<size_t>(gate_tensor_bytes));
    std::vector<unsigned char> expected_down(static_cast<size_t>(down_tensor_bytes));
    std::memcpy(expected_gate.data(), gguf.data() + header_size, static_cast<size_t>(gate_tensor_bytes));
    std::memcpy(expected_down.data(), gguf.data() + header_size + gate_tensor_bytes, static_cast<size_t>(down_tensor_bytes));
    const uint32_t gate_dims[3] = {hidden, intermediate * 2u, experts};
    const uint32_t down_dims[3] = {intermediate, hidden, experts};
    lm_tensor expected_gate_tensor{};
    lm_tensor expected_down_tensor{};
    assert(lm_tensor_make_q4_k_view(expected_gate.data(), expected_gate.size(), 3u, gate_dims, &expected_gate_tensor) == LM_OK);
    assert(lm_tensor_make_q4_k_view(expected_down.data(), expected_down.size(), 3u, down_dims, &expected_down_tensor) == LM_OK);
    std::vector<float> input(hidden, 1.0f);
    std::vector<float> expected(hidden, 0.0f);
    assert(lm_cpu_moe_selected_expert_mlp_q4_k(&route, &expected_gate_tensor, &expected_down_tensor,
                                               hidden, intermediate, input.data(), expected.data()) == LM_OK);
    std::vector<unsigned char> gate_scratch(static_cast<size_t>(gate_expert_bytes));
    std::vector<unsigned char> down_scratch(static_cast<size_t>(down_expert_bytes));
    std::vector<float> actual(hidden, 0.0f);
    assert(lm_model_moe_selected_expert_mlp_q4_k(model, 0u, 1u, &route, 0u, hidden, intermediate,
                                                 gate_scratch.data(), gate_scratch.size(), down_scratch.data(),
                                                 down_scratch.size(), input.data(), actual.data(), actual.size()) == LM_OK);
    for (uint32_t i = 0u; i < hidden; ++i) assert(actual[i] == expected[i]);
    assert(lm_model_moe_selected_expert_mlp_q4_k(model, 0u, 1u, &route, 0u, hidden, intermediate,
                                                 gate_scratch.data(), gate_scratch.size() - 1u, down_scratch.data(),
                                                 down_scratch.size(), input.data(), actual.data(), actual.size()) == LM_ERR_CAPACITY);
    lm_model_moe_layer_config moe_config{};
    moe_config.router_tensor = 2u;
    moe_config.gate_up_tensor = 0u;
    moe_config.down_tensor = 1u;
    moe_config.expected_layer = 0u;
    moe_config.hidden_size = hidden;
    moe_config.intermediate_size = intermediate;
    moe_config.expert_count = experts;
    moe_config.experts_per_token = 1u;
    moe_config.route_policy = LM_MOE_SOFTMAX_SELECTED_ONLY;
    std::vector<unsigned char> router_scratch(static_cast<size_t>(router_tensor_bytes));
    std::fill(actual.begin(), actual.end(), 0.0f);
    assert(lm_model_execute_moe_layer_f32_router_q4_k_cpu(model, &moe_config, router_scratch.data(), router_scratch.size(),
                                                          gate_scratch.data(), gate_scratch.size(), down_scratch.data(),
                                                          down_scratch.size(), input.data(), actual.data(), actual.size()) == LM_OK);
    for (uint32_t i = 0u; i < hidden; ++i) assert(actual[i] == expected[i]);
    assert(lm_model_execute_moe_layer_f32_router_q4_k_cpu(model, &moe_config, router_scratch.data(), router_scratch.size() - 1u,
                                                          gate_scratch.data(), gate_scratch.size(), down_scratch.data(),
                                                          down_scratch.size(), input.data(), actual.data(), actual.size()) == LM_ERR_CAPACITY);
    lm_model_moe_graph_binding moe_graph{};
    assert(lm_model_build_mixtral_moe_graph(model, experts, 1u, &moe_graph) == LM_OK);
    assert(moe_graph.layer_count == 1u && moe_graph.expert_count == experts &&
           moe_graph.experts_per_token == 1u && moe_graph.layers[0].router_tensor == 2u &&
           moe_graph.layers[0].gate_up_tensor == 0u && moe_graph.layers[0].down_tensor == 1u);
    std::fill(actual.begin(), actual.end(), 0.0f);
    assert(lm_model_execute_mixtral_moe_layer_f32_router_q4_k_cpu(model, &moe_graph, 0u, hidden, intermediate,
                                                                   router_scratch.data(), router_scratch.size(),
                                                                   gate_scratch.data(), gate_scratch.size(),
                                                                   down_scratch.data(), down_scratch.size(), input.data(),
                                                                   actual.data(), actual.size()) == LM_OK);
    for (uint32_t i = 0u; i < hidden; ++i) assert(actual[i] == expected[i]);
    assert(lm_model_execute_mixtral_moe_layer_f32_router_q4_k_cpu(model, &moe_graph, 1u, hidden, intermediate,
                                                                   router_scratch.data(), router_scratch.size(),
                                                                   gate_scratch.data(), gate_scratch.size(),
                                                                   down_scratch.data(), down_scratch.size(), input.data(),
                                                                   actual.data(), actual.size()) == LM_ERR_ARGUMENT);
    lm_model_close(model);
    std::remove(path);
}

static void test_safetensors_native_mlp() {
    const char *path = "test-native-f32.safetensors";
    const char *json = "{\"__metadata__\":{\"general.architecture\":\"llama\",\"llama.context_length\":\"4\",\"llama.embedding_length\":\"2\",\"llama.block_count\":\"1\",\"llama.attention.head_count\":\"1\",\"llama.attention.head_count_kv\":\"1\",\"llama.feed_forward_length\":\"2\",\"llama.rope.freq_base\":\"10000\",\"tokenizer.token.0\":\"a\",\"tokenizer.token.1\":\"b\"},\"token_embd.weight\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]},\"output.weight\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[16,32]},\"output_norm.weight\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[32,40]},\"blk.0.attn_norm.weight\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[40,48]},\"blk.0.attn_q.weight\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[48,64]},\"blk.0.attn_k.weight\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[64,80]},\"blk.0.attn_v.weight\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[80,96]},\"blk.0.attn_output.weight\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[96,112]},\"blk.0.ffn_norm.weight\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[112,120]},\"blk.0.ffn_gate.weight\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[120,136]},\"blk.0.ffn_down.weight\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[136,152]},\"blk.0.ffn_up.weight\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[152,168]}}";
    {
        std::ofstream file(path, std::ios::binary);
        const uint64_t header_bytes = std::strlen(json);
        file.write(reinterpret_cast<const char *>(&header_bytes), sizeof(header_bytes));
        file.write(json, static_cast<std::streamsize>(header_bytes));
        std::vector<float> payload(42u, 0.0f);
        payload[0] = 1.0f; payload[3] = 1.0f;
        payload[4] = 1.0f; payload[7] = 1.0f;
        for (uint32_t i = 0u; i < 2u; ++i) payload[8u + i] = 1.0f;
        for (uint32_t i = 0u; i < 2u; ++i) payload[10u + i] = 1.0f;
        payload[22] = 1.0f; payload[25] = 1.0f;
        payload[34] = 1.0f; payload[37] = 1.0f;
        file.write(reinterpret_cast<const char *>(payload.data()), static_cast<std::streamsize>(payload.size() * sizeof(float)));
    }
    char error[128] = {};
    lm_model_file *model = nullptr;
    assert(lm_model_open(path, &model, error, sizeof(error)) == LM_OK);
    lm_decoder_graph_binding graph{};
    assert(lm_model_build_llama_graph(model, &graph) == LM_OK && graph.layer_count == 1u);
    lm_model_architecture parsed_architecture{};
    assert(lm_model_get_architecture(model, &parsed_architecture) == LM_OK);
    assert(parsed_architecture.context_length == 4u && parsed_architecture.embedding_length == 2u &&
           parsed_architecture.block_count == 1u && parsed_architecture.head_count == 1u &&
           parsed_architecture.head_count_kv == 1u && parsed_architecture.intermediate_length == 2u &&
           std::fabs(parsed_architecture.rope_frequency_base - 10000.0f) < 1.0e-3f);
    uint32_t token_count = 0u;
    assert(lm_model_token_count(model, &token_count) == LM_OK && token_count == 2u);
    char token[8] = {};
    size_t token_bytes = 0u;
    assert(lm_model_token_at(model, 1u, token, sizeof(token), &token_bytes) == LM_OK &&
           token_bytes == 1u && token[0] == 'b');
    const char text[] = "ab";
    uint32_t encoded[2] = {};
    size_t encoded_count = 0u;
    assert(lm_model_token_encode(model, text, 2u, encoded, 2u, &encoded_count) == LM_OK &&
           encoded_count == 2u && encoded[0] == 0u && encoded[1] == 1u);
    char decoded[8] = {};
    size_t decoded_bytes = 0u;
    assert(lm_model_token_decode(model, encoded, encoded_count, decoded, sizeof(decoded), &decoded_bytes) == LM_OK &&
           decoded_bytes == 2u && std::memcmp(decoded, text, 2u) == 0);
    lm_native_mlp_config config{};
    config.matvec = {LM_BACKEND_CPU, 0u, nullptr, nullptr};
    config.layer_index = 0u;
    config.vocab_size = 2u;
    config.hidden_size = 2u;
    config.intermediate_size = 2u;
    config.matrix_format = LM_QUANT_NONE;
    config.rms_epsilon = 1.0e-5f;
    unsigned char scratch[256] = {};
    float logits[2] = {};
    assert(lm_model_execute_native_mlp_logits(model, &graph, &config, 0u, scratch, sizeof(scratch), logits, 2u) == LM_OK);
    const float scale = 1.0f / std::sqrt(0.5f + config.rms_epsilon);
    assert(std::fabs(logits[0] - scale) < 1.0e-5f && std::fabs(logits[1]) < 1.0e-5f);
    lm_native_mlp_config bad_backend = config;
    bad_backend.matvec.backend = LM_BACKEND_VULKAN;
    bad_backend.matvec.shader_path = "dot_f32_scalar.comp.spv";
    float backend_logits[2] = {};
#if LM_HAS_VULKAN
    assert(lm_model_execute_native_mlp_logits(model, &graph, &bad_backend, 0u, scratch, sizeof(scratch), backend_logits, 2u) == LM_OK);
    assert(std::fabs(backend_logits[0] - logits[0]) < 1.0e-5f &&
           std::fabs(backend_logits[1] - logits[1]) < 1.0e-5f);
#else
    assert(lm_model_execute_native_mlp_logits(model, &graph, &bad_backend, 0u, scratch, sizeof(scratch), backend_logits, 2u) == LM_ERR_UNSUPPORTED);
#endif
#if LM_HAS_VULKAN
    lm_vulkan_f32_context *context = nullptr;
    assert(lm_vulkan_f32_context_create("matvec_f32.comp.spv", 0u, &context) == LM_OK);
    bad_backend.matvec.vulkan_context = context;
    float reused_logits[2] = {};
    assert(lm_model_execute_native_mlp_logits(model, &graph, &bad_backend, 0u, scratch, sizeof(scratch), reused_logits, 2u) == LM_OK);
    assert(std::fabs(reused_logits[0] - logits[0]) < 1.0e-5f &&
           std::fabs(reused_logits[1] - logits[1]) < 1.0e-5f);
    assert(lm_model_execute_native_mlp_logits(model, &graph, &bad_backend, 0u, scratch, sizeof(scratch), reused_logits, 2u) == LM_OK);
    lm_vulkan_f32_context_destroy(context);
#endif
    lm_native_transformer_config transformer{};
    transformer.step.matvec = config.matvec;
    transformer.step.layer_index = 0u;
    transformer.step.vocab_size = 2u;
    transformer.step.hidden_size = 2u;
    transformer.step.intermediate_size = 2u;
    transformer.step.head_count = 1u;
    transformer.step.head_count_kv = 1u;
    transformer.step.token_offset = 0u;
    transformer.step.use_rope = 0u;
    transformer.step.rms_epsilon = config.rms_epsilon;
    transformer.step.matrix_format = LM_QUANT_NONE;
    transformer.has_architecture = 1u;
    transformer.architecture = {4u, 2u, 1u, 1u, 1u, 2u, 10000.0f};
    lm_kv_cache *transformer_cache = nullptr;
    lm_kv_cache *step_cache = nullptr;
    assert(lm_kv_cache_create_with_payload(1u, 4u, 8u, 8u, &transformer_cache) == LM_OK);
    assert(lm_kv_cache_create_with_payload(1u, 4u, 8u, 8u, &step_cache) == LM_OK);
    uint32_t transformer_page = UINT32_MAX;
    uint32_t step_page = UINT32_MAX;
    assert(lm_kv_cache_append(transformer_cache, &transformer_page, 1u) == LM_OK);
    assert(lm_kv_cache_append(step_cache, &step_page, 1u) == LM_OK);
    lm_kv_cache *layer_caches[1] = {transformer_cache};
    float transformer_logits[2] = {};
    float step_logits[2] = {};
    assert(lm_model_execute_native_transformer(model, &graph, &transformer, 0u, layer_caches, 1u,
                                               transformer_page, scratch, sizeof(scratch),
                                               transformer_logits, 2u) == LM_OK);
    assert(lm_model_execute_native_step(model, &graph, &transformer.step, 0u, step_cache, step_page,
                                        scratch, sizeof(scratch), step_logits, 2u) == LM_OK);
    assert(std::fabs(transformer_logits[0] - step_logits[0]) < 1.0e-6f);
    assert(std::fabs(transformer_logits[1] - step_logits[1]) < 1.0e-6f);
    transformer.has_architecture = 0u;
    assert(lm_model_execute_native_transformer(model, &graph, &transformer, 0u, layer_caches, 1u,
                                               transformer_page, scratch, sizeof(scratch),
                                               transformer_logits, 2u) == LM_OK);
    lm_native_generation_config generation{};
    generation.step = transformer.step;
    generation.step.token_offset = 0u;
    generation.sampling = {LM_SAMPLING_GREEDY, 0u, 1.0f, 1u};
    generation.max_new_tokens = 1u;
    generation.has_architecture = 1u;
    generation.architecture = transformer.architecture;
    const uint32_t prompt_tokens[1] = {0u};
    uint32_t generated_tokens[1] = {};
    size_t generated_count = 0u;
    assert(lm_model_generate_native(model, &graph, &generation, prompt_tokens, 1u, scratch, sizeof(scratch),
                                    generated_tokens, 1u, &generated_count) == LM_OK);
    assert(generated_count == 1u && generated_tokens[0] < 2u);
    lm_kv_cache_destroy(transformer_cache);
    lm_kv_cache_destroy(step_cache);
    lm_model_close(model);
    std::remove(path);
}

static void test_model_bound_f32_router() {
    const char *path = "test-model-bound-router.safetensors";
    const char *json = "{\"router.weight\":{\"dtype\":\"F32\",\"shape\":[3,2],\"data_offsets\":[0,24]}}";
    {
        std::ofstream file(path, std::ios::binary);
        const uint64_t header_bytes = std::strlen(json);
        file.write(reinterpret_cast<const char *>(&header_bytes), sizeof(header_bytes));
        file.write(json, static_cast<std::streamsize>(header_bytes));
        const float weights[6] = {1.0f, 0.0f, 0.0f, 2.0f, -1.0f, 1.0f};
        file.write(reinterpret_cast<const char *>(weights), sizeof(weights));
    }
    lm_model_file *model = nullptr;
    char error[128] = {};
    assert(lm_model_open(path, &model, error, sizeof(error)) == LM_OK);
    const float input[2] = {1.0f, 1.0f};
    unsigned char scratch[24] = {};
    lm_moe_route route{};
    assert(lm_model_moe_route_f32_cpu(model, 0u, scratch, sizeof(scratch), input, 2u, 3u, 1u,
                                      LM_MOE_SOFTMAX_SELECTED_ONLY, &route) == LM_OK);
    assert(route.expert_count == 3u && route.experts_per_token == 1u && route.selected[0] == 1u);
    assert(std::fabs(route.weights[0] - 1.0f) < 1.0e-6f);
    assert(lm_model_moe_route_f32_cpu(model, 0u, scratch, sizeof(scratch) - 1u, input, 2u, 3u, 1u,
                                      LM_MOE_SOFTMAX_SELECTED_ONLY, &route) == LM_ERR_CAPACITY);
    lm_model_close(model);
    std::remove(path);
}

static void test_native_mlp_profile() {
    auto put_u32 = [](std::vector<unsigned char> *bytes, uint32_t value) {
        for (unsigned i = 0u; i < 4u; ++i) bytes->push_back(static_cast<unsigned char>((value >> (8u * i)) & 0xffu));
    };
    auto put_u64 = [](std::vector<unsigned char> *bytes, uint64_t value) {
        for (unsigned i = 0u; i < 8u; ++i) bytes->push_back(static_cast<unsigned char>((value >> (8u * i)) & 0xffu));
    };
    struct Spec {
        const char *name;
        uint32_t type;
        uint32_t rank;
        uint64_t d0;
        uint64_t d1;
        uint64_t bytes;
        uint64_t offset;
    };
    std::vector<Spec> specs;
    uint64_t cursor = 0u;
    const auto align32 = [](uint64_t value) { return (value + 31u) & ~static_cast<uint64_t>(31u); };
    const auto add = [&specs, &cursor, &align32](const char *name, uint32_t type, uint32_t rank,
                                                  uint64_t d0, uint64_t d1, uint64_t bytes) {
        cursor = align32(cursor);
        specs.push_back({name, type, rank, d0, d1, bytes, cursor});
        cursor += bytes;
    };
    const uint32_t vocab = 32u;
    const uint32_t hidden = 64u;
    const uint32_t intermediate = 64u;
    add("token_embd.weight", 8u, 2u, hidden, vocab, static_cast<uint64_t>(hidden) * 34u);
    add("output.weight", 8u, 2u, vocab, hidden, static_cast<uint64_t>(vocab) * 68u);
    add("output_norm.weight", LM_DTYPE_F32, 1u, hidden, 0u, hidden * sizeof(float));
    add("blk.0.attn_norm.weight", LM_DTYPE_F32, 1u, hidden, 0u, hidden * sizeof(float));
    add("blk.0.attn_q.weight", 8u, 2u, hidden, hidden, static_cast<uint64_t>(hidden) * 68u);
    add("blk.0.attn_k.weight", 8u, 2u, hidden / 2u, hidden, static_cast<uint64_t>(hidden / 2u) * 68u);
    add("blk.0.attn_v.weight", 8u, 2u, hidden / 2u, hidden, static_cast<uint64_t>(hidden / 2u) * 68u);
    add("blk.0.attn_output.weight", 8u, 2u, hidden, hidden, static_cast<uint64_t>(hidden) * 68u);
    add("blk.0.ffn_norm.weight", LM_DTYPE_F32, 1u, hidden, 0u, hidden * sizeof(float));
    add("blk.0.ffn_gate.weight", 8u, 2u, intermediate, hidden, static_cast<uint64_t>(intermediate) * 68u);
    add("blk.0.ffn_down.weight", 8u, 2u, hidden, intermediate, static_cast<uint64_t>(hidden) * 68u);
    add("blk.0.ffn_up.weight", 8u, 2u, intermediate, hidden, static_cast<uint64_t>(intermediate) * 68u);
    add("blk.1.attn_norm.weight", LM_DTYPE_F32, 1u, hidden, 0u, hidden * sizeof(float));
    add("blk.1.attn_q.weight", 8u, 2u, hidden, hidden, static_cast<uint64_t>(hidden) * 68u);
    add("blk.1.attn_k.weight", 8u, 2u, hidden / 2u, hidden, static_cast<uint64_t>(hidden / 2u) * 68u);
    add("blk.1.attn_v.weight", 8u, 2u, hidden / 2u, hidden, static_cast<uint64_t>(hidden / 2u) * 68u);
    add("blk.1.attn_output.weight", 8u, 2u, hidden, hidden, static_cast<uint64_t>(hidden) * 68u);
    add("blk.1.ffn_norm.weight", LM_DTYPE_F32, 1u, hidden, 0u, hidden * sizeof(float));
    add("blk.1.ffn_gate.weight", 8u, 2u, intermediate, hidden, static_cast<uint64_t>(intermediate) * 68u);
    add("blk.1.ffn_down.weight", 8u, 2u, hidden, intermediate, static_cast<uint64_t>(hidden) * 68u);
    add("blk.1.ffn_up.weight", 8u, 2u, intermediate, hidden, static_cast<uint64_t>(intermediate) * 68u);

    std::vector<unsigned char> gguf(24u, 0u);
    gguf[0] = 'G'; gguf[1] = 'G'; gguf[2] = 'U'; gguf[3] = 'F';
    gguf[4] = 3u;
    gguf[16] = 1u;
    auto put_key_string = [&gguf, &put_u32, &put_u64](const char *key, const char *value) {
        put_u64(&gguf, std::strlen(key));
        gguf.insert(gguf.end(), key, key + std::strlen(key));
        put_u32(&gguf, 8u);
        put_u64(&gguf, std::strlen(value));
        gguf.insert(gguf.end(), value, value + std::strlen(value));
    };
    auto put_key_u32 = [&gguf, &put_u32, &put_u64](const char *key, uint32_t value) {
        put_u64(&gguf, std::strlen(key));
        gguf.insert(gguf.end(), key, key + std::strlen(key));
        put_u32(&gguf, 4u);
        put_u32(&gguf, value);
    };
    auto put_key_f32 = [&gguf, &put_u32, &put_u64](const char *key, float value) {
        uint32_t bits = 0u;
        std::memcpy(&bits, &value, sizeof(bits));
        put_u64(&gguf, std::strlen(key));
        gguf.insert(gguf.end(), key, key + std::strlen(key));
        put_u32(&gguf, 6u);
        put_u32(&gguf, bits);
    };
    put_key_string("general.architecture", "llama");
    put_key_u32("llama.context_length", 4096u);
    put_key_u32("llama.embedding_length", hidden);
    put_key_u32("llama.block_count", 2u);
    put_key_u32("llama.attention.head_count", 2u);
    put_key_u32("llama.attention.head_count_kv", 1u);
    put_key_u32("llama.feed_forward_length", intermediate);
    put_key_f32("llama.rope.freq_base", 10000.0f);
    const char *vocabulary_key = "tokenizer.ggml.tokens";
    put_u64(&gguf, std::strlen(vocabulary_key));
    gguf.insert(gguf.end(), vocabulary_key, vocabulary_key + std::strlen(vocabulary_key));
    put_u32(&gguf, 9u);
    put_u32(&gguf, 8u);
    put_u64(&gguf, vocab);
    for (uint32_t i = 0u; i < vocab; ++i) {
        const char *token = i == 0u ? "<s>" : (i == 1u ? "a" : (i == 2u ? "ab" : (i == 3u ? "b" : "")));
        put_u64(&gguf, std::strlen(token));
        gguf.insert(gguf.end(), token, token + std::strlen(token));
    }
    gguf[8] = static_cast<unsigned char>(specs.size());
    gguf[16] = 9u;
    for (const Spec &spec : specs) {
        const size_t length = std::strlen(spec.name);
        put_u64(&gguf, length);
        gguf.insert(gguf.end(), spec.name, spec.name + length);
        put_u32(&gguf, spec.rank);
        put_u64(&gguf, spec.d0);
        if (spec.rank == 2u) put_u64(&gguf, spec.d1);
        put_u32(&gguf, spec.type);
        put_u64(&gguf, spec.offset);
    }
    const size_t header_size = (gguf.size() + 31u) & ~static_cast<size_t>(31u);
    gguf.resize(header_size + static_cast<size_t>(cursor), 0u);
    const auto payload = [&gguf, header_size](uint64_t offset) { return gguf.data() + header_size + offset; };
    const auto fill_q8 = [&payload](uint64_t offset, uint32_t rows, uint32_t columns, const auto &row_value) {
        const uint32_t blocks = columns / 32u;
        for (uint32_t row = 0u; row < rows; ++row) {
            unsigned char *row_base = payload(offset) + static_cast<size_t>(row) * blocks * 34u;
            for (uint32_t block = 0u; block < blocks; ++block) {
                unsigned char *base = row_base + static_cast<size_t>(block) * 34u;
                base[1] = 0x3cu;
                for (uint32_t i = 0u; i < 32u; ++i) base[2u + i] = row_value(row);
            }
        }
    };
    fill_q8(specs[0].offset, hidden, vocab, [](uint32_t) { return static_cast<unsigned char>(1u); });
    fill_q8(specs[1].offset, vocab, hidden, [](uint32_t row) {
        return row == 0u ? static_cast<unsigned char>(1u) : (row == 1u ? static_cast<unsigned char>(2u) : static_cast<unsigned char>(0u));
    });
    fill_q8(specs[4].offset, hidden, hidden, [](uint32_t) { return static_cast<unsigned char>(1u); });
    fill_q8(specs[5].offset, hidden / 2u, hidden, [](uint32_t) { return static_cast<unsigned char>(1u); });
    fill_q8(specs[6].offset, hidden / 2u, hidden, [](uint32_t) { return static_cast<unsigned char>(1u); });
    fill_q8(specs[7].offset, hidden, hidden, [](uint32_t row) { return row == 0u ? static_cast<unsigned char>(1u) : static_cast<unsigned char>(0u); });
    fill_q8(specs[13].offset, hidden, hidden, [](uint32_t) { return static_cast<unsigned char>(1u); });
    fill_q8(specs[14].offset, hidden / 2u, hidden, [](uint32_t) { return static_cast<unsigned char>(1u); });
    fill_q8(specs[15].offset, hidden / 2u, hidden, [](uint32_t) { return static_cast<unsigned char>(1u); });
    fill_q8(specs[16].offset, hidden, hidden, [](uint32_t row) { return row == 0u ? static_cast<unsigned char>(1u) : static_cast<unsigned char>(0u); });
    fill_q8(specs[18].offset, intermediate, hidden, [](uint32_t) { return static_cast<unsigned char>(1u); });
    fill_q8(specs[19].offset, hidden, intermediate, [](uint32_t) { return static_cast<unsigned char>(0u); });
    fill_q8(specs[20].offset, intermediate, hidden, [](uint32_t) { return static_cast<unsigned char>(1u); });
    fill_q8(specs[9].offset, intermediate, hidden, [](uint32_t) { return static_cast<unsigned char>(1u); });
    fill_q8(specs[10].offset, hidden, intermediate, [](uint32_t) { return static_cast<unsigned char>(0u); });
    fill_q8(specs[11].offset, intermediate, hidden, [](uint32_t) { return static_cast<unsigned char>(1u); });
    for (uint32_t i = 0u; i < hidden; ++i) {
        float one = 1.0f;
        std::memcpy(payload(specs[2].offset) + i * sizeof(float), &one, sizeof(one));
        std::memcpy(payload(specs[3].offset) + i * sizeof(float), &one, sizeof(one));
        std::memcpy(payload(specs[8].offset) + i * sizeof(float), &one, sizeof(one));
        std::memcpy(payload(specs[12].offset) + i * sizeof(float), &one, sizeof(one));
        std::memcpy(payload(specs[17].offset) + i * sizeof(float), &one, sizeof(one));
    }
    const char *path = "test-native-mlp.gguf";
    {
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char *>(gguf.data()), static_cast<std::streamsize>(gguf.size()));
    }
    char error[128] = {};
    lm_model_file *model = nullptr;
    assert(lm_model_open(path, &model, error, sizeof(error)) == LM_OK);
    lm_decoder_graph_binding graph{};
    assert(lm_model_build_llama_graph(model, &graph) == LM_OK);
    assert(graph.layer_count == 2u);
    lm_model_architecture architecture{};
    assert(lm_model_get_architecture(model, &architecture) == LM_OK);
    assert(architecture.context_length == 4096u && architecture.embedding_length == hidden &&
           architecture.block_count == 2u && architecture.head_count == 2u &&
           architecture.head_count_kv == 1u && architecture.intermediate_length == intermediate &&
           architecture.rope_frequency_base == 10000.0f);
    lm_native_mlp_config config{};
    config.matvec = {LM_BACKEND_CPU, 0u, nullptr, nullptr};
    config.layer_index = 0u;
    config.vocab_size = vocab;
    config.hidden_size = hidden;
    config.intermediate_size = intermediate;
    config.matrix_format = LM_QUANT_GGML_Q8_0;
    config.rms_epsilon = 1.0e-5f;
    unsigned char scratch[8192] = {};
    float logits[vocab] = {};
    assert(lm_model_execute_native_mlp_logits(model, &graph, &config, 0u, scratch, sizeof(scratch),
                                              logits, vocab) == LM_OK);
    const float expected_scale = 1.0f / std::sqrt(1.0f + config.rms_epsilon);
    assert(std::fabs(logits[0] - 64.0f * expected_scale) < 1.0e-4f);
    assert(std::fabs(logits[1] - 128.0f * expected_scale) < 1.0e-4f);
    assert(logits[2] == 0.0f);
    lm_kv_cache *kv = nullptr;
    assert(lm_kv_cache_create_with_payload(1u, 2u, (hidden / 2u) * sizeof(float),
                                           (hidden / 2u) * sizeof(float), &kv) == LM_OK);
    uint32_t page = UINT32_MAX;
    assert(lm_kv_cache_append(kv, &page, 1u) == LM_OK);
    uint32_t key_bytes = 0u;
    uint32_t value_bytes = 0u;
    assert(lm_kv_cache_get_payload_layout(kv, &key_bytes, &value_bytes) == LM_OK);
    assert(key_bytes == (hidden / 2u) * sizeof(float) && value_bytes == (hidden / 2u) * sizeof(float));
    lm_native_attention_config attention{};
    attention.matvec = config.matvec;
    attention.layer_index = 0u;
    attention.hidden_size = hidden;
    attention.head_count = 2u;
    attention.head_count_kv = 1u;
    attention.token_offset = 0u;
    attention.rms_epsilon = config.rms_epsilon;
    attention.matrix_format = LM_QUANT_GGML_Q8_0;
    float attention_input[hidden] = {};
    for (float &value : attention_input) value = 1.0f;
    float attention_output[hidden] = {};
    assert(lm_model_execute_native_attention(model, &graph, &attention, attention_input, kv, page,
                                              scratch, sizeof(scratch), attention_output) == LM_OK);
    const float attention_scale = 1.0f / std::sqrt(1.0f + 1.0e-5f);
    assert(std::fabs(attention_output[0] - (1.0f + 4096.0f * attention_scale)) < 1.0e-2f);
    for (uint32_t i = 1u; i < hidden; ++i) assert(attention_output[i] == 1.0f);
    assert(lm_kv_cache_append(kv, &page, 1u) == LM_OK);
    attention.token_offset = 1u;
    std::fill(attention_output, attention_output + hidden, 0.0f);
    assert(lm_model_execute_native_attention(model, &graph, &attention, attention_input, kv, page,
                                              scratch, sizeof(scratch), attention_output) == LM_OK);
    assert(std::fabs(attention_output[0] - (1.0f + 4096.0f * attention_scale)) < 1.0e-2f);
    uint32_t devices = 0u;
    assert(lm_vulkan_device_count(&devices) == LM_OK || devices == 0u);
    if (devices != 0u) {
        lm_native_attention_config vulkan_attention = attention;
        vulkan_attention.matvec = {LM_BACKEND_VULKAN, 0u, "matvec_q8_0_f32.comp.spv", nullptr};
        float vulkan_attention_output[hidden] = {};
        assert(lm_model_execute_native_attention(model, &graph, &vulkan_attention, attention_input,
                                                  kv, page, scratch, sizeof(scratch),
                                                  vulkan_attention_output) == LM_OK);
        for (uint32_t i = 0u; i < hidden; ++i)
            assert(std::fabs(vulkan_attention_output[i] - attention_output[i]) < 1.0e-3f);
    }
    assert(lm_kv_cache_release(kv, page) == LM_OK);
    lm_kv_cache_destroy(kv);
    lm_native_mlp_config bad_shape = config;
    bad_shape.hidden_size = 16u;
    assert(lm_model_execute_native_mlp_logits(model, &graph, &bad_shape, 0u, scratch, sizeof(scratch),
                                              logits, vocab) == LM_ERR_UNSUPPORTED);
    assert(lm_model_execute_native_mlp_logits(model, &graph, &config, 0u, scratch, 1u,
                                              logits, vocab) == LM_ERR_CAPACITY);
    lm_native_generation_config generation{};
    generation.step.matvec = config.matvec;
    generation.step.layer_index = 0u;
    generation.step.vocab_size = vocab;
    generation.step.hidden_size = hidden;
    generation.step.intermediate_size = intermediate;
    generation.step.head_count = 2u;
    generation.step.head_count_kv = 1u;
    generation.step.use_rope = 0u;
    generation.step.rope_theta = 0.0f;
    generation.step.rms_epsilon = config.rms_epsilon;
    generation.step.matrix_format = config.matrix_format;
    generation.sampling = {LM_SAMPLING_GREEDY, 0u, 1.0f, 17u};
    generation.max_new_tokens = 3u;
    generation.has_stop_token = 0u;
    const uint32_t prompt[] = {0u, 0u};
    uint32_t generated[3] = {};
    size_t generated_count = 0u;
    assert(lm_model_generate_native(model, &graph, &generation, prompt, 2u, scratch,
                                    sizeof(scratch), generated, 3u, &generated_count) == LM_OK);
    assert(generated_count == 3u && generated[0] == 1u && generated[1] == 1u && generated[2] == 1u);
    generation.has_stop_token = 1u;
    generation.stop_token = 1u;
    assert(lm_model_generate_native(model, &graph, &generation, prompt, 2u, scratch,
                                    sizeof(scratch), generated, 3u, &generated_count) == LM_OK);
    assert(generated_count == 1u && generated[0] == 1u);
    generation.has_stop_token = 0u;
    assert(lm_model_generate_native(model, &graph, &generation, prompt, 2u, scratch,
                                    sizeof(scratch), generated, 2u, &generated_count) == LM_ERR_ARGUMENT);
    char generated_text[4] = {};
    size_t generated_bytes = 0u;
    assert(lm_model_generate_native_text(model, &graph, &generation, "ab", 2u, scratch,
                                         sizeof(scratch), generated_text, sizeof(generated_text),
                                         &generated_bytes) == LM_OK);
    assert(generated_bytes == 3u && std::strcmp(generated_text, "aaa") == 0);
    assert(lm_model_generate_native_text(model, &graph, &generation, "ab", 2u, scratch,
                                         sizeof(scratch), generated_text, 3u,
                                         &generated_bytes) == LM_ERR_CAPACITY);
    const uint32_t bad_prompt[] = {vocab};
    assert(lm_model_generate_native(model, &graph, &generation, bad_prompt, 1u, scratch,
                                    sizeof(scratch), generated, 3u, &generated_count) == LM_ERR_RANGE);
    if (lm_vulkan_device_count(&devices) == LM_OK && devices != 0u) {
        lm_native_mlp_config vulkan_config = config;
        vulkan_config.matvec = {LM_BACKEND_VULKAN, 0u, "matvec_q8_0_f32.comp.spv", nullptr};
        std::fill(logits, logits + vocab, 0.0f);
        assert(lm_model_execute_native_mlp_logits(model, &graph, &vulkan_config, 0u, scratch,
                                                  sizeof(scratch), logits, vocab) == LM_OK);
        assert(std::fabs(logits[0] - 64.0f * expected_scale) < 1.0e-4f);
        assert(std::fabs(logits[1] - 128.0f * expected_scale) < 1.0e-4f);
    }
    lm_model_close(model);
    std::remove(path);
}

static void test_model_and_cpu_math() {
    const char *gguf = "test-fixture.gguf";
    {
        std::ofstream file(gguf, std::ios::binary);
        const unsigned char header[] = {
            'G','G','U','F', 3,0,0,0,
            1,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,
            1,0,0,0,0,0,0,0,
            'x',
            1,0,0,0,
            1,0,0,0,0,0,0,0,
            0,0,0,0,
            0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0
        };
        file.write(reinterpret_cast<const char *>(header), sizeof(header));
        const unsigned char data[4] = {0,0,0,0};
        file.write(reinterpret_cast<const char *>(data), sizeof(data));
    }
    lm_model_info info;
    char error[128] = {};
    assert(lm_model_inspect(gguf, &info, error, sizeof(error)) == LM_OK);
    assert(info.format == LM_MODEL_GGUF && info.version == 3u && info.tensor_count == 1u);
    lm_model_file *model = nullptr;
    assert(lm_model_open(gguf, &model, error, sizeof(error)) == LM_OK);
    lm_model_info opened_info{};
    assert(lm_model_get_info(model, &opened_info) == LM_OK && opened_info.header_bytes == info.header_bytes);
    lm_model_tensor_info descriptor{};
    assert(lm_model_tensor_info_at(model, 0u, &descriptor) == LM_OK);
    assert(std::strcmp(descriptor.name, "x") == 0 && descriptor.rank == 1u && descriptor.dims[0] == 1u && descriptor.type == 0u && descriptor.relative_offset == 0u);
    assert(lm_model_tensor_info_at(model, 1u, &descriptor) == LM_ERR_RANGE);
    lm_file_span tensor_span{};
    assert(lm_model_tensor_span(model, 0u, 4u, &tensor_span) == LM_OK);
    unsigned char tensor_bytes[4] = {1u, 1u, 1u, 1u};
    assert(lm_file_span_read(&tensor_span, 0u, tensor_bytes, sizeof(tensor_bytes)) == LM_OK);
    assert(tensor_bytes[0] == 0u && tensor_bytes[1] == 0u && tensor_bytes[2] == 0u && tensor_bytes[3] == 0u);
    assert(lm_model_tensor_span(model, info.file_bytes - info.header_bytes + 1u, 1u, &tensor_span) == LM_ERR_RANGE);
    lm_model_close(model);
    std::remove(gguf);

    const char *moe_path = "test-moe.gguf";
    std::vector<unsigned char> moe(24u, 0u);
    moe[0] = 'G'; moe[1] = 'G'; moe[2] = 'U'; moe[3] = 'F';
    moe[4] = 3u;
    moe[16] = 2u;
    auto put_u32 = [&moe](uint32_t value) {
        for (unsigned i = 0u; i < 4u; ++i) moe.push_back(static_cast<unsigned char>((value >> (8u * i)) & 0xffu));
    };
    auto put_u64 = [&moe](uint64_t value) {
        for (unsigned i = 0u; i < 8u; ++i) moe.push_back(static_cast<unsigned char>((value >> (8u * i)) & 0xffu));
    };
    auto put_key = [&moe, &put_u32, &put_u64](const char *key, uint32_t value) {
        const size_t length = std::strlen(key);
        put_u64(length);
        for (size_t i = 0u; i < length; ++i) moe.push_back(static_cast<unsigned char>(key[i]));
        put_u32(4u);
        put_u32(value);
    };
    put_key("llama.expert_count", 8u);
    put_key("llama.expert_used_count", 2u);
    moe.resize((moe.size() + 31u) & ~static_cast<size_t>(31u), 0u);
    { std::ofstream file(moe_path, std::ios::binary); file.write(reinterpret_cast<const char *>(moe.data()), static_cast<std::streamsize>(moe.size())); }
    assert(lm_model_inspect(moe_path, &info, error, sizeof(error)) == LM_OK);
    assert(info.expert_count == 8u && info.experts_per_token == 2u);
    std::remove(moe_path);

    const float a[] = {1.0f, 2.0f, 3.0f};
    const float b[] = {4.0f, 5.0f, 6.0f};
    float dot = 0.0f;
    assert(lm_cpu_dot_f32(a, b, 3u, &dot) == LM_OK);
    assert(dot == 32.0f);
    const float logits[] = {0.0f, 1.0f, 2.0f};
    float probabilities[3] = {};
    assert(lm_cpu_softmax_f32(logits, probabilities, 3u) == LM_OK);
    assert(probabilities[2] > probabilities[1] && probabilities[1] > probabilities[0]);
    assert(probabilities[0] + probabilities[1] + probabilities[2] > 0.999f);
}

static void test_safetensors_parser() {
    const char *valid_path = "test-fixture.safetensors";
    const char *json = "{\"x\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]},\"m\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[8,24]}}";
    {
        std::ofstream file(valid_path, std::ios::binary);
        const uint64_t header_bytes = std::strlen(json);
        file.write(reinterpret_cast<const char *>(&header_bytes), sizeof(header_bytes));
        file.write(json, static_cast<std::streamsize>(header_bytes));
        const float matrix[] = {1.0f, 2.0f, 3.0f, 4.0f};
        const unsigned char data[8] = {};
        file.write(reinterpret_cast<const char *>(data), sizeof(data));
        file.write(reinterpret_cast<const char *>(matrix), sizeof(matrix));
    }
    lm_model_info info{};
    char error[128] = {};
    assert(lm_model_inspect(valid_path, &info, error, sizeof(error)) == LM_OK);
    assert(info.format == LM_MODEL_SAFETENSORS && info.tensor_count == 2u);
    lm_model_file *model = nullptr;
    assert(lm_model_open(valid_path, &model, error, sizeof(error)) == LM_OK);
    lm_model_tensor_info descriptor{};
    assert(lm_model_tensor_info_at(model, 0u, &descriptor) == LM_OK);
    assert(std::strcmp(descriptor.name, "x") == 0 && descriptor.rank == 1u && descriptor.dims[0] == 2u && descriptor.type == LM_DTYPE_F32 && descriptor.relative_offset == 0u);
    assert(lm_model_tensor_info_at(model, 1u, &descriptor) == LM_OK);
    assert(std::strcmp(descriptor.name, "m") == 0 && descriptor.rank == 2u && descriptor.dims[0] == 2u && descriptor.dims[1] == 2u);
    lm_model_tensor_binding binding{};
    assert(lm_model_tensor_bind_native(model, 0u, &binding) == LM_ERR_UNSUPPORTED);
    float matrix_scratch[4] = {};
    const float input[] = {1.0f, 2.0f};
    float output[2] = {};
    assert(lm_model_tensor_matvec_f32_cpu(model, 1u, matrix_scratch, sizeof(matrix_scratch),
                                          2u, 2u, input, output) == LM_OK);
    assert(output[0] == 5.0f && output[1] == 11.0f);
    assert(lm_model_tensor_matvec_f32_cpu(model, 1u, matrix_scratch, sizeof(float),
                                          2u, 2u, input, output) == LM_ERR_CAPACITY);
    lm_model_close(model);
    std::remove(valid_path);

    const char *bad_path = "bad-fixture.safetensors";
    const char *bad_json = "{\"a\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4]},\"b\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[2,6]}}";
    {
        std::ofstream file(bad_path, std::ios::binary);
        const uint64_t header_bytes = std::strlen(bad_json);
        file.write(reinterpret_cast<const char *>(&header_bytes), sizeof(header_bytes));
        file.write(bad_json, static_cast<std::streamsize>(header_bytes));
        const unsigned char data[6] = {};
        file.write(reinterpret_cast<const char *>(data), sizeof(data));
    }
    assert(lm_model_inspect(bad_path, &info, error, sizeof(error)) == LM_ERR_PARSE);
    std::remove(bad_path);
}

static void test_tokenizer() {
    auto put_u32 = [](std::vector<unsigned char> *bytes, uint32_t value) {
        for (unsigned i = 0u; i < 4u; ++i) bytes->push_back(static_cast<unsigned char>((value >> (8u * i)) & 0xffu));
    };
    auto put_u64 = [](std::vector<unsigned char> *bytes, uint64_t value) {
        for (unsigned i = 0u; i < 8u; ++i) bytes->push_back(static_cast<unsigned char>((value >> (8u * i)) & 0xffu));
    };
    const char *tokens[] = {"<s>", "a", "ab", "b"};
    std::vector<unsigned char> gguf(24u, 0u);
    gguf[0] = 'G'; gguf[1] = 'G'; gguf[2] = 'U'; gguf[3] = 'F'; gguf[4] = 3u;
    gguf[16] = 1u;
    const char *key = "tokenizer.ggml.tokens";
    put_u64(&gguf, std::strlen(key));
    gguf.insert(gguf.end(), key, key + std::strlen(key));
    put_u32(&gguf, 9u);
    put_u32(&gguf, 8u);
    put_u64(&gguf, 4u);
    for (const char *token : tokens) {
        put_u64(&gguf, std::strlen(token));
        gguf.insert(gguf.end(), token, token + std::strlen(token));
    }
    gguf.resize((gguf.size() + 31u) & ~static_cast<size_t>(31u), 0u);
    const char *path = "test-tokenizer.gguf";
    {
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char *>(gguf.data()), static_cast<std::streamsize>(gguf.size()));
    }
    char error[128] = {};
    lm_model_file *model = nullptr;
    assert(lm_model_open(path, &model, error, sizeof(error)) == LM_OK);
    uint32_t token_count = 0u;
    assert(lm_model_token_count(model, &token_count) == LM_OK && token_count == 4u);
    size_t token_bytes = 0u;
    assert(lm_model_token_at(model, 2u, nullptr, 0u, &token_bytes) == LM_OK && token_bytes == 2u);
    char token_text[3] = {};
    assert(lm_model_token_at(model, 2u, token_text, sizeof(token_text), &token_bytes) == LM_OK);
    assert(std::strcmp(token_text, "ab") == 0 && token_bytes == 2u);
    assert(lm_model_token_at(model, 2u, token_text, 2u, &token_bytes) == LM_ERR_CAPACITY);
    const char text[] = "aba";
    uint32_t encoded[2] = {};
    size_t encoded_count = 0u;
    assert(lm_model_token_encode(model, text, 3u, encoded, 2u, &encoded_count) == LM_OK);
    assert(encoded_count == 2u && encoded[0] == 2u && encoded[1] == 1u);
    assert(lm_model_token_encode(model, text, 3u, encoded, 1u, &encoded_count) == LM_ERR_CAPACITY);
    assert(encoded_count == 2u);
    assert(lm_model_token_encode(model, "z", 1u, encoded, 2u, &encoded_count) == LM_ERR_UNSUPPORTED);
    char decoded[4] = {};
    size_t decoded_bytes = 0u;
    assert(lm_model_token_decode(model, encoded, 2u, decoded, sizeof(decoded), &decoded_bytes) == LM_OK);
    assert(decoded_bytes == 3u && std::strcmp(decoded, "aba") == 0);
    assert(lm_model_token_decode(model, encoded, 2u, decoded, 3u, &decoded_bytes) == LM_ERR_CAPACITY);
    encoded[0] = 99u;
    assert(lm_model_token_decode(model, encoded, 1u, decoded, sizeof(decoded), &decoded_bytes) == LM_ERR_RANGE);
    lm_model_close(model);
    std::remove(path);
}

static void test_sampling() {
    const float logits[] = {1.0f, 3.0f, 3.0f, -2.0f};
    lm_sampling_config greedy{LM_SAMPLING_GREEDY, 0u, 1.0f, 1u};
    uint32_t token = 99u;
    float probability = 0.0f;
    assert(lm_sample_logits(logits, 4u, &greedy, &token, &probability) == LM_OK);
    assert(token == 1u && probability == 1.0f);
    lm_sampling_config top_k{LM_SAMPLING_TOP_K, 2u, 1.0f, 123u};
    uint32_t first_token = 99u;
    float first_probability = 0.0f;
    assert(lm_sample_logits(logits, 4u, &top_k, &first_token, &first_probability) == LM_OK);
    assert((first_token == 1u || first_token == 2u) && first_probability > 0.0f && first_probability < 1.0f);
    uint32_t second_token = 99u;
    float second_probability = 0.0f;
    assert(lm_sample_logits(logits, 4u, &top_k, &second_token, &second_probability) == LM_OK);
    assert(first_token == second_token && first_probability == second_probability);
    lm_sampling_config invalid{LM_SAMPLING_TOP_K, 5u, 1.0f, 1u};
    assert(lm_sample_logits(logits, 4u, &invalid, &token, &probability) == LM_ERR_ARGUMENT);
    assert(lm_sample_logits(nullptr, 4u, &greedy, &token, &probability) == LM_ERR_ARGUMENT);
}

static void test_cpu_decoder() {
    lm_model_tensor_info mapped_descriptor{};
    std::strcpy(mapped_descriptor.name, "blk.7.attn_q.weight");
    mapped_descriptor.rank = 2u;
    mapped_descriptor.dims[0] = 256u;
    mapped_descriptor.dims[1] = 256u;
    mapped_descriptor.type = 12u;
    lm_decoder_tensor_mapping mapped{};
    assert(lm_decoder_map_llama_tensor(&mapped_descriptor, &mapped) == LM_OK);
    assert(mapped.role == LM_DECODER_TENSOR_ATTN_Q && mapped.layer_index == 7u &&
           mapped.rank == 2u && mapped.dims[0] == 256u && mapped.type == 12u);
    std::strcpy(mapped_descriptor.name, "token_embd.weight");
    assert(lm_decoder_map_llama_tensor(&mapped_descriptor, &mapped) == LM_OK);
    assert(mapped.role == LM_DECODER_TENSOR_TOKEN_EMBEDDING && mapped.layer_index == 0u);
    std::strcpy(mapped_descriptor.name, "blk.7.attn_q.bias");
    assert(lm_decoder_map_llama_tensor(&mapped_descriptor, &mapped) == LM_ERR_UNSUPPORTED);
    const char *graph_names[] = {
        "token_embd.weight", "output.weight", "output_norm.weight",
        "blk.0.attn_norm.weight", "blk.0.attn_q.weight", "blk.0.attn_k.weight",
        "blk.0.attn_v.weight", "blk.0.attn_output.weight", "blk.0.ffn_norm.weight",
        "blk.0.ffn_gate.weight", "blk.0.ffn_down.weight", "blk.0.ffn_up.weight"};
    lm_model_tensor_info graph_descriptors[12] = {};
    for (unsigned i = 0u; i < 12u; ++i) {
        std::strcpy(graph_descriptors[i].name, graph_names[i]);
        graph_descriptors[i].rank = 1u;
        graph_descriptors[i].dims[0] = 256u;
    }
    lm_decoder_graph_plan graph_plan{};
    assert(lm_decoder_graph_plan_build(graph_descriptors, 12u, &graph_plan) == LM_OK);
    assert(graph_plan.layer_count == 1u && graph_plan.global_role_mask == 7u &&
           graph_plan.layer_role_mask[0] == 0xff8u);
    assert(lm_decoder_graph_plan_build(graph_descriptors, 11u, &graph_plan) == LM_ERR_UNSUPPORTED);
    std::strcpy(graph_descriptors[11].name, "blk.0.ffn_up.bias");
    assert(lm_decoder_graph_plan_build(graph_descriptors, 12u, &graph_plan) == LM_ERR_UNSUPPORTED);
    const float embedding[] = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    const float gamma[] = {1.0f, 1.0f};
    const float identity[] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float zeros[] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float output[] = {1.0f, 0.0f, 2.0f, 0.0f, 1.0f, 3.0f};
    lm_cpu_decoder_config config{};
    config.vocab_size = 3u;
    config.hidden_size = 2u;
    config.max_context = 2u;
    config.rms_epsilon = 1.0e-5f;
    config.rope_theta = 10000.0f;
    config.use_rope = 0u;
    config.embedding = embedding;
    config.rms_gamma_1 = gamma;
    config.wq = identity;
    config.wk = identity;
    config.wv = identity;
    config.wo = zeros;
    config.rms_gamma_2 = gamma;
    config.w1 = zeros;
    config.w2 = zeros;
    config.wout = output;
    lm_cpu_decoder *decoder = nullptr;
    assert(lm_cpu_decoder_create(&config, &decoder) == LM_OK);
    float logits[3] = {};
    assert(lm_cpu_decoder_step(decoder, 0u, logits, 3u) == LM_OK);
    assert(logits[0] == 1.0f && logits[1] == 0.0f && logits[2] == 2.0f);
    assert(lm_cpu_decoder_position(decoder) == 1u);
    assert(lm_cpu_decoder_step(decoder, 1u, logits, 3u) == LM_OK);
    assert(logits[0] == 0.0f && logits[1] == 1.0f && logits[2] == 3.0f);
    assert(lm_cpu_decoder_step(decoder, 0u, logits, 3u) == LM_ERR_CAPACITY);
    assert(lm_cpu_decoder_step(decoder, 3u, logits, 3u) == LM_ERR_RANGE);
    assert(lm_cpu_decoder_reset(decoder) == LM_OK && lm_cpu_decoder_position(decoder) == 0u);
    assert(lm_cpu_decoder_step(decoder, 0u, logits, 3u) == LM_OK);
    assert(logits[0] == 1.0f && logits[1] == 0.0f && logits[2] == 2.0f);
    lm_cpu_decoder_destroy(decoder);
}

static void test_moe_router() {
    const float logits[] = {1.0f, 3.0f, 3.0f, -2.0f};
    lm_moe_route route{};
    assert(lm_cpu_moe_route(logits, 4u, 2u, LM_MOE_SOFTMAX_ALL_THEN_TOPK, &route) == LM_OK);
    assert(route.selected[0] == 1u && route.selected[1] == 2u);
    assert(route.weights[0] == 0.5f && route.weights[1] == 0.5f);
    assert(std::fabs((route.weights[0] + route.weights[1]) - 1.0f) < 1.0e-6f);
    lm_moe_route selected_only{};
    assert(lm_cpu_moe_route(logits, 4u, 2u, LM_MOE_SOFTMAX_SELECTED_ONLY, &selected_only) == LM_OK);
    assert(selected_only.weights[0] == 0.5f && selected_only.weights[1] == 0.5f);
    const float selected_outputs[] = {2.0f, 4.0f, 6.0f, 8.0f};
    float combined[2] = {};
    assert(lm_cpu_moe_combine(&selected_only, selected_outputs, 2u, combined) == LM_OK);
    assert(combined[0] == 4.0f && combined[1] == 6.0f);
    const float bad_output[] = {2.0f, std::numeric_limits<float>::quiet_NaN(), 6.0f, 8.0f};
    assert(lm_cpu_moe_combine(&selected_only, bad_output, 2u, combined) == LM_ERR_RANGE);
    const float bad_logits[] = {0.0f, std::numeric_limits<float>::quiet_NaN()};
    assert(lm_cpu_moe_route(bad_logits, 2u, 1u, LM_MOE_SOFTMAX_ALL_THEN_TOPK, &route) == LM_ERR_RANGE);
    assert(lm_cpu_moe_route(logits, 4u, 3u, LM_MOE_SOFTMAX_ALL_THEN_TOPK, &route) == LM_OK);
    assert(lm_cpu_moe_route(logits, 4u, 0u, LM_MOE_SOFTMAX_ALL_THEN_TOPK, &route) == LM_ERR_ARGUMENT);
    lm_model_tensor_info gate_up{};
    std::strncpy(gate_up.name, "layers.7.feed_forward.experts.w1", sizeof(gate_up.name) - 1u);
    gate_up.rank = 3u;
    gate_up.dims[0] = 4u; gate_up.dims[1] = 8u; gate_up.dims[2] = 3u;
    lm_moe_tensor_mapping mapping{};
    assert(lm_moe_map_mixtral_tensor(&gate_up, 3u, &mapping) == LM_OK);
    assert(mapping.role == LM_MOE_TENSOR_GATE_UP_EXPERT && mapping.layer_index == 7u &&
           mapping.expert_axis == 2u && mapping.expert_count == 3u && mapping.dims[1] == 8u);
    lm_model_tensor_info down = gate_up;
    std::strncpy(down.name, "layers.7.feed_forward.experts.w2", sizeof(down.name) - 1u);
    down.dims[0] = 8u; down.dims[1] = 4u;
    assert(lm_moe_map_mixtral_tensor(&down, 3u, &mapping) == LM_OK);
    assert(mapping.role == LM_MOE_TENSOR_DOWN_EXPERT && mapping.layer_index == 7u);
    lm_model_tensor_info unknown = gate_up;
    std::strncpy(unknown.name, "model.layers.7.mlp.experts.gate_proj", sizeof(unknown.name) - 1u);
    assert(lm_moe_map_mixtral_tensor(&unknown, 3u, &mapping) == LM_ERR_UNSUPPORTED);
    lm_model_tensor_info wrong_experts = gate_up;
    wrong_experts.dims[2] = 4u;
    assert(lm_moe_map_mixtral_tensor(&wrong_experts, 3u, &mapping) == LM_ERR_PARSE);
    lm_model_tensor_info odd_gate = gate_up;
    odd_gate.dims[1] = 7u;
    assert(lm_moe_map_mixtral_tensor(&odd_gate, 3u, &mapping) == LM_ERR_PARSE);
    lm_moe_route mlp_route{};
    mlp_route.expert_count = 2u;
    mlp_route.experts_per_token = 1u;
    mlp_route.selected[0] = 1u;
    mlp_route.weights[0] = 1.0f;
    const uint32_t gate_up_dims[3] = {2u, 8u, 2u};
    const uint32_t down_dims[3] = {4u, 2u, 2u};
    std::vector<float> gate_up_values(32u, 1.0f);
    std::vector<float> down_values(16u, 1.0f);
    lm_tensor gate_up_tensor{};
    lm_tensor down_tensor{};
    assert(lm_tensor_make_view(gate_up_values.data(), gate_up_values.size() * sizeof(float), LM_DTYPE_F32,
                               3u, gate_up_dims, &gate_up_tensor) == LM_OK);
    assert(lm_tensor_make_view(down_values.data(), down_values.size() * sizeof(float), LM_DTYPE_F32,
                               3u, down_dims, &down_tensor) == LM_OK);
    const float mlp_input[2] = {0.5f, 0.5f};
    float selected_mlp_output[2] = {};
    assert(lm_cpu_moe_selected_expert_mlp(&mlp_route, &gate_up_tensor, &down_tensor, 2u, 4u,
                                          mlp_input, selected_mlp_output) == LM_OK);
    const float expected_mlp = 4.0f * (1.0f / (1.0f + std::exp(-1.0f)));
    assert(std::fabs(selected_mlp_output[0] - expected_mlp) < 1.0e-5f &&
           std::fabs(selected_mlp_output[1] - expected_mlp) < 1.0e-5f);
    float combined_mlp[2] = {};
    assert(lm_cpu_moe_combine(&mlp_route, selected_mlp_output, 2u, combined_mlp) == LM_OK);
    assert(std::fabs(combined_mlp[0] - expected_mlp) < 1.0e-5f &&
           std::fabs(combined_mlp[1] - expected_mlp) < 1.0e-5f);
    lm_moe_route native_route{};
    native_route.expert_count = 2u;
    native_route.experts_per_token = 1u;
    native_route.selected[0] = 1u;
    native_route.weights[0] = 1.0f;
    const uint32_t native_gate_up_dims[3] = {256u, 512u, 2u};
    const uint32_t native_down_dims[3] = {256u, 256u, 2u};
    std::vector<unsigned char> native_gate_up(2u * 512u * 144u, 0u);
    std::vector<unsigned char> native_down(2u * 256u * 144u, 0u);
    const auto fill_q4_k = [](std::vector<unsigned char> *bytes, uint32_t blocks) {
        for (uint32_t block = 0u; block < blocks; ++block) {
            const size_t base = static_cast<size_t>(block) * 144u;
            bytes->at(base + 1u) = 0x3cu;
            for (unsigned i = 0u; i < 4u; ++i) bytes->at(base + 4u + i) = 1u;
            for (unsigned i = 0u; i < 4u; ++i) bytes->at(base + 12u + i) = 1u;
            for (unsigned i = 0u; i < 128u; ++i) bytes->at(base + 16u + i) = 0x11u;
        }
    };
    fill_q4_k(&native_gate_up, 2u * 512u);
    fill_q4_k(&native_down, 2u * 256u);
    lm_tensor native_gate_up_tensor{};
    lm_tensor native_down_tensor{};
    assert(lm_tensor_make_q4_k_view(native_gate_up.data(), native_gate_up.size(), 3u,
                                     native_gate_up_dims, &native_gate_up_tensor) == LM_OK);
    assert(lm_tensor_make_q4_k_view(native_down.data(), native_down.size(), 3u,
                                     native_down_dims, &native_down_tensor) == LM_OK);
    std::vector<float> native_input(256u, 1.0f);
    float native_output[256] = {};
    assert(lm_cpu_moe_selected_expert_mlp_q4_k(&native_route, &native_gate_up_tensor,
                                                &native_down_tensor, 256u, 256u,
                                                native_input.data(), native_output) == LM_OK);
    for (float value : native_output) assert(value == 16777216.0f);
}


static void test_kv_pages() {
    lm_kv_cache *cache = nullptr;
    assert(lm_kv_cache_create(2u, 4u, &cache) == LM_OK);
    uint32_t first = UINT32_MAX;
    assert(lm_kv_cache_append(cache, &first, 3u) == LM_OK);
    uint32_t second = UINT32_MAX;
    assert(lm_kv_cache_fork(cache, first, &second) == LM_OK);
    assert(lm_kv_cache_append(cache, &second, 1u) == LM_OK);
    assert(lm_kv_cache_append(cache, &second, 1u) == LM_ERR_CAPACITY);
    lm_kv_stats stats;
    assert(lm_kv_cache_get_stats(cache, &stats) == LM_OK);
    assert(stats.used_pages == 2u && stats.free_pages == 0u && stats.shared_pages == 0u);
    assert(lm_kv_cache_rollback(cache, second, 1u) == LM_OK);
    assert(lm_kv_cache_release(cache, first) == LM_OK);
    assert(lm_kv_cache_release(cache, second) == LM_OK);
    assert(lm_kv_cache_get_stats(cache, &stats) == LM_OK && stats.free_pages == 2u);
    lm_kv_cache_destroy(cache);
}

static void test_kv_payload() {
    lm_kv_cache *metadata_only = nullptr;
    assert(lm_kv_cache_create(1u, 2u, &metadata_only) == LM_OK);
    uint32_t metadata_page = UINT32_MAX;
    assert(lm_kv_cache_append(metadata_only, &metadata_page, 1u) == LM_OK);
    const unsigned char one_key[3] = {1u, 1u, 1u};
    const unsigned char one_value[2] = {2u, 2u};
    assert(lm_kv_cache_write_payload(metadata_only, metadata_page, 0u, 1u,
                                     &one_key, &one_value) == LM_ERR_UNSUPPORTED);
    lm_kv_cache_destroy(metadata_only);

    lm_kv_cache *cache = nullptr;
    assert(lm_kv_cache_create_with_payload(2u, 4u, 3u, 2u, &cache) == LM_OK);
    uint32_t parent = UINT32_MAX;
    assert(lm_kv_cache_append(cache, &parent, 3u) == LM_OK);
    const unsigned char parent_keys[9] = {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u};
    const unsigned char parent_values[6] = {11u, 12u, 13u, 14u, 15u, 16u};
    assert(lm_kv_cache_write_payload(cache, parent, 0u, 3u, parent_keys, parent_values) == LM_OK);
    unsigned char read_keys[9] = {};
    unsigned char read_values[6] = {};
    assert(lm_kv_cache_read_payload(cache, parent, 0u, 3u, read_keys, read_values) == LM_OK);
    assert(std::memcmp(read_keys, parent_keys, sizeof(read_keys)) == 0);
    assert(std::memcmp(read_values, parent_values, sizeof(read_values)) == 0);
    assert(lm_kv_cache_read_payload(cache, parent, 3u, 1u, read_keys, read_values) == LM_ERR_RANGE);
    assert(lm_kv_cache_write_payload(cache, parent, 2u, 2u, parent_keys, parent_values) == LM_ERR_RANGE);

    uint32_t child = UINT32_MAX;
    assert(lm_kv_cache_fork(cache, parent, &child) == LM_OK);
    const unsigned char child_key[3] = {90u, 91u, 92u};
    const unsigned char child_value[2] = {93u, 94u};
    assert(lm_kv_cache_write_payload(cache, child, 1u, 1u, child_key, child_value) == LM_OK);
    assert(lm_kv_cache_read_payload(cache, parent, 1u, 1u, read_keys, read_values) == LM_OK);
    assert(std::memcmp(read_keys, parent_keys + 3u, 3u) == 0);
    assert(std::memcmp(read_values, parent_values + 2u, 2u) == 0);
    assert(lm_kv_cache_read_payload(cache, child, 1u, 1u, read_keys, read_values) == LM_OK);
    assert(std::memcmp(read_keys, child_key, 3u) == 0);
    assert(std::memcmp(read_values, child_value, 2u) == 0);

    assert(lm_kv_cache_rollback(cache, child, 2u) == LM_OK);
    assert(lm_kv_cache_read_payload(cache, child, 1u, 1u, read_keys, read_values) == LM_ERR_RANGE);
    assert(lm_kv_cache_release(cache, parent) == LM_OK);
    assert(lm_kv_cache_release(cache, child) == LM_OK);
    uint32_t reused = UINT32_MAX;
    assert(lm_kv_cache_append(cache, &reused, 1u) == LM_OK);
    assert(lm_kv_cache_write_payload(cache, reused, 0u, 1u, &one_key, &one_value) == LM_OK);
    assert(lm_kv_cache_read_payload(cache, reused, 0u, 1u, read_keys, read_values) == LM_OK);
    assert(read_keys[0] == one_key[0] && read_values[0] == one_value[0]);
    assert(lm_kv_cache_release(cache, reused) == LM_OK);
    lm_kv_cache_destroy(cache);

    lm_kv_cache *invalid = nullptr;
    assert(lm_kv_cache_create_with_payload(1u, 4u, 0u, 2u, &invalid) == LM_ERR_ARGUMENT);
    assert(lm_kv_cache_create_with_payload(UINT32_MAX, UINT32_MAX, UINT32_MAX,
                                           UINT32_MAX, &invalid) == LM_ERR_CAPACITY);
}

static void test_kernel_selection() {
    lm_kernel_caps cpu{0u, 0u, 0u};
    lm_kernel_choice choice{};
    assert(lm_kernel_select(LM_KERNEL_DOT_I8, LM_KERNEL_AUTO, &cpu, &choice) == LM_OK);
    assert(choice.path == LM_KERNEL_CPU_SCALAR);

    lm_kernel_caps dp4{1u, 1u, 1u};
    assert(lm_kernel_select(LM_KERNEL_DOT_I8, LM_KERNEL_AUTO, &dp4, &choice) == LM_OK);
    assert(choice.path == LM_KERNEL_VULKAN_DP4);
    assert(std::strcmp(choice.source_id, "vulkan/dp4") == 0);
    lm_kernel_contract contract{};
    assert(lm_kernel_contract_get(&choice, &contract) == LM_OK);
    assert(contract.input_dtype == LM_DTYPE_I8 && contract.output_dtype == LM_DTYPE_I32);
    assert(contract.minimum_alignment == 4u && contract.deterministic == 1u);
    assert(lm_kernel_select(LM_KERNEL_DOT_F32, LM_KERNEL_VULKAN_DP4, &dp4, &choice) == LM_ERR_UNSUPPORTED);
    assert(lm_kernel_select(LM_KERNEL_DOT_F32, LM_KERNEL_VULKAN_SCALAR, &dp4, &choice) == LM_OK);
    assert(choice.path == LM_KERNEL_VULKAN_SCALAR);
    assert(lm_kernel_select(LM_KERNEL_DOT_Q4_K, LM_KERNEL_AUTO, &dp4, &choice) == LM_OK);
    assert(choice.path == LM_KERNEL_VULKAN_SCALAR && std::strcmp(choice.source_id, "vulkan/q4_k") == 0);
    assert(lm_kernel_contract_get(&choice, &contract) == LM_OK);
    assert(contract.input_dtype == LM_DTYPE_U8 && contract.output_dtype == LM_DTYPE_F32 &&
           contract.minimum_alignment == 1u && contract.deterministic == 1u);
    assert(lm_kernel_select(LM_KERNEL_DOT_Q4_K, LM_KERNEL_AUTO, &cpu, &choice) == LM_OK);
    assert(choice.path == LM_KERNEL_CPU_SCALAR && std::strcmp(choice.source_id, "cpu/reference") == 0);
}

static void test_vulkan_dp4() {
    uint32_t device_count = 0u;
    const lm_status discovered = lm_vulkan_device_count(&device_count);
    if (discovered != LM_OK || device_count == 0u) return;
    const uint32_t a[] = {0x04030201u};
    const uint32_t b[] = {0x08070605u};
    int32_t gpu_result = 0;
    assert(lm_vulkan_dot_i8_dp4("dot_i8_dp4.comp.spv", 0u, a, b, 1u, &gpu_result) == LM_OK);
    assert(gpu_result == 70);
    unsigned char q4_k_bytes[144] = {};
    q4_k_bytes[0] = 0x00u;
    q4_k_bytes[1] = 0x3cu;
    for (unsigned i = 0u; i < 4u; ++i) q4_k_bytes[4u + i] = 1u;
    for (unsigned i = 0u; i < 4u; ++i) q4_k_bytes[12u + i] = 1u;
    for (unsigned i = 0u; i < 128u; ++i) q4_k_bytes[16u + i] = 0x11u;
    float q4_k_input[256] = {};
    for (float &value : q4_k_input) value = 1.0f;
    float q4_k_gpu_result = 0.0f;
    assert(lm_vulkan_dot_q4_k("dot_q4_k_f32.comp.spv", 0u, q4_k_bytes, 1u,
                              q4_k_input, &q4_k_gpu_result) == LM_OK);
    assert(q4_k_gpu_result == 256.0f);
    unsigned char q8_0_bytes[34] = {};
    q8_0_bytes[1] = 0x3cu;
    for (unsigned i = 0u; i < 32u; ++i) q8_0_bytes[2u + i] = 1u;
    float q8_0_input[32] = {};
    for (float &value : q8_0_input) value = 1.0f;
    float q8_0_gpu_result = 0.0f;
    assert(lm_vulkan_dot_q8_0("dot_q8_0_f32.comp.spv", 0u, q8_0_bytes, 1u,
                              q8_0_input, &q8_0_gpu_result) == LM_OK);
    assert(q8_0_gpu_result == 32.0f);
    unsigned char q8_0_rows[68] = {};
    q8_0_rows[1] = 0x3cu;
    for (unsigned i = 0u; i < 32u; ++i) q8_0_rows[2u + i] = 1u;
    q8_0_rows[35u] = 0x3cu;
    for (unsigned i = 0u; i < 32u; ++i) q8_0_rows[36u + i] = 2u;
    lm_tensor q8_0_matrix{};
    const uint32_t q8_0_dims[2] = {2u, 32u};
    assert(lm_tensor_make_q8_0_view(q8_0_rows, sizeof(q8_0_rows), 2u, q8_0_dims, &q8_0_matrix) == LM_OK);
    float q8_0_cpu_rows[2] = {};
    assert(lm_cpu_matvec_q8_0(&q8_0_matrix, q8_0_input, 2u, 32u, q8_0_cpu_rows) == LM_OK);
    assert(q8_0_cpu_rows[0] == 32.0f && q8_0_cpu_rows[1] == 64.0f);
    float q8_0_gpu_rows[2] = {};
    assert(lm_vulkan_matvec_q8_0("matvec_q8_0_f32.comp.spv", 0u, q8_0_rows, 2u, 1u,
                                 q8_0_input, q8_0_gpu_rows) == LM_OK);
    assert(q8_0_gpu_rows[0] == q8_0_cpu_rows[0] && q8_0_gpu_rows[1] == q8_0_cpu_rows[1]);
    lm_kernel_choice q8_0_choice{};
    const lm_kernel_caps q8_0_caps{1u, 1u, 1u};
    assert(lm_kernel_select(LM_KERNEL_DOT_Q8_0, LM_KERNEL_VULKAN_SCALAR, &q8_0_caps, &q8_0_choice) == LM_OK);
    assert(std::strcmp(q8_0_choice.source_id, "vulkan/q8_0") == 0);
    lm_kernel_io q8_0_io{q8_0_bytes, q8_0_input, 1u, &q8_0_gpu_result};
    q8_0_gpu_result = 0.0f;
    assert(lm_vulkan_dispatch(&q8_0_choice, "dot_q8_0_f32.comp.spv", 0u, &q8_0_io) == LM_OK);
    assert(q8_0_gpu_result == 32.0f);
    unsigned char q4_k_rows[288] = {};
    std::memcpy(q4_k_rows, q4_k_bytes, sizeof(q4_k_bytes));
    q4_k_rows[144u] = 0x00u;
    q4_k_rows[145u] = 0x3cu;
    for (unsigned i = 0u; i < 4u; ++i) q4_k_rows[148u + i] = 1u;
    for (unsigned i = 0u; i < 4u; ++i) q4_k_rows[156u + i] = 1u;
    for (unsigned i = 0u; i < 128u; ++i) q4_k_rows[160u + i] = 0x22u;
    const uint32_t matvec_dims[2] = {2u, 256u};
    lm_tensor matvec_tensor{};
    assert(lm_tensor_make_q4_k_view(q4_k_rows, sizeof(q4_k_rows), 2u, matvec_dims, &matvec_tensor) == LM_OK);
    float cpu_matvec[2] = {};
    assert(lm_cpu_matvec_q4_k(&matvec_tensor, q4_k_input, 2u, 256u, cpu_matvec) == LM_OK);
    assert(cpu_matvec[0] == 256.0f && cpu_matvec[1] == 512.0f);
    float gpu_matvec[2] = {};
    assert(lm_vulkan_matvec_q4_k("matvec_q4_k_f32.comp.spv", 0u, q4_k_rows, 2u, 1u,
                                 q4_k_input, gpu_matvec) == LM_OK);
    assert(gpu_matvec[0] == cpu_matvec[0] && gpu_matvec[1] == cpu_matvec[1]);
    const float scalar_a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float scalar_b[] = {5.0f, 6.0f, 7.0f, 8.0f};
    float scalar_result = 0.0f;
    assert(lm_vulkan_dot_f32("dot_f32_scalar.comp.spv", 0u, scalar_a, scalar_b, 4u, &scalar_result) == LM_OK);
    assert(scalar_result == 70.0f);
    lm_kernel_caps caps{1u, 1u, 1u};
    lm_kernel_choice scalar_choice{};
    assert(lm_kernel_select(LM_KERNEL_DOT_F32, LM_KERNEL_VULKAN_SCALAR, &caps, &scalar_choice) == LM_OK);
    lm_kernel_io scalar_io{scalar_a, scalar_b, 4u, &scalar_result};
    scalar_result = 0.0f;
    assert(lm_vulkan_dispatch(&scalar_choice, "dot_f32_scalar.comp.spv", 0u, &scalar_io) == LM_OK);
    assert(scalar_result == 70.0f);
    lm_kernel_choice dp4_choice{};
    assert(lm_kernel_select(LM_KERNEL_DOT_I8, LM_KERNEL_VULKAN_DP4, &caps, &dp4_choice) == LM_OK);
    lm_kernel_io dp4_io{a, b, 1u, &gpu_result};
    gpu_result = 0;
    assert(lm_vulkan_dispatch(&dp4_choice, "dot_i8_dp4.comp.spv", 0u, &dp4_io) == LM_OK);
    assert(gpu_result == 70);
    lm_kernel_choice q4_k_choice{};
    assert(lm_kernel_select(LM_KERNEL_DOT_Q4_K, LM_KERNEL_VULKAN_SCALAR, &caps, &q4_k_choice) == LM_OK);
    lm_kernel_io q4_k_io{q4_k_bytes, q4_k_input, 1u, &q4_k_gpu_result};
    q4_k_gpu_result = 0.0f;
    assert(lm_vulkan_dispatch(&q4_k_choice, "dot_q4_k_f32.comp.spv", 0u, &q4_k_io) == LM_OK);
    assert(q4_k_gpu_result == 256.0f);
}

static void test_probe_and_runtime() {
    lm_config config;
    lm_config_init(&config);
    config.trace = 1u;
    lm_runtime *runtime = nullptr;
    assert(lm_runtime_create(&config, &runtime) == LM_OK);
    ProbeState state{0u, 0u};
    lm_runtime_set_probe_sink(runtime, capture_probe, &state);
    lm_probe probe{};
    probe.trace_id = 17u;
    probe.stage = 9u;
    assert(lm_runtime_emit_probe(runtime, &probe) == LM_OK);
    assert(state.count == 1u && state.last_stage == 9u);
    assert(lm_runtime_dump_config(runtime, stdout) == LM_OK);
    lm_runtime_destroy(runtime);
}

int main() {
    test_defaults();
    test_cli();
    test_quantize_policy_gate();
    test_vulkan_backend_resolution();
    test_tensor_and_buffer();
    test_file_access();
    test_native_model_tensor_binding();
    test_model_bound_moe_q4_k();
    test_native_mlp_profile();
    test_model_and_cpu_math();
    test_safetensors_parser();
    test_safetensors_native_mlp();
    test_model_bound_f32_router();
    test_tokenizer();
    test_sampling();
    test_cpu_decoder();
    test_moe_router();
    test_kv_pages();
    test_kv_payload();
    test_kernel_selection();
    test_vulkan_dp4();
    test_probe_and_runtime();
    std::puts("core_tests=PASS");
    return 0;
}
