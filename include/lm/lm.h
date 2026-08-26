#ifndef LM_LM_H
#define LM_LM_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LM_ABI_VERSION 1u
#define LM_PATH_MAX 512u

typedef enum lm_status {
    LM_OK = 0,
    LM_ERR_ARGUMENT = -1,
    LM_ERR_PARSE = -2,
    LM_ERR_RANGE = -3,
    LM_ERR_UNSUPPORTED = -4,
    LM_ERR_CAPACITY = -5,
    LM_ERR_STATE = -6,
    LM_ERR_IO = -7
} lm_status;

typedef enum lm_backend_kind {
    LM_BACKEND_AUTO = 0,
    LM_BACKEND_CPU,
    LM_BACKEND_VULKAN,
    LM_BACKEND_ROCR,
    LM_BACKEND_ROCM,
    LM_BACKEND_CUDA,
    LM_BACKEND_OPENVINO,
    LM_BACKEND_DIRECTML
} lm_backend_kind;

typedef enum lm_load_mode {
    LM_LOAD_EAGER = 0,
    LM_LOAD_MMAP,
    LM_LOAD_LAZY,
    LM_LOAD_STREAM
} lm_load_mode;

typedef enum lm_weight_policy {
    LM_WEIGHT_PRESERVE = 0,
    LM_WEIGHT_QUANTIZE_CACHE
} lm_weight_policy;

typedef enum lm_kv_dtype {
    LM_KV_F16 = 0,
    LM_KV_BF16,
    LM_KV_Q8,
    LM_KV_Q6,
    LM_KV_Q4
} lm_kv_dtype;

typedef struct lm_probe {
    uint64_t trace_id;
    uint64_t parent_id;
    uint32_t kind;
    uint32_t stage;
    uint32_t bytes;
    uint32_t flags;
    uint64_t content_hash;
    uint64_t timestamp_ns;
} lm_probe;

typedef void (*lm_probe_sink)(void *user, const lm_probe *probe);

typedef struct lm_config {
    lm_backend_kind backend;
    lm_backend_kind resolved_backend;
    lm_load_mode load_mode;
    lm_weight_policy weight_policy;
    lm_kv_dtype kv_dtype;
    uint32_t device_index;
    uint32_t context_tokens;
    uint32_t threads;
    uint32_t kv_page_tokens;
    uint64_t vram_limit_bytes;
    uint64_t host_cache_bytes;
    uint64_t pinned_cache_bytes;
    uint64_t device_cache_bytes;
    uint8_t prefetch;
    uint8_t trace;
    uint8_t deterministic;
    char model_path[LM_PATH_MAX];
} lm_config;

typedef struct lm_runtime lm_runtime;
typedef struct lm_buffer lm_buffer;
typedef struct lm_file lm_file;
typedef struct lm_model_file lm_model_file;

lm_status lm_file_open(const char *path, lm_file **out_file);
void lm_file_close(lm_file *file);
lm_status lm_file_size(const lm_file *file, uint64_t *out_bytes);
lm_status lm_file_read(lm_file *file, uint64_t offset, void *dst, size_t bytes);

typedef struct lm_file_span {
    lm_file *file;
    uint64_t offset;
    uint64_t bytes;
} lm_file_span;

lm_status lm_file_span_make(lm_file *file, uint64_t offset, uint64_t bytes, lm_file_span *out_span);
lm_status lm_file_span_read(const lm_file_span *span, uint64_t offset, void *dst, size_t bytes);


typedef enum lm_dtype {
    LM_DTYPE_F32 = 0,
    LM_DTYPE_F16,
    LM_DTYPE_BF16,
    LM_DTYPE_I8,
    LM_DTYPE_I32,
    LM_DTYPE_U8
} lm_dtype;

typedef enum lm_quant_format {
    LM_QUANT_NONE = 0,
    LM_QUANT_BLOCK_STORAGE,
    LM_QUANT_GGML_Q4_0,
    LM_QUANT_GGML_Q8_0,
    LM_QUANT_GGML_Q4_K
} lm_quant_format;

typedef struct lm_tensor {
    void *data;
    uint64_t bytes;
    uint32_t rank;
    uint32_t dims[8];
    uint32_t strides[8];
    lm_dtype dtype;
    lm_quant_format quant_format;
    uint32_t quant_elements_per_block;
    uint32_t quant_bytes_per_block;
} lm_tensor;

size_t lm_dtype_size(lm_dtype dtype);
const char *lm_quant_format_name(lm_quant_format format);
lm_status lm_tensor_validate(const lm_tensor *tensor);
lm_status lm_tensor_make_view(void *data, uint64_t bytes, lm_dtype dtype,
                              uint32_t rank, const uint32_t *dims,
                              lm_tensor *out_tensor);
lm_status lm_tensor_make_quant_view(void *data, uint64_t bytes,
                                    uint32_t rank, const uint32_t *dims,
                                    uint32_t elements_per_block,
                                    uint32_t bytes_per_block,
                                    lm_tensor *out_tensor);
lm_status lm_tensor_make_q4_0_view(void *data, uint64_t bytes,
                                   uint32_t rank, const uint32_t *dims,
                                   lm_tensor *out_tensor);
lm_status lm_tensor_make_q8_0_view(void *data, uint64_t bytes,
                                   uint32_t rank, const uint32_t *dims,
                                   lm_tensor *out_tensor);
lm_status lm_tensor_make_q4_k_view(void *data, uint64_t bytes,
                                   uint32_t rank, const uint32_t *dims,
                                   lm_tensor *out_tensor);
lm_status lm_buffer_alloc(uint64_t bytes, lm_buffer **out_buffer);
void lm_buffer_free(lm_buffer *buffer);
lm_status lm_buffer_view(lm_buffer *buffer, lm_tensor *out_tensor);


typedef enum lm_kernel_op {
    LM_KERNEL_DOT_F32 = 0,
    LM_KERNEL_DOT_I8,
    LM_KERNEL_SOFTMAX_F32,
    LM_KERNEL_DOT_Q4_K,
    LM_KERNEL_DOT_Q8_0
} lm_kernel_op;

typedef enum lm_kernel_path {
    LM_KERNEL_AUTO = 0,
    LM_KERNEL_CPU_SCALAR,
    LM_KERNEL_VULKAN_SCALAR,
    LM_KERNEL_VULKAN_DP4
} lm_kernel_path;

typedef struct lm_kernel_caps {
    uint8_t vulkan;
    uint8_t shader_int_dot;
    uint8_t subgroup;
} lm_kernel_caps;

typedef struct lm_kernel_choice {
    lm_kernel_op op;
    lm_kernel_path path;
    const char *name;
    const char *source_id;
} lm_kernel_choice;

typedef struct lm_kernel_contract {
    lm_dtype input_dtype;
    lm_dtype output_dtype;
    uint32_t minimum_alignment;
    uint8_t deterministic;
    const char *source_id;
} lm_kernel_contract;

typedef struct lm_kernel_io {
    const void *input0;
    const void *input1;
    uint32_t count;
    void *output;
} lm_kernel_io;

lm_status lm_kernel_contract_get(const lm_kernel_choice *choice, lm_kernel_contract *out_contract);
lm_status lm_kernel_select(lm_kernel_op op, lm_kernel_path requested,
                           const lm_kernel_caps *caps, lm_kernel_choice *out_choice);
const char *lm_kernel_path_name(lm_kernel_path path);
lm_status lm_vulkan_dispatch(const lm_kernel_choice *choice, const char *shader_path,
                             uint32_t device_index, const lm_kernel_io *io);

typedef struct lm_vulkan_device_info {
    char name[128];
    uint32_t api_version;
    uint32_t driver_version;
    uint32_t vendor_id;
    uint32_t device_id;
    uint8_t is_cpu;
    uint8_t shader_int_dot;
    uint8_t subgroup;
} lm_vulkan_device_info;

lm_status lm_vulkan_device_count(uint32_t *out_count);
lm_status lm_vulkan_device_info_get(uint32_t index, lm_vulkan_device_info *out_info);
lm_status lm_vulkan_dot_i8_dp4(const char *spv_path, uint32_t device_index,
                               const uint32_t *a, const uint32_t *b,
                               uint32_t packed_words, int32_t *out_result);
lm_status lm_vulkan_dot_f32(const char *spv_path, uint32_t device_index,
                            const float *a, const float *b,
                            uint32_t count, float *out_result);
lm_status lm_vulkan_dot_q4_k(const char *spv_path, uint32_t device_index,
                             const void *packed_q4_k, uint32_t blocks,
                             const float *input, float *out_result);
lm_status lm_vulkan_dot_q8_0(const char *spv_path, uint32_t device_index,
                              const void *packed_q8_0, uint32_t blocks,
                              const float *input, float *out_result);
lm_status lm_vulkan_matvec_q4_k(const char *spv_path, uint32_t device_index,
                                const void *packed_q4_k, uint32_t rows,
                                uint32_t blocks_per_row, const float *input,
                                float *out);
lm_status lm_vulkan_matvec_q8_0(const char *spv_path, uint32_t device_index,
                                const void *packed_q8_0, uint32_t rows,
                                uint32_t blocks_per_row, const float *input,
                                float *out);
typedef struct lm_kv_cache lm_kv_cache;

typedef struct lm_kv_stats {
    uint32_t page_tokens;
    uint32_t total_pages;
    uint32_t free_pages;
    uint32_t used_pages;
    uint32_t shared_pages;
    uint64_t appended_tokens;
} lm_kv_stats;


typedef enum lm_model_format {
    LM_MODEL_UNKNOWN = 0,
    LM_MODEL_GGUF,
    LM_MODEL_SAFETENSORS
} lm_model_format;

typedef struct lm_model_info {
    lm_model_format format;
    uint32_t version;
    uint64_t file_bytes;
    uint64_t header_bytes;
    uint64_t tensor_count;
    uint32_t expert_count;
    uint32_t experts_per_token;
} lm_model_info;

typedef struct lm_model_tensor_info {
    char name[65];
    uint32_t rank;
    uint64_t dims[8];
    uint32_t type;
    uint64_t relative_offset;
} lm_model_tensor_info;

typedef struct lm_model_tensor_binding {
    lm_model_tensor_info descriptor;
    lm_file_span span; /* exact native payload; data is not resident here */
    uint64_t elements;
    lm_quant_format quant_format;
    uint32_t quant_elements_per_block;
    uint32_t quant_bytes_per_block;
} lm_model_tensor_binding;

typedef struct lm_native_matvec_config {
    lm_backend_kind backend;
    uint32_t device_index;
    const char *shader_path; /* required for Vulkan; ignored by CPU */
} lm_native_matvec_config;

lm_status lm_model_open(const char *path, lm_model_file **out_model, char *error_text, size_t error_capacity);
void lm_model_close(lm_model_file *model);
lm_status lm_model_get_info(const lm_model_file *model, lm_model_info *out_info);
lm_status lm_model_tensor_info_at(const lm_model_file *model, uint64_t index, lm_model_tensor_info *out_info);
lm_status lm_model_token_count(const lm_model_file *model, uint32_t *out_count);
lm_status lm_model_token_at(const lm_model_file *model, uint32_t token_id,
                            char *out_token, size_t out_capacity, size_t *out_bytes);
lm_status lm_model_token_encode(const lm_model_file *model, const char *text,
                                size_t text_bytes, uint32_t *out_tokens,
                                size_t token_capacity, size_t *out_count);
lm_status lm_model_token_decode(const lm_model_file *model, const uint32_t *tokens,
                                size_t token_count, char *out_text,
                                size_t out_capacity, size_t *out_bytes);
lm_status lm_model_tensor_span(const lm_model_file *model, uint64_t relative_offset, uint64_t bytes, lm_file_span *out_span);
lm_status lm_model_tensor_bind_native(const lm_model_file *model, uint64_t index, lm_model_tensor_binding *out_binding);
lm_status lm_model_tensor_binding_view(const lm_model_tensor_binding *binding, void *data, uint64_t bytes, lm_tensor *out_tensor);
lm_status lm_model_tensor_binding_read(const lm_model_tensor_binding *binding, void *data, uint64_t bytes, lm_tensor *out_tensor);
lm_status lm_model_tensor_binding_matvec_q4_k_cpu(const lm_model_tensor_binding *binding,
                                                  void *packed_scratch, uint64_t scratch_bytes,
                                                  uint32_t rows, uint32_t columns,
                                                  const float *input, float *out);
lm_status lm_model_tensor_binding_matvec_q4_k_vulkan(const lm_model_tensor_binding *binding,
                                                     void *packed_scratch, uint64_t scratch_bytes,
                                                     uint32_t rows, uint32_t columns,
                                                     const char *spv_path, uint32_t device_index,
                                                     const float *input, float *out);
lm_status lm_model_tensor_binding_matvec_q8_0_cpu(const lm_model_tensor_binding *binding,
                                                  void *packed_scratch, uint64_t scratch_bytes,
                                                  uint32_t rows, uint32_t columns,
                                                  const float *input, float *out);
lm_status lm_model_tensor_binding_matvec_q8_0_vulkan(const lm_model_tensor_binding *binding,
                                                     void *packed_scratch, uint64_t scratch_bytes,
                                                     uint32_t rows, uint32_t columns,
                                                     const char *spv_path, uint32_t device_index,
                                                     const float *input, float *out);
lm_status lm_model_tensor_binding_dot_q8_0_cpu(const lm_model_tensor_binding *binding,
                                               void *packed_scratch, uint64_t scratch_bytes,
                                               const float *input, uint64_t elements, float *out);
lm_status lm_model_tensor_binding_dot_q8_0_vulkan(const lm_model_tensor_binding *binding,
                                                  void *packed_scratch, uint64_t scratch_bytes,
                                                  const char *spv_path, uint32_t device_index,
                                                  const float *input, uint64_t elements, float *out);
lm_status lm_model_tensor_matvec_native(const lm_model_file *model, uint64_t tensor_index,
                                        const lm_native_matvec_config *config,
                                        void *packed_scratch, uint64_t scratch_bytes,
                                        uint32_t rows, uint32_t columns,
                                        const float *input, float *out);


void lm_config_init(lm_config *config);
lm_status lm_config_parse_argv(lm_config *config, int argc, char **argv,
                               const char **bad_argument);
const char *lm_status_name(lm_status status);
const char *lm_backend_name(lm_backend_kind backend);
const char *lm_load_mode_name(lm_load_mode mode);
const char *lm_weight_policy_name(lm_weight_policy policy);
const char *lm_kv_dtype_name(lm_kv_dtype dtype);
const char *lm_model_format_name(lm_model_format format);
lm_status lm_model_inspect(const char *path, lm_model_info *out_info,
                           char *error_text, size_t error_capacity);
lm_status lm_cpu_dot_f32(const float *a, const float *b, size_t count, float *out);
lm_status lm_cpu_softmax_f32(const float *input, float *output, size_t count);

typedef enum lm_sampling_mode {
    LM_SAMPLING_GREEDY = 0,
    LM_SAMPLING_TOP_K
} lm_sampling_mode;

typedef struct lm_sampling_config {
    lm_sampling_mode mode;
    uint32_t top_k;
    float temperature;
    uint64_t seed;
} lm_sampling_config;

lm_status lm_sample_logits(const float *logits, uint32_t vocab_size,
                           const lm_sampling_config *config, uint32_t *out_token,
                           float *out_probability);
lm_status lm_cpu_dot_q4_0(const lm_tensor *weights, const float *input,
                          uint64_t elements, float *out);
lm_status lm_cpu_dot_q8_0(const lm_tensor *weights, const float *input,
                          uint64_t elements, float *out);
lm_status lm_cpu_dot_q4_k(const lm_tensor *weights, const float *input,
                          uint64_t elements, float *out);
lm_status lm_cpu_matvec_q4_k(const lm_tensor *weights, const float *input,
                             uint32_t rows, uint32_t columns, float *out);
lm_status lm_cpu_matvec_q8_0(const lm_tensor *weights, const float *input,
                             uint32_t rows, uint32_t columns, float *out);


typedef struct lm_cpu_decoder lm_cpu_decoder;

typedef struct lm_cpu_decoder_config {
    uint32_t vocab_size;
    uint32_t hidden_size;
    uint32_t max_context;
    float rms_epsilon;
    float rope_theta;
    uint8_t use_rope;
    const float *embedding; /* vocab_size x hidden_size */
    const float *rms_gamma_1; /* hidden_size */
    const float *wq; /* hidden_size x hidden_size */
    const float *wk; /* hidden_size x hidden_size */
    const float *wv; /* hidden_size x hidden_size */
    const float *wo; /* hidden_size x hidden_size */
    const float *rms_gamma_2; /* hidden_size */
    const float *w1; /* hidden_size x hidden_size */
    const float *w2; /* hidden_size x hidden_size */
    const float *wout; /* hidden_size x vocab_size */
} lm_cpu_decoder_config;

lm_status lm_cpu_decoder_create(const lm_cpu_decoder_config *config, lm_cpu_decoder **out_decoder);
void lm_cpu_decoder_destroy(lm_cpu_decoder *decoder);
lm_status lm_cpu_decoder_reset(lm_cpu_decoder *decoder);
lm_status lm_cpu_decoder_step(lm_cpu_decoder *decoder, uint32_t token_id, float *out_logits, size_t logits_count);
uint32_t lm_cpu_decoder_position(const lm_cpu_decoder *decoder);

typedef enum lm_moe_route_policy {
    LM_MOE_SOFTMAX_ALL_THEN_TOPK = 0,
    LM_MOE_SOFTMAX_SELECTED_ONLY
} lm_moe_route_policy;

typedef struct lm_moe_route {
    uint32_t expert_count;
    uint32_t experts_per_token;
    uint32_t selected[16];
    float weights[16];
} lm_moe_route;

typedef enum lm_moe_tensor_role {
    LM_MOE_TENSOR_GATE_UP_EXPERT = 0,
    LM_MOE_TENSOR_DOWN_EXPERT
} lm_moe_tensor_role;

typedef struct lm_moe_tensor_mapping {
    lm_moe_tensor_role role;
    uint32_t layer_index;
    uint32_t expert_axis;
    uint32_t expert_count;
    uint32_t rank;
    uint64_t dims[8];
} lm_moe_tensor_mapping;

typedef enum lm_decoder_tensor_role {
    LM_DECODER_TENSOR_TOKEN_EMBEDDING = 0,
    LM_DECODER_TENSOR_OUTPUT,
    LM_DECODER_TENSOR_OUTPUT_NORM,
    LM_DECODER_TENSOR_ATTN_NORM,
    LM_DECODER_TENSOR_ATTN_Q,
    LM_DECODER_TENSOR_ATTN_K,
    LM_DECODER_TENSOR_ATTN_V,
    LM_DECODER_TENSOR_ATTN_OUTPUT,
    LM_DECODER_TENSOR_FFN_NORM,
    LM_DECODER_TENSOR_FFN_GATE,
    LM_DECODER_TENSOR_FFN_DOWN,
    LM_DECODER_TENSOR_FFN_UP
} lm_decoder_tensor_role;

typedef struct lm_decoder_tensor_mapping {
    lm_decoder_tensor_role role;
    uint32_t layer_index;
    uint32_t rank;
    uint64_t dims[8];
    uint32_t type;
} lm_decoder_tensor_mapping;

#define LM_DECODER_PLAN_MAX_LAYERS 256u
typedef struct lm_decoder_graph_plan {
    uint32_t layer_count;
    uint32_t global_role_mask;
    uint32_t layer_role_mask[LM_DECODER_PLAN_MAX_LAYERS];
} lm_decoder_graph_plan;

typedef struct lm_decoder_layer_binding {
    uint64_t attn_norm;
    uint64_t attn_q;
    uint64_t attn_k;
    uint64_t attn_v;
    uint64_t attn_output;
    uint64_t ffn_norm;
    uint64_t ffn_gate;
    uint64_t ffn_down;
    uint64_t ffn_up;
} lm_decoder_layer_binding;

typedef struct lm_decoder_graph_binding {
    uint64_t token_embedding;
    uint64_t output;
    uint64_t output_norm;
    uint32_t layer_count;
    lm_decoder_layer_binding layers[LM_DECODER_PLAN_MAX_LAYERS];
} lm_decoder_graph_binding;

lm_status lm_moe_map_mixtral_tensor(const lm_model_tensor_info *descriptor,
                                    uint32_t expected_experts,
                                    lm_moe_tensor_mapping *out_mapping);
lm_status lm_decoder_map_llama_tensor(const lm_model_tensor_info *descriptor,
                                      lm_decoder_tensor_mapping *out_mapping);
lm_status lm_decoder_graph_plan_build(const lm_model_tensor_info *descriptors,
                                      uint64_t descriptor_count,
                                      lm_decoder_graph_plan *out_plan);
lm_status lm_model_build_llama_graph(const lm_model_file *model,
                                     lm_decoder_graph_binding *out_binding);

/* Narrow profile: one layer's native embedding/MLP/output path only; attention is not scheduled. */
typedef struct lm_native_mlp_config {
    lm_native_matvec_config matvec;
    uint32_t layer_index;
    uint32_t vocab_size;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    lm_quant_format matrix_format;
    float rms_epsilon;
} lm_native_mlp_config;

lm_status lm_model_execute_native_mlp_logits(const lm_model_file *model,
                                             const lm_decoder_graph_binding *graph,
                                             const lm_native_mlp_config *config,
                                             uint32_t token_id,
                                             void *packed_scratch,
                                             uint64_t packed_scratch_bytes,
                                             float *out_logits,
                                             size_t logits_count);

/* Narrow attention: one head, equal query/KV width, caller-appended token slot, F32 KV payload. */
typedef struct lm_native_attention_config {
    lm_native_matvec_config matvec;
    uint32_t layer_index;
    uint32_t hidden_size;
    uint32_t token_offset;
    uint8_t use_rope;
    float rope_theta;
    float rms_epsilon;
    lm_quant_format matrix_format;
} lm_native_attention_config;

lm_status lm_model_execute_native_attention(const lm_model_file *model,
                                            const lm_decoder_graph_binding *graph,
                                            const lm_native_attention_config *config,
                                            const float *input,
                                            lm_kv_cache *kv_cache,
                                            uint32_t page_id,
                                            void *packed_scratch,
                                            uint64_t packed_scratch_bytes,
                                            float *out_attention);

typedef struct lm_native_step_config {
    lm_native_matvec_config matvec;
    uint32_t layer_index;
    uint32_t vocab_size;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t token_offset;
    uint8_t use_rope;
    float rope_theta;
    float rms_epsilon;
    lm_quant_format matrix_format;
} lm_native_step_config;

lm_status lm_model_execute_native_step(const lm_model_file *model,
                                       const lm_decoder_graph_binding *graph,
                                       const lm_native_step_config *config,
                                       uint32_t token_id,
                                       lm_kv_cache *kv_cache,
                                       uint32_t page_id,
                                       void *packed_scratch,
                                       uint64_t packed_scratch_bytes,
                                       float *out_logits,
                                       size_t logits_count);

/* Bounded local generation for the narrow profile; prompt IDs must come from the model tokenizer. */
typedef struct lm_native_generation_config {
    lm_native_step_config step;
    lm_sampling_config sampling;
    uint32_t max_new_tokens;
    uint32_t stop_token;
    uint8_t has_stop_token;
} lm_native_generation_config;

lm_status lm_model_generate_native(const lm_model_file *model,
                                   const lm_decoder_graph_binding *graph,
                                   const lm_native_generation_config *config,
                                   const uint32_t *prompt_tokens,
                                   size_t prompt_count,
                                   void *packed_scratch,
                                   uint64_t packed_scratch_bytes,
                                   uint32_t *out_tokens,
                                   size_t token_capacity,
                                   size_t *out_count);
lm_status lm_model_generate_native_text(const lm_model_file *model,
                                        const lm_decoder_graph_binding *graph,
                                        const lm_native_generation_config *config,
                                        const char *prompt, size_t prompt_bytes,
                                        void *packed_scratch,
                                        uint64_t packed_scratch_bytes,
                                        char *out_text, size_t out_capacity,
                                        size_t *out_bytes);

lm_status lm_cpu_moe_route(const float *router_logits, uint32_t expert_count,
                           uint32_t experts_per_token, lm_moe_route_policy policy,
                           lm_moe_route *out_route);
lm_status lm_cpu_moe_combine(const lm_moe_route *route, const float *selected_outputs,
                              uint32_t hidden_size, float *out_hidden);
lm_status lm_cpu_moe_selected_expert_mlp(const lm_moe_route *route,
                                         const lm_tensor *gate_up_weights,
                                         const lm_tensor *down_weights,
                                         uint32_t hidden_size, uint32_t intermediate_size,
                                         const float *input, float *selected_outputs);
lm_status lm_cpu_moe_selected_expert_mlp_q4_k(const lm_moe_route *route,
                                              const lm_tensor *gate_up_weights,
                                              const lm_tensor *down_weights,
                                              uint32_t hidden_size, uint32_t intermediate_size,
                                              const float *input, float *selected_outputs);

lm_status lm_kv_cache_create(uint32_t page_count, uint32_t page_tokens,
                             lm_kv_cache **out_cache);
/* Payload bytes are caller-defined and remain in source encoding; no conversion is performed.
 * The payload allocation is separate from control-plane metadata and is per page. */
lm_status lm_kv_cache_create_with_payload(uint32_t page_count, uint32_t page_tokens,
                                          uint32_t key_bytes_per_token,
                                          uint32_t value_bytes_per_token,
                                          lm_kv_cache **out_cache);
void lm_kv_cache_destroy(lm_kv_cache *cache);
lm_status lm_kv_cache_append(lm_kv_cache *cache, uint32_t *page_id,
                             uint32_t token_count);
lm_status lm_kv_cache_page_token_count(const lm_kv_cache *cache, uint32_t page_id,
                                       uint32_t *out_tokens);
lm_status lm_kv_cache_fork(lm_kv_cache *cache, uint32_t source_page,
                           uint32_t *out_page);
lm_status lm_kv_cache_rollback(lm_kv_cache *cache, uint32_t page_id,
                               uint32_t token_count);
lm_status lm_kv_cache_release(lm_kv_cache *cache, uint32_t page_id);
lm_status lm_kv_cache_get_stats(const lm_kv_cache *cache, lm_kv_stats *out_stats);
lm_status lm_kv_cache_get_payload_layout(const lm_kv_cache *cache,
                                         uint32_t *key_bytes_per_token,
                                         uint32_t *value_bytes_per_token);
/* Read/write cover already-appended tokens only; writes detach shared pages first. */
lm_status lm_kv_cache_write_payload(lm_kv_cache *cache, uint32_t page_id,
                                     uint32_t token_offset, uint32_t token_count,
                                     const void *keys, const void *values);
lm_status lm_kv_cache_read_payload(const lm_kv_cache *cache, uint32_t page_id,
                                    uint32_t token_offset, uint32_t token_count,
                                    void *keys, void *values);

lm_status lm_runtime_create(const lm_config *config, lm_runtime **out_runtime);
void lm_runtime_destroy(lm_runtime *runtime);
void lm_runtime_set_probe_sink(lm_runtime *runtime, lm_probe_sink sink, void *user);
lm_status lm_runtime_emit_probe(lm_runtime *runtime, const lm_probe *probe);
lm_status lm_runtime_dump_config(const lm_runtime *runtime, FILE *out);

#ifdef __cplusplus
}
#endif

#endif
