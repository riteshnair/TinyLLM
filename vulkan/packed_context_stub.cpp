#include "lm/lm.h"

struct lm_vulkan_packed_context {};

lm_status lm_vulkan_packed_context_create(const char *spv_path, uint32_t device_index,
                                          uint32_t block_bytes, uint32_t block_values,
                                          lm_vulkan_packed_context **out_context) {
    (void)spv_path;
    (void)device_index;
    (void)block_bytes;
    (void)block_values;
    if (!out_context) return LM_ERR_ARGUMENT;
    *out_context = nullptr;
    return LM_ERR_UNSUPPORTED;
}

void lm_vulkan_packed_context_destroy(lm_vulkan_packed_context *context) { (void)context; }

lm_status lm_vulkan_packed_context_matvec(lm_vulkan_packed_context *context,
                                          const void *packed, uint32_t rows,
                                          uint32_t blocks_per_row, const float *input,
                                          float *out) {
    (void)context;
    (void)packed;
    (void)rows;
    (void)blocks_per_row;
    (void)input;
    (void)out;
    return LM_ERR_UNSUPPORTED;
}
