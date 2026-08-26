#include "lm/lm.h"

struct lm_vulkan_f32_context {};

lm_status lm_vulkan_f32_context_create(const char *spv_path, uint32_t device_index,
                                       lm_vulkan_f32_context **out_context) {
    (void)spv_path;
    (void)device_index;
    if (!out_context) return LM_ERR_ARGUMENT;
    *out_context = nullptr;
    return LM_ERR_UNSUPPORTED;
}

void lm_vulkan_f32_context_destroy(lm_vulkan_f32_context *context) { (void)context; }

lm_status lm_vulkan_f32_context_matvec(lm_vulkan_f32_context *context,
                                       const float *matrix, uint32_t rows,
                                       uint32_t columns, const float *input,
                                       float *out) {
    (void)context;
    (void)matrix;
    (void)rows;
    (void)columns;
    (void)input;
    (void)out;
    return LM_ERR_UNSUPPORTED;
}
