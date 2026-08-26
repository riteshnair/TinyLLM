#include "lm/lm.h"

lm_status lm_vulkan_matvec_f32(const char *spv_path, uint32_t device_index,
                               const float *matrix, uint32_t rows,
                               uint32_t columns, const float *input,
                               float *out) {
    (void)spv_path;
    (void)device_index;
    (void)matrix;
    (void)rows;
    (void)columns;
    (void)input;
    (void)out;
    return LM_ERR_UNSUPPORTED;
}
