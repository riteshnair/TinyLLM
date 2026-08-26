#include "lm/lm.h"

lm_status lm_vulkan_matvec_q4_k(const char *spv_path, uint32_t device_index,
                                const void *packed_q4_k, uint32_t rows,
                                uint32_t blocks_per_row, const float *input,
                                float *out) {
    (void)spv_path;
    (void)device_index;
    (void)packed_q4_k;
    (void)rows;
    (void)blocks_per_row;
    (void)input;
    (void)out;
    return LM_ERR_UNSUPPORTED;
}
