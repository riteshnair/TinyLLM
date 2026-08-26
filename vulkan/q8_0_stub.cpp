#include "lm/lm.h"

lm_status lm_vulkan_dot_q8_0(const char *spv_path, uint32_t device_index,
                              const void *packed_q8_0, uint32_t blocks,
                              const float *input, float *out_result) {
    (void)spv_path;
    (void)device_index;
    (void)packed_q8_0;
    (void)blocks;
    (void)input;
    (void)out_result;
    return LM_ERR_UNSUPPORTED;
}
