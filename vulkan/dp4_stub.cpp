#include "lm/lm.h"

lm_status lm_vulkan_dot_i8_dp4(const char *spv_path, uint32_t device_index,
                               const uint32_t *a, const uint32_t *b,
                               uint32_t packed_words, int32_t *out_result) {
    (void)spv_path;
    (void)device_index;
    (void)a;
    (void)b;
    (void)packed_words;
    (void)out_result;
    return LM_ERR_UNSUPPORTED;
}
