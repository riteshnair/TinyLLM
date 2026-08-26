#include "lm/lm.h"

#include <cstring>

lm_status lm_vulkan_device_count(uint32_t *out_count) {
    if (!out_count) return LM_ERR_ARGUMENT;
    *out_count = 0u;
    return LM_ERR_UNSUPPORTED;
}

lm_status lm_vulkan_device_info_get(uint32_t index, lm_vulkan_device_info *out_info) {
    (void)index;
    if (!out_info) return LM_ERR_ARGUMENT;
    std::memset(out_info, 0, sizeof(*out_info));
    return LM_ERR_UNSUPPORTED;
}
