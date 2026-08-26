#include "lm/lm.h"

lm_status lm_vulkan_dispatch(const lm_kernel_choice *, const char *, uint32_t, const lm_kernel_io *) {
    return LM_ERR_UNSUPPORTED;
}
