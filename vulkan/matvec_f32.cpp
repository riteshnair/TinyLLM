#include "lm/lm.h"

#include <cmath>
#include <cstdint>
#include <limits>

lm_status lm_vulkan_matvec_f32(const char *spv_path, uint32_t device_index,
                               const float *matrix, uint32_t rows,
                               uint32_t columns, const float *input,
                               float *out) {
    if (!spv_path || !matrix || !input || !out || rows == 0u || columns == 0u) return LM_ERR_ARGUMENT;
    const uint64_t elements = static_cast<uint64_t>(rows) * columns;
    if (elements > std::numeric_limits<uint64_t>::max() / sizeof(float) ||
        elements * sizeof(float) > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) return LM_ERR_CAPACITY;
    for (uint32_t row = 0u; row < rows; ++row) {
        const lm_status status = lm_vulkan_dot_f32(spv_path, device_index, matrix + static_cast<size_t>(row) * columns,
                                                   input, columns, out + row);
        if (status != LM_OK) return status;
    }
    for (uint32_t row = 0u; row < rows; ++row) if (!std::isfinite(out[row])) return LM_ERR_RANGE;
    return LM_OK;
}
