#include "lm/lm.h"

#include <cstdint>

lm_status lm_vulkan_dispatch(const lm_kernel_choice *choice, const char *shader_path,
                             uint32_t device_index, const lm_kernel_io *io) {
    if (!choice || !shader_path || !io || !io->input0 || !io->input1 || !io->output || io->count == 0u)
        return LM_ERR_ARGUMENT;
    if (choice->path == LM_KERNEL_VULKAN_SCALAR && choice->op == LM_KERNEL_DOT_F32)
        return lm_vulkan_dot_f32(shader_path, device_index,
                                 static_cast<const float *>(io->input0),
                                 static_cast<const float *>(io->input1), io->count,
                                 static_cast<float *>(io->output));
    if (choice->path == LM_KERNEL_VULKAN_DP4 && choice->op == LM_KERNEL_DOT_I8)
        return lm_vulkan_dot_i8_dp4(shader_path, device_index,
                                    static_cast<const uint32_t *>(io->input0),
                                    static_cast<const uint32_t *>(io->input1), io->count,
                                    static_cast<int32_t *>(io->output));
    if (choice->path == LM_KERNEL_VULKAN_SCALAR && choice->op == LM_KERNEL_DOT_Q8_0)
        return lm_vulkan_dot_q8_0(shader_path, device_index, io->input0, io->count,
                                  static_cast<const float *>(io->input1),
                                  static_cast<float *>(io->output));
    if (choice->path == LM_KERNEL_VULKAN_SCALAR && choice->op == LM_KERNEL_DOT_Q4_K)
        return lm_vulkan_dot_q4_k(shader_path, device_index, io->input0, io->count,
                                  static_cast<const float *>(io->input1),
                                  static_cast<float *>(io->output));
    return LM_ERR_UNSUPPORTED;
}
