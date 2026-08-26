#include "lm/lm.h"

#include <cstdlib>
#include <cstring>
#include <limits>

struct lm_buffer {
    void *data;
    uint64_t bytes;
};

const char *lm_quant_format_name(lm_quant_format format) {
    switch (format) {
        case LM_QUANT_NONE: return "none";
        case LM_QUANT_BLOCK_STORAGE: return "block-storage";
        case LM_QUANT_GGML_Q4_0: return "ggml-q4_0";
        case LM_QUANT_GGML_Q8_0: return "ggml-q8_0";
        case LM_QUANT_GGML_Q4_K: return "ggml-q4_k";
        default: return "unknown";
    }
}

size_t lm_dtype_size(lm_dtype dtype) {
    switch (dtype) {
        case LM_DTYPE_F32: return 4u;
        case LM_DTYPE_F16: return 2u;
        case LM_DTYPE_BF16: return 2u;
        case LM_DTYPE_I8: return 1u;
        case LM_DTYPE_I32: return 4u;
        case LM_DTYPE_U8: return 1u;
        default: return 0u;
    }
}

lm_status lm_tensor_validate(const lm_tensor *tensor) {
    if (!tensor || tensor->rank == 0u || tensor->rank > 8u || !tensor->data) return LM_ERR_ARGUMENT;
    const size_t element_size = lm_dtype_size(tensor->dtype);
    if (element_size == 0u) return LM_ERR_ARGUMENT;
    uint64_t elements = 1u;
    for (uint32_t i = 0u; i < tensor->rank; ++i) {
        if (tensor->dims[i] == 0u || elements > std::numeric_limits<uint64_t>::max() / tensor->dims[i])
            return LM_ERR_RANGE;
        elements *= tensor->dims[i];
    }
    if (elements > std::numeric_limits<uint64_t>::max() / element_size) return LM_ERR_RANGE;
    if (tensor->quant_format == LM_QUANT_NONE) {
        if (tensor->quant_elements_per_block != 0u || tensor->quant_bytes_per_block != 0u) return LM_ERR_ARGUMENT;
        if (tensor->bytes < elements * element_size) return LM_ERR_CAPACITY;
        return LM_OK;
    }
    if ((tensor->quant_format != LM_QUANT_BLOCK_STORAGE && tensor->quant_format != LM_QUANT_GGML_Q4_0 && tensor->quant_format != LM_QUANT_GGML_Q8_0 && tensor->quant_format != LM_QUANT_GGML_Q4_K) ||
        tensor->quant_elements_per_block == 0u || tensor->quant_bytes_per_block == 0u ||
        elements % tensor->quant_elements_per_block != 0u)
        return LM_ERR_ARGUMENT;
    if (tensor->quant_format == LM_QUANT_GGML_Q4_0 &&
        (tensor->quant_elements_per_block != 32u || tensor->quant_bytes_per_block != 18u))
        return LM_ERR_ARGUMENT;
    if (tensor->quant_format == LM_QUANT_GGML_Q8_0 &&
        (tensor->quant_elements_per_block != 32u || tensor->quant_bytes_per_block != 34u))
        return LM_ERR_ARGUMENT;
    if (tensor->quant_format == LM_QUANT_GGML_Q4_K &&
        (tensor->quant_elements_per_block != 256u || tensor->quant_bytes_per_block != 144u))
        return LM_ERR_ARGUMENT;
    const uint64_t blocks = elements / tensor->quant_elements_per_block;
    if (blocks > std::numeric_limits<uint64_t>::max() / tensor->quant_bytes_per_block) return LM_ERR_RANGE;
    return tensor->bytes == blocks * tensor->quant_bytes_per_block ? LM_OK : LM_ERR_CAPACITY;
}

lm_status lm_tensor_make_quant_view(void *data, uint64_t bytes,
                                    uint32_t rank, const uint32_t *dims,
                                    uint32_t elements_per_block,
                                    uint32_t bytes_per_block,
                                    lm_tensor *out_tensor) {
    if (!data || !dims || !out_tensor || rank == 0u || rank > 8u || elements_per_block == 0u || bytes_per_block == 0u) return LM_ERR_ARGUMENT;
    std::memset(out_tensor, 0, sizeof(*out_tensor));
    out_tensor->data = data;
    out_tensor->bytes = bytes;
    out_tensor->rank = rank;
    out_tensor->dtype = LM_DTYPE_U8;
    uint32_t stride = 1u;
    for (uint32_t i = rank; i-- > 0u;) {
        if (dims[i] == 0u) return LM_ERR_RANGE;
        out_tensor->dims[i] = dims[i];
        out_tensor->strides[i] = stride;
        if (i > 0u && stride > std::numeric_limits<uint32_t>::max() / dims[i]) return LM_ERR_RANGE;
        stride *= dims[i];
    }
    out_tensor->quant_format = LM_QUANT_BLOCK_STORAGE;
    out_tensor->quant_elements_per_block = elements_per_block;
    out_tensor->quant_bytes_per_block = bytes_per_block;
    return lm_tensor_validate(out_tensor);
}

lm_status lm_tensor_make_q4_0_view(void *data, uint64_t bytes,
                                   uint32_t rank, const uint32_t *dims,
                                   lm_tensor *out_tensor) {
    const lm_status status = lm_tensor_make_quant_view(data, bytes, rank, dims, 32u, 18u, out_tensor);
    if (status != LM_OK) return status;
    out_tensor->quant_format = LM_QUANT_GGML_Q4_0;
    return lm_tensor_validate(out_tensor);
}

lm_status lm_tensor_make_q8_0_view(void *data, uint64_t bytes,
                                   uint32_t rank, const uint32_t *dims,
                                   lm_tensor *out_tensor) {
    const lm_status status = lm_tensor_make_quant_view(data, bytes, rank, dims, 32u, 34u, out_tensor);
    if (status != LM_OK) return status;
    out_tensor->quant_format = LM_QUANT_GGML_Q8_0;
    return lm_tensor_validate(out_tensor);
}

lm_status lm_tensor_make_q4_k_view(void *data, uint64_t bytes,
                                   uint32_t rank, const uint32_t *dims,
                                   lm_tensor *out_tensor) {
    const lm_status status = lm_tensor_make_quant_view(data, bytes, rank, dims, 256u, 144u, out_tensor);
    if (status != LM_OK) return status;
    out_tensor->quant_format = LM_QUANT_GGML_Q4_K;
    return lm_tensor_validate(out_tensor);
}

lm_status lm_tensor_make_view(void *data, uint64_t bytes, lm_dtype dtype,
                              uint32_t rank, const uint32_t *dims,
                              lm_tensor *out_tensor) {
    if (!data || !dims || !out_tensor || rank == 0u || rank > 8u) return LM_ERR_ARGUMENT;
    std::memset(out_tensor, 0, sizeof(*out_tensor));
    out_tensor->data = data;
    out_tensor->bytes = bytes;
    out_tensor->rank = rank;
    out_tensor->dtype = dtype;
    uint32_t stride = 1u;
    for (uint32_t i = rank; i-- > 0u;) {
        if (dims[i] == 0u) return LM_ERR_RANGE;
        out_tensor->dims[i] = dims[i];
        out_tensor->strides[i] = stride;
        if (i > 0u && stride > std::numeric_limits<uint32_t>::max() / dims[i]) return LM_ERR_RANGE;
        stride *= dims[i];
    }
    return lm_tensor_validate(out_tensor);
}

lm_status lm_buffer_alloc(uint64_t bytes, lm_buffer **out_buffer) {
    if (!out_buffer || bytes == 0u || bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
        return LM_ERR_ARGUMENT;
    *out_buffer = nullptr;
    lm_buffer *buffer = static_cast<lm_buffer *>(std::calloc(1u, sizeof(*buffer)));
    if (!buffer) return LM_ERR_CAPACITY;
    buffer->data = std::calloc(1u, static_cast<size_t>(bytes));
    if (!buffer->data) { std::free(buffer); return LM_ERR_CAPACITY; }
    buffer->bytes = bytes;
    *out_buffer = buffer;
    return LM_OK;
}

void lm_buffer_free(lm_buffer *buffer) {
    if (!buffer) return;
    std::free(buffer->data);
    std::free(buffer);
}

lm_status lm_buffer_view(lm_buffer *buffer, lm_tensor *out_tensor) {
    if (!buffer || !out_tensor) return LM_ERR_ARGUMENT;
    const uint32_t dims[1] = {static_cast<uint32_t>(buffer->bytes)};
    if (buffer->bytes > UINT32_MAX) return LM_ERR_RANGE;
    return lm_tensor_make_view(buffer->data, buffer->bytes, LM_DTYPE_U8, 1u, dims, out_tensor);
}
