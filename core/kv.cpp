#include "lm/lm.h"

#include <cstdlib>
#include <cstring>
#include <limits>
#include <cmath>
#include <cstdint>
#include <algorithm>

static uint16_t float_to_f16(float value) {
    uint32_t bits = 0u; std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16u) & 0x8000u;
    int exponent = static_cast<int>((bits >> 23u) & 0xffu) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffffu;
    if (exponent <= 0) return static_cast<uint16_t>(sign);
    if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    mantissa += 0x1000u;
    if ((mantissa & 0x800000u) != 0u) { mantissa = 0u; ++exponent; }
    return exponent >= 31 ? static_cast<uint16_t>(sign | 0x7c00u) :
           static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10u) | (mantissa >> 13u));
}

static float f16_to_float(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16u;
    const uint32_t exponent = (value >> 10u) & 31u;
    const uint32_t mantissa = value & 0x3ffu;
    uint32_t bits = sign;
    if (exponent == 0u) bits = mantissa == 0u ? sign : sign | ((static_cast<uint32_t>(127 - 15 + 1)) << 23u) | (mantissa << 13u);
    else if (exponent == 31u) bits = sign | 0x7f800000u | (mantissa << 13u);
    else bits = sign | ((exponent + 127u - 15u) << 23u) | (mantissa << 13u);
    float result = 0.0f; std::memcpy(&result, &bits, sizeof(result)); return result;
}

static uint32_t kv_codec_bytes(lm_kv_dtype dtype, uint32_t elements) {
    if (elements == 0u) return 0u;
    const uint64_t blocks = (static_cast<uint64_t>(elements) + 31u) / 32u;
    uint64_t bytes = 0u;
    if (dtype == LM_KV_F16 || dtype == LM_KV_BF16) bytes = static_cast<uint64_t>(elements) * 2u;
    else if (dtype == LM_KV_Q8) bytes = blocks * 36u;
    else if (dtype == LM_KV_Q6) bytes = blocks * 28u;
    else if (dtype == LM_KV_Q4) bytes = blocks * 20u;
    return bytes > UINT32_MAX ? 0u : static_cast<uint32_t>(bytes);
}

static lm_status kv_codec_write(const float *input, uint32_t elements, lm_kv_dtype dtype, void *output, uint32_t bytes) {
    if (!input || !output || kv_codec_bytes(dtype, elements) != bytes) return LM_ERR_ARGUMENT;
    for (uint32_t i = 0u; i < elements; ++i) if (!std::isfinite(input[i])) return LM_ERR_RANGE;
    if (dtype == LM_KV_F16) {
        uint16_t *dst = static_cast<uint16_t *>(output);
        for (uint32_t i = 0u; i < elements; ++i) dst[i] = float_to_f16(input[i]);
        return LM_OK;
    }
    if (dtype == LM_KV_BF16) {
        uint16_t *dst = static_cast<uint16_t *>(output);
        for (uint32_t i = 0u; i < elements; ++i) { uint32_t bits = 0u; std::memcpy(&bits, input + i, sizeof(bits)); dst[i] = static_cast<uint16_t>((bits + 0x8000u) >> 16u); }
        return LM_OK;
    }
    const uint32_t packed_bytes = dtype == LM_KV_Q8 ? 32u : dtype == LM_KV_Q6 ? 24u : 16u;
    const int max_value = dtype == LM_KV_Q8 ? 127 : dtype == LM_KV_Q6 ? 31 : 7;
    const int quant_offset = dtype == LM_KV_Q6 ? 32 : dtype == LM_KV_Q4 ? 8 : 0;
    const uint32_t blocks = (elements + 31u) / 32u;
    unsigned char *dst = static_cast<unsigned char *>(output);
    for (uint32_t block = 0u; block < blocks; ++block) {
        const uint32_t begin = block * 32u;
        const uint32_t count = elements - begin < 32u ? elements - begin : 32u;
        float maximum = 0.0f;
        for (uint32_t i = 0u; i < count; ++i) maximum = std::max(maximum, std::fabs(input[begin + i]));
        const float scale = maximum > 0.0f ? maximum / static_cast<float>(max_value) : 1.0f;
        std::memcpy(dst + block * (4u + packed_bytes), &scale, sizeof(scale));
        std::memset(dst + block * (4u + packed_bytes) + 4u, 0, packed_bytes);
        for (uint32_t i = 0u; i < count; ++i) {
            int quantized = static_cast<int>(std::lrint(input[begin + i] / scale));
            quantized = std::max(-max_value, std::min(max_value, quantized));
            const uint32_t stored = static_cast<uint32_t>(quantized + quant_offset);
            if (dtype == LM_KV_Q8) static_cast<int8_t *>(static_cast<void *>(dst + block * 36u + 4u))[i] = static_cast<int8_t>(quantized);
            else for (uint32_t bit = 0u; bit < (dtype == LM_KV_Q6 ? 6u : 4u); ++bit) if ((stored & (1u << bit)) != 0u) {
                const uint32_t position = i * (dtype == LM_KV_Q6 ? 6u : 4u) + bit;
                dst[block * (4u + packed_bytes) + 4u + position / 8u] |= static_cast<unsigned char>(1u << (position % 8u));
            }
        }
    }
    return LM_OK;
}

static lm_status kv_codec_read(const void *input, uint32_t elements, lm_kv_dtype dtype, uint32_t bytes, float *output) {
    if (!input || !output || kv_codec_bytes(dtype, elements) != bytes) return LM_ERR_ARGUMENT;
    if (dtype == LM_KV_F16) { const uint16_t *src = static_cast<const uint16_t *>(input); for (uint32_t i = 0u; i < elements; ++i) output[i] = f16_to_float(src[i]); return LM_OK; }
    if (dtype == LM_KV_BF16) { const uint16_t *src = static_cast<const uint16_t *>(input); for (uint32_t i = 0u; i < elements; ++i) { uint32_t bits = static_cast<uint32_t>(src[i]) << 16u; std::memcpy(output + i, &bits, sizeof(bits)); } return LM_OK; }
    const uint32_t packed_bytes = dtype == LM_KV_Q8 ? 32u : dtype == LM_KV_Q6 ? 24u : 16u;
    const int quant_offset = dtype == LM_KV_Q6 ? 32 : dtype == LM_KV_Q4 ? 8 : 0;
    const uint32_t bits_per_value = dtype == LM_KV_Q6 ? 6u : 4u;
    const unsigned char *src = static_cast<const unsigned char *>(input);
    for (uint32_t block = 0u; block < (elements + 31u) / 32u; ++block) {
        float scale = 1.0f; std::memcpy(&scale, src + block * (4u + packed_bytes), sizeof(scale));
        const uint32_t count = elements - block * 32u < 32u ? elements - block * 32u : 32u;
        for (uint32_t i = 0u; i < count; ++i) {
            int quantized = 0;
            if (dtype == LM_KV_Q8) quantized = static_cast<int>(static_cast<const int8_t *>(static_cast<const void *>(src + block * 36u + 4u))[i]);
            else { uint32_t stored = 0u; for (uint32_t bit = 0u; bit < bits_per_value; ++bit) { const uint32_t position = i * bits_per_value + bit; if ((src[block * (4u + packed_bytes) + 4u + position / 8u] & (1u << (position % 8u))) != 0u) stored |= 1u << bit; } quantized = static_cast<int>(stored) - quant_offset; }
            output[block * 32u + i] = static_cast<float>(quantized) * scale;
        }
    }
    return LM_OK;
}

struct lm_kv_storage {
    uint32_t tokens;
    uint32_t references;
    uint8_t used;
    void *key_payload;
    void *value_payload;
};

struct lm_kv_page {
    uint32_t storage_id;
    uint8_t used;
};

struct lm_kv_cache {
    uint32_t page_count;
    uint32_t page_tokens;
    uint32_t free_pages;
    uint32_t free_storages;
    uint32_t key_bytes_per_token;
    uint32_t value_bytes_per_token;
    lm_kv_dtype dtype;
    uint32_t key_elements_per_token;
    uint32_t value_elements_per_token;
    uint8_t typed;
    uint64_t appended_tokens;
    lm_kv_page *pages;
    lm_kv_storage *storages;
};

static int valid_page(const lm_kv_cache *cache, uint32_t page_id) {
    return cache && page_id < cache->page_count && cache->pages[page_id].used &&
           cache->pages[page_id].storage_id < cache->page_count &&
           cache->storages[cache->pages[page_id].storage_id].used;
}

static uint32_t find_free(const lm_kv_storage *items, uint32_t count) {
    for (uint32_t i = 0u; i < count; ++i) if (!items[i].used) return i;
    return UINT32_MAX;
}

static lm_status make_storage(lm_kv_cache *cache, uint32_t tokens, uint32_t *out_id) {
    if (!cache || !out_id || cache->free_storages == 0u) return LM_ERR_CAPACITY;
    const uint32_t id = find_free(cache->storages, cache->page_count);
    if (id == UINT32_MAX) return LM_ERR_CAPACITY;
    const uint64_t key_bytes = static_cast<uint64_t>(cache->page_tokens) * cache->key_bytes_per_token;
    const uint64_t value_bytes = static_cast<uint64_t>(cache->page_tokens) * cache->value_bytes_per_token;
    if (key_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
        value_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) return LM_ERR_CAPACITY;
    void *keys = key_bytes == 0u ? nullptr : std::calloc(1u, static_cast<size_t>(key_bytes));
    void *values = value_bytes == 0u ? nullptr : std::calloc(1u, static_cast<size_t>(value_bytes));
    if ((key_bytes != 0u && !keys) || (value_bytes != 0u && !values)) {
        std::free(keys);
        std::free(values);
        return LM_ERR_CAPACITY;
    }
    cache->storages[id].used = 1u;
    cache->storages[id].tokens = tokens;
    cache->storages[id].references = 1u;
    cache->storages[id].key_payload = keys;
    cache->storages[id].value_payload = values;
    cache->free_storages--;
    *out_id = id;
    return LM_OK;
}

static lm_status make_page_safe(lm_kv_cache *cache, uint32_t storage_id, uint32_t *out_id) {
    if (!cache || !out_id || cache->free_pages == 0u) return LM_ERR_CAPACITY;
    uint32_t id = UINT32_MAX;
    for (uint32_t i = 0u; i < cache->page_count; ++i) {
        if (!cache->pages[i].used) { id = i; break; }
    }
    if (id == UINT32_MAX) return LM_ERR_CAPACITY;
    cache->pages[id].used = 1u;
    cache->pages[id].storage_id = storage_id;
    cache->free_pages--;
    *out_id = id;
    return LM_OK;
}

static lm_status detach_for_write(lm_kv_cache *cache, uint32_t page_id) {
    if (!valid_page(cache, page_id)) return LM_ERR_ARGUMENT;
    lm_kv_page *page = &cache->pages[page_id];
    lm_kv_storage *old = &cache->storages[page->storage_id];
    if (old->references <= 1u) return LM_OK;
    uint32_t replacement = UINT32_MAX;
    const lm_status status = make_storage(cache, old->tokens, &replacement);
    if (status != LM_OK) return status;
    const uint64_t key_bytes = static_cast<uint64_t>(cache->page_tokens) * cache->key_bytes_per_token;
    const uint64_t value_bytes = static_cast<uint64_t>(cache->page_tokens) * cache->value_bytes_per_token;
    if (key_bytes != 0u) std::memcpy(cache->storages[replacement].key_payload, old->key_payload, static_cast<size_t>(key_bytes));
    if (value_bytes != 0u) std::memcpy(cache->storages[replacement].value_payload, old->value_payload, static_cast<size_t>(value_bytes));
    old->references--;
    page->storage_id = replacement;
    return LM_OK;
}

lm_status lm_kv_cache_create_with_payload(uint32_t page_count, uint32_t page_tokens,
                                          uint32_t key_bytes_per_token,
                                          uint32_t value_bytes_per_token,
                                          lm_kv_cache **out_cache) {
    if (!out_cache || page_count == 0u || page_tokens == 0u ||
        key_bytes_per_token == 0u || value_bytes_per_token == 0u)
        return LM_ERR_ARGUMENT;
    *out_cache = nullptr;
    const uint64_t token_bytes = static_cast<uint64_t>(key_bytes_per_token) + value_bytes_per_token;
    const uint64_t max_u64 = std::numeric_limits<uint64_t>::max();
    if (token_bytes > max_u64 / page_tokens) return LM_ERR_CAPACITY;
    const uint64_t page_total = static_cast<uint64_t>(page_tokens) * token_bytes;
    if (page_total > max_u64 / page_count) return LM_ERR_CAPACITY;
    const uint64_t total_payload = static_cast<uint64_t>(page_count) * page_total;
    if (page_total == 0u || total_payload > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
        return LM_ERR_CAPACITY;
    lm_kv_cache *cache = static_cast<lm_kv_cache *>(std::calloc(1u, sizeof(*cache)));
    if (!cache) return LM_ERR_CAPACITY;
    cache->pages = static_cast<lm_kv_page *>(std::calloc(page_count, sizeof(*cache->pages)));
    cache->storages = static_cast<lm_kv_storage *>(std::calloc(page_count, sizeof(*cache->storages)));
    if (!cache->pages || !cache->storages) {
        std::free(cache->pages);
        std::free(cache->storages);
        std::free(cache);
        return LM_ERR_CAPACITY;
    }
    cache->page_count = page_count;
    cache->page_tokens = page_tokens;
    cache->key_bytes_per_token = key_bytes_per_token;
    cache->value_bytes_per_token = value_bytes_per_token;
    cache->free_pages = page_count;
    cache->free_storages = page_count;
    *out_cache = cache;
    return LM_OK;
}

lm_status lm_kv_cache_create_typed(uint32_t page_count, uint32_t page_tokens,
                                   lm_kv_dtype dtype, uint32_t key_elements_per_token,
                                   uint32_t value_elements_per_token, lm_kv_cache **out_cache) {
    if (!out_cache || key_elements_per_token == 0u || value_elements_per_token == 0u ||
        (dtype != LM_KV_F16 && dtype != LM_KV_BF16 && dtype != LM_KV_Q8 &&
         dtype != LM_KV_Q6 && dtype != LM_KV_Q4)) return LM_ERR_ARGUMENT;
    const uint32_t key_bytes = kv_codec_bytes(dtype, key_elements_per_token);
    const uint32_t value_bytes = kv_codec_bytes(dtype, value_elements_per_token);
    if (key_bytes == 0u || value_bytes == 0u) return LM_ERR_CAPACITY;
    const lm_status created = lm_kv_cache_create_with_payload(page_count, page_tokens,
                                                               key_bytes, value_bytes, out_cache);
    if (created != LM_OK) return created;
    (*out_cache)->dtype = dtype;
    (*out_cache)->key_elements_per_token = key_elements_per_token;
    (*out_cache)->value_elements_per_token = value_elements_per_token;
    (*out_cache)->typed = 1u;
    return LM_OK;
}

lm_status lm_kv_cache_create(uint32_t page_count, uint32_t page_tokens,
                             lm_kv_cache **out_cache) {
    if (!out_cache || page_count == 0u || page_tokens == 0u) return LM_ERR_ARGUMENT;
    *out_cache = nullptr;
    lm_kv_cache *cache = static_cast<lm_kv_cache *>(std::calloc(1u, sizeof(*cache)));
    if (!cache) return LM_ERR_CAPACITY;
    cache->pages = static_cast<lm_kv_page *>(std::calloc(page_count, sizeof(*cache->pages)));
    cache->storages = static_cast<lm_kv_storage *>(std::calloc(page_count, sizeof(*cache->storages)));
    if (!cache->pages || !cache->storages) {
        std::free(cache->pages);
        std::free(cache->storages);
        std::free(cache);
        return LM_ERR_CAPACITY;
    }
    cache->page_count = page_count;
    cache->page_tokens = page_tokens;
    cache->free_pages = page_count;
    cache->free_storages = page_count;
    *out_cache = cache;
    return LM_OK;
}

void lm_kv_cache_destroy(lm_kv_cache *cache) {
    if (!cache) return;
    for (uint32_t i = 0u; i < cache->page_count; ++i) {
        std::free(cache->storages[i].key_payload);
        std::free(cache->storages[i].value_payload);
    }
    std::free(cache->pages);
    std::free(cache->storages);
    std::free(cache);
}

lm_status lm_kv_cache_append(lm_kv_cache *cache, uint32_t *page_id,
                             uint32_t token_count) {
    if (!cache || !page_id || token_count == 0u || token_count > cache->page_tokens)
        return LM_ERR_ARGUMENT;
    if (*page_id == UINT32_MAX) {
        uint32_t storage_id = UINT32_MAX;
        lm_status status = make_storage(cache, 0u, &storage_id);
        if (status != LM_OK) return status;
        status = make_page_safe(cache, storage_id, page_id);
        if (status != LM_OK) {
            std::free(cache->storages[storage_id].key_payload);
            std::free(cache->storages[storage_id].value_payload);
            std::memset(&cache->storages[storage_id], 0, sizeof(cache->storages[storage_id]));
            cache->free_storages++;
            return status;
        }
    }
    if (!valid_page(cache, *page_id)) return LM_ERR_ARGUMENT;
    lm_status status = detach_for_write(cache, *page_id);
    if (status != LM_OK) return status;
    lm_kv_storage *storage = &cache->storages[cache->pages[*page_id].storage_id];
    if (storage->tokens + token_count > cache->page_tokens) return LM_ERR_CAPACITY;
    storage->tokens += token_count;
    cache->appended_tokens += token_count;
    return LM_OK;
}

static lm_status validate_payload_range(const lm_kv_cache *cache, uint32_t page_id,
                                        uint32_t token_offset, uint32_t token_count) {
    if (!cache || cache->key_bytes_per_token == 0u || cache->value_bytes_per_token == 0u)
        return LM_ERR_UNSUPPORTED;
    if (!valid_page(cache, page_id) || token_count == 0u || token_offset > cache->page_tokens ||
        token_count > cache->page_tokens - token_offset)
        return LM_ERR_ARGUMENT;
    const lm_kv_storage *storage = &cache->storages[cache->pages[page_id].storage_id];
    if (token_offset > storage->tokens || token_count > storage->tokens - token_offset)
        return LM_ERR_RANGE;
    return LM_OK;
}

lm_status lm_kv_cache_get_payload_layout(const lm_kv_cache *cache,
                                         uint32_t *key_bytes_per_token,
                                         uint32_t *value_bytes_per_token) {
    if (!cache || !key_bytes_per_token || !value_bytes_per_token) return LM_ERR_ARGUMENT;
    if (cache->key_bytes_per_token == 0u || cache->value_bytes_per_token == 0u)
        return LM_ERR_UNSUPPORTED;
    *key_bytes_per_token = cache->key_bytes_per_token;
    *value_bytes_per_token = cache->value_bytes_per_token;
    return LM_OK;
}

lm_status lm_kv_cache_get_codec(const lm_kv_cache *cache, lm_kv_dtype *dtype,
                                uint32_t *key_elements_per_token,
                                uint32_t *value_elements_per_token) {
    if (!cache || !dtype || !key_elements_per_token || !value_elements_per_token) return LM_ERR_ARGUMENT;
    if (!cache->typed) return LM_ERR_UNSUPPORTED;
    *dtype = cache->dtype;
    *key_elements_per_token = cache->key_elements_per_token;
    *value_elements_per_token = cache->value_elements_per_token;
    return LM_OK;
}

lm_status lm_kv_cache_write_payload(lm_kv_cache *cache, uint32_t page_id,
                                     uint32_t token_offset, uint32_t token_count,
                                     const void *keys, const void *values) {
    if (!keys || !values) return LM_ERR_ARGUMENT;
    const lm_status valid = validate_payload_range(cache, page_id, token_offset, token_count);
    if (valid != LM_OK) return valid;
    const lm_status detached = detach_for_write(cache, page_id);
    if (detached != LM_OK) return detached;
    lm_kv_storage *storage = &cache->storages[cache->pages[page_id].storage_id];
    const size_t key_offset = static_cast<size_t>(token_offset) * cache->key_bytes_per_token;
    const size_t value_offset = static_cast<size_t>(token_offset) * cache->value_bytes_per_token;
    const size_t key_bytes = static_cast<size_t>(token_count) * cache->key_bytes_per_token;
    const size_t value_bytes = static_cast<size_t>(token_count) * cache->value_bytes_per_token;
    std::memcpy(static_cast<unsigned char *>(storage->key_payload) + key_offset, keys, key_bytes);
    std::memcpy(static_cast<unsigned char *>(storage->value_payload) + value_offset, values, value_bytes);
    return LM_OK;
}

lm_status lm_kv_cache_read_payload(const lm_kv_cache *cache, uint32_t page_id,
                                    uint32_t token_offset, uint32_t token_count,
                                    void *keys, void *values) {
    if (!keys || !values) return LM_ERR_ARGUMENT;
    const lm_status valid = validate_payload_range(cache, page_id, token_offset, token_count);
    if (valid != LM_OK) return valid;
    const lm_kv_storage *storage = &cache->storages[cache->pages[page_id].storage_id];
    const size_t key_offset = static_cast<size_t>(token_offset) * cache->key_bytes_per_token;
    const size_t value_offset = static_cast<size_t>(token_offset) * cache->value_bytes_per_token;
    const size_t key_bytes = static_cast<size_t>(token_count) * cache->key_bytes_per_token;
    const size_t value_bytes = static_cast<size_t>(token_count) * cache->value_bytes_per_token;
    std::memcpy(keys, static_cast<const unsigned char *>(storage->key_payload) + key_offset, key_bytes);
    std::memcpy(values, static_cast<const unsigned char *>(storage->value_payload) + value_offset, value_bytes);
    return LM_OK;
}

lm_status lm_kv_cache_write_f32(lm_kv_cache *cache, uint32_t page_id,
                                uint32_t token_offset, uint32_t token_count,
                                const float *keys, const float *values) {
    if (!cache || !cache->typed || !keys || !values) return LM_ERR_UNSUPPORTED;
    const lm_status valid = validate_payload_range(cache, page_id, token_offset, token_count);
    if (valid != LM_OK) return valid;
    if (token_count > UINT32_MAX / cache->key_elements_per_token || token_count > UINT32_MAX / cache->value_elements_per_token)
        return LM_ERR_CAPACITY;
    const size_t key_elements = static_cast<size_t>(token_count) * cache->key_elements_per_token;
    const size_t value_elements = static_cast<size_t>(token_count) * cache->value_elements_per_token;
    for (size_t i = 0u; i < key_elements; ++i) if (!std::isfinite(keys[i])) return LM_ERR_RANGE;
    for (size_t i = 0u; i < value_elements; ++i) if (!std::isfinite(values[i])) return LM_ERR_RANGE;
    const lm_status detached = detach_for_write(cache, page_id);
    if (detached != LM_OK) return detached;
    lm_kv_storage *storage = &cache->storages[cache->pages[page_id].storage_id];
    unsigned char *key_dst = static_cast<unsigned char *>(storage->key_payload) + static_cast<size_t>(token_offset) * cache->key_bytes_per_token;
    unsigned char *value_dst = static_cast<unsigned char *>(storage->value_payload) + static_cast<size_t>(token_offset) * cache->value_bytes_per_token;
    for (uint32_t token = 0u; token < token_count; ++token) {
        lm_status status = kv_codec_write(keys + static_cast<size_t>(token) * cache->key_elements_per_token,
                                          cache->key_elements_per_token, cache->dtype,
                                          key_dst + static_cast<size_t>(token) * cache->key_bytes_per_token,
                                          cache->key_bytes_per_token);
        if (status != LM_OK) return status;
        status = kv_codec_write(values + static_cast<size_t>(token) * cache->value_elements_per_token,
                                cache->value_elements_per_token, cache->dtype,
                                value_dst + static_cast<size_t>(token) * cache->value_bytes_per_token,
                                cache->value_bytes_per_token);
        if (status != LM_OK) return status;
    }
    return LM_OK;
}

lm_status lm_kv_cache_read_f32(const lm_kv_cache *cache, uint32_t page_id,
                               uint32_t token_offset, uint32_t token_count,
                               float *keys, float *values) {
    if (!cache || !cache->typed || !keys || !values) return LM_ERR_UNSUPPORTED;
    const lm_status valid = validate_payload_range(cache, page_id, token_offset, token_count);
    if (valid != LM_OK) return valid;
    const lm_kv_storage *storage = &cache->storages[cache->pages[page_id].storage_id];
    const unsigned char *key_src = static_cast<const unsigned char *>(storage->key_payload) + static_cast<size_t>(token_offset) * cache->key_bytes_per_token;
    const unsigned char *value_src = static_cast<const unsigned char *>(storage->value_payload) + static_cast<size_t>(token_offset) * cache->value_bytes_per_token;
    for (uint32_t token = 0u; token < token_count; ++token) {
        lm_status status = kv_codec_read(key_src + static_cast<size_t>(token) * cache->key_bytes_per_token,
                                         cache->key_elements_per_token, cache->dtype,
                                         cache->key_bytes_per_token,
                                         keys + static_cast<size_t>(token) * cache->key_elements_per_token);
        if (status != LM_OK) return status;
        status = kv_codec_read(value_src + static_cast<size_t>(token) * cache->value_bytes_per_token,
                               cache->value_elements_per_token, cache->dtype,
                               cache->value_bytes_per_token,
                               values + static_cast<size_t>(token) * cache->value_elements_per_token);
        if (status != LM_OK) return status;
    }
    return LM_OK;
}

lm_status lm_kv_cache_page_token_count(const lm_kv_cache *cache, uint32_t page_id,
                                       uint32_t *out_tokens) {
    if (!cache || !out_tokens || !valid_page(cache, page_id)) return LM_ERR_ARGUMENT;
    const lm_kv_page *page = &cache->pages[page_id];
    *out_tokens = cache->storages[page->storage_id].tokens;
    return LM_OK;
}

lm_status lm_kv_cache_fork(lm_kv_cache *cache, uint32_t source_page,
                           uint32_t *out_page) {
    if (!valid_page(cache, source_page) || !out_page) return LM_ERR_ARGUMENT;
    lm_kv_storage *storage = &cache->storages[cache->pages[source_page].storage_id];
    lm_status status = make_page_safe(cache, cache->pages[source_page].storage_id, out_page);
    if (status != LM_OK) return status;
    storage->references++;
    return LM_OK;
}

lm_status lm_kv_cache_rollback(lm_kv_cache *cache, uint32_t page_id,
                               uint32_t token_count) {
    if (!valid_page(cache, page_id)) return LM_ERR_ARGUMENT;
    lm_kv_storage *before = &cache->storages[cache->pages[page_id].storage_id];
    if (token_count > before->tokens) return LM_ERR_ARGUMENT;
    const uint32_t old_tokens = before->tokens;
    lm_status status = detach_for_write(cache, page_id);
    if (status != LM_OK) return status;
    lm_kv_storage *storage = &cache->storages[cache->pages[page_id].storage_id];
    storage->tokens -= token_count;
    const size_t first_key_byte = static_cast<size_t>(storage->tokens) * cache->key_bytes_per_token;
    const size_t first_value_byte = static_cast<size_t>(storage->tokens) * cache->value_bytes_per_token;
    const size_t removed_key_bytes = static_cast<size_t>(old_tokens - storage->tokens) * cache->key_bytes_per_token;
    const size_t removed_value_bytes = static_cast<size_t>(old_tokens - storage->tokens) * cache->value_bytes_per_token;
    if (removed_key_bytes != 0u) std::memset(static_cast<unsigned char *>(storage->key_payload) + first_key_byte, 0, removed_key_bytes);
    if (removed_value_bytes != 0u) std::memset(static_cast<unsigned char *>(storage->value_payload) + first_value_byte, 0, removed_value_bytes);
    if (cache->appended_tokens >= token_count) cache->appended_tokens -= token_count;
    return LM_OK;
}

lm_status lm_kv_cache_release(lm_kv_cache *cache, uint32_t page_id) {
    if (!valid_page(cache, page_id)) return LM_ERR_ARGUMENT;
    lm_kv_page *page = &cache->pages[page_id];
    lm_kv_storage *storage = &cache->storages[page->storage_id];
    if (storage->references == 0u) return LM_ERR_STATE;
    storage->references--;
    if (storage->references == 0u) {
        std::free(storage->key_payload);
        std::free(storage->value_payload);
        std::memset(storage, 0, sizeof(*storage));
        cache->free_storages++;
    }
    std::memset(page, 0, sizeof(*page));
    cache->free_pages++;
    return LM_OK;
}

lm_status lm_kv_cache_get_stats(const lm_kv_cache *cache, lm_kv_stats *out_stats) {
    if (!cache || !out_stats) return LM_ERR_ARGUMENT;
    std::memset(out_stats, 0, sizeof(*out_stats));
    out_stats->page_tokens = cache->page_tokens;
    out_stats->total_pages = cache->page_count;
    out_stats->free_pages = cache->free_pages;
    out_stats->appended_tokens = cache->appended_tokens;
    for (uint32_t i = 0u; i < cache->page_count; ++i) {
        if (cache->pages[i].used) out_stats->used_pages++;
    }
    for (uint32_t i = 0u; i < cache->page_count; ++i) {
        if (cache->storages[i].used && cache->storages[i].references > 1u)
            out_stats->shared_pages++;
    }
    return LM_OK;
}
