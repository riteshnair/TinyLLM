#include "lm/lm.h"

#include <cstdlib>
#include <cstring>
#include <limits>

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
