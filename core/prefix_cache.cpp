#include "lm/lm.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

namespace {

struct PrefixEntry {
    uint64_t model_identity;
    uint64_t settings_identity;
    uint64_t last_used;
    uint32_t page_id;
    std::vector<uint32_t> tokens;
};

void put32(unsigned char *out, uint32_t value) {
    for (uint32_t i = 0u; i < 4u; ++i) out[i] = static_cast<unsigned char>(value >> (i * 8u));
}

void put64(unsigned char *out, uint64_t value) {
    for (uint32_t i = 0u; i < 8u; ++i) out[i] = static_cast<unsigned char>(value >> (i * 8u));
}

uint32_t get32(const unsigned char *data) {
    uint32_t value = 0u;
    for (uint32_t i = 0u; i < 4u; ++i) value |= static_cast<uint32_t>(data[i]) << (i * 8u);
    return value;
}

uint64_t get64(const unsigned char *data) {
    uint64_t value = 0u;
    for (uint32_t i = 0u; i < 8u; ++i) value |= static_cast<uint64_t>(data[i]) << (i * 8u);
    return value;
}

} // namespace

struct lm_prefix_cache {
    uint32_t max_entries;
    uint32_t max_tokens;
    uint64_t clock;
    lm_prefix_cache_stats stats;
    std::vector<PrefixEntry> entries;
};

lm_status lm_prefix_cache_create(uint32_t max_entries, uint32_t max_tokens,
                                 lm_prefix_cache **out_cache) {
    if (!out_cache || max_entries == 0u || max_tokens == 0u || max_entries > (1u << 20u) ||
        max_tokens > (1u << 20u)) return LM_ERR_ARGUMENT;
    *out_cache = nullptr;
    try {
        lm_prefix_cache *cache = new lm_prefix_cache();
        cache->max_entries = max_entries;
        cache->max_tokens = max_tokens;
        cache->clock = 0u;
        cache->stats = {0u, max_entries, 0u, 0u, 0u};
        cache->entries.reserve(max_entries);
        *out_cache = cache;
        return LM_OK;
    } catch (const std::bad_alloc &) {
        return LM_ERR_CAPACITY;
    }
}

void lm_prefix_cache_destroy(lm_prefix_cache *cache) {
    delete cache;
}

lm_status lm_prefix_cache_insert(lm_prefix_cache *cache, uint64_t model_identity,
                                 uint64_t settings_identity, const uint32_t *tokens,
                                 size_t token_count, uint32_t page_id) {
    if (!cache || !tokens || token_count == 0u || token_count > cache->max_tokens) return LM_ERR_ARGUMENT;
    for (PrefixEntry &entry : cache->entries) {
        if (entry.model_identity != model_identity || entry.settings_identity != settings_identity ||
            entry.tokens.size() != token_count || std::memcmp(entry.tokens.data(), tokens, token_count * sizeof(uint32_t)) != 0)
            continue;
        entry.page_id = page_id;
        entry.last_used = ++cache->clock;
        return LM_OK;
    }
    try {
        if (cache->entries.size() == cache->max_entries) {
            const auto victim = std::min_element(cache->entries.begin(), cache->entries.end(),
                [](const PrefixEntry &a, const PrefixEntry &b) { return a.last_used < b.last_used; });
            if (victim == cache->entries.end()) return LM_ERR_STATE;
            *victim = PrefixEntry{model_identity, settings_identity, ++cache->clock, page_id,
                                  std::vector<uint32_t>(tokens, tokens + token_count)};
            ++cache->stats.evictions;
            return LM_OK;
        }
        cache->entries.push_back(PrefixEntry{model_identity, settings_identity, ++cache->clock, page_id,
                                             std::vector<uint32_t>(tokens, tokens + token_count)});
        cache->stats.entries = static_cast<uint32_t>(cache->entries.size());
        return LM_OK;
    } catch (const std::bad_alloc &) {
        return LM_ERR_CAPACITY;
    }
}

lm_status lm_prefix_cache_lookup(lm_prefix_cache *cache, uint64_t model_identity,
                                 uint64_t settings_identity, const uint32_t *tokens,
                                 size_t token_count, uint32_t *out_page_id,
                                 size_t *out_prefix_tokens) {
    if (!cache || !tokens || token_count == 0u || !out_page_id || !out_prefix_tokens ||
        token_count > cache->max_tokens) return LM_ERR_ARGUMENT;
    PrefixEntry *best = nullptr;
    for (PrefixEntry &entry : cache->entries) {
        if (entry.model_identity != model_identity || entry.settings_identity != settings_identity ||
            entry.tokens.size() > token_count ||
            std::memcmp(entry.tokens.data(), tokens, entry.tokens.size() * sizeof(uint32_t)) != 0) continue;
        if (!best || entry.tokens.size() > best->tokens.size()) best = &entry;
    }
    if (!best) {
        ++cache->stats.misses;
        *out_prefix_tokens = 0u;
        return LM_ERR_RANGE;
    }
    best->last_used = ++cache->clock;
    ++cache->stats.hits;
    *out_page_id = best->page_id;
    *out_prefix_tokens = best->tokens.size();
    return LM_OK;
}

lm_status lm_prefix_cache_erase_page(lm_prefix_cache *cache, uint32_t page_id) {
    if (!cache) return LM_ERR_ARGUMENT;
    cache->entries.erase(std::remove_if(cache->entries.begin(), cache->entries.end(),
                                        [page_id](const PrefixEntry &entry) { return entry.page_id == page_id; }),
                         cache->entries.end());
    cache->stats.entries = static_cast<uint32_t>(cache->entries.size());
    return LM_OK;
}

lm_status lm_prefix_cache_get_stats(const lm_prefix_cache *cache, lm_prefix_cache_stats *out_stats) {
    if (!cache || !out_stats) return LM_ERR_ARGUMENT;
    *out_stats = cache->stats;
    return LM_OK;
}

lm_status lm_prefix_cache_export_size(const lm_prefix_cache *cache, size_t *out_bytes) {
    if (!cache || !out_bytes) return LM_ERR_ARGUMENT;
    size_t total = 16u;
    for (const PrefixEntry &entry : cache->entries) {
        if (entry.tokens.size() > (std::numeric_limits<size_t>::max() - total - 24u) / sizeof(uint32_t))
            return LM_ERR_CAPACITY;
        total += 24u + entry.tokens.size() * sizeof(uint32_t);
    }
    *out_bytes = total;
    return LM_OK;
}

lm_status lm_prefix_cache_export(const lm_prefix_cache *cache, void *out_data,
                                 size_t data_capacity, size_t *out_bytes) {
    if (!cache || !out_bytes) return LM_ERR_ARGUMENT;
    size_t required = 0u;
    const lm_status sized = lm_prefix_cache_export_size(cache, &required);
    if (sized != LM_OK) return sized;
    *out_bytes = required;
    if (!out_data || data_capacity < required) return LM_ERR_CAPACITY;
    unsigned char *cursor = static_cast<unsigned char *>(out_data);
    std::memcpy(cursor, "TLPFX001", 8u);
    put32(cursor + 8u, 1u);
    put32(cursor + 12u, static_cast<uint32_t>(cache->entries.size()));
    cursor += 16u;
    for (const PrefixEntry &entry : cache->entries) {
        put64(cursor, entry.model_identity);
        put64(cursor + 8u, entry.settings_identity);
        put32(cursor + 16u, entry.page_id);
        put32(cursor + 20u, static_cast<uint32_t>(entry.tokens.size()));
        cursor += 24u;
        for (uint32_t token : entry.tokens) { put32(cursor, token); cursor += 4u; }
    }
    return LM_OK;
}

lm_status lm_prefix_cache_import(lm_prefix_cache *cache, const void *data, size_t data_bytes) {
    if (!cache || !data || data_bytes < 16u) return LM_ERR_ARGUMENT;
    const unsigned char *cursor = static_cast<const unsigned char *>(data);
    if (std::memcmp(cursor, "TLPFX001", 8u) != 0 || get32(cursor + 8u) != 1u) return LM_ERR_PARSE;
    const uint32_t count = get32(cursor + 12u);
    if (count > cache->max_entries) return LM_ERR_CAPACITY;
    cursor += 16u;
    size_t remaining = data_bytes - 16u;
    std::vector<PrefixEntry> imported;
    try { imported.reserve(count); } catch (const std::bad_alloc &) { return LM_ERR_CAPACITY; }
    for (uint32_t i = 0u; i < count; ++i) {
        if (remaining < 24u) return LM_ERR_PARSE;
        const uint64_t model_identity = get64(cursor);
        const uint64_t settings_identity = get64(cursor + 8u);
        const uint32_t page_id = get32(cursor + 16u);
        const uint32_t token_count = get32(cursor + 20u);
        cursor += 24u; remaining -= 24u;
        if (token_count == 0u || token_count > cache->max_tokens ||
            token_count > remaining / sizeof(uint32_t)) return LM_ERR_PARSE;
        PrefixEntry entry{model_identity, settings_identity, static_cast<uint64_t>(i + 1u), page_id, {}};
        try {
            entry.tokens.resize(token_count);
            for (uint32_t token = 0u; token < token_count; ++token)
                entry.tokens[token] = get32(cursor + static_cast<size_t>(token) * sizeof(uint32_t));
        } catch (const std::bad_alloc &) { return LM_ERR_CAPACITY; }
        imported.push_back(std::move(entry));
        cursor += static_cast<size_t>(token_count) * sizeof(uint32_t);
        remaining -= static_cast<size_t>(token_count) * sizeof(uint32_t);
    }
    if (remaining != 0u) return LM_ERR_PARSE;
    cache->entries.swap(imported);
    cache->clock = count;
    cache->stats.entries = count;
    cache->stats.hits = 0u;
    cache->stats.misses = 0u;
    cache->stats.evictions = 0u;
    return LM_OK;
}
