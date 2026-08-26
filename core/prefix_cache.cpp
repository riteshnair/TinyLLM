#include "lm/lm.h"

#include <algorithm>
#include <cstring>
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
