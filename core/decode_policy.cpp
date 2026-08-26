#include "lm/lm.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace {

bool valid_ngram(const lm_ngram_speculative_config *config) {
    return config && config->min_n != 0u && config->max_n >= config->min_n &&
           config->max_draft_tokens != 0u && config->max_n <= 4096u &&
           config->max_draft_tokens <= 4096u;
}

} // namespace

lm_status lm_speculative_ngram_propose(const uint32_t *history_tokens, size_t history_count,
                                       const lm_ngram_speculative_config *config,
                                       uint32_t *out_tokens, size_t token_capacity,
                                       size_t *out_count) {
    if (!out_count || (!history_tokens && history_count != 0u) || !valid_ngram(config) ||
        history_count > (1u << 20u)) return LM_ERR_ARGUMENT;
    size_t found_start = 0u;
    uint32_t found_n = 0u;
    const uint32_t max_n = std::min(config->max_n, static_cast<uint32_t>(history_count / 2u));
    for (uint32_t n = max_n; n >= config->min_n; --n) {
        const size_t suffix = history_count - n;
        for (size_t start = suffix; start != 0u; --start) {
            const size_t candidate = start - 1u;
            if (candidate + n > suffix || std::memcmp(history_tokens + candidate,
                                                       history_tokens + suffix,
                                                       static_cast<size_t>(n) * sizeof(uint32_t)) != 0) continue;
            found_start = candidate;
            found_n = n;
            break;
        }
        if (found_n != 0u || n == config->min_n) break;
    }
    const size_t available = found_n == 0u ? 0u : std::min(static_cast<size_t>(config->max_draft_tokens),
                                                            history_count - found_start - found_n);
    *out_count = available;
    if (available != 0u && (!out_tokens || token_capacity < available)) return LM_ERR_CAPACITY;
    if (available != 0u) std::memcpy(out_tokens, history_tokens + found_start + found_n,
                                      available * sizeof(uint32_t));
    return LM_OK;
}

lm_status lm_speculative_verify_greedy(const uint32_t *draft_tokens, size_t draft_count,
                                       const float *target_logits, size_t target_stride,
                                       uint32_t vocab_size, uint32_t *out_tokens,
                                       size_t token_capacity, size_t *out_count,
                                       lm_speculative_stats *stats) {
    if (!out_count || (!draft_tokens && draft_count != 0u) || !target_logits || !out_tokens ||
        draft_count > 4096u || vocab_size == 0u || target_stride < vocab_size ||
        token_capacity < draft_count + 1u) return LM_ERR_ARGUMENT;
    lm_speculative_stats local{};
    local.proposed_tokens = draft_count;
    local.verification_steps = 1u;
    size_t accepted = 0u;
    size_t draft_accepted = 0u;
    for (size_t row = 0u; row <= draft_count; ++row) {
        uint32_t best = 0u;
        float best_value = target_logits[row * target_stride];
        if (!std::isfinite(best_value)) return LM_ERR_RANGE;
        for (uint32_t token = 1u; token < vocab_size; ++token) {
            const float value = target_logits[row * target_stride + token];
            if (!std::isfinite(value)) return LM_ERR_RANGE;
            if (value > best_value) { best = token; best_value = value; }
        }
        if (row < draft_count) {
            if (best != draft_tokens[row]) {
                out_tokens[accepted++] = best;
                break;
            }
            out_tokens[accepted++] = draft_tokens[row];
            ++draft_accepted;
        } else {
            out_tokens[accepted++] = best;
        }
    }
    local.accepted_tokens = draft_accepted;
    local.rejected_tokens = draft_count - local.accepted_tokens;
    *out_count = accepted;
    if (stats) *stats = local;
    return LM_OK;
}

lm_status lm_speculative_adaptive_depth(const lm_speculative_stats *stats,
                                        uint32_t current_depth, uint32_t min_depth,
                                        uint32_t max_depth, uint32_t *out_depth) {
    if (!stats || !out_depth || min_depth == 0u || min_depth > max_depth ||
        current_depth < min_depth || current_depth > max_depth ||
        stats->accepted_tokens > stats->proposed_tokens || stats->verification_steps == 0u)
        return LM_ERR_ARGUMENT;
    uint32_t depth = current_depth;
    if (stats->proposed_tokens != 0u) {
        if (stats->accepted_tokens * 4u >= stats->proposed_tokens * 3u && depth < max_depth) ++depth;
        else if (stats->accepted_tokens * 4u <= stats->proposed_tokens && depth > min_depth) --depth;
    }
    *out_depth = depth;
    return LM_OK;
}
