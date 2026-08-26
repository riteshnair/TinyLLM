#include "lm/lm.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {

struct Candidate {
    uint32_t token;
    float logit;
};

uint64_t next_random(uint64_t *state) {
    uint64_t value = *state;
    if (value == 0u) value = 0x9e3779b97f4a7c15ull;
    value ^= value >> 12u;
    value ^= value << 25u;
    value ^= value >> 27u;
    *state = value;
    return value * 0x2545f4914f6cdd1dull;
}

} // namespace

lm_status lm_sample_logits(const float *logits, uint32_t vocab_size,
                           const lm_sampling_config *config, uint32_t *out_token,
                           float *out_probability) {
    if (!logits || !config || !out_token || !out_probability || vocab_size == 0u ||
        (config->mode != LM_SAMPLING_GREEDY && config->mode != LM_SAMPLING_TOP_K))
        return LM_ERR_ARGUMENT;
    uint32_t best = 0u;
    float best_logit = logits[0];
    if (!std::isfinite(best_logit)) return LM_ERR_RANGE;
    for (uint32_t token = 1u; token < vocab_size; ++token) {
        if (!std::isfinite(logits[token])) return LM_ERR_RANGE;
        if (logits[token] > best_logit) {
            best = token;
            best_logit = logits[token];
        }
    }
    if (config->mode == LM_SAMPLING_GREEDY || config->top_k == 1u) {
        *out_token = best;
        *out_probability = 1.0f;
        return LM_OK;
    }
    if (!(config->temperature > 0.0f) || !std::isfinite(config->temperature) ||
        config->top_k == 0u || config->top_k > vocab_size)
        return LM_ERR_ARGUMENT;
    std::vector<Candidate> candidates;
    candidates.reserve(vocab_size);
    for (uint32_t token = 0u; token < vocab_size; ++token)
        candidates.push_back(Candidate{token, logits[token] / config->temperature});
    const uint32_t count = std::min(config->top_k, vocab_size);
    std::partial_sort(candidates.begin(), candidates.begin() + count, candidates.end(),
                      [](const Candidate &a, const Candidate &b) {
                          if (a.logit != b.logit) return a.logit > b.logit;
                          return a.token < b.token;
                      });
    const float max_logit = candidates[0].logit;
    float total = 0.0f;
    for (uint32_t i = 0u; i < count; ++i) total += std::exp(candidates[i].logit - max_logit);
    if (!(total > 0.0f) || !std::isfinite(total)) return LM_ERR_RANGE;
    uint64_t random_state = config->seed;
    const float unit = static_cast<float>(next_random(&random_state) >> 40u) /
                       static_cast<float>(1ull << 24u);
    float threshold = unit * total;
    for (uint32_t i = 0u; i < count; ++i) {
        const float probability = std::exp(candidates[i].logit - max_logit) / total;
        if (threshold <= std::exp(candidates[i].logit - max_logit)) {
            *out_token = candidates[i].token;
            *out_probability = probability;
            return LM_OK;
        }
        threshold -= std::exp(candidates[i].logit - max_logit);
    }
    *out_token = candidates[count - 1u].token;
    *out_probability = std::exp(candidates[count - 1u].logit - max_logit) / total;
    return LM_OK;
}
