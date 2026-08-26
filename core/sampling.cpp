#include "lm/lm.h"

#include <algorithm>
#include <cmath>
#include <cstring>
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

bool descending(const Candidate &a, const Candidate &b) {
    if (a.logit != b.logit) return a.logit > b.logit;
    return a.token < b.token;
}

uint32_t history_count(const lm_sampling_config *config, uint32_t token) {
    uint32_t count = 0u;
    for (size_t i = 0u; i < config->history_count; ++i)
        if (config->history_tokens[i] == token && count != UINT32_MAX) ++count;
    return count;
}

} // namespace

void lm_sampling_config_init(lm_sampling_config *config) {
    if (!config) return;
    *config = {};
    config->mode = LM_SAMPLING_GREEDY;
    config->temperature = 1.0f;
    config->seed = 1u;
}

lm_status lm_sample_logits(const float *logits, uint32_t vocab_size,
                           const lm_sampling_config *config, uint32_t *out_token,
                           float *out_probability) {
    if (!logits || !config || !out_token || !out_probability || vocab_size == 0u ||
        (config->mode != LM_SAMPLING_GREEDY && config->mode != LM_SAMPLING_TOP_K &&
         config->mode != LM_SAMPLING_TOP_P && config->mode != LM_SAMPLING_TYPICAL))
        return LM_ERR_ARGUMENT;
    if (config->history_count != 0u && !config->history_tokens) return LM_ERR_ARGUMENT;
    if (config->mode != LM_SAMPLING_GREEDY &&
        (!(config->temperature > 0.0f) || !std::isfinite(config->temperature))) return LM_ERR_ARGUMENT;
    if (config->top_k > vocab_size) return LM_ERR_ARGUMENT;
    if (config->mode == LM_SAMPLING_TOP_K && config->top_k == 0u) return LM_ERR_ARGUMENT;
    if (config->mode == LM_SAMPLING_TOP_P && !(config->top_p > 0.0f && config->top_p <= 1.0f)) return LM_ERR_ARGUMENT;
    if (config->mode == LM_SAMPLING_TYPICAL && !(config->typical_p > 0.0f && config->typical_p <= 1.0f)) return LM_ERR_ARGUMENT;
    if (config->top_p != 0.0f && (!(config->top_p > 0.0f) || config->top_p > 1.0f)) return LM_ERR_ARGUMENT;
    if (config->min_p != 0.0f && (!(config->min_p > 0.0f) || config->min_p > 1.0f)) return LM_ERR_ARGUMENT;
    if (config->typical_p != 0.0f && (!(config->typical_p > 0.0f) || config->typical_p > 1.0f)) return LM_ERR_ARGUMENT;
    if (config->top_p != 0.0f && config->typical_p != 0.0f) return LM_ERR_ARGUMENT;
    if (!std::isfinite(config->repetition_penalty) || config->repetition_penalty < 0.0f ||
        (config->repetition_penalty != 0.0f && config->repetition_penalty < 1.0f) ||
        !std::isfinite(config->frequency_penalty) || config->frequency_penalty < 0.0f ||
        !std::isfinite(config->presence_penalty) || config->presence_penalty < 0.0f)
        return LM_ERR_ARGUMENT;

    std::vector<Candidate> candidates;
    std::vector<float> policy_logits;
    try {
        candidates.reserve(vocab_size);
        policy_logits.assign(logits, logits + vocab_size);
    } catch (const std::bad_alloc &) { return LM_ERR_CAPACITY; }
    if (config->processor) {
        const lm_status processed = config->processor(config->processor_user, policy_logits.data(), vocab_size,
                                                      config->history_tokens, config->history_count);
        if (processed != LM_OK) return processed;
    }
    const float temperature = config->temperature > 0.0f ? config->temperature : 1.0f;
    for (uint32_t token = 0u; token < vocab_size; ++token) {
        if (std::isnan(policy_logits[token]) || policy_logits[token] == std::numeric_limits<float>::infinity()) return LM_ERR_RANGE;
        const uint32_t count = history_count(config, token);
        float value = policy_logits[token];
        if (count != 0u && config->repetition_penalty > 1.0f)
            value = value < 0.0f ? value * config->repetition_penalty : value / config->repetition_penalty;
        value -= static_cast<float>(count) * config->frequency_penalty;
        if (count != 0u) value -= config->presence_penalty;
        value /= temperature;
        if (std::isnan(value) || value == std::numeric_limits<float>::infinity()) return LM_ERR_RANGE;
        candidates.push_back(Candidate{token, value});
    }
    std::sort(candidates.begin(), candidates.end(), descending);
    if (!std::isfinite(candidates[0].logit)) return LM_ERR_RANGE;
    size_t limit = candidates.size();
    if (config->top_k != 0u) limit = std::min(limit, static_cast<size_t>(config->top_k));
    if (limit == 0u) return LM_ERR_ARGUMENT;

    if (config->top_p != 0.0f || config->mode == LM_SAMPLING_TOP_P) {
        const float max_logit = candidates[0].logit;
        float total = 0.0f;
        for (size_t i = 0u; i < limit; ++i) total += std::exp(candidates[i].logit - max_logit);
        if (!(total > 0.0f) || !std::isfinite(total)) return LM_ERR_RANGE;
        const float cutoff = config->mode == LM_SAMPLING_TOP_P ? config->top_p : config->top_p;
        float cumulative = 0.0f;
        size_t kept = 0u;
        for (; kept < limit; ++kept) {
            cumulative += std::exp(candidates[kept].logit - max_logit) / total;
            if (cumulative >= cutoff) { ++kept; break; }
        }
        limit = std::max<size_t>(1u, kept);
    } else if (config->typical_p != 0.0f || config->mode == LM_SAMPLING_TYPICAL) {
        const float max_logit = candidates[0].logit;
        float total = 0.0f;
        for (size_t i = 0u; i < limit; ++i) total += std::exp(candidates[i].logit - max_logit);
        if (!(total > 0.0f) || !std::isfinite(total)) return LM_ERR_RANGE;
        float entropy = 0.0f;
        for (size_t i = 0u; i < limit; ++i) {
            const float probability = std::exp(candidates[i].logit - max_logit) / total;
            entropy -= probability * std::log(std::max(probability, std::numeric_limits<float>::min()));
        }
        std::sort(candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(limit),
                  [entropy, max_logit, total](const Candidate &a, const Candidate &b) {
                      const float pa = std::exp(a.logit - max_logit) / total;
                      const float pb = std::exp(b.logit - max_logit) / total;
                      const float da = std::fabs(-std::log(pa) - entropy);
                      const float db = std::fabs(-std::log(pb) - entropy);
                      if (da != db) return da < db;
                      return a.token < b.token;
                  });
        const float cutoff = config->typical_p;
        float cumulative = 0.0f;
        size_t kept = 0u;
        for (; kept < limit; ++kept) {
            cumulative += std::exp(candidates[kept].logit - max_logit) / total;
            if (cumulative >= cutoff) { ++kept; break; }
        }
        limit = std::max<size_t>(1u, kept);
    }
    if (config->min_p != 0.0f) {
        const float threshold = std::log(config->min_p);
        size_t kept = 0u;
        const float max_logit = candidates[0].logit;
        for (; kept < limit; ++kept)
            if (candidates[kept].logit - max_logit < threshold) break;
        limit = std::max<size_t>(1u, kept);
    }

    if (config->mode == LM_SAMPLING_GREEDY && config->top_k == 0u && config->top_p == 0.0f &&
        config->typical_p == 0.0f && config->min_p == 0.0f) {
        *out_token = candidates[0].token;
        *out_probability = 1.0f;
        return LM_OK;
    }
    const float max_logit = candidates[0].logit;
    float total = 0.0f;
    for (size_t i = 0u; i < limit; ++i) total += std::exp(candidates[i].logit - max_logit);
    if (!(total > 0.0f) || !std::isfinite(total)) return LM_ERR_RANGE;
    if (config->mode == LM_SAMPLING_GREEDY) {
        *out_token = candidates[0].token;
        *out_probability = std::exp(candidates[0].logit - max_logit) / total;
        return LM_OK;
    }
    uint64_t random_state = config->seed;
    const float unit = static_cast<float>(next_random(&random_state) >> 40u) /
                       static_cast<float>(1ull << 24u);
    float threshold = unit * total;
    for (size_t i = 0u; i < limit; ++i) {
        const float mass = std::exp(candidates[i].logit - max_logit);
        const float probability = mass / total;
        if (threshold <= mass) {
            *out_token = candidates[i].token;
            *out_probability = probability;
            return LM_OK;
        }
        threshold -= mass;
    }
    *out_token = candidates[limit - 1u].token;
    *out_probability = std::exp(candidates[limit - 1u].logit - max_logit) / total;
    return LM_OK;
}
