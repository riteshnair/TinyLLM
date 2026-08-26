#include "lm/lm.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

namespace {

lm_status parse_mixtral_name(const lm_model_tensor_info &descriptor, lm_moe_tensor_role *role, uint32_t *layer) {
    const char *name = descriptor.name;
    const char prefix[] = "layers.";
    const char gate_suffix[] = ".feed_forward.experts.w1";
    const char down_suffix[] = ".feed_forward.experts.w2";
    const size_t prefix_bytes = sizeof(prefix) - 1u;
    if (std::strncmp(name, prefix, prefix_bytes) != 0) return LM_ERR_UNSUPPORTED;
    const char *cursor = name + prefix_bytes;
    if (*cursor < '0' || *cursor > '9') return LM_ERR_UNSUPPORTED;
    uint64_t layer_value = 0u;
    while (*cursor >= '0' && *cursor <= '9') {
        const uint64_t digit = static_cast<uint64_t>(*cursor - '0');
        if (layer_value > (UINT32_MAX - digit) / 10u) return LM_ERR_RANGE;
        layer_value = layer_value * 10u + digit;
        ++cursor;
    }
    if (std::strcmp(cursor, gate_suffix) == 0) *role = LM_MOE_TENSOR_GATE_UP_EXPERT;
    else if (std::strcmp(cursor, down_suffix) == 0) *role = LM_MOE_TENSOR_DOWN_EXPERT;
    else return LM_ERR_UNSUPPORTED;
    *layer = static_cast<uint32_t>(layer_value);
    return LM_OK;
}

bool finite_logits(const float *logits, uint32_t count) {
    if (!logits) return false;
    for (uint32_t i = 0u; i < count; ++i) if (!std::isfinite(logits[i])) return false;
    return true;
}

int already_selected(const lm_moe_route *route, uint32_t id, uint32_t count) {
    for (uint32_t i = 0u; i < count; ++i) if (route->selected[i] == id) return 1;
    return 0;
}

} // namespace

lm_status lm_moe_map_mixtral_tensor(const lm_model_tensor_info *descriptor,
                                    uint32_t expected_experts,
                                    lm_moe_tensor_mapping *out_mapping) {
    if (!descriptor || !out_mapping || expected_experts == 0u) return LM_ERR_ARGUMENT;
    lm_moe_tensor_role role = LM_MOE_TENSOR_GATE_UP_EXPERT;
    uint32_t layer = 0u;
    const lm_status parsed = parse_mixtral_name(*descriptor, &role, &layer);
    if (parsed != LM_OK) return parsed;
    if (descriptor->rank != 3u || descriptor->dims[0] == 0u || descriptor->dims[1] == 0u ||
        descriptor->dims[2] != expected_experts) return LM_ERR_PARSE;
    if (role == LM_MOE_TENSOR_GATE_UP_EXPERT && (descriptor->dims[1] & 1u) != 0u) return LM_ERR_PARSE;
    std::memset(out_mapping, 0, sizeof(*out_mapping));
    out_mapping->role = role;
    out_mapping->layer_index = layer;
    out_mapping->expert_axis = 2u;
    out_mapping->expert_count = expected_experts;
    out_mapping->rank = descriptor->rank;
    for (uint32_t i = 0u; i < descriptor->rank; ++i) out_mapping->dims[i] = descriptor->dims[i];
    return LM_OK;
}

lm_status lm_cpu_moe_route(const float *router_logits, uint32_t expert_count,
                           uint32_t experts_per_token, lm_moe_route_policy policy,
                           lm_moe_route *out_route) {
    if (!router_logits || !out_route || expert_count == 0u || expert_count > 1u << 20u ||
        experts_per_token == 0u || experts_per_token > 16u || experts_per_token > expert_count ||
        (policy != LM_MOE_SOFTMAX_ALL_THEN_TOPK && policy != LM_MOE_SOFTMAX_SELECTED_ONLY))
        return LM_ERR_ARGUMENT;
    if (!finite_logits(router_logits, expert_count)) return LM_ERR_RANGE;
    std::memset(out_route, 0, sizeof(*out_route));
    out_route->expert_count = expert_count;
    out_route->experts_per_token = experts_per_token;
    for (uint32_t slot = 0u; slot < experts_per_token; ++slot) {
        int best = -1;
        for (uint32_t expert = 0u; expert < expert_count; ++expert) {
            if (already_selected(out_route, expert, slot)) continue;
            if (best < 0 || router_logits[expert] > router_logits[static_cast<uint32_t>(best)] ||
                (router_logits[expert] == router_logits[static_cast<uint32_t>(best)] && expert < static_cast<uint32_t>(best)))
                best = static_cast<int>(expert);
        }
        if (best < 0) return LM_ERR_STATE;
        out_route->selected[slot] = static_cast<uint32_t>(best);
    }

    float max_value = -std::numeric_limits<float>::infinity();
    if (policy == LM_MOE_SOFTMAX_ALL_THEN_TOPK) {
        for (uint32_t expert = 0u; expert < expert_count; ++expert) max_value = std::fmax(max_value, router_logits[expert]);
    } else {
        for (uint32_t slot = 0u; slot < experts_per_token; ++slot)
            max_value = std::fmax(max_value, router_logits[out_route->selected[slot]]);
    }
    float sum = 0.0f;
    for (uint32_t slot = 0u; slot < experts_per_token; ++slot) {
        out_route->weights[slot] = std::exp(router_logits[out_route->selected[slot]] - max_value);
        sum += out_route->weights[slot];
    }
    if (!(sum > 0.0f) || !std::isfinite(sum)) return LM_ERR_RANGE;
    for (uint32_t slot = 0u; slot < experts_per_token; ++slot) out_route->weights[slot] /= sum;
    return LM_OK;
}

lm_status lm_cpu_moe_combine(const lm_moe_route *route, const float *selected_outputs,
                              uint32_t hidden_size, float *out_hidden) {
    if (!route || !selected_outputs || !out_hidden || route->expert_count == 0u ||
        route->experts_per_token == 0u || route->experts_per_token > 16u || hidden_size == 0u)
        return LM_ERR_ARGUMENT;
    std::memset(out_hidden, 0, static_cast<size_t>(hidden_size) * sizeof(float));
    for (uint32_t slot = 0u; slot < route->experts_per_token; ++slot) {
        if (route->selected[slot] >= route->expert_count || !std::isfinite(route->weights[slot])) return LM_ERR_RANGE;
        const float *expert_output = selected_outputs + static_cast<size_t>(slot) * hidden_size;
        for (uint32_t i = 0u; i < hidden_size; ++i) {
            if (!std::isfinite(expert_output[i])) return LM_ERR_RANGE;
            out_hidden[i] += route->weights[slot] * expert_output[i];
        }
    }
    for (uint32_t i = 0u; i < hidden_size; ++i) if (!std::isfinite(out_hidden[i])) return LM_ERR_RANGE;
    return LM_OK;
}

lm_status lm_cpu_moe_selected_expert_mlp_q4_k(const lm_moe_route *route,
                                              const lm_tensor *gate_up_weights,
                                              const lm_tensor *down_weights,
                                              uint32_t hidden_size, uint32_t intermediate_size,
                                              const float *input, float *selected_outputs) {
    if (!route || !gate_up_weights || !down_weights || !input || !selected_outputs ||
        route->expert_count == 0u || route->experts_per_token == 0u || route->experts_per_token > 16u ||
        hidden_size == 0u || intermediate_size == 0u || hidden_size % 256u != 0u || intermediate_size % 256u != 0u)
        return LM_ERR_ARGUMENT;
    if (gate_up_weights->quant_format != LM_QUANT_GGML_Q4_K || down_weights->quant_format != LM_QUANT_GGML_Q4_K ||
        gate_up_weights->dtype != LM_DTYPE_U8 || down_weights->dtype != LM_DTYPE_U8)
        return LM_ERR_UNSUPPORTED;
    if (gate_up_weights->rank != 3u || down_weights->rank != 3u ||
        gate_up_weights->dims[0] != hidden_size || gate_up_weights->dims[1] != intermediate_size * 2u ||
        gate_up_weights->dims[2] != route->expert_count || down_weights->dims[0] != intermediate_size ||
        down_weights->dims[1] != hidden_size || down_weights->dims[2] != route->expert_count)
        return LM_ERR_RANGE;
    const uint64_t gate_up_row_bytes = (static_cast<uint64_t>(hidden_size) / 256u) * 144u;
    const uint64_t gate_up_expert_bytes = gate_up_row_bytes * static_cast<uint64_t>(intermediate_size) * 2u;
    const uint64_t down_row_bytes = (static_cast<uint64_t>(intermediate_size) / 256u) * 144u;
    const uint64_t down_expert_bytes = down_row_bytes * static_cast<uint64_t>(hidden_size);
    if (gate_up_expert_bytes > std::numeric_limits<uint64_t>::max() / route->expert_count ||
        down_expert_bytes > std::numeric_limits<uint64_t>::max() / route->expert_count ||
        gate_up_weights->bytes != gate_up_expert_bytes * route->expert_count ||
        down_weights->bytes != down_expert_bytes * route->expert_count)
        return LM_ERR_CAPACITY;
    std::vector<float> gate_up_output(static_cast<size_t>(intermediate_size) * 2u);
    const unsigned char *gate_up = static_cast<const unsigned char *>(gate_up_weights->data);
    const unsigned char *down = static_cast<const unsigned char *>(down_weights->data);
    for (uint32_t slot = 0u; slot < route->experts_per_token; ++slot) {
        if (route->selected[slot] >= route->expert_count || !std::isfinite(route->weights[slot])) return LM_ERR_RANGE;
        const uint32_t expert = route->selected[slot];
        const unsigned char *gate_up_expert = gate_up + static_cast<size_t>(expert * gate_up_expert_bytes);
        const unsigned char *down_expert = down + static_cast<size_t>(expert * down_expert_bytes);
        lm_tensor gate_up_view{};
        const uint32_t gate_up_dims[2] = {intermediate_size * 2u, hidden_size};
        lm_status status = lm_tensor_make_q4_k_view(const_cast<unsigned char *>(gate_up_expert), gate_up_expert_bytes,
                                                    2u, gate_up_dims, &gate_up_view);
        if (status != LM_OK) return status;
        status = lm_cpu_matvec_q4_k(&gate_up_view, input, intermediate_size * 2u, hidden_size, gate_up_output.data());
        if (status != LM_OK) return status;
        std::vector<float> activation(intermediate_size);
        for (uint32_t j = 0u; j < intermediate_size; ++j) {
            const float gate = gate_up_output[j];
            const float up = gate_up_output[intermediate_size + j];
            if (!std::isfinite(gate) || !std::isfinite(up)) return LM_ERR_RANGE;
            activation[j] = (gate / (1.0f + std::exp(-gate))) * up;
            if (!std::isfinite(activation[j])) return LM_ERR_RANGE;
        }
        lm_tensor down_view{};
        const uint32_t down_dims[2] = {hidden_size, intermediate_size};
        status = lm_tensor_make_q4_k_view(const_cast<unsigned char *>(down_expert), down_expert_bytes,
                                          2u, down_dims, &down_view);
        if (status != LM_OK) return status;
        status = lm_cpu_matvec_q4_k(&down_view, activation.data(), hidden_size, intermediate_size,
                                    selected_outputs + static_cast<size_t>(slot) * hidden_size);
        if (status != LM_OK) return status;
    }
    return LM_OK;
}

lm_status lm_cpu_moe_selected_expert_mlp(const lm_moe_route *route,
                                         const lm_tensor *gate_up_weights,
                                         const lm_tensor *down_weights,
                                         uint32_t hidden_size, uint32_t intermediate_size,
                                         const float *input, float *selected_outputs) {
    if (!route || !gate_up_weights || !down_weights || !input || !selected_outputs ||
        route->expert_count == 0u || route->experts_per_token == 0u || route->experts_per_token > 16u ||
        hidden_size == 0u || intermediate_size == 0u)
        return LM_ERR_ARGUMENT;
    if (gate_up_weights->dtype != LM_DTYPE_F32 || down_weights->dtype != LM_DTYPE_F32 ||
        gate_up_weights->quant_format != LM_QUANT_NONE || down_weights->quant_format != LM_QUANT_NONE)
        return LM_ERR_UNSUPPORTED;
    if (intermediate_size > UINT32_MAX / 2u || intermediate_size > (1u << 20u)) return LM_ERR_CAPACITY;
    if (gate_up_weights->rank != 3u || down_weights->rank != 3u ||
        gate_up_weights->dims[0] != hidden_size || gate_up_weights->dims[1] != intermediate_size * 2u ||
        gate_up_weights->dims[2] != route->expert_count || down_weights->dims[0] != intermediate_size ||
        down_weights->dims[1] != hidden_size || down_weights->dims[2] != route->expert_count ||
        lm_tensor_validate(gate_up_weights) != LM_OK || lm_tensor_validate(down_weights) != LM_OK)
        return LM_ERR_RANGE;
    const float *gate_up = static_cast<const float *>(gate_up_weights->data);
    const float *down = static_cast<const float *>(down_weights->data);
    const size_t gate_up_expert_stride = static_cast<size_t>(hidden_size) * intermediate_size * 2u;
    const size_t down_expert_stride = static_cast<size_t>(intermediate_size) * hidden_size;
    float *scratch = new (std::nothrow) float[intermediate_size];
    if (!scratch) return LM_ERR_CAPACITY;
    for (uint32_t slot = 0u; slot < route->experts_per_token; ++slot) {
        if (route->selected[slot] >= route->expert_count) { delete[] scratch; return LM_ERR_RANGE; }
        const uint32_t expert = route->selected[slot];
        const float *gate_up_expert = gate_up + static_cast<size_t>(expert) * gate_up_expert_stride;
        const float *down_expert = down + static_cast<size_t>(expert) * down_expert_stride;
        for (uint32_t j = 0u; j < intermediate_size; ++j) {
            float gate = 0.0f;
            float up = 0.0f;
            for (uint32_t i = 0u; i < hidden_size; ++i) {
                const float value = input[i];
                const float gate_weight = gate_up_expert[static_cast<size_t>(j) * hidden_size + i];
                const float up_weight = gate_up_expert[static_cast<size_t>(intermediate_size + j) * hidden_size + i];
                if (!std::isfinite(value) || !std::isfinite(gate_weight) || !std::isfinite(up_weight)) { delete[] scratch; return LM_ERR_RANGE; }
                gate += gate_weight * value;
                up += up_weight * value;
            }
            scratch[j] = (gate / (1.0f + std::exp(-gate))) * up;
            if (!std::isfinite(scratch[j])) { delete[] scratch; return LM_ERR_RANGE; }
        }
        float *output = selected_outputs + static_cast<size_t>(slot) * hidden_size;
        for (uint32_t h = 0u; h < hidden_size; ++h) {
            float sum = 0.0f;
            for (uint32_t j = 0u; j < intermediate_size; ++j) {
                const float weight = down_expert[static_cast<size_t>(h) * intermediate_size + j];
                if (!std::isfinite(weight)) { delete[] scratch; return LM_ERR_RANGE; }
                sum += weight * scratch[j];
            }
            if (!std::isfinite(sum)) { delete[] scratch; return LM_ERR_RANGE; }
            output[h] = sum;
        }
    }
    delete[] scratch;
    return LM_OK;
}
