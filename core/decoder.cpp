#include "lm/lm.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <cstdio>
#include <vector>

struct lm_cpu_decoder {
    lm_cpu_decoder_config config;
    uint32_t position;
    std::vector<float> keys;
    std::vector<float> values;
    std::vector<float> scratch;
};

namespace {

bool finite_array(const float *data, size_t count) {
    if (!data) return false;
    for (size_t i = 0u; i < count; ++i) if (!std::isfinite(data[i])) return false;
    return true;
}

bool matrix_finite(const float *data, uint32_t rows, uint32_t columns) {
    return finite_array(data, static_cast<size_t>(rows) * columns);
}

void matvec(const float *input, const float *matrix, uint32_t rows, uint32_t columns, float *output) {
    for (uint32_t column = 0u; column < columns; ++column) {
        float sum = 0.0f;
        for (uint32_t row = 0u; row < rows; ++row) sum += input[row] * matrix[static_cast<size_t>(row) * columns + column];
        output[column] = sum;
    }
}

bool rms_norm(const float *input, uint32_t size, float epsilon, const float *gamma, float *output) {
    if (!(epsilon > 0.0f) || !std::isfinite(epsilon)) return false;
    float mean_square = 0.0f;
    for (uint32_t i = 0u; i < size; ++i) mean_square += input[i] * input[i];
    mean_square /= static_cast<float>(size);
    const float scale = 1.0f / std::sqrt(mean_square + epsilon);
    if (!std::isfinite(scale)) return false;
    for (uint32_t i = 0u; i < size; ++i) output[i] = input[i] * scale * gamma[i];
    return finite_array(output, size);
}

void apply_rope(float *vector, uint32_t size, uint32_t position, float theta) {
    for (uint32_t i = 0u; i + 1u < size; i += 2u) {
        const float frequency = std::pow(theta, -static_cast<float>(i) / static_cast<float>(size));
        const float angle = static_cast<float>(position) * frequency;
        const float cosine = std::cos(angle);
        const float sine = std::sin(angle);
        const float first = vector[i];
        const float second = vector[i + 1u];
        vector[i] = first * cosine - second * sine;
        vector[i + 1u] = first * sine + second * cosine;
    }
}

} // namespace

lm_status lm_decoder_map_llama_tensor(const lm_model_tensor_info *descriptor,
                                      lm_decoder_tensor_mapping *out_mapping) {
    if (!descriptor || !out_mapping || descriptor->name[0] == '\0' || descriptor->rank == 0u || descriptor->rank > 8u)
        return LM_ERR_ARGUMENT;
    std::memset(out_mapping, 0, sizeof(*out_mapping));
    out_mapping->rank = descriptor->rank;
    out_mapping->type = descriptor->type;
    for (uint32_t i = 0u; i < descriptor->rank; ++i) out_mapping->dims[i] = descriptor->dims[i];
    if (std::strcmp(descriptor->name, "token_embd.weight") == 0)
        out_mapping->role = LM_DECODER_TENSOR_TOKEN_EMBEDDING;
    else if (std::strcmp(descriptor->name, "output.weight") == 0)
        out_mapping->role = LM_DECODER_TENSOR_OUTPUT;
    else if (std::strcmp(descriptor->name, "output_norm.weight") == 0)
        out_mapping->role = LM_DECODER_TENSOR_OUTPUT_NORM;
    else {
        unsigned layer = 0u;
        char suffix[64] = {};
        if (std::sscanf(descriptor->name, "blk.%u.%63s", &layer, suffix) != 2 || layer > UINT32_MAX)
            return LM_ERR_UNSUPPORTED;
        out_mapping->layer_index = static_cast<uint32_t>(layer);
        if (std::strcmp(suffix, "attn_norm.weight") == 0) out_mapping->role = LM_DECODER_TENSOR_ATTN_NORM;
        else if (std::strcmp(suffix, "attn_q.weight") == 0) out_mapping->role = LM_DECODER_TENSOR_ATTN_Q;
        else if (std::strcmp(suffix, "attn_k.weight") == 0) out_mapping->role = LM_DECODER_TENSOR_ATTN_K;
        else if (std::strcmp(suffix, "attn_v.weight") == 0) out_mapping->role = LM_DECODER_TENSOR_ATTN_V;
        else if (std::strcmp(suffix, "attn_output.weight") == 0) out_mapping->role = LM_DECODER_TENSOR_ATTN_OUTPUT;
        else if (std::strcmp(suffix, "ffn_norm.weight") == 0) out_mapping->role = LM_DECODER_TENSOR_FFN_NORM;
        else if (std::strcmp(suffix, "ffn_gate.weight") == 0) out_mapping->role = LM_DECODER_TENSOR_FFN_GATE;
        else if (std::strcmp(suffix, "ffn_down.weight") == 0) out_mapping->role = LM_DECODER_TENSOR_FFN_DOWN;
        else if (std::strcmp(suffix, "ffn_up.weight") == 0) out_mapping->role = LM_DECODER_TENSOR_FFN_UP;
        else return LM_ERR_UNSUPPORTED;
    }
    return LM_OK;
}

lm_status lm_decoder_graph_plan_build(const lm_model_tensor_info *descriptors,
                                      uint64_t descriptor_count,
                                      lm_decoder_graph_plan *out_plan) {
    if (!descriptors || !out_plan || descriptor_count == 0u ||
        descriptor_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
        return LM_ERR_ARGUMENT;
    std::memset(out_plan, 0, sizeof(*out_plan));
    const uint32_t global_required = (1u << LM_DECODER_TENSOR_TOKEN_EMBEDDING) |
                                     (1u << LM_DECODER_TENSOR_OUTPUT) |
                                     (1u << LM_DECODER_TENSOR_OUTPUT_NORM);
    const uint32_t layer_required = ((1u << (LM_DECODER_TENSOR_FFN_UP + 1u)) - 1u) & ~global_required;
    for (uint64_t i = 0u; i < descriptor_count; ++i) {
        lm_decoder_tensor_mapping mapping{};
        const lm_status status = lm_decoder_map_llama_tensor(&descriptors[static_cast<size_t>(i)], &mapping);
        if (status != LM_OK) return status;
        if (mapping.role <= LM_DECODER_TENSOR_OUTPUT_NORM) {
            const uint32_t bit = 1u << mapping.role;
            if ((out_plan->global_role_mask & bit) != 0u) return LM_ERR_PARSE;
            out_plan->global_role_mask |= bit;
        } else {
            if (mapping.layer_index >= LM_DECODER_PLAN_MAX_LAYERS) return LM_ERR_CAPACITY;
            const uint32_t bit = 1u << mapping.role;
            if ((out_plan->layer_role_mask[mapping.layer_index] & bit) != 0u) return LM_ERR_PARSE;
            out_plan->layer_role_mask[mapping.layer_index] |= bit;
            if (mapping.layer_index + 1u > out_plan->layer_count) out_plan->layer_count = mapping.layer_index + 1u;
        }
    }
    if (out_plan->global_role_mask != global_required || out_plan->layer_count == 0u) return LM_ERR_UNSUPPORTED;
    for (uint32_t layer = 0u; layer < out_plan->layer_count; ++layer)
        if (out_plan->layer_role_mask[layer] != layer_required) return LM_ERR_UNSUPPORTED;
    return LM_OK;
}

namespace {

constexpr uint32_t kNativeProfileMaxVocab = 65536u;
constexpr uint32_t kNativeProfileMaxHidden = 4096u;
constexpr uint32_t kNativeProfileMaxIntermediate = 16384u;
constexpr uint32_t kMaxNativeGeneratedTokens = 4096u;

lm_status profile_matrix(const lm_model_file *model, uint64_t index,
                        uint32_t rows, uint32_t columns, lm_quant_format format,
                        uint64_t *payload_bytes) {
    if (!model || !payload_bytes || rows == 0u || columns == 0u) return LM_ERR_ARGUMENT;
    lm_model_tensor_binding binding{};
    const lm_status bound = lm_model_tensor_bind_native(model, index, &binding);
    if (bound != LM_OK) return bound;
    if (binding.quant_format != format || binding.descriptor.rank != 2u ||
        binding.descriptor.dims[0] != rows || binding.descriptor.dims[1] != columns)
        return LM_ERR_UNSUPPORTED;
    *payload_bytes = binding.span.bytes;
    return LM_OK;
}

lm_status profile_vector(const lm_model_file *model, uint64_t index, uint32_t elements,
                        std::vector<float> *out) {
    if (!model || !out || elements == 0u) return LM_ERR_ARGUMENT;
    lm_model_tensor_info descriptor{};
    const lm_status described = lm_model_tensor_info_at(model, index, &descriptor);
    if (described != LM_OK) return described;
    if (descriptor.rank != 1u || descriptor.dims[0] != elements || descriptor.type != LM_DTYPE_F32)
        return LM_ERR_UNSUPPORTED;
    const uint64_t bytes = static_cast<uint64_t>(elements) * sizeof(float);
    lm_file_span span{};
    const lm_status spanned = lm_model_tensor_span(model, descriptor.relative_offset, bytes, &span);
    if (spanned != LM_OK) return spanned;
    try {
        out->assign(elements, 0.0f);
    } catch (const std::bad_alloc &) {
        return LM_ERR_CAPACITY;
    }
    const lm_status read = lm_file_span_read(&span, 0u, out->data(), static_cast<size_t>(bytes));
    if (read != LM_OK) return read;
    return finite_array(out->data(), out->size()) ? LM_OK : LM_ERR_RANGE;
}

lm_status profile_matvec(const lm_model_file *model, uint64_t index,
                         const lm_native_matvec_config *config, void *packed_scratch,
                         uint64_t packed_scratch_bytes, uint32_t rows, uint32_t columns,
                         const float *input, float *out) {
    return lm_model_tensor_matvec_native(model, index, config, packed_scratch,
                                         packed_scratch_bytes, rows, columns, input, out);
}

lm_status profile_mlp_hidden(const lm_model_file *model, const lm_decoder_graph_binding *graph,
                            const lm_native_mlp_config *config, const float *input,
                            void *packed_scratch, uint64_t packed_scratch_bytes, float *out) {
    if (!model || !graph || !config || !input || !packed_scratch || !out ||
        config->layer_index >= graph->layer_count || !finite_array(input, config->hidden_size))
        return LM_ERR_ARGUMENT;
    const lm_decoder_layer_binding &layer = graph->layers[config->layer_index];
    const uint64_t matrix_indices[] = {layer.ffn_gate, layer.ffn_up, layer.ffn_down};
    const uint32_t matrix_rows[] = {config->intermediate_size, config->intermediate_size, config->hidden_size};
    const uint32_t matrix_columns[] = {config->hidden_size, config->hidden_size, config->intermediate_size};
    for (uint32_t i = 0u; i < 3u; ++i) {
        uint64_t payload = 0u;
        const lm_status valid = profile_matrix(model, matrix_indices[i], matrix_rows[i],
                                               matrix_columns[i], config->matrix_format, &payload);
        if (valid != LM_OK) return valid;
        if (packed_scratch_bytes < payload) return LM_ERR_CAPACITY;
    }
    std::vector<float> gamma, normed, gate, up, activated, down;
    try {
        normed.assign(config->hidden_size, 0.0f);
        gate.assign(config->intermediate_size, 0.0f);
        up.assign(config->intermediate_size, 0.0f);
        activated.assign(config->intermediate_size, 0.0f);
        down.assign(config->hidden_size, 0.0f);
    } catch (const std::bad_alloc &) {
        return LM_ERR_CAPACITY;
    }
    lm_status status = profile_vector(model, layer.ffn_norm, config->hidden_size, &gamma);
    if (status != LM_OK) return status;
    float mean_square = 0.0f;
    for (uint32_t i = 0u; i < config->hidden_size; ++i) mean_square += input[i] * input[i];
    mean_square /= static_cast<float>(config->hidden_size);
    const float scale = 1.0f / std::sqrt(mean_square + config->rms_epsilon);
    if (!std::isfinite(scale)) return LM_ERR_RANGE;
    for (uint32_t i = 0u; i < config->hidden_size; ++i) normed[i] = input[i] * scale * gamma[i];
    if (!finite_array(normed.data(), normed.size())) return LM_ERR_RANGE;
    status = profile_matvec(model, layer.ffn_gate, &config->matvec, packed_scratch, packed_scratch_bytes,
                            config->intermediate_size, config->hidden_size, normed.data(), gate.data());
    if (status != LM_OK) return status;
    status = profile_matvec(model, layer.ffn_up, &config->matvec, packed_scratch, packed_scratch_bytes,
                            config->intermediate_size, config->hidden_size, normed.data(), up.data());
    if (status != LM_OK) return status;
    for (uint32_t i = 0u; i < config->intermediate_size; ++i) {
        const float sigmoid = 1.0f / (1.0f + std::exp(-gate[i]));
        activated[i] = gate[i] * sigmoid * up[i];
    }
    if (!finite_array(activated.data(), activated.size())) return LM_ERR_RANGE;
    status = profile_matvec(model, layer.ffn_down, &config->matvec, packed_scratch, packed_scratch_bytes,
                            config->hidden_size, config->intermediate_size, activated.data(), down.data());
    if (status != LM_OK) return status;
    for (uint32_t i = 0u; i < config->hidden_size; ++i) out[i] = input[i] + down[i];
    return finite_array(out, config->hidden_size) ? LM_OK : LM_ERR_RANGE;
}

} // namespace

lm_status lm_model_execute_native_mlp_logits(const lm_model_file *model,
                                             const lm_decoder_graph_binding *graph,
                                             const lm_native_mlp_config *config,
                                             uint32_t token_id,
                                             void *packed_scratch,
                                             uint64_t packed_scratch_bytes,
                                             float *out_logits,
                                             size_t logits_count) {
    if (!model || !graph || !config || !packed_scratch || !out_logits ||
        config->layer_index >= graph->layer_count || config->vocab_size == 0u ||
        config->hidden_size == 0u || config->intermediate_size == 0u ||
        config->vocab_size > kNativeProfileMaxVocab || config->hidden_size > kNativeProfileMaxHidden ||
        config->intermediate_size > kNativeProfileMaxIntermediate || token_id >= config->vocab_size ||
        logits_count < config->vocab_size || !(config->rms_epsilon > 0.0f) ||
        !std::isfinite(config->rms_epsilon) ||
        (config->matrix_format != LM_QUANT_GGML_Q8_0 && config->matrix_format != LM_QUANT_GGML_Q4_K))
        return LM_ERR_ARGUMENT;
    const lm_decoder_layer_binding &layer = graph->layers[config->layer_index];
    uint64_t max_payload = 0u;
    const uint64_t matrix_indices[] = {
        graph->token_embedding, graph->output, layer.ffn_gate, layer.ffn_up, layer.ffn_down};
    const uint32_t matrix_rows[] = {
        config->hidden_size, config->vocab_size, config->intermediate_size,
        config->intermediate_size, config->hidden_size};
    const uint32_t matrix_columns[] = {
        config->vocab_size, config->hidden_size, config->hidden_size,
        config->hidden_size, config->intermediate_size};
    for (uint32_t i = 0u; i < 5u; ++i) {
        uint64_t payload = 0u;
        const lm_status valid = profile_matrix(model, matrix_indices[i], matrix_rows[i],
                                               matrix_columns[i], config->matrix_format, &payload);
        if (valid != LM_OK) return valid;
        if (payload > max_payload) max_payload = payload;
    }
    if (packed_scratch_bytes < max_payload) return LM_ERR_CAPACITY;
    std::vector<float> embedding, ffn_norm, gate, up, activated, down, output_norm, one_hot, normed, residual;
    try {
        one_hot.assign(config->vocab_size, 0.0f);
        one_hot[token_id] = 1.0f;
        embedding.assign(config->hidden_size, 0.0f);
        gate.assign(config->intermediate_size, 0.0f);
        up.assign(config->intermediate_size, 0.0f);
        activated.assign(config->intermediate_size, 0.0f);
        down.assign(config->hidden_size, 0.0f);
        normed.assign(config->hidden_size, 0.0f);
        residual.assign(config->hidden_size, 0.0f);
    } catch (const std::bad_alloc &) {
        return LM_ERR_CAPACITY;
    }
    lm_status status = profile_matvec(model, graph->token_embedding, &config->matvec, packed_scratch,
                                      packed_scratch_bytes, config->hidden_size, config->vocab_size,
                                      one_hot.data(), embedding.data());
    if (status != LM_OK) return status;
    status = profile_vector(model, layer.ffn_norm, config->hidden_size, &ffn_norm);
    if (status != LM_OK) return status;
    status = profile_vector(model, graph->output_norm, config->hidden_size, &output_norm);
    if (status != LM_OK) return status;
    float mean_square = 0.0f;
    for (uint32_t i = 0u; i < config->hidden_size; ++i) mean_square += embedding[i] * embedding[i];
    mean_square /= static_cast<float>(config->hidden_size);
    const float scale = 1.0f / std::sqrt(mean_square + config->rms_epsilon);
    if (!std::isfinite(scale)) return LM_ERR_RANGE;
    for (uint32_t i = 0u; i < config->hidden_size; ++i) normed[i] = embedding[i] * scale * ffn_norm[i];
    if (!finite_array(normed.data(), normed.size())) return LM_ERR_RANGE;
    status = profile_matvec(model, layer.ffn_gate, &config->matvec, packed_scratch, packed_scratch_bytes,
                            config->intermediate_size, config->hidden_size, normed.data(), gate.data());
    if (status != LM_OK) return status;
    status = profile_matvec(model, layer.ffn_up, &config->matvec, packed_scratch, packed_scratch_bytes,
                            config->intermediate_size, config->hidden_size, normed.data(), up.data());
    if (status != LM_OK) return status;
    for (uint32_t i = 0u; i < config->intermediate_size; ++i) {
        const float sigmoid = 1.0f / (1.0f + std::exp(-gate[i]));
        activated[i] = gate[i] * sigmoid * up[i];
    }
    if (!finite_array(activated.data(), activated.size())) return LM_ERR_RANGE;
    status = profile_matvec(model, layer.ffn_down, &config->matvec, packed_scratch, packed_scratch_bytes,
                            config->hidden_size, config->intermediate_size, activated.data(), down.data());
    if (status != LM_OK) return status;
    for (uint32_t i = 0u; i < config->hidden_size; ++i) residual[i] = embedding[i] + down[i];
    if (!finite_array(residual.data(), residual.size())) return LM_ERR_RANGE;
    mean_square = 0.0f;
    for (uint32_t i = 0u; i < config->hidden_size; ++i) mean_square += residual[i] * residual[i];
    mean_square /= static_cast<float>(config->hidden_size);
    const float output_scale = 1.0f / std::sqrt(mean_square + config->rms_epsilon);
    if (!std::isfinite(output_scale)) return LM_ERR_RANGE;
    for (uint32_t i = 0u; i < config->hidden_size; ++i) normed[i] = residual[i] * output_scale * output_norm[i];
    if (!finite_array(normed.data(), normed.size())) return LM_ERR_RANGE;
    status = profile_matvec(model, graph->output, &config->matvec, packed_scratch, packed_scratch_bytes,
                            config->vocab_size, config->hidden_size, normed.data(), out_logits);
    if (status != LM_OK) return status;
    return finite_array(out_logits, config->vocab_size) ? LM_OK : LM_ERR_RANGE;
}

lm_status lm_model_execute_native_attention(const lm_model_file *model,
                                            const lm_decoder_graph_binding *graph,
                                            const lm_native_attention_config *config,
                                            const float *input,
                                            lm_kv_cache *kv_cache,
                                            uint32_t page_id,
                                            void *packed_scratch,
                                            uint64_t packed_scratch_bytes,
                                            float *out_attention) {
    if (!model || !graph || !config || !input || !kv_cache || !packed_scratch || !out_attention ||
        config->layer_index >= graph->layer_count || config->hidden_size == 0u ||
        config->hidden_size > kNativeProfileMaxHidden || config->token_offset >= (1u << 20u) ||
        (config->matrix_format != LM_QUANT_GGML_Q8_0 && config->matrix_format != LM_QUANT_GGML_Q4_K) ||
        (config->use_rope && (!(config->rope_theta > 1.0f) || !std::isfinite(config->rope_theta))) ||
        !(config->rms_epsilon > 0.0f) || !std::isfinite(config->rms_epsilon) ||
        !finite_array(input, config->hidden_size))
        return LM_ERR_ARGUMENT;
    uint32_t key_bytes = 0u;
    uint32_t value_bytes = 0u;
    lm_status status = lm_kv_cache_get_payload_layout(kv_cache, &key_bytes, &value_bytes);
    if (status != LM_OK) return status;
    if (key_bytes != config->hidden_size * sizeof(float) || value_bytes != config->hidden_size * sizeof(float))
        return LM_ERR_UNSUPPORTED;
    const lm_decoder_layer_binding &layer = graph->layers[config->layer_index];
    const uint64_t matrix_indices[] = {layer.attn_q, layer.attn_k, layer.attn_v, layer.attn_output};
    uint64_t max_payload = 0u;
    for (const uint64_t index : matrix_indices) {
        uint64_t payload = 0u;
        status = profile_matrix(model, index, config->hidden_size, config->hidden_size,
                                config->matrix_format, &payload);
        if (status != LM_OK) return status;
        if (payload > max_payload) max_payload = payload;
    }
    if (packed_scratch_bytes < max_payload) return LM_ERR_CAPACITY;
    std::vector<float> norm, q, k, v, projected, keys, values, scores, weights, attended;
    try {
        norm.assign(config->hidden_size, 0.0f);
        q.assign(config->hidden_size, 0.0f);
        k.assign(config->hidden_size, 0.0f);
        v.assign(config->hidden_size, 0.0f);
        projected.assign(config->hidden_size, 0.0f);
        attended.assign(config->hidden_size, 0.0f);
        const size_t context = static_cast<size_t>(config->token_offset) + 1u;
        keys.assign(context * config->hidden_size, 0.0f);
        values.assign(context * config->hidden_size, 0.0f);
        scores.assign(context, 0.0f);
        weights.assign(context, 0.0f);
    } catch (const std::bad_alloc &) {
        return LM_ERR_CAPACITY;
    }
    std::vector<float> gamma;
    status = profile_vector(model, layer.attn_norm, config->hidden_size, &gamma);
    if (status != LM_OK) return status;
    float mean_square = 0.0f;
    for (uint32_t i = 0u; i < config->hidden_size; ++i) mean_square += input[i] * input[i];
    mean_square /= static_cast<float>(config->hidden_size);
    const float norm_scale = 1.0f / std::sqrt(mean_square + config->rms_epsilon);
    if (!std::isfinite(norm_scale)) return LM_ERR_RANGE;
    for (uint32_t i = 0u; i < config->hidden_size; ++i) norm[i] = input[i] * norm_scale * gamma[i];
    if (!finite_array(norm.data(), norm.size())) return LM_ERR_RANGE;
    status = profile_matvec(model, layer.attn_q, &config->matvec,
                            packed_scratch, packed_scratch_bytes, config->hidden_size,
                            config->hidden_size, norm.data(), q.data());
    if (status != LM_OK) return status;
    status = profile_matvec(model, layer.attn_k, &config->matvec,
                            packed_scratch, packed_scratch_bytes, config->hidden_size,
                            config->hidden_size, norm.data(), k.data());
    if (status != LM_OK) return status;
    status = profile_matvec(model, layer.attn_v, &config->matvec,
                            packed_scratch, packed_scratch_bytes, config->hidden_size,
                            config->hidden_size, norm.data(), v.data());
    if (status != LM_OK) return status;
    if (config->use_rope) {
        apply_rope(q.data(), config->hidden_size, config->token_offset, config->rope_theta);
        apply_rope(k.data(), config->hidden_size, config->token_offset, config->rope_theta);
    }
    status = lm_kv_cache_write_payload(kv_cache, page_id, config->token_offset, 1u, k.data(), v.data());
    if (status != LM_OK) return status;
    status = lm_kv_cache_read_payload(kv_cache, page_id, 0u, config->token_offset + 1u,
                                      keys.data(), values.data());
    if (status != LM_OK) return status;
    const float attention_scale = 1.0f / std::sqrt(static_cast<float>(config->hidden_size));
    for (uint32_t t = 0u; t <= config->token_offset; ++t) {
        const float *key = keys.data() + static_cast<size_t>(t) * config->hidden_size;
        float dot = 0.0f;
        for (uint32_t i = 0u; i < config->hidden_size; ++i) dot += q[i] * key[i];
        scores[t] = dot * attention_scale;
    }
    status = lm_cpu_softmax_f32(scores.data(), weights.data(), config->token_offset + 1u);
    if (status != LM_OK) return status;
    for (uint32_t t = 0u; t <= config->token_offset; ++t) {
        const float *value = values.data() + static_cast<size_t>(t) * config->hidden_size;
        for (uint32_t i = 0u; i < config->hidden_size; ++i) attended[i] += weights[t] * value[i];
    }
    status = profile_matvec(model, layer.attn_output, &config->matvec,
                            packed_scratch, packed_scratch_bytes, config->hidden_size,
                            config->hidden_size, attended.data(), projected.data());
    if (status != LM_OK) return status;
    for (uint32_t i = 0u; i < config->hidden_size; ++i) out_attention[i] = input[i] + projected[i];
    return finite_array(out_attention, config->hidden_size) ? LM_OK : LM_ERR_RANGE;
}

lm_status lm_model_execute_native_step(const lm_model_file *model,
                                       const lm_decoder_graph_binding *graph,
                                       const lm_native_step_config *config,
                                       uint32_t token_id,
                                       lm_kv_cache *kv_cache,
                                       uint32_t page_id,
                                       void *packed_scratch,
                                       uint64_t packed_scratch_bytes,
                                       float *out_logits,
                                       size_t logits_count) {
    if (!model || !graph || !config || !kv_cache || !packed_scratch || !out_logits ||
        config->layer_index >= graph->layer_count || config->vocab_size == 0u ||
        config->hidden_size == 0u || config->intermediate_size == 0u || token_id >= config->vocab_size ||
        logits_count < config->vocab_size || config->vocab_size > kNativeProfileMaxVocab ||
        config->hidden_size > kNativeProfileMaxHidden || config->intermediate_size > kNativeProfileMaxIntermediate ||
        !(config->rms_epsilon > 0.0f) || !std::isfinite(config->rms_epsilon) ||
        (config->matrix_format != LM_QUANT_GGML_Q8_0 && config->matrix_format != LM_QUANT_GGML_Q4_K))
        return LM_ERR_ARGUMENT;
    lm_native_attention_config attention{};
    attention.matvec = config->matvec;
    attention.layer_index = config->layer_index;
    attention.hidden_size = config->hidden_size;
    attention.token_offset = config->token_offset;
    attention.use_rope = config->use_rope;
    attention.rope_theta = config->rope_theta;
    attention.rms_epsilon = config->rms_epsilon;
    attention.matrix_format = config->matrix_format;
    uint64_t embedding_payload = 0u;
    lm_status status = profile_matrix(model, graph->token_embedding, config->hidden_size,
                                      config->vocab_size, config->matrix_format, &embedding_payload);
    if (status != LM_OK) return status;
    if (packed_scratch_bytes < embedding_payload) return LM_ERR_CAPACITY;
    std::vector<float> one_hot, embedding, attended, normed, output_norm;
    try {
        one_hot.assign(config->vocab_size, 0.0f);
        one_hot[token_id] = 1.0f;
        embedding.assign(config->hidden_size, 0.0f);
        attended.assign(config->hidden_size, 0.0f);
        normed.assign(config->hidden_size, 0.0f);
    } catch (const std::bad_alloc &) {
        return LM_ERR_CAPACITY;
    }
    status = profile_matvec(model, graph->token_embedding, &config->matvec, packed_scratch,
                            packed_scratch_bytes, config->hidden_size, config->vocab_size,
                            one_hot.data(), embedding.data());
    if (status != LM_OK) return status;
    status = lm_model_execute_native_attention(model, graph, &attention, embedding.data(), kv_cache,
                                               page_id, packed_scratch, packed_scratch_bytes,
                                               attended.data());
    if (status != LM_OK) return status;
    std::vector<float> mlp_output;
    try {
        mlp_output.assign(config->hidden_size, 0.0f);
    } catch (const std::bad_alloc &) {
        return LM_ERR_CAPACITY;
    }
    lm_native_mlp_config mlp{};
    mlp.matvec = config->matvec;
    mlp.layer_index = config->layer_index;
    mlp.vocab_size = config->vocab_size;
    mlp.hidden_size = config->hidden_size;
    mlp.intermediate_size = config->intermediate_size;
    mlp.matrix_format = config->matrix_format;
    mlp.rms_epsilon = config->rms_epsilon;
    status = profile_mlp_hidden(model, graph, &mlp, attended.data(), packed_scratch,
                                packed_scratch_bytes, mlp_output.data());
    if (status != LM_OK) return status;
    status = profile_vector(model, graph->output_norm, config->hidden_size, &output_norm);
    if (status != LM_OK) return status;
    float mean_square = 0.0f;
    for (uint32_t i = 0u; i < config->hidden_size; ++i) mean_square += mlp_output[i] * mlp_output[i];
    mean_square /= static_cast<float>(config->hidden_size);
    const float output_scale = 1.0f / std::sqrt(mean_square + config->rms_epsilon);
    if (!std::isfinite(output_scale)) return LM_ERR_RANGE;
    for (uint32_t i = 0u; i < config->hidden_size; ++i) normed[i] = mlp_output[i] * output_scale * output_norm[i];
    if (!finite_array(normed.data(), normed.size())) return LM_ERR_RANGE;
    status = profile_matvec(model, graph->output, &config->matvec, packed_scratch, packed_scratch_bytes,
                            config->vocab_size, config->hidden_size, normed.data(), out_logits);
    if (status != LM_OK) return status;
    return finite_array(out_logits, config->vocab_size) ? LM_OK : LM_ERR_RANGE;
}

lm_status lm_model_execute_native_transformer(const lm_model_file *model,
                                              const lm_decoder_graph_binding *graph,
                                              const lm_native_transformer_config *config,
                                              uint32_t token_id,
                                              lm_kv_cache *const *layer_caches,
                                              uint32_t cache_count,
                                              uint32_t page_id,
                                              void *packed_scratch,
                                              uint64_t packed_scratch_bytes,
                                              float *out_logits,
                                              size_t logits_count) {
    if (!model || !graph || !config || !layer_caches || !packed_scratch || !out_logits ||
        graph->layer_count == 0u || graph->layer_count > LM_DECODER_PLAN_MAX_LAYERS ||
        cache_count != graph->layer_count || config->step.layer_index != 0u ||
        config->step.vocab_size == 0u || config->step.hidden_size == 0u ||
        config->step.intermediate_size == 0u || token_id >= config->step.vocab_size ||
        logits_count < config->step.vocab_size || config->step.vocab_size > kNativeProfileMaxVocab ||
        config->step.hidden_size > kNativeProfileMaxHidden ||
        config->step.intermediate_size > kNativeProfileMaxIntermediate ||
        !(config->step.rms_epsilon > 0.0f) || !std::isfinite(config->step.rms_epsilon) ||
        (config->step.matrix_format != LM_QUANT_GGML_Q8_0 &&
         config->step.matrix_format != LM_QUANT_GGML_Q4_K))
        return LM_ERR_ARGUMENT;
    for (uint32_t layer = 0u; layer < graph->layer_count; ++layer)
        if (!layer_caches[layer]) return LM_ERR_ARGUMENT;
    uint64_t payload = 0u;
    lm_status status = profile_matrix(model, graph->token_embedding, config->step.hidden_size,
                                      config->step.vocab_size, config->step.matrix_format, &payload);
    if (status != LM_OK) return status;
    status = profile_matrix(model, graph->output, config->step.vocab_size,
                            config->step.hidden_size, config->step.matrix_format, &payload);
    if (status != LM_OK) return status;
    std::vector<float> one_hot, hidden, next, attended, normed, output_norm;
    try {
        one_hot.assign(config->step.vocab_size, 0.0f);
        one_hot[token_id] = 1.0f;
        hidden.assign(config->step.hidden_size, 0.0f);
        next.assign(config->step.hidden_size, 0.0f);
        attended.assign(config->step.hidden_size, 0.0f);
        normed.assign(config->step.hidden_size, 0.0f);
    } catch (const std::bad_alloc &) {
        return LM_ERR_CAPACITY;
    }
    status = profile_matvec(model, graph->token_embedding, &config->step.matvec,
                            packed_scratch, packed_scratch_bytes, config->step.hidden_size,
                            config->step.vocab_size, one_hot.data(), hidden.data());
    if (status != LM_OK) return status;
    for (uint32_t layer_index = 0u; layer_index < graph->layer_count; ++layer_index) {
        lm_native_attention_config attention{};
        attention.matvec = config->step.matvec;
        attention.layer_index = layer_index;
        attention.hidden_size = config->step.hidden_size;
        attention.token_offset = config->step.token_offset;
        attention.use_rope = config->step.use_rope;
        attention.rope_theta = config->step.rope_theta;
        attention.rms_epsilon = config->step.rms_epsilon;
        attention.matrix_format = config->step.matrix_format;
        status = lm_model_execute_native_attention(model, graph, &attention, hidden.data(),
                                                   layer_caches[layer_index], page_id,
                                                   packed_scratch, packed_scratch_bytes,
                                                   attended.data());
        if (status != LM_OK) return status;
        lm_native_mlp_config mlp{};
        mlp.matvec = config->step.matvec;
        mlp.layer_index = layer_index;
        mlp.vocab_size = config->step.vocab_size;
        mlp.hidden_size = config->step.hidden_size;
        mlp.intermediate_size = config->step.intermediate_size;
        mlp.matrix_format = config->step.matrix_format;
        mlp.rms_epsilon = config->step.rms_epsilon;
        status = profile_mlp_hidden(model, graph, &mlp, attended.data(), packed_scratch,
                                    packed_scratch_bytes, next.data());
        if (status != LM_OK) return status;
        hidden.swap(next);
    }
    status = profile_vector(model, graph->output_norm, config->step.hidden_size, &output_norm);
    if (status != LM_OK) return status;
    float mean_square = 0.0f;
    for (uint32_t i = 0u; i < config->step.hidden_size; ++i) mean_square += hidden[i] * hidden[i];
    mean_square /= static_cast<float>(config->step.hidden_size);
    const float output_scale = 1.0f / std::sqrt(mean_square + config->step.rms_epsilon);
    if (!std::isfinite(output_scale)) return LM_ERR_RANGE;
    for (uint32_t i = 0u; i < config->step.hidden_size; ++i) normed[i] = hidden[i] * output_scale * output_norm[i];
    if (!finite_array(normed.data(), normed.size())) return LM_ERR_RANGE;
    status = profile_matvec(model, graph->output, &config->step.matvec, packed_scratch,
                            packed_scratch_bytes, config->step.vocab_size,
                            config->step.hidden_size, normed.data(), out_logits);
    if (status != LM_OK) return status;
    return finite_array(out_logits, config->step.vocab_size) ? LM_OK : LM_ERR_RANGE;
}

lm_status lm_model_generate_native(const lm_model_file *model,
                                   const lm_decoder_graph_binding *graph,
                                   const lm_native_generation_config *config,
                                   const uint32_t *prompt_tokens,
                                   size_t prompt_count,
                                   void *packed_scratch,
                                   uint64_t packed_scratch_bytes,
                                   uint32_t *out_tokens,
                                   size_t token_capacity,
                                   size_t *out_count) {
    if (!model || !graph || !config || !prompt_tokens || !packed_scratch || !out_tokens || !out_count ||
        prompt_count == 0u || config->max_new_tokens == 0u ||
        prompt_count > static_cast<size_t>(1u << 20u) || config->max_new_tokens > kMaxNativeGeneratedTokens ||
        prompt_count > static_cast<size_t>((1u << 20u) - config->max_new_tokens) ||
        token_capacity < config->max_new_tokens ||
        config->step.vocab_size == 0u || config->step.hidden_size == 0u ||
        config->step.hidden_size > kNativeProfileMaxHidden ||
        config->step.vocab_size > kNativeProfileMaxVocab ||
        config->step.intermediate_size == 0u ||
        config->step.intermediate_size > kNativeProfileMaxIntermediate ||
        (config->has_stop_token && (config->stop_token >= config->step.vocab_size)))
        return LM_ERR_ARGUMENT;
    for (size_t i = 0u; i < prompt_count; ++i)
        if (prompt_tokens[i] >= config->step.vocab_size) return LM_ERR_RANGE;
    const uint32_t context_tokens = static_cast<uint32_t>(prompt_count) + config->max_new_tokens;
    const uint32_t kv_bytes = config->step.hidden_size * static_cast<uint32_t>(sizeof(float));
    std::vector<lm_kv_cache *> layer_caches;
    std::vector<uint8_t> page_live;
    std::vector<float> logits;
    try {
        layer_caches.assign(graph->layer_count, nullptr);
        page_live.assign(graph->layer_count, 0u);
        logits.assign(config->step.vocab_size, 0.0f);
    } catch (const std::bad_alloc &) {
        return LM_ERR_CAPACITY;
    }
    auto cleanup = [&]() {
        for (uint32_t layer = 0u; layer < graph->layer_count; ++layer) {
            if (!layer_caches[layer]) continue;
            if (page_live[layer] != 0u) (void)lm_kv_cache_release(layer_caches[layer], 0u);
            lm_kv_cache_destroy(layer_caches[layer]);
        }
    };
    lm_status status = LM_OK;
    for (uint32_t layer = 0u; layer < graph->layer_count; ++layer) {
        status = lm_kv_cache_create_with_payload(1u, context_tokens, kv_bytes, kv_bytes,
                                                 &layer_caches[layer]);
        if (status != LM_OK) break;
    }
    if (status == LM_OK) {
        lm_native_transformer_config transformer{};
        transformer.step = config->step;
        for (size_t i = 0u; i < prompt_count; ++i) {
            for (uint32_t layer = 0u; layer < graph->layer_count; ++layer) {
                uint32_t page_id = page_live[layer] == 0u ? UINT32_MAX : 0u;
                status = lm_kv_cache_append(layer_caches[layer], &page_id, 1u);
                if (status != LM_OK || page_id != 0u) break;
                page_live[layer] = 1u;
            }
            if (status != LM_OK) break;
            transformer.step.token_offset = static_cast<uint32_t>(i);
            status = lm_model_execute_native_transformer(model, graph, &transformer,
                                                         prompt_tokens[i], layer_caches.data(),
                                                         static_cast<uint32_t>(layer_caches.size()), 0u,
                                                         packed_scratch, packed_scratch_bytes,
                                                         logits.data(), logits.size());
            if (status != LM_OK) break;
        }
        *out_count = 0u;
        for (uint32_t i = 0u; status == LM_OK && i < config->max_new_tokens; ++i) {
            float probability = 0.0f;
            uint32_t next_token = 0u;
            status = lm_sample_logits(logits.data(), config->step.vocab_size, &config->sampling,
                                      &next_token, &probability);
            if (status != LM_OK) break;
            out_tokens[*out_count] = next_token;
            ++*out_count;
            if (config->has_stop_token && next_token == config->stop_token) break;
            if (i + 1u == config->max_new_tokens) break;
            for (uint32_t layer = 0u; layer < graph->layer_count; ++layer) {
                uint32_t page_id = page_live[layer] == 0u ? UINT32_MAX : 0u;
                status = lm_kv_cache_append(layer_caches[layer], &page_id, 1u);
                if (status != LM_OK || page_id != 0u) break;
                page_live[layer] = 1u;
            }
            if (status != LM_OK) break;
            transformer.step.token_offset = static_cast<uint32_t>(prompt_count) + i;
            status = lm_model_execute_native_transformer(model, graph, &transformer, next_token,
                                                         layer_caches.data(),
                                                         static_cast<uint32_t>(layer_caches.size()), 0u,
                                                         packed_scratch, packed_scratch_bytes,
                                                         logits.data(), logits.size());
        }
    }
    cleanup();
    return status;
}

lm_status lm_model_generate_native_text(const lm_model_file *model,
                                        const lm_decoder_graph_binding *graph,
                                        const lm_native_generation_config *config,
                                        const char *prompt, size_t prompt_bytes,
                                        void *packed_scratch,
                                        uint64_t packed_scratch_bytes,
                                        char *out_text, size_t out_capacity,
                                        size_t *out_bytes) {
    if (!model || !graph || !config || (!prompt && prompt_bytes != 0u) || !packed_scratch ||
        !out_text || !out_bytes || out_capacity == 0u) return LM_ERR_ARGUMENT;
    const size_t token_capacity = static_cast<size_t>(config->max_new_tokens);
    if (token_capacity == 0u || prompt_bytes == std::numeric_limits<size_t>::max() ||
        config->max_new_tokens > kMaxNativeGeneratedTokens) return LM_ERR_ARGUMENT;
    std::vector<uint32_t> prompt_tokens;
    std::vector<uint32_t> generated;
    size_t prompt_count = 0u;
    lm_status status = LM_OK;
    try {
        prompt_tokens.resize(prompt_bytes + 1u);
        generated.resize(token_capacity);
    } catch (const std::bad_alloc &) {
        return LM_ERR_CAPACITY;
    }
    status = lm_model_token_encode(model, prompt, prompt_bytes, prompt_tokens.data(),
                                   prompt_tokens.size(), &prompt_count);
    if (status != LM_OK) return status;
    if (prompt_count == 0u) return LM_ERR_ARGUMENT;
    size_t generated_count = 0u;
    status = lm_model_generate_native(model, graph, config, prompt_tokens.data(), prompt_count,
                                      packed_scratch, packed_scratch_bytes, generated.data(),
                                      generated.size(), &generated_count);
    if (status != LM_OK) return status;
    return lm_model_token_decode(model, generated.data(), generated_count, out_text,
                                 out_capacity, out_bytes);
}

lm_status lm_cpu_decoder_create(const lm_cpu_decoder_config *config, lm_cpu_decoder **out_decoder) {
    if (!config || !out_decoder || config->vocab_size == 0u || config->hidden_size == 0u || config->max_context == 0u ||
        config->hidden_size > 4096u || config->max_context > 1u << 20u || !(config->rms_epsilon > 0.0f) ||
        !std::isfinite(config->rms_epsilon) || (config->use_rope && (!(config->rope_theta > 1.0f) || !std::isfinite(config->rope_theta))))
        return LM_ERR_ARGUMENT;
    if (!finite_array(config->embedding, static_cast<size_t>(config->vocab_size) * config->hidden_size) ||
        !finite_array(config->rms_gamma_1, config->hidden_size) || !matrix_finite(config->wq, config->hidden_size, config->hidden_size) ||
        !matrix_finite(config->wk, config->hidden_size, config->hidden_size) || !matrix_finite(config->wv, config->hidden_size, config->hidden_size) ||
        !matrix_finite(config->wo, config->hidden_size, config->hidden_size) || !finite_array(config->rms_gamma_2, config->hidden_size) ||
        !matrix_finite(config->w1, config->hidden_size, config->hidden_size) || !matrix_finite(config->w2, config->hidden_size, config->hidden_size) ||
        !matrix_finite(config->wout, config->hidden_size, config->vocab_size)) return LM_ERR_ARGUMENT;
    try {
        lm_cpu_decoder *decoder = new lm_cpu_decoder();
        decoder->config = *config;
        decoder->position = 0u;
        const size_t cache_elements = static_cast<size_t>(config->max_context) * config->hidden_size;
        decoder->keys.assign(cache_elements, 0.0f);
        decoder->values.assign(cache_elements, 0.0f);
        decoder->scratch.resize(static_cast<size_t>(config->hidden_size) * 9u + static_cast<size_t>(config->max_context) * 2u);
        *out_decoder = decoder;
        return LM_OK;
    } catch (const std::bad_alloc &) {
        *out_decoder = nullptr;
        return LM_ERR_CAPACITY;
    }
}

void lm_cpu_decoder_destroy(lm_cpu_decoder *decoder) {
    delete decoder;
}

lm_status lm_cpu_decoder_reset(lm_cpu_decoder *decoder) {
    if (!decoder) return LM_ERR_ARGUMENT;
    decoder->position = 0u;
    std::fill(decoder->keys.begin(), decoder->keys.end(), 0.0f);
    std::fill(decoder->values.begin(), decoder->values.end(), 0.0f);
    return LM_OK;
}

lm_status lm_cpu_decoder_step(lm_cpu_decoder *decoder, uint32_t token_id, float *out_logits, size_t logits_count) {
    if (!decoder || !out_logits || logits_count < decoder->config.vocab_size) return LM_ERR_ARGUMENT;
    if (token_id >= decoder->config.vocab_size) return LM_ERR_RANGE;
    if (decoder->position >= decoder->config.max_context) return LM_ERR_CAPACITY;
    const uint32_t h = decoder->config.hidden_size;
    float *x = decoder->scratch.data();
    float *norm1 = x + h;
    float *q = norm1 + h;
    float *k = q + h;
    float *v = k + h;
    float *attention = v + h;
    float *residual = attention + h;
    float *norm2 = residual + h;
    float *up = norm2 + h;
    float *scores = up + h;
    float *weights = scores + decoder->config.max_context;
    std::memcpy(x, decoder->config.embedding + static_cast<size_t>(token_id) * h, sizeof(float) * h);
    if (!rms_norm(x, h, decoder->config.rms_epsilon, decoder->config.rms_gamma_1, norm1)) return LM_ERR_RANGE;
    matvec(norm1, decoder->config.wq, h, h, q);
    matvec(norm1, decoder->config.wk, h, h, k);
    matvec(norm1, decoder->config.wv, h, h, v);
    if (decoder->config.use_rope) {
        apply_rope(q, h, decoder->position, decoder->config.rope_theta);
        apply_rope(k, h, decoder->position, decoder->config.rope_theta);
    }
    std::memcpy(decoder->keys.data() + static_cast<size_t>(decoder->position) * h, k, sizeof(float) * h);
    std::memcpy(decoder->values.data() + static_cast<size_t>(decoder->position) * h, v, sizeof(float) * h);
    const float scale = 1.0f / std::sqrt(static_cast<float>(h));
    for (uint32_t t = 0u; t <= decoder->position; ++t) {
        const float *key = decoder->keys.data() + static_cast<size_t>(t) * h;
        float dot = 0.0f;
        for (uint32_t i = 0u; i < h; ++i) dot += q[i] * key[i];
        scores[t] = dot * scale;
    }
    if (lm_cpu_softmax_f32(scores, weights, decoder->position + 1u) != LM_OK) return LM_ERR_RANGE;
    std::fill(attention, attention + h, 0.0f);
    for (uint32_t t = 0u; t <= decoder->position; ++t) {
        const float *value = decoder->values.data() + static_cast<size_t>(t) * h;
        for (uint32_t i = 0u; i < h; ++i) attention[i] += weights[t] * value[i];
    }
    matvec(attention, decoder->config.wo, h, h, residual);
    for (uint32_t i = 0u; i < h; ++i) residual[i] += x[i];
    if (!rms_norm(residual, h, decoder->config.rms_epsilon, decoder->config.rms_gamma_2, norm2)) return LM_ERR_RANGE;
    matvec(norm2, decoder->config.w1, h, h, up);
    for (uint32_t i = 0u; i < h; ++i) up[i] = up[i] / (1.0f + std::exp(-up[i]));
    matvec(up, decoder->config.w2, h, h, attention);
    for (uint32_t i = 0u; i < h; ++i) residual[i] += attention[i];
    matvec(residual, decoder->config.wout, h, decoder->config.vocab_size, out_logits);
    if (!finite_array(out_logits, decoder->config.vocab_size)) return LM_ERR_RANGE;
    ++decoder->position;
    return LM_OK;
}

uint32_t lm_cpu_decoder_position(const lm_cpu_decoder *decoder) {
    return decoder ? decoder->position : 0u;
}
