#include "lm/lm.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <new>
#include <string>
#include <vector>

struct lm_model_file {
    lm_file *file;
    lm_model_info info;
    lm_model_architecture architecture;
    uint8_t has_architecture;
    std::vector<lm_model_tensor_info> tensors;
    std::vector<std::string> tokens;
};

namespace {

constexpr uint64_t kMaxSafeTensorsHeader = 32ull << 20u;
constexpr uint64_t kMaxContainerItems = 1ull << 20u;
constexpr uint64_t kMaxVocabularyTokens = 65536u;
constexpr uint64_t kMaxVocabularyTokenBytes = 256u;

bool finite_array(const float *data, size_t count) {
    if (!data) return false;
    for (size_t i = 0u; i < count; ++i) if (!std::isfinite(data[i])) return false;
    return true;
}

void set_error(char *dst, size_t cap, const char *text) {
    if (!dst || cap == 0u) return;
    std::strncpy(dst, text, cap - 1u);
    dst[cap - 1u] = '\0';
}

uint64_t read_u64_le(const unsigned char *p) {
    uint64_t value = 0u;
    for (unsigned i = 0u; i < 8u; ++i) value |= static_cast<uint64_t>(p[i]) << (8u * i);
    return value;
}

uint32_t read_u32_le(const unsigned char *p) {
    uint32_t value = 0u;
    for (unsigned i = 0u; i < 4u; ++i) value |= static_cast<uint32_t>(p[i]) << (8u * i);
    return value;
}

uint64_t align_up(uint64_t value, uint64_t alignment) {
    const uint64_t remainder = value % alignment;
    return remainder == 0u ? value : value + (alignment - remainder);
}

lm_status native_contract(const lm_model_tensor_info &descriptor, lm_quant_format *format,
                         uint32_t *elements_per_block, uint32_t *bytes_per_block,
                         uint64_t *elements, uint64_t *bytes) {
    if (!format || !elements_per_block || !bytes_per_block || !elements || !bytes) return LM_ERR_ARGUMENT;
    if (descriptor.type == 2u) {
        *format = LM_QUANT_GGML_Q4_0;
        *elements_per_block = 32u;
        *bytes_per_block = 18u;
    } else if (descriptor.type == 8u) {
        *format = LM_QUANT_GGML_Q8_0;
        *elements_per_block = 32u;
        *bytes_per_block = 34u;
    } else if (descriptor.type == 12u) {
        *format = LM_QUANT_GGML_Q4_K;
        *elements_per_block = 256u;
        *bytes_per_block = 144u;
    } else {
        return LM_ERR_UNSUPPORTED;
    }
    if (descriptor.rank == 0u || descriptor.rank > 8u) return LM_ERR_PARSE;
    uint64_t count = 1u;
    for (uint32_t i = 0u; i < descriptor.rank; ++i) {
        if (descriptor.dims[i] == 0u || count > std::numeric_limits<uint64_t>::max() / descriptor.dims[i]) return LM_ERR_RANGE;
        count *= descriptor.dims[i];
    }
    if (count % *elements_per_block != 0u) return LM_ERR_PARSE;
    const uint64_t blocks = count / *elements_per_block;
    if (blocks > std::numeric_limits<uint64_t>::max() / *bytes_per_block) return LM_ERR_RANGE;
    *elements = count;
    *bytes = blocks * *bytes_per_block;
    return LM_OK;
}

class BinaryReader {
public:
    explicit BinaryReader(const char *path) : stream_(path, std::ios::binary), position_(0u), size_(0u) {
        if (!stream_) return;
        stream_.seekg(0, std::ios::end);
        const std::streamoff end = stream_.tellg();
        if (end < 0) { stream_.setstate(std::ios::failbit); return; }
        size_ = static_cast<uint64_t>(end);
        stream_.seekg(0, std::ios::beg);
    }

    bool good() const { return stream_.good(); }
    uint64_t position() const { return position_; }
    uint64_t size() const { return size_; }

    bool read(void *dst, uint64_t bytes) {
        if (bytes > size_ - (position_ <= size_ ? position_ : size_)) return false;
        if (bytes > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) return false;
        stream_.read(static_cast<char *>(dst), static_cast<std::streamsize>(bytes));
        if (stream_.gcount() != static_cast<std::streamsize>(bytes)) return false;
        position_ += bytes;
        return true;
    }

    bool skip(uint64_t bytes) {
        if (bytes > size_ - (position_ <= size_ ? position_ : size_)) return false;
        if (bytes > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) return false;
        stream_.seekg(static_cast<std::streamoff>(bytes), std::ios::cur);
        if (!stream_) return false;
        position_ += bytes;
        return true;
    }

    bool u8(uint8_t *value) { return read(value, 1u); }
    bool u32(uint32_t *value) { unsigned char bytes[4] = {}; if (!read(bytes, 4u)) return false; *value = read_u32_le(bytes); return true; }
    bool u64(uint64_t *value) { unsigned char bytes[8] = {}; if (!read(bytes, 8u)) return false; *value = read_u64_le(bytes); return true; }

    bool string(std::string *value, uint64_t max_bytes) {
        uint64_t length = 0u;
        if (!u64(&length) || length > max_bytes || length > size_ - position_) return false;
        value->assign(static_cast<size_t>(length), '\0');
        return length == 0u || read(value->data(), length);
    }

private:
    std::ifstream stream_;
    uint64_t position_;
    uint64_t size_;
};

bool valid_gguf_key(const std::string &key) {
    if (key.empty() || key.size() > 65535u) return false;
    bool segment_has_character = false;
    for (size_t i = 0u; i < key.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(key[i]);
        if (c == '.') {
            if (!segment_has_character) return false;
            segment_has_character = false;
        } else if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_') {
            segment_has_character = true;
        } else {
            return false;
        }
    }
    return segment_has_character;
}

bool skip_gguf_value(BinaryReader &reader, uint32_t type, uint32_t depth) {
    if (depth > 64u) return false;
    switch (type) {
        case 0u: case 1u: case 7u: return reader.skip(1u);
        case 2u: case 3u: return reader.skip(2u);
        case 4u: case 5u: case 6u: return reader.skip(4u);
        case 10u: case 11u: case 12u: return reader.skip(8u);
        case 8u: {
            uint64_t length = 0u;
            return reader.u64(&length) && length <= (16ull << 20u) && reader.skip(length);
        }
        case 9u: {
            uint32_t element_type = 0u;
            uint64_t length = 0u;
            if (!reader.u32(&element_type) || !reader.u64(&length) || length > kMaxContainerItems) return false;
            for (uint64_t i = 0u; i < length; ++i)
                if (!skip_gguf_value(reader, element_type, depth + 1u)) return false;
            return true;
        }
        default: return false;
    }
}

bool read_gguf_alignment(BinaryReader &reader, uint32_t type, uint64_t *alignment) {
    if (type == 4u) { uint32_t value = 0u; if (!reader.u32(&value)) return false; *alignment = value; return true; }
    if (type == 10u) return reader.u64(alignment);
    return false;
}

bool has_suffix(const std::string &value, const char *suffix) {
    const size_t suffix_size = std::strlen(suffix);
    return value.size() >= suffix_size && value.compare(value.size() - suffix_size, suffix_size, suffix) == 0;
}

bool read_moe_count(BinaryReader &reader, uint32_t type, uint64_t *count) {
    if (type == 4u) { uint32_t value = 0u; if (!reader.u32(&value)) return false; *count = value; return true; }
    if (type == 10u) return reader.u64(count);
    return false;
}

bool read_arch_u32(BinaryReader &reader, uint32_t type, uint32_t *value) {
    if (!value) return false;
    uint64_t parsed = 0u;
    if (!read_moe_count(reader, type, &parsed) || parsed == 0u || parsed > UINT32_MAX) return false;
    *value = static_cast<uint32_t>(parsed);
    return true;
}

bool read_arch_float(BinaryReader &reader, uint32_t type, float *value) {
    if (!value) return false;
    if (type == 6u) {
        uint32_t bits = 0u;
        if (!reader.u32(&bits)) return false;
        std::memcpy(value, &bits, sizeof(bits));
        return std::isfinite(*value) && *value > 0.0f;
    }
    if (type == 12u) {
        uint64_t bits = 0u;
        if (!reader.u64(&bits)) return false;
        double parsed = 0.0;
        std::memcpy(&parsed, &bits, sizeof(bits));
        if (!std::isfinite(parsed) || parsed <= 0.0 || parsed > static_cast<double>(std::numeric_limits<float>::max())) return false;
        *value = static_cast<float>(parsed);
        return std::isfinite(*value) && *value > 0.0f;
    }
    return false;
}

bool read_token_array(BinaryReader &reader, uint32_t type, std::vector<std::string> *tokens) {
    if (type != 9u || !tokens) return false;
    uint32_t element_type = 0u;
    uint64_t length = 0u;
    if (!reader.u32(&element_type) || !reader.u64(&length) || element_type != 8u ||
        length == 0u || length > kMaxVocabularyTokens)
        return false;
    std::vector<std::string> parsed;
    try {
        parsed.reserve(static_cast<size_t>(length));
        for (uint64_t i = 0u; i < length; ++i) {
            std::string token;
            if (!reader.string(&token, kMaxVocabularyTokenBytes)) return false;
            parsed.push_back(std::move(token));
        }
    } catch (const std::bad_alloc &) {
        return false;
    }
    *tokens = std::move(parsed);
    return true;
}

lm_status inspect_gguf(const char *path, lm_model_info *out_info, char *error_text, size_t error_capacity,
                       std::vector<lm_model_tensor_info> *out_tensors = nullptr,
                       std::vector<std::string> *out_tokens = nullptr,
                       lm_model_architecture *out_architecture = nullptr) {
    BinaryReader reader(path);
    if (!reader.good()) { set_error(error_text, error_capacity, "cannot open model file"); return LM_ERR_IO; }
    if (reader.size() < 24u) { set_error(error_text, error_capacity, "GGUF header is truncated"); return LM_ERR_PARSE; }
    unsigned char magic[4] = {};
    uint32_t version = 0u;
    uint64_t tensor_count = 0u;
    uint64_t metadata_count = 0u;
    if (!reader.read(magic, 4u) || !reader.u32(&version) || !reader.u64(&tensor_count) || !reader.u64(&metadata_count)) {
        set_error(error_text, error_capacity, "cannot read GGUF header"); return LM_ERR_IO;
    }
    if (std::memcmp(magic, "GGUF", 4u) != 0) { set_error(error_text, error_capacity, "invalid GGUF magic"); return LM_ERR_PARSE; }
    if (version != 3u) { set_error(error_text, error_capacity, "only GGUF version 3 is supported"); return LM_ERR_UNSUPPORTED; }
    if (tensor_count > kMaxContainerItems || metadata_count > kMaxContainerItems) {
        set_error(error_text, error_capacity, "GGUF item count exceeds safety limit"); return LM_ERR_CAPACITY;
    }
    uint64_t alignment = 32u;
    uint64_t expert_count = 0u;
    uint64_t experts_per_token = 0u;
    bool has_expert_count = false;
    bool has_experts_per_token = false;
    std::vector<std::string> tokens;
    lm_model_architecture architecture{};
    bool is_llama = false;
    bool has_context_length = false;
    bool has_embedding_length = false;
    bool has_block_count = false;
    bool has_head_count = false;
    bool has_head_count_kv = false;
    bool has_intermediate_length = false;
    bool has_rope_frequency_base = false;
    for (uint64_t i = 0u; i < metadata_count; ++i) {
        std::string key;
        uint32_t type = 0u;
        if (!reader.string(&key, 65535u) || !valid_gguf_key(key) || !reader.u32(&type) || type > 12u) {
            set_error(error_text, error_capacity, "invalid GGUF metadata entry"); return LM_ERR_PARSE;
        }
        const uint64_t old_alignment = alignment;
        if (key == "general.architecture") {
            std::string architecture_name;
            if (type != 8u || !reader.string(&architecture_name, 64u)) {
                set_error(error_text, error_capacity, "invalid GGUF architecture metadata"); return LM_ERR_PARSE;
            }
            if (architecture_name == "llama") is_llama = true;
        } else if (key == "general.alignment") {
            if (!read_gguf_alignment(reader, type, &alignment)) {
                set_error(error_text, error_capacity, "invalid GGUF alignment metadata"); return LM_ERR_PARSE;
            }
            if (alignment == 0u || alignment > (1ull << 20u) || (alignment % 8u) != 0u) {
                set_error(error_text, error_capacity, "GGUF alignment must be a bounded multiple of 8"); return LM_ERR_PARSE;
            }
        } else if (has_suffix(key, ".expert_count")) {
            if (!read_moe_count(reader, type, &expert_count)) {
                set_error(error_text, error_capacity, "invalid GGUF expert count metadata"); return LM_ERR_PARSE;
            }
            has_expert_count = true;
        } else if (has_suffix(key, ".expert_used_count")) {
            if (!read_moe_count(reader, type, &experts_per_token)) {
                set_error(error_text, error_capacity, "invalid GGUF experts-per-token metadata"); return LM_ERR_PARSE;
            }
            has_experts_per_token = true;
        } else if (key == "tokenizer.ggml.tokens") {
            if (!tokens.empty() || !read_token_array(reader, type, &tokens)) {
                set_error(error_text, error_capacity, "invalid GGUF token vocabulary metadata"); return LM_ERR_PARSE;
            }
        } else if (key == "llama.context_length") {
            if (has_context_length || !read_arch_u32(reader, type, &architecture.context_length)) return LM_ERR_PARSE;
            has_context_length = true;
        } else if (key == "llama.embedding_length") {
            if (has_embedding_length || !read_arch_u32(reader, type, &architecture.embedding_length)) return LM_ERR_PARSE;
            has_embedding_length = true;
        } else if (key == "llama.block_count") {
            if (has_block_count || !read_arch_u32(reader, type, &architecture.block_count)) return LM_ERR_PARSE;
            has_block_count = true;
        } else if (key == "llama.attention.head_count") {
            if (has_head_count || !read_arch_u32(reader, type, &architecture.head_count)) return LM_ERR_PARSE;
            has_head_count = true;
        } else if (key == "llama.attention.head_count_kv") {
            if (has_head_count_kv || !read_arch_u32(reader, type, &architecture.head_count_kv)) return LM_ERR_PARSE;
            has_head_count_kv = true;
        } else if (key == "llama.feed_forward_length") {
            if (has_intermediate_length || !read_arch_u32(reader, type, &architecture.intermediate_length)) return LM_ERR_PARSE;
            has_intermediate_length = true;
        } else if (key == "llama.rope.freq_base") {
            if (has_rope_frequency_base || !read_arch_float(reader, type, &architecture.rope_frequency_base)) return LM_ERR_PARSE;
            has_rope_frequency_base = true;
        } else if (!skip_gguf_value(reader, type, 0u)) {
            set_error(error_text, error_capacity, "invalid GGUF metadata value"); return LM_ERR_PARSE;
        }
        if (key != "general.alignment") alignment = old_alignment;
    }
    if (out_architecture) *out_architecture = lm_model_architecture{};
    if (is_llama && has_context_length && has_embedding_length && has_block_count && has_head_count &&
        has_head_count_kv && has_intermediate_length && has_rope_frequency_base &&
        architecture.head_count_kv <= architecture.head_count && architecture.embedding_length > 0u) {
        if (out_architecture) *out_architecture = architecture;
    }
    if ((has_expert_count != has_experts_per_token) || expert_count == 0u || experts_per_token == 0u ||
        expert_count > UINT32_MAX || experts_per_token > UINT32_MAX || experts_per_token > expert_count) {
        if (has_expert_count || has_experts_per_token) {
            set_error(error_text, error_capacity, "invalid GGUF MoE expert metadata"); return LM_ERR_PARSE;
        }
    }
    std::vector<uint64_t> relative_offsets;
    relative_offsets.reserve(static_cast<size_t>(tensor_count));
    std::vector<lm_model_tensor_info> parsed_tensors;
    parsed_tensors.reserve(static_cast<size_t>(tensor_count));
    for (uint64_t i = 0u; i < tensor_count; ++i) {
        std::string name;
        uint32_t dimensions = 0u;
        if (!reader.string(&name, 64u) || name.empty() || !reader.u32(&dimensions) || dimensions > 8u) {
            set_error(error_text, error_capacity, "invalid GGUF tensor descriptor"); return LM_ERR_PARSE;
        }
        lm_model_tensor_info tensor_info{};
        std::strncpy(tensor_info.name, name.c_str(), sizeof(tensor_info.name) - 1u);
        tensor_info.rank = dimensions;
        for (uint32_t d = 0u; d < dimensions; ++d) {
            uint64_t dimension = 0u;
            if (!reader.u64(&dimension) || dimension == 0u) {
                set_error(error_text, error_capacity, "invalid GGUF tensor dimension"); return LM_ERR_PARSE;
            }
            tensor_info.dims[d] = dimension;
        }
        uint32_t tensor_type = 0u;
        uint64_t offset = 0u;
        if (!reader.u32(&tensor_type) || tensor_type > 39u || !reader.u64(&offset)) {
            set_error(error_text, error_capacity, "invalid GGUF tensor type or offset"); return LM_ERR_PARSE;
        }
        if ((offset % alignment) != 0u) {
            set_error(error_text, error_capacity, "GGUF tensor offset is not aligned"); return LM_ERR_PARSE;
        }
        tensor_info.type = tensor_type;
        tensor_info.relative_offset = offset;
        parsed_tensors.push_back(tensor_info);
        relative_offsets.push_back(offset);
    }
    const uint64_t tensor_data_start = align_up(reader.position(), alignment);
    if (tensor_data_start > reader.size()) {
        set_error(error_text, error_capacity, "GGUF tensor-data alignment exceeds file"); return LM_ERR_PARSE;
    }
    std::sort(relative_offsets.begin(), relative_offsets.end());
    const uint64_t tensor_data_bytes = reader.size() - tensor_data_start;
    for (size_t i = 0u; i < relative_offsets.size(); ++i) {
        if (i > 0u && relative_offsets[i] == relative_offsets[i - 1u]) {
            set_error(error_text, error_capacity, "duplicate GGUF tensor offsets"); return LM_ERR_PARSE;
        }
        if (relative_offsets[i] > tensor_data_bytes) {
            set_error(error_text, error_capacity, "GGUF tensor offset exceeds file"); return LM_ERR_PARSE;
        }
    }
    for (const lm_model_tensor_info &tensor : parsed_tensors) {
        lm_quant_format format = LM_QUANT_NONE;
        uint32_t elements_per_block = 0u;
        uint32_t bytes_per_block = 0u;
        uint64_t elements = 0u;
        uint64_t expected_bytes = 0u;
        const lm_status contract = native_contract(tensor, &format, &elements_per_block,
                                                    &bytes_per_block, &elements, &expected_bytes);
        if (contract == LM_ERR_UNSUPPORTED) continue;
        if (contract != LM_OK) {
            set_error(error_text, error_capacity, "invalid native GGUF tensor shape"); return LM_ERR_PARSE;
        }
        uint64_t limit = tensor_data_bytes;
        for (const uint64_t offset : relative_offsets) {
            if (offset > tensor.relative_offset) { limit = offset; break; }
        }
        if (tensor.relative_offset > limit || expected_bytes > limit - tensor.relative_offset) {
            set_error(error_text, error_capacity, "native GGUF tensor payload exceeds its bounded range"); return LM_ERR_PARSE;
        }
    }
    if (out_tensors) *out_tensors = std::move(parsed_tensors);
    if (out_tokens) *out_tokens = std::move(tokens);
    out_info->format = LM_MODEL_GGUF;
    out_info->version = version;
    out_info->file_bytes = reader.size();
    out_info->header_bytes = tensor_data_start;
    out_info->tensor_count = tensor_count;
    out_info->expert_count = has_expert_count ? static_cast<uint32_t>(expert_count) : 0u;
    out_info->experts_per_token = has_experts_per_token ? static_cast<uint32_t>(experts_per_token) : 0u;
    return LM_OK;
}

class JsonCursor {
public:
    JsonCursor(const char *begin, const char *end) : p_(begin), end_(end) {}
    const char *position() const { return p_; }

    void whitespace() { while (p_ < end_ && (*p_ == ' ' || *p_ == '\n' || *p_ == '\r' || *p_ == '\t')) ++p_; }
    bool character(char expected) { whitespace(); if (p_ >= end_ || *p_ != expected) return false; ++p_; return true; }
    bool consume(char expected) { if (p_ >= end_ || *p_ != expected) return false; ++p_; return true; }

    bool string(std::string *out, size_t max_length) {
        whitespace();
        if (p_ >= end_ || *p_ != '"') return false;
        ++p_;
        out->clear();
        while (p_ < end_) {
            const unsigned char c = static_cast<unsigned char>(*p_++);
            if (c == '"') return true;
            if (c < 0x20u) return false;
            if (c == '\\') {
                if (p_ >= end_) return false;
                const char escaped = *p_++;
                if (escaped == 'u') { if (end_ - p_ < 4) return false; p_ += 4; }
                else if (escaped != '"' && escaped != '\\' && escaped != '/' && escaped != 'b' && escaped != 'f' && escaped != 'n' && escaped != 'r' && escaped != 't') return false;
            }
            if (out->size() < max_length) out->push_back(static_cast<char>(c));
        }
        return false;
    }

    bool unsigned_number(uint64_t *out) {
        whitespace();
        if (p_ >= end_ || *p_ < '0' || *p_ > '9') return false;
        uint64_t value = 0u;
        while (p_ < end_ && *p_ >= '0' && *p_ <= '9') {
            const uint64_t digit = static_cast<uint64_t>(*p_++ - '0');
            if (value > (std::numeric_limits<uint64_t>::max() - digit) / 10u) return false;
            value = value * 10u + digit;
        }
        *out = value;
        return true;
    }

    bool skip_value(uint32_t depth) {
        if (depth > 64u) return false;
        whitespace();
        if (p_ >= end_) return false;
        if (*p_ == '"') { std::string value; return string(&value, 1u << 20u); }
        if (*p_ == '{') {
            ++p_; whitespace(); if (p_ < end_ && *p_ == '}') { ++p_; return true; }
            for (;;) {
                std::string key; if (!string(&key, 1u << 20u) || !character(':') || !skip_value(depth + 1u)) return false;
                whitespace(); if (p_ < end_ && *p_ == '}') { ++p_; return true; }
                if (!character(',')) return false;
            }
        }
        if (*p_ == '[') {
            ++p_; whitespace(); if (p_ < end_ && *p_ == ']') { ++p_; return true; }
            for (;;) {
                if (!skip_value(depth + 1u)) return false;
                whitespace(); if (p_ < end_ && *p_ == ']') { ++p_; return true; }
                if (!character(',')) return false;
            }
        }
        const char *start = p_;
        while (p_ < end_ && *p_ != ',' && *p_ != ']' && *p_ != '}' && *p_ != ' ' && *p_ != '\n' && *p_ != '\r' && *p_ != '\t') ++p_;
        return p_ > start;
    }

private:
    const char *p_;
    const char *end_;
};

uint32_t safe_dtype_code(const std::string &dtype) {
    if (dtype == "F32") return LM_DTYPE_F32;
    if (dtype == "F16") return LM_DTYPE_F16;
    if (dtype == "BF16") return LM_DTYPE_BF16;
    if (dtype == "I8") return LM_DTYPE_I8;
    if (dtype == "I32") return LM_DTYPE_I32;
    if (dtype == "U8") return LM_DTYPE_U8;
    if (dtype == "U4") return 0x100u;
    if (dtype == "I4") return 0x101u;
    return 0xffffffffu;
}

uint64_t safe_dtype_bytes(const std::string &dtype, uint64_t elements, bool *known) {
    uint64_t element_bytes = 0u;
    if (dtype == "BOOL" || dtype == "U8" || dtype == "I8" || dtype == "F8_E4M3" || dtype == "F8_E5M2") element_bytes = 1u;
    else if (dtype == "U16" || dtype == "I16" || dtype == "F16" || dtype == "BF16") element_bytes = 2u;
    else if (dtype == "U32" || dtype == "I32" || dtype == "F32") element_bytes = 4u;
    else if (dtype == "U64" || dtype == "I64" || dtype == "F64") element_bytes = 8u;
    else if (dtype == "U4" || dtype == "I4") {
        *known = true;
        return elements / 2u + (elements % 2u);
    } else { *known = false; return 0u; }
    *known = true;
    if (elements > std::numeric_limits<uint64_t>::max() / element_bytes) return 0u;
    return elements * element_bytes;
}

lm_status inspect_safetensors(const char *path, lm_model_info *out_info, char *error_text, size_t error_capacity,
                              std::vector<lm_model_tensor_info> *out_tensors = nullptr) {
    BinaryReader reader(path);
    if (!reader.good()) { set_error(error_text, error_capacity, "cannot open model file"); return LM_ERR_IO; }
    if (reader.size() < 8u) { set_error(error_text, error_capacity, "SafeTensors header length is truncated"); return LM_ERR_PARSE; }
    uint64_t header_bytes = 0u;
    if (!reader.u64(&header_bytes) || header_bytes == 0u || header_bytes > kMaxSafeTensorsHeader || header_bytes > reader.size() - 8u) {
        set_error(error_text, error_capacity, "invalid SafeTensors header length"); return LM_ERR_PARSE;
    }
    std::vector<char> header(static_cast<size_t>(header_bytes));
    if (!reader.read(header.data(), header_bytes)) { set_error(error_text, error_capacity, "cannot read SafeTensors header"); return LM_ERR_IO; }
    JsonCursor json(header.data(), header.data() + header.size());
    if (!json.character('{')) { set_error(error_text, error_capacity, "SafeTensors header is not a JSON object"); return LM_ERR_PARSE; }
    struct Range { uint64_t begin; uint64_t end; };
    std::vector<Range> ranges;
    json.whitespace();
    if (json.position() < header.data() + header.size() && *json.position() != '}') {
        for (;;) {
            std::string name;
            if (!json.string(&name, 1u << 20u) || !json.character(':')) { set_error(error_text, error_capacity, "invalid SafeTensors key"); return LM_ERR_PARSE; }
            if (name == "__metadata__") {
                if (!json.skip_value(0u)) { set_error(error_text, error_capacity, "invalid SafeTensors metadata"); return LM_ERR_PARSE; }
            } else {
                if (!json.character('{')) { set_error(error_text, error_capacity, "SafeTensors tensor descriptor is not an object"); return LM_ERR_PARSE; }
                bool have_dtype = false, have_shape = false, have_offsets = false;
                std::string dtype;
                std::vector<uint64_t> shape;
                uint64_t elements = 1u, begin = 0u, end = 0u;
                json.whitespace();
                if (json.position() < header.data() + header.size() && *json.position() != '}') {
                    for (;;) {
                        std::string field;
                        if (!json.string(&field, 64u) || !json.character(':')) { set_error(error_text, error_capacity, "invalid SafeTensors tensor field"); return LM_ERR_PARSE; }
                        if (field == "dtype") {
                            if (!json.string(&dtype, 32u)) { set_error(error_text, error_capacity, "invalid SafeTensors dtype"); return LM_ERR_PARSE; }
                            have_dtype = true;
                        } else if (field == "shape") {
                            if (!json.character('[')) { set_error(error_text, error_capacity, "invalid SafeTensors shape"); return LM_ERR_PARSE; }
                            elements = 1u; json.whitespace();
                            if (json.position() < header.data() + header.size() && *json.position() != ']') {
                                for (;;) {
                                    uint64_t dimension = 0u;
                                    if (!json.unsigned_number(&dimension) || (dimension != 0u && elements > std::numeric_limits<uint64_t>::max() / dimension)) { set_error(error_text, error_capacity, "SafeTensors shape overflows"); return LM_ERR_RANGE; }
                                    elements *= dimension;
                                    shape.push_back(dimension);
                                    json.whitespace(); if (json.position() < header.data() + header.size() && *json.position() == ']') { if (!json.consume(']')) return LM_ERR_PARSE; break; }
                                    if (!json.character(',')) { set_error(error_text, error_capacity, "invalid SafeTensors shape array"); return LM_ERR_PARSE; }
                                }
                            } else return LM_ERR_PARSE;
                            have_shape = true;
                        } else if (field == "data_offsets") {
                            if (!json.character('[') || !json.unsigned_number(&begin) || !json.character(',') || !json.unsigned_number(&end) || !json.character(']') || begin > end) { set_error(error_text, error_capacity, "invalid SafeTensors data offsets"); return LM_ERR_PARSE; }
                            have_offsets = true;
                        } else if (!json.skip_value(0u)) { set_error(error_text, error_capacity, "invalid SafeTensors tensor field value"); return LM_ERR_PARSE; }
                        json.whitespace();
                        if (json.position() < header.data() + header.size() && *json.position() == '}') { if (!json.consume('}')) return LM_ERR_PARSE; break; }
                        if (!json.character(',')) { set_error(error_text, error_capacity, "invalid SafeTensors tensor object"); return LM_ERR_PARSE; }
                    }
                } else return LM_ERR_PARSE;
                bool dtype_known = false;
                const uint64_t expected_bytes = safe_dtype_bytes(dtype, elements, &dtype_known);
                if (!have_dtype || !have_shape || !have_offsets || !dtype_known) { set_error(error_text, error_capacity, "unsupported or incomplete SafeTensors descriptor"); return LM_ERR_UNSUPPORTED; }
                if (expected_bytes != end - begin) { set_error(error_text, error_capacity, "SafeTensors descriptor size does not match shape and dtype"); return LM_ERR_PARSE; }
                if (out_tensors) {
                    if (shape.size() > 8u || safe_dtype_code(dtype) == 0xffffffffu) {
                        set_error(error_text, error_capacity, "SafeTensors descriptor cannot be represented"); return LM_ERR_UNSUPPORTED;
                    }
                    lm_model_tensor_info tensor_info{};
                    std::strncpy(tensor_info.name, name.c_str(), sizeof(tensor_info.name) - 1u);
                    tensor_info.rank = static_cast<uint32_t>(shape.size());
                    for (size_t d = 0u; d < shape.size(); ++d) tensor_info.dims[d] = shape[d];
                    tensor_info.type = safe_dtype_code(dtype);
                    tensor_info.relative_offset = begin;
                    out_tensors->push_back(tensor_info);
                }
                ranges.push_back({begin, end});
                if (ranges.size() > kMaxContainerItems) { set_error(error_text, error_capacity, "SafeTensors tensor count exceeds safety limit"); return LM_ERR_CAPACITY; }
            }
            json.whitespace();
            if (json.position() < header.data() + header.size() && *json.position() == '}') { if (!json.consume('}')) return LM_ERR_PARSE; break; }
            if (!json.character(',')) { set_error(error_text, error_capacity, "invalid SafeTensors top-level object"); return LM_ERR_PARSE; }
        }
    } else return LM_ERR_PARSE;
    json.whitespace();
    if (json.position() != header.data() + header.size() || reader.size() - (8u + header_bytes) > std::numeric_limits<uint64_t>::max()) {
        set_error(error_text, error_capacity, "trailing bytes in SafeTensors JSON header"); return LM_ERR_PARSE;
    }
    const uint64_t data_bytes = reader.size() - (8u + header_bytes);
    for (const Range &range : ranges) {
        if (range.end > data_bytes) { set_error(error_text, error_capacity, "SafeTensors tensor range exceeds file"); return LM_ERR_PARSE; }
    }
    std::sort(ranges.begin(), ranges.end(), [](const Range &a, const Range &b) { return a.begin < b.begin; });
    for (size_t i = 1u; i < ranges.size(); ++i) {
        if (ranges[i].begin < ranges[i - 1u].end) { set_error(error_text, error_capacity, "overlapping SafeTensors tensor ranges"); return LM_ERR_PARSE; }
    }
    out_info->format = LM_MODEL_SAFETENSORS;
    out_info->version = 1u;
    out_info->file_bytes = reader.size();
    out_info->header_bytes = 8u + header_bytes;
    out_info->tensor_count = ranges.size();
    return LM_OK;
}

} // namespace

const char *lm_model_format_name(lm_model_format format) {
    switch (format) {
        case LM_MODEL_GGUF: return "gguf";
        case LM_MODEL_SAFETENSORS: return "safetensors";
        default: return "unknown";
    }
}

lm_status lm_model_inspect(const char *path, lm_model_info *out_info,
                           char *error_text, size_t error_capacity) {
    if (!path || !out_info) return LM_ERR_ARGUMENT;
    std::memset(out_info, 0, sizeof(*out_info));
    set_error(error_text, error_capacity, "");
    BinaryReader probe(path);
    if (!probe.good()) { set_error(error_text, error_capacity, "cannot open model file"); return LM_ERR_IO; }
    if (probe.size() >= 4u) {
        unsigned char magic[4] = {};
        if (!probe.read(magic, 4u)) return LM_ERR_IO;
        if (std::memcmp(magic, "GGUF", 4u) == 0) return inspect_gguf(path, out_info, error_text, error_capacity);
    }
    return inspect_safetensors(path, out_info, error_text, error_capacity);
}

lm_status lm_model_open(const char *path, lm_model_file **out_model, char *error_text, size_t error_capacity) {
    if (!path || !out_model) return LM_ERR_ARGUMENT;
    *out_model = nullptr;
    lm_model_info info{};
    const lm_status inspected = lm_model_inspect(path, &info, error_text, error_capacity);
    if (inspected != LM_OK) return inspected;
    lm_file *file = nullptr;
    std::vector<lm_model_tensor_info> tensors;
    std::vector<std::string> tokens;
    lm_model_architecture architecture{};
    if (info.format == LM_MODEL_GGUF) {
        const lm_status descriptors = inspect_gguf(path, &info, error_text, error_capacity, &tensors, &tokens, &architecture);
        if (descriptors != LM_OK) return descriptors;
    } else if (info.format == LM_MODEL_SAFETENSORS) {
        const lm_status descriptors = inspect_safetensors(path, &info, error_text, error_capacity, &tensors);
        if (descriptors != LM_OK) return descriptors;
    }
    const lm_status opened = lm_file_open(path, &file);
    if (opened != LM_OK) return opened;
    try {
        lm_model_file *model = new lm_model_file{file, info, architecture,
                                                   static_cast<uint8_t>(architecture.block_count != 0u),
                                                   std::move(tensors), std::move(tokens)};
        *out_model = model;
        return LM_OK;
    } catch (const std::bad_alloc &) {
        lm_file_close(file);
        return LM_ERR_CAPACITY;
    }
}

void lm_model_close(lm_model_file *model) {
    if (!model) return;
    lm_file_close(model->file);
    delete model;
}

lm_status lm_model_get_info(const lm_model_file *model, lm_model_info *out_info) {
    if (!model || !out_info) return LM_ERR_ARGUMENT;
    *out_info = model->info;
    return LM_OK;
}

lm_status lm_model_get_architecture(const lm_model_file *model, lm_model_architecture *out_architecture) {
    if (!model || !out_architecture) return LM_ERR_ARGUMENT;
    if (model->has_architecture == 0u) return LM_ERR_UNSUPPORTED;
    *out_architecture = model->architecture;
    return LM_OK;
}

lm_status lm_model_tensor_info_at(const lm_model_file *model, uint64_t index, lm_model_tensor_info *out_info) {
    if (!model || !out_info) return LM_ERR_ARGUMENT;
    if (index >= model->tensors.size()) return LM_ERR_RANGE;
    *out_info = model->tensors[static_cast<size_t>(index)];
    return LM_OK;
}

lm_status lm_model_token_count(const lm_model_file *model, uint32_t *out_count) {
    if (!model || !out_count) return LM_ERR_ARGUMENT;
    if (model->tokens.empty()) return LM_ERR_UNSUPPORTED;
    if (model->tokens.size() > UINT32_MAX) return LM_ERR_CAPACITY;
    *out_count = static_cast<uint32_t>(model->tokens.size());
    return LM_OK;
}

lm_status lm_model_token_at(const lm_model_file *model, uint32_t token_id,
                            char *out_token, size_t out_capacity, size_t *out_bytes) {
    if (!model || !out_bytes) return LM_ERR_ARGUMENT;
    if (model->tokens.empty()) return LM_ERR_UNSUPPORTED;
    if (token_id >= model->tokens.size()) return LM_ERR_RANGE;
    const std::string &token = model->tokens[token_id];
    *out_bytes = token.size();
    if (!out_token && out_capacity == 0u) return LM_OK;
    if (!out_token || out_capacity <= token.size()) return LM_ERR_CAPACITY;
    std::memcpy(out_token, token.data(), token.size());
    out_token[token.size()] = '\0';
    return LM_OK;
}

lm_status lm_model_token_encode(const lm_model_file *model, const char *text,
                                size_t text_bytes, uint32_t *out_tokens,
                                size_t token_capacity, size_t *out_count) {
    if (!model || !out_count || (!text && text_bytes != 0u)) return LM_ERR_ARGUMENT;
    if (model->tokens.empty()) return LM_ERR_UNSUPPORTED;
    if (text_bytes == 0u) { *out_count = 0u; return LM_OK; }
    if (!out_tokens || token_capacity == 0u) return LM_ERR_CAPACITY;
    std::vector<uint32_t> encoded;
    try {
        for (size_t position = 0u; position < text_bytes;) {
            uint32_t best_id = UINT32_MAX;
            size_t best_length = 0u;
            for (uint32_t id = 0u; id < model->tokens.size(); ++id) {
                const std::string &candidate = model->tokens[id];
                if (candidate.empty() || candidate.size() > text_bytes - position) continue;
                if (std::memcmp(text + position, candidate.data(), candidate.size()) != 0) continue;
                if (candidate.size() > best_length ||
                    (candidate.size() == best_length && id < best_id)) {
                    best_id = id;
                    best_length = candidate.size();
                }
            }
            if (best_id == UINT32_MAX) return LM_ERR_UNSUPPORTED;
            encoded.push_back(best_id);
            position += best_length;
        }
    } catch (const std::bad_alloc &) {
        return LM_ERR_CAPACITY;
    }
    *out_count = encoded.size();
    if (encoded.size() > token_capacity) return LM_ERR_CAPACITY;
    std::memcpy(out_tokens, encoded.data(), encoded.size() * sizeof(uint32_t));
    return LM_OK;
}

lm_status lm_model_token_decode(const lm_model_file *model, const uint32_t *tokens,
                                size_t token_count, char *out_text,
                                size_t out_capacity, size_t *out_bytes) {
    if (!model || !out_bytes || (!tokens && token_count != 0u)) return LM_ERR_ARGUMENT;
    if (model->tokens.empty()) return LM_ERR_UNSUPPORTED;
    size_t total = 0u;
    for (size_t i = 0u; i < token_count; ++i) {
        if (tokens[i] >= model->tokens.size() || model->tokens[tokens[i]].size() >
            std::numeric_limits<size_t>::max() - total) return LM_ERR_RANGE;
        total += model->tokens[tokens[i]].size();
    }
    *out_bytes = total;
    if (token_count == 0u) return LM_OK;
    if (!out_text && out_capacity == 0u) return LM_OK;
    if (!out_text || out_capacity <= total) return LM_ERR_CAPACITY;
    size_t position = 0u;
    for (size_t i = 0u; i < token_count; ++i) {
        const std::string &token = model->tokens[tokens[i]];
        std::memcpy(out_text + position, token.data(), token.size());
        position += token.size();
    }
    out_text[total] = '\0';
    return LM_OK;
}

lm_status lm_model_tensor_span(const lm_model_file *model, uint64_t relative_offset, uint64_t bytes, lm_file_span *out_span) {
    if (!model || !out_span) return LM_ERR_ARGUMENT;
    if (relative_offset > model->info.file_bytes - model->info.header_bytes ||
        bytes > model->info.file_bytes - model->info.header_bytes - relative_offset)
        return LM_ERR_RANGE;
    return lm_file_span_make(model->file, model->info.header_bytes + relative_offset, bytes, out_span);
}

lm_status lm_model_tensor_bind_native(const lm_model_file *model, uint64_t index, lm_model_tensor_binding *out_binding) {
    if (!model || !out_binding) return LM_ERR_ARGUMENT;
    if (index >= model->tensors.size()) return LM_ERR_RANGE;
    const lm_model_tensor_info &descriptor = model->tensors[static_cast<size_t>(index)];
    lm_quant_format format = LM_QUANT_NONE;
    uint32_t elements_per_block = 0u;
    uint32_t bytes_per_block = 0u;
    uint64_t elements = 0u;
    uint64_t expected_bytes = 0u;
    const lm_status contract = native_contract(descriptor, &format, &elements_per_block,
                                                &bytes_per_block, &elements, &expected_bytes);
    if (contract != LM_OK) return contract;
    const uint64_t data_bytes = model->info.file_bytes - model->info.header_bytes;
    uint64_t limit = data_bytes;
    for (const lm_model_tensor_info &other : model->tensors) {
        if (other.relative_offset > descriptor.relative_offset && other.relative_offset < limit)
            limit = other.relative_offset;
    }
    if (descriptor.relative_offset > limit || expected_bytes > limit - descriptor.relative_offset) return LM_ERR_PARSE;
    lm_file_span span{};
    const lm_status spanned = lm_model_tensor_span(model, descriptor.relative_offset, expected_bytes, &span);
    if (spanned != LM_OK) return spanned;
    std::memset(out_binding, 0, sizeof(*out_binding));
    out_binding->descriptor = descriptor;
    out_binding->span = span;
    out_binding->elements = elements;
    out_binding->quant_format = format;
    out_binding->quant_elements_per_block = elements_per_block;
    out_binding->quant_bytes_per_block = bytes_per_block;
    return LM_OK;
}

lm_status lm_model_tensor_binding_view(const lm_model_tensor_binding *binding, void *data, uint64_t bytes, lm_tensor *out_tensor) {
    if (!binding || !data || !out_tensor) return LM_ERR_ARGUMENT;
    if (binding->span.bytes != bytes) return LM_ERR_CAPACITY;
    uint32_t dims[8] = {};
    if (binding->descriptor.rank == 0u || binding->descriptor.rank > 8u) return LM_ERR_RANGE;
    for (uint32_t i = 0u; i < binding->descriptor.rank; ++i) {
        if (binding->descriptor.dims[i] == 0u || binding->descriptor.dims[i] > UINT32_MAX) return LM_ERR_RANGE;
        dims[i] = static_cast<uint32_t>(binding->descriptor.dims[i]);
    }
    if (binding->quant_format == LM_QUANT_GGML_Q4_0)
        return lm_tensor_make_q4_0_view(data, bytes, binding->descriptor.rank, dims, out_tensor);
    if (binding->quant_format == LM_QUANT_GGML_Q8_0)
        return lm_tensor_make_q8_0_view(data, bytes, binding->descriptor.rank, dims, out_tensor);
    if (binding->quant_format == LM_QUANT_GGML_Q4_K)
        return lm_tensor_make_q4_k_view(data, bytes, binding->descriptor.rank, dims, out_tensor);
    return LM_ERR_UNSUPPORTED;
}

lm_status lm_model_tensor_binding_read(const lm_model_tensor_binding *binding, void *data, uint64_t bytes, lm_tensor *out_tensor) {
    if (!binding || !data || !out_tensor) return LM_ERR_ARGUMENT;
    if (bytes != binding->span.bytes) return LM_ERR_CAPACITY;
    if (bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) return LM_ERR_RANGE;
    const lm_status read = lm_file_span_read(&binding->span, 0u, data, static_cast<size_t>(bytes));
    if (read != LM_OK) return read;
    return lm_model_tensor_binding_view(binding, data, bytes, out_tensor);
}

lm_status lm_model_tensor_matvec_f32_cpu(const lm_model_file *model, uint64_t tensor_index,
                                         void *matrix_scratch, uint64_t scratch_bytes,
                                         uint32_t rows, uint32_t columns,
                                         const float *input, float *out) {
    if (!model || !matrix_scratch || !input || !out || rows == 0u || columns == 0u) return LM_ERR_ARGUMENT;
    if (static_cast<uint64_t>(rows) > UINT64_MAX / columns) return LM_ERR_CAPACITY;
    const uint64_t elements = static_cast<uint64_t>(rows) * columns;
    if (elements > UINT64_MAX / sizeof(float)) return LM_ERR_CAPACITY;
    const uint64_t bytes = elements * sizeof(float);
    if (bytes > scratch_bytes || bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
        return LM_ERR_CAPACITY;
    lm_model_tensor_info descriptor{};
    lm_status status = lm_model_tensor_info_at(model, tensor_index, &descriptor);
    if (status != LM_OK) return status;
    if (descriptor.rank != 2u || descriptor.type != LM_DTYPE_F32 || descriptor.dims[0] != rows ||
        descriptor.dims[1] != columns) return LM_ERR_UNSUPPORTED;
    lm_file_span span{};
    status = lm_model_tensor_span(model, descriptor.relative_offset, bytes, &span);
    if (status != LM_OK) return status;
    status = lm_file_span_read(&span, 0u, matrix_scratch, static_cast<size_t>(bytes));
    if (status != LM_OK) return status;
    const float *matrix = static_cast<const float *>(matrix_scratch);
    for (uint32_t column = 0u; column < columns; ++column) {
        float sum = 0.0f;
        for (uint32_t row = 0u; row < rows; ++row) sum += input[row] * matrix[static_cast<size_t>(row) * columns + column];
        if (!std::isfinite(sum)) return LM_ERR_RANGE;
        out[column] = sum;
    }
    return LM_OK;
}

lm_status lm_model_moe_selected_expert_mlp_q4_k(const lm_model_file *model,
                                                uint64_t gate_up_tensor,
                                                uint64_t down_tensor,
                                                const lm_moe_route *route,
                                                uint32_t expected_layer,
                                                uint32_t hidden_size,
                                                uint32_t intermediate_size,
                                                void *gate_up_scratch,
                                                uint64_t gate_up_scratch_bytes,
                                                void *down_scratch,
                                                uint64_t down_scratch_bytes,
                                                const float *input,
                                                float *selected_outputs,
                                                size_t selected_outputs_count) {
    if (!model || !route || !gate_up_scratch || !down_scratch || !input || !selected_outputs ||
        route->expert_count == 0u || route->experts_per_token == 0u || route->experts_per_token > 16u ||
        hidden_size == 0u || intermediate_size == 0u || hidden_size % 256u != 0u ||
        intermediate_size % 256u != 0u || selected_outputs_count <
        static_cast<size_t>(route->experts_per_token) * hidden_size) return LM_ERR_ARGUMENT;
    if (!finite_array(input, hidden_size)) return LM_ERR_RANGE;
    lm_model_tensor_info gate_descriptor{};
    lm_model_tensor_info down_descriptor{};
    lm_status status = lm_model_tensor_info_at(model, gate_up_tensor, &gate_descriptor);
    if (status != LM_OK) return status;
    status = lm_model_tensor_info_at(model, down_tensor, &down_descriptor);
    if (status != LM_OK) return status;
    lm_moe_tensor_mapping gate_mapping{};
    lm_moe_tensor_mapping down_mapping{};
    status = lm_moe_map_mixtral_tensor(&gate_descriptor, route->expert_count, &gate_mapping);
    if (status != LM_OK) return status;
    status = lm_moe_map_mixtral_tensor(&down_descriptor, route->expert_count, &down_mapping);
    if (status != LM_OK) return status;
    if (gate_mapping.role != LM_MOE_TENSOR_GATE_UP_EXPERT ||
        down_mapping.role != LM_MOE_TENSOR_DOWN_EXPERT || gate_mapping.layer_index != expected_layer ||
        down_mapping.layer_index != expected_layer || gate_descriptor.dims[0] != hidden_size ||
        gate_descriptor.dims[1] != static_cast<uint64_t>(intermediate_size) * 2u ||
        down_descriptor.dims[0] != intermediate_size || down_descriptor.dims[1] != hidden_size)
        return LM_ERR_UNSUPPORTED;
    lm_model_tensor_binding gate_binding{};
    lm_model_tensor_binding down_binding{};
    status = lm_model_tensor_bind_native(model, gate_up_tensor, &gate_binding);
    if (status != LM_OK) return status;
    status = lm_model_tensor_bind_native(model, down_tensor, &down_binding);
    if (status != LM_OK) return status;
    if (gate_binding.quant_format != LM_QUANT_GGML_Q4_K || down_binding.quant_format != LM_QUANT_GGML_Q4_K ||
        gate_binding.span.bytes % route->expert_count != 0u || down_binding.span.bytes % route->expert_count != 0u)
        return LM_ERR_UNSUPPORTED;
    const uint64_t gate_expert_bytes = gate_binding.span.bytes / route->expert_count;
    const uint64_t down_expert_bytes = down_binding.span.bytes / route->expert_count;
    if (gate_expert_bytes > gate_up_scratch_bytes || down_expert_bytes > down_scratch_bytes ||
        gate_expert_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
        down_expert_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) return LM_ERR_CAPACITY;
    std::vector<float> gate_up_output;
    std::vector<float> activation;
    try {
        gate_up_output.assign(static_cast<size_t>(intermediate_size) * 2u, 0.0f);
        activation.assign(intermediate_size, 0.0f);
    } catch (const std::bad_alloc &) {
        return LM_ERR_CAPACITY;
    }
    const uint32_t gate_dims[2] = {intermediate_size * 2u, hidden_size};
    const uint32_t down_dims[2] = {hidden_size, intermediate_size};
    for (uint32_t slot = 0u; slot < route->experts_per_token; ++slot) {
        const uint32_t expert = route->selected[slot];
        if (expert >= route->expert_count || !std::isfinite(route->weights[slot])) return LM_ERR_RANGE;
        const uint64_t gate_offset = static_cast<uint64_t>(expert) * gate_expert_bytes;
        const uint64_t down_offset = static_cast<uint64_t>(expert) * down_expert_bytes;
        status = lm_file_span_read(&gate_binding.span, gate_offset, gate_up_scratch,
                                   static_cast<size_t>(gate_expert_bytes));
        if (status != LM_OK) return status;
        status = lm_file_span_read(&down_binding.span, down_offset, down_scratch,
                                   static_cast<size_t>(down_expert_bytes));
        if (status != LM_OK) return status;
        lm_tensor gate_view{};
        lm_tensor down_view{};
        status = lm_tensor_make_q4_k_view(gate_up_scratch, gate_expert_bytes, 2u, gate_dims, &gate_view);
        if (status != LM_OK) return status;
        status = lm_tensor_make_q4_k_view(down_scratch, down_expert_bytes, 2u, down_dims, &down_view);
        if (status != LM_OK) return status;
        status = lm_cpu_matvec_q4_k(&gate_view, input, intermediate_size * 2u, hidden_size,
                                    gate_up_output.data());
        if (status != LM_OK) return status;
        for (uint32_t i = 0u; i < intermediate_size; ++i) {
            const float gate = gate_up_output[i];
            const float up = gate_up_output[intermediate_size + i];
            activation[i] = (gate / (1.0f + std::exp(-gate))) * up;
            if (!std::isfinite(activation[i])) return LM_ERR_RANGE;
        }
        float *output = selected_outputs + static_cast<size_t>(slot) * hidden_size;
        status = lm_cpu_matvec_q4_k(&down_view, activation.data(), hidden_size, intermediate_size, output);
        if (status != LM_OK) return status;
        if (!finite_array(output, hidden_size)) return LM_ERR_RANGE;
    }
    return LM_OK;
}

static lm_status validate_q4_k_binding_matvec(const lm_model_tensor_binding *binding,
                                                uint32_t rows, uint32_t columns,
                                                uint64_t *payload_bytes, uint32_t *blocks_per_row) {
    if (!binding || !payload_bytes || !blocks_per_row || rows == 0u || columns == 0u || columns % 256u != 0u ||
        binding->quant_format != LM_QUANT_GGML_Q4_K || binding->descriptor.rank != 2u ||
        binding->descriptor.dims[0] != rows || binding->descriptor.dims[1] != columns)
        return LM_ERR_ARGUMENT;
    const uint64_t blocks = static_cast<uint64_t>(columns) / 256u;
    const uint64_t row_bytes = blocks * 144u;
    const uint64_t expected = row_bytes * static_cast<uint64_t>(rows);
    if (blocks > UINT32_MAX || row_bytes == 0u || expected != binding->span.bytes) return LM_ERR_CAPACITY;
    *payload_bytes = expected;
    *blocks_per_row = static_cast<uint32_t>(blocks);
    return LM_OK;
}

lm_status lm_model_tensor_binding_matvec_q4_k_cpu(const lm_model_tensor_binding *binding,
                                                  void *packed_scratch, uint64_t scratch_bytes,
                                                  uint32_t rows, uint32_t columns,
                                                  const float *input, float *out) {
    if (!packed_scratch || !input || !out) return LM_ERR_ARGUMENT;
    uint64_t payload_bytes = 0u;
    uint32_t blocks_per_row = 0u;
    const lm_status valid = validate_q4_k_binding_matvec(binding, rows, columns, &payload_bytes, &blocks_per_row);
    if (valid != LM_OK) return valid;
    if (scratch_bytes < payload_bytes) return LM_ERR_CAPACITY;
    lm_tensor tensor{};
    const lm_status read = lm_model_tensor_binding_read(binding, packed_scratch, payload_bytes, &tensor);
    if (read != LM_OK) return read;
    return lm_cpu_matvec_q4_k(&tensor, input, rows, columns, out);
}

lm_status lm_model_tensor_binding_matvec_q4_k_vulkan(const lm_model_tensor_binding *binding,
                                                     void *packed_scratch, uint64_t scratch_bytes,
                                                     uint32_t rows, uint32_t columns,
                                                     const char *spv_path, uint32_t device_index,
                                                     const float *input, float *out) {
    if (!packed_scratch || !input || !out || !spv_path) return LM_ERR_ARGUMENT;
    uint64_t payload_bytes = 0u;
    uint32_t blocks_per_row = 0u;
    const lm_status valid = validate_q4_k_binding_matvec(binding, rows, columns, &payload_bytes, &blocks_per_row);
    if (valid != LM_OK) return valid;
    if (scratch_bytes < payload_bytes) return LM_ERR_CAPACITY;
    const lm_status read = lm_file_span_read(&binding->span, 0u, packed_scratch, static_cast<size_t>(payload_bytes));
    if (read != LM_OK) return read;
    return lm_vulkan_matvec_q4_k(spv_path, device_index, packed_scratch, rows,
                                 blocks_per_row, input, out);
}

static lm_status validate_q8_0_binding_matvec(const lm_model_tensor_binding *binding,
                                                uint32_t rows, uint32_t columns,
                                                uint64_t *payload_bytes, uint32_t *blocks_per_row) {
    if (!binding || !payload_bytes || !blocks_per_row || rows == 0u || columns == 0u || columns % 32u != 0u ||
        binding->quant_format != LM_QUANT_GGML_Q8_0 || binding->descriptor.rank != 2u ||
        binding->descriptor.dims[0] != rows || binding->descriptor.dims[1] != columns)
        return LM_ERR_ARGUMENT;
    const uint64_t blocks = static_cast<uint64_t>(columns) / 32u;
    const uint64_t row_bytes = blocks * 34u;
    const uint64_t expected = row_bytes * static_cast<uint64_t>(rows);
    if (blocks > UINT32_MAX || row_bytes == 0u || expected != binding->span.bytes) return LM_ERR_CAPACITY;
    *payload_bytes = expected;
    *blocks_per_row = static_cast<uint32_t>(blocks);
    return LM_OK;
}

lm_status lm_model_tensor_binding_matvec_q8_0_cpu(const lm_model_tensor_binding *binding,
                                                  void *packed_scratch, uint64_t scratch_bytes,
                                                  uint32_t rows, uint32_t columns,
                                                  const float *input, float *out) {
    if (!packed_scratch || !input || !out) return LM_ERR_ARGUMENT;
    uint64_t payload_bytes = 0u;
    uint32_t blocks_per_row = 0u;
    const lm_status valid = validate_q8_0_binding_matvec(binding, rows, columns,
                                                          &payload_bytes, &blocks_per_row);
    if (valid != LM_OK) return valid;
    if (scratch_bytes < payload_bytes) return LM_ERR_CAPACITY;
    lm_tensor tensor{};
    const lm_status read = lm_model_tensor_binding_read(binding, packed_scratch, payload_bytes, &tensor);
    if (read != LM_OK) return read;
    return lm_cpu_matvec_q8_0(&tensor, input, rows, columns, out);
}

lm_status lm_model_tensor_binding_matvec_q8_0_vulkan(const lm_model_tensor_binding *binding,
                                                     void *packed_scratch, uint64_t scratch_bytes,
                                                     uint32_t rows, uint32_t columns,
                                                     const char *spv_path, uint32_t device_index,
                                                     const float *input, float *out) {
    if (!packed_scratch || !input || !out || !spv_path) return LM_ERR_ARGUMENT;
    uint64_t payload_bytes = 0u;
    uint32_t blocks_per_row = 0u;
    const lm_status valid = validate_q8_0_binding_matvec(binding, rows, columns,
                                                          &payload_bytes, &blocks_per_row);
    if (valid != LM_OK) return valid;
    if (scratch_bytes < payload_bytes) return LM_ERR_CAPACITY;
    const lm_status read = lm_file_span_read(&binding->span, 0u, packed_scratch,
                                             static_cast<size_t>(payload_bytes));
    if (read != LM_OK) return read;
    return lm_vulkan_matvec_q8_0(spv_path, device_index, packed_scratch,
                                 rows, blocks_per_row, input, out);
}

static lm_status validate_q8_0_binding_dot(const lm_model_tensor_binding *binding,
                                             uint64_t elements, uint64_t *payload_bytes,
                                             uint32_t *blocks) {
    if (!binding || !payload_bytes || !blocks || elements == 0u || elements % 32u != 0u ||
        binding->quant_format != LM_QUANT_GGML_Q8_0 || binding->elements != elements)
        return LM_ERR_ARGUMENT;
    const uint64_t block_count = elements / 32u;
    const uint64_t expected = block_count * 34u;
    if (block_count > UINT32_MAX || expected != binding->span.bytes) return LM_ERR_CAPACITY;
    *payload_bytes = expected;
    *blocks = static_cast<uint32_t>(block_count);
    return LM_OK;
}

lm_status lm_model_tensor_binding_dot_q8_0_cpu(const lm_model_tensor_binding *binding,
                                               void *packed_scratch, uint64_t scratch_bytes,
                                               const float *input, uint64_t elements, float *out) {
    if (!packed_scratch || !input || !out) return LM_ERR_ARGUMENT;
    uint64_t payload_bytes = 0u;
    uint32_t blocks = 0u;
    const lm_status valid = validate_q8_0_binding_dot(binding, elements, &payload_bytes, &blocks);
    if (valid != LM_OK) return valid;
    if (scratch_bytes < payload_bytes) return LM_ERR_CAPACITY;
    lm_tensor tensor{};
    const lm_status read = lm_model_tensor_binding_read(binding, packed_scratch, payload_bytes, &tensor);
    if (read != LM_OK) return read;
    return lm_cpu_dot_q8_0(&tensor, input, elements, out);
}

lm_status lm_model_tensor_binding_dot_q8_0_vulkan(const lm_model_tensor_binding *binding,
                                                  void *packed_scratch, uint64_t scratch_bytes,
                                                  const char *spv_path, uint32_t device_index,
                                                  const float *input, uint64_t elements, float *out) {
    if (!packed_scratch || !input || !out || !spv_path) return LM_ERR_ARGUMENT;
    uint64_t payload_bytes = 0u;
    uint32_t blocks = 0u;
    const lm_status valid = validate_q8_0_binding_dot(binding, elements, &payload_bytes, &blocks);
    if (valid != LM_OK) return valid;
    if (scratch_bytes < payload_bytes) return LM_ERR_CAPACITY;
    const lm_status read = lm_file_span_read(&binding->span, 0u, packed_scratch,
                                             static_cast<size_t>(payload_bytes));
    if (read != LM_OK) return read;
    return lm_vulkan_dot_q8_0(spv_path, device_index, packed_scratch, blocks, input, out);
}

lm_status lm_model_tensor_matvec_native(const lm_model_file *model, uint64_t tensor_index,
                                        const lm_native_matvec_config *config,
                                        void *packed_scratch, uint64_t scratch_bytes,
                                        uint32_t rows, uint32_t columns,
                                        const float *input, float *out) {
    if (!model || !config || !packed_scratch || !input || !out) return LM_ERR_ARGUMENT;
    lm_model_tensor_binding binding{};
    const lm_status bound = lm_model_tensor_bind_native(model, tensor_index, &binding);
    if (bound != LM_OK) return bound;
    lm_backend_kind backend = config->backend;
    if (backend == LM_BACKEND_AUTO) {
        uint32_t device_count = 0u;
        backend = (lm_vulkan_device_count(&device_count) == LM_OK && device_count != 0u)
                      ? LM_BACKEND_VULKAN : LM_BACKEND_CPU;
    }
    if (backend != LM_BACKEND_CPU && backend != LM_BACKEND_VULKAN) return LM_ERR_UNSUPPORTED;
    if (binding.quant_format == LM_QUANT_GGML_Q4_K) {
        if (backend == LM_BACKEND_CPU)
            return lm_model_tensor_binding_matvec_q4_k_cpu(&binding, packed_scratch, scratch_bytes,
                                                           rows, columns, input, out);
        if (!config->shader_path) return LM_ERR_ARGUMENT;
        return lm_model_tensor_binding_matvec_q4_k_vulkan(&binding, packed_scratch, scratch_bytes,
                                                          rows, columns, config->shader_path,
                                                          config->device_index, input, out);
    }
    if (binding.quant_format == LM_QUANT_GGML_Q8_0) {
        if (backend == LM_BACKEND_CPU)
            return lm_model_tensor_binding_matvec_q8_0_cpu(&binding, packed_scratch, scratch_bytes,
                                                           rows, columns, input, out);
        if (!config->shader_path) return LM_ERR_ARGUMENT;
        return lm_model_tensor_binding_matvec_q8_0_vulkan(&binding, packed_scratch, scratch_bytes,
                                                          rows, columns, config->shader_path,
                                                          config->device_index, input, out);
    }
    return LM_ERR_UNSUPPORTED;
}

lm_status lm_model_build_llama_graph(const lm_model_file *model,
                                     lm_decoder_graph_binding *out_binding) {
    if (!model || !out_binding || model->tensors.empty()) return LM_ERR_ARGUMENT;
    std::memset(out_binding, 0xff, sizeof(*out_binding));
    out_binding->layer_count = 0u;
    uint32_t global_mask = 0u;
    uint32_t layer_masks[LM_DECODER_PLAN_MAX_LAYERS] = {};
    for (size_t i = 0u; i < model->tensors.size(); ++i) {
        lm_decoder_tensor_mapping mapping{};
        const lm_status mapped = lm_decoder_map_llama_tensor(&model->tensors[i], &mapping);
        if (mapped != LM_OK) return mapped;
        const uint64_t index = static_cast<uint64_t>(i);
        if (mapping.role <= LM_DECODER_TENSOR_OUTPUT_NORM) {
            const uint32_t bit = 1u << mapping.role;
            if ((global_mask & bit) != 0u) return LM_ERR_PARSE;
            global_mask |= bit;
            if (mapping.role == LM_DECODER_TENSOR_TOKEN_EMBEDDING) out_binding->token_embedding = index;
            else if (mapping.role == LM_DECODER_TENSOR_OUTPUT) out_binding->output = index;
            else out_binding->output_norm = index;
        } else {
            if (mapping.layer_index >= LM_DECODER_PLAN_MAX_LAYERS) return LM_ERR_CAPACITY;
            const uint32_t bit = 1u << mapping.role;
            if ((layer_masks[mapping.layer_index] & bit) != 0u) return LM_ERR_PARSE;
            layer_masks[mapping.layer_index] |= bit;
            lm_decoder_layer_binding &layer = out_binding->layers[mapping.layer_index];
            if (mapping.role == LM_DECODER_TENSOR_ATTN_NORM) layer.attn_norm = index;
            else if (mapping.role == LM_DECODER_TENSOR_ATTN_Q) layer.attn_q = index;
            else if (mapping.role == LM_DECODER_TENSOR_ATTN_K) layer.attn_k = index;
            else if (mapping.role == LM_DECODER_TENSOR_ATTN_V) layer.attn_v = index;
            else if (mapping.role == LM_DECODER_TENSOR_ATTN_OUTPUT) layer.attn_output = index;
            else if (mapping.role == LM_DECODER_TENSOR_FFN_NORM) layer.ffn_norm = index;
            else if (mapping.role == LM_DECODER_TENSOR_FFN_GATE) layer.ffn_gate = index;
            else if (mapping.role == LM_DECODER_TENSOR_FFN_DOWN) layer.ffn_down = index;
            else if (mapping.role == LM_DECODER_TENSOR_FFN_UP) layer.ffn_up = index;
            if (mapping.layer_index + 1u > out_binding->layer_count)
                out_binding->layer_count = mapping.layer_index + 1u;
        }
    }
    const uint32_t global_required = 7u;
    const uint32_t layer_required = 0xff8u;
    if (global_mask != global_required || out_binding->layer_count == 0u) return LM_ERR_UNSUPPORTED;
    for (uint32_t layer = 0u; layer < out_binding->layer_count; ++layer)
        if (layer_masks[layer] != layer_required) return LM_ERR_UNSUPPORTED;
    return LM_OK;
}

lm_status lm_cpu_dot_f32(const float *a, const float *b, size_t count, float *out) {
    if (!a || !b || !out || count == 0u) return LM_ERR_ARGUMENT;
    float sum = 0.0f;
    for (size_t i = 0u; i < count; ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return LM_ERR_RANGE;
        sum += a[i] * b[i];
    }
    if (!std::isfinite(sum)) return LM_ERR_RANGE;
    *out = sum;
    return LM_OK;
}

float half_to_float(uint16_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16u;
    const uint32_t exponent = (bits >> 10u) & 0x1fu;
    const uint32_t fraction = bits & 0x03ffu;
    uint32_t value = 0u;
    if (exponent == 0u) {
        if (fraction == 0u) value = sign;
        else {
            uint32_t normalized = fraction;
            uint32_t shift = 0u;
            while ((normalized & 0x0400u) == 0u) { normalized <<= 1u; ++shift; }
            value = sign | ((127u - 15u - shift) << 23u) | ((normalized & 0x03ffu) << 13u);
        }
    } else if (exponent == 0x1fu) value = sign | 0x7f800000u | (fraction << 13u);
    else value = sign | ((exponent + 112u) << 23u) | (fraction << 13u);
    float result = 0.0f;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

lm_status lm_cpu_dot_q4_0(const lm_tensor *weights, const float *input,
                          uint64_t elements, float *out) {
    if (!weights || !input || !out || elements == 0u || elements % 32u != 0u ||
        weights->quant_format != LM_QUANT_GGML_Q4_0 || weights->dtype != LM_DTYPE_U8)
        return LM_ERR_ARGUMENT;
    if (lm_tensor_validate(weights) != LM_OK) return LM_ERR_RANGE;
    const uint64_t required_bytes = (elements / 32u) * 18u;
    if (required_bytes != weights->bytes) return LM_ERR_CAPACITY;
    const unsigned char *packed = static_cast<const unsigned char *>(weights->data);
    float sum = 0.0f;
    for (uint64_t block = 0u; block < elements / 32u; ++block) {
        const size_t base = static_cast<size_t>(block * 18u);
        const uint16_t scale_bits = static_cast<uint16_t>(packed[base]) | static_cast<uint16_t>(packed[base + 1u] << 8u);
        const float scale = half_to_float(scale_bits);
        if (!std::isfinite(scale)) return LM_ERR_RANGE;
        for (uint32_t i = 0u; i < 32u; ++i) {
            const float value = input[block * 32u + i];
            if (!std::isfinite(value)) return LM_ERR_RANGE;
            const unsigned char packed_pair = packed[base + 2u + i / 2u];
            const int quantized = static_cast<int>((i & 1u) == 0u ? (packed_pair & 0x0fu) : (packed_pair >> 4u)) - 8;
            sum += scale * static_cast<float>(quantized) * value;
        }
    }
    if (!std::isfinite(sum)) return LM_ERR_RANGE;
    *out = sum;
    return LM_OK;
}

lm_status lm_cpu_dot_q8_0(const lm_tensor *weights, const float *input,
                          uint64_t elements, float *out) {
    if (!weights || !input || !out || elements == 0u || elements % 32u != 0u ||
        weights->quant_format != LM_QUANT_GGML_Q8_0 || weights->dtype != LM_DTYPE_U8)
        return LM_ERR_ARGUMENT;
    if (lm_tensor_validate(weights) != LM_OK) return LM_ERR_RANGE;
    const uint64_t required_bytes = (elements / 32u) * 34u;
    if (required_bytes != weights->bytes) return LM_ERR_CAPACITY;
    const unsigned char *packed = static_cast<const unsigned char *>(weights->data);
    float sum = 0.0f;
    for (uint64_t block = 0u; block < elements / 32u; ++block) {
        const size_t base = static_cast<size_t>(block * 34u);
        const uint16_t scale_bits = static_cast<uint16_t>(packed[base]) | static_cast<uint16_t>(packed[base + 1u] << 8u);
        const float scale = half_to_float(scale_bits);
        if (!std::isfinite(scale)) return LM_ERR_RANGE;
        for (uint32_t i = 0u; i < 32u; ++i) {
            const float value = input[block * 32u + i];
            if (!std::isfinite(value)) return LM_ERR_RANGE;
            const int8_t quantized = static_cast<int8_t>(packed[base + 2u + i]);
            sum += scale * static_cast<float>(quantized) * value;
        }
    }
    if (!std::isfinite(sum)) return LM_ERR_RANGE;
    *out = sum;
    return LM_OK;
}

void q4_k_scale_min(unsigned index, const unsigned char *scales, unsigned char *scale, unsigned char *minimum) {
    if (index < 4u) {
        *scale = static_cast<unsigned char>(scales[index] & 63u);
        *minimum = static_cast<unsigned char>(scales[index + 4u] & 63u);
    } else {
        *scale = static_cast<unsigned char>((scales[index + 4u] & 15u) | ((scales[index - 4u] >> 6u) << 4u));
        *minimum = static_cast<unsigned char>((scales[index + 4u] >> 4u) | ((scales[index] >> 6u) << 4u));
    }
}

lm_status lm_cpu_dot_q4_k(const lm_tensor *weights, const float *input,
                          uint64_t elements, float *out) {
    if (!weights || !input || !out || elements == 0u || elements % 256u != 0u ||
        weights->quant_format != LM_QUANT_GGML_Q4_K || weights->dtype != LM_DTYPE_U8)
        return LM_ERR_ARGUMENT;
    if (lm_tensor_validate(weights) != LM_OK) return LM_ERR_RANGE;
    const uint64_t required_bytes = (elements / 256u) * 144u;
    if (required_bytes != weights->bytes) return LM_ERR_CAPACITY;
    const unsigned char *packed = static_cast<const unsigned char *>(weights->data);
    float sum = 0.0f;
    for (uint64_t block = 0u; block < elements / 256u; ++block) {
        const size_t base = static_cast<size_t>(block * 144u);
        const float d = half_to_float(static_cast<uint16_t>(packed[base]) | static_cast<uint16_t>(packed[base + 1u] << 8u));
        const float minimum_scale = half_to_float(static_cast<uint16_t>(packed[base + 2u]) | static_cast<uint16_t>(packed[base + 3u] << 8u));
        if (!std::isfinite(d) || !std::isfinite(minimum_scale)) return LM_ERR_RANGE;
        const unsigned char *scales = packed + base + 4u;
        const unsigned char *quantized = packed + base + 16u;
        unsigned scale_index = 0u;
        for (unsigned group = 0u; group < 4u; ++group) {
            unsigned char scale = 0u;
            unsigned char minimum = 0u;
            q4_k_scale_min(scale_index++, scales, &scale, &minimum);
            const float d1 = d * static_cast<float>(scale);
            const float m1 = minimum_scale * static_cast<float>(minimum);
            q4_k_scale_min(scale_index++, scales, &scale, &minimum);
            const float d2 = d * static_cast<float>(scale);
            const float m2 = minimum_scale * static_cast<float>(minimum);
            for (unsigned i = 0u; i < 32u; ++i) {
                const float value = input[block * 256u + group * 64u + i];
                if (!std::isfinite(value)) return LM_ERR_RANGE;
                sum += (d1 * static_cast<float>(quantized[group * 32u + i] & 15u) - m1) * value;
            }
            for (unsigned i = 0u; i < 32u; ++i) {
                const float value = input[block * 256u + group * 64u + 32u + i];
                if (!std::isfinite(value)) return LM_ERR_RANGE;
                sum += (d2 * static_cast<float>(quantized[group * 32u + i] >> 4u) - m2) * value;
            }
        }
    }
    if (!std::isfinite(sum)) return LM_ERR_RANGE;
    *out = sum;
    return LM_OK;
}

lm_status lm_cpu_matvec_q4_k(const lm_tensor *weights, const float *input,
                             uint32_t rows, uint32_t columns, float *out) {
    if (!weights || !input || !out || rows == 0u || columns == 0u || columns % 256u != 0u ||
        weights->quant_format != LM_QUANT_GGML_Q4_K || weights->dtype != LM_DTYPE_U8 ||
        weights->rank != 2u || weights->dims[0] != rows || weights->dims[1] != columns)
        return LM_ERR_ARGUMENT;
    if (lm_tensor_validate(weights) != LM_OK) return LM_ERR_RANGE;
    const uint64_t row_bytes = (static_cast<uint64_t>(columns) / 256u) * 144u;
    if (row_bytes > std::numeric_limits<size_t>::max() ||
        static_cast<uint64_t>(rows) > std::numeric_limits<uint64_t>::max() / row_bytes ||
        weights->bytes != static_cast<uint64_t>(rows) * row_bytes)
        return LM_ERR_CAPACITY;
    for (uint32_t row = 0u; row < rows; ++row) {
        lm_tensor row_tensor{};
        const uint32_t row_dims[1] = {columns};
        unsigned char *row_data = static_cast<unsigned char *>(weights->data) + static_cast<size_t>(row) * static_cast<size_t>(row_bytes);
        const lm_status view_status = lm_tensor_make_q4_k_view(row_data, row_bytes, 1u, row_dims, &row_tensor);
        if (view_status != LM_OK) return view_status;
        const lm_status dot_status = lm_cpu_dot_q4_k(&row_tensor, input, columns, &out[row]);
        if (dot_status != LM_OK) return dot_status;
    }
    return LM_OK;
}

lm_status lm_cpu_matvec_q8_0(const lm_tensor *weights, const float *input,
                             uint32_t rows, uint32_t columns, float *out) {
    if (!weights || !input || !out || rows == 0u || columns == 0u || columns % 32u != 0u ||
        weights->quant_format != LM_QUANT_GGML_Q8_0 || weights->dtype != LM_DTYPE_U8 ||
        weights->rank != 2u || weights->dims[0] != rows || weights->dims[1] != columns)
        return LM_ERR_ARGUMENT;
    if (lm_tensor_validate(weights) != LM_OK) return LM_ERR_RANGE;
    const uint64_t row_bytes = (static_cast<uint64_t>(columns) / 32u) * 34u;
    if (row_bytes > std::numeric_limits<size_t>::max() ||
        static_cast<uint64_t>(rows) > std::numeric_limits<uint64_t>::max() / row_bytes ||
        weights->bytes != static_cast<uint64_t>(rows) * row_bytes)
        return LM_ERR_CAPACITY;
    for (uint32_t row = 0u; row < rows; ++row) {
        lm_tensor row_tensor{};
        const uint32_t row_dims[1] = {columns};
        unsigned char *row_data = static_cast<unsigned char *>(weights->data) +
                                   static_cast<size_t>(row) * static_cast<size_t>(row_bytes);
        const lm_status view_status = lm_tensor_make_q8_0_view(row_data, row_bytes, 1u, row_dims, &row_tensor);
        if (view_status != LM_OK) return view_status;
        const lm_status dot_status = lm_cpu_dot_q8_0(&row_tensor, input, columns, &out[row]);
        if (dot_status != LM_OK) return dot_status;
    }
    return LM_OK;
}

lm_status lm_cpu_softmax_f32(const float *input, float *output, size_t count) {
    if (!input || !output || count == 0u) return LM_ERR_ARGUMENT;
    float max_value = -std::numeric_limits<float>::infinity();
    for (size_t i = 0u; i < count; ++i) {
        if (!std::isfinite(input[i])) return LM_ERR_RANGE;
        max_value = std::max(max_value, input[i]);
    }
    float sum = 0.0f;
    for (size_t i = 0u; i < count; ++i) {
        output[i] = std::exp(input[i] - max_value);
        sum += output[i];
    }
    if (!(sum > 0.0f) || !std::isfinite(sum)) return LM_ERR_RANGE;
    for (size_t i = 0u; i < count; ++i) output[i] /= sum;
    return LM_OK;
}
