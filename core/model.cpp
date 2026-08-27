#include "lm/lm.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <new>
#include <string>
#include <vector>

float half_to_float(uint16_t bits);
lm_status lm_tokenizer_create_gpt2(const std::vector<std::string> &vocabulary,
                                   const std::vector<std::string> &merge_strings,
                                   uint8_t add_bos, uint32_t bos_id,
                                   lm_tokenizer **out_tokenizer);

struct lm_model_file {
    lm_file *file;
    lm_file_shard_set *shards;
    lm_model_info info;
    lm_model_architecture architecture;
    uint8_t has_architecture;
    uint8_t hf_tied_output;
    std::vector<lm_model_tensor_info> tensors;
    std::vector<std::string> tokens;
    std::vector<uint64_t> shard_headers;
    std::vector<uint64_t> shard_sizes;
    lm_tokenizer *tokenizer = nullptr;
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

bool parse_mixtral_router_name(const char *name, uint32_t *layer) {
    if (!name || !layer) return false;
    const char prefix[] = "blk.";
    const char suffix[] = ".ffn_router.weight";
    const size_t prefix_bytes = sizeof(prefix) - 1u;
    const size_t suffix_bytes = sizeof(suffix) - 1u;
    if (std::strncmp(name, prefix, prefix_bytes) != 0) return false;
    const char *cursor = name + prefix_bytes;
    if (*cursor < '0' || *cursor > '9') return false;
    uint64_t value = 0u;
    while (*cursor >= '0' && *cursor <= '9') {
        const uint64_t digit = static_cast<uint64_t>(*cursor - '0');
        if (value > (UINT32_MAX - digit) / 10u) return false;
        value = value * 10u + digit;
        ++cursor;
    }
    if (std::strncmp(cursor, suffix, suffix_bytes) != 0 || cursor[suffix_bytes] != '\0') return false;
    *layer = static_cast<uint32_t>(value);
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
    bool u16(uint16_t *value) { unsigned char bytes[2] = {}; if (!read(bytes, 2u)) return false; *value = static_cast<uint16_t>(bytes[0] | (static_cast<uint16_t>(bytes[1]) << 8u)); return true; }
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

struct GgufSplitInfo {
    uint64_t count = 1u;
    uint64_t number = 0u;
    uint64_t tensor_count = 0u;
    bool has_count = false;
    bool has_number = false;
    bool has_tensor_count = false;
};

bool read_gguf_split_value(BinaryReader &reader, uint32_t type, uint64_t *value) {
    if (!value) return false;
    if (type == 2u) { uint16_t parsed = 0u; if (!reader.u16(&parsed)) return false; *value = parsed; return true; }
    if (type == 4u) { uint32_t parsed = 0u; if (!reader.u32(&parsed)) return false; *value = parsed; return true; }
    if (type == 10u) return reader.u64(value);
    return false;
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

bool read_string_array(BinaryReader &reader, uint32_t type, std::vector<std::string> *strings) {
    if (type != 9u || !strings) return false;
    uint32_t element_type = 0u;
    uint64_t length = 0u;
    if (!reader.u32(&element_type) || !reader.u64(&length) || element_type != 8u ||
        length > kMaxContainerItems) return false;
    std::vector<std::string> parsed;
    try {
        parsed.reserve(static_cast<size_t>(length));
        for (uint64_t i = 0u; i < length; ++i) {
            std::string value;
            if (!reader.string(&value, kMaxVocabularyTokenBytes)) return false;
            parsed.push_back(std::move(value));
        }
    } catch (const std::bad_alloc &) { return false; }
    *strings = std::move(parsed);
    return true;
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
                       lm_model_architecture *out_architecture = nullptr,
                       GgufSplitInfo *out_split = nullptr,
                       std::vector<std::string> *out_merges = nullptr,
                       bool *out_gpt2_tokenizer = nullptr) {
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
    GgufSplitInfo split;
    std::vector<std::string> tokens;
    std::vector<std::string> merges;
    bool gpt2_tokenizer = false;
    bool smollm_pre = false;
    bool add_bos_token = false;
    bool add_space_prefix = false;
    lm_model_architecture architecture{};
    architecture.rms_epsilon = 1.0e-5f;
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
        } else if (key == "split.count") {
            if (split.has_count || !read_gguf_split_value(reader, type, &split.count) || split.count == 0u) {
                set_error(error_text, error_capacity, "invalid GGUF split count metadata"); return LM_ERR_PARSE;
            }
            split.has_count = true;
        } else if (key == "split.no") {
            if (split.has_number || !read_gguf_split_value(reader, type, &split.number)) {
                set_error(error_text, error_capacity, "invalid GGUF split number metadata"); return LM_ERR_PARSE;
            }
            split.has_number = true;
        } else if (key == "split.tensors.count") {
            if (split.has_tensor_count || !read_gguf_split_value(reader, type, &split.tensor_count) || split.tensor_count == 0u) {
                set_error(error_text, error_capacity, "invalid GGUF split tensor count metadata"); return LM_ERR_PARSE;
            }
            split.has_tensor_count = true;
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
        } else if (key == "tokenizer.ggml.merges") {
            if (!merges.empty() || !read_string_array(reader, type, &merges)) {
                set_error(error_text, error_capacity, "invalid GGUF tokenizer merge metadata"); return LM_ERR_PARSE;
            }
        } else if (key == "tokenizer.ggml.model") {
            std::string model_name;
            if (type != 8u || !reader.string(&model_name, 64u)) {
                set_error(error_text, error_capacity, "invalid GGUF tokenizer model metadata"); return LM_ERR_PARSE;
            }
            gpt2_tokenizer = model_name == "gpt2";
        } else if (key == "tokenizer.ggml.pre") {
            std::string pre_name;
            if (type != 8u || !reader.string(&pre_name, 64u)) {
                set_error(error_text, error_capacity, "invalid GGUF tokenizer preprocessor metadata"); return LM_ERR_PARSE;
            }
            smollm_pre = pre_name == "smollm";
        } else if (key == "tokenizer.ggml.add_bos_token") {
            uint8_t value = 0u;
            if (type != 7u || !reader.u8(&value)) return LM_ERR_PARSE;
            add_bos_token = value != 0u;
        } else if (key == "tokenizer.ggml.add_space_prefix") {
            uint8_t value = 0u;
            if (type != 7u || !reader.u8(&value)) return LM_ERR_PARSE;
            add_space_prefix = value != 0u;
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
        } else if (key == "llama.attention.layer_norm_rms_epsilon") {
            if (!read_arch_float(reader, type, &architecture.rms_epsilon)) return LM_ERR_PARSE;
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
    if (split.count == 0u || split.number >= split.count || (split.has_count != split.has_number) ||
        (split.has_tensor_count && split.tensor_count < tensor_count)) {
        set_error(error_text, error_capacity, "invalid GGUF split metadata"); return LM_ERR_PARSE;
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
    const bool use_gpt2_tokenizer = gpt2_tokenizer && smollm_pre &&
        !add_bos_token && !add_space_prefix && !merges.empty();
    if (out_tensors) *out_tensors = std::move(parsed_tensors);
    if (out_tokens) *out_tokens = std::move(tokens);
    if (out_merges) *out_merges = std::move(merges);
    if (out_gpt2_tokenizer) *out_gpt2_tokenizer = use_gpt2_tokenizer;
    out_info->format = LM_MODEL_GGUF;
    out_info->version = version;
    out_info->file_bytes = reader.size();
    out_info->header_bytes = tensor_data_start;
    out_info->tensor_count = tensor_count;
    out_info->expert_count = has_expert_count ? static_cast<uint32_t>(expert_count) : 0u;
    out_info->experts_per_token = has_experts_per_token ? static_cast<uint32_t>(experts_per_token) : 0u;
    if (out_split) *out_split = split;
    return LM_OK;
}

class JsonCursor {
public:
    JsonCursor(const char *begin, const char *end) : p_(begin), end_(end) {}
    const char *position() const { return p_; }
    const char *end() const { return end_; }

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

    bool token(std::string *out, size_t max_length) {
        whitespace();
        if (p_ >= end_ || *p_ == ',' || *p_ == '}' || *p_ == ']') return false;
        const char *start = p_;
        while (p_ < end_ && *p_ != ',' && *p_ != '}' && *p_ != ']' &&
               *p_ != ' ' && *p_ != '\n' && *p_ != '\r' && *p_ != '\t') ++p_;
        if (p_ == start || static_cast<size_t>(p_ - start) > max_length) return false;
        out->assign(start, p_);
        return true;
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

bool parse_safe_u32(const std::string &text, uint32_t *out) {
    if (!out || text.empty()) return false;
    char *end = nullptr;
    const unsigned long long value = std::strtoull(text.c_str(), &end, 10);
    if (end != text.c_str() + text.size() || value == 0u || value > UINT32_MAX) return false;
    *out = static_cast<uint32_t>(value);
    return true;
}

bool parse_safe_float(const std::string &text, float *out) {
    if (!out || text.empty()) return false;
    char *end = nullptr;
    const float value = std::strtof(text.c_str(), &end);
    if (end != text.c_str() + text.size() || !std::isfinite(value) || value <= 0.0f) return false;
    *out = value;
    return true;
}

bool read_safetensors_metadata(JsonCursor &json, lm_model_architecture *architecture, bool *has_architecture,
                               std::vector<std::string> *tokens) {
    if (!architecture || !has_architecture || !json.character('{')) return false;
    *has_architecture = false;
    architecture->rms_epsilon = 1.0e-5f;
    bool is_llama = false;
    bool flags[7] = {};
    std::vector<uint8_t> token_seen;
    std::vector<std::string> ignored_tokens;
    std::vector<std::string> *token_output = tokens ? tokens : &ignored_tokens;
    json.whitespace();
    if (json.position() < json.end() && *json.position() != '}') {
        for (;;) {
            std::string key;
            if (!json.string(&key, 128u) || !json.character(':')) return false;
            const char *token_prefix = "tokenizer.token.";
            const size_t token_prefix_bytes = std::strlen(token_prefix);
            const bool tokenizer_key = key.size() > token_prefix_bytes &&
                                       key.compare(0u, token_prefix_bytes, token_prefix) == 0;
            const bool known = key == "general.architecture" || key == "llama.context_length" ||
                               key == "llama.embedding_length" || key == "llama.block_count" ||
                               key == "llama.attention.head_count" || key == "llama.attention.head_count_kv" ||
                               key == "llama.feed_forward_length" || key == "llama.rope.freq_base" || tokenizer_key;
            if (known) {
                std::string value;
                if (!json.string(&value, tokenizer_key ? kMaxVocabularyTokenBytes : 64u)) return false;
                if (tokenizer_key) {
                    char *end = nullptr;
                    const std::string suffix = key.substr(token_prefix_bytes);
                    const unsigned long long index = std::strtoull(suffix.c_str(), &end, 10);
                    if (suffix.empty() || end != suffix.c_str() + suffix.size() || index >= kMaxVocabularyTokens) return false;
                    if (token_output->size() <= static_cast<size_t>(index)) token_output->resize(static_cast<size_t>(index) + 1u);
                    if (token_seen.size() <= static_cast<size_t>(index)) token_seen.resize(static_cast<size_t>(index) + 1u, 0u);
                    if (token_seen[static_cast<size_t>(index)] != 0u) return false;
                    (*token_output)[static_cast<size_t>(index)] = value;
                    token_seen[static_cast<size_t>(index)] = 1u;
                } else if (key == "general.architecture") is_llama = value == "llama";
                else if (key == "llama.context_length") { if (flags[0] || !parse_safe_u32(value, &architecture->context_length)) return false; flags[0] = true; }
                else if (key == "llama.embedding_length") { if (flags[1] || !parse_safe_u32(value, &architecture->embedding_length)) return false; flags[1] = true; }
                else if (key == "llama.block_count") { if (flags[2] || !parse_safe_u32(value, &architecture->block_count)) return false; flags[2] = true; }
                else if (key == "llama.attention.head_count") { if (flags[3] || !parse_safe_u32(value, &architecture->head_count)) return false; flags[3] = true; }
                else if (key == "llama.attention.head_count_kv") { if (flags[4] || !parse_safe_u32(value, &architecture->head_count_kv)) return false; flags[4] = true; }
                else if (key == "llama.feed_forward_length") { if (flags[5] || !parse_safe_u32(value, &architecture->intermediate_length)) return false; flags[5] = true; }
                else if (key == "llama.rope.freq_base") { if (flags[6] || !parse_safe_float(value, &architecture->rope_frequency_base)) return false; flags[6] = true; }
            } else if (!json.skip_value(0u)) return false;
            json.whitespace();
            if (json.position() < json.end() && *json.position() == '}') { if (!json.consume('}')) return false; break; }
            if (!json.character(',')) return false;
        }
    } else if (!json.consume('}')) return false;
    for (size_t i = 0u; i < token_seen.size(); ++i) if (token_seen[i] == 0u) return false;
    *has_architecture = is_llama && flags[0] && flags[1] && flags[2] && flags[3] && flags[4] && flags[5] && flags[6] &&
                        architecture->head_count_kv <= architecture->head_count &&
                        architecture->embedding_length % architecture->head_count == 0u;
    return true;
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
                              std::vector<lm_model_tensor_info> *out_tensors = nullptr,
                              std::vector<std::string> *out_tokens = nullptr,
                              lm_model_architecture *out_architecture = nullptr) {
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
    if (out_architecture) *out_architecture = lm_model_architecture{};
    lm_model_architecture metadata_architecture{};
    bool has_metadata_architecture = false;
    if (!json.character('{')) { set_error(error_text, error_capacity, "SafeTensors header is not a JSON object"); return LM_ERR_PARSE; }
    struct Range { uint64_t begin; uint64_t end; };
    std::vector<Range> ranges;
    json.whitespace();
    if (json.position() < header.data() + header.size() && *json.position() != '}') {
        for (;;) {
            std::string name;
            if (!json.string(&name, 1u << 20u) || !json.character(':')) { set_error(error_text, error_capacity, "invalid SafeTensors key"); return LM_ERR_PARSE; }
            if (name == "__metadata__") {
                if (!read_safetensors_metadata(json, &metadata_architecture, &has_metadata_architecture, out_tokens)) {
                    set_error(error_text, error_capacity, "invalid SafeTensors metadata"); return LM_ERR_PARSE;
                }
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
    if (out_architecture && has_metadata_architecture) *out_architecture = metadata_architecture;
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
    std::vector<std::string> merges;
    bool gpt2_tokenizer = false;
    lm_model_architecture architecture{};
    if (info.format == LM_MODEL_GGUF) {
        const lm_status descriptors = inspect_gguf(path, &info, error_text, error_capacity, &tensors, &tokens, &architecture,
                                                    nullptr, &merges, &gpt2_tokenizer);
        if (descriptors != LM_OK) return descriptors;
    } else if (info.format == LM_MODEL_SAFETENSORS) {
        const lm_status descriptors = inspect_safetensors(path, &info, error_text, error_capacity, &tensors, &tokens, &architecture);
        if (descriptors != LM_OK) return descriptors;
    }
    const lm_status opened = lm_file_open(path, &file);
    if (opened != LM_OK) return opened;
    try {
        lm_model_file *model = new lm_model_file{file, nullptr, info, architecture,
                                                   static_cast<uint8_t>(architecture.block_count != 0u), 0u,
                                                   std::move(tensors), std::move(tokens),
                                                   std::vector<uint64_t>{info.header_bytes},
                                                   std::vector<uint64_t>{info.file_bytes}};
        if (gpt2_tokenizer) {
            const lm_status tokenizer_status = lm_tokenizer_create_gpt2(model->tokens, merges, 0u, 0u,
                                                                          &model->tokenizer);
            if (tokenizer_status != LM_OK) { lm_model_close(model); return tokenizer_status; }
        }
        *out_model = model;
        return LM_OK;
    } catch (const std::bad_alloc &) {
        lm_file_close(file);
        return LM_ERR_CAPACITY;
    }
}

static lm_status open_safetensors_sharded(const char *const *paths, size_t path_count,
                                          lm_model_file **out_model, char *error_text,
                                          size_t error_capacity) {
    lm_model_info combined{};
    std::vector<lm_model_tensor_info> tensors;
    std::vector<std::string> tokens;
    std::vector<uint64_t> headers;
    std::vector<uint64_t> sizes;
    lm_model_architecture architecture{};
    for (size_t shard = 0u; shard < path_count; ++shard) {
        if (!paths[shard]) return LM_ERR_ARGUMENT;
        lm_model_info part_info{};
        lm_model_architecture part_architecture{};
        std::vector<lm_model_tensor_info> part_tensors;
        std::vector<std::string> part_tokens;
        const lm_status parsed = inspect_safetensors(paths[shard], &part_info, error_text, error_capacity,
                                                      &part_tensors, &part_tokens, &part_architecture);
        if (parsed != LM_OK) return parsed;
        if (part_info.format != LM_MODEL_SAFETENSORS) {
            set_error(error_text, error_capacity, "mixed model formats in SafeTensors shard set"); return LM_ERR_PARSE;
        }
        if (shard == 0u) { combined = part_info; architecture = part_architecture; tokens = std::move(part_tokens); }
        else {
            const bool architecture_mismatch = architecture.context_length != part_architecture.context_length ||
                architecture.embedding_length != part_architecture.embedding_length || architecture.block_count != part_architecture.block_count ||
                architecture.head_count != part_architecture.head_count || architecture.head_count_kv != part_architecture.head_count_kv ||
                architecture.intermediate_length != part_architecture.intermediate_length || architecture.rope_frequency_base != part_architecture.rope_frequency_base;
            if (architecture_mismatch || part_tokens != tokens) {
                set_error(error_text, error_capacity, "inconsistent SafeTensors shard metadata"); return LM_ERR_PARSE;
            }
        }
        for (lm_model_tensor_info &tensor : part_tensors) {
            for (const lm_model_tensor_info &existing : tensors)
                if (std::strncmp(existing.name, tensor.name, sizeof(existing.name)) == 0) {
                    set_error(error_text, error_capacity, "duplicate tensor across SafeTensors shards"); return LM_ERR_PARSE;
                }
            tensor.shard_index = static_cast<uint32_t>(shard);
            tensors.push_back(tensor);
        }
        if (combined.file_bytes > UINT64_MAX - part_info.file_bytes) return LM_ERR_RANGE;
        if (shard != 0u) combined.file_bytes += part_info.file_bytes;
        headers.push_back(part_info.header_bytes); sizes.push_back(part_info.file_bytes);
    }
    if (tensors.empty() || tensors.size() > kMaxContainerItems) return LM_ERR_CAPACITY;
    lm_file_shard_set *shards = nullptr;
    const lm_status opened = lm_file_shard_set_open(paths, path_count, &shards);
    if (opened != LM_OK) return opened;
    combined.tensor_count = tensors.size();
    try {
        *out_model = new lm_model_file{nullptr, shards, combined, architecture,
                                       static_cast<uint8_t>(architecture.block_count != 0u), 0u,
                                       std::move(tensors), std::move(tokens),
                                       std::move(headers), std::move(sizes)};
        return LM_OK;
    } catch (const std::bad_alloc &) {
        lm_file_shard_set_close(shards); return LM_ERR_CAPACITY;
    }
}

lm_status lm_model_open_sharded(const char *const *paths, size_t path_count,
                                lm_model_file **out_model, char *error_text,
                                size_t error_capacity) {
    if (!paths || path_count == 0u || !out_model) return LM_ERR_ARGUMENT;
    *out_model = nullptr;
    if (path_count == 1u) return lm_model_open(paths[0], out_model, error_text, error_capacity);
    lm_model_info first_info{};
    const lm_status first_status = lm_model_inspect(paths[0], &first_info, error_text, error_capacity);
    if (first_status != LM_OK) return first_status;
    if (first_info.format == LM_MODEL_SAFETENSORS)
        return open_safetensors_sharded(paths, path_count, out_model, error_text, error_capacity);
    if (first_info.format != LM_MODEL_GGUF) return LM_ERR_UNSUPPORTED;
    lm_model_info combined{};
    std::vector<lm_model_tensor_info> tensors;
    std::vector<std::string> tokens;
    std::vector<std::string> merges;
    bool gpt2_tokenizer = false;
    std::vector<uint64_t> headers;
    std::vector<uint64_t> sizes;
    lm_model_architecture architecture{};
    GgufSplitInfo expected_split{};
    for (size_t shard = 0u; shard < path_count; ++shard) {
        if (!paths[shard]) return LM_ERR_ARGUMENT;
        lm_model_info part_info{};
        lm_model_architecture part_architecture{};
        GgufSplitInfo part_split{};
        std::vector<lm_model_tensor_info> part_tensors;
        std::vector<std::string> part_tokens;
        std::vector<std::string> part_merges;
        bool part_gpt2_tokenizer = false;
        const lm_status parsed = inspect_gguf(paths[shard], &part_info, error_text, error_capacity,
                                              &part_tensors, &part_tokens, &part_architecture, &part_split,
                                              &part_merges, &part_gpt2_tokenizer);
        if (parsed != LM_OK) return parsed;
        if (part_info.format != LM_MODEL_GGUF || !part_split.has_count || !part_split.has_number ||
            !part_split.has_tensor_count || part_split.count != path_count || part_split.number != shard ||
            part_split.tensor_count == 0u) {
            set_error(error_text, error_capacity, "invalid GGUF split membership"); return LM_ERR_PARSE;
        }
        if (shard == 0u) {
            combined = part_info;
            expected_split = part_split;
            architecture = part_architecture;
            tokens = std::move(part_tokens);
            merges = std::move(part_merges);
            gpt2_tokenizer = part_gpt2_tokenizer;
        } else {
            const bool architecture_mismatch = architecture.context_length != part_architecture.context_length ||
                architecture.embedding_length != part_architecture.embedding_length ||
                architecture.block_count != part_architecture.block_count ||
                architecture.head_count != part_architecture.head_count ||
                architecture.head_count_kv != part_architecture.head_count_kv ||
                architecture.intermediate_length != part_architecture.intermediate_length ||
                architecture.rope_frequency_base != part_architecture.rope_frequency_base;
            if (architecture_mismatch || part_tokens != tokens || part_merges != merges ||
                part_gpt2_tokenizer != gpt2_tokenizer) {
                set_error(error_text, error_capacity, "inconsistent GGUF split metadata"); return LM_ERR_PARSE;
            }
            if (part_split.tensor_count != expected_split.tensor_count) {
                set_error(error_text, error_capacity, "inconsistent GGUF split tensor count"); return LM_ERR_PARSE;
            }
        }
        for (lm_model_tensor_info &tensor : part_tensors) {
            for (const lm_model_tensor_info &existing : tensors) {
                if (std::strncmp(existing.name, tensor.name, sizeof(existing.name)) == 0) {
                    set_error(error_text, error_capacity, "duplicate tensor across GGUF splits"); return LM_ERR_PARSE;
                }
            }
            tensor.shard_index = static_cast<uint32_t>(shard);
            tensors.push_back(tensor);
        }
        if (combined.tensor_count > UINT64_MAX - part_info.tensor_count ||
            combined.file_bytes > UINT64_MAX - part_info.file_bytes)
            return LM_ERR_RANGE;
        if (shard != 0u) combined.tensor_count += part_info.tensor_count;
        combined.file_bytes += shard == 0u ? 0u : part_info.file_bytes;
        headers.push_back(part_info.header_bytes);
        sizes.push_back(part_info.file_bytes);
    }
    if (!expected_split.has_tensor_count || expected_split.tensor_count != tensors.size()) {
        set_error(error_text, error_capacity, "GGUF split tensor count does not match unified index"); return LM_ERR_PARSE;
    }
    if (tensors.empty() || tensors.size() > kMaxContainerItems) return LM_ERR_CAPACITY;
    lm_file_shard_set *shards = nullptr;
    const lm_status opened = lm_file_shard_set_open(paths, path_count, &shards);
    if (opened != LM_OK) return opened;
    combined.tensor_count = tensors.size();
    try {
        lm_model_file *model = new lm_model_file{nullptr, shards, combined, architecture,
                                                   static_cast<uint8_t>(architecture.block_count != 0u), 0u,
                                                   std::move(tensors), std::move(tokens),
                                                   std::move(headers), std::move(sizes)};
        if (gpt2_tokenizer) {
            const lm_status tokenizer_status = lm_tokenizer_create_gpt2(model->tokens, merges, 0u, 0u,
                                                                          &model->tokenizer);
            if (tokenizer_status != LM_OK) { lm_model_close(model); return tokenizer_status; }
        }
        *out_model = model;
        return LM_OK;
    } catch (const std::bad_alloc &) {
        lm_file_shard_set_close(shards);
        return LM_ERR_CAPACITY;
    }
}

void lm_model_close(lm_model_file *model) {
    if (!model) return;
    if (model->tokenizer) lm_tokenizer_destroy(model->tokenizer);
    if (model->shards) lm_file_shard_set_close(model->shards);
    else lm_file_close(model->file);
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

lm_status lm_model_set_tokenizer_json(lm_model_file *model, const char *path,
                                       char *error_text, size_t error_capacity) {
    if (!model || !path) return LM_ERR_ARGUMENT;
    lm_tokenizer *tokenizer = nullptr;
    const lm_status opened = lm_tokenizer_open_json(path, &tokenizer, error_text, error_capacity);
    if (opened != LM_OK) return opened;
    if (model->tokenizer) lm_tokenizer_destroy(model->tokenizer);
    model->tokenizer = tokenizer;
    return LM_OK;
}

lm_status lm_model_set_hf_config_json(lm_model_file *model, const char *path,
                                       char *error_text, size_t error_capacity) {
    if (!model || !path) return LM_ERR_ARGUMENT;
    if (model->info.format != LM_MODEL_SAFETENSORS) {
        set_error(error_text, error_capacity, "HF config attachment requires SafeTensors");
        return LM_ERR_UNSUPPORTED;
    }
    std::ifstream file(path, std::ios::binary);
    if (!file) { set_error(error_text, error_capacity, "cannot open config.json"); return LM_ERR_IO; }
    file.seekg(0, std::ios::end);
    const std::streamoff end = file.tellg();
    if (end <= 0 || static_cast<uint64_t>(end) > (1ull << 20u)) {
        set_error(error_text, error_capacity, "config.json is too large"); return LM_ERR_CAPACITY;
    }
    file.seekg(0, std::ios::beg);
    std::string source(static_cast<size_t>(end), '\0');
    if (!file.read(source.data(), static_cast<std::streamsize>(source.size()))) {
        set_error(error_text, error_capacity, "cannot read config.json"); return LM_ERR_IO;
    }
    JsonCursor json(source.data(), source.data() + source.size());
    if (!json.character('{')) { set_error(error_text, error_capacity, "config.json is not an object"); return LM_ERR_PARSE; }
    bool have_model_type = false, have_hidden = false, have_intermediate = false;
    bool have_layers = false, have_heads = false, have_rms = false, have_vocab = false;
    bool have_kv = false, have_context = false, have_rope = false;
    bool tie_word_embeddings = false;
    uint32_t hidden = 0u, intermediate = 0u, layers = 0u, heads = 0u, kv_heads = 0u, context = 2048u, vocab = 0u;
    float rms = 0.0f, rope = 10000.0f;
    json.whitespace();
    if (json.position() < json.end() && *json.position() != '}') {
        for (;;) {
            std::string key;
            if (!json.string(&key, 128u) || !json.character(':')) { set_error(error_text, error_capacity, "invalid config.json field"); return LM_ERR_PARSE; }
            auto read_u32 = [&json](uint32_t *out) {
                uint64_t value = 0u;
                return json.unsigned_number(&value) && value != 0u && value <= UINT32_MAX && ((*out = static_cast<uint32_t>(value)), true);
            };
            auto read_float = [&json](float *out) {
                std::string value;
                return json.token(&value, 64u) && parse_safe_float(value, out);
            };
            bool handled = true;
            if (key == "model_type") {
                std::string value; if (have_model_type || !json.string(&value, 32u) || value != "llama") { set_error(error_text, error_capacity, "config model_type must be llama"); return LM_ERR_UNSUPPORTED; } have_model_type = true;
            } else if (key == "hidden_size") { if (have_hidden || !read_u32(&hidden)) return LM_ERR_PARSE; have_hidden = true;
            } else if (key == "intermediate_size") { if (have_intermediate || !read_u32(&intermediate)) return LM_ERR_PARSE; have_intermediate = true;
            } else if (key == "num_hidden_layers") { if (have_layers || !read_u32(&layers)) return LM_ERR_PARSE; have_layers = true;
            } else if (key == "num_attention_heads") { if (have_heads || !read_u32(&heads)) return LM_ERR_PARSE; have_heads = true;
            } else if (key == "num_key_value_heads") { if (have_kv || !read_u32(&kv_heads)) return LM_ERR_PARSE; have_kv = true;
            } else if (key == "max_position_embeddings") { if (have_context || !read_u32(&context)) return LM_ERR_PARSE; have_context = true;
            } else if (key == "vocab_size") { if (have_vocab || !read_u32(&vocab)) return LM_ERR_PARSE; have_vocab = true;
            } else if (key == "rms_norm_eps") { if (have_rms || !read_float(&rms) || !(rms > 0.0f)) return LM_ERR_PARSE; have_rms = true;
            } else if (key == "rope_theta") { if (have_rope || !read_float(&rope) || !(rope > 0.0f)) return LM_ERR_PARSE; have_rope = true;
            } else if (key == "tie_word_embeddings") {
                std::string value; if (!json.token(&value, 8u) || (value != "true" && value != "false")) return LM_ERR_PARSE;
                tie_word_embeddings = value == "true";
            } else handled = false;
            if (!handled && !json.skip_value(0u)) { set_error(error_text, error_capacity, "invalid config.json value"); return LM_ERR_PARSE; }
            json.whitespace();
            if (json.position() < json.end() && *json.position() == '}') { if (!json.consume('}')) return LM_ERR_PARSE; break; }
            if (!json.character(',')) { set_error(error_text, error_capacity, "invalid config.json separator"); return LM_ERR_PARSE; }
        }
    } else if (!json.consume('}')) return LM_ERR_PARSE;
    json.whitespace();
    if (json.position() != json.end()) { set_error(error_text, error_capacity, "trailing config.json data"); return LM_ERR_PARSE; }
    if (!have_kv) kv_heads = heads;
    if (!have_model_type || !have_hidden || !have_intermediate || !have_layers || !have_heads || !have_rms || !have_vocab ||
        heads == 0u || (kv_heads == 0u && have_kv) || kv_heads > heads || hidden % heads != 0u || vocab == 0u) {
        if (have_kv && (kv_heads == 0u || kv_heads > heads)) { set_error(error_text, error_capacity, "invalid Llama attention head configuration"); return LM_ERR_PARSE; }
        set_error(error_text, error_capacity, "config.json lacks required Llama fields"); return LM_ERR_UNSUPPORTED;
    }
    if (model->tokenizer) {
        lm_tokenizer_info tokenizer_info{};
        if (lm_tokenizer_get_info(model->tokenizer, &tokenizer_info) != LM_OK || tokenizer_info.vocabulary_size != vocab) {
            set_error(error_text, error_capacity, "config and tokenizer vocabulary sizes differ"); return LM_ERR_PARSE;
        }
    }
    model->architecture = {context, hidden, layers, heads, kv_heads, intermediate, rope, rms};
    model->hf_tied_output = tie_word_embeddings ? 1u : 0u;
    model->has_architecture = 1u;
    return LM_OK;
}

lm_status lm_model_token_count(const lm_model_file *model, uint32_t *out_count) {
    if (!model || !out_count) return LM_ERR_ARGUMENT;
    if (model->tokenizer) {
        lm_tokenizer_info info{};
        const lm_status status = lm_tokenizer_get_info(model->tokenizer, &info);
        if (status != LM_OK) return status;
        *out_count = info.vocabulary_size;
        return LM_OK;
    }
    if (model->tokens.empty()) return LM_ERR_UNSUPPORTED;
    if (model->tokens.size() > UINT32_MAX) return LM_ERR_CAPACITY;
    *out_count = static_cast<uint32_t>(model->tokens.size());
    return LM_OK;
}

lm_status lm_model_token_at(const lm_model_file *model, uint32_t token_id,
                            char *out_token, size_t out_capacity, size_t *out_bytes) {
    if (!model || !out_bytes) return LM_ERR_ARGUMENT;
    if (model->tokenizer) {
        size_t bytes = 0u;
        const lm_status status = lm_tokenizer_decode(model->tokenizer, &token_id, 1u, nullptr, 0u, &bytes);
        if (status != LM_OK) return status;
        *out_bytes = bytes;
        if (!out_token && out_capacity == 0u) return LM_OK;
        if (!out_token || out_capacity <= bytes) return LM_ERR_CAPACITY;
        return lm_tokenizer_decode(model->tokenizer, &token_id, 1u, out_token, out_capacity, out_bytes);
    }
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
    if (model->tokenizer) return lm_tokenizer_encode(model->tokenizer, text, text_bytes, out_tokens, token_capacity, out_count);
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
    if (model->tokenizer) return lm_tokenizer_decode(model->tokenizer, tokens, token_count, out_text, out_capacity, out_bytes);
    if (model->tokens.empty()) return LM_ERR_UNSUPPORTED;
    size_t total = 0u;
    for (size_t i = 0u; i < token_count; ++i) {
        if (tokens[i] >= model->tokens.size() || model->tokens[tokens[i]].size() >
            std::numeric_limits<size_t>::max() - total) return LM_ERR_RANGE;
        total += model->tokens[tokens[i]].size();
    }
    *out_bytes = total;
    if (token_count == 0u) {
        if (out_text && out_capacity != 0u) out_text[0] = '\0';
        return LM_OK;
    }
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
    if (model->shards) return LM_ERR_UNSUPPORTED;
    if (relative_offset > model->info.file_bytes - model->info.header_bytes ||
        bytes > model->info.file_bytes - model->info.header_bytes - relative_offset)
        return LM_ERR_RANGE;
    return lm_file_span_make(model->file, model->info.header_bytes + relative_offset, bytes, out_span);
}

lm_status lm_model_tensor_span_at(const lm_model_file *model, uint64_t index, uint64_t bytes, lm_file_span *out_span) {
    if (!model || !out_span || index >= model->tensors.size()) return LM_ERR_ARGUMENT;
    const lm_model_tensor_info &descriptor = model->tensors[static_cast<size_t>(index)];
    if (!model->shards) return lm_model_tensor_span(model, descriptor.relative_offset, bytes, out_span);
    if (descriptor.shard_index >= model->shard_headers.size() ||
        descriptor.shard_index >= model->shard_sizes.size()) return LM_ERR_RANGE;
    const uint64_t header_bytes = model->shard_headers[descriptor.shard_index];
    const uint64_t data_bytes = model->shard_sizes[descriptor.shard_index] - header_bytes;
    if (descriptor.relative_offset > data_bytes || bytes > data_bytes - descriptor.relative_offset)
        return LM_ERR_RANGE;
    return lm_file_shard_set_span(model->shards, descriptor.shard_index,
                                  header_bytes + descriptor.relative_offset, bytes, out_span);
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
    uint64_t limit = model->info.file_bytes - model->info.header_bytes;
    if (model->shards) {
        if (descriptor.shard_index >= model->shard_headers.size() || descriptor.shard_index >= model->shard_sizes.size()) return LM_ERR_RANGE;
        limit = model->shard_sizes[descriptor.shard_index] - model->shard_headers[descriptor.shard_index];
    }
    for (const lm_model_tensor_info &other : model->tensors) {
        if (other.shard_index == descriptor.shard_index &&
            other.relative_offset > descriptor.relative_offset && other.relative_offset < limit)
            limit = other.relative_offset;
    }
    if (descriptor.relative_offset > limit || expected_bytes > limit - descriptor.relative_offset) return LM_ERR_PARSE;
    lm_file_span span{};
    const lm_status spanned = lm_model_tensor_span_at(model, index, expected_bytes, &span);
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

static bool matrix_dims_match(const lm_model_tensor_info &descriptor,
                               uint32_t rows, uint32_t columns) {
    if (descriptor.rank != 2u) return false;
    return (descriptor.dims[0] == rows && descriptor.dims[1] == columns) ||
           (descriptor.dims[0] == columns && descriptor.dims[1] == rows);
}

lm_status lm_model_tensor_matvec_f32_cpu(const lm_model_file *model, uint64_t tensor_index,
                                         void *matrix_scratch, uint64_t scratch_bytes,
                                         uint32_t rows, uint32_t columns,
                                         const float *input, float *out) {
    if (!model || !matrix_scratch || !input || !out || rows == 0u || columns == 0u) return LM_ERR_ARGUMENT;
    if (static_cast<uint64_t>(rows) > UINT64_MAX / columns) return LM_ERR_CAPACITY;
    const uint64_t elements = static_cast<uint64_t>(rows) * columns;
    if (elements > UINT64_MAX / sizeof(float)) return LM_ERR_CAPACITY;
    const uint64_t scratch_bytes_needed = elements * sizeof(float);
    if (scratch_bytes_needed > scratch_bytes || scratch_bytes_needed > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
        return LM_ERR_CAPACITY;
    lm_model_tensor_info descriptor{};
    lm_status status = lm_model_tensor_info_at(model, tensor_index, &descriptor);
    if (status != LM_OK) return status;
    if (!matrix_dims_match(descriptor, rows, columns)) return LM_ERR_UNSUPPORTED;
    const bool scalar16 = descriptor.type == LM_DTYPE_F16 || descriptor.type == LM_DTYPE_BF16;
    if (descriptor.type != LM_DTYPE_F32 || scalar16) {
        if (model->info.format != LM_MODEL_SAFETENSORS || !scalar16) return LM_ERR_UNSUPPORTED;
    }
    const uint64_t source_element_bytes = descriptor.type == LM_DTYPE_F32 ? sizeof(float) : sizeof(uint16_t);
    if (elements > UINT64_MAX / source_element_bytes) return LM_ERR_CAPACITY;
    const uint64_t source_bytes = elements * source_element_bytes;
    lm_file_span span{};
    status = lm_model_tensor_span_at(model, tensor_index, source_bytes, &span);
    if (status != LM_OK) return status;
    status = lm_file_span_read(&span, 0u, matrix_scratch, static_cast<size_t>(source_bytes));
    if (status != LM_OK) return status;
    if (descriptor.type == LM_DTYPE_F16 || descriptor.type == LM_DTYPE_BF16) {
        unsigned char *storage = static_cast<unsigned char *>(matrix_scratch);
        for (uint64_t i = elements; i != 0u; --i) {
            uint16_t bits = 0u;
            std::memcpy(&bits, storage + static_cast<size_t>(i - 1u) * sizeof(bits), sizeof(bits));
            uint32_t value = descriptor.type == LM_DTYPE_F16 ? 0u : static_cast<uint32_t>(bits) << 16u;
            if (descriptor.type == LM_DTYPE_F16) {
                const float converted = half_to_float(bits);
                std::memcpy(&value, &converted, sizeof(value));
            }
            std::memcpy(storage + static_cast<size_t>(i - 1u) * sizeof(float), &value, sizeof(value));
        }
    }
    const float *matrix = static_cast<const float *>(matrix_scratch);
    for (uint32_t row = 0u; row < rows; ++row) {
        float sum = 0.0f;
        for (uint32_t column = 0u; column < columns; ++column)
            sum += input[column] * matrix[static_cast<size_t>(row) * columns + column];
        if (!std::isfinite(sum)) return LM_ERR_RANGE;
        out[row] = sum;
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

lm_status lm_model_moe_route_f32_cpu(const lm_model_file *model, uint64_t router_tensor,
                                     void *matrix_scratch, uint64_t scratch_bytes,
                                     const float *input, uint32_t hidden_size,
                                     uint32_t expert_count, uint32_t experts_per_token,
                                     lm_moe_route_policy policy, lm_moe_route *out_route) {
    if (!model || !matrix_scratch || !input || !out_route || hidden_size == 0u || expert_count == 0u ||
        experts_per_token == 0u || experts_per_token > expert_count ||
        (policy != LM_MOE_SOFTMAX_ALL_THEN_TOPK && policy != LM_MOE_SOFTMAX_SELECTED_ONLY))
        return LM_ERR_ARGUMENT;
    if (!finite_array(input, hidden_size)) return LM_ERR_RANGE;
    lm_model_tensor_info descriptor{};
    lm_status status = lm_model_tensor_info_at(model, router_tensor, &descriptor);
    if (status != LM_OK) return status;
    if (descriptor.rank != 2u || descriptor.type != LM_DTYPE_F32 || descriptor.dims[0] != expert_count ||
        descriptor.dims[1] != hidden_size) return LM_ERR_UNSUPPORTED;
    std::vector<float> logits(expert_count, 0.0f);
    status = lm_model_tensor_matvec_f32_cpu(model, router_tensor, matrix_scratch, scratch_bytes,
                                            expert_count, hidden_size, input, logits.data());
    if (status != LM_OK) return status;
    return lm_cpu_moe_route(logits.data(), expert_count, experts_per_token, policy, out_route);
}

lm_status lm_model_execute_moe_layer_f32_router_q4_k_cpu(const lm_model_file *model,
                                                         const lm_model_moe_layer_config *config,
                                                         void *router_scratch, uint64_t router_scratch_bytes,
                                                         void *gate_up_scratch, uint64_t gate_up_scratch_bytes,
                                                         void *down_scratch, uint64_t down_scratch_bytes,
                                                         const float *input, float *out_hidden,
                                                         size_t out_hidden_count) {
    if (!model || !config || !router_scratch || !gate_up_scratch || !down_scratch || !input || !out_hidden ||
        out_hidden_count < config->hidden_size || config->hidden_size == 0u || config->intermediate_size == 0u ||
        config->expert_count == 0u || config->experts_per_token == 0u || config->experts_per_token > 16u)
        return LM_ERR_ARGUMENT;
    lm_moe_route route{};
    lm_status status = lm_model_moe_route_f32_cpu(model, config->router_tensor, router_scratch,
                                                  router_scratch_bytes, input, config->hidden_size,
                                                  config->expert_count, config->experts_per_token,
                                                  config->route_policy, &route);
    if (status != LM_OK) return status;
    std::vector<float> selected;
    try {
        selected.assign(static_cast<size_t>(config->experts_per_token) * config->hidden_size, 0.0f);
    } catch (const std::bad_alloc &) {
        return LM_ERR_CAPACITY;
    }
    status = lm_model_moe_selected_expert_mlp_q4_k(model, config->gate_up_tensor, config->down_tensor,
                                                   &route, config->expected_layer, config->hidden_size,
                                                   config->intermediate_size, gate_up_scratch,
                                                   gate_up_scratch_bytes, down_scratch, down_scratch_bytes,
                                                   input, selected.data(), selected.size());
    if (status != LM_OK) return status;
    return lm_cpu_moe_combine(&route, selected.data(), config->hidden_size, out_hidden);
}

static lm_status validate_q4_k_binding_matvec(const lm_model_tensor_binding *binding,
                                                uint32_t rows, uint32_t columns,
                                                uint64_t *payload_bytes, uint32_t *blocks_per_row) {
    if (!binding || !payload_bytes || !blocks_per_row || rows == 0u || columns == 0u || columns % 256u != 0u ||
        binding->quant_format != LM_QUANT_GGML_Q4_K || !matrix_dims_match(binding->descriptor, rows, columns))
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
    const lm_status read = lm_file_span_read(&binding->span, 0u, packed_scratch,
                                             static_cast<size_t>(payload_bytes));
    if (read != LM_OK) return read;
    const uint32_t logical_dims[2] = {rows, columns};
    lm_tensor tensor{};
    const lm_status view = lm_tensor_make_q4_k_view(packed_scratch, payload_bytes, 2u,
                                                    logical_dims, &tensor);
    if (view != LM_OK) return view;
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
        binding->quant_format != LM_QUANT_GGML_Q8_0 || !matrix_dims_match(binding->descriptor, rows, columns))
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
    const lm_status read = lm_file_span_read(&binding->span, 0u, packed_scratch,
                                             static_cast<size_t>(payload_bytes));
    if (read != LM_OK) return read;
    const uint32_t logical_dims[2] = {rows, columns};
    lm_tensor tensor{};
    const lm_status view = lm_tensor_make_q8_0_view(packed_scratch, payload_bytes, 2u,
                                                    logical_dims, &tensor);
    if (view != LM_OK) return view;
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
        const uint64_t blocks = static_cast<uint64_t>(columns) / 256u;
        const uint64_t expected = static_cast<uint64_t>(rows) * blocks * 144u;
        if (!matrix_dims_match(binding.descriptor, rows, columns) || columns == 0u ||
            columns % 256u != 0u || blocks > UINT32_MAX || expected != binding.span.bytes) return LM_ERR_RANGE;
        if (config->vulkan_context) {
            if (expected > scratch_bytes) return LM_ERR_CAPACITY;
            const lm_status read = lm_file_span_read(&binding.span, 0u, packed_scratch, static_cast<size_t>(expected));
            if (read != LM_OK) return read;
            return lm_vulkan_packed_context_matvec(static_cast<lm_vulkan_packed_context *>(config->vulkan_context),
                                                   packed_scratch, rows, static_cast<uint32_t>(blocks), input, out);
        }
        if (!config->shader_path) return LM_ERR_ARGUMENT;
        return lm_model_tensor_binding_matvec_q4_k_vulkan(&binding, packed_scratch, scratch_bytes,
                                                          rows, columns, config->shader_path,
                                                          config->device_index, input, out);
    }
    if (binding.quant_format == LM_QUANT_GGML_Q8_0) {
        if (backend == LM_BACKEND_CPU)
            return lm_model_tensor_binding_matvec_q8_0_cpu(&binding, packed_scratch, scratch_bytes,
                                                           rows, columns, input, out);
        const uint64_t blocks = static_cast<uint64_t>(columns) / 32u;
        const uint64_t expected = static_cast<uint64_t>(rows) * blocks * 34u;
        if (!matrix_dims_match(binding.descriptor, rows, columns) || columns == 0u ||
            columns % 32u != 0u || blocks > UINT32_MAX || expected != binding.span.bytes) return LM_ERR_RANGE;
        if (config->vulkan_context) {
            if (expected > scratch_bytes) return LM_ERR_CAPACITY;
            const lm_status read = lm_file_span_read(&binding.span, 0u, packed_scratch, static_cast<size_t>(expected));
            if (read != LM_OK) return read;
            return lm_vulkan_packed_context_matvec(static_cast<lm_vulkan_packed_context *>(config->vulkan_context),
                                                   packed_scratch, rows, static_cast<uint32_t>(blocks), input, out);
        }
        if (!config->shader_path) return LM_ERR_ARGUMENT;
        return lm_model_tensor_binding_matvec_q8_0_vulkan(&binding, packed_scratch, scratch_bytes,
                                                          rows, columns, config->shader_path,
                                                          config->device_index, input, out);
    }
    return LM_ERR_UNSUPPORTED;
}

static bool is_hf_rotary_auxiliary(const lm_model_file *model, const lm_model_tensor_info &tensor) {
    if (!model || model->info.format != LM_MODEL_SAFETENSORS || !model->has_architecture || tensor.rank != 1u || tensor.type != LM_DTYPE_F32)
        return false;
    unsigned layer = 0u;
    char suffix[64] = {};
    if (std::sscanf(tensor.name, "model.layers.%u.%63s", &layer, suffix) != 2 ||
        std::strcmp(suffix, "self_attn.rotary_emb.inv_freq") != 0 || layer >= model->architecture.block_count ||
        model->architecture.head_count == 0u || model->architecture.embedding_length % model->architecture.head_count != 0u)
        return false;
    return tensor.dims[0] == model->architecture.embedding_length / model->architecture.head_count / 2u;
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
        if (mapped != LM_OK) {
            if (is_hf_rotary_auxiliary(model, model->tensors[i])) continue;
            return mapped;
        }
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
    if ((global_mask & (1u << LM_DECODER_TENSOR_OUTPUT)) == 0u &&
        (global_mask & (1u << LM_DECODER_TENSOR_TOKEN_EMBEDDING)) != 0u &&
        (model->info.format == LM_MODEL_GGUF || model->hf_tied_output != 0u)) {
        out_binding->output = out_binding->token_embedding;
        out_binding->output_tied = 1u;
        global_mask |= 1u << LM_DECODER_TENSOR_OUTPUT;
    }
    if (global_mask != global_required || out_binding->layer_count == 0u) return LM_ERR_UNSUPPORTED;
    for (uint32_t layer = 0u; layer < out_binding->layer_count; ++layer)
        if (layer_masks[layer] != layer_required) return LM_ERR_UNSUPPORTED;
    return LM_OK;
}

lm_status lm_model_build_mixtral_moe_graph(const lm_model_file *model,
                                           uint32_t expert_count, uint32_t experts_per_token,
                                           lm_model_moe_graph_binding *out_binding) {
    if (!model || !out_binding || model->info.format != LM_MODEL_GGUF || expert_count == 0u ||
        experts_per_token == 0u || experts_per_token > expert_count || experts_per_token > 16u)
        return LM_ERR_ARGUMENT;
    std::memset(out_binding, 0xff, sizeof(*out_binding));
    out_binding->layer_count = 0u;
    out_binding->expert_count = expert_count;
    out_binding->experts_per_token = experts_per_token;
    uint8_t router_seen[LM_MOE_GRAPH_MAX_LAYERS] = {};
    uint8_t gate_seen[LM_MOE_GRAPH_MAX_LAYERS] = {};
    uint8_t down_seen[LM_MOE_GRAPH_MAX_LAYERS] = {};
    for (size_t i = 0u; i < model->tensors.size(); ++i) {
        const lm_model_tensor_info &descriptor = model->tensors[i];
        uint32_t router_layer = 0u;
        if (parse_mixtral_router_name(descriptor.name, &router_layer)) {
            if (router_layer >= LM_MOE_GRAPH_MAX_LAYERS || router_seen[router_layer] != 0u ||
                descriptor.rank != 2u || descriptor.type != LM_DTYPE_F32) return LM_ERR_UNSUPPORTED;
            out_binding->layers[router_layer].router_tensor = static_cast<uint64_t>(i);
            router_seen[router_layer] = 1u;
            if (router_layer + 1u > out_binding->layer_count) out_binding->layer_count = router_layer + 1u;
            continue;
        }
        lm_moe_tensor_mapping mapping{};
        const lm_status mapped = lm_moe_map_mixtral_tensor(&descriptor, expert_count, &mapping);
        if (mapped == LM_ERR_UNSUPPORTED) continue;
        if (mapped != LM_OK || mapping.layer_index >= LM_MOE_GRAPH_MAX_LAYERS) return mapped;
        lm_model_moe_layer_binding &layer = out_binding->layers[mapping.layer_index];
        if (mapping.role == LM_MOE_TENSOR_GATE_UP_EXPERT) {
            if (gate_seen[mapping.layer_index] != 0u) return LM_ERR_PARSE;
            layer.gate_up_tensor = static_cast<uint64_t>(i);
            gate_seen[mapping.layer_index] = 1u;
        } else {
            if (down_seen[mapping.layer_index] != 0u) return LM_ERR_PARSE;
            layer.down_tensor = static_cast<uint64_t>(i);
            down_seen[mapping.layer_index] = 1u;
        }
        if (mapping.layer_index + 1u > out_binding->layer_count)
            out_binding->layer_count = mapping.layer_index + 1u;
    }
    if (out_binding->layer_count == 0u) return LM_ERR_UNSUPPORTED;
    for (uint32_t layer = 0u; layer < out_binding->layer_count; ++layer)
        if (router_seen[layer] == 0u || gate_seen[layer] == 0u || down_seen[layer] == 0u)
            return LM_ERR_UNSUPPORTED;
    return LM_OK;
}

lm_status lm_model_execute_mixtral_moe_layer_f32_router_q4_k_cpu(
    const lm_model_file *model, const lm_model_moe_graph_binding *graph, uint32_t layer_index,
    uint32_t hidden_size, uint32_t intermediate_size,
    void *router_scratch, uint64_t router_scratch_bytes,
    void *gate_up_scratch, uint64_t gate_up_scratch_bytes,
    void *down_scratch, uint64_t down_scratch_bytes,
    const float *input, float *out_hidden, size_t out_hidden_count) {
    if (!model || !graph || layer_index >= graph->layer_count || graph->layer_count == 0u ||
        graph->layer_count > LM_MOE_GRAPH_MAX_LAYERS) return LM_ERR_ARGUMENT;
    lm_model_moe_layer_config config{};
    config.router_tensor = graph->layers[layer_index].router_tensor;
    config.gate_up_tensor = graph->layers[layer_index].gate_up_tensor;
    config.down_tensor = graph->layers[layer_index].down_tensor;
    config.expected_layer = layer_index;
    config.hidden_size = hidden_size;
    config.intermediate_size = intermediate_size;
    config.expert_count = graph->expert_count;
    config.experts_per_token = graph->experts_per_token;
    config.route_policy = LM_MOE_SOFTMAX_SELECTED_ONLY;
    return lm_model_execute_moe_layer_f32_router_q4_k_cpu(model, &config, router_scratch, router_scratch_bytes,
                                                         gate_up_scratch, gate_up_scratch_bytes,
                                                         down_scratch, down_scratch_bytes, input,
                                                         out_hidden, out_hidden_count);
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
