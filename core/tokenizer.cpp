#include "lm/lm.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr size_t kMaxJsonBytes = 32u << 20u;
constexpr size_t kMaxVocabulary = 1u << 20u;
constexpr size_t kMaxMerges = 1u << 20u;
constexpr size_t kMaxStringBytes = 1u << 20u;

void set_error(char *dst, size_t cap, const char *text) {
    if (!dst || cap == 0u) return;
    std::strncpy(dst, text, cap - 1u);
    dst[cap - 1u] = '\0';
}

struct JsonValue {
    enum Kind { Null, Boolean, Number, String, Array, Object } kind = Null;
    double number = 0.0;
    bool boolean = false;
    std::string string;
    std::vector<JsonValue> array;
    std::map<std::string, JsonValue> object;
};

void append_codepoint(std::string *out, uint32_t codepoint) {
    if (codepoint <= 0x7fu) out->push_back(static_cast<char>(codepoint));
    else if (codepoint <= 0x7ffu) { out->push_back(static_cast<char>(0xc0u | (codepoint >> 6u))); out->push_back(static_cast<char>(0x80u | (codepoint & 0x3fu))); }
    else if (codepoint <= 0xffffu) { out->push_back(static_cast<char>(0xe0u | (codepoint >> 12u))); out->push_back(static_cast<char>(0x80u | ((codepoint >> 6u) & 0x3fu))); out->push_back(static_cast<char>(0x80u | (codepoint & 0x3fu))); }
    else { out->push_back(static_cast<char>(0xf0u | (codepoint >> 18u))); out->push_back(static_cast<char>(0x80u | ((codepoint >> 12u) & 0x3fu))); out->push_back(static_cast<char>(0x80u | ((codepoint >> 6u) & 0x3fu))); out->push_back(static_cast<char>(0x80u | (codepoint & 0x3fu))); }
}

class JsonParser {
public:
    JsonParser(const char *begin, const char *end) : p_(begin), end_(end) {}
    bool parse(JsonValue *out) { if (!out || !value(out, 0u)) return false; whitespace(); return p_ == end_; }
private:
    void whitespace() { while (p_ != end_ && (*p_ == ' ' || *p_ == '\n' || *p_ == '\r' || *p_ == '\t')) ++p_; }
    bool literal(const char *text) { const size_t n = std::strlen(text); if (static_cast<size_t>(end_ - p_) < n || std::strncmp(p_, text, n) != 0) return false; p_ += n; return true; }
    bool hex4(uint32_t *out) { if (!out || static_cast<size_t>(end_ - p_) < 4u) return false; uint32_t value = 0u; for (unsigned i = 0u; i < 4u; ++i) { const char c = p_[i]; uint32_t digit = 0u; if (c >= '0' && c <= '9') digit = static_cast<uint32_t>(c - '0'); else if (c >= 'a' && c <= 'f') digit = static_cast<uint32_t>(c - 'a' + 10); else if (c >= 'A' && c <= 'F') digit = static_cast<uint32_t>(c - 'A' + 10); else return false; value = (value << 4u) | digit; } p_ += 4u; *out = value; return true; }
    bool string_value(std::string *out) {
        if (!out || p_ == end_ || *p_++ != '"') return false;
        std::string result;
        try { result.reserve(16u); } catch (const std::bad_alloc &) { return false; }
        while (p_ != end_ && *p_ != '"') {
            const unsigned char c = static_cast<unsigned char>(*p_++);
            if (c < 0x20u) return false;
            if (c != '\\') { result.push_back(static_cast<char>(c)); if (result.size() > kMaxStringBytes) return false; continue; }
            if (p_ == end_) return false;
            const char escape = *p_++;
            if (escape == '"' || escape == '\\' || escape == '/') result.push_back(escape);
            else if (escape == 'b') result.push_back('\b');
            else if (escape == 'f') result.push_back('\f');
            else if (escape == 'n') result.push_back('\n');
            else if (escape == 'r') result.push_back('\r');
            else if (escape == 't') result.push_back('\t');
            else if (escape == 'u') {
                uint32_t high = 0u; if (!hex4(&high)) return false;
                if (high >= 0xd800u && high <= 0xdbffu) {
                    if (static_cast<size_t>(end_ - p_) < 6u || p_[0] != '\\' || p_[1] != 'u') return false;
                    p_ += 2u; uint32_t low = 0u; if (!hex4(&low) || low < 0xdc00u || low > 0xdfffu) return false;
                    high = 0x10000u + ((high - 0xd800u) << 10u) + (low - 0xdc00u);
                } else if (high >= 0xdc00u && high <= 0xdfffu) return false;
                append_codepoint(&result, high);
            } else return false;
            if (result.size() > kMaxStringBytes) return false;
        }
        if (p_ == end_ || *p_++ != '"') return false;
        *out = std::move(result); return true;
    }
    bool number_value(JsonValue *out) {
        const char *start = p_; char *finish = nullptr; const double number = std::strtod(start, &finish);
        if (finish == start || !std::isfinite(number)) return false;
        p_ = finish; out->kind = JsonValue::Number; out->number = number; return true;
    }
    bool value(JsonValue *out, unsigned depth) {
        if (!out || depth > 64u) return false;
        whitespace(); if (p_ == end_) return false;
        if (*p_ == 'n') { if (!literal("null")) return false; out->kind = JsonValue::Null; return true; }
        if (*p_ == 't') { if (!literal("true")) return false; out->kind = JsonValue::Boolean; out->boolean = true; return true; }
        if (*p_ == 'f') { if (!literal("false")) return false; out->kind = JsonValue::Boolean; out->boolean = false; return true; }
        if (*p_ == '"') { out->kind = JsonValue::String; return string_value(&out->string); }
        if (*p_ == '[') return array_value(out, depth);
        if (*p_ == '{') return object_value(out, depth);
        if (*p_ == '-' || (*p_ >= '0' && *p_ <= '9')) return number_value(out);
        return false;
    }
    bool array_value(JsonValue *out, unsigned depth) {
        ++p_; out->kind = JsonValue::Array; whitespace();
        if (p_ != end_ && *p_ == ']') { ++p_; return true; }
        for (;;) {
            if (out->array.size() >= kMaxMerges) return false;
            JsonValue item; if (!value(&item, depth + 1u)) return false; out->array.push_back(std::move(item)); whitespace();
            if (p_ != end_ && *p_ == ']') { ++p_; return true; }
            if (p_ == end_ || *p_++ != ',') return false;
        }
    }
    bool object_value(JsonValue *out, unsigned depth) {
        ++p_; out->kind = JsonValue::Object; whitespace();
        if (p_ != end_ && *p_ == '}') { ++p_; return true; }
        for (;;) {
            std::string key; if (!string_value(&key)) return false; whitespace();
            if (p_ == end_ || *p_++ != ':') return false;
            JsonValue item; if (!value(&item, depth + 1u)) return false;
            if (!out->object.emplace(std::move(key), std::move(item)).second) return false;
            whitespace(); if (p_ != end_ && *p_ == '}') { ++p_; return true; }
            if (p_ == end_ || *p_++ != ',') return false;
        }
    }
    const char *p_;
    const char *end_;
};

const JsonValue *field(const JsonValue &object, const char *name) {
    const auto found = object.object.find(name); return found == object.object.end() ? nullptr : &found->second;
}

bool string_field(const JsonValue &object, const char *name, std::string *out, bool required) {
    const JsonValue *value = field(object, name);
    if (!value) return !required;
    if (value->kind != JsonValue::String || !out) return false;
    *out = value->string; return true;
}

bool bool_field(const JsonValue &object, const char *name, bool *out, bool default_value) {
    const JsonValue *value = field(object, name);
    if (!value) { *out = default_value; return true; }
    if (value->kind != JsonValue::Boolean) return false;
    *out = value->boolean;
    return true;
}

bool strict_optional_null(const JsonValue &root, const char *name) {
    const JsonValue *value = field(root, name); return !value || value->kind == JsonValue::Null;
}

struct Pair { uint32_t left; uint32_t right; bool operator<(const Pair &other) const { return left < other.left || (left == other.left && right < other.right); } };
struct Merge { uint32_t rank; uint32_t output; };

bool parse_merge(const JsonValue &value, std::string *left, std::string *right) {
    if (value.kind == JsonValue::String) {
        const size_t split = value.string.find(' ');
        return split != std::string::npos && split != 0u && split + 1u < value.string.size() && value.string.find(' ', split + 1u) == std::string::npos &&
               (left->assign(value.string, 0u, split), right->assign(value.string, split + 1u), true);
    }
    if (value.kind == JsonValue::Array && value.array.size() == 2u && value.array[0].kind == JsonValue::String && value.array[1].kind == JsonValue::String &&
        !value.array[0].string.empty() && !value.array[1].string.empty()) { *left = value.array[0].string; *right = value.array[1].string; return true; }
    return false;
}

bool valid_utf8(const std::string &text) {
    for (size_t i = 0u; i < text.size();) {
        const unsigned char c = static_cast<unsigned char>(text[i++]); uint32_t count = 0u, codepoint = 0u;
        if (c <= 0x7fu) continue;
        if (c >= 0xc2u && c <= 0xdfu) { count = 1u; codepoint = c & 0x1fu; }
        else if (c >= 0xe0u && c <= 0xefu) { count = 2u; codepoint = c & 0x0fu; }
        else if (c >= 0xf0u && c <= 0xf4u) { count = 3u; codepoint = c & 0x07u; }
        else return false;
        if (i + count > text.size()) return false;
        for (uint32_t j = 0u; j < count; ++j) { const unsigned char tail = static_cast<unsigned char>(text[i++]); if ((tail & 0xc0u) != 0x80u) return false; codepoint = (codepoint << 6u) | (tail & 0x3fu); }
        if ((count == 2u && codepoint < 0x800u) || (count == 3u && codepoint < 0x10000u) || codepoint > 0x10ffffu || (codepoint >= 0xd800u && codepoint <= 0xdfffu)) return false;
    }
    return true;
}

bool next_utf8(const char *text, size_t bytes, size_t *position, std::string *symbol) {
    const size_t start = *position;
    if (start >= bytes) return false;
    const unsigned char c = static_cast<unsigned char>(text[start]);
    size_t length = 1u;
    if (c >= 0xc2u && c <= 0xdfu) length = 2u;
    else if (c >= 0xe0u && c <= 0xefu) length = 3u;
    else if (c >= 0xf0u && c <= 0xf4u) length = 4u;
    else if (c > 0x7fu) return false;
    if (length > bytes - start) return false;
    symbol->assign(text + start, length);
    if (!valid_utf8(*symbol)) return false;
    *position += length;
    return true;
}

} // namespace

struct lm_tokenizer {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, uint32_t> token_to_id;
    std::map<Pair, Merge> merges;
    uint8_t unk_present = 0u;
    uint32_t unk_id = 0u;
};

lm_status lm_tokenizer_open_json(const char *path, lm_tokenizer **out_tokenizer, char *error_text, size_t error_capacity) {
    if (!path || !out_tokenizer) return LM_ERR_ARGUMENT;
    *out_tokenizer = nullptr;
    set_error(error_text, error_capacity, "");
    std::ifstream file(path, std::ios::binary); if (!file) { set_error(error_text, error_capacity, "cannot open tokenizer.json"); return LM_ERR_IO; }
    file.seekg(0, std::ios::end); const std::streamoff end = file.tellg(); if (end < 0 || static_cast<uint64_t>(end) > kMaxJsonBytes) { set_error(error_text, error_capacity, "tokenizer.json is too large"); return LM_ERR_CAPACITY; }
    file.seekg(0, std::ios::beg); std::string source(static_cast<size_t>(end), '\0'); if (!source.empty()) file.read(source.data(), static_cast<std::streamsize>(source.size())); if (!file) { set_error(error_text, error_capacity, "cannot read tokenizer.json"); return LM_ERR_IO; }
    JsonValue root; JsonParser parser(source.data(), source.data() + source.size()); if (!parser.parse(&root) || root.kind != JsonValue::Object) { set_error(error_text, error_capacity, "invalid tokenizer.json"); return LM_ERR_PARSE; }
    const JsonValue *model = field(root, "model"); if (!model || model->kind != JsonValue::Object) { set_error(error_text, error_capacity, "tokenizer model is missing"); return LM_ERR_PARSE; }
    std::string model_type; if (!string_field(*model, "type", &model_type, true) || model_type != "BPE") { set_error(error_text, error_capacity, "only tokenizer.json BPE models are supported"); return LM_ERR_UNSUPPORTED; }
    if (!strict_optional_null(root, "normalizer") || !strict_optional_null(root, "pre_tokenizer") || !strict_optional_null(root, "post_processor") || !strict_optional_null(root, "decoder")) { set_error(error_text, error_capacity, "tokenizer pipeline component is unsupported"); return LM_ERR_UNSUPPORTED; }
    const JsonValue *added = field(root, "added_tokens"); if (added && (added->kind != JsonValue::Array || !added->array.empty())) { set_error(error_text, error_capacity, "tokenizer added tokens are unsupported"); return LM_ERR_UNSUPPORTED; }
    const JsonValue *vocab = field(*model, "vocab"); const JsonValue *merges = field(*model, "merges");
    if (!vocab || vocab->kind != JsonValue::Object || !merges || merges->kind != JsonValue::Array || vocab->object.empty() || vocab->object.size() > kMaxVocabulary || merges->array.size() > kMaxMerges) { set_error(error_text, error_capacity, "invalid tokenizer BPE vocabulary or merges"); return LM_ERR_PARSE; }
    auto *tokenizer = new (std::nothrow) lm_tokenizer(); if (!tokenizer) return LM_ERR_CAPACITY;
    try {
        uint32_t maximum_id = 0u; std::vector<uint8_t> seen;
        for (const auto &entry : vocab->object) {
            if (entry.second.kind != JsonValue::Number || entry.second.number < 0.0 || entry.second.number > static_cast<double>(UINT32_MAX) || std::floor(entry.second.number) != entry.second.number || !valid_utf8(entry.first)) { delete tokenizer; set_error(error_text, error_capacity, "invalid tokenizer vocabulary entry"); return LM_ERR_PARSE; }
            const uint32_t id = static_cast<uint32_t>(entry.second.number);
            if (id > 4u * 1024u * 1024u) { delete tokenizer; return LM_ERR_CAPACITY; }
            if (id > maximum_id) maximum_id = id;
            tokenizer->token_to_id.emplace(entry.first, id);
        }
        tokenizer->id_to_token.assign(static_cast<size_t>(maximum_id) + 1u, std::string()); seen.assign(static_cast<size_t>(maximum_id) + 1u, 0u);
        for (const auto &entry : vocab->object) { const uint32_t id = tokenizer->token_to_id.at(entry.first); if (seen[id] != 0u) { delete tokenizer; return LM_ERR_PARSE; } seen[id] = 1u; tokenizer->id_to_token[id] = entry.first; }
        std::string unk; if (!string_field(*model, "unk_token", &unk, false)) { delete tokenizer; return LM_ERR_PARSE; }
        if (!unk.empty()) { const auto found = tokenizer->token_to_id.find(unk); if (found == tokenizer->token_to_id.end()) { delete tokenizer; return LM_ERR_PARSE; } tokenizer->unk_present = 1u; tokenizer->unk_id = found->second; }
        std::string prefix, suffix; if (!string_field(*model, "continuing_subword_prefix", &prefix, false) || !string_field(*model, "end_of_word_suffix", &suffix, false)) { delete tokenizer; return LM_ERR_PARSE; }
        if (!prefix.empty() || !suffix.empty()) { delete tokenizer; set_error(error_text, error_capacity, "BPE subword affixes are unsupported in raw UTF-8 mode"); return LM_ERR_UNSUPPORTED; }
        bool flag = false; if (!bool_field(*model, "byte_fallback", &flag, false) || flag || !bool_field(*model, "fuse_unk", &flag, false) || flag || !bool_field(*model, "ignore_merges", &flag, false) || flag) { delete tokenizer; set_error(error_text, error_capacity, "BPE byte or merge policy is unsupported"); return LM_ERR_UNSUPPORTED; }
        const JsonValue *dropout = field(*model, "dropout"); if (dropout && (dropout->kind != JsonValue::Null && (dropout->kind != JsonValue::Number || dropout->number != 0.0))) { delete tokenizer; set_error(error_text, error_capacity, "BPE dropout is unsupported"); return LM_ERR_UNSUPPORTED; }
        for (size_t rank = 0u; rank < merges->array.size(); ++rank) {
            std::string left, right; if (!parse_merge(merges->array[rank], &left, &right) || !valid_utf8(left) || !valid_utf8(right)) { delete tokenizer; set_error(error_text, error_capacity, "invalid BPE merge"); return LM_ERR_PARSE; }
            const auto left_id = tokenizer->token_to_id.find(left), right_id = tokenizer->token_to_id.find(right); if (left_id == tokenizer->token_to_id.end() || right_id == tokenizer->token_to_id.end()) { delete tokenizer; return LM_ERR_PARSE; }
            const std::string output = left + right; const auto output_id = tokenizer->token_to_id.find(output); if (output_id == tokenizer->token_to_id.end()) { delete tokenizer; return LM_ERR_PARSE; }
            const Pair pair{left_id->second, right_id->second}; if (!tokenizer->merges.emplace(pair, Merge{static_cast<uint32_t>(rank), output_id->second}).second) { delete tokenizer; return LM_ERR_PARSE; }
        }
    } catch (const std::bad_alloc &) { delete tokenizer; return LM_ERR_CAPACITY; }
    *out_tokenizer = tokenizer; return LM_OK;
}

void lm_tokenizer_destroy(lm_tokenizer *tokenizer) { delete tokenizer; }

lm_status lm_tokenizer_get_info(const lm_tokenizer *tokenizer, lm_tokenizer_info *out_info) {
    if (!tokenizer || !out_info) return LM_ERR_ARGUMENT;
    out_info->vocabulary_size = tokenizer->token_to_id.size() > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(tokenizer->token_to_id.size());
    out_info->merge_count = tokenizer->merges.size() > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(tokenizer->merges.size());
    out_info->raw_utf8_bpe = 1u;
    return LM_OK;
}

lm_status lm_tokenizer_encode(const lm_tokenizer *tokenizer, const char *text, size_t text_bytes, uint32_t *out_tokens, size_t token_capacity, size_t *out_count) {
    if (!tokenizer || !out_count || (!text && text_bytes != 0u)) return LM_ERR_ARGUMENT;
    *out_count = 0u;
    if (text_bytes == 0u) return LM_OK;
    if (!out_tokens || token_capacity == 0u) return LM_ERR_CAPACITY;
    std::vector<std::pair<uint32_t, std::string>> symbols;
    try {
        for (size_t position = 0u; position < text_bytes;) {
            std::string symbol; if (!next_utf8(text, text_bytes, &position, &symbol)) return LM_ERR_PARSE; const auto found = tokenizer->token_to_id.find(symbol);
            if (found == tokenizer->token_to_id.end()) { if (!tokenizer->unk_present) return LM_ERR_UNSUPPORTED; symbols.emplace_back(tokenizer->unk_id, std::string()); }
            else symbols.emplace_back(found->second, std::move(symbol));
        }
        for (;;) {
            size_t best = symbols.size(); Merge selected{}; bool have = false;
            for (size_t i = 0u; i + 1u < symbols.size(); ++i) { const auto found = tokenizer->merges.find(Pair{symbols[i].first, symbols[i + 1u].first}); if (found != tokenizer->merges.end() && (!have || found->second.rank < selected.rank)) { best = i; selected = found->second; have = true; } }
            if (!have) break;
            symbols[best] = {selected.output, symbols[best].second + symbols[best + 1u].second}; symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best + 1u));
        }
    } catch (const std::bad_alloc &) { return LM_ERR_CAPACITY; }
    *out_count = symbols.size(); if (symbols.size() > token_capacity) return LM_ERR_CAPACITY; for (size_t i = 0u; i < symbols.size(); ++i) out_tokens[i] = symbols[i].first; return LM_OK;
}

lm_status lm_tokenizer_decode(const lm_tokenizer *tokenizer, const uint32_t *tokens, size_t token_count, char *out_text, size_t out_capacity, size_t *out_bytes) {
    if (!tokenizer || !out_bytes || (!tokens && token_count != 0u)) return LM_ERR_ARGUMENT;
    size_t total = 0u;
    for (size_t i = 0u; i < token_count; ++i) { if (tokens[i] >= tokenizer->id_to_token.size() || (tokenizer->id_to_token[tokens[i]].empty() && tokenizer->token_to_id.find(std::string()) == tokenizer->token_to_id.end())) return LM_ERR_RANGE; const size_t bytes = tokenizer->id_to_token[tokens[i]].size(); if (total > std::numeric_limits<size_t>::max() - bytes) return LM_ERR_CAPACITY; total += bytes; }
    *out_bytes = total; if (!out_text && out_capacity == 0u) return LM_OK; if (!out_text || out_capacity <= total) return LM_ERR_CAPACITY; size_t position = 0u; for (size_t i = 0u; i < token_count; ++i) { const std::string &value = tokenizer->id_to_token[tokens[i]]; std::memcpy(out_text + position, value.data(), value.size()); position += value.size(); } out_text[position] = '\0'; return LM_OK;
}
