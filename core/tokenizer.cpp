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
            whitespace();
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
            whitespace();
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
    if (value->kind == JsonValue::Null && !required) { if (out) out->clear(); return true; }
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

bool object_type(const JsonValue *value, const char *type) {
    std::string actual;
    return value && value->kind == JsonValue::Object && string_field(*value, "type", &actual, true) && actual == type;
}

bool number_field_equals(const JsonValue &object, const char *name, double expected) {
    const JsonValue *value = field(object, name);
    return value && value->kind == JsonValue::Number && value->number == expected;
}

bool string_value_equals(const JsonValue *value, const char *expected) {
    return value && value->kind == JsonValue::String && value->string == expected;
}

bool exact_sentencepiece_normalizer(const JsonValue &root) {
    const JsonValue *normalizer = field(root, "normalizer");
    if (!object_type(normalizer, "Sequence")) return false;
    const JsonValue *items = field(*normalizer, "normalizers");
    if (!items || items->kind != JsonValue::Array || items->array.size() != 2u) return false;
    const JsonValue &prepend = items->array[0];
    const JsonValue &replace = items->array[1];
    const JsonValue *pattern = field(replace, "pattern");
    return object_type(&prepend, "Prepend") && string_value_equals(field(prepend, "prepend"), "\xE2\x96\x81") &&
           object_type(&replace, "Replace") && string_value_equals(field(replace, "content"), "\xE2\x96\x81") &&
           pattern && pattern->kind == JsonValue::Object &&
           string_value_equals(field(*pattern, "String"), " ");
}

bool exact_sentencepiece_decoder(const JsonValue &root) {
    const JsonValue *decoder = field(root, "decoder");
    if (!object_type(decoder, "Sequence")) return false;
    const JsonValue *items = field(*decoder, "decoders");
    if (!items || items->kind != JsonValue::Array || items->array.size() != 4u) return false;
    const JsonValue &replace = items->array[0];
    const JsonValue &fallback = items->array[1];
    const JsonValue &fuse = items->array[2];
    const JsonValue &strip = items->array[3];
    const JsonValue *pattern = field(replace, "pattern");
    return object_type(&replace, "Replace") && pattern && pattern->kind == JsonValue::Object &&
           string_value_equals(field(*pattern, "String"), "\xE2\x96\x81") &&
           string_value_equals(field(replace, "content"), " ") && object_type(&fallback, "ByteFallback") &&
           object_type(&fuse, "Fuse") && object_type(&strip, "Strip") &&
           string_value_equals(field(strip, "content"), " ") &&
           number_field_equals(strip, "start", 1.0) && number_field_equals(strip, "stop", 0.0);
}

bool exact_sentencepiece_post_processor(const JsonValue &root, uint32_t *bos_id) {
    const JsonValue *processor = field(root, "post_processor");
    if (!object_type(processor, "TemplateProcessing") || !bos_id) return false;
    const JsonValue *single = field(*processor, "single");
    if (!single || single->kind != JsonValue::Array || single->array.size() != 2u) return false;
    const JsonValue &special = single->array[0];
    const JsonValue &sequence = single->array[1];
    const JsonValue *special_value = field(special, "SpecialToken");
    const JsonValue *sequence_value = field(sequence, "Sequence");
    if (!special_value || special_value->kind != JsonValue::Object || !sequence_value ||
        sequence_value->kind != JsonValue::Object || !string_value_equals(field(*special_value, "id"), "<s>") ||
        !number_field_equals(*special_value, "type_id", 0.0) ||
        !string_value_equals(field(*sequence_value, "id"), "A") ||
        !number_field_equals(*sequence_value, "type_id", 0.0)) return false;
    const JsonValue *specials = field(*processor, "special_tokens");
    const JsonValue *bos = specials && specials->kind == JsonValue::Object ? field(*specials, "<s>") : nullptr;
    const JsonValue *ids = bos && bos->kind == JsonValue::Object ? field(*bos, "ids") : nullptr;
    if (!ids || ids->kind != JsonValue::Array || ids->array.size() != 1u ||
        ids->array[0].kind != JsonValue::Number || ids->array[0].number < 0.0 ||
        ids->array[0].number > static_cast<double>(UINT32_MAX) ||
        std::floor(ids->array[0].number) != ids->array[0].number) return false;
    *bos_id = static_cast<uint32_t>(ids->array[0].number);
    return true;
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

bool normalize_sentencepiece(const char *text, size_t text_bytes, std::string *out) {
    if (!text || !out || !valid_utf8(std::string(text, text_bytes))) return false;
    const std::string marker("\xE2\x96\x81");
    try {
        out->clear();
        out->reserve(text_bytes + marker.size());
        out->append(marker);
        for (size_t position = 0u; position < text_bytes;) {
            std::string symbol;
            if (!next_utf8(text, text_bytes, &position, &symbol)) return false;
            if (symbol == " ") out->append(marker);
            else out->append(symbol);
        }
    } catch (const std::bad_alloc &) { return false; }
    return true;
}

bool decode_byte_fallback(const std::string &token, std::string *out) {
    if (!out || token.size() != 6u || token[0] != '<' || token[1] != '0' ||
        token[2] != 'x' || token[5] != '>') return false;
    auto digit = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    const int high = digit(token[3]);
    const int low = digit(token[4]);
    if (high < 0 || low < 0) return false;
    out->assign(1u, static_cast<char>((high << 4) | low));
    return true;
}

} // namespace

struct lm_tokenizer {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, uint32_t> token_to_id;
    std::map<Pair, Merge> merges;
    uint8_t unk_present = 0u;
    uint32_t unk_id = 0u;
    uint8_t sentencepiece = 0u;
    uint8_t byte_fallback = 0u;
    uint8_t fuse_unk = 0u;
    uint8_t add_bos = 0u;
    uint32_t bos_id = 0u;
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
    const bool sentencepiece = exact_sentencepiece_normalizer(root) &&
        strict_optional_null(root, "pre_tokenizer") && exact_sentencepiece_decoder(root);
    uint32_t sentencepiece_bos_id = 0u;
    const bool sentencepiece_post = exact_sentencepiece_post_processor(root, &sentencepiece_bos_id);
    if (!sentencepiece && (!strict_optional_null(root, "normalizer") ||
                           !strict_optional_null(root, "pre_tokenizer") ||
                           !strict_optional_null(root, "post_processor") ||
                           !strict_optional_null(root, "decoder"))) {
        set_error(error_text, error_capacity, "tokenizer pipeline component is unsupported");
        return LM_ERR_UNSUPPORTED;
    }
    if (sentencepiece != sentencepiece_post) {
        set_error(error_text, error_capacity, "incomplete SentencePiece tokenizer pipeline");
        return LM_ERR_UNSUPPORTED;
    }
    const JsonValue *added = field(root, "added_tokens");
    if (added && added->kind != JsonValue::Array) {
        set_error(error_text, error_capacity, "invalid tokenizer added tokens");
        return LM_ERR_PARSE;
    }
    if (!sentencepiece && added && !added->array.empty()) {
        set_error(error_text, error_capacity, "tokenizer added tokens are unsupported");
        return LM_ERR_UNSUPPORTED;
    }
    const JsonValue *vocab = field(*model, "vocab"); const JsonValue *merges = field(*model, "merges");
    if (!vocab || vocab->kind != JsonValue::Object || !merges || merges->kind != JsonValue::Array || vocab->object.empty() || vocab->object.size() > kMaxVocabulary || merges->array.size() > kMaxMerges) { set_error(error_text, error_capacity, "invalid tokenizer BPE vocabulary or merges"); return LM_ERR_PARSE; }
    auto *tokenizer = new (std::nothrow) lm_tokenizer(); if (!tokenizer) return LM_ERR_CAPACITY;
    tokenizer->sentencepiece = sentencepiece ? 1u : 0u;
    tokenizer->add_bos = sentencepiece ? 1u : 0u;
    tokenizer->bos_id = sentencepiece_bos_id;
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
        if (added && sentencepiece) {
            for (const JsonValue &entry : added->array) {
                if (entry.kind != JsonValue::Object) { delete tokenizer; return LM_ERR_PARSE; }
                const JsonValue *id_value = field(entry, "id");
                const JsonValue *content = field(entry, "content");
                const JsonValue *special = field(entry, "special");
                if (!id_value || id_value->kind != JsonValue::Number || id_value->number < 0.0 ||
                    id_value->number > static_cast<double>(UINT32_MAX) || std::floor(id_value->number) != id_value->number ||
                    !content || content->kind != JsonValue::String || !special || special->kind != JsonValue::Boolean ||
                    !special->boolean || !valid_utf8(content->string)) { delete tokenizer; set_error(error_text, error_capacity, "invalid added special token"); return LM_ERR_UNSUPPORTED; }
                const uint32_t id = static_cast<uint32_t>(id_value->number);
                if (id >= tokenizer->id_to_token.size() || tokenizer->id_to_token[id] != content->string) { delete tokenizer; set_error(error_text, error_capacity, "added token is absent from vocab"); return LM_ERR_PARSE; }
            }
        }
        std::string unk; if (!string_field(*model, "unk_token", &unk, false)) { delete tokenizer; set_error(error_text, error_capacity, "invalid unknown token field"); return LM_ERR_PARSE; }
        if (!unk.empty()) { const auto found = tokenizer->token_to_id.find(unk); if (found == tokenizer->token_to_id.end()) { delete tokenizer; set_error(error_text, error_capacity, "unknown token is absent from vocab"); return LM_ERR_PARSE; } tokenizer->unk_present = 1u; tokenizer->unk_id = found->second; }
        std::string prefix, suffix; if (!string_field(*model, "continuing_subword_prefix", &prefix, false) || !string_field(*model, "end_of_word_suffix", &suffix, false)) { delete tokenizer; set_error(error_text, error_capacity, "invalid BPE affix field"); return LM_ERR_PARSE; }
        if (!prefix.empty() || !suffix.empty()) { delete tokenizer; set_error(error_text, error_capacity, "BPE subword affixes are unsupported in raw UTF-8 mode"); return LM_ERR_UNSUPPORTED; }
        bool byte_fallback = false, fuse_unk = false, ignore_merges = false;
        if (!bool_field(*model, "byte_fallback", &byte_fallback, false) ||
            !bool_field(*model, "fuse_unk", &fuse_unk, false) ||
            !bool_field(*model, "ignore_merges", &ignore_merges, false) ||
            (!sentencepiece && (byte_fallback || fuse_unk)) || ignore_merges) {
            delete tokenizer; set_error(error_text, error_capacity, "BPE byte or merge policy is unsupported"); return LM_ERR_UNSUPPORTED;
        }
        tokenizer->byte_fallback = byte_fallback ? 1u : 0u;
        tokenizer->fuse_unk = fuse_unk ? 1u : 0u;
        const JsonValue *dropout = field(*model, "dropout"); if (dropout && (dropout->kind != JsonValue::Null && (dropout->kind != JsonValue::Number || dropout->number != 0.0))) { delete tokenizer; set_error(error_text, error_capacity, "BPE dropout is unsupported"); return LM_ERR_UNSUPPORTED; }
        for (size_t rank = 0u; rank < merges->array.size(); ++rank) {
            std::string left, right; if (!parse_merge(merges->array[rank], &left, &right) || !valid_utf8(left) || !valid_utf8(right)) { delete tokenizer; set_error(error_text, error_capacity, "invalid BPE merge"); return LM_ERR_PARSE; }
            const auto left_id = tokenizer->token_to_id.find(left), right_id = tokenizer->token_to_id.find(right); if (left_id == tokenizer->token_to_id.end() || right_id == tokenizer->token_to_id.end()) { delete tokenizer; set_error(error_text, error_capacity, "BPE merge input is absent from vocab"); return LM_ERR_PARSE; }
            const std::string output = left + right; const auto output_id = tokenizer->token_to_id.find(output); if (output_id == tokenizer->token_to_id.end()) { delete tokenizer; set_error(error_text, error_capacity, "BPE merge output is absent from vocab"); return LM_ERR_PARSE; }
            const Pair pair{left_id->second, right_id->second}; if (!tokenizer->merges.emplace(pair, Merge{static_cast<uint32_t>(rank), output_id->second}).second) { delete tokenizer; set_error(error_text, error_capacity, "duplicate BPE merge"); return LM_ERR_PARSE; }
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
    std::string normalized;
    const char *source = text;
    size_t source_bytes = text_bytes;
    if (tokenizer->sentencepiece) {
        if (text_bytes != 0u && !normalize_sentencepiece(text, text_bytes, &normalized)) return LM_ERR_PARSE;
        if (text_bytes != 0u) { source = normalized.data(); source_bytes = normalized.size(); }
    }
    if (text_bytes == 0u && !tokenizer->add_bos) return LM_OK;
    if (!out_tokens || token_capacity == 0u) return LM_ERR_CAPACITY;
    std::vector<std::pair<uint32_t, std::string>> symbols;
    try {
        for (size_t position = 0u; position < source_bytes;) {
            std::string symbol; if (!next_utf8(source, source_bytes, &position, &symbol)) return LM_ERR_PARSE; const auto found = tokenizer->token_to_id.find(symbol);
            if (found == tokenizer->token_to_id.end()) {
                if (tokenizer->byte_fallback) {
                    static const char hex[] = "0123456789ABCDEF";
                    for (unsigned char byte : symbol) {
                        std::string byte_token("<0x00>"); byte_token[3] = hex[byte >> 4u]; byte_token[4] = hex[byte & 0x0fu];
                        const auto byte_id = tokenizer->token_to_id.find(byte_token);
                        if (byte_id == tokenizer->token_to_id.end()) return LM_ERR_UNSUPPORTED;
                        symbols.emplace_back(byte_id->second, std::move(byte_token));
                    }
                } else {
                    if (!tokenizer->unk_present) return LM_ERR_UNSUPPORTED;
                    if (!tokenizer->fuse_unk || symbols.empty() || symbols.back().first != tokenizer->unk_id)
                        symbols.emplace_back(tokenizer->unk_id, std::string());
                }
            } else symbols.emplace_back(found->second, std::move(symbol));
        }
        for (;;) {
            size_t best = symbols.size(); Merge selected{}; bool have = false;
            for (size_t i = 0u; i + 1u < symbols.size(); ++i) { const auto found = tokenizer->merges.find(Pair{symbols[i].first, symbols[i + 1u].first}); if (found != tokenizer->merges.end() && (!have || found->second.rank < selected.rank)) { best = i; selected = found->second; have = true; } }
            if (!have) break;
            symbols[best] = {selected.output, symbols[best].second + symbols[best + 1u].second}; symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best + 1u));
        }
    } catch (const std::bad_alloc &) { return LM_ERR_CAPACITY; }
    const size_t required = symbols.size() + (tokenizer->add_bos ? 1u : 0u);
    *out_count = required;
    if (required > token_capacity) return LM_ERR_CAPACITY;
    size_t output = 0u;
    if (tokenizer->add_bos) out_tokens[output++] = tokenizer->bos_id;
    for (size_t i = 0u; i < symbols.size(); ++i) out_tokens[output++] = symbols[i].first;
    return LM_OK;
}

lm_status lm_tokenizer_decode(const lm_tokenizer *tokenizer, const uint32_t *tokens, size_t token_count, char *out_text, size_t out_capacity, size_t *out_bytes) {
    if (!tokenizer || !out_bytes || (!tokens && token_count != 0u)) return LM_ERR_ARGUMENT;
    std::string decoded;
    try {
        for (size_t i = 0u; i < token_count; ++i) {
            if (tokens[i] >= tokenizer->id_to_token.size() ||
                (tokenizer->id_to_token[tokens[i]].empty() && tokenizer->token_to_id.find(std::string()) == tokenizer->token_to_id.end()))
                return LM_ERR_RANGE;
            const std::string &value = tokenizer->id_to_token[tokens[i]];
            if (tokenizer->sentencepiece && (value == "<s>" || value == "</s>")) continue;
            std::string byte;
            if (tokenizer->sentencepiece && decode_byte_fallback(value, &byte)) decoded.append(byte);
            else decoded.append(value);
        }
        if (tokenizer->sentencepiece) {
            const std::string marker("\xE2\x96\x81");
            for (size_t position = 0u; (position = decoded.find(marker, position)) != std::string::npos;) {
                decoded.replace(position, marker.size(), " ");
                ++position;
            }
            if (!decoded.empty() && decoded[0] == ' ') decoded.erase(0u, 1u);
        }
    } catch (const std::bad_alloc &) { return LM_ERR_CAPACITY; }
    *out_bytes = decoded.size();
    if (!out_text && out_capacity == 0u) return LM_OK;
    if (!out_text || out_capacity <= decoded.size()) return LM_ERR_CAPACITY;
    if (!decoded.empty()) std::memcpy(out_text, decoded.data(), decoded.size());
    out_text[decoded.size()] = '\0';
    return LM_OK;
}
