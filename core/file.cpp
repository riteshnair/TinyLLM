#include "lm/lm.h"

#include <fstream>
#include <limits>

struct lm_file {
    std::ifstream stream;
    uint64_t bytes;
};

lm_status lm_file_open(const char *path, lm_file **out_file) {
    if (!path || !out_file) return LM_ERR_ARGUMENT;
    *out_file = nullptr;
    lm_file *file = new lm_file();
    file->stream.open(path, std::ios::binary | std::ios::ate);
    if (!file->stream) { delete file; return LM_ERR_IO; }
    const std::streamoff end = file->stream.tellg();
    if (end < 0) { delete file; return LM_ERR_IO; }
    file->bytes = static_cast<uint64_t>(end);
    file->stream.seekg(0, std::ios::beg);
    if (!file->stream) { delete file; return LM_ERR_IO; }
    *out_file = file;
    return LM_OK;
}

void lm_file_close(lm_file *file) {
    delete file;
}

lm_status lm_file_size(const lm_file *file, uint64_t *out_bytes) {
    if (!file || !out_bytes) return LM_ERR_ARGUMENT;
    *out_bytes = file->bytes;
    return LM_OK;
}

lm_status lm_file_span_make(lm_file *file, uint64_t offset, uint64_t bytes, lm_file_span *out_span) {
    if (!file || !out_span) return LM_ERR_ARGUMENT;
    if (offset > file->bytes || bytes > file->bytes - offset) return LM_ERR_RANGE;
    out_span->file = file;
    out_span->offset = offset;
    out_span->bytes = bytes;
    return LM_OK;
}

lm_status lm_file_span_read(const lm_file_span *span, uint64_t offset, void *dst, size_t bytes) {
    if (!span || !span->file || (!dst && bytes != 0u)) return LM_ERR_ARGUMENT;
    if (offset > span->bytes || static_cast<uint64_t>(bytes) > span->bytes - offset) return LM_ERR_RANGE;
    if (span->offset > std::numeric_limits<uint64_t>::max() - offset) return LM_ERR_RANGE;
    return lm_file_read(span->file, span->offset + offset, dst, bytes);
}

lm_status lm_file_read(lm_file *file, uint64_t offset, void *dst, size_t bytes) {
    if (!file || (!dst && bytes != 0u)) return LM_ERR_ARGUMENT;
    if (offset > file->bytes || static_cast<uint64_t>(bytes) > file->bytes - offset)
        return LM_ERR_RANGE;
    if (bytes == 0u) return LM_OK;
    if (offset > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) return LM_ERR_RANGE;
    file->stream.clear();
    file->stream.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!file->stream) return LM_ERR_IO;
    file->stream.read(static_cast<char *>(dst), static_cast<std::streamsize>(bytes));
    return file->stream.gcount() == static_cast<std::streamsize>(bytes) ? LM_OK : LM_ERR_IO;
}
