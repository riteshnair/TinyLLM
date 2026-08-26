#include "lm/lm.h"

#include <fstream>
#include <limits>
#include <cstring>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

struct lm_file {
    std::ifstream stream;
    uint64_t bytes;
#if defined(__unix__) || defined(__APPLE__)
    int descriptor;
    const unsigned char *mapped;
    size_t mapped_bytes;
#endif
};

lm_status lm_file_open(const char *path, lm_file **out_file) {
    if (!path || !out_file) return LM_ERR_ARGUMENT;
    *out_file = nullptr;
    lm_file *file = new lm_file{};
#if defined(__unix__) || defined(__APPLE__)
    file->descriptor = -1;
    file->mapped = nullptr;
    file->mapped_bytes = 0u;
#endif
    file->stream.open(path, std::ios::binary | std::ios::ate);
    if (!file->stream) { delete file; return LM_ERR_IO; }
    const std::streamoff end = file->stream.tellg();
    if (end < 0) { delete file; return LM_ERR_IO; }
    file->bytes = static_cast<uint64_t>(end);
    file->stream.seekg(0, std::ios::beg);
    if (!file->stream) { delete file; return LM_ERR_IO; }
#if defined(__unix__) || defined(__APPLE__)
    if (file->bytes != 0u && file->bytes <= static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        file->descriptor = open(path, O_RDONLY);
        if (file->descriptor >= 0) {
            void *mapped = mmap(nullptr, static_cast<size_t>(file->bytes), PROT_READ, MAP_PRIVATE,
                                file->descriptor, 0);
            if (mapped != MAP_FAILED) {
                file->mapped = static_cast<const unsigned char *>(mapped);
                file->mapped_bytes = static_cast<size_t>(file->bytes);
            } else {
                close(file->descriptor);
                file->descriptor = -1;
            }
        }
    }
#endif
    *out_file = file;
    return LM_OK;
}

void lm_file_close(lm_file *file) {
    if (!file) return;
#if defined(__unix__) || defined(__APPLE__)
    if (file->mapped) munmap(const_cast<unsigned char *>(file->mapped), file->mapped_bytes);
    if (file->descriptor >= 0) close(file->descriptor);
#endif
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
#if defined(__unix__) || defined(__APPLE__)
    if (file->mapped && offset <= static_cast<uint64_t>(file->mapped_bytes) &&
        bytes <= file->mapped_bytes - static_cast<size_t>(offset)) {
        std::memcpy(dst, file->mapped + static_cast<size_t>(offset), bytes);
        return LM_OK;
    }
#endif
    if (offset > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) return LM_ERR_RANGE;
    file->stream.clear();
    file->stream.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!file->stream) return LM_ERR_IO;
    file->stream.read(static_cast<char *>(dst), static_cast<std::streamsize>(bytes));
    return file->stream.gcount() == static_cast<std::streamsize>(bytes) ? LM_OK : LM_ERR_IO;
}
