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

#include <algorithm>
#include <new>
#include <vector>

struct lm_file_window {
    lm_file *file;
    uint64_t window_bytes;
    uint64_t offset;
    uint64_t valid_bytes;
    std::vector<unsigned char> data;
};

struct lm_file_shard_set {
    std::vector<lm_file *> files;
    std::vector<uint64_t> starts;
    uint64_t bytes;
};

void lm_file_shard_set_close(lm_file_shard_set *set);

lm_status lm_file_window_create(lm_file *file, uint64_t window_bytes, lm_file_window **out_window) {
    if (!file || !out_window || window_bytes == 0u || window_bytes > (64ull << 20u)) return LM_ERR_ARGUMENT;
    *out_window = nullptr;
    lm_file_window *window = new (std::nothrow) lm_file_window{};
    if (!window) return LM_ERR_CAPACITY;
    try {
        window->data.resize(static_cast<size_t>(window_bytes));
    } catch (...) {
        delete window;
        return LM_ERR_CAPACITY;
    }
    window->file = file;
    window->window_bytes = window_bytes;
    *out_window = window;
    return LM_OK;
}

void lm_file_window_destroy(lm_file_window *window) { delete window; }

lm_status lm_file_window_prefetch(lm_file_window *window, uint64_t offset, uint64_t bytes) {
    if (!window || !window->file) return LM_ERR_ARGUMENT;
    if (bytes > window->window_bytes || offset > window->file->bytes || bytes > window->file->bytes - offset)
        return LM_ERR_RANGE;
    if (bytes == 0u) {
        window->offset = offset;
        window->valid_bytes = 0u;
        return LM_OK;
    }
    const lm_status status = lm_file_read(window->file, offset, window->data.data(), static_cast<size_t>(bytes));
    if (status != LM_OK) return status;
    window->offset = offset;
    window->valid_bytes = bytes;
    return LM_OK;
}

lm_status lm_file_window_read(lm_file_window *window, uint64_t offset, void *dst, size_t bytes) {
    if (!window || !window->file || (!dst && bytes != 0u)) return LM_ERR_ARGUMENT;
    if (static_cast<uint64_t>(bytes) > window->window_bytes) return LM_ERR_CAPACITY;
    if (offset > window->file->bytes || static_cast<uint64_t>(bytes) > window->file->bytes - offset) return LM_ERR_RANGE;
    const uint64_t request_end = offset + static_cast<uint64_t>(bytes);
    const uint64_t window_end = window->offset + window->valid_bytes;
    if (offset < window->offset || request_end > window_end) {
        const uint64_t load = std::min(window->window_bytes, window->file->bytes - offset);
        const lm_status status = lm_file_window_prefetch(window, offset, load);
        if (status != LM_OK) return status;
    }
    if (bytes != 0u) std::memcpy(dst, window->data.data() + static_cast<size_t>(offset - window->offset), bytes);
    return LM_OK;
}

lm_status lm_file_shard_set_open(const char *const *paths, size_t count, lm_file_shard_set **out_set) {
    if (!paths || !out_set || count == 0u || count > 64u) return LM_ERR_ARGUMENT;
    *out_set = nullptr;
    lm_file_shard_set *set = new (std::nothrow) lm_file_shard_set{};
    if (!set) return LM_ERR_CAPACITY;
    try {
        set->files.reserve(count);
        set->starts.reserve(count);
    } catch (...) {
        delete set;
        return LM_ERR_CAPACITY;
    }
    uint64_t total = 0u;
    for (size_t i = 0u; i < count; ++i) {
        if (!paths[i]) { lm_file_shard_set_close(set); return LM_ERR_ARGUMENT; }
        lm_file *file = nullptr;
        const lm_status opened = lm_file_open(paths[i], &file);
        if (opened != LM_OK) { lm_file_shard_set_close(set); return opened; }
        uint64_t size = 0u;
        const lm_status sized = lm_file_size(file, &size);
        if (sized != LM_OK || size > std::numeric_limits<uint64_t>::max() - total) {
            lm_file_close(file);
            lm_file_shard_set_close(set);
            return sized != LM_OK ? sized : LM_ERR_CAPACITY;
        }
        set->starts.push_back(total);
        set->files.push_back(file);
        total += size;
    }
    set->bytes = total;
    *out_set = set;
    return LM_OK;
}

void lm_file_shard_set_close(lm_file_shard_set *set) {
    if (!set) return;
    for (lm_file *file : set->files) lm_file_close(file);
    delete set;
}

lm_status lm_file_shard_set_size(const lm_file_shard_set *set, uint64_t *out_bytes) {
    if (!set || !out_bytes) return LM_ERR_ARGUMENT;
    *out_bytes = set->bytes;
    return LM_OK;
}

lm_status lm_file_shard_set_read(lm_file_shard_set *set, uint64_t offset, void *dst, size_t bytes) {
    if (!set || (!dst && bytes != 0u)) return LM_ERR_ARGUMENT;
    if (offset > set->bytes || static_cast<uint64_t>(bytes) > set->bytes - offset) return LM_ERR_RANGE;
    uint64_t cursor = offset;
    size_t remaining = bytes;
    unsigned char *output = static_cast<unsigned char *>(dst);
    for (size_t i = 0u; remaining != 0u && i < set->files.size(); ++i) {
        uint64_t shard_size = 0u;
        const lm_status sized = lm_file_size(set->files[i], &shard_size);
        if (sized != LM_OK) return sized;
        const uint64_t start = set->starts[i];
        if (cursor < start || cursor >= start + shard_size) continue;
        const uint64_t local = cursor - start;
        const size_t chunk = static_cast<size_t>(std::min<uint64_t>(remaining, shard_size - local));
        const lm_status status = lm_file_read(set->files[i], local, output, chunk);
        if (status != LM_OK) return status;
        cursor += chunk;
        output += chunk;
        remaining -= chunk;
    }
    return remaining == 0u ? LM_OK : LM_ERR_STATE;
}
