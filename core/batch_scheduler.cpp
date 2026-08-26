#include "lm/lm.h"

#include <algorithm>
#include <new>
#include <vector>

struct lm_batch_scheduler {
    uint32_t capacity;
    size_t cursor;
    lm_batch_scheduler_stats stats;
    std::vector<uint32_t> request_ids;
};

lm_status lm_batch_scheduler_create(uint32_t capacity, lm_batch_scheduler **out_scheduler) {
    if (!out_scheduler || capacity == 0u || capacity > 4096u) return LM_ERR_ARGUMENT;
    *out_scheduler = nullptr;
    try {
        lm_batch_scheduler *scheduler = new lm_batch_scheduler();
        scheduler->capacity = capacity;
        scheduler->cursor = 0u;
        scheduler->stats = {0u, capacity, 0u, 0u, 0u, 0u};
        scheduler->request_ids.reserve(capacity);
        *out_scheduler = scheduler;
        return LM_OK;
    } catch (const std::bad_alloc &) {
        return LM_ERR_CAPACITY;
    }
}

void lm_batch_scheduler_destroy(lm_batch_scheduler *scheduler) {
    delete scheduler;
}

lm_status lm_batch_scheduler_submit(lm_batch_scheduler *scheduler, uint32_t request_id) {
    if (!scheduler || request_id == 0u) return LM_ERR_ARGUMENT;
    if (std::find(scheduler->request_ids.begin(), scheduler->request_ids.end(), request_id) != scheduler->request_ids.end())
        return LM_ERR_STATE;
    if (scheduler->request_ids.size() >= scheduler->capacity) return LM_ERR_CAPACITY;
    try { scheduler->request_ids.push_back(request_id); }
    catch (const std::bad_alloc &) { return LM_ERR_CAPACITY; }
    scheduler->stats.active = static_cast<uint32_t>(scheduler->request_ids.size());
    ++scheduler->stats.submitted;
    return LM_OK;
}

lm_status lm_batch_scheduler_cancel(lm_batch_scheduler *scheduler, uint32_t request_id) {
    if (!scheduler || request_id == 0u) return LM_ERR_ARGUMENT;
    const auto found = std::find(scheduler->request_ids.begin(), scheduler->request_ids.end(), request_id);
    if (found == scheduler->request_ids.end()) return LM_ERR_RANGE;
    const size_t index = static_cast<size_t>(found - scheduler->request_ids.begin());
    scheduler->request_ids.erase(found);
    if (scheduler->cursor > index && scheduler->cursor != 0u) --scheduler->cursor;
    if (scheduler->cursor >= scheduler->request_ids.size()) scheduler->cursor = 0u;
    scheduler->stats.active = static_cast<uint32_t>(scheduler->request_ids.size());
    ++scheduler->stats.cancelled;
    return LM_OK;
}

lm_status lm_batch_scheduler_step(lm_batch_scheduler *scheduler, lm_batch_step_callback callback,
                                  void *user, uint32_t *out_stepped) {
    if (!scheduler || !callback || !out_stepped) return LM_ERR_ARGUMENT;
    *out_stepped = 0u;
    const size_t initial = scheduler->request_ids.size();
    if (initial == 0u) return LM_OK;
    size_t visited = 0u;
    while (visited < initial && !scheduler->request_ids.empty()) {
        if (scheduler->cursor >= scheduler->request_ids.size()) scheduler->cursor = 0u;
        const uint32_t request_id = scheduler->request_ids[scheduler->cursor];
        uint32_t token = 0u;
        uint8_t finished = 0u;
        const lm_status status = callback(user, request_id, &token, &finished);
        if (status != LM_OK) return status;
        ++*out_stepped;
        ++scheduler->stats.steps;
        ++visited;
        if (finished != 0u) {
            scheduler->request_ids.erase(scheduler->request_ids.begin() + static_cast<std::ptrdiff_t>(scheduler->cursor));
            ++scheduler->stats.completed;
            if (scheduler->cursor >= scheduler->request_ids.size()) scheduler->cursor = 0u;
        } else if (!scheduler->request_ids.empty()) {
            scheduler->cursor = (scheduler->cursor + 1u) % scheduler->request_ids.size();
        }
    }
    scheduler->stats.active = static_cast<uint32_t>(scheduler->request_ids.size());
    return LM_OK;
}

lm_status lm_batch_scheduler_get_stats(const lm_batch_scheduler *scheduler,
                                       lm_batch_scheduler_stats *out_stats) {
    if (!scheduler || !out_stats) return LM_ERR_ARGUMENT;
    *out_stats = scheduler->stats;
    return LM_OK;
}
