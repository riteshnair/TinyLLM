#include "lm/lm.h"

#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#include <cstring>
#endif

lm_status lm_rocr_runtime_probe(lm_rocr_runtime_info *out_info) {
    if (!out_info) return LM_ERR_ARGUMENT;
    std::memset(out_info, 0, sizeof(*out_info));
#if defined(__unix__) || defined(__APPLE__)
    const char *names[] = {"libhsa-runtime64.so", "libhsa-runtime64.so.1", "libhsa-runtime64.dylib"};
    void *library = nullptr;
    const char *selected = nullptr;
    for (const char *name : names) {
        library = dlopen(name, RTLD_NOW | RTLD_LOCAL);
        if (library) { selected = name; break; }
    }
    if (!library) return LM_ERR_UNSUPPORTED;
    out_info->runtime_present = 1u;
    std::strncpy(out_info->library, selected, sizeof(out_info->library) - 1u);
    using hsa_init_fn = int (*)();
    using hsa_shutdown_fn = int (*)();
    const hsa_init_fn init = reinterpret_cast<hsa_init_fn>(dlsym(library, "hsa_init"));
    const hsa_shutdown_fn shutdown = reinterpret_cast<hsa_shutdown_fn>(dlsym(library, "hsa_shut_down"));
    if (init && shutdown && init() == 0) {
        out_info->initialized = 1u;
        (void)shutdown();
    }
    dlclose(library);
    return out_info->initialized ? LM_OK : LM_ERR_UNSUPPORTED;
#else
    return LM_ERR_UNSUPPORTED;
#endif
}
