# Vulkan backend

The backend is intentionally narrow and replaceable. The portable core selects a kernel path from an operation, a request, and a discovered capability set. The current supported paths are CPU scalar, Vulkan scalar, and Vulkan DP4 for packed signed int8 dot products.

## Current implementation

| Component | Status |
|---|---|
| Instance and physical-device discovery | Implemented |
| Compute-queue discovery | Implemented |
| Host-visible storage-buffer setup | Implemented for the reference dispatch |
| Packed int8 DP4 shader | Implemented in `shaders/dot_i8_dp4.comp` |
| Scalar F32 dot shader | Implemented in `shaders/dot_f32_scalar.comp` |
| SPIR-V build target | Implemented when `glslangValidator` is available |
| CPU/Vulkan differential fixture | Implemented for scalar F32 dot and packed-int8 DP4 |
| Contract-based operation dispatch | Implemented for scalar F32 and packed-int8 DP4 |
| Cooperative matrices | Explicitly excluded |
| Full tensor graph dispatch | Not yet implemented; current dispatch is operation-level only |

## Replacing a shader

A shader variant must preserve the operation contract and update only the registry/source identifier, specialization metadata, or backend implementation. `lm_vulkan_dispatch` is the narrow operation-level boundary that consumes the selected choice and delegates to the replaceable backend entry point. It must declare its buffer bindings, push constants, element packing, alignment, dispatch shape, required device features, numerical tolerance, and fallback path. The CPU reference test remains mandatory.

The selection order for `LM_KERNEL_AUTO` is:

```text
Vulkan DP4 when required integer-dot capability exists
-> Vulkan scalar when Vulkan exists
-> CPU scalar
```

A requested path that cannot be proven safe returns `LM_ERR_UNSUPPORTED`; it is never silently substituted. Cooperative-matrix variants are not part of the current registry or build.

## Build

From `code/`:

```sh
cmake -S . -B build -DLM_ENABLE_VULKAN=ON -DLM_BUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The generated `dot_i8_dp4.comp.spv` and `dot_f32_scalar.comp.spv` remain in `build/` and are ignored by Git. They are loaded by the differential tests from the test working directory.
