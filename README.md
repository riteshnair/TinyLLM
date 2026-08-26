# TinyLLM

TinyLLM is a compact, Vulkan-first modular LLM inference engine with a strict C99-compatible public ABI and C++17 implementation. The CPU path remains the correctness oracle and fallback; native GGUF packed storage is preserved without model-wide F32 expansion. Start with [`ENGINEERING_GUIDE.md`](ENGINEERING_GUIDE.md), the live [`MODULE_STATUS.md`](MODULE_STATUS.md), and the focused research notes under [`docs/`](docs/).

## Current validated slice

The repository currently provides bounded GGUF/SafeTensors inspection, exact native GGML Q4_0/Q8_0/Q4_K storage contracts, lazy file-span bindings, replaceable Vulkan packed dot and matvec shaders, paged KV metadata plus bounded opaque payloads with copy-on-write, exact bounded GGUF vocabulary parsing, deterministic sampling, and a narrow native single-layer path.

The narrow path can validate a one-layer, one-head, equal-Q/KV-width profile; execute native Q8_0 MLP and attention operations; use an explicit F32 KV payload; produce logits; encode a bounded prompt; and run a bounded deterministic generation loop. It is exposed through the library API and the CLI’s `--generate` entrypoint. This is a real tested vertical slice, not arbitrary-checkpoint compatibility.

The engine deliberately returns named unsupported errors for capabilities outside the validated profile. Full multi-layer architecture metadata, broad tokenizer semantics, GQA variants, quantized KV policy, model residency scheduling, batching, streaming, server/WebUI integration, distributed execution, and future ROCr/HSA backends remain pending.

## Build

```sh
cmake -S . -B build -DLM_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Run the control-plane CLI:

```sh
./build/tiny-lm \
  --backend cpu \
  --load lazy \
  --kv-dtype q8 \
  --context 2048 \
  --threads 2 \
  --trace \
  --model fixture.gguf
```

Use `--help` for the supported switches. `--backend vulkan` performs capability/device validation and selects the Vulkan path when a compute device is available; it does not imply arbitrary full-model compatibility.

For the validated narrow profile, use:

```sh
./build/tiny-lm --backend cpu --model narrow.gguf \\
  --generate --prompt "ab" --max-new-tokens 8
```

The CLI accepts only models that satisfy the narrow profile’s exact tensor names, shapes, native matrix types, and GGUF vocabulary contract.

## Source layout

| Path | Responsibility |
|---|---|
| `include/lm/lm.h` | Stable public C ABI |
| `core/lm.cpp` | Configuration, backend selection, runtime and probes |
| `core/model.cpp` | Bounded model inspection, exact vocabulary, and native model bindings |
| `core/kv.cpp` | Paged KV metadata and bounded opaque payload COW |
| `core/file.cpp` | Bounded read-only on-demand file access |
| `core/tensor.cpp` | Dtype, tensor-view, and host-buffer primitives |
| `core/decoder.cpp` | Narrow native MLP, attention, logits, tokenizer bridge, and generation path |
| `core/kernel.cpp` | Replaceable CPU/Vulkan/DP4 kernel registry |
| `vulkan/device.cpp` | Vulkan device discovery |
| `vulkan/dp4.cpp` | Reference packed-int8 DP4 dispatch |
| `vulkan/shaders/dot_i8_dp4.comp` | DP4 compute shader source |
| `cli/main.cpp` | Local control CLI, device listing, and narrow generation entrypoint |
| `tests/test_core.cpp` | Focused unit and smoke tests |
| `CMakeLists.txt` | Minimal build definition |

Generated files belong in `build/`, which is ignored by Git. Keep model files outside the repository unless they are small, synthetic test fixtures.

## Development contract

Every new module must define inputs, outputs, ownership, synchronization, errors, backend applicability, budgets, tests, CLI fields, acceptance criteria, and exclusions before implementation. Keep the scalar CPU path as the reference. Add Vulkan or vendor variants only after a differential test exists. Use the probe bus and deterministic fixtures to locate the first divergent boundary.
