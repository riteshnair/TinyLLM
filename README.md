# TinyLLM

TinyLLM is a compact, Vulkan-first modular LLM inference engine with a strict C99-compatible public ABI and C++17 implementation. The CPU path remains the correctness oracle and fallback; native GGUF packed storage is preserved without model-wide F32 expansion. Start with [`ENGINEERING_GUIDE.md`](ENGINEERING_GUIDE.md), the live [`MODULE_STATUS.md`](MODULE_STATUS.md), and the focused research notes under [`docs/`](docs/).

## Current validated slice

The repository currently provides bounded GGUF/SafeTensors inspection, exact native GGML Q4_0/Q8_0/Q4_K storage contracts, model-aware GGUF split-set and explicit SafeTensors file-set tensor binding, SafeTensors metadata and contiguous token-string contracts, demand-paged file-span bindings, bounded windows and cross-shard reads, replaceable Vulkan packed and F32 scalar dot/matvec shaders, reusable Vulkan F32 and packed dispatch contexts with grow-only buffers, paged KV metadata plus opaque payloads and explicit F16/BF16/Q8/Q6/Q4 CPU codecs with copy-on-write, wired into native transformer attention, deterministic sampling, extractive context compaction, and a narrow native multi-layer path.

The standalone tokenizer API validates raw UTF-8 tokenizer.json BPE models with deterministic ordered merges. The narrow GGUF path validates a multi-layer Llama metadata profile with two query heads and one KV head when every layer satisfies the same explicit shape and native-format contract; executes native Q8_0 MLP and attention operations; uses an explicit F32 or typed KV payload per layer; produces logits; encodes a bounded prompt; and runs a bounded deterministic generation loop. SafeTensors additionally supports metadata-bound Llama scalar architecture, contiguous `tokenizer.token.N` vocabulary entries, exact F32/F16/BF16 caller-scratch matrices, CPU F16/BF16 matrix and narrow MLP inference, scalar Vulkan F32 matvec differentials, and CPU transformer/numeric-token generation; the F16/BF16 path converts only the requested matrix and remains explicitly CPU-only. The library also exposes token callbacks, sequential text batches, and a loopback `--server` adapter with completion/chat routes, role-preserving `messages` parsing, and one SSE event per generated token followed by `[DONE]`. These are real tested vertical slices, not arbitrary-checkpoint compatibility.

The engine deliberately returns named unsupported errors for capabilities outside the validated profiles. Architecture families beyond the bounded Llama scalar contract, SafeTensors F16/BF16 Vulkan execution, typed-KV Vulkan attention, byte-level/normalized tokenizer semantics, tokenizer post-processing and special-token policies, prefix-aware KV identity and cache residency scheduling, packed model-graph device residency, asynchronous shard scheduling, heterogeneous MoE formats, concurrent batching/admission, cancellation, authentication/TLS/WebUI assets, distributed execution, learned semantic compression, and direct ROCr/HSA model kernels remain unsupported. A dynamic ROCr runtime probe is present, but this host has no HSA runtime or AMDGPU device and no direct code-object execution is claimed.

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
./build/tiny-lm --backend cpu --model narrow.gguf \
  --generate --prompt "ab" --max-new-tokens 8
```

The CLI accepts the narrow validated multi-layer GGUF profile and checks every layer’s exact tensor names, shapes, native matrix type, and GGUF vocabulary contract before generation; the library API additionally exposes model-bound selected-expert Q4_K staging for the validated expert-major rank-3 Mixtral contract.

## Source layout

| Path | Responsibility |
|---|---|
| `include/lm/lm.h` | Stable public C ABI |
| `core/lm.cpp` | Configuration, backend selection, runtime and probes |
| `core/model.cpp` | Bounded model inspection, exact vocabulary, model-aware shard sets, and native model bindings |
| `core/tokenizer.cpp` | Strict raw-UTF-8 tokenizer.json BPE parser and merge-rank executor |
| `core/kv.cpp` | Paged KV metadata, opaque payload COW, and explicit typed KV codecs |
| `core/file.cpp` | Bounded demand-paged or stream-backed file access |
| `core/tensor.cpp` | Dtype, tensor-view, and host-buffer primitives |
| `core/decoder.cpp` | Narrow native MLP, attention, logits, tokenizer bridge, and generation path |
| `core/kernel.cpp` | Replaceable CPU/Vulkan/DP4 kernel registry |
| `vulkan/device.cpp` | Vulkan device discovery |
| `vulkan/dp4.cpp` | Reference packed-int8 DP4 dispatch |
| `vulkan/matvec_f32.cpp` | SafeTensors F32 scalar Vulkan matvec route |
| `vulkan/f32_context.cpp` | Reusable F32 Vulkan dispatch context |
| `vulkan/packed_context.cpp` | Reusable Q4_K/Q8_0 Vulkan dispatch context |
| `vulkan/shaders/dot_i8_dp4.comp` | DP4 compute shader source |
| `cli/main.cpp` | Local control CLI, device listing, and narrow multi-layer generation entrypoint |
| `tests/test_core.cpp` | Focused unit and smoke tests |
| `CMakeLists.txt` | Minimal build definition |

Generated files belong in `build/`, which is ignored by Git. Keep model files outside the repository unless they are small, synthetic test fixtures.

## Development contract

Every new module must define inputs, outputs, ownership, synchronization, errors, backend applicability, budgets, tests, CLI fields, acceptance criteria, and exclusions before implementation. Keep the scalar CPU path as the reference. Add Vulkan or vendor variants only after a differential test exists. Use the probe bus and deterministic fixtures to locate the first divergent boundary.
