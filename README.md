# TinyLLM

TinyLLM is a compact, Vulkan-first modular LLM inference engine with a strict C99-compatible public ABI and C++17 implementation. The CPU path remains the correctness oracle and fallback; native GGUF packed storage is preserved without model-wide F32 expansion. Start with [`ENGINEERING_GUIDE.md`](ENGINEERING_GUIDE.md), the live [`MODULE_STATUS.md`](MODULE_STATUS.md), and the focused research notes under [`docs/`](docs/).

## Current validated slice

The repository currently provides bounded GGUF/SafeTensors inspection, exact native GGML Q4_0/Q8_0/Q4_K storage contracts, model-aware GGUF split-set and explicit SafeTensors file-set tensor binding, SafeTensors metadata and contiguous token-string contracts, demand-paged file-span bindings, bounded windows and cross-shard reads, replaceable Vulkan packed and F32 scalar dot/matvec shaders, reusable Vulkan F32 and packed dispatch contexts with grow-only buffers, paged KV metadata plus opaque payloads and explicit F16/BF16/Q8/Q6/Q4 CPU codecs with copy-on-write, wired into native transformer attention, deterministic advanced sampling, logits processors, stop-string suppression, extractive context compaction, a bounded prefix-cache table with identity-safe longest-prefix lookup, LRU eviction, and validated export/import, native CPU sliding-window attention over a bounded KV range, an explicit linear RoPE position-scale policy, history-aware generation sampling, a reusable allowlist logits processor, synchronous native-generation trace events, and a narrow native multi-layer path.

The standalone tokenizer API validates raw UTF-8 tokenizer.json BPE models with deterministic ordered merges and one exact SentencePiece-style BPE pipeline used by the validated real checkpoint: `▁` normalization, BOS post-processing, byte fallback, fused unknown handling, and decoder marker restoration. The narrow GGUF path validates a multi-layer Llama metadata profile when every layer satisfies the same explicit shape and native-format contract; preserves Q8_0 storage; executes CPU and Vulkan Q8_0 MLP, attention, logits, and deterministic generation; and explicitly aliases omitted GGUF output weights to the token embedding. SafeTensors additionally supports explicit standard-HF Llama `config.json` and `tokenizer.json` attachment, standard HF tensor names, exact F32/F16/BF16 caller-scratch matrices, and CPU transformer generation; F16/BF16 conversion remains limited to requested matrices and CPU-only execution. The downloaded real-model evidence is recorded in [`docs/real_model_validation_notes.md`](docs/real_model_validation_notes.md). The library also exposes token callbacks, sequential text batches, and a loopback `--server` adapter with completion/chat routes, role-preserving `messages` parsing, and one SSE event per generated token followed by `[DONE]`. These are narrow validated profiles, not arbitrary-checkpoint compatibility.

The engine deliberately returns named unsupported errors for capabilities outside the validated profiles. Architecture families beyond the bounded Llama scalar contract, SafeTensors F16/BF16 Vulkan execution, typed-KV Vulkan attention, tokenizer pipelines beyond the exact raw-UTF8, embedded SmoLLM GPT-2 byte-level BPE, and validated SentencePiece-style contracts, implicit tokenizer/config discovery, full model/KV scheduler integration of prefix reuse, cache residency scheduling, packed model-graph device residency, asynchronous shard scheduling, heterogeneous MoE formats, native concurrent batching/admission, HTTP cancellation, authentication/TLS/WebUI assets, distributed execution, learned semantic compression, NTK/dynamic RoPE policies, and direct ROCr/HSA model kernels remain unsupported. The speculative API currently provides deterministic n-gram proposal, target-greedy verification, rollback-equivalent replacement semantics, and adaptive depth policy; it is not yet wired into the model generation loop. A dynamic ROCr runtime probe is present, but this host has no HSA runtime or AMDGPU device and no direct code-object execution is claimed.

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

Use `--help` for the supported switches. For standard HF SafeTensors generation, pass both `--tokenizer PATH` and `--config PATH`; no unrelated sidecars are guessed. `--backend vulkan` performs capability/device validation and selects the Vulkan path when a compute device is available; it does not imply arbitrary full-model compatibility.

For the validated narrow profile, use:

```sh
./build/tiny-lm --backend cpu --model narrow.gguf \
  --generate --prompt "ab" --max-new-tokens 8
```

The CLI accepts the narrow validated multi-layer GGUF profile and checks every layer’s exact tensor names, shapes, native matrix type, and GGUF vocabulary contract before generation; decoding controls include greedy, top-k, top-p, typical, min-p, temperature, repetition, frequency, and presence policies. Native generation now passes prompt and generated-token history to sampling processors and penalties, and callers can use `lm_logits_allowlist_processor` with a bounded token allowlist. Library callers and the CLI may request a bounded attention window (`0` for full causal KV), the explicit linear RoPE position policy (`rope_scale=0` or `1` for unchanged positions; positive values divide absolute positions), and bounded prompt prefill chunks; these controls currently execute through the narrow native sequential path. The library API additionally exposes model-bound selected-expert Q4_K staging for the validated expert-major rank-3 Mixtral contract, standalone speculative/prefix-cache policies, a callback-level continuous-batch scheduler, and generation trace stages: begin, prefill token, decode token, and completion. Trace callbacks are synchronous and caller-owned; they do not provide profiling or concurrency guarantees. The repository-wide source allowance is approximately 30 MiB for human-maintained core and backend code; weights, mapped files, generated shaders, and runtime payloads remain separate budgets.

## Source layout

| Path | Responsibility |
|---|---|
| `include/lm/lm.h` | Stable public C ABI |
| `core/lm.cpp` | Configuration, backend selection, runtime and probes |
| `core/model.cpp` | Bounded model inspection, exact vocabulary, model-aware shard sets, and native model bindings |
| `core/tokenizer.cpp` | Strict tokenizer.json BPE parser, exact validated normalizer/post-processor branch, and merge-rank executor |
| `core/kv.cpp` | Paged KV metadata, opaque payload COW, and explicit typed KV codecs |
| `core/file.cpp` | Bounded demand-paged or stream-backed file access |
| `core/tensor.cpp` | Dtype, tensor-view, and host-buffer primitives |
| `core/decoder.cpp` | Narrow native MLP, windowed attention, linear-RoPE option, logits, tokenizer bridge, and stop-aware generation path |
| `core/sampling.cpp` | Deterministic sampling, penalties, filtering, allowlist masking, and logits processors |
| `core/decode_policy.cpp` | N-gram proposal, greedy verification, and adaptive speculation policy |
| `core/prefix_cache.cpp` | Bounded identity-safe prefix table, LRU policy, and framed export/import |
| `core/batch_scheduler.cpp` | Fair bounded request admission, cancellation, completion, and step telemetry |
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
