# TinyLLM Trace Deep Dive

## Scope

This audit followed the public C99 ABI through model opening, explicit SafeTensors configuration and tokenizer attachment, graph binding, native prompt prefill, KV allocation and mutation, attention, sampling, text decoding, CLI dispatch, loopback HTTP dispatch, standalone policy modules, and the Vulkan/CPU backend split. The audit treated source comments and status tables as claims to verify rather than as evidence.

## Verified repairs in this tranche

| Boundary | Finding | Repair | Evidence |
|---|---|---|---|
| Native generation and sampling | Sampling processors and penalties did not receive the active prompt and generated-token history in the native generation loop. | Generation now owns a bounded history containing configured prior tokens, the active prompt after rolling, and generated tokens. | Three-step history probe observes counts 2, 3, and 4. |
| Deterministic stochastic sampling | The sampler reseeded from `seed` on every call, so stochastic generation repeated the same random draw at every step. | Added optional caller-owned `rng_state`; native generation uses a per-generation state when no external state is supplied. | State advances across repeated calls while direct no-state calls remain seed-deterministic. |
| Structured token constraints | The callback ABI supported custom masking but had no reusable built-in allowlist contract. | Added `lm_token_allowlist` and `lm_logits_allowlist_processor`, with duplicate, empty, and vocabulary-range validation. | Unit tests cover direct masking, sampler composition, malformed allowlists, and explicit named errors. |
| Generation tracing | Runtime probes existed, but native generation had no ingress/prefill/decode/egress trace path. | Added synchronous generation trace sink stages: begin, prefill token, decode token, completion. Probe byte counts are token payload bytes. CLI `--trace` now emits them. | Synthetic ordering/timestamp/ID test and real GGUF CPU trace run pass. |
| Generation error state | Native generation could return before initializing `out_count`, leaving callers with an unspecified count on failure. | `out_count` is set to zero before execution and remains zero on validation failures. | Invalid prompt and invalid sampling-history regressions pass. |
| Generation allocation safety | Streaming generation allocated `max_new_tokens` before the main bounded validation; text generation staged arbitrary prompt byte counts. | Reject oversized generation limits before allocation and bound text prompt staging to the native input ceiling. | No-Vulkan and sanitizer suites pass. |
| Prefix-cache import | Serialized token words were read through a potentially unaligned `uint32_t*` cast. | Import now decodes each word byte-wise using the framed little-endian helper. | Deliberately unaligned serialized-buffer regression passes. |
| Standard-HF tied output | Alias logic existed but lacked a dedicated inference regression. | Added a SafeTensors fixture with missing `output.weight`, `tie_word_embeddings=true`, graph alias assertions, actual native generation, and a false-flag negative case. | Positive and negative tests pass. |

## Validation matrix

| Validation | Result |
|---|---|
| Vulkan Debug build and CTest | PASS |
| No-Vulkan Debug build and CTest | PASS |
| ASan/UBSan no-Vulkan build and CTest | PASS |
| Strict C99 public-header compile with warnings as errors | PASS |
| Forbidden matrix-extension route scan | PASS |
| `git diff --check` | PASS |
| Human-maintained source measurement | 554,140 bytes |
| Real SmolLM Q8_0 GGUF CPU generation | PASS, `generated=..` |
| Real SmolLM Q8_0 GGUF Vulkan generation through llvmpipe | PASS, `generated=..` |
| Real standard-HF F32 SafeTensors CPU generation | PASS, `generated=ino película` |
| Configured window/RoPE/chunked-prefill real runs | PASS on all three validated paths |
| Traced configured real GGUF CPU run | PASS with ordered begin/prefill/decode/completion events |

## Remaining coding backlog

The following items remain genuine implementation work rather than documentation gaps:

| Priority | Pending item | Current boundary | Required completion evidence |
|---:|---|---|---|
| P0 | Integrated n-gram or draft-model speculation | Proposal and greedy verification are standalone; target KV append/rollback is not integrated. | Greedy equality against normal generation, KV rollback/commit correctness, acceptance telemetry, and real-model run. |
| P0 | Model-integrated continuous batching | Scheduler is callback-level; each HTTP request opens and executes independently. | Two-request native KV state machine, serial-equivalence test, fairness, cancellation, and bounded admission. |
| P0 | Prefix KV reuse | Prefix cache stores metadata/page IDs but does not own or retain native KV payloads. | Page ownership/refcount contract, compatible identity hash, reuse/COW test, eviction, and generation integration. |
| P1 | Prompt-cache payload serialization | Only metadata framing is implemented. | KV payload export/import with model/settings identity validation and corruption tests. |
| P1 | Per-layer KV precision | One global typed-KV dtype is supported. | Layer/head codec arrays, allocation and attention propagation, codec differential tests. |
| P1 | Backend autotuning and telemetry | Kernel registry and trace probes exist, but no benchmark-driven selector exists. | Bounded benchmark harness, capability hash, selected-kernel report, fallback reason, and llvmpipe evidence. |
| P1 | Vulkan native attention and typed-KV attention | Projection/matvec paths are Vulkan-capable; attention and typed-KV attention remain CPU-side or explicitly unsupported. | Shader contracts, synchronization, CPU differential tests, and backend-specific capability gates. |
| P1 | Multi-device foundation | Device inventory exists; partition and transfer contracts do not. | Explicit partition/transfer ABI, identity validation, unsupported transport errors, and single-device regression. |
| P1 | Broader model/tokenizer coverage | The supported execution profile is narrow Llama plus exact tokenizer branches. | Per-architecture contracts, real checkpoints, parser fixtures, CPU oracle, and backend differential tests. |
| P2 | HTTP production features and WebUI | Loopback synchronous adapter only. | Socket-level tests, persistent model ownership, cancellation, scheduler admission, authentication/TLS, and UI assets. |
| P2 | Direct ROCr/HSA execution | Runtime probe only. | Real AMDGPU/ROCr host, queue/code-object contract, native kernel, and end-to-end checkpoint evidence. |

## Explicit non-claims

TinyLLM does not claim arbitrary transformer compatibility, model-wide F32 expansion, SafeTensors F16/BF16 Vulkan execution, typed-KV Vulkan attention, discrete-GPU performance, direct ROCr execution, learned semantic compression, distributed execution, or integrated speculative decoding. The available Vulkan device in this environment is llvmpipe/lavapipe, so its results are functional correctness evidence only.

## Source references

The detailed module contracts remain in `ENGINEERING_GUIDE.md`, the operational truth table remains in `MODULE_STATUS.md`, the 20-feature status remains in `docs/feature_roadmap_20.md`, and reproducible real-model hashes and commands remain in `docs/real_model_validation_notes.md`.
