# TinyLLM 20-feature expansion plan

## Scope and engineering rule

TinyLLM now permits approximately **30 MiB of human-maintained core plus backend source**, excluding model weights, mapped files, runtime allocations, device memory, generated shader binaries, and optional external tools. The larger allowance is a ceiling, not a target: each feature remains modular, bounded, source-measured, and independently validated. Native GGUF and SafeTensors encodings remain native; no model-wide F32 expansion is permitted.

The feature selection is informed by PagedAttention [1], EAGLE-2 dynamic draft trees [2], current TensorRT-LLM speculative-decoding modes [3], and current vLLM serving capabilities [4] [5]. Research speedups are not copied as performance claims; TinyLLM must produce local correctness and benchmark evidence.

## Current implementation status

The current expansion tranche implements and tests the following policy primitives: advanced sampling and penalties (features 1–3), stop-string suppression (feature 4), caller-owned logits processors suitable for token masks (feature 5), deterministic n-gram proposal, greedy target verification, and adaptive depth (features 6–8), identity-safe longest-prefix lookup with LRU eviction (features 9–10), and a bounded fair round-robin batch scheduler with admission, cancellation, completion, and telemetry (feature 14). These are real library contracts with deterministic unit tests; speculative verification is not yet wired into target-model KV execution, and the batch scheduler callback is not itself a model executor.

The remaining roadmap items remain explicitly planned rather than silently advertised as complete. Each will require a vertical slice with a CPU oracle, backend differential tests where applicable, and real-model evidence before its status changes.

## Twenty useful features

| # | Feature | Concrete TinyLLM contract | Validation gate | Priority |
|---:|---|---|---|---|
| 1 | Top-p/nucleus sampling | Select from the smallest cumulative-probability set with deterministic seed behavior. | Distribution bounds, seed repeatability, greedy equality. | P0 |
| 2 | Min-p and typical sampling | Add probability-floor and entropy-distance filters with explicit invalid-config errors. | Hand-computed logits and boundary cases. | P0 |
| 3 | Repetition/frequency/presence penalties | Apply history-aware logit transforms without mutating model weights. | Repeated-token fixtures and finite-logit checks. | P0 |
| 4 | Stop sequences | Support stop-token and bounded stop-string termination without emitting the stop suffix. | Prefix/split-boundary and capacity tests. | P0 |
| 5 | Logits-processor callback | Allow bounded user policy to mask or transform logits before sampling. | C99 callback ABI, error propagation, no silent fallback. | P0 |
| 6 | N-gram speculative decoding | Propose prompt/recent-history continuations and verify them with target logits. | Greedy output equality, rollback, acceptance metrics. | P0 |
| 7 | Draft-model speculative ABI | Define target/draft proposal, vocabulary compatibility, verify, accept, and rollback interfaces. | Same-vocabulary synthetic draft; incompatible vocabulary rejected. | P0 |
| 8 | Adaptive speculation depth | Adjust proposal length from rolling acceptance/cost metrics under configured bounds. | Deterministic policy simulation and no overrun. | P1 |
| 9 | Prefix KV cache | Hash token prefixes plus model/settings identity and reuse committed full pages. | Same-prefix reuse, mismatch invalidation, COW isolation. | P0 |
| 10 | KV eviction and tiering | Add LRU metadata and explicit host/device/disk residency transitions. | Budget pressure, pinning, readback equality, failure reporting. | P1 |
| 11 | Context shifting | Preserve a configured prefix while rolling the conversation tail when the context window fills. | Long prompt generation and token-position checks. | P0 |
| 12 | Sliding-window attention | Honor architecture/request window limits without reading discarded KV pages. | Windowed attention oracle and page-read bounds. | P1 |
| 13 | Chunked prefill | Split long prompts into bounded work units and interleave decode work. | Token/logit equality against unchunked prefill. | P0 |
| 14 | Continuous batching | Admit, pause, compact, and retire requests at decode-step boundaries. | Two-request deterministic batch versus serial oracle. | P0; bounded scheduler primitive implemented, model integration pending |
| 15 | Prompt-cache serialization | Export/import validated prefix metadata and native KV payloads with model identity checks. | Round trip, corruption rejection, incompatible model rejection. | P1 |
| 16 | RoPE context extension | Add explicit linear/NTK-style scaling policy fields with architecture validation. | Position-frequency oracle and unsupported-policy tests. | P1 |
| 17 | Mixed per-layer KV precision | Select codecs by layer/head policy instead of one global KV dtype. | Codec round trips and attention differential tests. | P1 |
| 18 | Structured decoding masks | Add choice, stop/allowed-token, and bounded JSON-prefix constraints through the logits processor. | Exact token-mask fixtures and malformed policy rejection. | P1 |
| 19 | Backend autotuning and telemetry | Benchmark registered kernel paths at startup or on demand and report choice, latency, bytes, and fallback reason. | Deterministic mock-capability probe plus real llvmpipe report. | P1 |
| 20 | Multi-device execution foundation | Add explicit device inventory, tensor/pipeline partition contracts, and bounded KV transfer records; no implicit distributed success. | Single-device identity, partition validation, unsupported transport errors. | P1 |

## Deliberate boundaries

Learned EAGLE/MLP/MTP heads, arbitrary sparse attention, semantic KV autoencoders, SSM architectures, LoRA weight injection, multimodal encoders, direct ROCr/HSA execution, CUDA/ROCm/OpenVINO/DirectML implementations, and full distributed transport require additional model and hardware contracts. They may consume the extension points above, but must not be represented as implemented until real checkpoints, backend probes, and differential tests exist.

## Implementation order

The first implementation tranche should be features 1–9 and 11–14 because they build directly on existing sampling, page/COW KV, context-compaction, native generation, and batch foundations. The second tranche should implement 10, 15–19. Feature 20 should begin with validation-only partition and transfer contracts before any network or vendor transport is added.

## References

[1]: https://arxiv.org/abs/2309.06180 "Efficient Memory Management for Large Language Model Serving with PagedAttention"
[2]: https://arxiv.org/html/2406.16858v1 "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"
[3]: https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html "Speculative Sampling — TensorRT-LLM"
[4]: https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/ "Automatic Prefix Caching - vLLM"
[5]: https://docs.vllm.ai/en/latest/features/speculative_decoding.html "Speculative Decoding - vLLM"
