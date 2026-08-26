# Fast Troubleshooting Playbook for the Tiny LLM Engine

## Core principle

Troubleshooting becomes fast when every failure is **deterministic, localized, observable, replayable, and cheap to reduce**. The engine should be designed so a failure can be reduced from “the model produced a bad token” to “the Q/K page for layer 7, page 12, backend Vulkan, kernel variant DP4, changed after transfer” without manually stepping through the entire system.

The most effective practice is to make every boundary carry a small identity and contract. Large tensors should not be logged by default. Instead, record shape, dtype, device, byte range, checksum, min/max, norm, a few sentinels, and the transformation name. Full payload capture is enabled only for a selected trace packet or failing fixture.

## 1. Use a failure-localization ladder

Every test and runtime error should identify the first layer at which an invariant fails.

| Level | Question | Example failure |
|---|---|---|
| L0 build | Did the intended source and flags compile? | Wrong backend macro or stale generated shader |
| L1 platform | Are files, threads, clocks, and devices usable? | Mapping offset not aligned |
| L2 format | Is the model artifact valid? | Tensor range exceeds file length |
| L3 tensor | Are shape, stride, dtype, and alignment correct? | Q4 block interpreted as FP16 |
| L4 operator | Does one kernel preserve the reference contract? | RMSNorm output differs |
| L5 graph | Are operators connected in the intended order? | RoPE applied twice |
| L6 cache | Are pages, prefixes, forks, and rollback correct? | Speculative rejection commits a page |
| L7 scheduler | Are requests and phases ordered correctly? | Decode starves behind prefill |
| L8 backend | Does CPU match Vulkan/ROCm/CUDA? | Transfer uses stale staging buffer |
| L9 API | Is the request translated correctly? | OpenAI `max_tokens` ignored |
| L10 user result | Is the final generation correct and usable? | Context compression drops a constraint |

The error should report the first failed level, module, operation, request ID, trace ID, tensor/page identity, backend, device, and a compact evidence record. Do not report only “inference failed.”

## 2. Maintain one executable contract per module

Before implementation, create a short contract for each module:

```text
purpose:
inputs and units:
outputs and postconditions:
ownership and lifetime:
threading and synchronization:
error and cancellation behavior:
backend differences:
resource budget:
tests:
```

The contract becomes executable through assertions, unit tests, and boundary checks. This prevents comments and implementation from drifting apart. A module cannot be considered complete if its ownership or failure behavior is unspecified.

## 3. Make invariants explicit

Invariants should be grouped into cheap always-on checks and expensive debug checks.

| Area | Always-on invariant | Debug/trace invariant |
|---|---|---|
| Tensor | Shape product and byte range are valid | Recompute strides and sample values |
| Dtype | Dtype is supported by the selected kernel | Verify block scales and dequantized sentinels |
| Memory | Allocation does not exceed the selected budget | Red zones, poison patterns, and leak tracking |
| File map | Offset and length remain inside the mapped file | Hash the selected block before and after transfer |
| KV page | Page identity and token span are consistent | Recompute prefix chain and reference count |
| Device transfer | Fence is signaled before read | Compare host/device checksums and sentinel values |
| Scheduler | Request state transition is legal | Emit full state-transition trace |
| Sampling | Probabilities are finite and normalized | Compare candidate distribution with CPU reference |
| Context | Final prompt is within the model limit | Report every removed, summarized, and retrieved span |

Use an explicit state machine for request, page, transfer, backend, and device states. Illegal transitions should fail immediately in debug builds and return a structured error in production builds.

## 4. Use the trace packet or “ball through the pipe” method

A trace packet should travel through the same logical boundaries as real work. It carries a fixed-size header and optional payload reference.

```cpp
struct lm_probe {
    uint64_t trace_id;       // End-to-end identity.
    uint64_t parent_id;      // Parent operation or request.
    uint32_t kind;            // Token, tensor, page, transfer, request, or kernel.
    uint32_t stage;           // Monotonic pipeline stage.
    uint32_t bytes;           // Logical payload size.
    uint32_t flags;           // Capture, replay, fault-injection, or sensitive-data flags.
    uint64_t content_hash;    // Hash of canonical metadata or sampled content.
    uint64_t timestamp_ns;    // Monotonic timestamp.
};
```

Each tap records `enter`, `transform`, `handoff`, `wait`, `error`, and `exit`. A normal production build can compile taps into a branch that is nearly free when disabled. A debug build writes a compact binary event stream. The trace packet can be intercepted or replaced at a boundary through a probe callback without modifying the downstream module.

Do not log raw prompts or model output by default. Use hashes, redacted metadata, and opt-in capture so debugging does not become a data-leak path.

## 5. Deterministic replay is more valuable than interactive debugging

Record enough information to replay a failure without the original server state:

| Recorded item | Purpose |
|---|---|
| Model and tokenizer hashes | Prevent replay against incompatible artifacts |
| Resolved CLI/config | Reproduce backend and cache policy |
| Device capability hash | Detect hardware-dependent behavior |
| Seed and sampling settings | Reproduce stochastic choices |
| Input token IDs | Avoid changes in tokenizer behavior |
| Cache page identities | Reproduce prefix and eviction decisions |
| Kernel variant IDs | Reproduce specialized execution |
| Probe events | Locate the first divergence |
| Optional tensor snapshots | Compare exact intermediate values |

A replay file should support three modes: metadata-only, sampled-tensor, and full selected-payload. The replay runner should stop at the first mismatch between CPU reference and accelerated backend. This is faster and more reliable than attaching a debugger to a long-running server.

## 6. Differential testing across backends

The CPU scalar backend should be the reference oracle. Compare one operator at a time, then one layer, then one token step, and finally an entire request.

```text
scalar CPU -> SIMD CPU -> Vulkan reference -> Vulkan tuned -> optional vendor backend
```

Use tolerances by dtype and operation. Do not use one global epsilon. Quantized and stochastic operations need explicit error metrics, such as maximum absolute error, relative error, cosine similarity, logit rank agreement, and final-token agreement over a fixture set.

When a full output differs, bisect the graph using intermediate checksums. The first divergent tensor is the likely fault location; later differences are often consequences rather than causes.

## 7. Fault injection and failure-path testing

The engine should intentionally fail in controlled ways. Add injectable failures for allocation, file mapping, prefetch, staging upload, transfer completion, shader compilation, device loss, cache-page allocation, page eviction, network send, cancellation, and JSON parsing.

Every failure injection must verify that resources are released, request state is coherent, pages are not leaked, fences are not reused prematurely, and a retry or clear error is produced. This is particularly important for lazy loading and speculative decoding, where rollback paths are more error-prone than the happy path.

## 8. Minimize failing inputs automatically

When a failure is detected, reduce it to the smallest reproducer:

| Failure type | Reduction strategy |
|---|---|
| Prompt bug | Remove messages and tokens while preserving failure |
| Context bug | Reduce history and summary records |
| Model-format bug | Extract the smallest tensor/header fixture |
| Kernel bug | Reduce shape, dtype, batch, and page count |
| Cache bug | Reduce page chain, fork depth, and request count |
| Scheduler bug | Reduce number of requests and transitions |
| Backend bug | Reduce to one operator and one device feature |
| Network bug | Reduce packet sequence and transport events |

A reducer should save every smaller failing artifact with its configuration and seed. This prevents the common situation where a large test case is fixed by guesswork because nobody can isolate it.

## 9. Keep commits and changes bisectable

Each commit should compile and have a narrow purpose. Avoid commits that simultaneously change the cache, scheduler, Vulkan kernel, and WebUI. Record the expected benchmark impact and changed invariants in the commit message.

Use a strict sequence for performance work:

```text
baseline -> one change -> correctness gate -> benchmark -> trace comparison -> commit
```

Automatic `git bisect run` should invoke the smallest relevant deterministic fixture, not the entire model test suite. A ten-second bisect test is more useful than a two-hour test that nobody runs.

## 10. Separate correctness, performance, and compatibility modes

The binary should expose explicit modes:

| Mode | Purpose |
|---|---|
| `--mode=reference` | Scalar CPU, deterministic, maximum checks |
| `--mode=debug` | Assertions, poison memory, probes, traces, fault injection |
| `--mode=compare` | CPU versus selected backend at each boundary |
| `--mode=benchmark` | Stable clocks, warmup, fixed input, low logging |
| `--mode=server` | Production request handling with bounded metrics |
| `--mode=replay` | Consume a trace/replay artifact and stop at divergence |

Do not use benchmark mode to hide correctness failures. Do not use server mode to silently change precision, cache policy, or context behavior.

## 11. Treat configuration as part of the bug identity

A bug report must include the resolved configuration, not just the command typed by the user. Save `--dump-config`, device capabilities, selected kernel IDs, cache mode, page size, quantization, lazy-loading policy, thread count, and seed.

Use configuration validation before model loading. Reject combinations such as unsupported KV dtype, page size incompatible with a kernel, requested backend feature unavailable on the device, or context/output reservations exceeding the model limit.

## 12. Add performance observability without heavy frameworks

The core needs only fixed counters and monotonic timestamps. Record:

| Counter | Why it matters |
|---|---|
| Prefill tokens and time | Prompt throughput |
| Decode tokens and time | Interactive generation speed |
| Time to first token | User-visible latency |
| Scheduler wait time | Queue and fairness problems |
| KV hits/misses/evictions | Cache quality |
| Page migrations | Memory pressure and PCIe cost |
| Disk faults and readahead hits | Lazy-loading effectiveness |
| Bytes transferred | Hidden bandwidth bottlenecks |
| Kernel variant and occupancy proxy | Tuning comparison |
| Peak control-plane RAM | 40 MB budget enforcement |
| Fallback count | Unsupported or unstable accelerator paths |

Export metrics as text or a small binary record. Keep tracing disabled by default and avoid a logging dependency in the core.

## 13. Use golden fixtures at every boundary

Maintain small fixtures for valid and invalid GGUF/SafeTensors metadata, one tensor per dtype, one quantization block per supported format, one KV page, one prefix fork, one transfer, one fused kernel, one context compression event, one speculative rollback, and one API request.

A fixture should state what it proves and the expected invariant. Large models belong in nightly tests; small fixtures belong in every commit.

## 14. Make generated shaders and backends reproducible

Every SPIR-V artifact should record source template hash, compiler version, target environment, device capability assumptions, specialization constants, and tuning result. A shader mismatch should be diagnosed as an artifact-identity failure, not guessed at from visual output.

The same rule applies to ROCr/ROCm, CUDA, OpenVINO, and DirectML plugins. Their ABI version, runtime version, device identity, and selected kernel path belong in the trace header.

## 15. Priority order for implementation

| Priority | Practice | Immediate value |
|---:|---|---|
| 1 | Scalar CPU reference plus tiny golden fixtures | Establishes an oracle |
| 2 | Contracts and explicit invariants | Stops errors at boundaries |
| 3 | Trace packet with compact taps | Shows where data stops or changes |
| 4 | Deterministic replay | Makes failures repeatable |
| 5 | CPU/backend differential comparison | Localizes accelerator divergence |
| 6 | Fault injection | Validates cleanup and rollback |
| 7 | Automatic input reduction | Produces small actionable bugs |
| 8 | Bisectable commits and tests | Finds the change that caused regressions |
| 9 | Budget and performance counters | Prevents hidden memory/latency regressions |
| 10 | Optional full payload capture | Solves difficult cases without permanent overhead |

## Final recommendation

The best troubleshooting architecture is not a large debugger. It is a small set of repeatable mechanisms built into every boundary: **contracts, invariants, trace IDs, compact probes, deterministic replay, differential backends, fault injection, automatic reduction, and bisectable commits**.

These methods directly support the project’s small-source goal because they reduce debugging code duplication. The same probe and contract interfaces can observe CPU, Vulkan, ROCr, ROCm, CUDA, lazy-loading, KV pages, speculative decoding, WebUI requests, and future diffusion modules. The result is faster development without sacrificing the ability to remove or replace components.
