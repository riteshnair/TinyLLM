# MySkills Engineering Guide

> **Purpose:** This file is the source-of-truth guide for human programmers and coding agents using the MySkills collection. It explains how to select skills, design modules, implement safely, validate quickly, trace data through the system, and extend the tiny CPU-first/Vulkan-first LLM engine without creating hidden coupling.

## 1. Start here

Use the smallest skill set that matches the active task. Load `systems-ml-stack-router` only when the task crosses several boundaries or the correct specialist is unclear. For a single focused task, load the one domain skill and the relevant shared reference; do not load the entire library.

| Task boundary | Primary skill | Add only when needed |
|---|---|---|
| End-to-end architecture | `system-architecture` | `modular-component-boundaries`, `traceability` |
| New or changed code | `implementation-integrity` | `code-contract-comments`, `debug-core` |
| C99 implementation | `c99-systems` | `memory-management`, `toolchains` |
| C++ implementation | `cpp-systems` | `memory-management`, `toolchains` |
| CPU/Vulkan LLM runtime | `llm-components` | `vulkan-compute`, `gguf-format`, `safetensors-format` |
| ROCr/ROCm backend | `rocr-runtime` or `rocm-stack` | `rdna`/`cdna`, `kernel-tuning`, `version-policy.md` |
| CUDA backend | `cuda-stack` | `kernel-tuning`, `x86-architecture`, `version-policy.md` |
| Model artifact | `gguf-format` or `safetensors-format` | `model-format-checklist.md` |
| Cross-platform port | `cross-porting` | `cross-compilation`, `porting-checklist.md` |
| Fast troubleshooting | `debug-core` | `debug-reproduce`, `debug-localize`, `debug-invariants`, `debug-verify` |
| Backend/service separation | `backend-component-demarcation` | `plugin-adapter`, `modular-component-boundaries` |

The shared references in `_systems-ml-shared/` are not skills. Read them only by topic. This is the principal token-saving rule.

## 2. Progressive loading rules

The suite uses three levels of disclosure. **Metadata** routes the task. The matching `SKILL.md` provides the minimum workflow. A named reference is loaded only when its subject is active.

| Level | Default behavior | Example |
|---|---|---|
| Metadata | Always available for routing | Skill name and trigger description |
| Skill body | Load only matching specialist | `vulkan-compute/SKILL.md` for Vulkan work |
| Reference | Load only for a concrete need | `version-policy.md` for release/support claims |

After completing a specialist operation, retain only the result, decisions, unresolved risks, and evidence labels. Do not carry the entire specialist body into unrelated work. If the runtime cannot literally unload context, follow this protocol by writing a compact summary and using that summary instead of re-reading the full material.

## 3. Engineering mission and hard boundaries

The target engine is a small, modifiable inference runtime with a mandatory scalar CPU path, a first-class Vulkan backend, and optional ROCr/ROCm, CUDA, OpenVINO, and DirectML adapters. It supports GGUF and SafeTensors, paged KV storage, context rolling/compression, disk-backed lazy loading, CLI configuration, local chat, OpenAI/Llama-compatible serving, and future speculative, multi-token, diffusion, or TurboQuant-like plugins.

The mandatory core is constrained to approximately **2 MB of human-maintained C99/C++ source**. The runtime control plane may use up to **40 MB** for metadata, allocators, scheduler state, token buffers, page tables, staging rings, and small scratch arenas. Model weights, mapped files, device memory, host cache, pinned cache, and KV storage are separate measured budgets.

Keep the following outside the core target: HTTP/WebUI, JSON/TLS, model conversion, shader generation, vendor SDKs, cluster orchestration, profilers, fuzzers, and experimental algorithm implementations. They may call the stable core ABI but must not spread platform conditionals through the core.

## 4. Implementation workflow

Every change follows this loop:

```text
pre-study map -> module contract -> failing test -> smallest correct path
-> scalar CPU/reference result -> budget check -> accelerated variant
-> differential test -> probe trace -> benchmark -> focused commit
```

Do not begin with a large framework. Begin with one vertical slice that builds, runs, fails clearly, and produces a measurable result. Each slice should leave the repository in a buildable state.

### Module contract

Before coding, create a short module note with these fields:

```text
MODULE: one-sentence responsibility
IN: exact inputs, units, alignment, and validity
OUT: exact outputs and postconditions
DEPENDS: direct dependencies only
OWNS: allocations, mappings, handles, pages, queues, and locks
THREADING: concurrency and synchronization contract
ERRORS: recoverable, fatal, and cancellation behavior
BACKENDS: CPU/Vulkan/ROCr/ROCm/CUDA/OpenVINO/DirectML support
BUDGET: source, control-plane, device, startup, and I/O budget
TESTS: unit, malformed, differential, stress, and benchmark tests
CLI: switches and resolved configuration fields
DONE: measurable acceptance criteria
NOT IN SCOPE: explicit exclusions
```

A module is not ready when its ownership, errors, or acceptance criteria are vague. This pre-study prevents large downstream rewrites.

## 5. Source tree and dependency direction

Use one-way dependencies:

```text
platform primitives
    -> tensors, dtypes, allocators, file maps, capabilities
    -> model readers, graph descriptors, tokenizer adapter, KV cache
    -> CPU kernels, scheduler, context manager, sampling, decode policies
    -> Vulkan backend and shader selection
    -> optional vendor plugins and serving/cluster modules
```

Recommended layout:

```text
include/lm/       Stable C99 public ABI
core/             Status, bytes, tensor views, allocators
platform/         Mapping, files, threads, clocks, atomics
model/            GGUF and SafeTensors readers
graph/            Model descriptors and execution plan
cache/            Paged KV API and policies
cpu/              Scalar and SIMD reference kernels
vulkan/           Discovery, memory, queues, dispatch
backends/         Optional ROCr, ROCm, CUDA, OpenVINO, DirectML
context/          Rolling window, compression, retrieval hooks
decode/           Sampling, speculation, MTP, diffusion plugins
io/               Lazy loading, prefetch, residency
cli/              Switches and configuration resolution
server/           OpenAI/Llama API and SSE
webui/            Static UI and adapter
cluster/          Optional external node-agent protocol
tools/            Converters, shader generation, tuning, profiling
tests/            Unit, golden, malformed, differential, fuzz, benchmark
docs/             Module pre-study notes and design records
```

The lower layers must not include HTTP, UI, vendor SDK types, model-specific scheduling policy, or hidden global state.

## 6. Public ABI and ownership

The C99 ABI should use opaque handles and plain data structures. It defines status codes, tensor views, allocator callbacks, backend capabilities, model readers, KV cache operations, request events, decode policies, and metrics. C++ wrappers may provide RAII, containers, exceptions at the boundary, and policy composition, but the C ABI remains usable from C99 and other languages.

Every resource has one owner and one release path. Tensor views do not own memory. Mapped files own mappings. Allocators own allocations. Page tables own references to pages. Backends own device handles and synchronization objects. A function comment must state preconditions, postconditions, ownership, errors, cancellation, and synchronization where non-obvious.

Example:

```cpp
// Appends tokens to a private tail page. Committed prefix pages remain
// immutable and may be shared. On capacity failure, the request and page
// table are unchanged and LM_ERR_CAPACITY is returned.
lm_status kv_append(kv_cache* cache, request_id id,
                    const token_id* tokens, uint32_t count);
```

## 7. Backend selection and memory policy

Backend selection is capability-driven. The user may request `auto`, `cpu`, `vulkan`, `rocr`, `rocm`, `cuda`, `openvino`, or `directml`; the selector discovers actual features before loading weights and emits a dry-run plan.

```text
built-in defaults < config file < environment < command line
```

Useful switches include:

```text
--backend auto|cpu|vulkan|rocr|rocm|cuda|openvino|directml
--device <id> --list-devices --dump-capabilities --dump-config --dry-run
--vram-limit <bytes> --host-cache-bytes <bytes> --pinned-cache-bytes <bytes>
--offload none|weights|kv|auto --rebar auto|on|off
--load eager|mmap|lazy|stream --prefetch layers|blocks|off
--kv-mode contiguous|paged --kv-page-tokens <n> --kv-dtype f16|bf16|q8|q6|q4
```

The resolved plan includes backend, device, capability hash, memory heaps, queue types, tensor layout, KV dtype, page size, staging capacity, selected kernels, and fallback warnings. Unsupported requests must fail before model loading rather than silently changing behavior.

The memory tiers are device-local, host-pinned, pageable host, and optional storage. VRAM overflow uses explicit residency and transfer queues. ReBAR may improve mapping/transfer behavior where supported, but it does not eliminate PCIe costs or create unlimited coherent VRAM.

## 8. Model formats and lazy loading

GGUF and SafeTensors readers are strict, read-only, and independent from execution. Validate headers, metadata, names, shapes, dtypes, offsets, alignment, shards, ranges, and integer overflow before exposing tensor spans. Conversion belongs in tools, not the core.

Large models use file mapping plus explicit residency. The preferred pipeline is:

```text
disk -> validated map -> host view -> pinned staging -> device-local tensor/page
```

`mmap` or Windows file mapping avoids an eager full copy but does not guarantee predictable decode latency. Use layer/block prefetch, two- or three-buffer staging, adaptive readahead, hot-weight retention, and explicit fault counters. Never allow a configuration that causes a disk read on every decode token without reporting a severe warning or failing `--dry-run`.

## 9. KV cache and context

Use a replaceable page-based KV interface rather than binding the scheduler to a single contiguous cache. A page records token span, prefix identity, device location, residency, reference count, quantization, and state. Required operations are prefix lookup, allocate, append, fork, commit, rollback, release, evict, migrate, export, and import.

Prefix identity includes model hash, tokenizer/version, token IDs, position/RoPE parameters, adapter identity, attention mode, cache dtype, and execution configuration. Prefix reuse across incompatible identities is a correctness error.

For native 4,096-token models, enforce:

```text
prompt_budget = native_context - reserved_output - safety_margin
```

Use rolling windows, structured summaries, selective retrieval, and validated prefix reuse. Compression improves conversation continuity but does not change the model’s native context limit. Preserve system/developer instructions, constraints, decisions, identifiers, tool results, citations, and unresolved work.

KV quantization is an explicit codec policy. Begin with FP16/BF16, then Q8/Q6, and only then experimental Q4, KIVI-like, or TurboQuant-like codecs. Each codec declares its bit rate, scale metadata, encode/decode cost, supported kernels, error metric, and model/backend restrictions.

## 10. CPU and Vulkan execution

CPU scalar execution is the correctness oracle. Add SIMD only after scalar tests pass. Vulkan begins with capability discovery, a reference compute path, staging and synchronization, and CPU differential checks.

The Vulkan kernel family should include vector/scalar reference, tiled exact attention, paged decode attention, GQA/MQA attention, packed integer dot-product/DP4 paths, and optional cooperative-matrix variants. Cooperative matrix is never a hard dependency. Use DP4, subgroup, vectorized scalar, or CPU fallback if a driver path is unavailable or unreliable.

Generated SPIR-V records source hash, compiler, target environment, device capability hash, specialization constants, and tuning result. Runtime autotuning has a bounded time budget and a safe fallback; it does not generate unbounded shader variants.

## 11. Scheduling, decoding, and future research

Separate prefill and decode in the request state machine. Add features in this order: single request, microbatch prefill, continuous decode batch, chunked prefill, prefix-aware admission, fairness/cancellation, and external prefill/decode disaggregation.

Use one decode-policy interface for normal sampling, draft-model speculation, self-speculation, MTP, DFlash-like block diffusion drafting, and future algorithms. The contract is proposal, verification, acceptance, commit/rollback, and metrics. Experimental methods must identify required model metadata and quality limitations; they must not be represented by fake production code.

Diffusion language-model generation is a separate model/execution path. It can share tensors, backends, memory, kernels, CLI, and serving, but its denoising/remasking schedule must not be forced into an autoregressive decoder abstraction that hides different semantics.

## 12. Probe bus and fast troubleshooting

Use a low-overhead probe packet to trace data from ingress to egress. It carries trace ID, parent operation, kind, stage, logical bytes, flags, content hash, and timestamp. Taps record enter, transform, handoff, wait, error, and exit.

Do not log full prompts, outputs, or tensors by default. Record shape, dtype, device, byte range, checksum, min/max, norm, and selected sentinels. Enable full payload capture only for a selected failing fixture and protect sensitive data.

A complete failure report includes the first failed layer, module, operation, request ID, trace ID, tensor/page identity, backend, device, configuration hash, and evidence label. Use these levels:

```text
build -> platform -> format -> tensor -> operator -> graph -> cache
-> scheduler -> backend -> API -> final result
```

The minimum debugging stack is `debug-core`, `debug-reproduce`, `debug-localize`, `debug-invariants`, `debug-reference`, and `debug-verify`. Add `debug-reduce` for large reproducers and `debug-deep` only when the fast loop cannot localize the issue.

## 13. Validation gates

Every feature passes four gates.

| Gate | Required evidence |
|---|---|
| Correctness | Scalar CPU fixture, deterministic seed, operation and end-to-end checks |
| Safety | Malformed input, allocation failure, cancellation, device loss, OOM, and cleanup tests |
| Performance | TTFT, TPOT, throughput, peak control-plane RAM, cache hits, faults, transfers, and fallback counts |
| Maintainability | No placeholders, narrow dependencies, source budget, documented ownership, buildable optional modules |

Use differential execution:

```text
scalar CPU -> SIMD CPU -> Vulkan reference -> Vulkan tuned -> vendor plugin
```

When output differs, compare intermediate tensors and checksums until the first divergence is found. Do not debug only from final text output.

Use deterministic replay files containing model/tokenizer hashes, resolved configuration, device capability hash, seed, input token IDs, page identities, kernel IDs, probe events, and optional selected tensor snapshots. Add fault injection for allocation, mapping, staging, transfer, shader compilation, device loss, eviction, network, cancellation, and JSON parsing.

Reduce every failure automatically where possible: remove prompt tokens, messages, pages, requests, tensor dimensions, operators, or transport events while preserving the failure. Keep the smallest reproducer in the regression suite.

## 14. No-false-code policy

Production code must not contain a placeholder, fake success path, unimplemented return that appears valid, hidden demo branch, unchecked mock, or comment that claims behavior the code does not provide. If an operation is intentionally unsupported, return a named error such as `LM_ERR_UNSUPPORTED` and report it in the resolved configuration.

A change is not complete until its implementation-integrity review confirms that the actual path is exercised. The report must distinguish `PASS`, `FAIL`, `NOT RUN`, and `UNVALIDATED`.

## 15. How human and AI programmers should work

Before editing, read this guide and the one-page module note for the target component. Load the smallest relevant skill and reference. State the intended boundary and acceptance test. Make one narrow change. Run the smallest relevant test immediately. Update the probe or invariant if the new boundary is difficult to observe. Record the resolved configuration and evidence.

For AI-assisted changes, the agent must first state the files it expects to change, the contract it will preserve, the tests it will run, and the assumptions it will not make. It must not invent APIs, versions, hardware support, benchmark numbers, or successful execution. If a dependency or capability is unknown, it must report `UNVALIDATED` and add a probe or test rather than guessing.

## 16. Definition of done

A module is ready for integration when its contract is written, dependencies are minimal, ownership is explicit, unsupported cases fail clearly, malformed inputs are tested, CPU/reference behavior is known, accelerated behavior is differentially checked, source and runtime budgets are measured, CLI configuration is documented, probes identify its boundaries, and a focused regression test passes.

The fastest path is not the one with the most features in the first version. It is the one that makes every feature cheap to validate, easy to remove, easy to port, and difficult to misunderstand.

## Related documents

- [`README.md`](README.md) — Skill catalog, routing, migrations, and installation.
- [`_systems-ml-shared/shared-execution-protocol.md`](_systems-ml-shared/shared-execution-protocol.md) — Evidence-first execution protocol.
- [`_systems-ml-shared/quality-gates.md`](_systems-ml-shared/quality-gates.md) — Completeness and anti-placeholder checks.
- [`_systems-ml-shared/version-policy.md`](_systems-ml-shared/version-policy.md) — Release, preview, compatibility, and support claims.
- [`_systems-ml-shared/porting-checklist.md`](_systems-ml-shared/porting-checklist.md) — Cross-platform capability mapping.
- [`_systems-ml-shared/model-format-checklist.md`](_systems-ml-shared/model-format-checklist.md) — GGUF and SafeTensors validation.
- [`_systems-ml-shared/suite-manifest.md`](_systems-ml-shared/suite-manifest.md) — Canonical skill inventory and loading policy.
- [`docs/llm_engine_architecture_evaluation.md`](docs/llm_engine_architecture_evaluation.md) — Detailed engine architecture evaluation.
- [`docs/llm_engine_build_plan.md`](docs/llm_engine_build_plan.md) — Vertical-slice build plan.
- [`docs/fast_troubleshooting_playbook.md`](docs/fast_troubleshooting_playbook.md) — Troubleshooting and probe methods.
