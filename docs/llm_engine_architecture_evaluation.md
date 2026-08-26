# Tiny CPU-First, Vulkan-Primary LLM Engine

## Executive assessment

The direction makes sense **if the project is defined as a small, stable inference core with replaceable execution modules**, not as a miniature copy of vLLM, SGLang, TensorRT-LLM, and llama.cpp all at once. The proposed engine can be valuable because its differentiator would be a very small C99/C++ core, native GGUF and SafeTensors loading, a replaceable KV-cache subsystem, CPU fallback, Vulkan-first acceleration, explicit hardware discovery, and clean extension points for future decoding and backends.

The strict source constraint is the most important architectural decision. A human-maintained core below approximately **2 MB of C99/C++ source** is feasible only if the core excludes the HTTP server, WebUI, shader compiler, model conversion tools, distributed orchestration, and vendor SDKs. Those features must be optional modules or external processes. A runtime control-plane footprint around **20 MB** is plausible, but it cannot include model weights or a full 4,096-token KV cache for ordinary transformer models. The weights and cache must live in mapped files, VRAM, host memory pools, or explicitly managed overflow storage.

The central recommendation is therefore:

> **Build a small C ABI and a C++ orchestration layer. Keep CPU execution mandatory, Vulkan execution first-class, and every other backend optional. Treat KV cache, model loading, scheduling, transport, serving, and speculative decoding as replaceable interfaces.**

## What to borrow from existing engines

The strongest ideas should be adopted as interfaces and policies, not by pulling complete frameworks into the core.

| Engine or research line | Best idea to borrow | What not to import into the tiny core |
|---|---|---|
| llama.cpp | Compact local execution, GGUF integration, broad model pragmatism, CPU fallback, simple server modes | Its cache assumptions, large model-specific compatibility surface, and every optional backend implementation |
| vLLM/PagedAttention | Page/block KV allocation, prefix sharing, copy-on-write, continuous batching | Python control-plane assumptions and CUDA-specific kernels |
| SGLang | Radix/prefix reuse, structured execution, cache-aware scheduling | Full programming-language/runtime surface |
| TensorRT-LLM | Aggressive engine specialization, fused kernels, quantization-aware deployment | NVIDIA engine-building dependency and CUDA-only assumptions |
| AMD ATOM/ROCm ecosystem | AMD-specific distributed placement, tuned kernels, ROCm/ROCr integration | AMD-only orchestration and accelerator-specific code in the base runtime |
| FlashAttention and related kernels | Tiled, IO-aware exact attention and fused memory movement | A single universal kernel; each backend needs its own tile and synchronization strategy |
| DistServe | A clean prefill/decode boundary and explicit KV transfer | Mandatory cluster orchestration in a local inference binary |

The research literature consistently identifies model size, attention memory traffic, autoregressive serialization, KV-cache growth, batching, and scheduling as the main sources of inference cost. The efficient-inference survey organizes solutions into data-, model-, and system-level optimization, while the inference-systems survey emphasizes kernels, batching, scheduling, quantization, cache persistence, memory management, and distributed serving [1] [2].

## Hard budget interpretation

The budget must be stated precisely or it will become impossible to enforce.

| Budget | Recommended interpretation | Feasible target |
|---|---|---:|
| Core source | `core/` C99/C++ only; no WebUI, HTTP, vendor SDK, shader compiler, or conversion tools | 1.2–2.0 MB |
| Core binary | Optional target; measure stripped size per platform and backend | 0.5–3 MB, depending on Vulkan loader use |
| Control-plane RAM | Allocators, scheduler, metadata, token buffers, page tables, sockets, and small scratch buffers | Approximately 20 MB |
| Model weights | Memory-mapped or device-resident artifact; excluded from control-plane budget | Model-dependent |
| KV cache | Device/host cache pool, not counted as “20 MB runtime” | Model- and context-dependent |
| Optional modules | Server, WebUI, converter, profiler, shader generator, cluster agent, and vendor adapters | Separate budgets |
| Generated shaders | SPIR-V binaries and tuned variants | External assets or cache files |
| Tests and tools | Golden reference, fuzzers, benchmark harness, converters | Separate build targets |

For a representative 32-layer GQA model with 4,096 tokens, eight KV heads, head dimension 128, and two bytes per element, the raw K/V storage is approximately **512 MiB** before padding, metadata, scales, and allocator overhead. One byte per element is approximately 256 MiB, and half a byte per element is approximately 128 MiB before scales. Therefore a 20 MB process-control budget cannot mean “all inference memory”; it can mean that the engine itself has a small control plane while weights and KV storage are external, mapped, or device-resident.

## Proposed layered architecture

### Layer A: C99 stable core ABI

The C99 layer should be deliberately boring and stable. It should define opaque handles and plain data structures for contexts, tensors, devices, command streams, cache pages, model readers, and request states. It should not know about C++ containers, exceptions, Python, vendor runtimes, or a particular model family.

The core ABI should contain only:

| Core component | Responsibility |
|---|---|
| `lm_status` | Stable error codes and diagnostic category |
| `lm_tensor_view` | Shape, stride, dtype, byte range, and device location |
| `lm_allocator` | Host/device allocation and lifetime callbacks |
| `lm_backend_v1` | Device discovery, tensor operations, synchronization, and optional kernels |
| `lm_model_reader_v1` | Checked metadata/tensor access without format-specific execution logic |
| `lm_kv_cache_v1` | Page allocation, lookup, fork, commit, evict, migrate, and export/import |
| `lm_decode_policy_v1` | Proposal, verification, acceptance, rollback, and sampling hooks |
| `lm_request_v1` | Token input, generation limits, cancellation, and output events |
| `lm_metrics_v1` | Counters and timing samples without a logging framework dependency |

The C ABI is the portability seam. The C++ layer can provide safer wrappers, RAII, vectors, strings, server routing, and policy composition without forcing them into the low-level ABI.

### Layer B: C++ orchestration

The C++ layer should contain the scheduler, model graph, tokenizer adapter, request lifecycle, page-table cache manager, context manager, CLI parser, and optional server adapter. It should use a small number of internal types and avoid pulling a large general-purpose framework into the core.

The dependency direction should be one-way:

```text
CLI / local chat / HTTP adapter
                |
        Request scheduler
                |
      Context + decode policy
                |
       Model graph + KV cache
                |
        C99 backend ABI
          /      |       \
       CPU    Vulkan   optional plugins
                         /   |    \
                      ROCm CUDA OpenVINO/DirectML
```

The CPU backend is mandatory and is also the executable reference implementation. The Vulkan backend is first-class but replaceable. ROCm, CUDA, OpenVINO, and DirectML should be dynamically loaded or separately linked modules that implement the same ABI.

### Layer C: optional modules

The following should not count against the 2 MB core source target:

| Optional module | Reason to keep it outside the core |
|---|---|
| `server_http` | HTTP parsing, SSE, JSON, TLS, routing, and authentication expand attack and code surface |
| `webui_adapter` | Static HTML/JavaScript and browser protocol are not inference concerns |
| `model_convert` | GGUF/SafeTensors conversion requires larger metadata and validation logic |
| `shader_gen` | Runtime shader compilation and tuning need SDK/toolchain dependencies |
| `backend_rocm` | ROCm/ROCr libraries and version compatibility are platform-specific |
| `backend_cuda` | CUDA driver/runtime and kernel toolchains are NVIDIA-specific |
| `backend_openvino` | OpenVINO graph/runtime integration is an external deployment surface |
| `backend_directml` | DirectML and Windows graphics integration are optional Windows modules |
| `cluster_agent` | Discovery, transport, retries, and distributed scheduling are not local inference primitives |
| `bench` and `fuzz` | Test and measurement code should never be linked into production by default |

## Model formats: GGUF and SafeTensors

The format boundary must be strict and independent from execution. A model reader should expose checked tensor descriptors and a read/map operation. The executor should not parse GGUF or SafeTensors internally.

GGUF support should validate the header, version, metadata count, metadata types, tensor count, alignment, tensor offsets, tensor byte ranges, and quantization block layout. It should preserve tensor names and model metadata and support memory-mapped access only after all ranges are checked. The format reader should never execute bytes from a model file.

SafeTensors support should validate header length, JSON structure, tensor names, dtype, shape, offsets, non-overlap, file length, and shard/index completeness. The reader should support zero-copy mapping only after offset and lifetime validation. Conversion between SafeTensors and GGUF belongs in `model_convert`, not the core. Both readers must have malformed-file tests for truncation, duplicate names, invalid shapes, unsupported dtypes, missing shards, and integer overflow.

AMD is not a model format. AMD-specific handling belongs at the tensor-layout, quantization, backend-kernel, and device-placement layers. A GGUF or SafeTensors artifact should remain portable; the backend may select AMD-specific packed layouts after validating the source tensor.

## KV cache: replace the llama.cpp-style assumption

The KV cache should be a **virtualized page pool**, not a single large contiguous buffer and not a fixed per-slot reservation. This addresses the concern that a simple slot-oriented cache is convenient but less flexible for heterogeneous devices, prefix sharing, forked speculative decoding, eviction, and multi-request scheduling.

### Minimum page model

Each cache page should contain a fixed number of sequential token positions, such as 16, 32, or 64. The exact page size must be configurable because it trades fragmentation against lookup overhead. A page-table entry should contain:

| Field | Purpose |
|---|---|
| Prefix identity | Hash of model identity, adapter identity, tokenizer state, positions, and preceding page |
| Token span | Start position and number of tokens |
| Layer range | Full-layer page or layer-group page policy |
| Device location | CPU, host-pinned, Vulkan device, other backend, or disk spill |
| State | Resident, inflight, clean, dirty, evicted, importing, exporting |
| Reference count | Prefix sharing and copy-on-write |
| Quantization | FP16, BF16, Q8, Q6, Q4, or backend-specific encoding |
| Checksum/version | Optional integrity and compatibility check |

The page API should include `lookup_prefix`, `allocate`, `fork`, `append`, `commit`, `release`, `evict`, `migrate`, `export`, and `import`. Speculative decoding needs fork and rollback without copying every tensor. Prefix sharing needs immutable committed pages and copy-on-write at the first modified page.

### Cache policies

The first implementation should ship with three policies: contiguous single-request cache for bring-up, paged cache for production, and paged cache with prefix reuse. Later policies can add LRU-plus-recent eviction, score-based eviction, layer-aware eviction, host overflow, quantized KV, and persistent cache files.

The arXiv KV-cache survey classifies optimizations into token-level selection/budgeting/merging/quantization/low-rank methods, model-level grouping and architectural methods, and system-level virtual memory, prefix sharing, scheduling, multi-GPU, heterogeneous, I/O, and SSD methods [3]. The tiny engine should implement the system-level page interface first and add lossy methods only as explicit policies with quality tests.

KIVI demonstrates why KV quantization must not be treated as a generic uniform cast: its paper reports asymmetric 2-bit handling with per-channel key quantization and per-token value quantization, together with model- and implementation-dependent quality and throughput results [7]. A safe rollout order is FP16/BF16, then Q8, then Q6/Q4 only after per-model validation.

### Prefix-cache correctness

A reusable prefix must be keyed by more than raw token IDs. The identity should include model hash, tokenizer/version, prompt token IDs, position/RoPE parameters, adapter/LoRA identity, attention mode, cache dtype, and relevant execution configuration. Reusing a cache across incompatible identities is a correctness bug, not merely a performance issue.

The vLLM PagedAttention paper reports near-zero KV fragmentation and flexible cache sharing in its evaluated design, with 2–4x throughput improvements at comparable latency under its benchmark conditions [4]. This supports adopting the page abstraction, but the number must not be treated as a universal prediction for CPU or Vulkan.

## Context extension for a native 4,096-token model

A context manager can make a 4,096-token model useful in long conversations, but it cannot make the model natively understand an unlimited context. The engine must never silently feed more than the model’s configured maximum positions.

Use this budget:

```text
prompt_budget = native_context - reserved_output - safety_margin
```

The context manager should apply the following order:

| Stage | Action | Core or optional |
|---|---|---|
| 1 | Count tokens with the model tokenizer | Core adapter |
| 2 | Preserve system/developer policy, current task, and tool state | Core policy |
| 3 | Reuse the longest valid prefix from the KV cache | Core cache |
| 4 | Trim low-value old turns using a rolling window | Core policy |
| 5 | Compress older turns into a structured summary | Optional compressor |
| 6 | Retrieve selected older facts or documents | Optional retrieval module |
| 7 | Re-tokenize and validate the final prompt against 4,096 | Core gate |

The compressor should not summarize blindly. It should preserve commitments, constraints, identifiers, decisions, unresolved questions, tool results, citations, and user preferences. A compact record format is better than prose alone:

```text
[FACT] key=value; source=turn/document; confidence=...
[DECISION] decision; rationale=...; supersedes=...
[TODO] action; owner=...; status=...
[CONSTRAINT] exact requirement; scope=...
[TOOL] command/result summary; timestamp=...
```

Context extension should be implemented as **rolling context plus selective memory**, not as an unverified RoPE trick. RoPE scaling or model-context extension should be enabled only when the model was trained or validated for the requested scaling. Compression is lossy and must expose a quality/evidence status to the UI.

## Attention and KV-cache execution

The engine should distinguish prefill from decode. Prefill processes many prompt tokens and is usually more compute- and bandwidth-friendly for batching. Decode is sequential per request and is often limited by weight/KV movement, synchronization, and small-batch utilization.

FlashAttention shows the value of IO-aware tiling: it reduces reads and writes between high-bandwidth memory and on-chip memory while retaining exact attention for supported layouts [5]. The Vulkan backend should implement a small family of kernels rather than one universal kernel:

| Kernel family | Initial use |
|---|---|
| Vector/scalar reference | Correctness and unsupported features |
| Tiled QK + online softmax + PV | Prefill and longer prompt chunks |
| Paged decode attention | Decode over page-table KV storage |
| GQA/MQA specialized attention | Models with grouped or multi-query KV heads |
| Dot-product/DP4 path | Quantized or packed integer vectors when device features permit |
| Optional block-sparse path | Only for models/policies that declare compatible sparsity |

On Vulkan, cooperative matrix must be an optional capability, never a hard dependency. If Windows or a particular driver has issues with cooperative-matrix paths, use a feature-discovered dot-product/DP4 path, subgroup operations, vectorized scalar code, or a CPU fallback. The shader selector should choose from prevalidated variants; it should not assume that a device extension name guarantees useful performance.

## Vulkan-first execution design

The Vulkan backend should perform hardware discovery once and produce a capability record:

| Capability group | Examples |
|---|---|
| Device identity | Vendor, device, driver, API version, UUID |
| Memory | Heaps, budgets, host visibility, coherent/cached properties |
| Compute | Workgroup limits, subgroup size/control, shared memory |
| Arithmetic | FP16, BF16 where exposed, integer dot product, packed formats |
| Queues | Compute/transfer queues, queue family support, timeline semaphores |
| Portability | Portability subset restrictions, descriptor/indexing limitations |
| Interop | External memory/semaphore availability, if later needed |

The runtime should generate a stable capability hash and use it to select cached SPIR-V and tuning results. Dynamic shader generation should be separated into two stages:

1. **Build-time generation:** produce a compact family of SPIR-V variants and metadata.
2. **Runtime specialization:** select workgroup size, tile size, vector width, page size, and algorithm based on discovered features and a small bounded autotune cache.

Generating arbitrary shaders at runtime is powerful but can break the 2 MB and startup budgets. The small runtime should select and parameterize trusted variants. A separate `shader_gen` tool can compile larger templates and write signed or checksummed assets.

## CPU fallback

CPU is the reference backend and must be usable without Vulkan, CUDA, ROCm, OpenVINO, DirectML, or Python. It should provide:

- Scalar correctness kernels for every supported operator.
- SIMD variants selected by CPUID where available.
- Thread-pool execution with deterministic single-thread mode.
- Memory-mapped weights and bounded scratch arenas.
- The same tensor, graph, KV, sampling, and request APIs as Vulkan.
- A comparison mode that runs a small fixture on CPU and Vulkan and checks tolerances.

The CPU path should not attempt to compete with highly tuned BLAS libraries in every case. It should prioritize portability, correctness, and serving as the golden reference for backend validation.

## Optional backends

The backend ABI should be versioned and capability-based.

| Backend | Integration boundary | Recommended status |
|---|---|---|
| CPU | Built into core | Mandatory reference |
| Vulkan | `backend_vulkan` | First accelerator |
| ROCm/ROCr | Separate dynamic/shared module | Later AMD accelerator path |
| CUDA | Separate module | Later NVIDIA accelerator path |
| OpenVINO | Separate module | CPU/GPU/NPU deployment adapter |
| DirectML | Separate Windows module | Compatibility path, not the primary tuning target |

No backend should expose vendor-specific types through the core C ABI. A plugin can expose optional extensions after the stable base interface is negotiated. The engine should fail with a clear capability error when a requested operation is not supported, rather than silently falling back to a slower or numerically different path.

## Quantization and model optimization

The runtime should support already-quantized artifacts first and leave calibration/conversion to external tools. This is the best fit for a tiny source budget.

| Technique | Benefit | Runtime fit |
|---|---|---|
| Weight-only 4/5/6/8-bit | Reduces model storage and memory bandwidth | High; implement a few packed dot-product kernels |
| GPTQ | Accurate post-training weight quantization | Format/converter concern; runtime consumes packed weights [8] |
| AWQ | Activation-aware weight protection and hardware-friendly packing | Converter plus backend kernel; paper reports >3x over a stated FP16 baseline in TinyChat conditions [9] |
| SmoothQuant | Moves activation difficulty into weights | Mostly offline transformation; adapter only needs packed result |
| KV Q8/Q6 | Reduces cache bandwidth with moderate quality risk | Medium; add after paged FP16/Q8 correctness |
| KV Q4/KIVI-like | Strong cache reduction | Optional, model/backend-specific; require quality gates [7] |
| Sparsity/pruning | Less computation or storage | Low initial priority; needs model metadata and specialized kernels |
| Distillation | Smaller model with retained capability | Training/offline concern, not core runtime |
| GQA/MQA | Lower KV heads and cache bandwidth | Model property; runtime must support it |
| MoE | Fewer active parameters per token | Runtime support can be modular; expert routing/cache is complex |

## Decoding technologies and future-proofing

Speculative decoding is an excellent plugin boundary. The original speculative-decoding paper shows that a smaller proposal model can generate several candidates and the target model can verify them in parallel without changing the target distribution under the algorithm’s assumptions [10]. The core should therefore expose:

```text
proposal = draft.propose(state, max_tokens)
result   = target.verify(state, proposal)
accepted = policy.accept(result)
state    = cache.commit_or_rollback(state, accepted)
```

The same interface can support draft-model speculation, self-speculation, tree speculation, multi-token prediction, MTP, dFlash-style proposals, dFlash2, dPark, and future methods without hardcoding their names into the executor. Each technique should provide a plugin identifier, required model metadata, proposal shape, verification method, acceptance policy, rollback needs, and quality/latency counters. If an algorithm has not been precisely specified and validated, the runtime must label it experimental rather than claiming support.

## Batching and scheduling

The minimum scheduler should support one request, static batches, and continuous admission of requests. It should measure prompt tokens, generated tokens, cache pages, estimated work, and cancellation state.

The recommended order is:

| Stage | Scheduler feature | Why |
|---|---|---|
| 1 | Single-request prefill/decode | Establish correctness |
| 2 | Microbatch prefill | Improve prompt throughput without complex fairness |
| 3 | Continuous decode batch | Avoid waiting for a whole batch to finish |
| 4 | Chunked prefill | Reduce long-prompt interference with decode |
| 5 | Prefix-aware admission | Prefer reusable cache prefixes |
| 6 | Fairness/SLO policy | Prevent one long request from blocking others |
| 7 | Prefill/decode disaggregation | External node-agent feature for clusters |

DistServe’s results motivate keeping prefill and decode as explicit phases: colocating them can cause interference, while disaggregation lets each phase use different resource and parallelism settings at the cost of transferring intermediate state, mainly KV cache [6]. For a single laptop, use local phase scheduling before introducing network disaggregation.

## Multi-laptop and multi-desktop inference

Multi-machine inference is possible, but “many heads and many tails” is not automatically faster. The correct topology depends on the workload.

| Goal | Best first topology | Reason |
|---|---|---|
| Serve more independent users | Request parallelism | No per-token network synchronization |
| Large prompt batch | Replicated prefill workers | Prefill is more parallel and tolerates batching |
| One model does not fit one device | Layer/pipeline partitioning | Necessary capacity trade-off, but adds activation transfers |
| Large decode batch | Tensor parallel within a low-latency host | Network latency can dominate token-by-token decode |
| Interactive single-user decode | One host/device with local CPU fallback | Cross-laptop synchronization usually hurts TPOT |
| Large cluster | External scheduler plus prefill/decode pools | Keeps orchestration outside the inference core |

The core should expose a stateless request-phase protocol and KV export/import handles, but the cluster agent should handle discovery, authentication, retries, backpressure, transport compression, and placement. Use TCP/QUIC or a user-provided transport module; do not force a network stack into the 2 MB core.

## VRAM overflow, PCIe, BAR, and ReBAR

VRAM overflow must be explicit residency management, not an accidental page fault. ReBAR can improve the CPU’s visibility and transfer behavior on supported systems, but it does not turn VRAM into unlimited coherent system RAM and it does not remove PCIe bandwidth or latency costs.

The memory manager should use:

| Mechanism | Design rule |
|---|---|
| Residency tiers | Device-local, host-pinned, pageable host, optional storage |
| Transfer queue | Batch copies and overlap them with independent compute |
| Pinned buffers | Use bounded pools; never pin unbounded user memory |
| Chunking | Transfer layer/tensor blocks, not one tiny allocation per token |
| Prefetch | Predict next layer or page; cancel on request cancellation |
| Eviction | Pin active weights and hot prefixes; evict cold KV pages first |
| Synchronization | Timeline semaphores/fences and explicit ownership transitions |
| Backpressure | Refuse or queue work before allocations fail catastrophically |
| Telemetry | Report residency, bytes moved, transfer time, faults, and evictions |

A useful first overflow policy is weight placement plus KV paging: keep hot quantized weights and the active KV pages on the accelerator, while moving cold pages to pinned host memory. Per-token PCIe round trips should be treated as a failure mode. If a layer requires host data every decode step, the layer partition or quantization strategy is probably wrong.

## OpenAI API, local CLI, and Llama WebUI

The serving layer should be a separate C++ module with a small dependency footprint. It should expose an OpenAI-compatible subset:

| Endpoint | Minimum behavior |
|---|---|
| `GET /v1/models` | Return model ID, format, context limit, backend, and capabilities |
| `POST /v1/chat/completions` | Chat messages, generation parameters, streaming, stop, cancellation |
| `POST /v1/completions` | Raw prompt compatibility |
| `GET /health` | Backend and model readiness |
| `GET /metrics` | Optional counters and latency summaries |

Use Server-Sent Events for streaming first. Keep JSON parsing, HTTP, authentication, and WebUI static assets outside the core. The Llama-compatible adapter should translate common request fields to the internal request contract and return a compatibility warning when a requested option has no exact equivalent.

The local chat CLI should use the same request API as HTTP, not a second inference path. It should support interactive input, file prompts, prompt templates, streaming output, cancellation, context statistics, and a command to print the resolved configuration.

## Command-line configuration

Every deployment-sensitive choice should be configurable by switches. The parser can be a small custom C99/C++ parser rather than a large CLI library.

| Area | Switches |
|---|---|
| Model | `--model`, `--format auto|gguf|safetensors`, `--tokenizer`, `--model-hash` |
| Backend | `--backend auto|cpu|vulkan|rocm|cuda|openvino|directml`, `--device`, `--list-devices` |
| CPU | `--threads`, `--cpu-affinity`, `--simd auto|scalar|avx2|avx512|neon`, `--deterministic` |
| Vulkan | `--vk-device`, `--vk-queue`, `--vk-feature-policy strict|safe|aggressive`, `--shader-cache` |
| Context | `--ctx-size 4096`, `--max-new-tokens`, `--reserve-output`, `--rolling-window`, `--compress-context` |
| KV cache | `--kv-mode contiguous|paged`, `--kv-page-tokens`, `--kv-dtype f16|bf16|q8|q6|q4`, `--kv-prefix-cache`, `--kv-bytes` |
| Memory | `--vram-limit`, `--host-cache-bytes`, `--offload none|weights|kv|auto`, `--pinned-bytes`, `--rebar auto|on|off` |
| Scheduling | `--batch`, `--microbatch`, `--prefill-chunk`, `--continuous-batch`, `--priority`, `--slo-ttft`, `--slo-tpot` |
| Decoding | `--temperature`, `--top-k`, `--top-p`, `--min-p`, `--draft-model`, `--draft-tokens`, `--speculate`, `--mtp`, `--dflash` |
| Serving | `--chat`, `--server`, `--api openai|llama`, `--listen`, `--port`, `--ui-dir`, `--cors`, `--auth-file` |
| Distribution | `--node-id`, `--cluster`, `--role standalone|prefill|decode|worker`, `--transport`, `--head`, `--tail` |
| Tuning | `--tune`, `--tune-budget-ms`, `--profile`, `--dump-capabilities`, `--dump-config`, `--trace` |

Use deterministic precedence:

```text
built-in defaults < config file < environment variables < command-line switches
```

`--dump-config` should print the fully resolved configuration, device capability hash, selected kernels, cache policy, and unsupported requested features. Invalid combinations must fail before loading model weights.

## Source-size tightening plan

The project should enforce source budgets mechanically in CI.

| Component | Suggested source budget |
|---|---:|
| C99 ABI, status, tensors, allocator | 120 KB |
| CPU reference backend | 300 KB |
| Model graph and basic operators | 250 KB |
| GGUF/SafeTensors descriptor readers | 220 KB |
| Paged KV cache and context window | 220 KB |
| Sampling and request state | 120 KB |
| Minimal scheduler | 150 KB |
| Vulkan ABI adapter and dispatch | 180 KB |
| CLI and config | 80 KB |
| Tests embedded in core target | 0 KB; separate target |
| Total core target | Approximately 1.64 MB |

These are planning envelopes, not promises. Generated code, comments, and platform-specific modules should be measured separately. Detailed comments are valuable, but long design explanations belong in documentation and references; function comments should explain contracts, ownership, synchronization, failure, and non-obvious performance constraints.

## Recommended implementation phases

### Phase 0: Contract and reference fixture

Define one small supported decoder-only transformer model, one tokenizer, one GGUF fixture, one SafeTensors fixture, and a deterministic CPU reference. Validate tokenization, tensor shapes, logits, sampling, stop behavior, and a short golden response before adding Vulkan.

### Phase 1: Tiny CPU runtime

Implement the C99 ABI, allocator, tensor views, model reader, basic graph, FP16/BF16/FP32 reference kernels, paged KV API with a contiguous policy, sampling, and a command-line chat mode. Keep the core source under the budget from the first commit.

### Phase 2: Vulkan backend

Add device discovery, capability hashing, staging/pinned transfer pools, SPIR-V reference kernels, tiled attention, quantized matrix multiplication, and CPU/Vulkan differential tests. Start with vector/scalar and dot-product paths. Enable cooperative matrix only as an opt-in variant after feature and correctness checks.

### Phase 3: Production cache and context

Add page-table KV storage, prefix reuse, copy-on-write forks, cancellation, rolling 4,096-token context, structured compression hooks, and cache telemetry. Add Q8 KV only after FP16 correctness and quality baselines are stable.

### Phase 4: Serving and WebUI

Add the thin OpenAI-compatible server, Llama-compatible adapter, SSE streaming, local WebUI integration, health/metrics, and `--dump-config`. Keep the UI and HTTP module out of the core target.

### Phase 5: Specialized optimization

Add autotuned Vulkan variants, weight-only quantization layouts, CPU SIMD kernels, GQA/MQA specialization, chunked prefill, continuous batching, and optional speculative decoding. Every optimization must be selectable by CLI and compared to the CPU reference.

### Phase 6: Optional heterogeneous backends and cluster

Add ROCm/ROCr, CUDA, OpenVINO, and DirectML adapters. Then add the external node agent for request parallelism, prefill/decode pools, pipeline partitioning, and KV export/import. Do not add multi-laptop synchronization before the single-host scheduler and cache are measurable.

## Final recommendation

Proceed, but tighten the scope around **four non-negotiable boundaries**: a tiny stable core ABI, a page-based replaceable KV cache, a CPU reference plus Vulkan-first accelerator, and optional modules for serving, conversion, vendor runtimes, shaders, and cluster execution. This design can support the flexibility you want without becoming a Python/PyTorch-heavy framework.

The most important technical correction is to avoid claiming that context compression, RoPE scaling, KV eviction, or distributed execution increases the model’s native context length. For a 4,096-token model, the engine can preserve useful long-running conversations by rolling, summarizing, retrieving, and reusing validated prefixes, but each actual model invocation must remain within the trained/validated position limit.

The most important performance correction is to prioritize **memory movement and cache policy** over exotic arithmetic. For interactive decode, a well-designed paged KV cache, quantized weights, cache-aware scheduling, bounded host overflow, and avoidance of per-token PCIe transfers will usually matter more than enabling every matrix extension. Treat DP4/dot-product and cooperative-matrix paths as selectable Vulkan kernels, measure them on each GPU, and retain the CPU path as the correctness oracle.

## References

[1]: https://arxiv.org/abs/2404.14294 "A Survey on Efficient Inference for Large Language Models"
[2]: https://arxiv.org/html/2506.21901v1 "A Survey of LLM Inference Systems"
[3]: https://arxiv.org/html/2412.19442v3 "A Survey on Large Language Model Acceleration based on KV Cache Management"
[4]: https://arxiv.org/abs/2309.06180 "Efficient Memory Management for Large Language Model Serving with PagedAttention"
[5]: https://arxiv.org/abs/2205.14135 "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
[6]: https://arxiv.org/html/2401.09670v3 "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving"
[7]: https://arxiv.org/abs/2402.02750 "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache"
[8]: https://arxiv.org/abs/2210.17323 "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
[9]: https://arxiv.org/abs/2306.00978 "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"
[10]: https://arxiv.org/abs/2211.17192 "Fast Inference from Transformers via Speculative Decoding"
[11]: https://docs.vllm.ai/en/latest/design/prefix_caching/ "vLLM Automatic Prefix Caching"
[12]: https://docs.sglang.ai/ "SGLang Documentation"
[13]: https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html "TensorRT-LLM KV Cache System"
[14]: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md "llama.cpp Server Documentation"
[15]: https://registry.khronos.org/vulkan/specs/latest/html/ "Khronos Vulkan Specification"
[16]: https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_shader_integer_dot_product.html "Vulkan Integer Dot Product Extension"
[17]: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md "GGUF Format Documentation"
[18]: https://huggingface.co/docs/safetensors/index "SafeTensors Documentation"


## Disk-backed lazy loading and demand paging

Disk-backed model loading is strongly compatible with the small-control-plane design. The model reader should map the GGUF or SafeTensors file, validate all metadata and ranges, and expose tensor spans without copying the entire artifact into RAM. The residency manager then moves only the required tensor blocks into host-pinned or device-local memory.

The important distinction is between **file mapping** and **transparent demand paging**. A memory map can avoid an eager user-space copy, but a page fault during a decode step can still stall generation. The engine should therefore use explicit prefetch and residency states rather than relying entirely on the operating system’s page-fault behavior.

### Residency pipeline

```text
disk file -> OS file mapping -> pageable host view -> pinned staging pool -> device-local tensor/page
```

Each mapped tensor should have a compact descriptor containing file offset, byte length, alignment, dtype, shape, quantization block size, checksum/version, current residency, last-use timestamp, and an in-flight transfer token. The model file is immutable; the runtime owns only staging buffers, packed device layouts, and cache metadata.

| Policy | Recommended behavior |
|---|---|
| Weight granularity | Page or tensor-block, ideally aligned to quantization blocks and storage pages |
| Prefetch | Read the next layer’s blocks while the current layer executes |
| Readahead | Adaptive: increase for sequential layer execution, decrease after cancellation or cache misses |
| Host cache | Bounded LRU/clock cache in pageable or pinned memory |
| Device cache | Keep hot weights, active layers, and current KV pages resident |
| Eviction | Evict cold KV pages before repeatedly faulting hot weights |
| Fault behavior | Count and expose faults; never hide disk stalls behind a false throughput number |
| Integrity | Validate file bounds and optional checksums before making a block executable |
| Cancellation | Cancel queued prefetches and release unused staging buffers |
| Persistence | Keep file mappings open, but close device resources cleanly on device loss |

For sequential layer execution, a two- or three-buffer staging ring is usually more useful than arbitrary page faults: one buffer is being consumed by the backend, one is transferring, and one is being filled from the mapped file. The scheduler should overlap file read, host-to-device transfer, and compute only when the backend exposes independent queues and synchronization. On systems where the model is larger than RAM, the OS may still reclaim mapped pages; the engine must report this as a residency event rather than promising a fixed latency.

### Lazy-loading modes

The CLI should expose explicit modes instead of making performance behavior implicit:

```text
--load eager|mmap|lazy|stream
--weights-residency all|hot|auto
--weight-page-bytes <n>
--prefetch layers|blocks|off
--readahead-bytes <n>
--host-cache-bytes <n>
--pinned-cache-bytes <n>
--device-cache-bytes <n>
--evict weights|kv|auto
--io-threads <n>
--io-priority normal|low
--verify-model strict|fast|off
--dump-residency
```

`eager` is useful for testing and repeatable benchmarks. `mmap` avoids a full copy but may still experience faults. `lazy` uses explicit block residency and prefetch. `stream` is a constrained mode for models that can execute layer-by-layer with no need to retain all weights, but it may repeatedly reread weights and is normally unsuitable for high-throughput decode.

### Reality of decode-time lazy loading

Lazy loading is most useful for startup, model selection, large-model capacity, and infrequent requests. It is not free acceleration. If every generated token causes a disk read, storage latency dominates and the engine may become unusably slow. The scheduler should estimate the next working set and refuse or downgrade a configuration that cannot keep the active layer set in host/device memory.

For interactive decoding, keep the following resident whenever possible: the active model layer weights, tokenizer tables, rotary/position data, current KV pages, quantization scales, and backend pipeline objects. Load cold layers or alternate experts only when the model architecture and schedule make that transfer worthwhile. For MoE, expert weights can be demand-loaded, but expert cache thrashing can exceed the cost of keeping a dense layer resident.

### 40 MB control-plane allocation

A practical 40 MB runtime-control budget can accommodate a bounded page table, tensor descriptors, scheduler state, tokenizer buffers, request metadata, transfer rings, and compressed context records. It still cannot accommodate large model weights or a full ordinary 4,096-token KV cache. Those remain in mapped files, host cache pools, VRAM, or explicitly spilled storage.

The selector should calculate a plan before model execution:

```text
control_plane <= 40 MB
host_staging <= configured limit
pinned_host <= configured limit
vram_resident <= device budget - safety margin
kv_resident <= remaining device/host budget
```

If the plan cannot satisfy the requested context, batch, output, and residency constraints, `--dry-run` must fail with a breakdown rather than silently falling back to disk on every token.

### Backend-specific lazy-loading behavior

CPU can read directly from a validated mapped tensor and rely on OS readahead, but should use explicit block prefetch for predictable latency. Vulkan should use a staging ring, transfer queue, and device-local buffers; descriptor updates and synchronization must be batched. ROCr/ROCm and CUDA should use their own pinned-memory and asynchronous-copy mechanisms behind the same residency ABI. The core should know only about `map`, `prefetch`, `upload`, `resident`, `evict`, `fence`, and `release` operations.

The backend selector should produce a dry-run report such as:

```text
backend=vulkan
model=gguf
weights=lazy:mmap
vram_budget=6144MiB
host_cache=2048MiB
pinned_cache=256MiB
kv=pages:q8
prefetch=layer:2
transfer_queue=available
cooperative_matrix=disabled
dp4=enabled
cpu_fallback=enabled
warnings=none
```

This keeps all platform decisions visible and makes the engine easier to port, test, and modify.

[19]: https://man7.org/linux/man-pages/man2/mmap.2.html "Linux mmap(2)"
[20]: https://learn.microsoft.com/en-us/windows/win32/memory/file-mapping "Microsoft Windows File Mapping"
[21]: https://docs.vulkan.org/samples/latest/samples/extensions/memory_budget/README.html "Vulkan Memory Budget Sample"
[22]: https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html "CUDA Programming Guide: Unified and System Memory"
