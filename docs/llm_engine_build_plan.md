# Fast-Build Blueprint for the Tiny LLM Engine

## 1. Assessment

A pre-analyzed outline for each module will materially reduce development time. It prevents the most expensive form of rework: implementing a feature before its ownership, data contracts, failure behavior, memory lifetime, backend requirements, and tests are defined. It also makes the project easier to divide among contributors and easier to port between CPU, Vulkan, ROCr/ROCm, CUDA, OpenVINO, and DirectML.

The outline must remain short and executable. It should not become a large specification that nobody updates. Each module gets a one-page contract and a small test plan; detailed research remains in separate design notes.

The project should be built through a sequence of **vertical slices**. Every slice ends with a runnable binary, a deterministic test, and a measured budget. Do not build all abstractions first and postpone execution until the end.

## 2. Non-negotiable project rules

| Rule | Enforcement |
|---|---|
| Core source target | Keep mandatory C99/C++ source under approximately 2 MB; CI reports bytes by module |
| Runtime control plane | Keep metadata, scheduler, tokenizer buffers, page tables, and scratch state under 40 MB by default |
| Model memory | Weights and KV cache are explicitly excluded from the control-plane budget and are separately accounted for |
| Backend order | CPU reference first, Vulkan first accelerator, ROCr/ROCm/CUDA/OpenVINO/DirectML optional |
| Dependencies | No Python, PyTorch, or vendor SDK in the mandatory core build |
| Correctness | CPU reference is the oracle; every accelerated kernel has differential tests |
| Configuration | All policy decisions have CLI switches and appear in `--dump-config` |
| No silent fallback | Unsupported features produce a visible capability warning or a hard error |
| No placeholder code | Production targets cannot contain TODO stubs, fake return values, demo paths, or unchecked mock behavior |
| Ownership | Every allocation, mapped file, device buffer, page, fence, and request has one documented owner |
| Portability | Platform-specific code is isolated behind small interfaces and compile-time feature guards |
| Performance claims | Record model, context, batch, backend, device, precision, cache mode, and measurement method |

## 3. Module pre-analysis template

Every module should be described using the following compact template before implementation.

```text
MODULE: canonical name and one-sentence responsibility
OWNER: owning layer and maintainer
IN: exact input types, units, alignment, and validity rules
OUT: exact output types and postconditions
DEPENDS: direct dependencies only; no hidden global calls
OWNS: allocations, handles, mappings, pages, queues, and locks
THREADING: single-thread, concurrent, or externally synchronized
ERRORS: recoverable errors, fatal errors, and cancellation behavior
BACKENDS: CPU/Vulkan/ROCr/ROCm/CUDA/OpenVINO/DirectML applicability
BUDGET: source bytes, control-plane bytes, device bytes, and startup cost
TESTS: unit, malformed input, differential, stress, and benchmark cases
CLI: switches and resolved-config fields
DONE: measurable acceptance criteria
NOT IN SCOPE: explicit exclusions to prevent scope growth
```

A module is not ready to code until `IN`, `OUT`, `OWNS`, `ERRORS`, `TESTS`, and `DONE` are unambiguous. This is the fastest way to avoid large rewrites later.

## 4. Dependency layers

The source tree should mirror dependency direction. Higher layers may depend on lower layers; lower layers must never include a server, UI, vendor SDK, or model-specific policy.

```text
layer 0  platform primitives, status, atomics, byte utilities
layer 1  tensor views, dtypes, allocators, file mapping, capability records
layer 2  model readers, graph descriptors, tokenizer adapter, page-table KV cache
layer 3  CPU kernels, scheduler, context manager, sampler, decode policies
layer 4  Vulkan backend and generated-kernel selection
layer 5  optional ROCr/ROCm/CUDA/OpenVINO/DirectML adapters
layer 6  CLI, local chat, OpenAI/Llama API, WebUI adapter, cluster agent
layer 7  converters, shader generator, benchmark suite, fuzzers, profilers
```

The core build should compile layers 0–3. The Vulkan build adds layer 4. All other layers are optional targets. Layer 6 must call the same request API as the local CLI; it must never create a second inference implementation.

## 5. Recommended vertical-slice order

### Slice 0: Build and budget harness

Create the build matrix, formatting rule, compiler warnings, sanitizers, source-size counter, runtime allocation tracker, test runner, and a failure-report format. This slice has no model inference but prevents later uncertainty.

Acceptance criteria are a clean build in C99 and C++ mode where intended, warnings treated as errors for core targets, reproducible test invocation, and a report showing source bytes and peak control-plane memory.

### Slice 1: Status, bytes, tensor views, and allocator

Implement checked integer arithmetic, byte spans, shapes, strides, dtypes, error codes, and allocator callbacks. Do not implement a general tensor class or expression template system. Tensor views should describe memory; they should not own memory.

Tests must cover overflow, zero dimensions, negative/invalid strides if unsupported, alignment, use-after-release detection in debug builds, and allocation-failure behavior.

### Slice 2: File mapping and model readers

Implement a read-only mapped-file abstraction and strict GGUF and SafeTensors descriptor readers. Validate headers, ranges, alignment, names, shapes, dtypes, sharding, and integer overflow before exposing any tensor bytes.

Acceptance criteria are successful mapping without a full user-space copy, rejection of malformed fixtures, and a descriptor dump that is identical across CPU and Vulkan builds.

### Slice 3: One model on scalar CPU

Support one small decoder-only transformer family and one tokenizer adapter. Implement scalar RMS normalization, rotary position handling, matrix multiplication, attention, feed-forward, logits, sampling, stop conditions, and cancellation.

The first model should be deliberately small enough for rapid tests. The target is not broad model compatibility; the target is a complete, trustworthy path from file to generated token.

### Slice 4: CPU SIMD and memory mapping

Add CPUID-selected SIMD kernels only after scalar outputs are correct. Add weight-block access through mapped files and bounded host caches. Measure whether mapping and readahead improve startup and steady-state performance instead of assuming they do.

### Slice 5: Replaceable KV-cache API

Begin with a contiguous single-request policy behind the final cache interface. Then add fixed-size pages, prefix identity, reference counting, copy-on-write fork, rollback, eviction, migration, and telemetry. Do not couple the scheduler directly to a particular cache layout.

Minimum cache operations are `lookup_prefix`, `allocate`, `append`, `fork`, `commit`, `rollback`, `release`, `evict`, `migrate`, `export`, and `import`.

### Slice 6: Context manager

Implement token counting, output reservation, rolling-window trimming, system-policy preservation, structured summary slots, and final context-limit validation. Compression and retrieval are optional providers; the core only requires a prompt assembly contract.

The manager must guarantee that a native 4,096-token model is never invoked above its validated limit. It should report what was removed, summarized, retrieved, or reused.

### Slice 7: Vulkan reference backend

Discover device features and memory heaps, create a minimal compute path, run one operator, and compare it to CPU. Add staging buffers, transfer synchronization, and device-loss handling before adding aggressive kernels.

Use a scalar/vector SPIR-V reference path first. DP4/integer dot-product and cooperative-matrix variants are later selectable kernels. A driver exposing an extension is not sufficient evidence that the path is correct or fast.

### Slice 8: Vulkan optimized kernels

Add tiled matrix multiplication, online-softmax tiled attention, paged decode attention, GQA/MQA variants, quantized weight unpacking, and bounded autotuning. Store tuning by device UUID, driver identity, capability hash, shader version, model layout, and dtype.

Autotuning must have a time budget, a safe fallback, and a cache invalidation rule. It must not generate unbounded shader variants at runtime.

### Slice 9: Lazy loading and residency

Map model files, expose tensor blocks, add layer/block prefetch, bounded pinned staging, device residency, readahead, eviction, and dry-run planning. Prevent decode-time disk faults by tracking the next working set.

The plan must distinguish control-plane RAM, pageable host cache, pinned host cache, device-local memory, KV memory, and storage. `--dump-residency` reports faults, bytes moved, transfer time, evictions, and hit rates.

### Slice 10: Quantization policies

Consume pre-quantized GGUF/SafeTensors artifacts first. Implement a small set of packed weight kernels. Add KV Q8/Q6 only after cache correctness. Add Q4, KIVI-like, and TurboQuant-like policies as opt-in modules with model-specific quality tests.

TurboQuant-like transforms should be isolated behind a cache codec interface. The codec must declare bits per channel, scales/metadata, encode/decode cost, error metric, and supported attention path. Never label a codec “lossless” when it is not.

### Slice 11: Scheduling and decode policies

Add microbatch prefill, continuous decode batches, chunked prefill, prefix-aware admission, fairness, cancellation, and queue backpressure. Keep prefill and decode as explicit phases.

Add speculative decoding through a proposal/verify/accept/rollback interface. That interface can later host draft models, self-speculation, MTP, DFlash-like block diffusion drafting, and other experimental methods without modifying the scheduler core.

### Slice 12: CLI and local chat

Use one small parser with default/config-file/environment/CLI precedence. Required switches include backend, device, model format, context, KV mode, KV dtype, cache bytes, offload, lazy loading, prefetch, readahead, threads, batch, prefill chunk, speculation, sampling, deterministic mode, tuning, profiling, `--dry-run`, `--dump-config`, and `--dump-capabilities`.

The local chat mode should use the same request lifecycle and output events as the future HTTP server.

### Slice 13: OpenAI/Llama API and WebUI adapter

Add HTTP and SSE as an optional module. Implement `/health`, `/v1/models`, `/v1/chat/completions`, and `/v1/completions` first. Add a Llama-compatible translation layer and a static WebUI adapter after the API is stable.

The WebUI must display backend, device, context budget, compression events, KV residency, and warnings when a request was rolled or summarized. It must not hide context loss.

### Slice 14: Optional backends and multi-device

Implement ROCr/ROCm, CUDA, OpenVINO, and DirectML as capability-negotiated plugins. Start with one operation and differential tests per backend. Add multi-device request parallelism before tensor/pipeline parallelism.

For multiple laptops/desktops, use an external node agent. The core exposes request-phase and KV export/import contracts; the agent owns discovery, authentication, retries, backpressure, and transport.

## 6. Optimization order by expected return

| Priority | Optimization | Expected value | Risk | When to add |
|---:|---|---|---|---|
| 1 | Quantized weight layout plus packed dot products | High for bandwidth-bound decode | Medium | After scalar CPU path |
| 2 | Paged KV cache with prefix reuse | High for long context and concurrency | Medium | Before continuous batching |
| 3 | Fused/tiled attention and GEMM | High on Vulkan | High | After differential Vulkan path |
| 4 | Lazy loading with layer prefetch | High for startup and capacity | Medium | After mapping works |
| 5 | Continuous batching and chunked prefill | High for server throughput | Medium | After request lifecycle |
| 6 | KV Q8/Q6 | High memory reduction with manageable risk | Medium | After paged cache |
| 7 | Speculative decoding | Potentially very high TPOT reduction | High and workload-dependent | After rollback works |
| 8 | TurboQuant-like KV codecs | Potentially very high cache reduction | High quality/kernel risk | After Q8/Q6 baseline |
| 9 | Multi-token or DFlash-like drafting | Potentially high | Experimental/model-dependent | Plugin stage |
| 10 | Sparse attention, token dropping, layer skipping | Can be high but quality-sensitive | Very high | Research branch |
| 11 | Cross-host tensor/pipeline parallelism | Capacity gain; may reduce latency only in some cases | Very high | After single-host profiling |
| 12 | SSD KV persistence or expert paging | Capacity gain, often latency-expensive | High | Specialized deployments |

“Insanely faster” should be treated as a hypothesis to measure, not a design requirement. Large gains usually come from combining several compatible improvements: lower-bit weights, fewer KV bytes, fewer serial decode iterations, better cache reuse, fused memory movement, and correct scheduling. A technique that saves memory but adds expensive dequantization or random transfers may reduce end-to-end speed.

## 7. Validation gates

### Gate A: correctness

Every model operation has scalar CPU fixtures with known inputs and tolerances. Accelerated outputs are compared per tensor and end-to-end. Sampling has deterministic seeds. Cache fork/rollback tests verify exact token state. Format readers have malformed-file tests and fuzz targets.

### Gate B: resource safety

Run with allocation failure injection, mapped-file truncation, device loss, cancellation during transfer, out-of-memory, invalid CLI combinations, and concurrent request teardown. No test may rely on a placeholder return value.

### Gate C: performance

Record time to first token, time per output token, tokens/second, prefill throughput, peak control-plane RAM, host cache bytes, pinned bytes, VRAM bytes, page faults, bytes transferred, KV hit rate, and scheduler queue time. Benchmark one variable at a time before testing combinations.

### Gate D: maintainability

Require one owner per resource, explicit comments for non-obvious synchronization and memory contracts, no hidden backend calls, no new dependency without a budget note, and a module size report. Every optional module must compile out cleanly.

## 8. Commenting and code-quality standard

Function comments should explain **contract, ownership, synchronization, failure, and performance reason**. They should not repeat the function name. For example:

```cpp
// Appends tokens to a private tail page. Committed prefix pages remain immutable
// and may be shared by other requests. On allocation failure, the request is
// unchanged and LM_ERR_CAPACITY is returned; no caller-owned page is released.
lm_status kv_append(kv_cache* cache, request_id id,
                    const token_id* tokens, uint32_t count);
```

Use visual demarcation in source files:

```text
// ===== Public C ABI: ownership and error contract =========================
// ===== Internal cache policy: page table and eviction ======================
// ===== Backend adapter: Vulkan-only synchronization ========================
// ===== Optional server boundary: never included by core ====================
```

Do not comment obvious syntax. Comment why a layout, fence, alignment, page size, quantizer, or fallback exists and what breaks if it changes.

## 9. Fast-development workflow

For each module, write the one-page outline, add the public contract, add failure tests, implement the smallest correct path, run the CPU reference, add one benchmark, then optimize only after recording a baseline. Keep each commit buildable. Avoid combining format parsing, kernels, scheduling, and UI changes in one commit.

Use a narrow change loop:

```text
outline -> contract -> failing test -> smallest implementation -> reference result
        -> budget check -> accelerated variant -> differential test -> benchmark
```

A module is complete only when its `DONE` conditions pass. Research ideas that are not yet implemented belong in an experimental design note or plugin branch, never in a fake production path.

## 10. Recommended repository layout

```text
engine/
  include/lm/                 stable C99 public ABI
  core/                       status, bytes, tensor views, allocators
  platform/                   mmap, file mapping, threads, clocks, atomics
  model/                      GGUF and SafeTensors readers
  cache/                      paged KV interface and policies
  graph/                      model descriptors and execution plan
  cpu/                        scalar and SIMD reference kernels
  vulkan/                     discovery, memory, queues, SPIR-V dispatch
  backends/                   optional ROCr, ROCm, CUDA, OpenVINO, DirectML
  decode/                     sampling, speculative, MTP, diffusion plugins
  context/                    rolling window, compression, retrieval hooks
  io/                         lazy loading, prefetch, residency
  cli/                        command-line and config resolution
  server/                     OpenAI/Llama API and SSE
  webui/                      static assets and adapter
  cluster/                    optional external node agent protocol
tools/
  convert/                    GGUF/SafeTensors conversion
  shaders/                    shader generation and offline tuning
tests/
  unit/ fuzz/ golden/ diff/ bench/
docs/
  modules/                    one-page pre-analysis files
```

## Conclusion

A proper pre-analyzed outline for each module will help reach the goal faster, provided it remains a **small executable contract** rather than a giant document. The correct strategy is to pre-analyze interfaces and risks, then implement through vertical slices that produce working and measurable software every few days of work.

The project should optimize in this order: correctness, memory movement, cache policy, quantized layouts, fused kernels, scheduling, speculative decoding, then experimental diffusion and extreme quantization. This order gives the fastest path to real performance while preserving the small source base that makes the engine easy to modify.
