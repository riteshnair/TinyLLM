# TinyLLM feature research notes (working draft)

This file is a research scratchpad and will be rewritten into the final feature plan. Sources were opened from arXiv and official TensorRT-LLM documentation.

## Primary findings

| Area | Finding relevant to TinyLLM | Source |
|---|---|---|
| Paged KV memory | PagedAttention applies virtual-memory-style fixed blocks to dynamic KV state, reducing fragmentation and enabling flexible sharing within/across requests. The original vLLM paper reports 2–4x throughput improvements over prior systems in its evaluated settings; this is a paper result, not a TinyLLM promise. | [1] |
| Speculative decoding | The target model validates multiple draft tokens in one forward pass. This is most useful at low batch when the accelerator is underutilized; correctness requires a rejection/acceptance procedure that preserves the target distribution. | [2] |
| Dynamic draft trees | EAGLE-2 expands only promising tree nodes and reranks by approximate global acceptance probability; verification needs an ancestor-only attention mask so branches cannot see one another. | [3] |
| Production speculative modes | TensorRT-LLM documents draft-target, n-gram, Medusa, ReDrafter, EAGLE, lookahead, and disaggregated-serving integrations. TinyLLM should start with deterministic n-gram/draft-target speculation, then add tree/EAGLE interfaces only behind an explicit draft ABI. | [2] |
| KV compression | Recent work explores quantization, selective retention, semantic compression, and host offload. RocketKV reports large compression ratios in its own evaluation, but such methods need model/task-specific accuracy checks and should not be silently enabled. | [4] [5] |
| Long-context alternatives | Mamba and newer linear/recurrent attention research motivate a separate architecture plug-in rather than forcing SSM models through the Llama graph. | [6] [7] |

## Initial engineering consequences

1. Implement a target/draft speculative interface with exact token rollback and acceptance accounting before considering learned tree drafts.
2. Add KV page allocation, reference counting, prefix hashes, copy-on-write, eviction metadata, and optional quantization as separate contracts.
3. Add chunked prefill, continuous scheduling, prompt-prefix reuse, and batch compaction around the existing page control plane.
4. Add benchmark instrumentation for time-to-first-token, inter-token latency, accepted draft length, KV bytes, page faults, and backend fallback reasons.
5. Keep learned compression, arbitrary sparse attention, SSM architectures, and vendor-specific kernels behind explicit capability probes and unsupported errors until real checkpoints and differential tests exist.

## References

[1]: https://arxiv.org/abs/2309.06180 "Efficient Memory Management for Large Language Model Serving with PagedAttention"
[2]: https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html "Speculative Sampling — TensorRT-LLM"
[3]: https://arxiv.org/html/2406.16858v1 "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"
[4]: https://arxiv.org/abs/2401.18079 "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization"
[5]: https://arxiv.org/html/2502.14051v3 "RocketKV: Accelerating Long-Context LLM Inference via Compression"
[6]: https://arxiv.org/abs/2312.00752 "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
[7]: https://arxiv.org/html/2507.19595v3 "Efficient Attention Mechanisms for Large Language Models"

## Additional verified findings

| Area | Finding relevant to TinyLLM | Source |
|---|---|---|
| Prefix caching | Automatic prefix caching reuses KV blocks for identical token prefixes and mainly reduces prefill work; it does not accelerate decode when there is no reusable prefix. A content-addressed block key plus model/settings identity is safer than a raw pointer cache. | [8] |
| Speculative correctness | Speculative decoding should be exposed as a target-distribution-preserving verifier, not as an unconditional draft shortcut. Draft length and acceptance metrics must be observable because gains depend on target/draft cost and acceptance. | [2] [3] |
| Tree verification | Tree drafts require an ancestor-only mask and connected-tree selection; flattening branches into a normal causal sequence is incorrect. | [3] |
| Serving architecture | Current engine documentation separates prefill and decode concerns and exposes chunked prefill, prefix caching, KV offload, structured outputs, LoRA, disaggregated prefill, and data/context/expert parallel deployment as distinct features. TinyLLM should keep these as separate scheduler capabilities rather than one monolithic mode. | [9] |

[8]: https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/ "Automatic Prefix Caching - vLLM"
[9]: https://docs.vllm.ai/en/stable/cli/serve/ "vLLM serve CLI and feature configuration"

## Current engine feature inventory

The current vLLM documentation exposes a useful separation of concerns: automatic prefix caching, batch invariance, context extension, disaggregated prefill, KV offload, LoRA adapters, per-request metrics, structured outputs, tool calling, quantization, speculative decoding, data/context/expert parallel deployment, and sleep or fault-tolerance modes. These are candidate contracts, not claims that TinyLLM should copy their Python or accelerator dependencies.

The current speculative-decoding documentation lists EAGLE, MTP, draft-model, PARD, MLP speculator, n-gram, suffix, hidden-state, custom proposer, dynamic speculation, and adaptive verification. It explicitly describes both heterogeneous-vocabulary draft models through token-level intersection and greedy-sampling equality tests as an important correctness check. TinyLLM should implement the smaller, independently testable subset first: n-gram proposal, same-vocabulary draft proposal, exact rejection/rollback, adaptive proposal length, acceptance metrics, and a future tree-proposer ABI.

The structured-output documentation identifies choice, regex, JSON schema, context-free grammar, and structural tags as serving features. A compact TinyLLM version can begin with a tokenizer-level allowed-token callback and deterministic choice/regex/JSON-prefix masks; a full grammar compiler should remain a separate bounded module.

## References added

[10]: https://docs.vllm.ai/en/latest/features/structured_outputs.html "Structured Outputs - vLLM"
[11]: https://docs.vllm.ai/en/latest/features/speculative_decoding.html "Speculative Decoding - vLLM"
[12]: https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/ "Automatic Prefix Caching - vLLM latest"

## Additional primary-source observations

The latest vLLM feature documentation lists model-based speculative methods (EAGLE, MTP, draft model, PARD, MLP), lightweight methods (n-gram and suffix), hidden-state/custom proposers, dynamic speculation, adaptive verification, heterogeneous-vocabulary token-level intersection, and per-request acceptance metrics. It emphasizes greedy-sampling equality and rejection-sampler convergence as correctness checks rather than assuming speculation is automatically lossless. [11]

The same documentation exposes structured-output modes for choice, regex, JSON schema, context-free grammar, and structural tags, with explicit per-request configuration. TinyLLM can safely provide a compact token-mask/logits-processor ABI first, then add bounded grammar compilation separately. [10]

[10]: https://docs.vllm.ai/en/latest/features/structured_outputs.html "Structured Outputs - vLLM"
[11]: https://docs.vllm.ai/en/latest/features/speculative_decoding.html "Speculative Decoding - vLLM"
