# Model format implementation notes

The parser implementation is based on the public specifications accessed 2026-08-26.

## GGUF

The GGUF specification defines a little-endian v3 header with `magic`, `version`, `tensor_count`, and `metadata_kv_count`, followed by metadata key/value records and tensor info records. Strings carry a uint64 byte length. Tensor info contains a name, dimension count, uint64 dimensions, a uint32 type, and a uint64 offset relative to the aligned tensor-data region. The default global alignment is 32 bytes; `general.alignment` may override it but must be a multiple of 8. Metadata values include scalar types and recursively nested arrays.

Source: [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).

## SafeTensors

The SafeTensors container begins with an 8-byte little-endian JSON-header length, then the JSON header bytes, then raw tensor data. Each tensor descriptor contains `dtype`, `shape`, and `data_offsets`; offsets are relative to the start of the data section. The header must be a JSON object, and tensor data ranges must be validated against the file bounds and against each other before exposing a view.

Source: [SafeTensors documentation](https://huggingface.co/docs/safetensors/en/index).

## Implementation boundary

The current engine only has structural SafeTensors checks and GGUF count checks. The next parser slice must use bounded reads, reject overflow, reject duplicate or overlapping tensor ranges, cap metadata nesting/counts, and avoid adding a general-purpose JSON dependency to the mandatory core.

## MoE routing

The current GGUF specification includes architecture metadata for expert models, including `*.expert_count` and `*.expert_used_count` fields; exact keys are architecture-scoped and must be read from the model’s declared architecture rather than guessed from filenames. The tensor mapping used by llama.cpp includes expert-specific feed-forward tensors such as `feed_forward.experts.w3`, so tensor-name mapping must treat expert index and layer index as structural fields.

The Mixtral configuration reference identifies `num_local_experts` as the number of expert MLPs available on a device and `num_experts_per_tok` as the token-choice top-k routing value. A correct runtime must preserve all expert weights in their native encoded layout, compute router logits, select the configured top-k experts, normalize or apply the model-specified routing weights, evaluate only selected experts, and combine their outputs. Loading all experts into F32 would be an invalid default for the memory design.

Sources: [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md), [Mixtral configuration and architecture reference](https://huggingface.co/docs/transformers/en/model_doc/mixtral), and [llama.cpp tensor-name mapping](https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/gguf/tensor_mapping.py).

## Native GGML block layouts

The official `ggml-common.h` defines the classic blocks used by GGUF. `Q4_0` has 32 values and 18 bytes per block (`fp16 d` plus 16 packed nibbles). `Q8_0` has 32 values and 34 bytes per block (`fp16 d` plus 32 signed bytes). K-quants use a 256-value super-block: `Q2_K` is 84 bytes, `Q3_K` is 110 bytes, `Q4_K` is 144 bytes, `Q5_K` is 176 bytes, and `Q6_K` is 210 bytes, based on the exact field declarations and static-size assertions in the authoritative header. Their dequantization is not interchangeable with the classic Q4_0/Q8_0 formulas, so each needs a separate decoder and differential fixture.

Source: [llama.cpp ggml-common.h](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h).
