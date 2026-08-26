# GGUF split-shard contract notes

The upstream llama.cpp split-tool documentation states that split files are produced and consumed as a set rather than concatenated raw byte streams. The first model path identifies the split set, and the tool uses `split-max-size` or `split-max-tensors` when creating parts. Source: [llama.cpp tools/gguf-split README](https://github.com/ggml-org/llama.cpp/blob/master/tools/gguf-split/README.md).

The upstream model loader source reads `split.count`, requires `split.no`, loads the first split before siblings, checks each sibling index, and reads tensor descriptors from each sibling using that file’s own metadata context. It validates `split.tensors.count` against the unified tensor map and rejects duplicate tensor names. Source: [llama.cpp src/llama-model-loader.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-model-loader.cpp), loader region around split initialization and sibling loading.

TinyLLM must therefore preserve per-shard file ownership and each shard’s own tensor-data header offset. It must not treat shards as a concatenated byte stream, must reject missing or misnumbered parts, and must bind each tensor span to its owning read-only file while retaining native payload bytes.
