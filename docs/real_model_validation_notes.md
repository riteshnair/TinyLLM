# Real checkpoint validation notes

This note records external model artifacts used for TinyLLM validation. The downloaded files remain local under `validation_models/` and are not repository source or committed model assets. The runs below validate the complete prompt-to-tokenizer-to-transformer-to-generation path; they do not claim model quality or discrete-GPU performance.

## Sources

| Artifact | Source |
|---|---|
| SmolLM 135M Instruct Q8_0 GGUF | [HuggingFaceTB/smollm-135M-instruct-v0.2-Q8_0-GGUF](https://huggingface.co/HuggingFaceTB/smollm-135M-instruct-v0.2-Q8_0-GGUF) |
| Tiny random Llama SafeTensors | [trl-internal-testing/tiny-random-LlamaForCausalLM](https://huggingface.co/trl-internal-testing/tiny-random-LlamaForCausalLM) |

The SmolLM model card identifies the artifact as a small Llama-family instruction model. The SafeTensors model card identifies a small F32 Llama test checkpoint. TinyLLM uses the model files as data only; no downloaded artifact is executed as code.

## Downloaded hashes

| File | Exact size | SHA-256 |
|---|---:|---|
| `validation_models/gguf/smollm-135m-instruct-add-basics-q8_0.gguf` | 144,811,552 bytes | `a98d3857b95b96c156d954780d28f39dcb35b642e72892ee08ddff70719e6220` |
| `validation_models/safetensors/model.safetensors` | 4,131,512 bytes | `aaa2c22c920d4e9c0943dc1f9d73fd4c687c2aa9c1ebcb878281c234075cec50` |
| `validation_models/safetensors/config.json` | 531 bytes | `3305bd985f960e7d680f4710ffd2830a55a5f149e8d8d2d18ff0b3fc508cc97f` |
| `validation_models/safetensors/tokenizer.json` | 1,842,792 bytes | `fcbc3208f7992717b368108564186a19695b26653a578435f13f77d2082112d8` |

## Compatibility and inference results

The real GGUF opens and inspects successfully as GGUF v3 with 272 tensors, context 2048, hidden size 576, 30 layers, 9 attention heads, 3 KV heads, intermediate size 1536, vocabulary size 49152, and Q8_0 matrix tensors. It omits `output.weight`, so the graph explicitly aliases output to the GGUF token embedding. Its native descriptors use dimensions such as `token_embd.weight [576,49152]`, `blk.0.ffn_down.weight [1536,576]`, and `blk.0.ffn_gate.weight [576,1536]`; the validated CPU path creates logical row/output views without transposing or expanding the stored quantization.

The real SafeTensors file opens and inspects successfully as SafeTensors v1 with 23 F32 tensors. It uses standard Hugging Face names such as `model.embed_tokens.weight`, `model.layers.0.self_attn.q_proj.weight`, `model.layers.0.mlp.gate_proj.weight`, `model.norm.weight`, and `lm_head.weight`. The explicit config attachment accepts `model_type=llama`, hidden size 16, intermediate size 64, two layers, four attention heads, the permitted default of four KV heads, RMS epsilon `1e-6`, and the permitted defaults for context and RoPE base. Its header has no TinyLLM-specific `__metadata__`; configuration is therefore supplied explicitly with `--config`.

The tokenizer attachment accepts the exact pipeline present in the downloaded tokenizer: BPE merges, `Prepend("▁")`, ASCII-space replacement, TemplateProcessing BOS insertion, decoder marker replacement/leading-strip behavior, byte fallback, and fused unknown handling. An independent Hugging Face `tokenizers` check produced prompt IDs `[1, 15043]` and tokens `["<s>", "▁Hello"]`; TinyLLM produced the same prompt IDs through its attached tokenizer. The TinyLLM unit suite also covers the exact pipeline with BOS, a merge, UTF-8 byte fallback, and round-trip decoding.

| Model and backend | Command working directory | Prompt | New tokens | Result |
|---|---|---|---:|---|
| SmolLM GGUF Q8_0, CPU | `/home/ubuntu/TinyLLM` | `Hello` | 2 | **PASS**, exit 0, `generated=..` |
| SmolLM GGUF Q8_0, Vulkan | `/home/ubuntu/TinyLLM` | `Hello` | 2 | **PASS**, exit 0, `generated=..` |
| Tiny random Llama SafeTensors F32, CPU | `/home/ubuntu/TinyLLM` | `Hello` | 2 | **PASS**, exit 0, `generated=ino película` |

The Vulkan run used the repository-root command after executable-relative shader lookup was added. The only available Vulkan device in this sandbox is lavapipe/llvmpipe, so this is functional Vulkan coverage rather than evidence of discrete-GPU speed or production performance. The generated strings are deterministic run outputs for this validation command, not a quality evaluation of either checkpoint.

## Reproduction commands

After downloading the four artifacts into the paths in the table and verifying the listed SHA-256 values, run:

```sh
cd /home/ubuntu/TinyLLM
./build-real/tiny-lm --backend cpu --model validation_models/gguf/smollm-135m-instruct-add-basics-q8_0.gguf --generate --prompt 'Hello' --max-new-tokens 2
./build-real/tiny-lm --backend vulkan --model validation_models/gguf/smollm-135m-instruct-add-basics-q8_0.gguf --generate --prompt 'Hello' --max-new-tokens 2
./build-real/tiny-lm --backend cpu --model validation_models/safetensors/model.safetensors --tokenizer validation_models/safetensors/tokenizer.json --config validation_models/safetensors/config.json --generate --prompt 'Hello' --max-new-tokens 2
```

These commands exercise the validated narrow Llama profile only. They do not imply support for arbitrary transformer architectures, arbitrary tokenizer pipelines, all quantization formats, SafeTensors Vulkan execution, typed-KV Vulkan, or direct ROCr execution.
