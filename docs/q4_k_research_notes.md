# Q4_K implementation notes

Source inspected on 2026-08-26:
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-quants.c

Authoritative constants and structure:
- `QK_K = 256` values per super-block.
- `K_SCALE_SIZE = 12` bytes.
- `block_q4_K` contains two FP16 values (`d`, `dmin`), 12 packed scale/min bytes, and 128 packed 4-bit values.
- Static size is 144 bytes per 256-value super-block.
- Q4_K represents values as `x = a*q + b`; it is not interchangeable with Q4_0.

Implementation guardrails:
- Do not add a generic dequantizer or whole-tensor F32 cache.
- Add one bounded native decoder and a CPU differential fixture first.
- Preserve the packed source bytes and use F32 only as a bounded output/accumulator fixture.
- Bind GGUF raw type only after exact descriptor byte sizing and decoder tests pass.
- Keep unsupported K formats named unsupported.
