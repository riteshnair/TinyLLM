MODULE: Parse and execute a bounded tokenizer.json BPE model without external runtime dependencies.
IN: UTF-8 tokenizer.json path with a BPE `model.vocab`, ordered `model.merges`, and optional supported scalar policy fields.
OUT: Owned vocabulary/rank handle; deterministic token IDs from raw UTF-8 code-point symbols; decoded vocabulary strings.
DEPENDS: C++17 standard library only; stable C99 ABI.
OWNS: Parsed vocabulary, reverse ID table, and merge map; caller owns input/output buffers and source file.
THREADING: A tokenizer handle is read-only after open; concurrent encode/decode calls do not mutate it.
ERRORS: Return LM_ERR_IO for file failure, LM_ERR_PARSE for malformed JSON/schema/UTF-8/vocabulary/merge output, LM_ERR_CAPACITY for bounds/allocation, LM_ERR_RANGE for invalid decode IDs, and LM_ERR_UNSUPPORTED for non-BPE models or unsupported pipeline/policy fields.
BACKENDS: CPU reference only; token IDs are backend-neutral inputs to CPU/Vulkan model execution.
BUDGET: 32 MiB bounded source JSON, 1 MiB vocabulary/merge count limits, no model-weight copy, compact C++17 implementation.
TESTS: Ordered merge-rank fixture, unknown-token fallback, UTF-8 validation, decode capacity, malformed merge, unsupported normalizer, C99 header smoke, and CPU/no-Vulkan CTest.
CLI: No implicit model integration or tokenizer auto-discovery; callers use the library API until an explicit model/tokenizer association contract exists.
DONE: A valid raw UTF-8 BPE fixture encodes and decodes deterministically; unsupported byte-level, normalizer, pre-tokenizer, post-processor, decoder, added-token, affix, dropout, and special-policy behavior fails explicitly.
NOT IN SCOPE: Byte-level BPE, Unicode normalization, pre-tokenization, post-processing, special-token insertion, tokenizer.json index discovery, WordPiece, Unigram, SentencePiece, or approximate fallback behavior.
