MODULE: Parse and execute a bounded tokenizer.json BPE model without external runtime dependencies.
IN: UTF-8 tokenizer.json path with a BPE `model.vocab`, ordered `model.merges`, and optional supported scalar policy fields.
OUT: Owned vocabulary/rank handle; deterministic token IDs from raw UTF-8 code-point symbols or the exact validated SentencePiece-style normalized symbol stream; decoded vocabulary strings.
DEPENDS: C++17 standard library only; stable C99 ABI.
OWNS: Parsed vocabulary, reverse ID table, merge map, and exact pipeline flags; caller owns input/output buffers and source file.
THREADING: A tokenizer handle is read-only after open; concurrent encode/decode calls do not mutate it.
ERRORS: Return LM_ERR_IO for file failure, LM_ERR_PARSE for malformed JSON/schema/UTF-8/vocabulary/merge output, LM_ERR_CAPACITY for bounds/allocation, LM_ERR_RANGE for invalid decode IDs, and LM_ERR_UNSUPPORTED for non-BPE models or unsupported pipeline/policy fields.
BACKENDS: CPU reference only; token IDs are backend-neutral inputs to CPU/Vulkan model execution.
BUDGET: 32 MiB bounded source JSON, 1 MiB vocabulary/merge count limits, no model-weight copy, compact C++17 implementation.
TESTS: Ordered merge-rank fixture, unknown-token fallback, UTF-8 validation, decode capacity, malformed merge, unsupported normalizer, exact `Prepend("▁")` plus space replacement, TemplateProcessing BOS insertion, byte fallback, decoder marker restoration/leading-strip, C99 header smoke, and CPU/no-Vulkan CTest.
CLI: No implicit model integration or tokenizer auto-discovery; standard HF SafeTensors callers pass `--tokenizer PATH` explicitly, and model association is performed through `lm_model_set_tokenizer_json`.
DONE: A valid raw UTF-8 BPE fixture and the exact validated SentencePiece-style BPE fixture encode and decode deterministically. The exact branch is limited to the observed `Prepend("▁")`, ASCII-space replacement, BOS TemplateProcessing, byte fallback/fused-unknown policy, and decoder marker/strip contract.
NOT IN SCOPE: Other byte-level BPE, Unicode normalization, arbitrary pre-tokenization, arbitrary post-processing, arbitrary decoder pipelines, tokenizer.json index discovery, WordPiece, Unigram, unvalidated SentencePiece variants, or approximate fallback behavior.
