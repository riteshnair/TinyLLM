# tokenizer.json BPE notes

The Hugging Face tokenizer pipeline separates normalization, pre-tokenization, model execution, and post-processing. A tokenizer.json BPE model stores a `model` object with `type: "BPE"`, a token-to-ID `vocab` object, ordered `merges`, and optional BPE settings such as `unk_token`, `continuing_subword_prefix`, `end_of_word_suffix`, `byte_fallback`, and `fuse_unk`. Merge order is the rank: the lowest-ranked available adjacent pair is merged first, with left-to-right tie handling. The model vocabulary and merge outputs must be mutually consistent.

This TinyLLM module implements a strict raw UTF-8 BPE contract: it accepts a tokenizer.json whose model is BPE, uses the exact ordered vocabulary/ranks, and applies merges to UTF-8 code-point symbols. It rejects unsupported normalizers, pre-tokenizers, post-processors, decoders, byte-level transformations, dropout, and added-token policies rather than silently approximating them. This is intentional: byte-level BPE and special-token behavior require their own exact contracts and tests.

Sources: Hugging Face Tokenizers model API (`https://huggingface.co/docs/tokenizers/en/api/models`), Hugging Face tokenizer construction guide (`https://huggingface.co/learn/llm-course/chapter6/8`), and the Tokenizers BPE implementation (`https://docs.rs/tokenizers/latest/src/tokenizers/models/bpe/model.rs.html`).

Acceptance: valid raw UTF-8 BPE fixture, merge-rank ordering, UTF-8 decoding, unknown-token behavior, malformed JSON/schema rejection, unsupported pipeline component rejection, capacity/range checks, and C99 header smoke compile.
