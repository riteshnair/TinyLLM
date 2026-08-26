MODULE: Open and bind tensor payloads from GGUF split shards or an explicit SafeTensors file set while preserving each tensor’s native file-relative bytes.
IN: Ordered GGUF paths with the first split at index zero and validated GGUF v3 split metadata, or ordered SafeTensors paths with validated headers and tensor descriptors.
OUT: One model handle whose tensor descriptors carry their owning shard and exact relative payload offset; all supported global metadata and names are consistent across shards.
DEPENDS: lm_file, GGUF parser, native tensor contracts.
OWNS: One read-only lm_file per shard and the model’s shard ownership table; close releases every shard exactly once.
THREADING: Read operations are not concurrent-safe through one stream-backed handle; independent model handles may be used by callers.
ERRORS: Reject missing, duplicated, misnumbered, incomplete, inconsistent, overlapping, or out-of-range shard tensors; reject ambiguous legacy span lookup on a sharded model and never fall back to the first file. Return LM_ERR_PARSE, LM_ERR_RANGE, LM_ERR_IO, or LM_ERR_UNSUPPORTED.
BACKENDS: CPU and Vulkan consume the same lazy shard spans; no vendor-specific code.
BUDGET: No model-wide payload copy; bounded metadata and file descriptors; source remains within the repository-wide 30 MiB core-plus-backend allowance.
TESTS: Two-shard GGUF fixture with tensors on both shards, malformed split metadata, missing/reversed shard, duplicate tensor, and native CPU/Vulkan differential; two-file SafeTensors F32 fixture with nonzero-shard matvec and duplicate-name rejection.
CLI: --model continues to name split zero; sibling names are derived from -00001-of-00002.gguf or supplied through the library API.
DONE: A tensor on a nonzero shard binds and reads exact native bytes, all format-specific shard metadata is checked, ambiguous lookup fails explicitly, and clean dual-backend plus sanitizer tests pass.
NOT IN SCOPE: Standard SafeTensors index-file discovery, concatenating raw files, implicit network download, async I/O, scheduler-level residency, or conversion of source encodings.
