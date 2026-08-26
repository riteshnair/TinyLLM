# Engine Implementation Skill Bundle

This directory contains the curated skills used to build and maintain the tiny CPU-first, Vulkan-primary LLM engine. It is a vendored working bundle for human programmers and coding agents. The repository-root skill directories remain canonical; this copy travels with the implementation so contributors can use the same guidance without relying on a global installation.

## Conditional loading

Do not load every file for every task. Start with the smallest matching skill, then add one skill only when the change crosses a concrete boundary.

| Work | Start with | Add only when needed |
|---|---|---|
| Architecture or module split | `system-architecture` | `modular-component-boundaries`, `traceability` |
| New or changed production code | `implementation-integrity` | `code-contract-comments`, `debug-core` |
| C99 core | `c99-systems` | `memory-management`, `toolchains` |
| C++ orchestration | `cpp-systems` | `memory-management`, `toolchains` |
| GGUF or SafeTensors | `gguf-format` or `safetensors-format` | `_shared/model-format-checklist.md` |
| CPU/Vulkan execution | `llm-components` | `vulkan-compute`, `graphics-shader-kernels`, `kernel-tuning` |
| ROCr/ROCm | `rocr-runtime` or `rocm-stack` | `cdna`/`rdna` from the root bundle, `kernel-tuning`, `_shared/version-policy.md` |
| CUDA | `cuda-stack` | `kernel-tuning`, `_shared/version-policy.md` |
| Windows graphics/ML | `directx-ai-ml` | `windows-system-architecture`, `graphics-shader-kernels` |
| Lazy loading and memory | `memory-management` | `linux-systems` or `windows-system-architecture`, `_shared/porting-checklist.md` |
| Fast troubleshooting | `debug-core` | `debug-reproduce`, `debug-localize`, `debug-invariants`, `debug-verify` |
| Cross-platform port | `cross-porting` | `cross-compilation`, `plugin-adapter`, `_shared/porting-checklist.md` |
| Toolchain or build | `toolchains` | `cross-compilation`, `dox-validate` |

Use `systems-ml-stack-router` only when the task spans multiple areas or the correct specialist is unclear. The six files under `_shared/` are on-demand references, not auto-loaded skills.

## Required implementation sequence

Every change follows:

```text
module outline -> public contract -> failing test -> smallest correct path
-> CPU/reference result -> source/RAM budget -> accelerated variant
-> differential test -> probe trace -> benchmark -> focused commit
```

Before editing, record the module’s inputs, outputs, ownership, synchronization, errors, backend applicability, budgets, tests, CLI switches, acceptance criteria, and explicit exclusions. Keep optional server, WebUI, vendor, shader-generation, conversion, and cluster code outside the mandatory core.

## Quality requirements

Production code must not contain fake success paths, placeholder returns, demo branches masquerading as production, unchecked mocks, or claims unsupported by executed evidence. Unsupported features return a named error and appear in the resolved configuration. Function comments must document non-obvious preconditions, ownership, lifetime, errors, synchronization, and performance constraints.

Use the probe-bus pattern to follow a trace packet from ingress to egress. Record IDs, stages, hashes, shapes, dtypes, device locations, page identities, and timing; capture raw prompts, outputs, or tensors only by explicit opt-in. Use deterministic replay and CPU-versus-backend differential comparison to locate the first divergence.

## Shared references

- `_shared/shared-execution-protocol.md` — evidence labels and execution workflow.
- `_shared/quality-gates.md` — completeness and anti-placeholder checks.
- `_shared/porting-checklist.md` — platform capability mapping.
- `_shared/model-format-checklist.md` — GGUF/SafeTensors validation.
- `_shared/version-policy.md` — release, preview, driver, SDK, and hardware claims.
- `_shared/suite-manifest.md` — bundle inventory and loading policy.

## Bundle policy

Update the root canonical skill first, then refresh this vendored copy. Do not create a second divergent implementation of a skill under `code/skills`. If a module needs a new specialist, add it to the root suite, define a narrow trigger, validate it, then add it to this bundle and update this table.
