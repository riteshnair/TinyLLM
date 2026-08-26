# Narrow MoE tensor mapping notes

Source inspected on 2026-08-26:
- https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/tensor_mapping.py

The current upstream mapping source shows multiple architecture-specific patterns. A narrow first profile should not guess across them. The Mixtral-specific entries include `layers.{bid}.feed_forward.experts.w1` for merged gate/up expert weights and `layers.{bid}.feed_forward.experts.w2` for merged down expert weights. Other current architectures use different prefixes and roles, including `model.layers.{bid}.mlp.experts.gate_proj`, `up_proj`, and `down_proj`.

Implementation decision:

Use an explicit profile identifier and recognize only one exact profile initially. The first profile should parse the exact Mixtral-style names, require a clear layer index, expert tensor role, and expert dimension contract, and return named unsupported for other patterns. Do not silently interpret merged expert tensors as separate gate/up matrices; mapping must record the role and rank/dimensions for a later execution contract.
