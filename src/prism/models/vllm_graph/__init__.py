"""Graph-conditioned vLLM engine: GREP Ψ-injection through the multimodal channel.

Lifted from ``notebooks/2026-08-07 fable-vllm-graph-demo.ipynb`` and ported from
Qwen2 to Gemma-4. The Ψ transport tensor rides vLLM's multimodal "image"
channel into a registered wrapper model whose patched attention layers add
W·Ψ post-RoPE, mirroring ``gnn_llm._prism_pe_attention_forward``.

Submodules (import explicitly — everything except :mod:`psi` imports vllm,
which is only installed in the rollout/eval environments):

- ``psi``        driver-side Ψ transport construction (torch + prism only)
- ``processor``  multimodal plumbing (parser / processor / dummy builder)
- ``model``      the registered ``GraphGemma4ForCausalLM`` wrapper
- ``attention``  the Gemma-4 attention patch
- ``engine``     ``build_graph_llm`` — engine construction from a checkpoint
"""
