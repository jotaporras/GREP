"""Post-RoPE Ψ-injection patch for vLLM's Gemma-4 attention.

Mirrors ``gnn_llm._prism_pe_attention_forward`` exactly: Ψ passes through the
layer's own (fused) qkv projection weight, and the un-normed, un-rotated
contributions are added AFTER q/k norm and RoPE —

- ``q += W_q·Ψ`` on every layer;
- ``k += W_k·Ψ``;
- ``v += W_v·Ψ`` gated by ``wrapper._pe_inject_value`` and skipped on
  ``use_k_eq_v`` layers, where the HF model has no ``v_proj`` at all and the HF
  injection therefore leaves value untouched.

KV-shared configs (``num_kv_shared_layers > 0``) are REFUSED at install: in HF,
``shared_kv_states`` is captured BEFORE the injection interface runs, so shared
layers attend Ψ-FREE keys while the source layer's own attention sees Ψ-carrying
ones — two different k tensors from one layer. vLLM's single paged cache cannot
express that (whatever the source layer writes is what sharers read), so any
port would silently diverge. The 31B target has zero KV-shared layers; fail
loud rather than approximate. (Note gnn_llm.py:38-40's comment assumes the
shared k/v "already carry Ψ" — the Gemma4 capture point makes that untrue.)

Identity-RoPE (``disable_graph_token_rope`` checkpoints): the transport's span
mask zeroes ``positions`` at injected spans before ``rotary_emb``, reproducing
``graph_token_position_ids`` inside the engine. Decode tokens are never in the
span mask, so they keep natural positions — same prompt-only semantics as the
HF path.

With Ψ absent — or on decode steps, whose packed batch length no longer matches
the armed Ψ — every branch falls through and the forward is the stock Gemma-4
computation.
"""
from __future__ import annotations


def install_psi_injection(wrapper) -> None:
    """Replace each ``Gemma4Attention.forward`` with the Ψ-aware equivalent.

    ``wrapper`` is the registered wrapper model
    (:class:`prism.models.vllm_graph.model.GraphGemma4ForCausalLM`); the patch
    reads ``wrapper._psi_packed`` / ``wrapper._span_mask_packed`` armed per
    forward by ``embed_input_ids``.
    """
    layers = wrapper.language_model.model.layers
    for layer in layers:
        attn = layer.self_attn
        # Ψ=0 at non-graph tokens, so proj(Ψ)=0 only if the projection is
        # bias-free (same contract as GraphAugmentedLLM._install_pe_injection);
        # a biased base model would mean the HF and vLLM models disagree —
        # fail loud instead of silently diverging.
        if attn.qkv_proj.bias is not None:
            raise ValueError(
                f"{type(attn).__name__}.qkv_proj has a bias; prism Ψ-injection assumes "
                "bias-free attention projections so non-graph tokens stay untouched."
            )
        if attn.is_kv_shared_layer:
            raise ValueError(
                "KV-shared layers cannot carry the Ψ channel under vLLM: HF captures "
                "shared_kv_states before Ψ is added (shared layers attend Ψ-free keys), "
                "but vLLM's paged cache persists whatever the source layer computed — "
                "the two semantics cannot both hold. The gemma-4-31B target has "
                "num_kv_shared_layers=0; refusing rather than silently diverging."
            )
        attn.forward = _make_forward(wrapper, attn)


def _make_forward(wrapper, attn):
    def forward(positions, hidden_states, **kwargs):
        qkv, _ = attn.qkv_proj(hidden_states)
        q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)

        q = attn.q_norm(q.unflatten(-1, (attn.num_heads, attn.head_dim)))
        q = q.flatten(-2, -1)

        psi = wrapper._psi_packed
        inject = psi is not None and psi.shape[0] == hidden_states.shape[0]
        if psi is not None and not inject:
            wrapper.dbg["attn_skip_shape"] += 1
        pos = positions
        if inject:
            wrapper.dbg["attn_hit"] += 1
            # Through the MODULE, not F.linear on .weight: under bnb the raw
            # weight is a packed uint8 blob, and under a LoRARequest the module
            # carries the adapter — both mirror the HF path, where Ψ passes the
            # (PEFT-adapted, quant-aware) q/k/v projections themselves.
            qkv_psi, _ = attn.qkv_proj(
                psi.to(dtype=hidden_states.dtype, device=hidden_states.device))
            q_psi, k_psi, v_psi = qkv_psi.split(
                [attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
            if wrapper._identity_rope:
                # graph_token_position_ids, engine-side: injected spans rotate
                # at position 0. Prompt-only by construction (decode rows are
                # never in the span mask).
                pos = positions.clone()
                pos[wrapper._span_mask_packed] = 0

        k = attn.k_norm(k.unflatten(-1, (attn.num_kv_heads, attn.head_dim)))
        k = k.flatten(-2, -1)
        q, k = attn.rotary_emb(pos, q, k)
        v = attn.v_norm(v.unflatten(-1, (attn.num_kv_heads, attn.head_dim)))
        v = v.flatten(-2, -1)
        if inject:
            q = q + q_psi
            k = k + k_psi
            if wrapper._pe_inject_value and not attn.use_k_eq_v:
                v = v + v_psi

        attn_output = attn.attn(q, k, v)
        output, _ = attn.o_proj(attn_output)
        return output

    return forward
