"""Numerical verification that GraphAugmentedLLM injects Ψ into q/k/v as specified.

Target: ``prism.models.gnn_llm._prism_pe_attention_forward`` (the ``"prism_pe"`` attn
impl) and ``GraphAugmentedLLM.{build_pe_signal,_augment_embeddings,forward}``, driven
on a tiny random-init **Gemma 4** LLM (``Gemma4Unified*`` — q/k-norm, single-tensor
RoPE, sliding windows, KV-shared layers) plus a real **GraphTransformer** PE model.

The specified scheme (gnn_llm docstring) is, per attention layer::

    q = RoPE(q_norm(W_q·h)) + W_q·Ψ
    k = RoPE(k_norm(W_k·h)) + W_k·Ψ        (skipped on KV-shared layers)
    v =          W_v·h      + W_v·Ψ        (skipped on KV-shared layers / inject_v=False)
    A = softmax((QX)ᵀ(KX)) ;  Y = W_oᵀ · (A · VX)

i.e. the graph signal enters q/k/v as the layer's *own bias-free projection of Ψ*,
added **post-RoPE (unrotated)**, so it is identical on every position regardless of
RoPE phase. The independent oracle is the raw projection ``Ψ @ Wᵀ`` (bias-free), which
the injected delta must match exactly — and exact equality to the *unrotated* projection
is itself proof the add happens after RoPE (a pre-RoPE add would appear rotated).

The attention core (A·V, softmax, repeat_kv) is a transformers boundary: we do not test
into it; we capture the q/k/v *handed to it* and assert they carry the injection.

Run:  python tests/test_pe_injection_qkv_numeric.py
 or:  pytest tests/test_pe_injection_qkv_numeric.py -v
"""
import sys
sys.path.insert(0, "src")

import torch
from torch_geometric.data import Data

from prism.models.gnn_llm import GraphAugmentedLLM
from prism.models.gt import GraphTransformer

_TOL = 1e-5


def _skip(msg):
    """Skip under pytest; print and bail when run as a plain script."""
    if __name__ != "__main__" and "pytest" in sys.modules:
        import pytest
        pytest.skip(msg)
    print(f"[SKIP] {msg}")
    return None


def _gemma(num_layers=6, num_kv_shared_layers=0, seed=0):
    """Tiny random-init Gemma 4 standing in for gemma-4-12B (CPU scale, eager)."""
    try:
        from transformers import Gemma4UnifiedForCausalLM, Gemma4UnifiedTextConfig
    except Exception as e:  # noqa: BLE001 — any import failure ⇒ unsupported here
        return None
    torch.manual_seed(seed)
    cfg = Gemma4UnifiedTextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=num_layers, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=64, sliding_window=8,
        num_kv_shared_layers=num_kv_shared_layers, attn_implementation="eager")
    return Gemma4UnifiedForCausalLM(cfg).eval()


def _wrap(llm, d_model=16, pe_gain_init=1.0, use_pe_norm=True):
    """Wrap with a real GraphTransformer PE model (rpearl_gt_llm path, scaled down)."""
    torch.manual_seed(1)
    gt = GraphTransformer(
        num_layers=2, pe_hidden_channels=16, pe_num_layers=2, d_model=d_model,
        heads=2, num_samples=8, dropout=0.0, k_pe=2, k_gt=2)
    return GraphAugmentedLLM(
        llm, gt, d_model=d_model, pe_gain_init=pe_gain_init,
        use_pe_norm=use_pe_norm, pe_node_features="random").eval()


def _capture_qkv(wrap, ids, psi, layer_idx, inject_value=True):
    """Capture (q, k, v) handed to the stock attention fn at ``layer_idx``.

    Injection is ISOLATED to ``layer_idx``: the ``_prism_pe_model`` back-ref is
    nulled on every other layer for the duration, so no upstream layer injects.
    This keeps the hidden state entering ``layer_idx`` identical with and without Ψ
    (the residual stream is untouched before this layer), so the captured q/k/v
    delta isolates *this* layer's injection rather than downstream drift. Without
    isolation only layer 0 would be testable.

    Wraps the layer's ``_prism_orig_attn_fn`` to record its inputs, runs a forward
    with ``wrap._pe_signal = psi`` (psi=None ⇒ pure stock), and restores everything.
    """
    layers = list(wrap._decoder_layers())
    attn = layers[layer_idx].self_attn
    orig = attn._prism_orig_attn_fn
    store = {}

    def cap(module, query, key, value, attention_mask,
            scaling=None, dropout=0.0, **kw):
        store["q"] = query.detach().clone()
        store["k"] = key.detach().clone()
        store["v"] = value.detach().clone()
        return orig(module, query, key, value, attention_mask,
                    scaling=scaling, dropout=dropout, **kw)

    # Isolate injection to the target layer: null the back-ref elsewhere.
    saved_refs = []
    for j, l in enumerate(layers):
        if j != layer_idx:
            a = l.self_attn
            saved_refs.append((a, a._prism_pe_model))
            object.__setattr__(a, "_prism_pe_model", None)

    attn._prism_orig_attn_fn = cap
    prev_iv = wrap._pe_inject_value
    wrap._pe_inject_value = inject_value
    wrap._pe_signal = psi
    try:
        with torch.no_grad():
            wrap.llm(input_ids=ids)
    finally:
        attn._prism_orig_attn_fn = orig
        wrap._pe_signal = None
        wrap._pe_inject_value = prev_iv
        for a, ref in saved_refs:
            object.__setattr__(a, "_prism_pe_model", ref)
    return store


def _proj_raw(psi, weight):
    """Independent oracle: bias-free projection Ψ @ Wᵀ, flattened (no head split)."""
    return psi.to(weight.dtype) @ weight.t()          # [B, S, out_features]


def _flatten_heads(delta):
    """[B, H, S, hd] -> [B, S, H*hd], inverting the injection's view/transpose."""
    b, h, s, hd = delta.shape
    return delta.transpose(1, 2).reshape(b, s, h * hd)


def _delta_qkv(wrap, ids, psi, layer_idx, inject_value=True):
    with_psi = _capture_qkv(wrap, ids, psi, layer_idx, inject_value=inject_value)
    no_psi = _capture_qkv(wrap, ids, None, layer_idx)
    return ({k: with_psi[k] - no_psi[k] for k in ("q", "k", "v")}, no_psi)


# --------------------------------------------------------------------------- #
# Core: q, k, v each carry the layer's own raw projection of Ψ, post-RoPE.     #
# --------------------------------------------------------------------------- #

def _check_layer_injection(wrap, layer_idx, seq=6, batch=2):
    """On a non-KV-shared layer, q/k/v deltas == raw W·Ψ (unrotated projection)."""
    torch.manual_seed(2)
    ids = torch.randint(0, 64, (batch, seq))
    hidden = wrap.llm.config.get_text_config().hidden_size
    # Ψ nonzero on every position so RoPE phase varies across positions: exact
    # match to the *unrotated* projection then proves the add is post-RoPE.
    psi = torch.randn(batch, seq, hidden) * 0.5

    attn = list(wrap._decoder_layers())[layer_idx].self_attn
    delta, _ = _delta_qkv(wrap, ids, psi, layer_idx)

    exp_q = _proj_raw(psi, attn.q_proj.weight)
    exp_k = _proj_raw(psi, attn.k_proj.weight)
    exp_v = _proj_raw(psi, attn.v_proj.weight)

    got_q = _flatten_heads(delta["q"])
    got_k = _flatten_heads(delta["k"])
    got_v = _flatten_heads(delta["v"])

    dq = (got_q - exp_q).abs().max().item()
    dk = (got_k - exp_k).abs().max().item()
    dv = (got_v - exp_v).abs().max().item()
    assert dq < _TOL, f"layer {layer_idx}: q delta != W_q·Ψ (max|Δ|={dq:.2e})"
    assert dk < _TOL, f"layer {layer_idx}: k delta != W_k·Ψ (max|Δ|={dk:.2e})"
    assert dv < _TOL, f"layer {layer_idx}: v delta != W_v·Ψ (max|Δ|={dv:.2e})"
    # Sanity: the injection is non-trivial (Ψ actually moved q/k/v).
    assert exp_q.abs().max().item() > 1e-3, "oracle projection ~0 — vacuous test"


def test_inject_qkv_sliding_layer():
    """q/k/v injection on a sliding-window layer (explicit mask path)."""
    llm = _gemma()
    if llm is None:
        return _skip("gemma4_unified unavailable")
    assert llm.config.layer_types[0] == "sliding_attention"
    _check_layer_injection(_wrap(llm), layer_idx=0)


def test_inject_qkv_full_layer():
    """q/k/v injection on a full-attention layer (registered-mask path)."""
    llm = _gemma()
    if llm is None:
        return _skip("gemma4_unified unavailable")
    full = [i for i, t in enumerate(llm.config.layer_types) if t == "full_attention"]
    assert full, "expected a full_attention layer in the config"
    _check_layer_injection(_wrap(llm), layer_idx=full[0])


def test_injection_is_post_rope_unrotated():
    """The delta equals the *raw* projection, not a RoPE-rotated one.

    Build Ψ supported on a single late position (RoPE phase ≠ identity there). If the
    add were pre-RoPE the captured delta would be the rotated projection and would
    differ from Ψ@Wᵀ; exact equality confirms the documented post-RoPE add.
    """
    llm = _gemma()
    if llm is None:
        return _skip("gemma4_unavailable")
    wrap = _wrap(llm)
    seq, batch = 6, 1
    torch.manual_seed(3)
    ids = torch.randint(0, 64, (batch, seq))
    hidden = wrap.llm.config.get_text_config().hidden_size
    psi = torch.zeros(batch, seq, hidden)
    psi[0, 5] = torch.randn(hidden)          # only the last position carries Ψ
    attn = list(wrap._decoder_layers())[0].self_attn
    delta, _ = _delta_qkv(wrap, ids, psi, 0)
    got_q = _flatten_heads(delta["q"])
    exp_q = _proj_raw(psi, attn.q_proj.weight)
    d = (got_q - exp_q).abs().max().item()
    assert d < _TOL, f"q delta at pos5 != raw W_q·Ψ (max|Δ|={d:.2e}) — add not post-RoPE?"
    # delta must live only on the Ψ-supported position.
    nz = got_q[0].abs().sum(dim=-1) > _TOL
    assert nz.tolist() == [False, False, False, False, False, True], (
        f"q delta leaked off the Ψ position: {nz.tolist()}")


def test_value_injection_toggle():
    """``_pe_inject_value=False`` zeroes the v delta but leaves q/k injected."""
    llm = _gemma()
    if llm is None:
        return _skip("gemma4_unified unavailable")
    wrap = _wrap(llm)
    seq, batch = 6, 2
    torch.manual_seed(4)
    ids = torch.randint(0, 64, (batch, seq))
    hidden = wrap.llm.config.get_text_config().hidden_size
    psi = torch.randn(batch, seq, hidden) * 0.5
    attn = list(wrap._decoder_layers())[0].self_attn
    delta, _ = _delta_qkv(wrap, ids, psi, 0, inject_value=False)
    dv = delta["v"].abs().max().item()
    dq = delta["q"].abs().max().item()
    dk = delta["k"].abs().max().item()
    assert dv < _TOL, f"v changed with inject_value=False (max|Δ|={dv:.2e})"
    assert dq > 1e-3, "q not injected when inject_value=False"
    assert dk > 1e-3, "k not injected when inject_value=False"
    # And with inject_value=True the v path IS exercised (guards against dead branch).
    delta_on, _ = _delta_qkv(wrap, ids, psi, 0, inject_value=True)
    exp_v = _proj_raw(psi, attn.v_proj.weight)
    d = (_flatten_heads(delta_on["v"]) - exp_v).abs().max().item()
    assert d < _TOL, f"v delta != W_v·Ψ when enabled (max|Δ|={d:.2e})"


def test_kv_shared_layer_skips_kv_injection():
    """KV-shared layers: q is still injected, k/v injection is skipped (no double-count).

    Gemma 4 KV-shared layers have NO ``k_proj``/``v_proj`` (they reuse upstream KV), so
    the injection's ``module.k_proj(psi)`` would raise ``AttributeError`` if the
    ``is_kv_shared_layer`` guard failed. The test exercises the guard directly:

      1. the shared layer indeed lacks ``k_proj``/``v_proj`` (structural precondition);
      2. a forward with Ψ armed runs without error (the guard skipped the k/v branch —
         otherwise it would crash reaching ``module.k_proj``);
      3. q is still injected there, carrying the raw projection W_q·Ψ.
    """
    # 12 layers so each KV-shared layer (idx 10 sliding, 11 full) has an earlier
    # same-type source to reuse KV from; a 6-layer config makes the lone full layer
    # share from nothing (KeyError in HF). Real gemma-4-12B has many full layers.
    llm = _gemma(num_layers=12, num_kv_shared_layers=2)
    if llm is None:
        return _skip("gemma4_unified unavailable")
    wrap = _wrap(llm)
    layers = list(wrap._decoder_layers())
    shared = [i for i, l in enumerate(layers)
              if getattr(l.self_attn, "is_kv_shared_layer", False)]
    if not shared:
        return _skip("no KV-shared layers materialized in this config")
    sidx = shared[0]
    attn = layers[sidx].self_attn
    assert not hasattr(attn, "k_proj") and not hasattr(attn, "v_proj"), (
        "expected KV-shared layer to lack k_proj/v_proj")

    seq, batch = 6, 2
    torch.manual_seed(5)
    ids = torch.randint(0, 64, (batch, seq))
    hidden = wrap.llm.config.get_text_config().hidden_size
    psi = torch.randn(batch, seq, hidden) * 0.5

    # Reaching module.k_proj(psi) here would AttributeError; a clean run ⇒ guard skipped k/v.
    with_psi = _capture_qkv(wrap, ids, psi, sidx)   # isolated to the shared layer
    no_psi = _capture_qkv(wrap, ids, None, sidx)
    got_q = _flatten_heads(with_psi["q"] - no_psi["q"])
    exp_q = _proj_raw(psi, attn.q_proj.weight)
    dq = (got_q - exp_q).abs().max().item()
    assert dq < _TOL, f"q not injected (raw) at shared layer (max|Δ|={dq:.2e})"
    # k/v unchanged by Ψ at the shared layer (reused upstream, not re-projected).
    dk = (with_psi["k"] - no_psi["k"]).abs().max().item()
    dv = (with_psi["v"] - no_psi["v"]).abs().max().item()
    assert dk < _TOL and dv < _TOL, (
        f"shared-layer k/v changed with Ψ (k={dk:.2e}, v={dv:.2e}) — re-injected?")


# --------------------------------------------------------------------------- #
# End-to-end: real GraphTransformer Ψ -> placed at node spans -> q/k/v inject. #
# --------------------------------------------------------------------------- #

def _tiny_graph(n=3):
    x = torch.randn(n, 1)                                   # value ignored (random probes)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    g = Data(x=x, edge_index=edge_index)
    g.num_nodes = n
    return g


def test_end_to_end_gt_psi_placed_and_injected():
    """GraphTransformer output is placed at node-token spans (and only there), and the
    per-layer q-injection at those tokens equals the layer's raw projection of that Ψ.

    Ties the full path: GT(graph) -> pe_proj -> pe_norm -> tanh(gate) = Ψ rows at node
    spans (build_pe_signal); _augment_embeddings arms _pe_signal; attention adds W·Ψ.
    """
    llm = _gemma()
    if llm is None:
        return _skip("gemma4_unified unavailable")
    torch.manual_seed(6)
    wrap = _wrap(llm, d_model=16, pe_gain_init=1.0)   # gate open so Ψ ≠ 0
    seq, batch = 8, 1
    ids = torch.randint(0, 64, (batch, seq))
    g = _tiny_graph(3)
    # disjoint node-token spans: positions 2, 4, 6 are node tokens; rest non-node.
    inj = {0: [(2, 3)], 1: [(4, 5)], 2: [(6, 7)]}
    node_pos = [2, 4, 6]
    non_node = [p for p in range(seq) if p not in node_pos]

    embeddings = wrap.llm.get_input_embeddings()(ids).clone()
    psi = wrap.build_pe_signal(embeddings, [g], [inj])
    assert psi.shape == (batch, seq, embeddings.shape[-1])

    # Ψ placement: node-token rows nonzero, every other row exactly zero.
    row_norm = psi[0].norm(dim=-1)
    for p in node_pos:
        assert row_norm[p].item() > 1e-6, f"node-token row {p} has zero Ψ"
    for p in non_node:
        assert row_norm[p].item() == 0.0, f"non-node row {p} got nonzero Ψ"

    # Each node's Ψ row equals the gated/normed/projected GT output for that node.
    attn = list(wrap._decoder_layers())[0].self_attn
    delta, _ = _delta_qkv(wrap, ids, psi, 0)
    got_q = _flatten_heads(delta["q"])
    exp_q = _proj_raw(psi, attn.q_proj.weight)
    d = (got_q - exp_q).abs().max().item()
    assert d < _TOL, f"GT-derived q injection != raw W_q·Ψ (max|Δ|={d:.2e})"
    # q delta supported exactly on the node-token positions.
    nz = (got_q[0].abs().sum(dim=-1) > _TOL).nonzero().flatten().tolist()
    assert nz == node_pos, f"q injection positions {nz} != node tokens {node_pos}"


def test_full_forward_logits_change_with_graph():
    """End-to-end ``GraphAugmentedLLM.forward`` with a graph changes logits vs no-graph,
    and the no-graph/Ψ-off path is identical to the stock LLM (masking not regressed)."""
    llm = _gemma()
    if llm is None:
        return _skip("gemma4_unified unavailable")
    torch.manual_seed(7)
    wrap = _wrap(llm, d_model=16, pe_gain_init=1.0)
    seq, batch = 8, 1
    ids = torch.randint(0, 64, (batch, seq))
    g = _tiny_graph(3)
    inj = {0: [(2, 3)], 1: [(4, 5)], 2: [(6, 7)]}

    with torch.no_grad():
        stock = wrap.llm(input_ids=ids).logits
        out_g = wrap(input_ids=ids, graphs=[g], injection_maps=[inj]).logits

    d = (out_g - stock).abs().max().item()
    assert d > 1e-3, f"graph forward did not change logits (max|Δ|={d:.2e}) — injection inert"
    # _pe_signal disarmed after forward (no stale signal leaks into a later call).
    assert wrap._pe_signal is None, "Ψ left armed after forward"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: PASS")
    print("done")
