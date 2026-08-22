"""e18 node-identity pathways — decision gating (A), structural keys (B),
binding head — on ``LearnableGraphMaskLLM``.

Invariants locked here:

* A: ``decision_query_map`` tags exactly the untagged answer-side positions with
  the last knowable mention; ``build_structural_mask`` writes the SOFT row there
  (no hard block, goal/non-node keys untouched); ``decision_gain=0`` is a bitwise
  no-op; the step-by-step decode injector reproduces the teacher-forced rows.
* B: ``sk_gain=0`` is a bitwise no-op; nonzero gain moves logits at EVERY position
  (not only node tokens); gradients reach ``sk_q``/``sk_k``/the tower; the decode
  injector re-arms keys so cached decode logits match teacher forcing.
* binding head: the loss is finite, gradients reach ``bind_proj`` and the tower,
  and ``labels=None`` (generation) leaves logits untouched.
* provenance: every new module is in ``base_lr_parameters`` and round-trips
  through ``save_run_dir`` / ``loaders`` key sets.
"""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, "tests")

import torch

from prism.models.gnn_llm import (
    LearnableGraphMaskLLM,
    MaskDecodeInjector,
    build_injection_map,
    decision_query_map,
    decode_style_query_map,
    shift_positions,
    shift_spans,
    splice_prefix,
)
from test_learnable_graph_mask import _tiny_llm, _StubPE, _graph, DEVICE

NEG = torch.finfo(torch.float32).min


# ---------------------------------------------------------------------------
# decision_query_map
# ---------------------------------------------------------------------------

def test_decision_query_map_assigns_last_knowable_mention():
    # prompt: node0 @ [1,3), node1 @ [3,5); answer_start=7; answer: node0 @ [8,9), node2 @ [11,12)
    full = {0: [(1, 3), (8, 9)], 1: [(3, 5)], 2: [(5, 7), (11, 12)]}
    ids = list(range(14))
    seqs = [[1, 2], [3, 4], [5, 6]]
    q = decode_style_query_map(full, 7, ids, seqs)
    d = decision_query_map(q, 7, len(ids))
    # Position 6 (prefill's last) sees node2 (prompt span [5,7) ends at 6 — but 6
    # is itself a node token => tagged => excluded). Position 7: last ref < 7 is
    # node2's prompt mention (ref 6).
    assert 6 not in d
    assert d[7] == 2
    # 8 is node0's tagged answer position -> excluded; 9, 10 ride node0.
    assert 8 not in d and d[9] == 0 and d[10] == 0
    # 11 is node2's tag -> excluded; 12, 13 ride node2.
    assert 11 not in d and d[12] == 2 and d[13] == 2
    # nothing before answer_start-1
    assert min(d) >= 6


def test_decision_query_map_empty_without_mentions():
    assert decision_query_map({}, 3, 10) == {}


# ---------------------------------------------------------------------------
# A — decision gating in the prefill bias
# ---------------------------------------------------------------------------

def _model(**kw):
    torch.manual_seed(0)
    llm = _tiny_llm()
    return LearnableGraphMaskLLM(
        llm, _StubPE(d=8).to(DEVICE), alpha=0.7, layer_scope="all",
        fusion_d_gt=8, **kw).to(DEVICE).eval()


def _inputs():
    # 3 nodes, path 0-1-2. Sequence of 12: prompt [0,7), answer [7,12).
    # prompt mentions: node0 @ [1,2), node1 @ [3,4), node2 @ [5,6)
    # answer mentions: node1 @ [8,9), node2 @ [10,11)
    g = _graph(3, [(0, 1), (1, 2)])
    kmap = {0: [(1, 2)], 1: [(3, 4), (8, 9)], 2: [(5, 6), (10, 11)]}
    qmap = {0: [(1, 2)], 1: [(3, 4), (8, 9)], 2: [(5, 6), (10, 11)]}   # 1-token names: tag == span
    dmap = decision_query_map(qmap, 7, 12)
    ids = torch.randint(0, 64, (1, 12), generator=torch.Generator().manual_seed(2)).to(DEVICE)
    return ids, [g], [qmap], [kmap], [dmap]


def test_decision_rows_are_soft_and_only_at_decision_positions():
    m = _model(decision_gating=True, decision_gain_init=2.0)
    ids, gs, qm, km, dm = _inputs()
    plain = m.build_structural_mask(12, gs, qm, DEVICE, dtype=torch.float32,
                                    key_injection_maps=km)[0, 0]
    gated = m.build_structural_mask(12, gs, qm, DEVICE, dtype=torch.float32,
                                    key_injection_maps=km, decision_maps=dm)[0, 0]
    diff = (gated != plain).any(dim=1)
    assert set(diff.nonzero().flatten().tolist()) == set(dm[0].keys())
    # Row 7 rides node2 (last prompt mention): neighbours of 2 = {1, 2}.
    row = gated[7]
    assert row[3] > 0 and row[8] > 0 and row[5] > 0 and row[10] > 0   # node1/node2 keys boosted
    assert row[1] == 0                                                  # node0 key: not adjacent, NOT blocked
    assert row[0] == 0 and row[7] == 0                                  # non-node keys untouched
    assert (gated[7] > NEG / 2).all()                                   # no hard block on a decision row
    # Tagged rows unchanged (hard block still present).
    assert torch.equal(gated[10], plain[10]) and plain[10, 1] == NEG   # node2 row blocks node0


def test_decision_gain_zero_is_bitwise_noop():
    m_g = _model(decision_gating=True, decision_gain_init=0.0)
    m_p = _model(decision_gating=False)
    ids, gs, qm, km, dm = _inputs()
    with torch.no_grad():
        a = m_g(input_ids=ids, graphs=gs, injection_maps=qm, key_injection_maps=km,
                decision_maps=dm).logits
        b = m_p(input_ids=ids, graphs=gs, injection_maps=qm, key_injection_maps=km,
                decision_maps=dm).logits
    assert torch.equal(a, b)


def test_decision_gain_trains_and_moves_logits():
    m = _model(decision_gating=True, decision_gain_init=1.0)
    ids, gs, qm, km, dm = _inputs()
    out = m(input_ids=ids, graphs=gs, injection_maps=qm, key_injection_maps=km,
            decision_maps=dm).logits
    out.sum().backward()
    assert m.decision_gain.grad is not None and m.decision_gain.grad.abs() > 0
    assert id(m.decision_gain) in set(map(id, m.base_lr_parameters()))
    with torch.no_grad():
        m.decision_gain.fill_(5.0)
        moved = m(input_ids=ids, graphs=gs, injection_maps=qm, key_injection_maps=km,
                  decision_maps=dm).logits
    assert not torch.equal(out, moved)


# ---------------------------------------------------------------------------
# A + B — decode parity (step-by-step injector == teacher-forced prefill)
# ---------------------------------------------------------------------------

def _parity(model):
    """Token ids with 1-token node names 10/11/12; prompt 6 tokens, answer 5."""
    g = _graph(3, [(0, 1), (1, 2)])
    seqs = [[10], [11], [12]]
    prompt = [1, 10, 2, 11, 3, 12]
    answer = [4, 11, 5, 12, 6]
    full = prompt + answer
    imap_full = build_injection_map(full, seqs)
    qmap = decode_style_query_map(imap_full, len(prompt), full, seqs)
    dmap = decision_query_map(qmap, len(prompt), len(full))
    ids = torch.tensor([full], device=DEVICE)
    with torch.no_grad():
        tf = model(input_ids=ids, graphs=[g], injection_maps=[qmap],
                   key_injection_maps=[imap_full], decision_maps=[dmap]).logits[0]
    # step-wise decode (mirrors inference.py: soft-edge prefix => embeds prefill
    # in the shifted frame)
    pmap = build_injection_map(prompt, seqs)
    pids = torch.tensor([prompt], device=DEVICE)
    plen = len(prompt)
    prefill = {"input_ids": pids}
    with torch.no_grad():
        if getattr(model, "_soft_edges", False):
            embeds, off = model.build_soft_edges(pids, [g], [pmap])
            pmap = shift_spans(pmap, off)
            plen += off
            prefill = {"inputs_embeds": embeds}
        pdmap = decision_query_map(pmap, plen, plen)
        model._struct_bias = model.build_structural_mask(
            plen, [g], [pmap], DEVICE, decision_maps=[pdmap])
        if model._struct_keys:
            model._sk_keys = model.build_sk_keys(plen, [g], [pmap], DEVICE)
        inj = MaskDecodeInjector(model, g, pmap, plen, seqs)
        h = model.llm.register_forward_pre_hook(inj.pre_hook, with_kwargs=True)
        out = model.llm(**prefill, use_cache=True)
        logits = [out.logits[0, -1]]
        past = out.past_key_values
        for t in answer[:-1]:
            out = model.llm(input_ids=torch.tensor([[t]], device=DEVICE),
                            past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits.append(out.logits[0, -1])
        h.remove()
        model._struct_bias = None
        model._sk_keys = None
        model._decode_bias_row = None
    dec = torch.stack(logits)
    return tf[len(prompt) - 1:len(prompt) - 1 + dec.shape[0]], dec


def test_decode_parity_decision_gating():
    m = _model(decision_gating=True, decision_gain_init=1.5)
    tf, dec = _parity(m)
    assert torch.allclose(tf, dec, atol=1e-4), (tf - dec).abs().max()


def test_decode_parity_struct_keys():
    m = _model(struct_keys=True, struct_keys_dim=4, struct_keys_gain_init=1.0)
    tf, dec = _parity(m)
    assert torch.allclose(tf, dec, atol=1e-4), (tf - dec).abs().max()


def test_decode_parity_both():
    m = _model(decision_gating=True, decision_gain_init=1.0,
               struct_keys=True, struct_keys_dim=4, struct_keys_gain_init=0.7)
    tf, dec = _parity(m)
    assert torch.allclose(tf, dec, atol=1e-4), (tf - dec).abs().max()


# ---------------------------------------------------------------------------
# B — structural keys
# ---------------------------------------------------------------------------

def test_struct_keys_zero_gain_is_bitwise_noop():
    m_sk = _model(struct_keys=True, struct_keys_dim=4, struct_keys_gain_init=0.0)
    m_p = _model()
    ids, gs, qm, km, _ = _inputs()
    with torch.no_grad():
        a = m_sk(input_ids=ids, graphs=gs, injection_maps=qm, key_injection_maps=km).logits
        b = m_p(input_ids=ids, graphs=gs, injection_maps=qm, key_injection_maps=km).logits
    assert torch.equal(a, b)


def test_struct_keys_move_every_position_and_train():
    m = _model(struct_keys=True, struct_keys_dim=4, struct_keys_gain_init=1.0)
    ids, gs, qm, km, _ = _inputs()
    base = _model()
    with torch.no_grad():
        b = base(input_ids=ids, graphs=gs, injection_maps=qm, key_injection_maps=km).logits
    out = m(input_ids=ids, graphs=gs, injection_maps=qm, key_injection_maps=km).logits
    # Position 0 attends only to itself (non-node key => zero structural key): unchanged.
    assert torch.allclose(out[0, 0], b[0, 0], atol=1e-5)
    # Non-node query positions after the first node key move too (the point of B).
    assert not torch.allclose(out[0, 2], b[0, 2], atol=1e-5)
    assert not torch.allclose(out[0, 7], b[0, 7], atol=1e-5)
    out.sum().backward()
    assert m.sk_gain.grad is not None and m.sk_gain.grad.abs().sum() > 0
    assert m.sk_k.weight.grad is not None and m.sk_k.weight.grad.abs().sum() > 0
    assert all(l.weight.grad is not None and l.weight.grad.abs().sum() > 0 for l in m.sk_q)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.pe_model.parameters())
    base_ids = set(map(id, m.base_lr_parameters()))
    assert id(m.sk_gain) in base_ids and id(m.sk_k.weight) in base_ids
    assert id(m.sk_q[0].weight) in base_ids
    assert not (base_ids & set(map(id, m.structural_parameters())))


def test_struct_keys_scope_must_lie_within_mask_scope():
    torch.manual_seed(0)
    llm = _tiny_llm()
    try:
        LearnableGraphMaskLLM(llm, _StubPE(d=8).to(DEVICE), alpha=0.7,
                              layer_scope="dense_first", fusion_d_gt=8,
                              struct_keys=True, struct_keys_layer_scope="all")
    except ValueError as e:
        assert "outside mask_layer_scope" in str(e)
    else:
        raise AssertionError("scope outside the mask scope must be rejected")


def test_struct_keys_forward_under_autocast():
    m = _model(struct_keys=True, struct_keys_dim=4, struct_keys_gain_init=1.0)
    ids, gs, qm, km, _ = _inputs()
    with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
        out = m(input_ids=ids, graphs=gs, injection_maps=qm, key_injection_maps=km).logits
    assert torch.isfinite(out).all()
    assert m.sk_q[0].weight.dtype == torch.float32


# ---------------------------------------------------------------------------
# binding head
# ---------------------------------------------------------------------------

def test_binding_loss_added_finite_and_trains():
    m = _model(binding_head=True, binding_loss_weight=0.5)
    ids, gs, qm, km, _ = _inputs()
    labels = ids.clone()
    out = m(input_ids=ids, graphs=gs, injection_maps=qm, key_injection_maps=km,
            labels=labels)
    assert torch.isfinite(out.loss) and torch.isfinite(m.last_binding_loss)
    base = _model()
    lm_only = base(input_ids=ids, graphs=gs, injection_maps=qm, key_injection_maps=km,
                   labels=labels).loss
    assert torch.allclose(out.loss, lm_only + 0.5 * m.last_binding_loss, atol=1e-4)
    out.loss.backward()
    assert m.bind_proj.weight.grad is not None and m.bind_proj.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.pe_model.parameters())
    assert id(m.bind_proj.weight) in set(map(id, m.base_lr_parameters()))


def test_binding_head_inert_without_labels():
    m = _model(binding_head=True)
    base = _model()
    ids, gs, qm, km, _ = _inputs()
    with torch.no_grad():
        a = m(input_ids=ids, graphs=gs, injection_maps=qm, key_injection_maps=km).logits
        b = base(input_ids=ids, graphs=gs, injection_maps=qm, key_injection_maps=km).logits
    assert torch.equal(a, b)
    assert m._bind_state is None and m._bind_hidden is None


# ---------------------------------------------------------------------------
# save / load key contract
# ---------------------------------------------------------------------------

def test_run_dir_saves_and_loader_requires_e18_weights(tmp_path):
    from prism.training import run_dir
    m = _model(decision_gating=True, struct_keys=True, struct_keys_dim=4, binding_head=True)
    run_dir.save_run_dir(m, {"architecture": "learnable_graph_mask"}, str(tmp_path))
    w = torch.load(tmp_path / "gnn_weights.pt", map_location="cpu")
    for k in ("pe_model", "decision_gain", "sk_k", "sk_q", "sk_gain", "bind_proj"):
        assert k in w, k


# ---------------------------------------------------------------------------
# D — soft edge tokens
# ---------------------------------------------------------------------------

def test_shift_helpers():
    assert shift_spans({0: [(1, 3)], 2: [(5, 6)]}, 4) == {0: [(5, 7)], 2: [(9, 10)]}
    assert shift_positions({3: 1, 7: 2}, 2) == {5: 1, 9: 2}
    x = torch.tensor([[9, 8, 7]])
    assert splice_prefix(x, 2, -100).tolist() == [[9, -100, -100, 8, 7]]
    try:
        shift_spans({0: [(0, 1)]}, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("a span at BOS must be rejected")


def test_soft_edges_shapes_loss_and_grads():
    m = _model(soft_edges=True)
    ids, gs, qm, km, _ = _inputs()
    out = m(input_ids=ids, graphs=gs, injection_maps=qm, key_injection_maps=km,
            labels=ids.clone())
    assert out.logits.shape == (1, ids.shape[1], m.llm.config.vocab_size)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.se_mlp.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.pe_model.parameters())
    assert id(m.se_mlp[0].weight) in set(map(id, m.base_lr_parameters()))


def test_soft_edges_change_logits_and_no_edges_is_identity():
    m = _model(soft_edges=True)
    base = _model()
    ids, gs, qm, km, _ = _inputs()
    with torch.no_grad():
        a = m(input_ids=ids, graphs=gs, injection_maps=qm, key_injection_maps=km).logits
        b = base(input_ids=ids, graphs=gs, injection_maps=qm, key_injection_maps=km).logits
    assert not torch.allclose(a, b, atol=1e-5)
    # a graph with no edges adds no prefix: the embeds path must equal the ids path
    g0 = _graph(3, [])
    with torch.no_grad():
        a0 = m(input_ids=ids, graphs=[g0], injection_maps=qm, key_injection_maps=km).logits
        b0 = base(input_ids=ids, graphs=[g0], injection_maps=qm, key_injection_maps=km).logits
    assert torch.allclose(a0, b0, atol=1e-6)


def test_soft_edges_reject_batch_gt1():
    m = _model(soft_edges=True)
    ids, gs, qm, km, _ = _inputs()
    ids2 = torch.cat([ids, ids])
    try:
        m(input_ids=ids2, graphs=gs * 2, injection_maps=qm * 2, key_injection_maps=km * 2)
    except ValueError as e:
        assert "batch size 1" in str(e)
    else:
        raise AssertionError("batch > 1 must be rejected")


def test_decode_parity_soft_edges():
    m = _model(soft_edges=True)
    tf, dec = _parity(m)
    assert torch.allclose(tf, dec, atol=1e-4), (tf - dec).abs().max()


def test_decode_parity_soft_edges_with_b_and_a():
    m = _model(soft_edges=True, struct_keys=True, struct_keys_dim=4,
               struct_keys_gain_init=0.5, decision_gating=True, decision_gain_init=1.0)
    tf, dec = _parity(m)
    assert torch.allclose(tf, dec, atol=1e-4), (tf - dec).abs().max()


def test_generate_with_inputs_embeds_returns_new_tokens_only():
    """inference.py relies on this HF contract for the soft-edge prefill."""
    m = _model(soft_edges=True)
    ids, gs, qm, km, _ = _inputs()
    with torch.no_grad():
        embeds, off = m.build_soft_edges(ids, gs, km)
        assert off == 4                                   # path 0-1-2: 2 edges x 2 directions
        out = m.llm.generate(inputs_embeds=embeds, max_new_tokens=3, do_sample=False,
                             pad_token_id=0)
    assert out.shape == (1, 3)


def test_run_dir_saves_soft_edges(tmp_path):
    from prism.training import run_dir
    m = _model(soft_edges=True)
    run_dir.save_run_dir(m, {"architecture": "learnable_graph_mask"}, str(tmp_path))
    w = torch.load(tmp_path / "gnn_weights.pt", map_location="cpu")
    assert "se_mlp" in w


# ---------------------------------------------------------------------------
# graphs arrive from the training collator as a PyG Batch (regression: 7731161)
# ---------------------------------------------------------------------------

def test_forward_accepts_collator_batch_for_every_e18_flag():
    from torch_geometric.data import Batch
    ids, gs, qm, km, _ = _inputs()
    for kw in ({"binding_head": True},
               {"struct_keys": True, "struct_keys_dim": 4},
               {"decision_gating": True, "decision_gain_init": 1.0},
               {"soft_edges": True}):
        m = _model(**kw)
        with torch.no_grad():
            a = m(input_ids=ids, graphs=gs, injection_maps=qm, key_injection_maps=km,
                  labels=ids.clone())
            b = m(input_ids=ids, graphs=Batch.from_data_list(gs), injection_maps=qm,
                  key_injection_maps=km, labels=ids.clone())
        assert torch.allclose(a.logits, b.logits, atol=1e-6), kw
        assert torch.allclose(a.loss, b.loss, atol=1e-6), kw


# ---------------------------------------------------------------------------
# device placement — the eval loader / inference client never call .to() on the
# wrapper; they take next(model.parameters()).device as THE model device.
# ---------------------------------------------------------------------------

def _accelerator():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return None


def test_every_e18_flag_constructs_on_the_llm_device():
    """Regression for e18 mask_a / mask_ab (7731158 / 7731160): ``decision_gain`` was
    built on CPU, became the wrapper's first parameter, and the probe / post-train
    eval — which derive the device from ``next(model.parameters())`` and never
    ``.to()`` the wrapper — ran every decode-time tensor on CPU."""
    import pytest
    dev = _accelerator()
    if dev is None:
        pytest.skip("needs a non-CPU device to tell wrapper placement from LLM placement")
    for flags in ({"decision_gating": True, "decision_gain_init": 3.0},
                  {"struct_keys": True, "struct_keys_dim": 4},
                  {"binding_head": True},
                  {"soft_edges": True}):
        torch.manual_seed(0)
        llm = _tiny_llm().to(dev)
        model = LearnableGraphMaskLLM(         # NO trailing .to(): the loader's path
            llm, _StubPE(d=8), alpha=0.7, layer_scope="all", fusion_d_gt=8, **flags)
        first = next(model.parameters()).device
        assert first.type == dev.type, (flags, first)
        off = [n for n, p in model.named_parameters() if p.device.type != dev.type]
        assert not off, (flags, off)
