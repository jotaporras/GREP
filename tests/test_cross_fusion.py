"""e17 candidate C — post-LLM gated cross-attention tests.

Invariants this file locks: with ``xf_gain = 0`` (the init), an enabled
cross-fusion block is a BITWISE no-op; a nonzero gain moves logits at EVERY
position (all tokens query the graph, unlike A's node-positions-only write);
gradients reach the gain (live at init) and, once open, the whole block and
the tower; padding masks exclude phantom nodes; and cached decode steps work
against the same static K/V bank.
"""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, "tests")

import torch

from prism.models.gnn_llm import LearnableGraphMaskLLM
from test_learnable_graph_mask import _tiny_llm, _StubPE, _graph, DEVICE


def _model(cross_fusion=True):
    torch.manual_seed(0)
    llm = _tiny_llm()
    return LearnableGraphMaskLLM(
        llm, _StubPE(d=8).to(DEVICE), alpha=0.7, layer_scope="all",
        cross_fusion=cross_fusion, cross_fusion_heads=2,
        cross_fusion_dim=8, fusion_d_gt=8).to(DEVICE)


def _inputs(model):
    g = _graph(3, [(0, 1), (1, 2)])
    imap = {0: [(1, 2)], 1: [(3, 4)], 2: [(5, 6)]}
    ids = torch.randint(0, 64, (1, 8), generator=torch.Generator().manual_seed(2)
                        ).to(DEVICE)
    return ids, [g], [imap]


def test_zero_gain_is_bitwise_noop():
    m_xf = _model(cross_fusion=True)
    m_plain = _model(cross_fusion=False)
    ids, gs, imaps = _inputs(m_xf)
    with torch.no_grad():
        out_xf = m_xf(input_ids=ids, graphs=gs, injection_maps=imaps).logits
        out_plain = m_plain(input_ids=ids, graphs=gs, injection_maps=imaps).logits
    assert torch.equal(out_xf, out_plain)


def test_nonzero_gain_moves_all_positions():
    m = _model(cross_fusion=True)
    ids, gs, imaps = _inputs(m)
    with torch.no_grad():
        base = m(input_ids=ids, graphs=gs, injection_maps=imaps).logits
        m.xf_gain.data.fill_(1.0)
        moved = m(input_ids=ids, graphs=gs, injection_maps=imaps).logits
    diff = (moved - base).abs().amax(dim=-1)[0]           # [S]
    # Every position queries the graph — unlike candidate A, non-node
    # positions move too.
    assert (diff > 0).all()


def test_gradients_reach_gain_block_and_tower():
    m = _model(cross_fusion=True)
    ids, gs, imaps = _inputs(m)
    loss = m(input_ids=ids, graphs=gs, injection_maps=imaps).logits.sum()
    loss.backward()
    assert m.xf_gain.grad is not None and m.xf_gain.grad.abs().sum() > 0
    # Open the gain: block weights and the tower (through K/V=Ψ) get grad.
    m2 = _model(cross_fusion=True)
    m2.xf_gain.data.fill_(1.0)
    loss2 = m2(input_ids=ids, graphs=gs, injection_maps=imaps).logits.sum()
    loss2.backward()
    assert m2.xf_q.weight.grad.abs().sum() > 0
    assert m2.xf_k.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in m2.pe_model.parameters())


def test_padding_mask_excludes_phantom_nodes():
    # Two graphs of different node counts in one batch: the padded row must
    # not attend the phantom node slots.
    m = _model(cross_fusion=True)
    g_small = _graph(2, [(0, 1)])
    g_big = _graph(5, [(0, 1), (1, 2), (2, 3), (3, 4)], seed=3)
    kv, mask = m.build_xf_kv([g_small, g_big], DEVICE)
    assert kv.shape[:2] == (2, 5)
    assert mask[0].tolist() == [True, True, False, False, False]
    m.xf_gain.data.fill_(1.0)
    ids = torch.randint(0, 64, (2, 6), generator=torch.Generator().manual_seed(4)
                        ).to(DEVICE)
    with torch.no_grad():
        m._xf_kv = (kv, mask)
        out_pad = m.llm(input_ids=ids).logits
        # Corrupting the PADDED slots of row 0 must not change row 0's logits.
        kv2 = kv.clone()
        kv2[0, 2:] = 1e3
        m._xf_kv = (kv2, mask)
        out_corrupt = m.llm(input_ids=ids).logits
        m._xf_kv = None
    assert torch.equal(out_pad[0], out_corrupt[0])


def test_decode_uses_static_kv():
    m = _model(cross_fusion=True)
    ids, gs, imaps = _inputs(m)
    with torch.no_grad():
        m.xf_gain.data.fill_(1.0)
        m._xf_kv = m.build_xf_kv(gs, DEVICE)
        step = ids[:, :1]
        out_armed = m.llm(input_ids=step).logits
        m._xf_kv = None
        out_plain = m.llm(input_ids=step).logits
    assert not torch.equal(out_armed, out_plain)


def test_xf_params_train_at_base_lr():
    m = _model(cross_fusion=True)
    struct = set(map(id, m.structural_parameters()))
    base = set(map(id, m.base_lr_parameters()))
    for p in (m.xf_gain, m.xf_q.weight, m.xf_k.weight, m.xf_v.weight,
              m.xf_o.weight, m.xf_ln.weight):
        assert id(p) in base
        assert id(p) not in struct
    assert not (struct & base)
    assert _model(cross_fusion=False).base_lr_parameters() == []
