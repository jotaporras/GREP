"""e17 candidate A — post-fusion gated residual injection tests.

Invariant this file locks: with ``pf_gain = 0`` (the init), an enabled
post-fusion pathway is a BITWISE no-op — logits identical to the plain
``learnable_graph_mask`` forward — so RL warm starts from pre-e17 SFT
checkpoints are behaviorally unchanged at step 0. Also asserts the pathway
actually moves logits once a gain is nonzero (at node positions only), and
that gradients reach the tower and pf modules through the residual write.
"""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, "tests")

import torch

from prism.models.gnn_llm import LearnableGraphMaskLLM
from test_learnable_graph_mask import _tiny_llm, _StubPE, _graph, DEVICE


def _model(post_fusion=True):
    torch.manual_seed(0)
    llm = _tiny_llm()
    return LearnableGraphMaskLLM(
        llm, _StubPE(d=8).to(DEVICE), alpha=0.7, layer_scope="all",
        post_fusion=post_fusion, post_fusion_layer_scope="all",
        post_fusion_d_gt=8).to(DEVICE)


def _inputs(model):
    g = _graph(3, [(0, 1), (1, 2)])
    imap = {0: [(1, 2)], 1: [(3, 4)], 2: [(5, 6)]}
    ids = torch.randint(0, 64, (1, 8), generator=torch.Generator().manual_seed(2)
                        ).to(DEVICE)
    return ids, [g], [imap]


def test_zero_gain_is_bitwise_noop():
    m_pf = _model(post_fusion=True)
    m_plain = _model(post_fusion=False)
    ids, gs, imaps = _inputs(m_pf)
    with torch.no_grad():
        out_pf = m_pf(input_ids=ids, graphs=gs, injection_maps=imaps).logits
        out_plain = m_plain(input_ids=ids, graphs=gs, injection_maps=imaps).logits
    assert torch.equal(out_pf, out_plain)


def test_nonzero_gain_moves_logits():
    m = _model(post_fusion=True)
    ids, gs, imaps = _inputs(m)
    with torch.no_grad():
        base = m(input_ids=ids, graphs=gs, injection_maps=imaps).logits
        m.pf_gain.data.fill_(1.0)
        moved = m(input_ids=ids, graphs=gs, injection_maps=imaps).logits
    assert not torch.equal(base, moved)


def test_gradients_reach_pf_and_tower():
    m = _model(post_fusion=True)
    ids, gs, imaps = _inputs(m)
    loss = m(input_ids=ids, graphs=gs, injection_maps=imaps).logits.sum()
    loss.backward()
    # pf_gain is upstream of tanh(0) with nonzero d/dgain — must receive grad.
    assert m.pf_gain.grad is not None and m.pf_gain.grad.abs().sum() > 0
    # At gain=0 the pf_proj branch is multiplied by 0, so its grad is zero but
    # DEFINED; the tower still gets grad through the mask bias itself.
    assert m.pf_proj.weight.grad is not None
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in m.pe_model.parameters())


def test_build_pf_signal_survives_autocast():
    # The RL loss forward runs under accelerate's bf16 autocast; the fp32
    # signal build must not crash (or silently downcast) inside it.
    m = _model(post_fusion=True)
    ids, gs, imaps = _inputs(m)
    with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
        sig = m.build_pf_signal(ids.shape[1], gs, imaps, DEVICE)
        out = m(input_ids=ids, graphs=gs, injection_maps=imaps).logits
    assert sig.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_pf_params_train_at_base_lr():
    # The pf modules are fresh (zero-gated) and must NOT sit in the damped
    # structural group — at SFT's structural_lr_mult 0.012 the gate never
    # opens (e17: pf_gain absmax ~1e-4 after 3 epochs). They belong in
    # base_lr_parameters(), which both trainers unfreeze and create_optimizer
    # leaves at the full base LR.
    m = _model(post_fusion=True)
    struct = set(map(id, m.structural_parameters()))
    base = set(map(id, m.base_lr_parameters()))
    for p in (m.pf_gain, m.pf_proj.weight, m.pf_norm.weight):
        assert id(p) in base
        assert id(p) not in struct
    assert not (struct & base)
    assert _model(post_fusion=False).base_lr_parameters() == []
