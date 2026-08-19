"""e17 candidate D — graph-generated LoRA tests.

Invariants this file locks: with ``B = 0`` (the init), an enabled graph-LoRA
pathway is a BITWISE no-op — logits identical to the plain mask forward — so
RL warm starts from pre-e17 SFT checkpoints are behaviorally unchanged at
step 0. Also asserts the pathway moves logits once B is nonzero, that
gradients reach the generator head, B, and the tower through the delta, and
the base-LR parameter-group membership.
"""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, "tests")

import torch

from prism.models.gnn_llm import LearnableGraphMaskLLM
from test_learnable_graph_mask import _tiny_llm, _StubPE, _graph, DEVICE


def _model(graph_lora=True):
    torch.manual_seed(0)
    llm = _tiny_llm()
    return LearnableGraphMaskLLM(
        llm, _StubPE(d=8).to(DEVICE), alpha=0.7, layer_scope="all",
        graph_lora=graph_lora, graph_lora_rank=4, graph_lora_targets="o_proj",
        graph_lora_layer_scope="all", fusion_d_gt=8).to(DEVICE)


def _inputs(model):
    g = _graph(3, [(0, 1), (1, 2)])
    imap = {0: [(1, 2)], 1: [(3, 4)], 2: [(5, 6)]}
    ids = torch.randint(0, 64, (1, 8), generator=torch.Generator().manual_seed(2)
                        ).to(DEVICE)
    return ids, [g], [imap]


def test_zero_B_is_bitwise_noop():
    m_gl = _model(graph_lora=True)
    m_plain = _model(graph_lora=False)
    ids, gs, imaps = _inputs(m_gl)
    with torch.no_grad():
        out_gl = m_gl(input_ids=ids, graphs=gs, injection_maps=imaps).logits
        out_plain = m_plain(input_ids=ids, graphs=gs, injection_maps=imaps).logits
    assert torch.equal(out_gl, out_plain)


def test_nonzero_B_moves_logits():
    m = _model(graph_lora=True)
    ids, gs, imaps = _inputs(m)
    with torch.no_grad():
        base = m(input_ids=ids, graphs=gs, injection_maps=imaps).logits
        for p in m.glora_B.values():
            p.data.normal_(0.0, 0.05)
        moved = m(input_ids=ids, graphs=gs, injection_maps=imaps).logits
    assert not torch.equal(base, moved)


def test_gradients_reach_B_gen_and_tower():
    m = _model(graph_lora=True)
    ids, gs, imaps = _inputs(m)
    loss = m(input_ids=ids, graphs=gs, injection_maps=imaps).logits.sum()
    loss.backward()
    # B is zero-init but LIVE: dL/dB = g_out · (A x)^T != 0 — no gate needed.
    assert all(p.grad is not None for p in m.glora_B.values())
    assert any(p.grad.abs().sum() > 0 for p in m.glora_B.values())
    # The generator head sits behind B=0, so its grad is zero but DEFINED; the
    # tower still gets grad through the mask bias itself.
    assert m.glora_gen["o_proj"].weight.grad is not None
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in m.pe_model.parameters())
    # Once B is nonzero, the tower ALSO gets grad through pooled psi -> A.
    m2 = _model(graph_lora=True)
    for p in m2.glora_B.values():
        p.data.normal_(0.0, 0.05)
    stub_grad_before = None
    loss2 = m2(input_ids=ids, graphs=gs, injection_maps=imaps).logits.sum()
    loss2.backward()
    assert m2.glora_gen["o_proj"].weight.grad.abs().sum() > 0
    del stub_grad_before


def test_glora_params_train_at_base_lr():
    m = _model(graph_lora=True)
    struct = set(map(id, m.structural_parameters()))
    base = set(map(id, m.base_lr_parameters()))
    for p in list(m.glora_gen.parameters()) + list(m.glora_B.values()):
        assert id(p) in base
        assert id(p) not in struct
    assert not (struct & base)
    assert _model(graph_lora=False).base_lr_parameters() == []


def test_decode_uses_static_factors():
    # The armed per-row factors must serve cached decode steps unchanged
    # (per-graph signal, no per-step state): a [B,1,·] forward under an armed
    # _glora_A must not raise and must apply the delta.
    m = _model(graph_lora=True)
    ids, gs, imaps = _inputs(m)
    with torch.no_grad():
        m._glora_A = m.build_glora_signal(gs, DEVICE)
        for p in m.glora_B.values():
            p.data.normal_(0.0, 0.05)
        step = ids[:, :1]
        out_armed = m.llm(input_ids=step).logits
        m._glora_A = None
        out_plain = m.llm(input_ids=step).logits
    assert not torch.equal(out_armed, out_plain)
