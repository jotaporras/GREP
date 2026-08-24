"""e19 hop-separated post-fusion tests (docs/2026-08-23 e19 design note).

Invariants locked here, mirroring ``test_post_fusion.py``:

- With ALL gains at 0 an enabled hop-mode pathway ("shift" and "depth") is a
  BITWISE no-op vs the plain mask forward.
- e19 v2 ControlNet-style init: OPEN gates (``post_fusion_gain_init``) with
  ZERO RMSNorm scales are ALSO a bitwise no-op at init — no perturbation tax —
  while the norm scale receives full-strength gradient through the open gate
  (the e17 zero-GATE init killed that gradient; the first e19 fleet's open
  gates + unit norm paid a ~3x loss-floor tax it never recovered from).
- Channels are genuinely SEPARATE: opening different single channels produces
  different logits (per-channel matrices + gates, the experiment's point).
- Hop parameters train at base LR (not in the damped structural group).
- ``GraphTransformer.forward_taps`` has shape [N, L+1, d] and its last channel
  is bitwise the pure-PE ``forward`` output (depth mode's parity contract).
- Depth mode fails loud on a Ψ producer without ``forward_taps``; shift mode
  fails loud when the graph outgrows the codebook.
- ``save_run_dir`` writes ``pf_ch_gain`` (both hop modes) and ``pf_hop_gt``
  (shift only), and the loader's fail-loud key sets match.
"""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, "tests")

import pytest
import torch
from torch import nn

from prism.models import gt as gt_module
from prism.models.gnn_llm import LearnableGraphMaskLLM
from test_learnable_graph_mask import _tiny_llm, _StubPE, _graph, DEVICE


class _StubTapPE(_StubPE):
    """_StubPE plus the ``forward_taps`` contract depth mode needs: channel k is
    a distinct (scaled) copy of Ψ, so per-channel separation is observable."""

    num_layers = 2

    def forward_taps(self, g, permutation=None):
        x = self.forward(g, permutation=permutation)
        return torch.stack([x * float(k + 1)
                            for k in range(self.num_layers + 1)], dim=1)


def _model(hop_mode="none", post_fusion=True, gain_init=0.0, hop_k=3,
           codebook_size=256):
    torch.manual_seed(0)
    llm = _tiny_llm()
    pe = (_StubTapPE(d=8) if hop_mode == "depth" else _StubPE(d=8)).to(DEVICE)
    return LearnableGraphMaskLLM(
        llm, pe, alpha=0.7, layer_scope="all",
        post_fusion=post_fusion, post_fusion_layer_scope="all",
        post_fusion_d_gt=8, post_fusion_hop_mode=hop_mode,
        post_fusion_hop_k=hop_k, post_fusion_gain_init=gain_init,
        post_fusion_codebook_size=codebook_size,
        post_fusion_hop_gt_layers=2, post_fusion_hop_gt_heads=2,
        post_fusion_hop_gt_k=1).to(DEVICE)


def _inputs():
    g = _graph(3, [(0, 1), (1, 2)])
    imap = {0: [(1, 2)], 1: [(3, 4)], 2: [(5, 6)]}
    ids = torch.randint(0, 64, (1, 8), generator=torch.Generator().manual_seed(2)
                        ).to(DEVICE)
    return ids, [g], [imap]


@pytest.mark.parametrize("hop_mode", ["shift", "depth"])
def test_zero_gain_is_bitwise_noop(hop_mode):
    m_pf = _model(hop_mode=hop_mode)
    m_plain = _model(hop_mode=hop_mode, post_fusion=False)
    ids, gs, imaps = _inputs()
    with torch.no_grad():
        out_pf = m_pf(input_ids=ids, graphs=gs, injection_maps=imaps).logits
        out_plain = m_plain(input_ids=ids, graphs=gs, injection_maps=imaps).logits
    assert torch.equal(out_pf, out_plain)


@pytest.mark.parametrize("hop_mode", ["shift", "depth"])
def test_open_gates_are_bitwise_noop_at_init(hop_mode):
    # e19 v2 (ControlNet-style init): gates OPEN at 1.0 but the RMSNorm scales
    # are zero-init, so the model is bitwise the plain mask at step 0 — no
    # perturbation tax (the first fleet stalled ~3x above the mask_a floor).
    m = _model(hop_mode=hop_mode, gain_init=1.0)
    assert torch.all(m.pf_gain == 1.0) and torch.all(m.pf_ch_gain == 1.0)
    for norm in m.pf_norm:
        assert torch.all(norm.weight == 0.0)
    m_plain = _model(hop_mode=hop_mode, post_fusion=False)
    ids, gs, imaps = _inputs()
    with torch.no_grad():
        out_pf = m(input_ids=ids, graphs=gs, injection_maps=imaps).logits
        out_plain = m_plain(input_ids=ids, graphs=gs, injection_maps=imaps).logits
    assert torch.equal(out_pf, out_plain)


@pytest.mark.parametrize("hop_mode", ["shift", "depth"])
def test_nonzero_norm_scale_moves_logits(hop_mode):
    m = _model(hop_mode=hop_mode, gain_init=1.0)
    ids, gs, imaps = _inputs()
    with torch.no_grad():
        base = m(input_ids=ids, graphs=gs, injection_maps=imaps).logits
        for norm in m.pf_norm:
            norm.weight.fill_(1.0)
        moved = m(input_ids=ids, graphs=gs, injection_maps=imaps).logits
    assert not torch.equal(base, moved)


@pytest.mark.parametrize("hop_mode", ["shift", "depth"])
def test_channels_are_separate(hop_mode):
    # Opening ONLY channel j must produce a different output per j — if the
    # channels collapsed into one aggregate (the e17 failure hypothesis), the
    # single-channel outputs would coincide.
    m = _model(hop_mode=hop_mode)
    m.pf_gain.data.fill_(1.0)
    for norm in m.pf_norm:
        norm.weight.data.fill_(1.0)
    ids, gs, imaps = _inputs()
    outs = []
    with torch.no_grad():
        for j in range(m.pf_ch_gain.numel()):
            m.pf_ch_gain.data.zero_()
            m.pf_ch_gain.data[j] = 1.0
            outs.append(m(input_ids=ids, graphs=gs,
                          injection_maps=imaps).logits)
    for a in range(len(outs)):
        for b in range(a + 1, len(outs)):
            assert not torch.equal(outs[a], outs[b]), (a, b)


@pytest.mark.parametrize("hop_mode", ["shift", "depth"])
def test_gradients_reach_norm_scale_at_init(hop_mode):
    # The ControlNet property: at init (zero norm scale, open gates) the ONLY
    # live gradient is the norm scale's — full strength through the open gate.
    # Everything upstream (W_k, tower, gains) is defined-but-zero until the
    # scale moves; e17's zero-GATE init killed this gradient too.
    m = _model(hop_mode=hop_mode, gain_init=1.0)
    ids, gs, imaps = _inputs()
    loss = m(input_ids=ids, graphs=gs, injection_maps=imaps).logits.sum()
    loss.backward()
    for norm in m.pf_norm:
        assert norm.weight.grad is not None and norm.weight.grad.abs().sum() > 0
    assert m.pf_gain.grad is not None
    assert m.pf_ch_gain.grad is not None
    for proj in m.pf_proj:
        assert proj.weight.grad is not None


@pytest.mark.parametrize("hop_mode", ["shift", "depth"])
def test_gradients_reach_channels_once_scale_opens(hop_mode):
    m = _model(hop_mode=hop_mode, gain_init=1.0)
    for norm in m.pf_norm:
        norm.weight.data.fill_(1.0)
    ids, gs, imaps = _inputs()
    loss = m(input_ids=ids, graphs=gs, injection_maps=imaps).logits.sum()
    loss.backward()
    assert m.pf_gain.grad is not None and m.pf_gain.grad.abs().sum() > 0
    assert (m.pf_ch_gain.grad is not None
            and m.pf_ch_gain.grad.abs().sum() > 0)
    for proj in m.pf_proj:
        assert (proj.weight.grad is not None
                and proj.weight.grad.abs().sum() > 0)
    if hop_mode == "shift":
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in m.pf_hop_gt.parameters())


@pytest.mark.parametrize("hop_mode", ["shift", "depth"])
def test_hop_params_train_at_base_lr(hop_mode):
    m = _model(hop_mode=hop_mode)
    struct = set(map(id, m.structural_parameters()))
    base = set(map(id, m.base_lr_parameters()))
    probes = [m.pf_gain, m.pf_ch_gain, m.pf_proj[0].weight, m.pf_norm[0].weight]
    if hop_mode == "shift":
        probes.append(m.pf_hop_gt.codebook.weight)
    for p in probes:
        assert id(p) in base
        assert id(p) not in struct
    assert not (struct & base)


def test_forward_taps_shape_and_last_tap_parity():
    torch.manual_seed(0)
    gt = gt_module.GraphTransformer(
        num_layers=2, pe_hidden_channels=8, pe_num_layers=2, d_model=16,
        heads=2, num_samples=8, dropout=0.0, k_pe=2, k_gt=1,
        node_feature_dim=None, fixed_seed_mode=True, fixed_seed_value=11
    ).to(DEVICE).eval()
    g = _graph(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    with torch.no_grad():
        taps = gt.forward_taps(g)
        full = gt(g)
    assert taps.shape == (6, gt.num_layers + 1, 16)
    assert torch.equal(taps[:, -1, :], full)


def test_depth_mode_requires_forward_taps():
    torch.manual_seed(0)
    with pytest.raises(ValueError, match="forward_taps"):
        LearnableGraphMaskLLM(
            _tiny_llm(), _StubPE(d=8).to(DEVICE), alpha=0.7, layer_scope="all",
            post_fusion=True, post_fusion_layer_scope="all",
            post_fusion_d_gt=8, post_fusion_hop_mode="depth")


def test_shift_mode_codebook_overflow_fails_loud():
    m = _model(hop_mode="shift", codebook_size=4)
    ids, gs, imaps = _inputs()          # 3 nodes: fits
    with torch.no_grad():
        m(input_ids=ids, graphs=gs, injection_maps=imaps)
    big = _graph(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    with pytest.raises(ValueError, match="codebook"):
        m.pf_psi(big)


def test_hop_and_pointer_fusion_are_mutually_exclusive():
    m = _model(hop_mode="shift")
    with pytest.raises(ValueError, match="pointer_fusion"):
        m.enable_pointer_fusion(d_gt=8)


@pytest.mark.parametrize("hop_mode", ["shift", "depth"])
def test_run_dir_saves_and_loader_requires_hop_weights(tmp_path, hop_mode):
    from prism.training.run_dir import save_run_dir
    m = _model(hop_mode=hop_mode, gain_init=1.0)
    cfg = {"architecture": "learnable_graph_mask", "post_fusion": True,
           "post_fusion_hop_mode": hop_mode}
    save_run_dir(m, cfg, str(tmp_path))
    weights = torch.load(tmp_path / "gnn_weights.pt", map_location="cpu")
    assert "pf_ch_gain" in weights
    assert torch.equal(weights["pf_ch_gain"], m.pf_ch_gain.data.cpu())
    if hop_mode == "shift":
        assert "pf_hop_gt" in weights
        assert set(weights["pf_hop_gt"]) == set(
            k for k, _ in m.pf_hop_gt.state_dict().items())
    else:
        assert "pf_hop_gt" not in weights
