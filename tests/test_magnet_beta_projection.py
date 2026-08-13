r"""Tests for the R-PEARL β projection and the removal of MagNet's per-node LayerNorm.

Both changes exist to make PEARL (arXiv:2502.01122) Assumption 4.2 —
``||H(S)|| <= beta``, with ``C_sigma = 1`` and ``beta = 1/F`` in Thm 4.3 / Cor 4.6 —
an enforced invariant rather than an assumed one. What these pin down:

  a) After a projected optimizer step, ``sum_k ||H_k||_2 <= 1/F`` for EVERY
     MagChebConv, to 1e-6.
  b) The projection scales the MagChebConv PRE-ACTIVATION output by exactly the
     scalar it divided by (rel err < 1e-5) — the layer bias is scaled with the taps,
     so the whole affine map, not just its linear part, is homogeneous in s.
     Deliberately tested at the CONV level: modrelu has a FIXED deadzone radius
     softplus(b) and is therefore not positively homogeneous, so the end-to-end
     MagNet output does NOT scale by s. That interaction is the reason (b) is not an
     end-to-end test.
  c) ``enforce_beta_bound=False`` leaves every weight bit-identical.
  d) Conjugation equivariance: reversing every edge conjugates the pre-unwind
     representation. Holds with hidden_norm='none' and 'global_rms'; the pre-fix
     per-node LayerNorm path breaks it and is kept here as a strict xfail so the fix
     is proven to do something rather than merely asserted to.

Runs on CUDA when available, CPU otherwise — tiny modules, no training. MPS is
excluded: it has no complex scatter-add kernel, so MagNet cannot forward there (see
``device()``).

Run:  uv run --with pytest -m pytest tests/test_magnet_beta_projection.py -q
"""
import sys

sys.path.insert(0, "src")

import pytest
import torch
from torch import nn
from torch_geometric.data import Data

from prism.models import magnet
from prism.models.beta_projection import (beta_slack, project_beta_,
                                          register_beta_projection)
from prism.models.magnet import MagChebConv, MagNet, modrelu
from prism.models.r_pearl import RandomGNNPositionalEncodings

# Small enough to be instant, wide enough that 1/F is a real constraint (F=8 -> 0.125).
HIDDEN, LAYERS, K = 8, 4, 3


def device():
    """CUDA when present, else CPU. MPS is deliberately NOT used.

    PyG aggregation reaches ``Tensor.scatter_add_``, which raises
    ``scatter(): Yet not supported for complex`` on MPS — a hard error that
    ``PYTORCH_ENABLE_MPS_FALLBACK=1`` does not rescue, so MagNet cannot forward
    there. The ``MagChebConv.aggregate`` split that fixes it is stashed in
    ``.magnet_mps_complex_scatter.patch``; restore MPS here when it is re-applied.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def directed_graph(dev, num_nodes: int = 7, seed: int = 0) -> Data:
    """A genuinely ASYMMETRIC edge_index — a cycle plus two chords, one direction only.

    MagNet symmetrizes A and puts the asymmetry in the phase, so a graph that already
    carries both directions would make Theta = 0 and turn test (d) into a tautology.
    """
    g = torch.Generator().manual_seed(seed)
    src = torch.arange(num_nodes)
    dst = (src + 1) % num_nodes
    edge_index = torch.stack([torch.cat([src, torch.tensor([0, 2])]),
                              torch.cat([dst, torch.tensor([3, 5])])])
    x = torch.randn(num_nodes, 1, generator=g)
    return Data(x=x.to(dev), edge_index=edge_index.to(dev))


def make_magnet(dev, hidden_norm: str = "none", seed: int = 0, in_channels: int = 1):
    torch.manual_seed(seed)
    return MagNet(in_channels, HIDDEN, LAYERS, skip_connection=True, dropout=0.0,
                  k=K, hidden_norm=hidden_norm).to(dev).eval()


def tap_norm_sum(conv: MagChebConv) -> float:
    """sum_k ||H_k||_2 — the quantity Assumption 4.2 bounds, measured not assumed.

    Computed here from the raw weights (on CPU, since MPS has no SVD kernel) rather
    than through ``beta_slack``, so the assertion is independent of the code it checks.
    """
    return float(sum(torch.linalg.matrix_norm(lin.weight.detach().cpu(), ord=2)
                     for lin in conv.lins))


# --------------------------------------------------------------------------------
# (a) the invariant holds after a projected step
# --------------------------------------------------------------------------------
def test_beta_bound_holds_after_projected_step():
    dev = device()
    model = make_magnet(dev)
    graph = directed_graph(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-1)
    register_beta_projection(opt, model, enabled=True)

    # A real step: a large LR is what would push the taps out of the constraint set.
    for _ in range(3):
        opt.zero_grad()
        model(graph).square().mean().backward()
        opt.step()

    for name, conv in model.named_modules():
        if not isinstance(conv, MagChebConv):
            continue
        budget = 1.0 / conv.out_channels
        assert tap_norm_sum(conv) <= budget + 1e-6, (
            f"{name}: sum_k ||H_k||_2 = {tap_norm_sum(conv):.6g} > 1/F = {budget:.6g}")


def test_beta_bound_violated_without_the_hook():
    """The projection is load-bearing: glorot init alone blows the 1/F budget."""
    model = make_magnet(device())
    violated = [tap_norm_sum(c) > 1.0 / c.out_channels
                for c in model.modules() if isinstance(c, MagChebConv)]
    assert all(violated), "1/F is not a binding constraint at init; test (a) proves nothing"


def test_slack_is_the_measured_overshoot():
    """beta_slack is sum_k ||H_k||_2 / beta, and lands exactly on 1 after projection."""
    model = make_magnet(device())
    conv = next(c for c in model.modules() if isinstance(c, MagChebConv))
    expected = tap_norm_sum(conv) * conv.out_channels
    assert float(beta_slack(conv).detach()) == pytest.approx(expected, rel=1e-6)

    project_beta_(model)
    assert float(beta_slack(conv).detach()) == pytest.approx(1.0, rel=1e-5)


# --------------------------------------------------------------------------------
# (b) the projection scales the conv pre-activation by exactly s
# --------------------------------------------------------------------------------
def test_projection_scales_conv_preactivation_by_the_scalar():
    dev = device()
    torch.manual_seed(3)
    conv = MagChebConv(4, HIDDEN, K).to(dev).eval()
    graph = directed_graph(dev)
    x = torch.randn(graph.num_nodes, 4, device=dev) + 0j

    with torch.no_grad():
        before = conv(x, graph.edge_index)
    s = float(beta_slack(conv).detach())
    assert s > 1.0, "nothing to project; the test would pass vacuously"

    project_beta_(conv)
    with torch.no_grad():
        after = conv(x, graph.edge_index)

    rel = ((after - before / s).abs().max() / (before / s).abs().max()).item()
    assert rel < 1e-5, f"pre-activation is not exactly 1/s of the original (rel {rel:.3g})"


def test_bias_is_scaled_with_the_taps():
    """Scaling the taps but not the bias would leave the map only AFFINELY related."""
    dev = device()
    torch.manual_seed(4)
    conv = MagChebConv(4, HIDDEN, K).to(dev).eval()
    with torch.no_grad():
        conv.bias.normal_()
    bias_before = conv.bias.detach().clone()
    s = float(beta_slack(conv).detach())

    project_beta_(conv)
    assert torch.allclose(conv.bias, bias_before / s, rtol=1e-5)


def test_projection_is_one_global_scalar_not_per_row():
    """Per-row/per-channel rescaling is forbidden: it leaves the sum_k h_k S^k family."""
    dev = device()
    torch.manual_seed(5)
    conv = MagChebConv(4, HIDDEN, K).to(dev).eval()
    before = [lin.weight.detach().clone() for lin in conv.lins]
    s = float(beta_slack(conv).detach())

    project_beta_(conv)
    for w_before, lin in zip(before, conv.lins):
        ratio = w_before / lin.weight
        assert ratio.std() < 1e-4, "the rescale varies across entries; it is not global"
        assert ratio.mean().item() == pytest.approx(s, rel=1e-4)


def test_charge_and_modrelu_bias_are_untouched():
    """r_logit parameterizes S, the modReLU bias sets a deadzone radius; neither is H."""
    model = make_magnet(device())
    r_before = [c.r_logit.detach().clone() for c in model.convs]
    b_before = [b.detach().clone() for b in model.biases]

    project_beta_(model)
    assert all(torch.equal(a, c.r_logit) for a, c in zip(r_before, model.convs))
    assert all(torch.equal(a, b) for a, b in zip(b_before, model.biases))


def test_unwind_linear_is_outside_the_projection_scope():
    """The readout is not a graph filter, so Assumption 4.2 does not reach it."""
    model = make_magnet(device())
    unwind_before = model.unwind.weight.detach().clone()
    project_beta_(model)
    assert torch.equal(unwind_before, model.unwind.weight)


def test_projection_is_a_noop_on_the_undirected_backbone():
    """TAGConv holds no MagChebConv, so the hook must find nothing to scale."""
    pe = RandomGNNPositionalEncodings(pe_hidden_channels=HIDDEN, pe_num_layers=2,
                                      d_model=4, num_samples=2, dropout=0.0, k=2,
                                      directed=False)
    before = {n: p.detach().clone() for n, p in pe.named_parameters()}
    assert project_beta_(pe) == {}
    assert all(torch.equal(before[n], p) for n, p in pe.named_parameters())


# --------------------------------------------------------------------------------
# (c) flag off -> bit-identical
# --------------------------------------------------------------------------------
def test_flag_off_leaves_weights_bit_identical():
    dev = device()
    model = make_magnet(dev)
    graph = directed_graph(dev)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    handle = register_beta_projection(opt, model, enabled=False)
    assert handle is None, "a disabled projection must not register a step hook"

    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt.zero_grad()
    model(graph).square().mean().backward()
    opt.step()

    for n, p in model.named_parameters():
        assert torch.equal(before[n], p), f"{n} moved with enforce_beta_bound=False"


# --------------------------------------------------------------------------------
# (d) conjugation equivariance
# --------------------------------------------------------------------------------
def pre_unwind(model: MagNet, graph: Data) -> torch.Tensor:
    """MagNet.forward stopping before the unwind Linear, which discards the phase.

    ``unwind([Re, Im])`` maps x and conj(x) to different reals by construction, so
    the conjugation property is only observable upstream of it.
    """
    x = x_prev = graph.x + 0j
    for i, conv in enumerate(model.convs[:-1]):
        x = conv(x_prev, graph.edge_index)
        if model.hidden_norm == "global_rms" and i < len(model.convs) - 2:
            x = x / magnet.global_rms(x)
        x = modrelu(x, model.biases[i])
        if model.skip_connection and i > 0:
            x = x + x_prev
        x_prev = x
    return model.convs[-1](x, graph.edge_index)


def reversed_graph(graph: Data) -> Data:
    return Data(x=graph.x, edge_index=graph.edge_index.flip(0))


@pytest.mark.parametrize("hidden_norm", ["none", "global_rms"])
def test_edge_reversal_conjugates_the_representation(hidden_norm):
    dev = device()
    model = make_magnet(dev, hidden_norm=hidden_norm)
    graph = directed_graph(dev)

    with torch.no_grad():
        fwd = pre_unwind(model, graph)
        rev = pre_unwind(model, reversed_graph(graph))

    err = (rev - fwd.conj()).abs().max().item()
    assert err < 1e-5, f"hidden_norm={hidden_norm}: not conjugation-equivariant ({err:.3g})"
    assert fwd.imag.abs().max().item() > 1e-4, "phase is identically zero; test is vacuous"


class LayerNormMagNet(MagNet):
    """The PRE-FIX path: nn.LayerNorm applied separately to x.real and x.imag.

    Kept in the test file only. It is a per-node normalization across channels — not
    1-Lipschitz, and its learnable affine shift is applied to Re and Im independently,
    so it does not commute with conjugation. Reconstructed here rather than left
    reachable from config so the violating path cannot be selected by a real run.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norms = nn.ModuleList(
            [nn.LayerNorm(HIDDEN) for _ in range(len(self.convs) - 2)])
        for norm in self.norms:  # a nonzero shift is what breaks equivariance
            nn.init.normal_(norm.bias, std=0.5)


def pre_unwind_layernorm(model: LayerNormMagNet, graph: Data) -> torch.Tensor:
    x = x_prev = graph.x + 0j
    for i, conv in enumerate(model.convs[:-1]):
        x = conv(x_prev, graph.edge_index)
        if i < len(model.norms):
            x = torch.complex(model.norms[i](x.real), model.norms[i](x.imag))
        x = modrelu(x, model.biases[i])
        if model.skip_connection and i > 0:
            x = x + x_prev
        x_prev = x
    return model.convs[-1](x, graph.edge_index)


@pytest.mark.xfail(strict=True, reason="per-node LayerNorm on Re/Im is not "
                                       "conjugation-equivariant; this is the bug the "
                                       "hidden_norm gate removes")
def test_layernorm_path_breaks_conjugation_equivariance():
    dev = device()
    torch.manual_seed(0)
    model = LayerNormMagNet(1, HIDDEN, LAYERS, skip_connection=True, dropout=0.0,
                            k=K).to(dev).eval()
    graph = directed_graph(dev)

    with torch.no_grad():
        fwd = pre_unwind_layernorm(model, graph)
        rev = pre_unwind_layernorm(model, reversed_graph(graph))

    assert (rev - fwd.conj()).abs().max().item() < 1e-5


# --------------------------------------------------------------------------------
# gating acceptance test: is the projected encoder still a function of the graph?
# --------------------------------------------------------------------------------
def graph_sensitivity(model: MagNet, graph: Data) -> float:
    """How much the encoder output moves when every edge is reversed.

    Zero means the PE has stopped reading the graph — a positional encoder that
    encodes no position. Aggregate norms cannot see this; only a targeted probe can.
    """
    with torch.no_grad():
        return (model(graph) - model(reversed_graph(graph))).abs().max().item()


@pytest.mark.parametrize("budget", [1.0, None])
def test_projected_encoder_still_reads_the_graph(budget):
    """The premise check: projection must bound the filters, not silence them.

    ``budget=None`` is Cor 4.6 read as ``beta = 1/F`` on the BLOCK operator (the
    default) and is a strict xfail at production width: F=256 with L=5 drives every
    hidden activation below modrelu's ABSOLUTE deadzone radius softplus(-4.6) ~ 0.01,
    which zeroes them, so the output stops depending on the graph entirely. The
    paper's ``beta`` bounds each of the F^2 SCALAR filters, which bounds the assembled
    block operator by F*beta = 1 — that is ``budget=1.0``, and it stays alive. If this
    xfail ever passes, the default budget has been changed and this test should become
    a plain assertion.
    """
    dev = device()
    torch.manual_seed(0)
    # Production width from experiments/base_config.yaml, not the toy width above.
    model = MagNet(1, 256, 5, skip_connection=True, dropout=0.0, k=3).to(dev).eval()
    graph = directed_graph(dev)
    project_beta_(model, budget)

    sensitivity = graph_sensitivity(model, graph)
    if budget is None:
        pytest.xfail("beta = 1/F on the block operator silences the encoder at F=256")
    assert sensitivity > 1e-4, f"PE no longer depends on the graph ({sensitivity:.3g})"


# --------------------------------------------------------------------------------
# global_rms contract
# --------------------------------------------------------------------------------
def test_global_rms_is_one_detached_non_affine_scalar():
    dev = device()
    x = torch.complex(torch.randn(11, HIDDEN, device=dev),
                      torch.randn(11, HIDDEN, device=dev)).requires_grad_(False)
    scale = magnet.global_rms(x)

    assert scale.ndim == 0, "global_rms must be ONE scalar, not per-node or per-channel"
    assert not scale.requires_grad
    expected = (x.real.square() + x.imag.square()).mean().sqrt()
    assert scale.item() == pytest.approx(expected.item(), rel=1e-6)
    # Modulus-based, hence invariant under conjugation and under a global gauge.
    gauge = torch.exp(1j * torch.tensor(0.7, device=dev))
    assert magnet.global_rms(x.conj()).item() == pytest.approx(scale.item(), rel=1e-6)
    assert magnet.global_rms(x * gauge).item() == pytest.approx(scale.item(), rel=1e-6)


def test_global_rms_carries_no_gradient_and_survives_zeros():
    dev = device()
    x = torch.zeros(5, HIDDEN, dtype=torch.complex64, device=dev)
    assert torch.isfinite(magnet.global_rms(x)).all(), "all-zero input produced inf/NaN"

    y = torch.complex(torch.randn(5, HIDDEN, device=dev),
                      torch.randn(5, HIDDEN, device=dev)).requires_grad_(True)
    (y / magnet.global_rms(y)).abs().sum().backward()
    assert torch.isfinite(y.grad).all()


def test_magnet_has_no_layernorm_parameters():
    """The default path must carry no per-node normalization at all."""
    model = make_magnet(device())
    assert not any(isinstance(m, nn.LayerNorm) for m in model.modules())
    assert not hasattr(model, "norms")


def test_hidden_norm_is_validated():
    with pytest.raises(AssertionError, match="Invalid hidden_norm"):
        MagNet(1, HIDDEN, LAYERS, hidden_norm="layer")


def test_hidden_norm_reaches_the_backbone_through_r_pearl():
    pe = RandomGNNPositionalEncodings(pe_hidden_channels=HIDDEN, pe_num_layers=3,
                                      d_model=4, num_samples=2, dropout=0.0, k=2,
                                      directed=True, hidden_norm="global_rms")
    assert pe.pe_gcn.hidden_norm == "global_rms"


# --------------------------------------------------------------------------------
# The trainer's REPORT of the projection (GraphSFTTrainer._register_beta_projection).
# The flag alone says nothing about whether anything is constrained, so the log must
# distinguish "N layers bounded" from "inert" — and measuring must not change what is
# enforced.
# --------------------------------------------------------------------------------
def _report(model, enabled: bool, capsys):
    """Drive _register_beta_projection without the HF Trainer machinery."""
    from prism.training.trainers import GraphSFTTrainer

    bare = GraphSFTTrainer.__new__(GraphSFTTrainer)
    bare.model, bare.gnn_config = model, {"enforce_beta_bound": enabled}
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    bare._register_beta_projection(opt)
    return capsys.readouterr().out


def test_projection_report_names_the_measured_layer_count(capsys):
    out = _report(make_magnet(device()), True, capsys)
    assert f"{LAYERS} MagChebConv layer(s)" in out, out
    assert "INERT" not in out
    # The enforced inequality is the BLOCK norm <= 1; "<= 1/F" misreports it by F.
    assert "sum_k ||H_k||_2 <= 1 " in out and "1/F" not in out, out
    assert "pre-projection slack" in out, out


def test_projection_report_says_inert_without_magcheb(capsys):
    """gnn.directed=false -> TAGConv backbone -> nothing to bound. Say so."""
    pe = RandomGNNPositionalEncodings(pe_hidden_channels=HIDDEN, pe_num_layers=2,
                                      d_model=4, num_samples=2, dropout=0.0, k=2,
                                      directed=False)
    assert not any(isinstance(m, MagChebConv) for m in pe.modules())
    out = _report(pe, True, capsys)
    assert "INERT" in out and "nothing is bounded" in out, out


def test_projection_report_is_silent_when_disabled(capsys):
    assert _report(make_magnet(device()), False, capsys) == ""


def test_measuring_the_slack_does_not_change_what_is_enforced(capsys):
    """The trainer projects once to READ the pre-projection slack, then registers (which
    projects again). The second pass must not change what is enforced.

    It is a no-op up to fp32 ROUNDING, not bit-exactly: after a projection the recomputed
    slack is 1 ± 1e-7, so a layer landing on the high side divides once more by that
    factor. MEASURED at seed 3: one layer of four, relative change 1.2e-7, one time at
    registration only (the per-step hook still projects exactly once per step). The
    invariant is what must hold, so that is what is asserted — plus a tolerance tight
    enough that a real second projection (slack ~4-5 here) could never pass.
    """
    dev = device()
    reference = make_magnet(dev, seed=3)
    register_beta_projection(torch.optim.SGD(reference.parameters(), lr=0.1),
                             reference, enabled=True)

    measured = make_magnet(dev, seed=3)
    _report(measured, True, capsys)

    ref = dict(reference.named_parameters())
    for name, p in measured.named_parameters():
        assert torch.allclose(p.detach(), ref[name].detach(), rtol=1e-5, atol=1e-7), \
            f"{name} diverged by {(p.detach() - ref[name].detach()).abs().max():.3e}"
    for conv in measured.convs:
        assert tap_norm_sum(conv) <= 1 + 1e-5, "the bound must hold after both passes"
