r"""The four structural properties of e17's attention bias, on the REAL encoder.

``MagCompGraphLLM.bias_block`` computes, with ``x = nu(n)``, ``y = nu(m)`` the scene
nodes tokens ``n, m`` mention (anchor substituted when unmentioned)::

    P    = C[:c,:c] * C[nu,nu]
    rho  = corr(P)  ==  corr(C)[n,m] * corr(C)[x,y]      (exact: P's diagonal factorises)
    bias = beta * log((1 + rho) / 2)

These tests pin the four properties the arm's theory rests on, MEASURED through the
trained MagE-GT and real composites rather than asserted:

  1. SCALE INVARIANCE      C -> sC leaves the bias unchanged. This is the property whose
                           absence produced 6.7% (beta-projection ON, C ~1e-8 of its
                           fitted scale) and 10% (projection OFF, bias_absmax 1.9e4,
                           every graph-side grad_norm exactly 0).
  2. PERMUTATION EQUIVARIANCE  relabelling scene nodes must not move the bias by more
                           than the Monte-Carlo probe noise the model already trains
                           against (fixed_seed_mode=false redraws every forward).
  3. STABILITY             a relative perturbation of C must produce a bounded — here
                           CONTRACTIVE — relative change in the bias (Cor 4.6's
                           hypothesis, with a measured constant).
  4. TRANSFERABILITY       an encoder trained at n_30 must behave the same at n_10 and
                           n_100: bias magnitude and discriminative power invariant to
                           graph size.

SKIPPED unless CUDA, the trained checkpoint and the n_* corpora are all present, so this
is a no-op on a dev machine and real on lb1. Read-only on ``data/``.

Run:  CUDA_VISIBLE_DEVICES=0 uv run --with pytest -m pytest tests/test_e17_bias_properties.py -q
"""
import copy
import glob
import os
import sys

import pytest
import torch

sys.path.insert(0, "src")

CKPT = os.environ.get("E17_PROP_CKPT", "outputs/e17_mag_gt/suite3/mag_gt.pt")
M = int(os.environ.get("E17_PROP_M", 320))
EPS = 1e-6
DSETS = {
    "n_10": "data/n_10/gen/*/split/test_graphs",
    "n_30": "data/n_30/gen/nav100_n30_gemma_data/split/test_graphs",
    "n_100": "data/n_100/gen/nav_n100_gemma_data/test_graphs",
}

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.exists(CKPT)
    or not glob.glob(DSETS["n_30"]),
    reason="needs CUDA + the trained MagE-GT + the n_* corpora (lb1)",
)


@pytest.fixture(scope="module")
def rig():
    """Encoder, tokenizer and config — built once for the whole module."""
    from hydra import initialize_config_dir, compose
    from transformers import AutoTokenizer
    from prism.models import gt as gt_module

    with initialize_config_dir(config_dir=os.path.abspath("experiments"), version_base=None):
        cfg = compose(config_name="e17_ms_stage3")
    cfg.gnn.num_samples = M
    tok = AutoTokenizer.from_pretrained(cfg.model.path)

    def build():
        pe = gt_module.build_psi_producer(cfg.gnn, node_feature_dim=None)
        missing, unexpected = pe.load_state_dict(
            torch.load(CKPT, map_location="cpu"), strict=False)
        assert not unexpected, f"checkpoint does not match the architecture: {unexpected[:4]}"
        return pe.to("cuda").eval()

    return cfg, tok, build, build()


def _composite(cfg, tok, sample, perm=None, drop_edge=False):
    from prism.data import utils, compact_prompt
    from prism.models.gnn_llm import (build_composite_graph, build_injection_map,
                                      node_token_variants, find_last_graph_scope,
                                      defer_open_mentions)
    g = cfg.gnn
    msgs = [{"role": "user",
             "content": compact_prompt._task_turn_content(sample.graph, sample.task,
                                                          include_edges=False)}]
    ids = tok(compact_prompt.render(msgs, tok, add_generation_prompt=True),
              add_special_tokens=False)["input_ids"]
    scene = utils.scene_graph_dict_to_pyg(sample.graph, edge_weights="binary")
    if drop_edge and scene.edge_index.shape[1] > 2:
        keep = torch.ones(scene.edge_index.shape[1], dtype=torch.bool)
        keep[0] = False
        scene = copy.copy(scene)
        scene.edge_index = scene.edge_index[:, keep]
    seqs = node_token_variants(scene.node_names, tok)
    scope = find_last_graph_scope(ids, tok)
    imap = defer_open_mentions(build_injection_map(ids, seqs, scope_start=scope), seqs, ids)
    comp = build_composite_graph(
        scene, ids, imap, scope, context_window=int(g.mask_cycle_size), device="cuda",
        cycle_weight=float(g.mask_cycle_weight), cycle_causal=bool(g.mask_cycle_causal),
        crosslink_weight=float(g.mask_crosslink_weight),
        anchor=bool(g.mask_anchor_enabled), anchor_weight=float(g.mask_anchor_weight),
        permutation=perm)
    return comp, scene


def _cov(cfg, pe, comp):
    with torch.no_grad():
        C, _ = pe.pe_model.covariance_token_block(
            comp, comp.num_nodes, pe_pool=cfg.gnn.pe_pool, gt=pe)
    return C.float()


def _bias(C, comp):
    """Mirror of MagCompGraphLLM.bias_block under mask_bias_mode='log'."""
    c, t2n = comp.num_token_nodes, comp.tok2node
    P = C[:c, :c] * C[t2n, :][:, t2n]
    d = P.diagonal()
    d = d.clamp_min(d.max() * EPS).sqrt()
    den = d.unsqueeze(1) * d.unsqueeze(0)
    rho = P / den.clamp_min(torch.finfo(den.dtype).tiny)
    return ((1 + rho) * 0.5).clamp_min(EPS).log(), rho


def _first_sample(pattern):
    from prism.data import data
    paths = glob.glob(pattern)
    if not paths:
        return None
    sbg, _ = data.load_samples_by_graph(paths[0])
    return sbg[sorted(sbg)[0]][0]


def _auc(score, label):
    s, y = score.flatten().float(), label.flatten().float()
    order = s.argsort()
    ranks = torch.empty_like(s)
    ranks[order] = torch.arange(1, s.numel() + 1, device=s.device, dtype=s.dtype)
    npos, nneg = y.sum(), (1 - y).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[y > 0].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def _adjacency_labels(comp, scene):
    """Token pair is RELATED iff the scene nodes they mention are adjacent or identical."""
    c, ns, t2n = comp.num_token_nodes, comp.num_scene_nodes, comp.tok2node
    A = torch.zeros(ns + 1, ns + 1, dtype=torch.bool, device="cuda")
    ei = scene.edge_index.to("cuda")
    if ei.numel():
        A[ei[0], ei[1]] = True
        A[ei[1], ei[0]] = True
    A.fill_diagonal_(True)
    node_of = (t2n - c).clamp(0, ns)
    mentioned = t2n != (c + ns)
    return A[node_of][:, node_of], mentioned.unsqueeze(1) & mentioned.unsqueeze(0)


# --------------------------------------------------------------------------- #
# 1. SCALE INVARIANCE
# --------------------------------------------------------------------------- #
def test_bias_is_invariant_to_the_scale_of_C(rig):
    cfg, tok, _, pe = rig
    comp, _ = _composite(cfg, tok, _first_sample(DSETS["n_30"]))
    C = _cov(cfg, pe, comp)
    b0, _ = _bias(C, comp)
    for s in (1e-8, 1e-4, 1e2, 1e4, 1e8):
        d = float((_bias(C * s, comp)[0] - b0).abs().max())
        assert d < 1e-4, f"bias moved by {d:.3e} when C was scaled by {s:g}"


# --------------------------------------------------------------------------- #
# 2. PERMUTATION EQUIVARIANCE — against the probe-noise floor, not against zero
# --------------------------------------------------------------------------- #
def test_scene_permutation_stays_within_probe_noise(rig):
    from prism.models.utils import Permutation
    cfg, tok, _, pe = rig
    sample = _first_sample(DSETS["n_30"])
    comp, _ = _composite(cfg, tok, sample)
    ref, _ = _bias(_cov(cfg, pe, comp), comp)

    # fixed_seed_mode=false redraws probes every forward: THAT is the noise the model
    # is trained against, and the bar a permutation has to stay under.
    floor = max(float((_bias(_cov(cfg, pe, comp), comp)[0] - ref).abs().max())
                for _ in range(3))
    worst = 0.0
    for seed in (0, 1, 2):
        cp, _ = _composite(cfg, tok, sample, perm=Permutation(seed))
        worst = max(worst, float((_bias(_cov(cfg, pe, cp), cp)[0] - ref).abs().max()))
    assert worst <= 1.5 * floor, (
        f"scene relabelling moved the bias {worst:.4e}, over 1.5x the {floor:.4e} "
        "probe-noise floor — that is a genuine equivariance break, not sampling")


def test_bias_is_deterministic_when_the_probe_draw_is_pinned(rig):
    cfg, tok, build, _ = rig
    pe = build()
    pe.pe_model.fixed_seed_mode, pe.pe_model.fixed_seed_value = True, 0
    comp, _ = _composite(cfg, tok, _first_sample(DSETS["n_30"]))
    a, _ = _bias(_cov(cfg, pe, comp), comp)
    b, _ = _bias(_cov(cfg, pe, comp), comp)
    assert float((a - b).abs().max()) < 1e-5, "pinned probes must give a reproducible bias"


# --------------------------------------------------------------------------- #
# 3. STABILITY
# --------------------------------------------------------------------------- #
def test_perturbing_C_changes_the_bias_sub_linearly(rig):
    cfg, tok, _, pe = rig
    comp, _ = _composite(cfg, tok, _first_sample(DSETS["n_30"]))
    C = _cov(cfg, pe, comp)
    b0, rho = _bias(C, comp)
    # the gate's own Lipschitz constant, d/drho log((1+rho)/2) = 1/(1+rho)
    assert float((1.0 / (1.0 + rho)).max()) < 20.0, "gate derivative unbounded: rho near -1"
    gen = torch.Generator(device="cuda").manual_seed(0)
    for delta in (1e-3, 1e-2, 1e-1):
        E = torch.randn(C.shape, generator=gen, device="cuda")
        E = (E + E.t()) / 2
        Cp = C + delta * C.abs().mean() * E
        rel_in = float((Cp - C).norm() / C.norm())
        rel_out = float((_bias(Cp, comp)[0] - b0).norm() / b0.norm())
        assert rel_out <= rel_in, (
            f"amplification {rel_out / max(rel_in, 1e-12):.3f}x > 1: not contractive")


def test_dropping_one_scene_edge_is_a_proportionate_change(rig):
    cfg, tok, _, pe = rig
    sample = _first_sample(DSETS["n_30"])
    comp, _ = _composite(cfg, tok, sample)
    b0, _ = _bias(_cov(cfg, pe, comp), comp)
    cd, _ = _composite(cfg, tok, sample, drop_edge=True)
    bd, _ = _bias(_cov(cfg, pe, cd), cd)
    n = min(bd.shape[0], b0.shape[0])
    rel = float((bd[:n, :n] - b0[:n, :n]).norm() / b0[:n, :n].norm())
    assert rel < 0.25, f"one dropped scene edge moved the bias {rel:.3f} — not stable"


# --------------------------------------------------------------------------- #
# 4. TRANSFERABILITY — trained at n_30, must hold at n_10 and n_100
# --------------------------------------------------------------------------- #
def test_bias_scale_and_ranking_transfer_across_graph_size(rig):
    cfg, tok, _, pe = rig
    got = {}
    for tag, pattern in DSETS.items():
        sample = _first_sample(pattern)
        if sample is None:
            continue
        comp, scene = _composite(cfg, tok, sample)
        b, rho = _bias(_cov(cfg, pe, comp), comp)
        lab, both = _adjacency_labels(comp, scene)
        got[tag] = (float(b.abs().median()), _auc(rho[both], lab[both]),
                    comp.num_scene_nodes, comp.num_token_nodes)
    assert len(got) >= 2, f"need >= 2 corpora to test transfer, found {list(got)}"
    meds = [v[0] for v in got.values()]
    spread = (max(meds) - min(meds)) / max(meds)
    assert spread < 0.10, (
        f"|bias| median varies {spread:.1%} across graph sizes {got} — does not transfer")
    for tag, (med, auc, ns, c) in got.items():
        assert auc > 0.65, f"{tag} (n_scene={ns}, c={c}): AUC {auc:.4f} — ranking lost"
