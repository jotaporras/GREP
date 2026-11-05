"""e22 GEGRv2 port — graded hop-table mask (``mask_form="hop_table"``) and the
post-fusion tap selector (``post_fusion_hop_taps``). docs/2026-11-05 e22_greg_port.md.

Invariants locked here:

- ``node_hops`` agrees with ``node_adjacency`` (``hops <= k`` is the adjacency) and
  gives the true shortest-path distance on a path graph, ``k+1`` beyond.
- The mask bias is ``mask_level[hop]`` on pairs within k_hops (ADDITIVE, not a
  log-gate) and ``finfo.min`` beyond; the GEGRv2 default init is 0.7/0.7/0.35/0.175.
- Ψ is absent from the mask: the bias carries gradient to ``mask_level`` only, and a
  model without post-fusion sends NO gradient to ``pe_model``.
- ``mask_level`` trains at base LR (``base_lr_parameters``), not in the damped
  structural group, and sits on the LLM device (first-parameter rule).
- Decision rows (e18-A) in hop-table form are ``decision_gain · level[hop]``.
- ``save_run_dir`` writes ``mask_level`` and the loader fails loud without it.
- ``post_fusion_hop_taps``: "first"/"last"/"ends" select the forward_taps channels,
  size the per-channel modules accordingly, and are rejected outside depth mode.
"""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, "tests")

import pytest
import torch
from torch_geometric.data import Data

from prism.models import gnn_llm
from prism.models.gnn_llm import (LearnableGraphMaskLLM, node_adjacency, node_hops,
                                  hop_table_level_init, select_pf_taps)
from test_learnable_graph_mask import _tiny_llm, _StubPE, _graph, DEVICE, NEG
from test_hop_fusion import _StubTapPE


def _wrap(k_hops=3, level_init=None, decision=False, **kw):
    torch.manual_seed(0)
    llm = _tiny_llm()
    pe = _StubPE(feat_dim=4, d=8)
    return LearnableGraphMaskLLM(
        llm, pe, layer_scope="all", mask_form="hop_table", k_hops=k_hops,
        mask_level_init=level_init, decision_gating=decision,
        decision_gain_init=3.0 if decision else 0.0, **kw).eval()


def _path(n=5):
    return _graph(n, [(i, i + 1) for i in range(n - 1)])


def _mask(wrap, seq, graphs, imap):
    return wrap.build_structural_mask(seq, graphs, imap, DEVICE, dtype=torch.float32)[0, 0].cpu()


# ---------------------------------------------------------------- node_hops

def test_node_hops_is_shortest_path_distance_capped_at_k_plus_1():
    g = _path(5)
    hops = node_hops(g, DEVICE, k_hops=3).cpu()
    expect = torch.tensor([[abs(i - j) if abs(i - j) <= 3 else 4 for j in range(5)]
                           for i in range(5)])
    assert torch.equal(hops, expect)


@pytest.mark.parametrize("k_hops", [1, 2, 3])
def test_node_hops_support_equals_node_adjacency(k_hops):
    g = _graph(6, [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5)])
    adj = node_adjacency(g, DEVICE, k_hops=k_hops)
    hops = node_hops(g, DEVICE, k_hops=k_hops)
    assert torch.equal(hops <= k_hops, adj)


def test_node_hops_use_edges_false_is_self_loops_only():
    g = _path(4)
    hops = node_hops(g, DEVICE, k_hops=3, use_edges=False).cpu()
    assert torch.equal(hops, torch.full((4, 4), 4).fill_diagonal_(0))


# ---------------------------------------------------------------- mask values

def test_default_level_init_matches_gegrv2():
    assert hop_table_level_init(3) == pytest.approx([0.7, 0.7, 0.35, 0.175])
    w = _wrap(k_hops=3)
    assert torch.allclose(w.mask_level.detach().cpu(),
                          torch.tensor([0.7, 0.7, 0.35, 0.175]))


def test_bias_is_level_of_hop_additive_and_blocked_beyond_k():
    level = [0.5, 1.5, -0.25, -0.75]
    w = _wrap(k_hops=3, level_init=level)
    g = _path(5)
    imap = [{i: [(2 * i + 1, 2 * i + 2)] for i in range(5)}]
    m = _mask(w, 12, [g], imap)
    for i in range(5):
        for j in range(5):
            d = abs(i - j)
            v = m[2 * i + 1, 2 * j + 1].item()
            if d <= 3:
                assert v == pytest.approx(level[d]), (i, j, v)
            else:
                assert v == NEG, (i, j, v)
    # non-node rows/cols untouched
    assert m[0].abs().sum() == 0 and m[:, 0].abs().sum() == 0
    assert m[11].abs().sum() == 0


def test_level_init_length_is_validated():
    with pytest.raises(ValueError, match="k_hops \\+ 1"):
        _wrap(k_hops=3, level_init=[0.1, 0.2])
    with pytest.raises(ValueError, match="mask_form"):
        torch.manual_seed(0)
        LearnableGraphMaskLLM(_tiny_llm(), _StubPE(feat_dim=4, d=8), mask_form="nope")


def test_grad_reaches_levels_not_psi():
    w = _wrap(k_hops=2)
    w.train()
    g = _path(4)
    imap = [{i: [(2 * i + 1, 2 * i + 2)] for i in range(4)}]
    m = w.build_structural_mask(10, [g], imap, DEVICE, dtype=torch.float32)
    finite = m[m > NEG / 2]
    finite.sum().backward()
    assert w.mask_level.grad is not None and w.mask_level.grad.abs().sum() > 0
    assert all(p.grad is None for p in w.pe_model.parameters()), \
        "Ψ must be absent from the hop-table mask"


def test_levels_train_at_base_lr_on_llm_device():
    w = _wrap()
    assert any(p is w.mask_level for p in w.base_lr_parameters())
    assert all(p is not w.mask_level for p in w.structural_parameters())
    assert w.mask_level.device == next(w.llm.parameters()).device
    assert next(w.parameters()).device == w.mask_level.device


def test_psi_form_is_unchanged_by_the_new_kwargs():
    """Default construction (mask_form='psi') must not gain a mask_level parameter
    or change its base-LR group."""
    torch.manual_seed(0)
    w = LearnableGraphMaskLLM(_tiny_llm(), _StubPE(feat_dim=4, d=8), layer_scope="all")
    assert not hasattr(w, "mask_level")
    assert w.base_lr_parameters() == []
    assert w._mask_form == "psi"


# ---------------------------------------------------------------- decision rows (e18-A)

def test_decision_rows_are_gain_times_level():
    """Non-negative levels: the row is gain · level[hop] within k, 0 beyond
    (negative levels are clamped; see test_decision_rows_never_below_non_neighbour)."""
    level = [0.5, 1.5, 0.25, 0.75]
    w = _wrap(k_hops=3, level_init=level, decision=True)
    g = _path(5)
    adj, node_m = w._node_mask_logits(g, DEVICE)
    soft = w.decision_node_values(adj, node_m).cpu()
    for i in range(5):
        for j in range(5):
            d = abs(i - j)
            assert soft[i, j].item() == pytest.approx(3.0 * level[d] if d <= 3 else 0.0)


# ---------------------------------------------------------------- save / load

def test_run_dir_saves_levels_and_loader_requires_them(tmp_path):
    from prism.training.run_dir import save_run_dir
    w = _wrap(k_hops=3, level_init=[0.1, 0.2, 0.3, 0.4])
    cfg = {"architecture": "learnable_graph_mask", "mask_form": "hop_table",
           "mask_k_hops": 3}
    save_run_dir(w, cfg, str(tmp_path))
    weights = torch.load(tmp_path / "gnn_weights.pt", map_location="cpu")
    assert torch.equal(weights["mask_level"], torch.tensor([0.1, 0.2, 0.3, 0.4]))
    # psi form writes no levels
    torch.manual_seed(0)
    w_psi = LearnableGraphMaskLLM(_tiny_llm(), _StubPE(feat_dim=4, d=8), layer_scope="all")
    save_run_dir(w_psi, {"architecture": "learnable_graph_mask"}, str(tmp_path / "psi"))
    assert "mask_level" not in torch.load(tmp_path / "psi" / "gnn_weights.pt",
                                          map_location="cpu")


def test_loader_source_requires_mask_level():
    """The loader's fail-loud key set covers mask_level (source-level contract: the
    full loader needs a Gemma checkpoint on disk)."""
    import inspect
    from prism.models import loaders
    src = inspect.getsource(loaders)
    assert '_require("mask_form=hop_table", ("mask_level",))' in src
    assert 'model.mask_level.data.copy_(gnn_weights["mask_level"])' in src


# ---------------------------------------------------------------- tap selector

def test_select_pf_taps_shapes():
    taps = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)
    assert torch.equal(select_pf_taps(taps, "all"), taps)
    assert torch.equal(select_pf_taps(taps, "first"), taps[:, :1])
    assert torch.equal(select_pf_taps(taps, "last"), taps[:, 3:4])
    ends = select_pf_taps(taps, "ends")
    assert ends.shape == (2, 2, 3)
    assert torch.equal(ends[:, 0], taps[:, 0]) and torch.equal(ends[:, 1], taps[:, 3])


def _pf_model(hop_taps, hop_mode="depth", mask_form="hop_table"):
    torch.manual_seed(0)
    llm = _tiny_llm()
    pe = (_StubTapPE(d=8) if hop_mode == "depth" else _StubPE(d=8)).to(DEVICE)
    return LearnableGraphMaskLLM(
        llm, pe, layer_scope="all", mask_form=mask_form, k_hops=3,
        post_fusion=True, post_fusion_layer_scope="all", post_fusion_d_gt=8,
        post_fusion_hop_mode=hop_mode, post_fusion_hop_taps=hop_taps,
        post_fusion_gain_init=1.0, post_fusion_hop_gt_layers=2,
        post_fusion_hop_gt_heads=2, post_fusion_hop_gt_k=1).to(DEVICE)


@pytest.mark.parametrize("hop_taps,n_ch,pick", [
    ("all", 3, [0, 1, 2]), ("first", 1, [0]), ("last", 1, [2]), ("ends", 2, [0, 2])])
def test_hop_taps_size_modules_and_pf_psi(hop_taps, n_ch, pick):
    m = _pf_model(hop_taps)
    assert len(m.pf_proj) == n_ch and len(m.pf_norm) == n_ch
    assert m.pf_ch_gain.shape == (n_ch,)
    g = _path(4)
    with torch.no_grad():
        full = m.pe_model.forward_taps(g).float()
        psi = m.pf_psi(g)
    assert psi.shape == (4, n_ch, 8)
    assert torch.equal(psi, full[:, pick])
    # the write consumes the selected stack without a shape error
    with torch.no_grad():
        vec = m._pf_project(psi)
    assert vec.shape == (4, m.pf_out_features)


def test_first_tap_write_reaches_rpearl_channel_only():
    """With taps=first the write reads channel 0 (the R-PEARL tap): a change to
    channel 2 must not move the projected vector."""
    m = _pf_model("first")
    for norm in m.pf_norm:
        torch.nn.init.ones_(norm.weight)
    g = _path(4)
    with torch.no_grad():
        full = m.pe_model.forward_taps(g).float()
        a = m._pf_project(select_pf_taps(full, "first"))
        full2 = full.clone()
        full2[:, 2] += 5.0
        b = m._pf_project(select_pf_taps(full2, "first"))
        full3 = full.clone()
        full3[:, 0] += 5.0
        c = m._pf_project(select_pf_taps(full3, "first"))
    assert torch.equal(a, b)
    assert not torch.allclose(a, c)


def test_hop_taps_rejected_outside_depth_mode():
    with pytest.raises(ValueError, match="post_fusion_hop_mode='depth'"):
        _pf_model("first", hop_mode="shift")
    with pytest.raises(ValueError, match="post_fusion_hop_taps"):
        _pf_model("middle")


def test_psi_form_with_taps_still_works():
    m = _pf_model("first", mask_form="psi")
    assert m._pf_hop_taps == "first" and len(m.pf_proj) == 1


def test_forward_runs_with_hop_table_and_first_tap_write():
    m = _pf_model("first")
    g = _path(4)
    imap = {i: [(2 * i + 1, 2 * i + 2)] for i in range(4)}
    ids = torch.randint(0, 64, (1, 10), generator=torch.Generator().manual_seed(2)).to(DEVICE)
    m.train()
    out = m(input_ids=ids, graphs=[g], injection_maps=[imap], labels=ids)
    out.loss.backward()
    assert torch.isfinite(out.loss)
    assert m.mask_level.grad is not None
    # the write's norm scale gets gradient at init (open gate, zero scale)
    assert m.pf_norm[0].weight.grad is not None and m.pf_norm[0].weight.grad.abs().sum() > 0


# ---------------------------------------------------------------- review follow-ups (2026-11-05)

def test_node_hops_support_matches_adjacency_under_permutation_and_directed():
    from prism.models.utils import Permutation
    g = _graph(6, [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5)])
    for k in (1, 2, 3):
        for sym in (True, False):
            perm = Permutation(seed=3)
            adj = node_adjacency(g, DEVICE, k_hops=k, symmetrize=sym, permutation=perm)
            hops = node_hops(g, DEVICE, k_hops=k, symmetrize=sym, permutation=perm)
            assert torch.equal(hops <= k, adj), (k, sym)


def test_decode_rows_equal_prefill_node_block():
    """mask_node_values (decode) and build_structural_mask (prefill) draw the same
    hop-table values for every node pair."""
    w = _wrap(k_hops=2, level_init=[0.5, 1.5, -0.3])
    g = _path(5)
    seq = 7
    imap = {i: [(i + 1, i + 2)] for i in range(5)}  # node i at position i+1 (BOS at 0)
    pre = _mask(w, seq, [g], [imap])
    dec = gnn_llm.mask_node_values(w, g, DEVICE).cpu()
    for i in range(5):
        for j in range(5):
            assert pre[i + 1, j + 1] == dec[i, j]


def test_decision_rows_never_below_non_neighbour():
    """Negative trained levels (GEGRv2 e028 end state) must not push a within-k node
    below a non-neighbour, which gets 0."""
    w = _wrap(k_hops=3, level_init=[0.45, 1.6, -0.24, -0.65], decision=True)
    g = _path(6)
    adj, node_m = w._node_mask_logits(g, DEVICE)
    rows = w.decision_node_values(adj, node_m).detach().cpu()
    assert (rows >= 0).all()
    assert rows[0, 1] == pytest.approx(3.0 * 1.6)
    assert rows[0, 2] == 0.0 and rows[0, 5] == 0.0


def test_forward_taps_n_blocks_is_a_prefix_of_the_full_stack():
    from prism.models import gt
    torch.manual_seed(0)
    cfg = dict(d_model=8, dropout=0.0, pe_hidden_channels=8, pe_num_layers=2,
               num_samples=4, k_pe=2, gt_num_layers=2, gt_heads=2, k_gt=1, eps=1e-6,
               use_layer_norm=True, pe_gt_from=None, semantic_gt_from=None)
    tower = gt.build_psi_producer(cfg, node_feature_dim=4).to(DEVICE).eval()
    g = _path(5)
    g.x = torch.randn(5, 4)
    with torch.no_grad():
        full = tower.forward_taps(g)
        first = tower.forward_taps(g, n_blocks=0)
    assert full.shape[1] == 3 and first.shape == (5, 1, 8)
    assert torch.equal(first, full[:, :1])


def test_pf_psi_first_tap_skips_gt_blocks():
    m = _pf_model("first")
    calls = []
    orig = m.pe_model.forward_taps
    def spy(g, permutation=None, n_blocks=None):
        calls.append(n_blocks)
        return orig(g, permutation=permutation, n_blocks=n_blocks)
    m.pe_model.forward_taps = spy
    with torch.no_grad():
        psi = m.pf_psi(_path(4))
    assert calls == [0] and psi.shape == (4, 1, 8)


def test_train_v3_records_the_e22_keys():
    """The SFT train_config.json is what the eval loader rebuilds from: an
    unrecorded mask_form reloads a psi-form mask and silently ignores mask_level
    (found in the 2026-11-05 review)."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "src" / "prism" / "training" / "train_v3.py").read_text()
    for key in ("mask_form", "mask_level_init", "post_fusion_hop_taps", "fusion_lr_mult"):
        assert f'"{key}"' in src, key


def _ensure_spine_importable():
    """prism.training.trainers imports prism.eval.evaluate, which needs the external
    ``spine`` package (present on betty, absent on the Mac). Stub ONLY spine's import
    surface when it is missing; every prism module stays real."""
    import sys, types
    try:
        import spine  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    def _mod(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(sys.modules[parent], child, m)
        return m
    class _InMemoryLLM:  # subclassed at import by prism.models.inference
        pass
    _mod("spine", models=types.SimpleNamespace(InMemoryLLM=_InMemoryLLM))
    _mod("spine.spine", SPINE=object)
    _mod("spine.spine_util", UpdatePromptFormer=object)
    _mod("spine.mapping"); _mod("spine.mapping.graph_util", GraphHandler=object,
                                parse_graph_coord=lambda *a, **k: None)
    _mod("spine.prompts"); _mod("spine.prompts.examples", EXAMPLE_1=[], EXAMPLE_2=[])
    _mod("spine.prompts.prompts", get_base_prompt_update_graph=lambda *a, **k: None,
         SYS_PROMPT={"role": "system", "content": "x"})


def test_fusion_lr_mult_puts_levels_and_pf_in_their_own_group():
    _ensure_spine_importable()
    from prism.training.trainers import create_two_group_optimizer

    class _Args:
        learning_rate = 1e-3
        weight_decay = 0.01

    class _Trainer:
        def __init__(self, model, cfg):
            self.model, self.gnn_config, self.args, self.optimizer = model, cfg, _Args(), None
        def get_decay_parameter_names(self, model):
            return [n for n, p in model.named_parameters() if p.ndim > 1]
        def get_optimizer_cls_and_kwargs(self, args, model=None):
            return torch.optim.AdamW, {"lr": args.learning_rate}

    m = _pf_model("first")
    for p in m.parameters():
        p.requires_grad = True
    t = _Trainer(m, {"structural_lr_mult": 0.012, "fusion_lr_mult": 5.0})
    opt = create_two_group_optimizer(t, m.structural_parameters(), m.base_lr_parameters())
    lr_of = {id(p): grp["lr"] for grp in opt.param_groups for p in grp["params"]}
    assert lr_of[id(m.mask_level)] == pytest.approx(5e-3)
    assert lr_of[id(m.pf_gain)] == pytest.approx(5e-3)
    assert all(lr_of[id(p)] == pytest.approx(1.2e-5) for p in m.pe_model.parameters())
    assert all(lr_of[id(p)] == pytest.approx(1e-3) for p in m.llm.parameters())
    # default multiplier 1.0 on both groups: stock optimizer path
    t2 = _Trainer(m, {"structural_lr_mult": 1.0, "fusion_lr_mult": 1.0})
    assert create_two_group_optimizer(t2, m.structural_parameters(), m.base_lr_parameters()) is None
