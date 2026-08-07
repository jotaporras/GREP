"""Tests for the navigator Ψ producer (``gt.NavigatorPE`` / ``gt.build_psi_producer``).

The notebook (``notebooks/2026-06-28 e9_gnn_navigation.ipynb``) trains a shortest-path navigator whose
two halves — a probe-PE ``GraphTransformer`` (``path_navigator_gt.pt``) and a
``SemanticGraphTransformer`` head (``path_navigator_agt.pt``) — are reused here as the Ψ
producer of the two Ψ-consuming architectures: ``learnable_graph_mask`` (Ψ Ψᵀ attention
bias) and ``wire_llm`` (Ψ as q/k rotation angles). What these tests pin down:

  1. ONE factory builds the producer for BOTH architectures, at training time and at
     eval-rebuild time (``gt.build_psi_producer``), so the two can never drift.
  2. The head's input width is the PE GT's ``d_model`` and its depth/heads/k_gt come from
     the same ``gt_*`` keys — the notebook builds both from one ``model_hparams`` dict.
  3. The pretrained-weight load is STRICT: hyperparameter drift, or a producer topology
     that disagrees with the checkpoint (legacy ``TwoStagePE`` vs standalone GT), raises
     instead of silently evaluating a randomly-initialised Ψ.
  3b. The PE/AGT SPLIT: ``NavigatorPE`` is the PE stage alone (Ψ = pe_gt(graph), NO
     ``semantic_gt``), ``NavigatorGT`` is the one that owns the AGT head, and the legacy
     two-stage Ψ survives ONLY under ``gt.TwoStagePE`` so pre-split checkpoints reload as
     the function they were trained as (never remapped into a different one).
  4. save → load round trip of a two-stage checkpoint through the exact dict
     ``trainers.GraphSFTTrainer.save_model`` writes.
  5. The permutation-equivariance path reaches BOTH halves (the head must refine Ψ over
     the same relabelled graph the PE was computed on).
  6. ``NavigatorGT`` — the AGT stage — reproduces the notebook's ``GNNShortestPathNavigator``
     (§3, cell 75): head hparams, ``self.shape``, the ``graph.x + PE`` forward composition
     and its cache, the ``generate`` rollout (seeding, sinusoidal step code, blurry-vision
     + visited masking, ``N - 1`` bound, Cauchy cleanup, ``[len, 1]`` return), the
     tag-distribution knobs, and degenerate-graph safety.

Mostly construction / state-dict only: tiny random-init modules, no training. The rollout
tests run on MPS when available (CUDA otherwise, CPU last) — they are seconds-scale.

Run:  uv run --with pytest -m pytest tests/test_navigator_pe.py -q
"""
import os
import sys
import tempfile

sys.path.insert(0, "src")

import torch
from torch import nn
from torch_geometric.data import Data

from prism.models import gt as gt_module
from prism.models import loaders
from prism.models.utils import Permutation


# Tiny stand-in for the notebook's model_hparams (num_layers=3, d_model=1024, heads=8,
# k_gt=2, ...). Same KEY NAMES the gnn config / train_config.json use.
_CFG = dict(gt_num_layers=2, gt_heads=2, k_gt=2, d_model=8, dropout=0.0, eps=1e-6,
            use_layer_norm=True, pe_hidden_channels=4, pe_num_layers=2,
            num_samples=4, k_pe=2)


def _cfg(**over):
    return {**_CFG, **over}


def _nav_cfg(**over):
    """Config selecting navigator mode (both sources set; paths are never read here)."""
    return _cfg(pe_gt_from="path_navigator_gt.pt",
                semantic_gt_from="path_navigator_agt.pt", **over)


def _skip(msg):
    if __name__ != "__main__" and "pytest" in sys.modules:
        import pytest
        pytest.skip(msg)
    print(f"[SKIP] {msg}")
    return None


class _Holder(nn.Module):
    """Minimal stand-in for the LLM wrappers: the loaders only touch ``.pe_model``."""

    def __init__(self, pe_model):
        super().__init__()
        self.pe_model = pe_model


def _graph(n=5):
    src = list(range(n - 1))
    dst = list(range(1, n))
    ei = torch.tensor([src + dst, dst + src], dtype=torch.long)
    g = Data(x=torch.randn(n, 3), edge_index=ei, num_nodes=n)
    g.node_names = [f"node{i}" for i in range(n)]
    return g


# --------------------------------------------------- the PE / AGT class split

def test_navigator_pe_is_the_pe_stage_only():
    """``NavigatorPE`` holds the probe-PE GT and NOTHING else — no semantic_gt, no
    classifier, no generate. Ψ is exactly ``pe_gt(graph)``."""
    torch.manual_seed(0)
    pe_gt = gt_module.build_psi_producer(_cfg())              # a bare GraphTransformer
    nav_pe = gt_module.NavigatorPE(pe_gt).eval()
    assert not hasattr(nav_pe, "semantic_gt"), \
        "NavigatorPE must NOT contain the AGT head — that belongs to NavigatorGT"
    for attr in ("classifier", "generate"):
        assert not hasattr(nav_pe, attr), attr
    assert set(nav_pe.state_dict()) == {"pe_gt." + k for k in pe_gt.state_dict()}
    g = _graph(5)
    torch.manual_seed(1)
    a = nav_pe(g)
    torch.manual_seed(1)
    b = pe_gt(g)
    assert torch.allclose(a, b), "NavigatorPE.forward must be pe_gt(graph), unmodified"


def test_navigator_gt_owns_the_agt_head():
    """``NavigatorGT`` = NavigatorPE (PE stage) + the AGT: semantic_gt + classifier +
    generate. The inheritance direction is what the notebook's navigator is."""
    nav = _navigator()
    assert isinstance(nav, gt_module.NavigatorPE), "NavigatorGT must extend NavigatorPE"
    assert isinstance(nav.semantic_gt, gt_module.SemanticGraphTransformer)
    assert isinstance(nav.classifier, nn.Linear) and callable(nav.generate)
    keys = set(nav.state_dict())
    assert any(k.startswith("pe_gt.") for k in keys)
    assert any(k.startswith("semantic_gt.") for k in keys)
    assert any(k.startswith("classifier.") for k in keys)


def test_two_stage_pe_is_the_legacy_producer_not_navigator_pe():
    """The legacy Ψ = SemanticGT(PE_GT(·)) lives in ``TwoStagePE``, a NavigatorPE subclass,
    and keeps the pre-split KEY LAYOUT so old checkpoints load tensor-for-tensor."""
    two = gt_module.build_psi_producer(_nav_cfg())
    assert isinstance(two, gt_module.TwoStagePE)
    assert isinstance(two, gt_module.NavigatorPE)
    assert type(two) is not gt_module.NavigatorPE
    keys = set(two.state_dict())
    assert any(k.startswith("pe_gt.") for k in keys) and any(
        k.startswith("semantic_gt.") for k in keys)


def test_psi_producer_is_gt_only_when_semantic_gt_from_is_null():
    """The Ψ a current run trains is the PE GT ALONE — a bare GraphTransformer whose keys
    carry no ``pe_gt.``/``semantic_gt.`` prefix (what e14 passes: semantic_gt_from=null)."""
    psi = gt_module.build_psi_producer(_cfg(pe_gt_from="path_navigator_gt.pt"))
    assert type(psi) is gt_module.GraphTransformer
    assert not any(k.startswith(("pe_gt.", "semantic_gt.")) for k in psi.state_dict())


def test_remap_shim_loads_a_navigator_pe_layout_checkpoint_into_a_bare_gt():
    """A ``pe_gt.*``-prefixed checkpoint holds the SAME function as a bare GT, so the
    loader strips the prefix and loads it losslessly (the key-remap shim)."""
    torch.manual_seed(0)
    trained = gt_module.build_psi_producer(_cfg())
    wrapped = gt_module.NavigatorPE(trained)
    torch.manual_seed(9)
    fresh = gt_module.build_psi_producer(_cfg())
    assert not _sd_equal(trained.state_dict(), fresh.state_dict())      # no-op guard
    loaders._load_psi_producer_state(fresh, wrapped.state_dict(), "ckpt/", _cfg())
    assert _sd_equal(trained.state_dict(), fresh.state_dict())


def test_remap_shim_refuses_to_drop_the_semantic_half():
    """A two-stage checkpoint into a PE-only Ψ must RAISE. Silently stripping
    ``semantic_gt.*`` would reload the run as a different Ψ under the same weights."""
    torch.manual_seed(0)
    state = gt_module.build_psi_producer(_nav_cfg()).state_dict()
    bare = gt_module.build_psi_producer(_cfg())
    try:
        loaders._load_psi_producer_state(bare, state, "ckpt/", _cfg())
    except RuntimeError as e:
        assert "semantic_gt_from" in str(e) and "DIFFERENT function" in str(e), str(e)
        return
    raise AssertionError("expected a RuntimeError refusing the lossy two-stage remap")


def _sd_equal(a, b):
    return set(a) == set(b) and all(torch.equal(v, b[k]) for k, v in a.items())


# ---------------------------------------------------------------- the factory

def test_factory_builds_navigator_only_when_both_sources_are_set():
    """pe_gt_from + semantic_gt_from ⇒ NavigatorPE; either alone / neither ⇒ standalone GT."""
    nav = gt_module.build_psi_producer(_nav_cfg())
    assert isinstance(nav, gt_module.NavigatorPE), type(nav)
    assert isinstance(gt_module.build_psi_producer(_cfg()), gt_module.GraphTransformer)
    gt_only = gt_module.build_psi_producer(_cfg(pe_gt_from="path_navigator_gt.pt"))
    assert isinstance(gt_only, gt_module.GraphTransformer), \
        "pe_gt_from alone must stay a standalone GT (the GT-only Ψ arm)"


def test_factory_rejects_semantic_source_without_pe_gt():
    """A head with no PE GT is not a Ψ producer — fail loud, naming both knobs."""
    try:
        gt_module.build_psi_producer(_cfg(semantic_gt_from="path_navigator_agt.pt"))
    except ValueError as e:
        assert "pe_gt_from" in str(e) and "semantic_gt_from" in str(e), str(e)
        return
    raise AssertionError("expected a ValueError naming gnn.pe_gt_from / gnn.semantic_gt_from")


def test_navigator_head_shape_follows_the_pe_gt():
    """head.node_feature_dim == pe_gt.d_model, and the head's depth/heads/k_gt are the gt_* keys.

    The notebook builds the head from the SAME ``model_hparams`` as the PE GT, so any other
    rebuild would fail the strict load of ``path_navigator_agt.pt``.
    """
    nav = gt_module.build_psi_producer(_nav_cfg())
    assert nav.semantic_gt.input_proj.in_features == nav.pe_gt.d_model
    assert nav.semantic_gt.d_model == _CFG["d_model"]
    assert len(nav.semantic_gt.blocks) == _CFG["gt_num_layers"]
    assert nav.semantic_gt.blocks[0].attn.heads == _CFG["gt_heads"]
    assert nav.semantic_gt.k_hops == _CFG["k_gt"]


def test_navigator_forward_returns_psi_rows():
    nav = gt_module.build_psi_producer(_nav_cfg()).eval()
    with torch.no_grad():
        psi = nav(_graph(5))
    assert psi.shape == (5, _CFG["d_model"]), psi.shape
    assert torch.isfinite(psi).all()


# ------------------------------------------------- pretrained-weight loading

def _write_state(module, path):
    torch.save(module.state_dict(), path)
    return path


def test_load_navigator_pe_into_round_trips_both_submodules():
    """The notebook's two state dicts land in pe_gt / semantic_gt, tensor-for-tensor."""
    torch.manual_seed(0)
    trained = gt_module.build_psi_producer(_nav_cfg())
    torch.manual_seed(1)
    fresh = gt_module.build_psi_producer(_nav_cfg())
    fresh_sd = fresh.state_dict()
    assert any(not torch.equal(v, fresh_sd[k]) for k, v in trained.state_dict().items()), \
        "fixture would pass vacuously — the two builds must differ"

    with tempfile.TemporaryDirectory() as td:
        loaders.load_navigator_pe_into(
            _Holder(fresh),
            _write_state(trained.pe_gt, os.path.join(td, "path_navigator_gt.pt")),
            _write_state(trained.semantic_gt, os.path.join(td, "path_navigator_agt.pt")))

    for name, (a, b) in (("pe_gt", (trained.pe_gt, fresh.pe_gt)),
                         ("semantic_gt", (trained.semantic_gt, fresh.semantic_gt))):
        sa, sb = a.state_dict(), b.state_dict()
        assert set(sa) == set(sb)
        for k in sa:
            assert torch.equal(sa[k], sb[k]), f"{name}.{k} not loaded"


def test_load_navigator_pe_into_rejects_a_semantic_source_for_a_standalone_gt():
    """semantic_gt_from with a GT-only producer is a config error, not a partial load."""
    torch.manual_seed(0)
    trained = gt_module.build_psi_producer(_nav_cfg())
    holder = _Holder(gt_module.build_psi_producer(_cfg()))   # standalone GT
    with tempfile.TemporaryDirectory() as td:
        gt_path = _write_state(trained.pe_gt, os.path.join(td, "gt.pt"))
        agt_path = _write_state(trained.semantic_gt, os.path.join(td, "agt.pt"))
        try:
            loaders.load_navigator_pe_into(holder, gt_path, agt_path)
        except RuntimeError as e:
            assert "semantic_gt_from" in str(e) and "pe_gt_from" in str(e), str(e)
            return
    raise AssertionError("expected a RuntimeError for semantic_gt_from on a standalone GT")


def test_load_navigator_pe_into_fails_loud_on_hparam_drift():
    """A head/GT that does not reproduce the pretrained shapes must raise, not silently drop."""
    torch.manual_seed(0)
    trained = gt_module.build_psi_producer(_nav_cfg(gt_num_layers=3))
    fresh = gt_module.build_psi_producer(_nav_cfg())          # depth 2 vs the saved 3
    with tempfile.TemporaryDirectory() as td:
        gt_path = _write_state(trained.pe_gt, os.path.join(td, "gt.pt"))
        agt_path = _write_state(trained.semantic_gt, os.path.join(td, "agt.pt"))
        try:
            loaders.load_navigator_pe_into(_Holder(fresh), gt_path, agt_path)
        except RuntimeError:
            return
    raise AssertionError("expected a RuntimeError on gt_num_layers drift")


def test_gt_only_load_fails_loud_on_hparam_drift():
    """Same guarantee on the GT-only arm (strict=False + explicit key check)."""
    torch.manual_seed(0)
    trained = gt_module.build_psi_producer(_cfg(gt_num_layers=3))
    fresh = gt_module.build_psi_producer(_cfg())
    with tempfile.TemporaryDirectory() as td:
        gt_path = _write_state(trained, os.path.join(td, "gt.pt"))
        try:
            loaders.load_navigator_pe_into(_Holder(fresh), gt_path, None)
        except RuntimeError as e:
            assert "gt_num_layers" in str(e) or "hyperparameters" in str(e), str(e)
            return
    raise AssertionError("expected a RuntimeError on GT-only hparam drift")


# ------------------------------------------------------- checkpoint round trip

def _saved_gnn_weights(pe_model):
    """EXACTLY the payload ``trainers.GraphSFTTrainer.save_model`` writes for the two
    Ψ-consuming archs (learnable_graph_mask: {"pe_model"}; wire_llm: + wire tensors)."""
    return {"pe_model": {k: v.clone() for k, v in pe_model.state_dict().items()}}


def test_navigator_checkpoint_round_trip_through_the_eval_rebuild():
    """save_model → build_psi_producer(train_config gnn) → strict load ⇒ identical Ψ producer.

    This is the whole eval boundary for a navigator run: the checkpoint holds ONE flat
    ``pe_model`` state dict whose ``pe_gt.*`` / ``semantic_gt.*`` keys only exist if the
    rebuild also chose NavigatorPE.
    """
    torch.manual_seed(0)
    trained = gt_module.build_psi_producer(_nav_cfg())
    state = _saved_gnn_weights(trained)
    assert any(k.startswith("pe_gt.") for k in state["pe_model"]) and \
        any(k.startswith("semantic_gt.") for k in state["pe_model"]), \
        "a NavigatorPE checkpoint must carry both submodules"

    torch.manual_seed(2)
    rebuilt = gt_module.build_psi_producer(_nav_cfg())
    loaders._load_psi_producer_state(rebuilt, state["pe_model"], "ckpt/", _nav_cfg())
    for k, v in trained.state_dict().items():
        assert torch.equal(v, rebuilt.state_dict()[k]), f"{k} did not survive the round trip"


def test_eval_rebuild_without_the_navigator_keys_raises():
    """train_config.json missing semantic_gt_from ⇒ the rebuild is the PE-only Ψ while the
    checkpoint holds the LEGACY two-stage Ψ = SemanticGT(PE_GT(·)). Must raise, naming the
    knob — a strict=False load would drop every tensor and evaluate a random Ψ in silence,
    and a key-remap that merely DROPPED semantic_gt.* would evaluate a different function."""
    torch.manual_seed(0)
    state = _saved_gnn_weights(gt_module.build_psi_producer(_nav_cfg()))
    standalone = gt_module.build_psi_producer(_cfg())        # provenance keys lost
    try:
        loaders._load_psi_producer_state(standalone, state["pe_model"], "ckpt/", _cfg())
    except RuntimeError as e:
        assert "semantic_gt_from" in str(e), str(e)
        assert "GraphTransformer" in str(e), str(e)
        return
    raise AssertionError("expected a RuntimeError naming the navigator provenance keys")


def test_eval_rebuild_of_a_standalone_gt_checkpoint_still_works():
    """The non-navigator path is unchanged: same topology ⇒ clean strict load."""
    torch.manual_seed(0)
    trained = gt_module.build_psi_producer(_cfg())
    state = _saved_gnn_weights(trained)
    torch.manual_seed(3)
    rebuilt = gt_module.build_psi_producer(_cfg())
    loaders._load_psi_producer_state(rebuilt, state["pe_model"], "ckpt/", _cfg())
    for k, v in trained.state_dict().items():
        assert torch.equal(v, rebuilt.state_dict()[k]), k


def test_multistage_carry_of_a_navigator_checkpoint(architecture="learnable_graph_mask"):
    """``init_pe_from`` (stage N+1) carries the WHOLE legacy TwoStagePE, both halves."""
    torch.manual_seed(0)
    trained = gt_module.build_psi_producer(_nav_cfg())
    torch.manual_seed(4)
    holder = _Holder(gt_module.build_psi_producer(_nav_cfg()))
    with tempfile.TemporaryDirectory() as td:
        torch.save(_saved_gnn_weights(trained), os.path.join(td, "gnn_weights.pt"))
        loaders.load_pe_weights_into(holder, td, architecture)
    for k, v in trained.state_dict().items():
        assert torch.equal(v, holder.pe_model.state_dict()[k]), f"{k} not carried"


def test_multistage_carry_of_a_navigator_checkpoint_wire():
    """Same carry for wire_llm. ε/σ/pe_gain are absent here (GT-shaped source), so the
    loader must report a cold start rather than raise — asserted via the holder having no
    wire tensors to load."""
    torch.manual_seed(0)
    trained = gt_module.build_psi_producer(_nav_cfg())
    torch.manual_seed(4)
    holder = _Holder(gt_module.build_psi_producer(_nav_cfg()))
    with tempfile.TemporaryDirectory() as td:
        torch.save(_saved_gnn_weights(trained), os.path.join(td, "gnn_weights.pt"))
        loaders.load_pe_weights_into(holder, td, "wire_llm")
    for k, v in trained.state_dict().items():
        assert torch.equal(v, holder.pe_model.state_dict()[k]), f"{k} not carried"


def test_multistage_topology_mismatch_raises():
    """Carrying a legacy TwoStagePE checkpoint into a PE-only Ψ run must raise."""
    torch.manual_seed(0)
    state = _saved_gnn_weights(gt_module.build_psi_producer(_nav_cfg()))
    holder = _Holder(gt_module.build_psi_producer(_cfg()))
    with tempfile.TemporaryDirectory() as td:
        torch.save(state, os.path.join(td, "gnn_weights.pt"))
        try:
            loaders.load_pe_weights_into(holder, td, "learnable_graph_mask")
        except RuntimeError as e:
            assert "semantic_gt_from" in str(e), str(e)
            return
    raise AssertionError("expected a RuntimeError on Ψ-producer topology mismatch")


# --------------------------------------------------------------- permutation

class _CaptureGT(nn.Module):
    """Deterministic stand-in for the PE GT that records the permutation it was given."""

    def __init__(self, n, d):
        super().__init__()
        self.n, self.d, self.seen = n, d, "unset"

    def forward(self, data, permutation=None, **kw):
        self.seen = permutation
        return torch.arange(self.n * self.d, dtype=torch.float).view(self.n, self.d)


class _CaptureHead(nn.Module):
    """Records the graph the head is fed (the point of the permutation fix)."""

    def forward(self, data, permutation=None):
        self.edge_index = data.edge_index
        self.x = data.x
        return data.x


def test_permutation_reaches_both_halves():
    """The head must see the SAME relabelled edge_index the PE GT was run on.

    Feeding it the original topology would mix a permuted-graph PE with unpermuted edges
    — a silent equivariance break that the eval would report as a model property.
    """
    g = _graph(5)
    pe_gt, head = _CaptureGT(5, _CFG["d_model"]), _CaptureHead()
    nav = gt_module.TwoStagePE(pe_gt, head)
    perm = Permutation(seed=7)

    nav(g, permutation=perm)
    assert perm is pe_gt.seen, "permutation not threaded to the PE GT"
    expected = perm.apply(g.edge_index, g.num_nodes)
    assert torch.equal(head.edge_index, expected), \
        "the head was fed the ORIGINAL topology while the PE was computed on the permuted one"
    assert not torch.equal(head.edge_index, g.edge_index), \
        "fixture is degenerate — this permutation does not move any edge"


def test_no_permutation_leaves_the_graph_untouched():
    g = _graph(5)
    head = _CaptureHead()
    nav = gt_module.TwoStagePE(_CaptureGT(5, _CFG["d_model"]), head)
    nav(g)
    assert torch.equal(head.edge_index, g.edge_index)
    assert head.x.shape == (5, _CFG["d_model"])


# ------------------------------------------------- config validation + wiring

def _base_gnn_cfg(arch, **over):
    """The real experiments/base_config.yaml, tiny-ified — so the test exercises the
    SHIPPED defaults and key names, not a hand-written stand-in."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("experiments/base_config.yaml")
    cfg.gnn.arch = arch
    cfg.gnn.pe_node_features = "random"
    for k, v in {**_CFG, **over}.items():
        cfg.gnn[k] = v
    return cfg


def test_validate_config_accepts_navigator_for_both_psi_archs_and_rejects_others():
    from prism.training import train_v3

    for arch in ("learnable_graph_mask", "wire_llm"):
        cfg = _base_gnn_cfg(arch)
        cfg.gnn.pe_gt_from = "path_navigator_gt.pt"
        cfg.gnn.semantic_gt_from = "path_navigator_agt.pt"
        train_v3._validate_config(cfg)             # must not raise

    cfg = _base_gnn_cfg("rpearl_llm")
    cfg.gnn.pe_gt_from = "path_navigator_gt.pt"
    try:
        train_v3._validate_config(cfg)
    except ValueError as e:
        assert "pe_gt_from" in str(e), str(e)
        return
    raise AssertionError("expected a ValueError for pe_gt_from on a non-Ψ-producer arch")


def test_validate_config_rejects_semantic_without_pe_gt():
    from prism.training import train_v3

    cfg = _base_gnn_cfg("wire_llm")
    cfg.gnn.semantic_gt_from = "path_navigator_agt.pt"
    try:
        train_v3._validate_config(cfg)
    except ValueError as e:
        assert "pe_gt_from" in str(e), str(e)
        return
    raise AssertionError("expected a ValueError for semantic_gt_from without pe_gt_from")


def _tiny_llama(hidden=32):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=64, hidden_size=hidden, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        attn_implementation="eager")).eval()


def _tiny_gemma4(num_layers=4):
    try:
        from transformers import Gemma4ForCausalLM, Gemma4TextConfig
    except Exception:  # noqa: BLE001 — family unavailable in this transformers build
        return None
    torch.manual_seed(0)
    return Gemma4ForCausalLM(Gemma4TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=num_layers,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, global_head_dim=16,
        max_position_embeddings=64, sliding_window=8, attn_implementation="eager")).eval()


class _StubCollator:
    """build_planner_model only wires attributes onto the collator; the real one needs a
    live tokenizer, which these construction tests deliberately avoid."""

    def __init__(self, *a, **kw):
        pass


def _build(arch, llm):
    """``architectures.build_planner_model`` with the collator stubbed out."""
    from prism.data import data as data_module
    from prism.models import architectures

    cfg = _base_gnn_cfg(arch)
    cfg.gnn.pe_gt_from = "path_navigator_gt.pt"
    cfg.gnn.semantic_gt_from = "path_navigator_agt.pt"
    real = data_module.SpineDataCollator
    data_module.SpineDataCollator = _StubCollator
    try:
        model, _ = architectures.build_planner_model(cfg.gnn, llm, tokenizer=None)
    finally:
        data_module.SpineDataCollator = real
    return model


def test_build_planner_model_uses_the_navigator_for_both_archs():
    """Both architectures must reach NavigatorPE through the same factory, and expose BOTH
    halves as structural parameters (the boosted-LR / freeze_pe group)."""
    for arch, llm in (("learnable_graph_mask", _tiny_llama()),
                      ("wire_llm", _tiny_gemma4())):
        if llm is None:
            _skip("gemma4 unavailable — wire_llm half not exercised")
            continue
        model = _build(arch, llm)
        assert isinstance(model.pe_model, gt_module.NavigatorPE), (arch, type(model.pe_model))
        structural = {id(p) for p in model.structural_parameters()}
        for half in ("pe_gt", "semantic_gt"):
            params = list(getattr(model.pe_model, half).parameters())
            assert params, f"{arch}: {half} has no parameters"
            assert all(id(p) in structural for p in params), \
                f"{arch}: {half} params are outside structural_parameters() — they would " \
                "miss the structural LR group and stay frozen under PEFT"


# =============================================================================
# NavigatorGT — the AGT stage vs the notebook's GNNShortestPathNavigator (cell 75)
# =============================================================================

# The notebook's create_gnn('gt') model_hparams, verbatim (cell 29). The head is built
# from THESE keys: node_feature_dim=d_model, num_layers, d_model, heads, dropout, k_gt.
_NB_HPARAMS = dict(num_layers=3, pe_hidden_channels=256, pe_num_layers=5, d_model=1024,
                   heads=8, num_samples=320, dropout=0.1, k_pe=3, k_gt=2, eps=1e-6,
                   use_layer_norm=True)

_SUITE8 = "outputs/e9_multistage_training/suite8/path_navigator.pt"


def _dev():
    """MPS locally, CUDA on the cluster, CPU last (project rule: not CPU by choice)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _navigator(**nav_kw):
    """Tiny NavigatorGT with the notebook's *structure* (only the widths shrunk)."""
    torch.manual_seed(0)
    return gt_module.NavigatorGT(
        gt_module.GraphTransformer(num_layers=_CFG["gt_num_layers"],
                                   pe_hidden_channels=_CFG["pe_hidden_channels"],
                                   pe_num_layers=_CFG["pe_num_layers"],
                                   d_model=_CFG["d_model"], heads=_CFG["gt_heads"],
                                   num_samples=_CFG["num_samples"], dropout=0.0,
                                   k_pe=_CFG["k_pe"], k_gt=_CFG["k_gt"], eps=_CFG["eps"],
                                   use_layer_norm=True),
        gt_module.SemanticGraphTransformer(node_feature_dim=_CFG["d_model"],
                                           d_model=_CFG["d_model"],
                                           num_layers=_CFG["gt_num_layers"],
                                           heads=_CFG["gt_heads"], dropout=0.0,
                                           k_gt=_CFG["k_gt"]),
        **nav_kw).eval()


def _line_graph(n, device):
    """Path graph 0-1-...-(n-1): every hop distance is exactly |i - j|."""
    src, dst = list(range(n - 1)), list(range(1, n))
    ei = torch.tensor([src + dst, dst + src], dtype=torch.long, device=device)
    g = Data(x=torch.zeros(n, 1, device=device), edge_index=ei, num_nodes=n)
    return g


# ------------------------------------------------------------ construction parity

def test_navigator_gt_head_matches_the_notebook_hparams():
    """The head is SemanticGraphTransformer(node_feature_dim=d_model, num_layers, d_model,
    heads, dropout, k_gt) read off ONE model_hparams dict — notebook cell 75."""
    head = gt_module.SemanticGraphTransformer(
        node_feature_dim=_NB_HPARAMS["d_model"], num_layers=_NB_HPARAMS["num_layers"],
        d_model=_NB_HPARAMS["d_model"], heads=_NB_HPARAMS["heads"],
        dropout=_NB_HPARAMS["dropout"], k_gt=_NB_HPARAMS["k_gt"])
    import yaml
    with open("experiments/e9_navigator_gt.yaml") as f:
        semantic = yaml.safe_load(f)["navigator"]["semantic"]
    cfg_head = gt_module.SemanticGraphTransformer(**semantic)
    assert set(head.state_dict()) == set(cfg_head.state_dict()), \
        "the shipped YAML `semantic` block does not rebuild the notebook head"
    for k, v in head.state_dict().items():
        assert v.shape == cfg_head.state_dict()[k].shape, k


def test_navigator_gt_shape_is_the_pe_gt_d_model():
    """Notebook: self.shape = gnn.out_features, and create_gnn('gt') sets
    `gnn.out_features = gnn.d_model`. Repo: pe_gt.d_model. Same number by construction."""
    nav = _navigator()
    assert nav.shape == nav.pe_gt.d_model == _CFG["d_model"]
    assert nav.classifier.in_features == nav.shape and nav.classifier.out_features == 1
    assert isinstance(nav.classifier, nn.Linear)
    # And the head consumes exactly that width.
    assert nav.semantic_gt.input_proj.in_features == nav.shape


# ---------------------------------------------------------------- forward parity

def test_navigator_gt_forward_is_classifier_of_head_of_x_plus_pe():
    """forward == classifier(head(Data(x = graph.x + cached_pe, edge_index))) — cell 75."""
    device = _dev()
    nav = _navigator().to(device)
    g = _line_graph(6, device)
    g.x = torch.randn(6, nav.shape, device=device)
    with torch.no_grad():
        out = nav(g)
        assert out.shape == (6, 1)
        feed = Data(x=g.x + nav.cached_pe, edge_index=g.edge_index)
        expected = nav.classifier(nav.semantic_gt(feed))
    assert torch.allclose(out, expected, atol=1e-5), (out - expected).abs().max()


def test_navigator_gt_pe_cache_follows_graph_identity():
    """The notebook re-uses the PE while the SAME Data object is fed (probes ignore x);
    a new graph object, or invalidate_cache(), forces a re-sample."""
    device = _dev()
    nav = _navigator().to(device)
    g = _line_graph(6, device)
    g.x = torch.randn(6, nav.shape, device=device)
    with torch.no_grad():
        nav(g)
        first = nav.cached_pe
        g.x = torch.randn(6, nav.shape, device=device)   # re-seed: PE must NOT move
        nav(g)
        assert nav.cached_pe is first
        nav.invalidate_cache()
        nav(g)
    assert nav.cached_pe is not first


# --------------------------------------------------------------- generate parity

def test_generate_returns_a_simple_path_honouring_visited_and_the_hop_mask():
    device = _dev()
    nav = _navigator(mask_hops=2).to(device)
    n = 10
    g = _line_graph(n, device)
    with torch.no_grad():
        out = nav.generate(g, 0, n - 1)
    assert out.dim() == 2 and out.shape[1] == 1, out.shape
    path = out.view(-1).tolist()
    assert path[0] == 0
    assert len(set(path)) == len(path), f"visited set violated: {path}"
    assert len(path) - 1 <= min(nav.MAX_LENGTH, n - 1), path
    # Blurry vision: every emitted hop is within MASK_HOPS BFS hops (|i - j| on a line).
    assert all(abs(b - a) <= nav.MASK_HOPS for a, b in zip(path, path[1:])), path


def test_generate_restores_the_cauchy_prior_and_caches_hops():
    """Notebook cleanup: graph.x is handed back as Cauchy(base_loc, CAUCHY_SCALE) [N, 1];
    graph.hops is built once and reused."""
    device = _dev()
    nav = _navigator().to(device)
    g = _line_graph(7, device)
    assert not hasattr(g, "hops") or g.hops is None
    with torch.no_grad():
        nav.generate(g, 0, 6)
    assert g.x.shape == (7, 1), g.x.shape
    hops = g.hops
    assert torch.equal(hops[0], torch.arange(7, dtype=hops.dtype, device=hops.device))
    with torch.no_grad():
        nav.generate(g, 1, 5)
    assert g.hops is hops, "the topology-only hop matrix must not be rebuilt per rollout"


def test_generate_start_equals_goal_is_a_single_node():
    device = _dev()
    nav = _navigator().to(device)
    with torch.no_grad():
        out = nav.generate(_line_graph(5, device), 3, 3)
    assert out.view(-1).tolist() == [3]


def test_generate_on_degenerate_graphs_is_finite_and_terminates():
    """Isolated node, unreachable goal, N = 1: no inf/NaN, no infinite loop."""
    device = _dev()
    nav = _navigator().to(device)

    # N = 1 -> max_hops = 0, the loop must not run.
    single = Data(x=torch.zeros(1, 1, device=device),
                  edge_index=torch.empty(2, 0, dtype=torch.long, device=device), num_nodes=1)
    with torch.no_grad():
        assert nav.generate(single, 0, 0).view(-1).tolist() == [0]

    # Two disjoint components: goal unreachable -> walk stops, no inf leaks into the path.
    ei = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long, device=device)
    split = Data(x=torch.zeros(4, 1, device=device), edge_index=ei, num_nodes=4)
    with torch.no_grad():
        path = nav.generate(split, 0, 3).view(-1).tolist()
    assert path[0] == 0 and set(path) <= {0, 1}, path
    assert len(set(path)) == len(path)

    # Isolated node as the start: its only in-mask node is itself, which is visited.
    ei = torch.tensor([[1, 2], [2, 1]], dtype=torch.long, device=device)
    iso = Data(x=torch.zeros(3, 1, device=device), edge_index=ei, num_nodes=3)
    with torch.no_grad():
        assert iso is not None and nav.generate(iso, 0, 2).view(-1).tolist() == [0]


# ------------------------------------------------- the tag-distribution knobs

def test_tag_std_reproduces_the_notebook_precedence_slip_and_is_overridable():
    """`self.STD = target_var ** 1/2` is (target_var ** 1) / 2 = 0.005, NOT sqrt(0.01)=0.1.

    Matched deliberately (the instruction is to match the notebook CLASS); the suite8
    TRAINING loop used 0.1, so `tag_scale` exists to decode at the fitted scale.
    """
    assert _navigator().STD == 0.005
    assert _navigator(target_var=0.04).STD == 0.02          # 0.04/2, not sqrt(0.04)=0.2
    assert _navigator(tag_scale=0.1).STD == 0.1             # the training-loop scale
    # `df` still parameterises the (back-compat) Student's-T arm.
    assert abs(_navigator().SCALE - (0.01 * 3.0 / 5.0) ** 0.5) < 1e-12


def test_tag_dist_default_is_normal_and_studentt_stays_available():
    assert _navigator().TAG_DIST == "normal"
    assert _navigator(tag_dist="studentt").TAG_DIST == "studentt"
    try:
        _navigator(tag_dist="cauchy")
    except ValueError as e:
        assert "tag_dist" in str(e)
        return
    raise AssertionError("expected a ValueError naming tag_dist")


def test_tag_knobs_actually_reach_the_seeded_features():
    """The knobs must fork the SEED, not just sit on the object.

    The goal row is ``tag(goal_loc) + sin(pe_phase)``; subtracting the deterministic step
    code leaves a draw whose spread IS the tag scale, so the two configs are separated by
    the scale ratio rather than by the PE's own O(0.5) spread.
    """
    device = _dev()
    g = _line_graph(6, device)
    D = _CFG["d_model"]
    pe_phase = torch.sin((torch.arange(D, device=device) % 2) * (torch.pi / 2))
    spreads = []
    for scale in (0.005, 5.0):
        nav = _navigator(tag_scale=scale).to(device)
        seen = {}

        def _spy(graph, permutation=None, _seen=seen):
            _seen.setdefault("x", graph.x.clone())
            return torch.zeros(graph.num_nodes, 1, device=graph.x.device)

        nav.forward = _spy                      # isolate the seeding from the model
        with torch.no_grad():
            nav.generate(g, 0, 5)
        spreads.append(float((seen["x"][5] - pe_phase).std()))
    assert spreads[1] > 50 * spreads[0], spreads


def test_generate_studentt_arm_still_produces_a_valid_path():
    """Back-compat: configs that set df/target_var and tag_dist=studentt keep working."""
    device = _dev()
    nav = _navigator(tag_dist="studentt", mask_hops=2).to(device)
    with torch.no_grad():
        path = nav.generate(_line_graph(8, device), 0, 7).view(-1).tolist()
    assert path[0] == 0 and len(set(path)) == len(path)


def test_nav_decode_keys_cover_every_decode_switch_and_the_yaml_loads():
    """_NAV_DECODE_KEYS is the single audited switch list: every non-module __init__ arg
    of NavigatorGT must be routable from the YAML, and every YAML key must be accepted."""
    import inspect
    import yaml
    from prism.eval import scalability_evaluation as se

    params = set(inspect.signature(gt_module.NavigatorGT.__init__).parameters)
    params -= {"self", "pe_gt", "semantic_gt"}
    assert params == set(se._NAV_DECODE_KEYS), params ^ set(se._NAV_DECODE_KEYS)

    with open("experiments/e9_navigator_gt.yaml") as f:
        cfg = yaml.safe_load(f)["navigator"]
    decode = {k: cfg[k] for k in se._NAV_DECODE_KEYS if k in cfg}
    nav = _navigator(**decode)
    assert nav.TAG_DIST == "normal" and nav.STD == 0.005 and nav.MASK_HOPS == 3


# -------------------------------------------------- the real suite8 checkpoint

def test_from_pretrained_strict_loads_the_real_suite8_navigator():
    """The shipped YAML must rebuild suite8's path_navigator.pt with NO missing/unexpected
    keys — otherwise every navigator number is from a partially random model."""
    import yaml
    if not os.path.exists(_SUITE8):
        return _skip(f"{_SUITE8} not present")
    with open("experiments/e9_navigator_gt.yaml") as f:
        cfg = yaml.safe_load(f)["navigator"]
    from prism.eval import scalability_evaluation as se
    decode = {k: cfg[k] for k in se._NAV_DECODE_KEYS if k in cfg}
    nav = gt_module.NavigatorGT.from_pretrained(          # raises on any key mismatch
        _SUITE8, gt_kwargs=cfg["gt"], semantic_kwargs=cfg["semantic"], **decode)
    raw = torch.load(_SUITE8, map_location="cpu")
    assert len(raw) == len(nav.state_dict()), (len(raw), len(nav.state_dict()))
    assert nav.shape == cfg["gt"]["d_model"] == cfg["semantic"]["node_feature_dim"]
    # The notebook's classifier is Linear(shape, 1) — pin the loaded shapes.
    assert nav.classifier.weight.shape == (1, nav.shape)
    assert torch.equal(nav.classifier.weight, raw["classifier.weight"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"--- {name}")
            fn()
    print("all navigator PE tests passed")
