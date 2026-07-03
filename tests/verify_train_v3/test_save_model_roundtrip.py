"""DL-mode verification of ``GraphSFTTrainer.save_model`` checkpoint round-trips and the
``create_optimizer`` ``get_optimizer_cls_and_kwargs`` TypeError-fallback branch
(``prism.training.train_v3``).

WHAT IS UNDER TEST
  ``GraphSFTTrainer.save_model`` (train_v3.py:551-624) serialises the graph-side weights to
  ``gnn_weights.pt`` (+ ``rpearl_weights.pt`` where applicable) and ``train_config.json``,
  branching by ``gnn_config['architecture']``. The eval boundary later reloads those files via
  ``prism.models.loaders.graph_augmented_llm_from_pretrained``. The contract is the SAVE↔LOAD
  ROUND-TRIP: the keys ``save_model`` writes, and the live submodules it reads them from, must
  exactly match the keys/submodules the loader reads them back into — otherwise a checkpoint is
  silently corrupted (the loader uses ``strict=False`` on ``pe_model``/``gt_model``, so a key
  mismatch drops weights silently rather than erroring).

THE ORACLE (independent of save_model's body)
  Derived from ``loaders.py`` — the CONSUMER. For each architecture we mirror the loader's exact
  load sequence into a FRESH, differently-seeded model B, then assert every persisted submodule of
  B now equals model A (which produced the checkpoint). A "no-op guard" first asserts A and B
  differ pre-load, so post-load equality proves the bytes actually transferred (catching the
  strict=False silent-drop failure mode). We never copy save_model's own logic into the assertions.

  Loader contract (loaders.py:123-268):
    rpearl_llm     : key 'pe_model' -> model.pe_model ; 'pe_proj','pe_gain', opt 'pe_norm'
    rpearl_gt_llm  : key 'gt_model' -> model.pe_model ; 'pe_proj','pe_gain', opt 'pe_norm'
    gt_llm         : key 'pe_model' -> model.pe_model ; 'pe_proj','pe_gain', opt 'pe_norm'
    graph_mask_llm : NO weights; rebuilt from the saved config (mask_* keys)

FIXTURE
  Tiny random-init Gemma4Unified base (q/k-norm, single-tensor RoPE, sliding windows, KV-shared
  layers — stands in for Gemma 4 12B), CPU, fixed seed. Models are built with the SAME real
  constructors ``architectures.build_planner_model`` uses. ``save_model``'s LoRA-adapter branch
  (``self.model.peft_config is not None``) is intentionally NOT exercised here — these models are
  un-PEFT-wrapped, so getattr returns None and the branch is skipped; the adapter save is a PEFT
  boundary, separate from the GNN-weight serialisation under test.

"""
import json
import os
import sys
from types import ModuleType, SimpleNamespace

sys.path.insert(0, "src")

import torch


def _skip(msg):
    try:
        import pytest
        pytest.skip(msg, allow_module_level=False)
    except Exception:
        print(f"[SKIP] {msg}")
    return "SKIPPED"


# --- Boundary stub: env lacks optional `liger_kernel`, which `trl` imports at module load.
# train_v3 only needs trl's SFTConfig/SFTTrainer classes; inject a minimal fake so the
# orchestration code under test imports. Stubs a boundary; changes no behaviour exercised below.
if "liger_kernel" not in sys.modules:
    class _LigerDummy:
        pass

    def _liger_getattr(attr):
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        return _LigerDummy

    import importlib.machinery
    for _name in ("liger_kernel", "liger_kernel.transformers"):
        _m = ModuleType(_name)
        _m.__file__ = "<stub>"
        _m.__spec__ = importlib.machinery.ModuleSpec(_name, loader=None)
        _m.__path__ = []
        _m.__getattr__ = _liger_getattr
        sys.modules[_name] = _m

from prism.models import gnn_llm
from prism.models import loaders as model_loaders
from prism.models import r_pearl as r_pearl_module
from prism.models import gt as gt_module
from prism.training.train_v3 import GraphSFTTrainer

_SCRATCH = os.environ.get("TMPDIR", "/tmp")
_H = 32  # tiny Gemma4 hidden size (text width fed to word-embedding node features)


def _gemma4_missing():
    try:
        from transformers import Gemma4UnifiedForCausalLM, Gemma4UnifiedTextConfig  # noqa: F401
        return None
    except Exception as e:  # noqa: BLE001
        return f"gemma4_unified unavailable: {e}"


def _tiny_llm(seed):
    from transformers import Gemma4UnifiedForCausalLM, Gemma4UnifiedTextConfig
    torch.manual_seed(seed)
    cfg = Gemma4UnifiedTextConfig(
        vocab_size=64, hidden_size=_H, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=64, attn_implementation="eager")
    return Gemma4UnifiedForCausalLM(cfg)


# --- Model factories: the SAME constructor calls architectures.build_planner_model makes. ---
def _gt(seed):
    torch.manual_seed(seed)
    return gt_module.GraphTransformer(
        num_layers=2, pe_hidden_channels=16, pe_num_layers=2, d_model=24, heads=2,
        num_samples=8, dropout=0.1, k_pe=2, k_gt=2, eps=1e-6, use_layer_norm=True,
        probe_distribution="gaussian", max_gather_rows=200000,
        fixed_seed_mode=False, fixed_seed_value=0, pe_readout="second_moment",
        center_second_moment=True)


def _build(arch, seed, use_pe_norm=True):
    """Build one tiny model for ``arch`` (random-init at ``seed``)."""
    torch.manual_seed(seed)
    if arch == "rpearl_llm":
        pe = r_pearl_module.RandomGNNPositionalEncodings(
            pe_hidden_channels=16, pe_num_layers=2, d_model=24, num_samples=8,
            dropout=0.1, k=2, eps=1e-6, use_layer_norm=True)
        return gnn_llm.GraphAugmentedLLM(_tiny_llm(seed), pe, d_model=24, eps=1e-6,
                                         pe_gain_init=0.0, use_pe_norm=use_pe_norm)
    if arch == "rpearl_gt_llm":
        return gnn_llm.GraphAugmentedLLM(_tiny_llm(seed), _gt(seed), d_model=24, eps=1e-6,
                                         pe_gain_init=0.0, use_pe_norm=use_pe_norm)
    if arch == "gt_llm":
        torch.manual_seed(seed)
        sgt = gt_module.SemanticGraphTransformer(
            node_feature_dim=_H, d_model=24, num_layers=2, heads=2, dropout=0.1, k_gt=2)
        return gnn_llm.GraphAugmentedLLM(_tiny_llm(seed), sgt, d_model=24, eps=1e-6,
                                         pe_gain_init=0.0, use_pe_norm=use_pe_norm,
                                         pe_node_features="word_embeddings")
    if arch == "graph_mask_llm":
        return gnn_llm.GraphMaskLLM(_tiny_llm(seed), k_hops=1, symmetrize=True, use_edges=True)
    raise ValueError(arch)


def _perturb_structural(model):
    """Push every persisted graph-side param off its init so a dropped load is detectable."""
    torch.manual_seed(999)
    with torch.no_grad():
        for p in model.structural_parameters():
            p.add_(torch.randn_like(p))
        if getattr(model, "pe_norm", None) is not None:
            for p in model.pe_norm.parameters():
                p.add_(torch.randn_like(p))


def _trainer_for(model, gnn_config, out_dir):
    """A GraphSFTTrainer instance carrying just the state save_model reads (no SFTTrainer.__init__)."""
    t = object.__new__(GraphSFTTrainer)
    t.model = model
    t.gnn_config = gnn_config
    t.args = SimpleNamespace(output_dir=out_dir)
    return t


def _sd_equal(a, b):
    ka, kb = set(a.keys()), set(b.keys())
    if ka != kb:
        return False
    return all(torch.equal(a[k], b[k]) for k in ka)


def _outdir(name):
    d = os.path.join(_SCRATCH, "save_model_rt", name)
    os.makedirs(d, exist_ok=True)
    # clean any prior artifacts so absence-assertions are meaningful
    for f in ("train_config.json", "gnn_config.json", "gnn_weights.pt", "rpearl_weights.pt"):
        p = os.path.join(d, f)
        if os.path.exists(p):
            os.remove(p)
    return d


# ==========================================================================
# Round-trip: rpearl_llm / gt_llm  (loader 'else' & 'gt_llm' branches; key 'pe_model')
# ==========================================================================
def _roundtrip_pe_model_key(arch):
    """save_model writes the PE under key 'pe_model'; loader reads 'pe_model'->model.pe_model.
    Mirror loaders.py:262-268 (resp. 167-173) into a fresh model B and assert full transfer."""
    A = _build(arch, seed=0)
    _perturb_structural(A)
    out = _outdir(arch)
    _trainer_for(A, {"architecture": arch}, out).save_model(output_dir=out)

    assert os.path.exists(os.path.join(out, "gnn_weights.pt"))
    assert not os.path.exists(os.path.join(out, "rpearl_weights.pt")), \
        f"{arch} must not emit rpearl_weights.pt (only rpearl_gt_llm does)"
    w = torch.load(os.path.join(out, "gnn_weights.pt"), map_location="cpu")
    assert set(w.keys()) == {"pe_model", "pe_proj", "pe_gain", "pe_norm"}, \
        f"{arch} unexpected gnn_weights keys: {sorted(w.keys())}"

    B = _build(arch, seed=1)                          # different init
    assert not _sd_equal(A.pe_model.state_dict(), B.pe_model.state_dict())  # no-op guard
    # --- loader protocol (loaders.py) ---
    B.pe_model.load_state_dict(w["pe_model"], strict=False)
    B.pe_proj.load_state_dict(w["pe_proj"])
    B.pe_gain.data.copy_(w["pe_gain"])
    B.pe_norm.load_state_dict(w["pe_norm"])
    # --- round-trip equality ---
    assert _sd_equal(A.pe_model.state_dict(), B.pe_model.state_dict()), "pe_model failed to round-trip"
    assert _sd_equal(A.pe_proj.state_dict(), B.pe_proj.state_dict()), "pe_proj failed to round-trip"
    assert torch.equal(A.pe_gain.data, B.pe_gain.data), "pe_gain failed to round-trip"
    assert _sd_equal(A.pe_norm.state_dict(), B.pe_norm.state_dict()), "pe_norm failed to round-trip"


def test_roundtrip_rpearl_llm():
    if _gemma4_missing():
        return _skip(_gemma4_missing())
    _roundtrip_pe_model_key("rpearl_llm")


def test_roundtrip_gt_llm():
    if _gemma4_missing():
        return _skip(_gemma4_missing())
    _roundtrip_pe_model_key("gt_llm")


# ==========================================================================
# Round-trip: rpearl_gt_llm  (loader key rename 'gt_model' -> model.pe_model)
# ==========================================================================
def test_roundtrip_rpearl_gt_llm():
    """save_model stores the GraphTransformer under key 'gt_model'; loader reads 'gt_model' INTO
    model.pe_model (loaders.py:145). Also emits rpearl_weights.pt (inner R-PEARL)."""
    if _gemma4_missing():
        return _skip(_gemma4_missing())
    arch = "rpearl_gt_llm"
    A = _build(arch, seed=0)
    _perturb_structural(A)
    out = _outdir(arch)
    _trainer_for(A, {"architecture": arch}, out).save_model(output_dir=out)

    assert os.path.exists(os.path.join(out, "rpearl_weights.pt"))
    w = torch.load(os.path.join(out, "gnn_weights.pt"), map_location="cpu")
    assert set(w.keys()) == {"gt_model", "pe_proj", "pe_gain", "pe_norm"}, \
        f"unexpected gnn_weights keys: {sorted(w.keys())}"
    rp = torch.load(os.path.join(out, "rpearl_weights.pt"), map_location="cpu")
    assert set(rp.keys()) == {"rpearl"}

    B = _build(arch, seed=1)
    assert not _sd_equal(A.pe_model.state_dict(), B.pe_model.state_dict())  # no-op guard
    B.pe_model.load_state_dict(w["gt_model"], strict=False)   # loader: 'gt_model' -> pe_model
    B.pe_proj.load_state_dict(w["pe_proj"])
    B.pe_gain.data.copy_(w["pe_gain"])
    B.pe_norm.load_state_dict(w["pe_norm"])
    assert _sd_equal(A.pe_model.state_dict(), B.pe_model.state_dict()), "GT failed to round-trip via 'gt_model'"
    # rpearl_weights.pt carries the inner R-PEARL (model.pe_model.pe_model) for analysis/reuse.
    assert _sd_equal(A.pe_model.pe_model.state_dict(), rp["rpearl"]), "inner R-PEARL mismatch in rpearl_weights.pt"


# ==========================================================================
# graph_mask_llm — parameter-free: only the config file, NO weight files
# ==========================================================================
def test_graph_mask_writes_only_config_no_weights():
    """save_model graph_mask branch is a no-op for weights; the mask is rebuilt from
    the saved mask_* keys. With no LoRA adapter attached, the only artifact is
    train_config.json (metadata top-level, params under "gnn")."""
    if _gemma4_missing():
        return _skip(_gemma4_missing())
    A = _build("graph_mask_llm", seed=0)
    cfg = {"architecture": "graph_mask_llm", "mask_k_hops": 2,
           "mask_symmetrize": True, "mask_use_edges": False}
    out = _outdir("graph_mask_llm")
    _trainer_for(A, cfg, out).save_model(output_dir=out)

    assert not os.path.exists(os.path.join(out, "gnn_weights.pt")), "graph_mask must emit no weights"
    assert not os.path.exists(os.path.join(out, "rpearl_weights.pt"))
    assert not os.path.exists(os.path.join(out, "gnn_config.json")), \
        "new saves must not write the legacy gnn_config.json"
    with open(os.path.join(out, "train_config.json")) as f:
        loaded = json.load(f)
    assert loaded == {"architecture": "graph_mask_llm",
                      "gnn": {"mask_k_hops": 2, "mask_symmetrize": True,
                              "mask_use_edges": False}}
    # ...and the production reader flattens it back to exactly the trainer's dict.
    assert model_loaders.load_gnn_config(out) == cfg


# ==========================================================================
# train_config.json is always written and round-trips through the production reader
# ==========================================================================
def test_saved_config_roundtrips_through_load_gnn_config():
    """Every save_model branch writes train_config.json first; loaders.load_gnn_config
    must flatten it back to exactly the trainer's gnn_config dict (the eval boundary
    reads base_model/architecture/rebuild hyperparams from it)."""
    if _gemma4_missing():
        return _skip(_gemma4_missing())
    A = _build("rpearl_llm", seed=0)
    cfg = {"architecture": "rpearl_llm", "base_model": "google/gemma-4-12B-it",
           "d_model": 24, "pe_hidden_channels": 16, "eps": 1e-6, "use_pe_norm": True}
    out = _outdir("cfgjson")
    _trainer_for(A, cfg, out).save_model(output_dir=out)
    assert model_loaders.load_gnn_config(out) == cfg


def test_load_gnn_config_reads_legacy_flat_format(tmp_path=None):
    """Legacy checkpoints (flat gnn_config.json, no train_config.json) still load."""
    import tempfile
    d = tempfile.mkdtemp(prefix="legacy_cfg_")
    cfg = {"architecture": "rpearl_gt_llm", "base_model": "google/gemma-4-31B-it",
           "d_model": 2048, "text_edge_list": "none"}
    with open(os.path.join(d, "gnn_config.json"), "w") as f:
        json.dump(cfg, f)
    assert model_loaders.load_gnn_config(d) == cfg


# ==========================================================================
# create_optimizer — the get_optimizer_cls_and_kwargs(args, model) TypeError fallback
# ==========================================================================
def test_create_optimizer_typeerror_fallback_to_single_arg():
    """train_v3.py:522-529 calls get_optimizer_cls_and_kwargs(args, opt_model) and, on TypeError
    (older transformers whose signature took only `args`), retries with (args). Simulate the old
    signature: the 2-arg form raises TypeError, the 1-arg form succeeds. The boosted two-LR
    grouping must still be produced — i.e. the fallback branch is reached and functional."""
    if _gemma4_missing():
        return _skip(_gemma4_missing())
    from trl import SFTConfig
    from peft import LoraConfig, get_peft_model

    base_lr, mult = 2e-4, 3.0
    model = _build("rpearl_llm", seed=0)
    model = get_peft_model(model, LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    for p in model.structural_parameters():               # mirror GraphSFTTrainer.__init__ re-enable
        p.requires_grad = True

    t = object.__new__(GraphSFTTrainer)
    t.model = model
    t.gnn_config = {"structural_lr_mult": mult}
    t.optimizer = None
    t.args = SFTConfig(output_dir=os.path.join(_SCRATCH, "create_opt_fallback"),
                       learning_rate=base_lr, weight_decay=0.05, optim="adamw_torch",
                       report_to=[], bf16=False, fp16=False)

    real = GraphSFTTrainer.get_optimizer_cls_and_kwargs   # the real (static) impl

    def _only_one_arg(*a, **k):
        # 2+ args (the args, model form) -> emulate the old transformers signature error.
        if len(a) + len(k) >= 2:
            raise TypeError("get_optimizer_cls_and_kwargs() takes 1 positional argument (old API)")
        return real(a[0])
    t.get_optimizer_cls_and_kwargs = _only_one_arg

    opt = t.create_optimizer()                            # must NOT raise; uses the 1-arg fallback
    lrs = {round(g["lr"], 12) for g in opt.param_groups}
    assert lrs == {round(base_lr, 12), round(base_lr * mult, 12)}, \
        f"fallback path lost the two-LR grouping: {sorted(lrs)}"
    struct_ids = {id(p) for p in model.structural_parameters() if p.requires_grad}
    boosted = {id(p) for g in opt.param_groups if round(g["lr"], 12) == round(base_lr * mult, 12)
               for p in g["params"]}
    assert boosted == struct_ids, "structural params not in the boosted group after fallback"


# ==========================================================================
# Standalone runner (pytest absent from the env that carries trl/transformers)
# ==========================================================================
if __name__ == "__main__":
    passed, failed, skipped = 0, [], 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                r = fn()
                if r == "SKIPPED":
                    skipped += 1
                    print(f"{name}: SKIP")
                else:
                    passed += 1
                    print(f"{name}: PASS")
            except Exception as e:  # noqa: BLE001 — report, don't abort the suite
                failed.append((name, f"{type(e).__name__}: {e}"))
                print(f"{name}: FAIL — {type(e).__name__}: {e}")
    print(f"\n{passed} passed, {len(failed)} failed, {skipped} skipped")
    for name, err in failed:
        print(f"  FAIL {name}: {err}")
    sys.exit(1 if failed else 0)
