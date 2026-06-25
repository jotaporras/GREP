"""DL-mode verification of ``GraphSFTTrainer.create_optimizer`` (prism.training.train_v3).

Contract (from the method docstring, restated independently):
  Two learning-rate groups —
    * the *structural* path (``model.structural_parameters()`` = graph encoder + Ψ→hidden
      projection + injection gate) at ``structural_lr_mult × base_lr``;
    * everything else trainable (LLM/LoRA, and the deliberately-excluded ``pe_norm``) at
      ``base_lr``.
  Each LR group is further split into decay / no-decay (no-decay = norm/bias params, at
  ``weight_decay=0``). Only ``requires_grad`` params are included. When the multiplier is
  ``1.0`` (or the architecture is parameter-free → empty structural set), it falls back to
  the stock single-LR optimizer.

We exercise the REAL ``create_optimizer`` against a REAL tiny graph model. Because the full
``SFTTrainer.__init__`` needs a real tokenizer + tokenized dataset (network/offline-blocked
on CPU), we construct the trainer object directly and reproduce the post-``__init__`` STATE
that ``create_optimizer`` consumes: a PEFT-wrapped ``GraphAugmentedLLM`` whose structural
params have had ``requires_grad`` re-enabled (exactly as ``GraphSFTTrainer.__init__`` does
after PEFT freezes them). The grouping logic itself is the real method under test — not
reimplemented here.

Base-LLM fixture: a tiny random-init ``Gemma4Unified`` standing in for Gemma 4 12B (q/k-norm,
single-tensor RoPE, sliding windows, KV-shared layers). CPU, fixed seed.

Run:  conda run -n GREP-PRISM uv run python tests/verify_train_v3/test_create_optimizer.py
      (uv/conda share one env here; pytest is absent → standalone runner footer.)
"""
import os
import sys
from types import ModuleType

sys.path.insert(0, "src")

import torch
from torch import nn


def _skip(msg):
    """Skip under pytest; print [SKIP] and bail when run as a script."""
    try:
        import pytest
        pytest.skip(msg, allow_module_level=False)
    except Exception:
        print(f"[SKIP] {msg}")
    return None


# --- Boundary stub: this env lacks optional `liger_kernel`, which `trl` imports at module
# load. train_v3 only needs trl's SFTConfig/SFTTrainer classes, so inject a minimal fake. ---
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

from trl import SFTConfig
from peft import LoraConfig, get_peft_model

from prism.models import gnn_llm
from prism.models import r_pearl as r_pearl_module
from prism.training.train_v3 import GraphSFTTrainer

_BASE_LR = 2e-4
_WD = 0.05
_SCRATCH = os.environ.get("TMPDIR", "/tmp")


def _tiny_graph_model():
    """Tiny rpearl ``GraphAugmentedLLM`` on a random-init Gemma4Unified base (CPU)."""
    from transformers import Gemma4UnifiedForCausalLM, Gemma4UnifiedTextConfig

    torch.manual_seed(0)
    cfg = Gemma4UnifiedTextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=64, attn_implementation="eager")
    llm = Gemma4UnifiedForCausalLM(cfg)
    pe = r_pearl_module.RandomGNNPositionalEncodings(
        pe_hidden_channels=16, pe_num_layers=2, d_model=24, num_samples=8,
        dropout=0.1, k=2, eps=1e-6, use_layer_norm=True)
    return gnn_llm.GraphAugmentedLLM(llm, pe, d_model=24, eps=1e-6, pe_gain_init=0.0,
                                     use_pe_norm=True)


def _make_trainer(mult, *, reenable_structural=True, freeze_param=None):
    """Build the post-__init__ STATE create_optimizer reads, then a real GraphSFTTrainer
    instance (without running SFTTrainer.__init__). Returns (trainer, model).

    reenable_structural mirrors GraphSFTTrainer.__init__'s re-enable of requires_grad on
    structural_parameters()+pe_norm (PEFT freezes them on wrap). freeze_param lets a test
    keep one structural param frozen to check the requires_grad filter.
    """
    model = _tiny_graph_model()
    lora = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)   # freezes base (incl. structural) params

    if reenable_structural:
        for p in model.structural_parameters():
            p.requires_grad = True
        if getattr(model, "pe_norm", None) is not None:
            for p in model.pe_norm.parameters():
                p.requires_grad = True
    if freeze_param is not None:
        freeze_param(model).requires_grad = False

    args = SFTConfig(
        output_dir=os.path.join(_SCRATCH, "create_opt_test"),
        learning_rate=_BASE_LR, weight_decay=_WD,
        optim="adamw_torch",   # CPU-safe (real run uses adamw_torch_fused on GPU)
        report_to=[], bf16=False, fp16=False,
    )

    trainer = object.__new__(GraphSFTTrainer)
    trainer.model = model
    trainer.gnn_config = {"structural_lr_mult": mult}
    trainer.optimizer = None
    trainer.optimizer_cls_and_kwargs = None   # force HF fallback to build its own groups
    trainer.args = args
    return trainer, model


def _group_ids_by_lr(optimizer):
    """{lr: set(id(param))} across all param_groups."""
    by_lr = {}
    for g in optimizer.param_groups:
        by_lr.setdefault(round(g["lr"], 12), set()).update(id(p) for p in g["params"])
    return by_lr


# ==========================================================================
# Boosted two-LR grouping (the headline contract)
# ==========================================================================
def test_structural_params_land_in_boosted_lr_group_only():
    """mult=3.0 ⇒ every trainable structural param sits at 3×base LR and NOWHERE else;
    every other trainable param (LoRA, pe_norm) sits at base LR and NOWHERE else."""
    if _gemma4_missing():
        return _skip(_gemma4_missing())
    mult = 3.0
    trainer, model = _make_trainer(mult)
    opt = trainer.create_optimizer()

    struct_ids = {id(p) for p in model.structural_parameters() if p.requires_grad}
    other_ids = {id(p) for n, p in model.named_parameters()
                 if p.requires_grad and id(p) not in struct_ids}
    assert struct_ids and other_ids, "fixture must have both structural and other trainable params"

    by_lr = _group_ids_by_lr(opt)
    boosted = round(_BASE_LR * mult, 12)
    base = round(_BASE_LR, 12)
    assert set(by_lr.keys()) == {boosted, base}, f"unexpected LR set: {sorted(by_lr)}"
    assert by_lr[boosted] == struct_ids, "boosted group must be exactly the structural params"
    assert by_lr[base] == other_ids, "base group must be exactly the non-structural trainable params"


def test_pe_norm_stays_at_base_lr_not_boosted():
    """Docstring invariant: pe_norm is excluded from the structural group → base LR."""
    if _gemma4_missing():
        return _skip(_gemma4_missing())
    mult = 4.0
    trainer, model = _make_trainer(mult)
    opt = trainer.create_optimizer()
    pe_norm_ids = {id(p) for p in model.pe_norm.parameters()}
    by_lr = _group_ids_by_lr(opt)
    assert pe_norm_ids <= by_lr[round(_BASE_LR, 12)]
    assert pe_norm_ids.isdisjoint(by_lr[round(_BASE_LR * mult, 12)])


def test_decay_split_weight_decay_values():
    """Within each LR, decay params carry args.weight_decay and norm/bias params carry 0.0."""
    if _gemma4_missing():
        return _skip(_gemma4_missing())
    trainer, model = _make_trainer(3.0)
    decay_names = set(trainer.get_decay_parameter_names(model))
    id_to_name = {id(p): n for n, p in model.named_parameters()}
    opt = trainer.create_optimizer()
    for g in opt.param_groups:
        for p in g["params"]:
            name = id_to_name[id(p)]
            expected_wd = _WD if name in decay_names else 0.0
            assert g["weight_decay"] == expected_wd, (
                f"{name}: weight_decay={g['weight_decay']} expected {expected_wd}")
    # No-decay norm params really were routed to a 0.0 group (sanity the split is non-trivial).
    assert any(g["weight_decay"] == 0.0 for g in opt.param_groups)
    assert any(g["weight_decay"] == _WD for g in opt.param_groups)


def test_frozen_structural_param_excluded_from_all_groups():
    """A structural param left frozen (e.g. freeze_pe path) must appear in NO group — the
    grouping filters on requires_grad."""
    if _gemma4_missing():
        return _skip(_gemma4_missing())
    # Freeze pe_gain specifically; the rest of the structural path stays trainable.
    trainer, model = _make_trainer(3.0, freeze_param=lambda m: m.pe_gain)
    opt = trainer.create_optimizer()
    all_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    assert id(model.pe_gain) not in all_ids, "frozen pe_gain must be excluded"
    # but the rest of the structural params are still present & boosted
    other_struct = [p for p in model.structural_parameters()
                    if p.requires_grad and id(p) != id(model.pe_gain)]
    by_lr = _group_ids_by_lr(opt)
    assert {id(p) for p in other_struct} <= by_lr[round(_BASE_LR * 3.0, 12)]


# ==========================================================================
# Fallback to the stock optimizer
# ==========================================================================
def test_fallback_single_lr_when_mult_is_one():
    """mult=1.0 ⇒ stock optimizer: all params at base LR, no boosted group, but every
    trainable param (incl. structural) still present."""
    if _gemma4_missing():
        return _skip(_gemma4_missing())
    trainer, model = _make_trainer(1.0)
    opt = trainer.create_optimizer()
    lrs = {round(g["lr"], 12) for g in opt.param_groups}
    assert lrs == {round(_BASE_LR, 12)}, f"expected only base LR, got {sorted(lrs)}"
    all_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    struct_ids = {id(p) for p in model.structural_parameters() if p.requires_grad}
    assert struct_ids <= all_ids, "structural params must still be optimized at base LR"


# ==========================================================================
# Model contract the grouping relies on
# ==========================================================================
def test_structural_parameters_disjoint_from_lora_params():
    """create_optimizer partitions by id; structural_parameters() and the LoRA params it
    must NOT double-count have to be disjoint sets."""
    if _gemma4_missing():
        return _skip(_gemma4_missing())
    _, model = _make_trainer(3.0)
    struct_ids = {id(p) for p in model.structural_parameters()}
    lora_ids = {id(p) for n, p in model.named_parameters() if "lora_" in n}
    assert lora_ids, "fixture should attach LoRA params"
    assert struct_ids.isdisjoint(lora_ids)


# ==========================================================================
# Optional-dep gate + standalone runner
# ==========================================================================
def _gemma4_missing():
    try:
        from transformers import Gemma4UnifiedForCausalLM, Gemma4UnifiedTextConfig  # noqa: F401
        return None
    except Exception as e:  # noqa: BLE001
        return f"gemma4_unified unavailable: {e}"


if __name__ == "__main__":
    passed, failed, skipped = 0, [], 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                r = fn()
                if r is None and _gemma4_missing():
                    skipped += 1
                    print(f"{name}: SKIP")
                else:
                    passed += 1
                    print(f"{name}: PASS")
            except Exception as e:  # noqa: BLE001
                failed.append((name, f"{type(e).__name__}: {e}"))
                print(f"{name}: FAIL — {type(e).__name__}: {e}")
    print(f"\n{passed} passed, {len(failed)} failed, {skipped} skipped")
    for name, err in failed:
        print(f"  FAIL {name}: {err}")
    sys.exit(1 if failed else 0)
