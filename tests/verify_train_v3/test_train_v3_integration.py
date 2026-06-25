"""Verification suite for ``prism.training.train_v3`` and its integration with the
Hydra config tree under ``experiments/e9_hydra_training/``.

CS-ONLY scope (no ``DEEP-LEARNING`` flag): this file verifies *deterministic
orchestration* only — config composition/plumbing, output-dir construction, model-name
slugging, precision-capability flags, pad-token plumbing, ``loss_target`` label masking
(pure tensor bookkeeping), and trainer MRO. It does NOT touch model architecture,
forward/backward, gradients, or any learned/stochastic behaviour.

The oracle for each contract is restated independently from the docstrings/spec of the
target — never copied from its body.

Environment note: ``train_v3`` imports ``hydra``/``omegaconf``, which live in the conda
env (``GREP-PRISM``) but NOT in the ``uv`` project env. Run this file with:

    conda run -n GREP-PRISM python tests/verify_train_v3/test_train_v3_integration.py

(pytest is absent from the conda env, so this file uses the standalone runner footer.)
"""
import os
import sys
import warnings

sys.path.insert(0, "src")

import torch
from types import ModuleType, SimpleNamespace

import omegaconf
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir

# --- Boundary stub: this env lacks the optional third-party `liger_kernel`, which `trl`
# imports unconditionally at module load (trl/trainer/sft_trainer.py). train_v3 only needs
# trl's SFTConfig/SFTTrainer *classes* (never the liger fast-path), so we inject a minimal
# fake module so the deterministic orchestration code under test becomes importable. This
# stubs a boundary; it does NOT change any behaviour exercised below.
if "liger_kernel" not in sys.modules:
    class _Dummy:  # stand-in for any liger symbol trl binds at import (never used)
        pass

    def _liger_getattr(attr):
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)  # let introspection (inspect/import machinery) work
        return _Dummy

    import importlib.machinery
    for _name in ("liger_kernel", "liger_kernel.transformers"):
        _m = ModuleType(_name)
        _m.__file__ = "<stub>"
        _m.__spec__ = importlib.machinery.ModuleSpec(_name, loader=None)
        _m.__path__ = []
        _m.__getattr__ = _liger_getattr
        sys.modules[_name] = _m

from prism.training import train_v3
from prism.training.train_v3 import (
    _LOSS_TARGET_COLUMN,
    BaselineSFTTrainer,
    GraphSFTTrainer,
    LossTargetMixin,
    _bf16_supported,
    _construct_output_dir,
    _ensure_pad_tokens,
    _fp16_supported,
    _model_short_name,
    _validate_config,
)
from prism.eval.evaluate import GraphTokenAccuracyMixin
from prism.models import composite_graph

_CFG_DIR = os.path.abspath("experiments/e9_hydra_training")


def _compose(overrides):
    """Compose the e9 config with the given CLI-style overrides (fresh Hydra each call)."""
    with initialize_config_dir(version_base=None, config_dir=_CFG_DIR):
        return compose(config_name="config", overrides=list(overrides))


# ==========================================================================
# Config integration — the e9 config tree must provide every field train_v3 reads
# ==========================================================================
# Independent restatement of the flat keys train_v3.train_model() / its trainers read
# off the composed config (config.<name>). Derived by reading the source, NOT by
# importing it. Every default group composes unconditionally, so all of these must be
# present in the default composition regardless of architecture.
_FIELDS_TRAIN_V3_READS = [
    # wandb / output-dir / overwrite
    "wandb_project", "wandb_run_name", "wandb_tag", "checkpoint_dir", "save_name",
    "name", "report_to", "overwrite_ok", "enable_visualizer",
    # model / device / quant
    "base_model", "device", "bit4", "architecture",
    # data
    "dataset_num_proc", "dataloader_num_workers", "max_seq_length", "text_edge_list",
    # training hyperparams (plumbed into SFTConfig)
    "per_device_train_batch_size", "per_device_eval_batch_size",
    "gradient_accumulation_steps", "warmup_steps", "epochs", "max_steps",
    "learning_rate", "weight_decay", "gradient_checkpointing", "no_train",
    # lora
    "r", "lora_alpha", "lora_dropout", "target_modules",
    # multistage
    "freeze_llm", "freeze_lora", "freeze_pe", "init_lora_from", "init_pe_from",
    "loss_target", "lora_warmup_steps", "lam_c_warmup_steps",
    # structural / r-pearl
    "d_model", "dropout", "use_layer_norm", "eps", "pe_gain_init", "use_pe_norm",
    "pe_hidden_channels", "pe_num_layers", "num_samples", "k_pe", "pe_node_features",
    # gt
    "k_gt", "gt_num_layers", "gt_heads",
    # graph mask
    "mask_k_hops", "mask_symmetrize", "mask_use_edges",
    # eval / post-train
    "eval_data", "eval_num_graphs", "eval_use_icl", "eval_epoch_interval",
    "post_train_eval_graphs",
]


def test_default_compose_provides_every_field_train_v3_reads():
    """Integration: default composition supplies every flat key train_v3 accesses."""
    cfg = _compose([])
    missing = [k for k in _FIELDS_TRAIN_V3_READS if k not in cfg]
    assert not missing, f"default config is missing fields train_v3 reads: {missing}"


def test_default_compose_passes_validation():
    """``_validate_config`` accepts the default composition and coerces ``eps`` to float."""
    cfg = _compose([])
    _validate_config(cfg)
    assert isinstance(cfg.eps, float)


def test_every_architecture_passes_validation_and_provides_its_fields():
    """Each ``architecture`` value composes and validates; arch-specific groups are present."""
    for arch in ("llm", "rpearl_llm", "rpearl_gt_llm", "gt_llm", "graph_mask_llm"):
        cfg = _compose([f"architecture={arch}"])
        _validate_config(cfg)
        # gt_* and mask_* always compose (their groups are in the default list), so the
        # arch-conditional gnn_config reads in train_v3 never hit a missing key.
        for k in ("k_gt", "gt_num_layers", "gt_heads", "mask_k_hops",
                  "mask_symmetrize", "mask_use_edges"):
            assert k in cfg, f"architecture={arch}: missing {k}"


def test_composite_graph_rebuild_params_for_every_design_option():
    """Integration contract: for ``architecture=composite_graph_gt``, EVERY ``composite_graphs``
    design option must compose into a config from which train_v3 can build ``gnn_config``.

    train_v3 calls ``composite_graph.composite_graph_gnn_rebuild_params(config)`` unconditionally
    when building ``gnn_config`` for ``composite_graph_gt`` (train_v3.py:204). The README lists
    ``centered`` / ``c_bias`` / ``c_per_layer`` as first-class recipes, so each must work.
    """
    failures = {}
    for opt in ("default", "centered", "c_bias", "c_per_layer"):
        cfg = _compose(["architecture=composite_graph_gt", f"composite_graphs={opt}"])
        try:
            params = composite_graph.composite_graph_gnn_rebuild_params(cfg)
            assert isinstance(params, dict) and params, "expected a non-empty rebuild dict"
        except Exception as e:  # noqa: BLE001 — record the break, don't swallow it
            failures[opt] = f"{type(e).__name__}: {e}"
    assert not failures, (
        "composite_graphs options fail to build gnn_config for composite_graph_gt "
        f"(train_v3 would crash before training): {failures}"
    )


def test_readme_recipe_overrides_compose_and_validate():
    """A spread of documented README recipes compose and pass ``_validate_config``."""
    recipes = [
        ["architecture=llm"],
        ["architecture=llm", "text_edge_list=none"],
        ["architecture=rpearl_llm", "multistage=stage1_sft", "training=default"],
        ["architecture=gt_llm", "gt=L5_d4096", "pe_node_features=word_embeddings"],
        ["architecture=graph_mask_llm", "mask_k_hops=2", "mask_use_edges=false",
         "freeze_llm=true"],
        ["architecture=rpearl_gt_llm", "llm=gemma4_31b", "overview=e8",
         "eval=single_graph", "bit4=true"],
        ["architecture=llm", "training=zeroshot", "overview=e8", "eval=single_graph",
         "data=legacy", "device=cuda0"],
    ]
    for ov in recipes:
        cfg = _compose(ov)
        _validate_config(cfg)


# ==========================================================================
# structural_lr_mult plumbing — the bridge between config and create_optimizer
# ==========================================================================
# create_optimizer reads the boost factor from gnn_config: mult = gnn_config.get(
# "structural_lr_mult", 1.0) (train_v3.py:494). The ONLY conduit that injects it into
# gnn_config is composite_graph_gnn_rebuild_params(config) (train_v3.py:204) — and that
# helper returns {} for any architecture != composite_graph_gt. So the boosted LR group
# is reachable ONLY for composite_graph_gt; for every other graph arch the override is
# silently dropped and create_optimizer falls back to 1.0. These two tests pin both halves
# of that observed contract so a future change to the conduit is caught.
def test_structural_lr_mult_carried_into_gnn_config_for_composite():
    """For composite_graph_gt, config.structural_lr_mult survives into the rebuild params
    create_optimizer reads — so the boosted-LR group actually engages with the configured
    multiplier (here c_bias ships 3.0)."""
    cfg = _compose(["architecture=composite_graph_gt", "composite_graphs=c_bias"])
    params = composite_graph.composite_graph_gnn_rebuild_params(cfg)
    assert params.get("structural_lr_mult") == cfg.structural_lr_mult == 3.0


def test_structural_lr_mult_override_dropped_for_non_composite_arch():
    """REGRESSION GUARD on a known integration gap: a CLI structural_lr_mult override on a
    non-composite graph arch (gt_llm) is NOT plumbed into gnn_config — the sole conduit
    returns {} — so create_optimizer silently uses 1.0 despite config carrying 8.0. This
    documents current behaviour; if structural_lr_mult is ever made to apply to gt_llm,
    flip this assertion. (create_optimizer's docstring frames the structural group as
    'GT + R-PEARL + gate', which gt_llm has, so the silent drop is a latent foot-gun.)"""
    cfg = _compose(["architecture=gt_llm", "structural_lr_mult=8.0"])
    assert cfg.structural_lr_mult == 8.0  # the field IS set and validated...
    params = composite_graph.composite_graph_gnn_rebuild_params(cfg)
    assert "structural_lr_mult" not in params  # ...but never reaches gnn_config → dropped


# ==========================================================================
# _validate_config — coercion + domain rejection (loud failure on bad input)
# ==========================================================================
def _validation_cfg(**kw):
    base = {"eps": "1e-6", "loss_target": "all", "text_edge_list": "present"}
    base.update(kw)
    return OmegaConf.create(base)


def test_validate_config_coerces_string_eps_to_float():
    cfg = _validation_cfg(eps="1e-5")
    _validate_config(cfg)
    assert cfg.eps == 1e-5 and isinstance(cfg.eps, float)


def test_validate_config_rejects_unknown_loss_target():
    raised = False
    try:
        _validate_config(_validation_cfg(loss_target="bogus"))
    except ValueError as e:
        raised = "loss_target" in str(e)
    assert raised, "expected ValueError naming loss_target for an unknown target"


def test_validate_config_edge_list_requires_text_edges_present():
    raised = False
    try:
        _validate_config(_validation_cfg(loss_target="edge_list", text_edge_list="none"))
    except ValueError as e:
        raised = "text_edge_list" in str(e)
    assert raised, "loss_target=edge_list with text_edge_list!=present must raise"


def test_validate_config_accepts_all_valid_targets():
    for tgt in ("all", "responses", "edge_list"):
        cfg = _validation_cfg(loss_target=tgt, text_edge_list="present")
        _validate_config(cfg)
        assert cfg.loss_target == tgt


# ==========================================================================
# _model_short_name — filesystem-safe slug from a HF model id
# ==========================================================================
def test_model_short_name_documented_examples():
    assert _model_short_name("meta-llama/Llama-3.1-8B-Instruct") == "llama-3.1-8b"
    assert _model_short_name("Qwen/Qwen2.5-0.5B-Instruct") == "qwen2.5-0.5b"


def test_model_short_name_on_e9_base_models():
    # The actual bases in the llm/ group must slug cleanly (used in the checkpoint dir).
    assert _model_short_name("google/gemma-4-12B-it") == "gemma-4-12b-it"
    assert _model_short_name("google/gemma-4-31B-it") == "gemma-4-31b-it"


def test_model_short_name_strips_instruct_and_collapses_hyphens():
    assert _model_short_name("org/Foo--Bar-Instruct") == "foo-bar"
    assert _model_short_name("Plain-Name") == "plain-name"  # no org prefix, no -Instruct


# ==========================================================================
# _construct_output_dir — {checkpoint_dir}/{subdir}, wandb id always appended
# ==========================================================================
def _outdir_cfg(**kw):
    base = {
        "save_name": None, "checkpoint_dir": "ckpts", "base_model": "org/Llama-3.1-8B-Instruct",
        "name": "run", "architecture": "rpearl_llm", "r": 16, "bit4": False,
    }
    base.update(kw)
    return OmegaConf.create(base)


def test_construct_output_dir_auto_name():
    out = _construct_output_dir(_outdir_cfg(), "abc123")
    assert out == os.path.join("ckpts", "run_rpearl_llm_llama-3.1-8b_r16_abc123")


def test_construct_output_dir_4bit_suffix():
    out = _construct_output_dir(_outdir_cfg(bit4=True), "wid")
    assert out == os.path.join("ckpts", "run_rpearl_llm_llama-3.1-8b_r16_4bit_wid")


def test_construct_output_dir_save_name_override_still_appends_wandb_id():
    out = _construct_output_dir(_outdir_cfg(save_name="myrun"), "zzz")
    assert out == os.path.join("ckpts", "myrun_zzz")


# ==========================================================================
# _bf16_supported / _fp16_supported — precision-capability flags
# ==========================================================================
def test_precision_flags_are_mutually_exclusive():
    # Contract: fp16 helper returns False whenever bf16 is supported (train_v3.py:317),
    # so the two SFTConfig precision flags can never both be True.
    assert not (_bf16_supported() and _fp16_supported())


def test_precision_flags_false_without_cuda():
    if not torch.cuda.is_available():
        assert _bf16_supported() is False
        assert _fp16_supported() is False


# ==========================================================================
# _ensure_pad_tokens — pad-token plumbing between tokenizer and model.config
# ==========================================================================
def test_ensure_pad_tokens_aliases_eos_when_missing():
    tok = SimpleNamespace(pad_token=None, eos_token="</s>", pad_token_id=7)
    model = SimpleNamespace(config=SimpleNamespace(pad_token_id=None))
    _ensure_pad_tokens(tok, model)
    assert tok.pad_token == "</s>"            # tokenizer pad aliased to eos
    assert model.config.pad_token_id == 7     # model config filled from tokenizer pad id


def test_ensure_pad_tokens_is_idempotent_when_already_set():
    tok = SimpleNamespace(pad_token="<pad>", eos_token="</s>", pad_token_id=3)
    model = SimpleNamespace(config=SimpleNamespace(pad_token_id=99))
    _ensure_pad_tokens(tok, model)
    assert tok.pad_token == "<pad>"           # not clobbered
    assert model.config.pad_token_id == 99    # not clobbered


# ==========================================================================
# LossTargetMixin (train_v3 copy) — label masking, column popping, norm flag
# ==========================================================================
class _Recorder:
    """Stands in for SFTTrainer below the mixin: records what reaches compute_loss."""

    def __init__(self):
        self.model_accepts_loss_kwargs = True
        self.seen_labels = None
        self.seen_keys = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self.seen_keys = set(inputs.keys())
        labels = inputs.get("labels")
        self.seen_labels = labels.clone() if labels is not None else None
        return torch.tensor(0.0)


class _LossTrainer(LossTargetMixin, _Recorder):
    def __init__(self, loss_target):
        _Recorder.__init__(self)
        self._set_loss_target(loss_target)


def test_loss_target_column_map_is_stable():
    assert _LOSS_TARGET_COLUMN == {"responses": "assistant_idx", "edge_list": "edge_list_idx"}


def test_set_loss_target_disables_loss_kwargs_only_when_masking():
    assert _LossTrainer("all").model_accepts_loss_kwargs is True
    assert _LossTrainer("responses").model_accepts_loss_kwargs is False
    assert _LossTrainer("edge_list").model_accepts_loss_kwargs is False


def test_compute_loss_masks_to_responses_and_pops_index_columns():
    t = _LossTrainer("responses")
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "labels": torch.tensor([[10, 11, 12, 13, 14]]),
        "assistant_idx": [[1, 3]],
        "edge_list_idx": [[0, 2, 4]],
    }
    loss = t.compute_loss(None, inputs)
    assert isinstance(loss, torch.Tensor) and loss.ndim == 0
    assert t.seen_labels.tolist() == [[-100, 11, -100, 13, -100]]
    assert "assistant_idx" not in t.seen_keys and "edge_list_idx" not in t.seen_keys


def test_compute_loss_edge_list_uses_edge_column():
    t = _LossTrainer("edge_list")
    inputs = {
        "labels": torch.tensor([[10, 11, 12, 13, 14]]),
        "assistant_idx": [[0, 1]],            # ignored for edge_list target
        "edge_list_idx": [[2, 4]],
    }
    t.compute_loss(None, inputs)
    assert t.seen_labels.tolist() == [[-100, -100, 12, -100, 14]]


def test_compute_loss_all_target_leaves_labels_but_still_pops_columns():
    t = _LossTrainer("all")
    inputs = {
        "labels": torch.tensor([[10, 11, 12]]),
        "assistant_idx": [[1]],
        "edge_list_idx": [[2]],
    }
    t.compute_loss(None, inputs)
    assert t.seen_labels.tolist() == [[10, 11, 12]]
    assert "assistant_idx" not in t.seen_keys and "edge_list_idx" not in t.seen_keys


def test_mask_labels_multi_row():
    t = _LossTrainer("responses")
    inputs = {"labels": torch.tensor([[10, 11, 12], [20, 21, 22]])}
    t._mask_labels_to_positions(inputs, [[1], [0, 2]], "responses")
    assert inputs["labels"].tolist() == [[-100, 11, -100], [20, -100, 22]]


def test_mask_labels_empty_falls_back_to_full_sequence_with_warning():
    t = _LossTrainer("responses")
    inputs = {"labels": torch.tensor([[1, 2, 3]])}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t._mask_labels_to_positions(inputs, [[]], "responses")
    assert any("no supervised tokens" in str(w.message) for w in caught)
    assert inputs["labels"].tolist() == [[1, 2, 3]]    # unchanged (no all-masked NaN)


def test_real_trainers_order_mask_before_diagnostic():
    for cls in (GraphSFTTrainer, BaselineSFTTrainer):
        names = [c.__name__ for c in cls.__mro__]
        assert names.index("LossTargetMixin") < names.index("GraphTokenAccuracyMixin")
        assert names.index("GraphTokenAccuracyMixin") < names.index("SFTTrainer")


# ==========================================================================
# Standalone runner (pytest is absent from the conda env that has hydra)
# ==========================================================================
if __name__ == "__main__":
    passed, failed = 0, []
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                passed += 1
                print(f"{name}: PASS")
            except Exception as e:  # noqa: BLE001 — report, don't abort the suite
                failed.append((name, f"{type(e).__name__}: {e}"))
                print(f"{name}: FAIL — {type(e).__name__}: {e}")
    print(f"\n{passed} passed, {len(failed)} failed")
    for name, err in failed:
        print(f"  FAIL {name}: {err}")
    sys.exit(1 if failed else 0)
