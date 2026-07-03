"""Verification suite for ``prism.training.train_v3`` and its integration with the
nested Hydra config at ``experiments/base_config.yaml``.

CS-ONLY scope (no ``DEEP-LEARNING`` flag): this file verifies *deterministic
orchestration* only — config composition/plumbing, output-dir construction, model-name
slugging, precision-capability flags, pad-token plumbing, ``loss_target`` label masking
(pure tensor bookkeeping), and trainer MRO. It does NOT touch model architecture,
forward/backward, gradients, or any learned/stochastic behaviour.

The oracle for each contract is restated independently from the docstrings/spec of the
target — never copied from its body.

Config note: the former group tree (``experiments/e9_hydra_training/`` with overview/data/
training/... groups) was flattened into a single nested ``experiments/base_config.yaml``
(sections model/gnn/data/lora/trainer/eval/wandb). Per-experiment files inherit it via
``defaults: [base_config, _self_]``. Group selections (``multistage=stage1_sft``) are gone;
runs are reconstructed with flat key overrides (``gnn.arch=...``, ``trainer.freeze_pe=true``).

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

_CFG_DIR = os.path.abspath("experiments")

# Per-experiment files that inherit base_config via `defaults: [base_config, _self_]`.
_EXPERIMENT_CONFIGS = (
    "e9_ms_stage1", "e9_ms_stage2", "e9_ms_stage3",
    "e9_baseline_llm", "e9_baseline_llm_no_edges",
)


def _compose(overrides, config_name="base_config"):
    """Compose ``config_name`` with the given CLI-style overrides (fresh Hydra each call)."""
    with initialize_config_dir(version_base=None, config_dir=_CFG_DIR):
        return compose(config_name=config_name, overrides=list(overrides))


_MISSING = object()


def _present(cfg, dotted):
    """True if a dotted path resolves to a present node (None counts as present)."""
    return OmegaConf.select(cfg, dotted, default=_MISSING) is not _MISSING


# ==========================================================================
# Config integration — base_config must provide every field train_v3 reads
# ==========================================================================
# Independent restatement of the keys train_v3.train_model() / its trainers read off the
# composed config (config.<section>.<name>). Derived by reading the source, NOT by importing
# it. Every section composes unconditionally, so all of these must be present in the default
# composition regardless of architecture.
_FIELDS_TRAIN_V3_READS = [
    # wandb / output-dir / overwrite
    "wandb.project", "wandb.run_name", "wandb.tag", "trainer.checkpoint_dir",
    "trainer.save_name", "name", "trainer.report_to", "trainer.overwrite_ok",
    # model / device / quant
    "model.path", "trainer.device", "trainer.bit4", "gnn.arch",
    # data
    "trainer.dataset_num_proc", "trainer.dataloader_num_workers", "data.max_seq_length",
    "data.text_edge_list",
    # training hyperparams (plumbed into SFTConfig)
    "trainer.per_device_train_batch_size", "trainer.per_device_eval_batch_size",
    "trainer.gradient_accumulation_steps", "trainer.warmup_steps", "trainer.epochs",
    "trainer.max_steps", "trainer.learning_rate", "trainer.weight_decay",
    "trainer.gradient_checkpointing", "trainer.no_train",
    # lora
    "lora.r", "lora.alpha", "lora.dropout", "lora.target_modules",
    # multistage (folded into trainer)
    "trainer.freeze_llm", "trainer.freeze_lora", "trainer.freeze_pe",
    "trainer.init_lora_from", "trainer.init_pe_from", "trainer.loss_target",
    # structural / r-pearl
    "gnn.d_model", "gnn.dropout", "gnn.use_layer_norm", "gnn.eps", "gnn.pe_gain_init",
    "gnn.use_pe_norm", "gnn.pe_hidden_channels", "gnn.pe_num_layers", "gnn.num_samples",
    "gnn.k_pe", "gnn.pe_node_features",
    # gt
    "gnn.k_gt", "gnn.gt_num_layers", "gnn.gt_heads",
    # graph mask
    "gnn.mask_k_hops", "gnn.mask_symmetrize", "gnn.mask_use_edges",
    # eval / post-train
    "eval.data", "eval.num_graphs", "eval.use_icl", "eval.epoch_interval",
    "eval.post_train_graphs",
]


def test_default_compose_provides_every_field_train_v3_reads():
    """Integration: default composition supplies every key train_v3 accesses."""
    cfg = _compose([])
    missing = [k for k in _FIELDS_TRAIN_V3_READS if not _present(cfg, k)]
    assert not missing, f"base_config is missing fields train_v3 reads: {missing}"


def test_default_compose_passes_validation():
    """``_validate_config`` accepts the default composition and coerces ``gnn.eps`` to float."""
    cfg = _compose([])
    _validate_config(cfg)
    assert isinstance(cfg.gnn.eps, float)


def test_every_architecture_passes_validation_and_provides_its_fields():
    """Each ``gnn.arch`` value composes and validates; arch-specific keys are present."""
    for arch in ("llm", "rpearl_llm", "rpearl_gt_llm", "gt_llm", "graph_mask_llm"):
        cfg = _compose([f"gnn.arch={arch}"])
        _validate_config(cfg)
        # gt_* and mask_* always compose (one flat gnn section), so the arch-conditional
        # gnn_config reads in train_v3 never hit a missing key.
        for k in ("gnn.k_gt", "gnn.gt_num_layers", "gnn.gt_heads", "gnn.mask_k_hops",
                  "gnn.mask_symmetrize", "gnn.mask_use_edges"):
            assert _present(cfg, k), f"gnn.arch={arch}: missing {k}"


def test_experiment_files_compose_and_validate():
    """Every per-experiment file inherits base_config and passes validation."""
    failures = {}
    for name in _EXPERIMENT_CONFIGS:
        try:
            cfg = _compose([], config_name=name)
            _validate_config(cfg)
            # Each experiment still resolves every field train_v3 reads.
            missing = [k for k in _FIELDS_TRAIN_V3_READS if not _present(cfg, k)]
            assert not missing, f"missing {missing}"
        except Exception as e:  # noqa: BLE001 — record the break, don't swallow it
            failures[name] = f"{type(e).__name__}: {e}"
    assert not failures, f"experiment configs fail to compose/validate: {failures}"


def test_stage_experiment_overrides_take_effect():
    """The multistage stage files actually carry their distinguishing overrides."""
    s1 = _compose([], config_name="e9_ms_stage1")
    assert s1.trainer.freeze_pe is True and s1.gnn.arch == "rpearl_llm"
    s2 = _compose([], config_name="e9_ms_stage2")
    assert s2.trainer.freeze_lora is True and s2.trainer.loss_target == "edge_list"
    assert s2.data.text_edge_list == "present"  # edge_list target needs edges present
    s3 = _compose([], config_name="e9_ms_stage3")
    assert s3.data.text_edge_list == "none" and s3.trainer.loss_target == "responses"


def test_flat_override_recipes_compose_and_validate():
    """A spread of documented flat-override recipes compose and pass ``_validate_config``."""
    recipes = [
        ["gnn.arch=llm"],
        ["gnn.arch=llm", "data.text_edge_list=none"],
        ["gnn.arch=rpearl_llm", "trainer.freeze_pe=true"],                       # stage-1 SFT
        ["gnn.arch=gt_llm", "gnn.gt_num_layers=5", "gnn.gt_heads=32", "gnn.k_gt=1",
         "gnn.d_model=4096", "gnn.pe_node_features=word_embeddings"],            # gt L5_d4096
        ["gnn.arch=graph_mask_llm", "gnn.mask_k_hops=2", "gnn.mask_use_edges=false",
         "trainer.freeze_llm=true"],
        ["gnn.arch=rpearl_gt_llm", "model.path=google/gemma-4-31B-it",
         "eval.data=data/x/test_graphs/data_gen_023.json", "trainer.bit4=true"],
        ["gnn.arch=llm", "trainer.no_train=true", "trainer.bit4=true",
         "data.dataset_proportion=0.01", "data.text_edge_list=none", "trainer.device=0"],
    ]
    for ov in recipes:
        cfg = _compose(ov)
        _validate_config(cfg)


# ==========================================================================
# _validate_config — coercion + domain rejection (loud failure on bad input)
# ==========================================================================
def _validation_cfg(eps="1e-6", loss_target="all", text_edge_list="present"):
    """A minimal NESTED config carrying just the fields ``_validate_config`` reads:
    ``config.gnn.eps``, ``config.gnn.arch``, ``config.trainer.loss_target``,
    ``config.data.text_edge_list``."""
    return OmegaConf.create({
        "gnn": {"eps": eps, "arch": "rpearl_llm"},
        "trainer": {"loss_target": loss_target},
        "data": {"text_edge_list": text_edge_list},
    })


def test_validate_config_coerces_string_eps_to_float():
    cfg = _validation_cfg(eps="1e-5")
    _validate_config(cfg)
    assert cfg.gnn.eps == 1e-5 and isinstance(cfg.gnn.eps, float)


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
        assert cfg.trainer.loss_target == tgt


# ==========================================================================
# _model_short_name — filesystem-safe slug from a HF model id
# ==========================================================================
def test_model_short_name_documented_examples():
    assert _model_short_name("meta-llama/Llama-3.1-8B-Instruct") == "llama-3.1-8b"
    assert _model_short_name("Qwen/Qwen2.5-0.5B-Instruct") == "qwen2.5-0.5b"


def test_model_short_name_on_e9_base_models():
    # The actual bases in the model.path comments must slug cleanly (used in the ckpt dir).
    assert _model_short_name("google/gemma-4-12B-it") == "gemma-4-12b-it"
    assert _model_short_name("google/gemma-4-31B-it") == "gemma-4-31b-it"


def test_model_short_name_strips_instruct_and_collapses_hyphens():
    assert _model_short_name("org/Foo--Bar-Instruct") == "foo-bar"
    assert _model_short_name("Plain-Name") == "plain-name"  # no org prefix, no -Instruct


# ==========================================================================
# _construct_output_dir — {checkpoint_dir}/{subdir}, wandb id always appended
# ==========================================================================
def _outdir_cfg(save_name=None, checkpoint_dir="ckpts", path="org/Llama-3.1-8B-Instruct",
                name="run", arch="rpearl_llm", r=16, bit4=False):
    """A NESTED config carrying just the fields ``_construct_output_dir`` reads:
    trainer.{save_name,checkpoint_dir,bit4}, model.path, name, gnn.arch, lora.r."""
    return OmegaConf.create({
        "name": name,
        "model": {"path": path},
        "gnn": {"arch": arch},
        "lora": {"r": r},
        "trainer": {"save_name": save_name, "checkpoint_dir": checkpoint_dir, "bit4": bit4},
    })


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
    # Contract: fp16 helper returns False whenever bf16 is supported, so the two SFTConfig
    # precision flags can never both be True.
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
