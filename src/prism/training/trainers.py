"""Trainer classes for the GREP-PRISM training stack (used by ``train_v3``).

* :class:`LossTargetMixin` — restricts the LM loss to a configurable token span
  (assistant turns / edge-list bullets) via precomputed per-example index columns.
* :class:`GraphSFTTrainer` — graph-augmented architectures: two-group structural
  LR, grad-norm capture for the debug callback, and checkpoint serialisation
  (``train_config.json`` + ``gnn_weights.pt``).
* :class:`BaselineSFTTrainer` — plain-LLM baseline with the same metrics/config
  surface so runs stay comparable.
"""
import json
import os
import warnings

import torch
from trl import SFTTrainer

from prism.eval import callbacks
from prism.eval import evaluate


# Maps loss_target values to their precomputed per-example index column. "all" is
# absent (full-sequence). Add a new target by precomputing a column and registering it.
_LOSS_TARGET_COLUMN = {"responses": "assistant_idx", "edge_list": "edge_list_idx"}


class LossTargetMixin:
    """Restricts the next-token loss to a configurable token span (``loss_target``).

    ``all`` (default) = full sequence; ``responses`` = assistant turns;
    ``edge_list`` = edge bullets. Supervised positions come from per-example index
    columns precomputed in ``preprocess_dataset``, mapped via ``_LOSS_TARGET_COLUMN``.

    Masks ``inputs['labels']`` then defers to ``super().compute_loss`` (cooperative,
    MRO-safe). Mix in BEFORE :class:`prism.eval.evaluate.GraphTokenAccuracyMixin` so masking happens first.
    """

    def _set_loss_target(self, loss_target: str):
        """Configure the supervised-token span. Call after ``super().__init__``.

        For masked targets (responses / edge_list), disables the loss-kwargs path so
        the LM loss uses reduction="mean" over only the supervised tokens, not HF's
        full-batch normalization (which would under-normalize the masked span).
        """
        self.loss_target = loss_target
        if loss_target != "all":
            self.model_accepts_loss_kwargs = False

    def _mask_labels_to_positions(self, inputs, idx_lists, target):
        """Mask ``inputs['labels']`` to -100 outside the per-example supervised positions.

        ``idx_lists`` is a per-example list of token positions (e.g. assistant_idx /
        edge_list_idx). If a batch ends up with NO supervised tokens, fall back to its
        full-sequence labels with a warning (better than a NaN all-masked loss).
        """
        labels = inputs["labels"]
        keep = torch.zeros_like(labels, dtype=torch.bool)
        for b, idx in enumerate(idx_lists):
            if not idx or b >= labels.shape[0]:
                continue
            pos = torch.as_tensor(
                [p for p in idx if 0 <= p < labels.shape[1]],
                device=labels.device,
                dtype=torch.long,
            )
            if pos.numel():
                keep[b, pos] = True
        if keep.any():
            inputs["labels"] = labels.masked_fill(~keep, -100)
        else:
            warnings.warn(
                f"loss_target={target!r} but no supervised tokens located in this "
                "batch; falling back to full-sequence loss for it."
            )

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # Pop the mask columns so they never reach the model forward, then restrict
        # the loss to the configured span ('all' / absent column => no masking).
        mask_columns = {
            name: inputs.pop(name, None) for name in set(_LOSS_TARGET_COLUMN.values())
        }
        target = getattr(self, "loss_target", "all")
        col = _LOSS_TARGET_COLUMN.get(target)
        if (
            col is not None
            and mask_columns.get(col) is not None
            and inputs.get("labels") is not None
        ):
            self._mask_labels_to_positions(inputs, mask_columns[col], target)
        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )


GraphTokenAccuracyMixin = evaluate.GraphTokenAccuracyMixin


from prism.training.run_dir import save_run_dir  # noqa: F401 — shared run-dir contract


def create_two_group_optimizer(trainer, structural):
    """Two learning-rate groups on ``trainer``: ``structural`` params at
    ``structural_lr_mult`` × base LR, everything else trainable at base LR.

    Shared by the SFT and GRPO trainers (each passes its own model's
    ``structural_parameters()`` — the GRPO trainer's policy is PEFT-wrapped, so
    it resolves them on the unwrapped core). Returns ``None`` when the stock
    optimizer applies (multiplier 1.0, parameter-free arch, or an optimizer
    already exists), so callers fall through to ``super().create_optimizer()``.
    """
    mult = float(trainer.gnn_config.get("structural_lr_mult", 1.0))
    opt_model = trainer.model
    if trainer.optimizer is not None or mult == 1.0 or not structural:
        return None

    structural_ids = {id(p) for p in structural}

    decay_names = set(trainer.get_decay_parameter_names(opt_model))
    base_lr = trainer.args.learning_rate
    groups = []
    # Structural at boosted LR, LLM/LoRA at base LR; each split into decay / no-decay.
    for is_struct, lr in ((True, base_lr * mult), (False, base_lr)):
        named = [
            (n, p)
            for n, p in opt_model.named_parameters()
            if p.requires_grad and (id(p) in structural_ids) == is_struct
        ]
        decay = [p for n, p in named if n in decay_names]
        no_decay = [p for n, p in named if n not in decay_names]
        if decay:
            groups.append(
                {"params": decay, "lr": lr, "weight_decay": trainer.args.weight_decay}
            )
        if no_decay:
            groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})

    try:
        optimizer_cls, optimizer_kwargs = trainer.get_optimizer_cls_and_kwargs(
            trainer.args, opt_model
        )
    except TypeError:
        optimizer_cls, optimizer_kwargs = trainer.get_optimizer_cls_and_kwargs(
            trainer.args
        )
    optimizer_kwargs.pop("params", None)
    optimizer_kwargs.pop("lr", None)
    trainer.optimizer = optimizer_cls(groups, lr=base_lr, **optimizer_kwargs)
    n_struct = sum(p.numel() for p in structural if p.requires_grad)
    print(
        f"[train] structural LR group: {mult}x base = {base_lr * mult:.2e} "
        f"({n_struct / 1e6:.2f}M params); LLM/LoRA at base LR {base_lr:.2e}"
    )
    return trainer.optimizer


class GraphSFTTrainer(LossTargetMixin, GraphTokenAccuracyMixin, SFTTrainer):
    def __init__(
        self,
        *args,
        gnn_config: dict,
        freeze_pe: bool = False,
        loss_target: str = "all",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gnn_config = gnn_config
        self.freeze_pe = freeze_pe
        self._set_loss_target(loss_target)
        if freeze_pe:
            # Stage-1 SFT: PE stays at init (gate closed); only LoRA trains.
            # Freeze EXPLICITLY rather than relying on PEFT having frozen everything: with
            # trainer.freeze_llm=true no adapter is attached (peft_config=None), so nothing
            # ever froze the graph side and freeze_pe was a silent no-op.
            for p in self.model.structural_parameters():
                p.requires_grad = False
            for p in self._base_lr_parameters():
                p.requires_grad = False
            if getattr(self.model, "pe_norm", None) is not None:
                for p in self.model.pe_norm.parameters():
                    p.requires_grad = False
            return
        # Re-enable gradients on the graph-side params PEFT froze (each model class
        # reports its own via structural_parameters(); parameter-free archs return []).
        for p in self.model.structural_parameters():
            p.requires_grad = True
        # Same treatment for graph-side params that belong at BASE LR rather than in the
        # boosted group: PEFT froze them, they must be re-enabled, but create_optimizer
        # must not sweep them into the multiplier. WIRE's ω/σ frequency table reports
        # itself here (see WireGraphLLM.base_lr_parameters).
        for p in self._base_lr_parameters():
            p.requires_grad = True
        # pe_norm is a non-LoRA module trained at base LR (kept out of the boosted
        # structural group); re-enable it here so magnitude calibration adapts.
        if getattr(self.model, "pe_norm", None) is not None:
            for p in self.model.pe_norm.parameters():
                p.requires_grad = True

    def _base_lr_parameters(self):
        """Graph-side params the model wants trained at base LR; [] if it declares none."""
        fn = getattr(self.model, "base_lr_parameters", None)
        return list(fn()) if callable(fn) else []

    def create_optimizer(self):
        """Two learning-rate groups: structural path (GT + R-PEARL + gate) at
        ``structural_lr_mult`` × base LR; LLM/LoRA at base LR. Falls back to the
        stock optimizer when the multiplier is 1.0.
        """
        opt = create_two_group_optimizer(self, self.model.structural_parameters())
        return opt if opt is not None else super().create_optimizer()

    def training_step(self, model, inputs, num_items_in_batch=None, **kwargs):
        loss = super().training_step(
            model, inputs, num_items_in_batch=num_items_in_batch, **kwargs
        )
        # Capture grad norms before zero_grad() (backward already ran in super).
        for cb in self.callback_handler.callbacks:
            if isinstance(cb, callbacks.GradientDebugCallback):
                cb._capture_grad_norms(model)
                break
        return loss

    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir or self.args.output_dir
        save_run_dir(self.model, self.gnn_config, output_dir)
        # Save the LoRA adapter whenever one is attached (including frozen carried-forward
        # adapters). freeze_llm=True attaches no adapter, so this is skipped for that path.
        if getattr(self.model, "peft_config", None) is not None:
            super().save_model(output_dir, _internal_call)


class BaselineSFTTrainer(LossTargetMixin, GraphTokenAccuracyMixin, SFTTrainer):
    """Plain-``llm`` baseline trainer. Identical to ``SFTTrainer`` except it logs the
    ``graph_acc/*`` metric (via :class:`GraphTokenAccuracyMixin`) so the baseline is
    comparable to the graph architectures. Paired with
    :class:`prism.data.data.TokenIndexCollator`, which carries the precomputed
    graph-token index columns the metric needs.

    Persists the same ``train_config.json`` the graph trainer writes (minus the
    ``"gnn"`` section) so the standalone eval boundary can recover the train-time
    ``text_edge_list`` policy. Without it an LLM trained with ``text_edge_list=none``
    would be silently evaluated with edge bullets re-added (train/eval mismatch).
    """

    def __init__(self, *args, train_config: dict, loss_target: str = "all", **kwargs):
        super().__init__(*args, **kwargs)
        self.train_config = train_config
        self._set_loss_target(loss_target)

    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "train_config.json"), "w") as f:
            json.dump(self.train_config, f, indent=2)
        super().save_model(output_dir, _internal_call)
