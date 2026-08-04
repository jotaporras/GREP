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
            if getattr(self.model, "pe_norm", None) is not None:
                for p in self.model.pe_norm.parameters():
                    p.requires_grad = False
            return
        # Re-enable gradients on the graph-side params PEFT froze (each model class
        # reports its own via structural_parameters(); parameter-free archs return []).
        for p in self.model.structural_parameters():
            p.requires_grad = True
        # pe_norm is a non-LoRA module trained at base LR (kept out of the boosted
        # structural group); re-enable it here so magnitude calibration adapts.
        if getattr(self.model, "pe_norm", None) is not None:
            for p in self.model.pe_norm.parameters():
                p.requires_grad = True

    def create_optimizer(self):
        """Two learning-rate groups: structural path (GT + R-PEARL + gate) at
        ``structural_lr_mult`` × base LR; LLM/LoRA at base LR. Falls back to the
        stock optimizer when the multiplier is 1.0.
        """
        mult = float(self.gnn_config.get("structural_lr_mult", 1.0))
        opt_model = self.model
        structural = self.model.structural_parameters()
        # No multiplier, or a parameter-free arch (empty structural set) → stock optimizer.
        if self.optimizer is not None or mult == 1.0 or not structural:
            return super().create_optimizer()

        structural_ids = {id(p) for p in structural}

        decay_names = set(self.get_decay_parameter_names(opt_model))
        base_lr = self.args.learning_rate
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
                    {"params": decay, "lr": lr, "weight_decay": self.args.weight_decay}
                )
            if no_decay:
                groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})

        try:
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
                self.args, opt_model
            )
        except TypeError:
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
                self.args
            )
        optimizer_kwargs.pop("params", None)
        optimizer_kwargs.pop("lr", None)
        self.optimizer = optimizer_cls(groups, lr=base_lr, **optimizer_kwargs)
        n_struct = sum(p.numel() for p in structural if p.requires_grad)
        print(
            f"[train] structural LR group: {mult}x base = {base_lr * mult:.2e} "
            f"({n_struct / 1e6:.2f}M params); LLM/LoRA at base LR {base_lr:.2e}"
        )
        return self.optimizer

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

    # Shared run-metadata keys stored at the TOP LEVEL of train_config.json; every
    # other gnn_config entry (the arch hyperparameters) nests under "gnn". Loaders
    # flatten this back (and still read legacy flat gnn_config.json checkpoints).
    # spine_tools / icl_examples belong here (not under "gnn"): they are PROMPT policy,
    # and eval.checkpoint.resolve_prompt_policy reads them at the top level. Nesting them
    # made that resolver always return the "predates the knob" fallback ("none", 0), so an
    # ICL/tool-trained graph checkpoint was silently re-evaluated zero-shot and tool-free.
    _RUN_META_KEYS = ("architecture", "base_model", "text_edge_list", "injection_scope",
                      "edge_weights", "spine_tools", "icl_examples")

    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        run_config = {k: self.gnn_config[k] for k in self._RUN_META_KEYS
                      if k in self.gnn_config}
        run_config["gnn"] = {k: v for k, v in self.gnn_config.items()
                             if k not in self._RUN_META_KEYS}
        with open(os.path.join(output_dir, "train_config.json"), "w") as f:
            json.dump(run_config, f, indent=2)
        if self.gnn_config.get("architecture") == "graph_mask_llm":
            # Parameter-free: mask rebuilt from config; train_config.json + LoRA adapter suffice.
            pass
        elif self.gnn_config.get("architecture") == "learnable_graph_mask":
            # Save the standalone GraphTransformer (Psi producer); the mask + adjacency
            # rebuild from gnn_config and the LoRA adapter is saved below.
            torch.save(
                {"pe_model": self.model.pe_model.state_dict()},
                os.path.join(output_dir, "gnn_weights.pt"),
            )
        elif self.gnn_config.get("architecture") == "wire_llm":
            # WIRE: the Ψ producer, the angle gate, and the frequency store. Which store
            # is populated depends on gnn.wire_vanilla — the learnable ω table
            # (wire_vanilla=true, the paper's form) or the frozen ε directions plus the
            # learned per-layer σ (the expectation arm). BOTH key sets are always
            # written (the unused one is an empty dict) so the checkpoint key set does
            # not depend on the mode: a key present in one mode and absent in the other
            # is exactly the silent-corruption case loaders.py guards against.
            # ε/ω are SAVED rather than reconstructed from wire_omega_seed: regenerating
            # them would make the checkpoint depend on torch RNG determinism across
            # versions/devices, which is exactly the silent-drift failure mode. (The seed
            # is still recorded in train_config.json.)
            torch.save(
                {
                    "pe_model": self.model.pe_model.state_dict(),
                    "pe_gain": self.model.pe_gain.data,
                    "wire_eps": self.model._wire_eps.state_dict(),
                    "wire_sigma": self.model._wire_sigma.state_dict(),
                    "wire_omega": self.model._wire_omega.state_dict(),
                },
                os.path.join(output_dir, "gnn_weights.pt"),
            )
        elif self.gnn_config.get("architecture") == "rpearl_gt_llm":
            # Full GT: save the whole GraphTransformer (includes R-PEARL inside) + projection head.
            torch.save(
                {
                    "gt_model": self.model.pe_model.state_dict(),
                    "pe_proj": self.model.pe_proj.state_dict(),
                    "pe_gain": self.model.pe_gain.data,
                    **(
                        {"pe_norm": self.model.pe_norm.state_dict()}
                        if self.model.pe_norm is not None
                        else {}
                    ),
                },
                os.path.join(output_dir, "gnn_weights.pt"),
            )
            # Also save the inner R-PEARL separately for analysis / reuse.
            torch.save(
                {
                    "rpearl": self.model.pe_model.pe_model.state_dict(),
                },
                os.path.join(output_dir, "rpearl_weights.pt"),
            )
        else:
            torch.save(
                {
                    "pe_model": self.model.pe_model.state_dict(),
                    "pe_proj": self.model.pe_proj.state_dict(),
                    "pe_gain": self.model.pe_gain.data,
                    **(
                        {"pe_norm": self.model.pe_norm.state_dict()}
                        if self.model.pe_norm is not None
                        else {}
                    ),
                },
                os.path.join(output_dir, "gnn_weights.pt"),
            )
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
