import os
import re
import warnings

from prism.data import data
from prism.eval import callbacks
from prism.eval import evaluate
from prism.models import architectures
from prism.models import composite_graph
from prism.models import loaders as model_loaders

import json

from dotenv import load_dotenv

load_dotenv()

import hydra
import omegaconf
import wandb
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer


def train_model(config: omegaconf.DictConfig):
    wandb_run_id = _setup_wandb(
        config.wandb_project, config.wandb_run_name, config.wandb_tag
    )
    # Dir: {name}_{architecture}_{model_slug}_r{r}[_4bit]
    output_dir = _construct_output_dir(config, wandb_run_id)

    if os.path.isdir(output_dir) and os.listdir(output_dir) and not config.overwrite_ok:
        raise RuntimeError(
            f"Checkpoint directory already exists and is non-empty: {output_dir}\n"
            f"Set overwrite_ok: true in your config to allow overwriting, "
            f"or delete the directory manually."
        )

    # Quantization / dtype
    bnb_config = None
    if config.bit4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
            if _bf16_supported()
            else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    # Model & tokenizer
    device_map = {"": 0} if config.device >= 0 else "auto"
    llm = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype="auto",
        quantization_config=bnb_config,  # None if not 4-bit
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, use_fast=True)
    _ensure_pad_tokens(tokenizer, llm)
    tokenizer.padding_side = "right"  # ty: ignore[invalid-assignment]

    model, collator = architectures.build_planner_model(config, llm, tokenizer)

    # --- Multistage init: weight-only carry-over from a prior stage (NOT HF resume). --
    # Load PE first (lives outside the LoRA adapter), then attach the carried adapter.
    is_graph_arch = config.architecture in (
        "rpearl_llm",
        "rpearl_gt_llm",
        "gt_llm",
        "graph_mask_llm",
        "composite_graph_gt",
    )
    if (config.init_pe_from or config.init_lora_from) and not is_graph_arch:
        raise ValueError(
            "init_pe_from / init_lora_from are only supported for graph architectures."
        )
    if config.init_pe_from:
        model_loaders.load_pe_weights_into(
            model, config.init_pe_from, config.architecture
        )
    attach_existing_adapter = config.init_lora_from is not None
    if attach_existing_adapter:
        model = PeftModel.from_pretrained(
            model, config.init_lora_from, is_trainable=not config.freeze_lora
        )
        if config.gradient_checkpointing and not config.freeze_lora:
            # Replicate TRL's get_peft_model behavior so LoRA gradients flow back
            # through the frozen base under gradient checkpointing.
            model.enable_input_require_grads()
        print(
            f"[multistage] attached {'frozen' if config.freeze_lora else 'trainable'} "
            f"LoRA adapter from {config.init_lora_from}"
        )

    train_dataset, eval_dataset = data.load_and_split_dataset(config, tokenizer)

    # LoRA config; for multimodal bases, exclude vision/audio towers (see architectures.peft_tower_exclude).
    tower_exclude = architectures.peft_tower_exclude(model)
    lora_config = LoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=config.target_modules,
        exclude_modules=tower_exclude,
        task_type="CAUSAL_LM",
    )
    if tower_exclude:
        print(
            f"[peft] multimodal base detected — excluding LoRA targets matching "
            f"{tower_exclude!r} (vision/audio towers)"
        )

    optim = "adamw_bnb_8bit" if config.bit4 else "adamw_torch_fused"

    # SFT trainer configuration
    sft_args = SFTConfig(
        dataset_num_proc=config.dataset_num_proc,
        dataloader_num_workers=config.dataloader_num_workers,
        packing=False,  # packing combines multiple examples into a single input_id. Keep disabled to avoid graph contamination.
        max_length=config.max_seq_length,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        num_train_epochs=config.epochs,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        lr_scheduler_type="linear",
        logging_steps=15,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # precision
        fp16=_fp16_supported(),
        bf16=_bf16_supported(),
        # misc
        seed=3407,
        output_dir=output_dir,
        report_to=config.report_to,
        run_name=config.wandb_run_name,
        optim=optim,
        remove_unused_columns=False,
        # Checkpointing / Validation: step-based when max_steps is set (dev), else epoch-based.
        save_strategy="steps" if config.max_steps > 0 else "epoch",
        save_steps=max(1, config.max_steps // 2) if config.max_steps > 0 else 500,
        save_total_limit=3,
        eval_strategy="steps" if config.max_steps > 0 else "epoch",
        eval_steps=max(1, config.max_steps // 2) if config.max_steps > 0 else 0.5,
        do_eval=True,
    )

    if config.architecture in (
        "rpearl_llm",
        "rpearl_gt_llm",
        "gt_llm",
        "graph_mask_llm",
        "composite_graph_gt",
    ):
        gnn_config = {
            "architecture": config.architecture,
            "base_model": config.base_model,
            "pe_hidden_channels": config.pe_hidden_channels,
            "pe_num_layers": config.pe_num_layers,
            "d_model": config.d_model,
            "num_samples": config.num_samples,
            "dropout": config.dropout,
            "k_pe": config.k_pe,
            "use_layer_norm": config.use_layer_norm,
            "text_edge_list": config.text_edge_list,
            "eps": config.eps,
            "pe_gain_init": config.pe_gain_init,
            "use_pe_norm": config.use_pe_norm,
            "pe_node_features": config.pe_node_features,
            **(
                {
                    "k_gt": config.k_gt,
                    "gt_num_layers": config.gt_num_layers,
                    "gt_heads": config.gt_heads,
                }
                if config.architecture
                in ("rpearl_gt_llm", "gt_llm", "composite_graph_gt")
                else {}
            ),
            # graph_mask_llm rebuild params (read back by loaders for eval).
            **(
                {
                    "mask_k_hops": config.mask_k_hops,
                    "mask_symmetrize": config.mask_symmetrize,
                    "mask_use_edges": config.mask_use_edges,
                }
                if config.architecture == "graph_mask_llm"
                else {}
            ),
            **composite_graph.composite_graph_gnn_rebuild_params(config),
        }
        trainer = GraphSFTTrainer(
            model=model,
            data_collator=collator,
            processing_class=tokenizer,
            # No fresh adapter when one was carried forward (it's already attached)
            # or when freeze_llm requests a PE-only run on the raw base.
            peft_config=lora_config
            if (not config.freeze_llm and not attach_existing_adapter)
            else None,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=sft_args,
            gnn_config=gnn_config,
            freeze_pe=config.freeze_pe,
            loss_target=config.loss_target,
        )
    else:
        trainer = BaselineSFTTrainer(
            model=model,
            data_collator=collator,
            processing_class=tokenizer,
            peft_config=lora_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=sft_args,
            train_config={
                "architecture": config.architecture,
                "base_model": config.base_model,
                "text_edge_list": config.text_edge_list,
            },
            loss_target=config.loss_target,
        )

    # Log all training config parameters to wandb
    if wandb.run is not None:
        wandb.config.update(
            omegaconf.OmegaConf.to_container(config, resolve=True),
            allow_val_change=True,
        )

    eval_samples_by_graph = data.load_eval_samples_by_graph(
        config.eval_data, config.eval_num_graphs
    )
    trainer.add_callback(
        callbacks.EvalCallback(
            eval_samples_by_graph,
            tokenizer=tokenizer,
            use_icl=config.eval_use_icl,
            include_edge_list=(config.text_edge_list == "present"),
            eval_epoch_interval=config.eval_epoch_interval,
        )
    )

    if config.architecture in ("rpearl_llm", "rpearl_gt_llm", "gt_llm"):
        trainer.add_callback(callbacks.GradientDebugCallback())
    elif config.architecture == "composite_graph_gt":
        # Per-component grad norms, GT output magnitude, gate value, injection count.
        trainer.add_callback(callbacks.GradientDebugCallback())
        # Fiedler, scene-mass, gate, contrib-ratio diagnostics (+ visualizer if enabled).
        trainer.add_callback(
            callbacks.AugGraphDebugCallback(
                enable_visualizer=config.enable_visualizer,
                visualizer_dir=os.path.join(output_dir, "visuals"),
            )
        )
        if config.lora_warmup_steps > 0:
            trainer.add_callback(callbacks.LoraWarmupCallback(config.lora_warmup_steps))
        if getattr(config, "lam_c_warmup_steps", 0) > 0 and getattr(
            config, "c_bias", False
        ):
            trainer.add_callback(
                callbacks.LamCWarmupCallback(config.lam_c_warmup_steps)
            )

    # no_train: evaluate the untrained base model zero-shot instead of training.
    if config.no_train:
        print("[no_train] Skipping optimization — evaluating the base model zero-shot.")
        evaluate.run_zero_shot_eval(
            trainer.model, tokenizer, config, sft_args.output_dir, eval_samples_by_graph
        )
    else:
        trainer.train()

    # Save model artifacts
    trainer.save_model()
    tokenizer.save_pretrained(sft_args.output_dir)

    if config.post_train_eval_graphs:
        evaluate.run_post_train_cross_eval(
            trainer.model,
            tokenizer,
            config,
            sft_args.output_dir,
        )

    return trainer


# ----------------------------
# Utilities
# ----------------------------
def _bf16_supported() -> bool:
    """Conservative check for bfloat16 support."""
    try:
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:
        return False


def _fp16_supported() -> bool:
    """Conservative check for fp16 support."""
    if _bf16_supported():
        return False
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def _model_short_name(base_model: str) -> str:
    """Extract a short filesystem-safe slug from a HuggingFace model ID.

    Examples:
        meta-llama/Llama-3.1-8B-Instruct → llama-3.1-8b
        Qwen/Qwen2.5-0.5B-Instruct       → qwen2.5-0.5b
    """
    name = base_model.split("/")[-1]  # drop org prefix
    name = re.sub(r"-[Ii]nstruct$", "", name)  # drop -Instruct suffix
    name = name.lower()
    name = re.sub(r"-+", "-", name)  # collapse consecutive hyphens
    return name


def _ensure_pad_tokens(tokenizer: PreTrainedTokenizer, model: PreTrainedModel) -> None:
    """Ensure tokenizer and model config have a pad token for training.

    Many chat models ship without a dedicated PAD token; we alias EOS instead.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id


def _setup_wandb(wandb_project: str, wandb_run_name: str, wandb_tag: str) -> str:
    os.environ["WANDB_PROJECT"] = wandb_project
    os.environ["WANDB_RUN_GROUP"] = wandb_tag
    os.environ["WANDB_TAGS"] = wandb_tag

    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        tags=[wandb_tag],
        group=wandb_tag,
    )
    return wandb.run.id


def _construct_output_dir(config: omegaconf.DictConfig, wandb_run_id: str) -> str:
    """Checkpoint output dir ``{checkpoint_dir}/{subdir}``.

    ``subdir`` is the ``--save_name`` override or the auto-generated
    ``{name}_{architecture}_{model_slug}_r{r}[_4bit]``. The wandb run ID is always appended.
    """
    if config.save_name is not None:
        subdir = f"{config.save_name}_{wandb_run_id}"
    else:
        model_slug = _model_short_name(config.base_model)
        subdir = (
            f"{config.name}_{config.architecture}_{model_slug}_r{config.r}"
            + ("_4bit" if config.bit4 else "")
            + f"_{wandb_run_id}"
        )
    return str(os.path.join(config.checkpoint_dir, subdir))


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

    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "gnn_config.json"), "w") as f:
            json.dump(self.gnn_config, f, indent=2)
        if self.gnn_config.get("architecture") == "composite_graph_gt":
            # Save the Graph Transformer (R-PEARL inside) and the cold-start gate.
            gnn_state = {
                "gt_model": self.model.gt_model.state_dict(),
                "injection": self.model.injection.state_dict(),
            }
            # In-attention injection variant: persist the dedicated q/k(/v) projections.
            if hasattr(self.model, "pe_q_proj"):
                gnn_state["pe_q_proj"] = self.model.pe_q_proj.state_dict()
                gnn_state["pe_k_proj"] = self.model.pe_k_proj.state_dict()
                if getattr(self.model, "pe_v_proj", None) is not None:
                    gnn_state["pe_v_proj"] = self.model.pe_v_proj.state_dict()
            # c_bias: persist scalar gains λ_C/λ_S/λ_V.
            if getattr(self.model, "c_bias", False):
                gnn_state["c_bias_gains"] = {
                    name: getattr(self.model, name).detach().cpu()
                    for name in ("lam_c", "lam_psi", "lam_v")
                    if getattr(self.model, name, None) is not None
                }
            torch.save(gnn_state, os.path.join(output_dir, "gnn_weights.pt"))
            torch.save(
                {
                    "rpearl": self.model.gt_model.pe_model.state_dict(),
                },
                os.path.join(output_dir, "rpearl_weights.pt"),
            )
        elif self.gnn_config.get("architecture") == "graph_mask_llm":
            # Parameter-free: mask rebuilt from config; gnn_config.json + LoRA adapter suffice.
            pass
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

    Persists a ``train_config.json`` (the plain-LLM analogue of the graph trainer's
    ``gnn_config.json``) so the standalone eval boundary can recover the train-time
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


# ----------------------------
# Config (Hydra)
# ----------------------------
# All fields, defaults, and per-field docs live in the experiments/e9_hydra_training
# config tree; the composed OmegaConf object is consumed directly (no static dataclass).
def _validate_config(config: omegaconf.DictConfig) -> None:
    """Coerce/validate composed fields (replaces the former TrainConfig.__post_init__)."""
    config.eps = float(config.eps)
    if config.loss_target not in ("all", "responses", "edge_list"):
        raise ValueError(
            f"loss_target must be 'all', 'responses', or 'edge_list', got {config.loss_target!r}"
        )
    if config.loss_target == "edge_list" and config.text_edge_list != "present":
        raise ValueError(
            "loss_target='edge_list' requires text_edge_list='present'."
        )


# ----------------------------
# CLI (Hydra)
# ----------------------------
@hydra.main(
    version_base=None,
    config_path="../../../experiments/e9_hydra_training",
    config_name="config",
)
def main(config: omegaconf.DictConfig) -> None:
    """Compose a run from the config groups, e.g. `... architecture=gt_llm gt=L5_d4096`."""
    _validate_config(config)
    train_model(config)


if __name__ == "__main__":
    main()
