import os
import re
import warnings

from prism.data import data
from prism.eval import callbacks
from prism.eval import checkpoint
from prism.eval import evaluate
# Trainer classes live in trainers.py; re-exported here so existing imports
# (tests, diagnostics) keep resolving through train_v3.
from prism.training.trainers import (  # noqa: F401
    _LOSS_TARGET_COLUMN,
    BaselineSFTTrainer,
    GraphSFTTrainer,
    LossTargetMixin,
)
from prism.models import architectures
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
        config.wandb.project, config.wandb.run_name, config.wandb.tag,
        report_to=config.trainer.report_to,
    )
    # Dir: {name}_{architecture}_{model_slug}_r{r}[_4bit]
    output_dir = _construct_output_dir(config, wandb_run_id)

    if os.path.isdir(output_dir) and os.listdir(output_dir) and not config.trainer.overwrite_ok:
        raise RuntimeError(
            f"Checkpoint directory already exists and is non-empty: {output_dir}\n"
            f"Set overwrite_ok: true in your config to allow overwriting, "
            f"or delete the directory manually."
        )

    # Quantization / dtype
    bnb_config = None
    if config.trainer.bit4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
            if _bf16_supported()
            else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    # Model & tokenizer
    device_map = {"": 0} if config.trainer.device >= 0 else "auto"
    llm = AutoModelForCausalLM.from_pretrained(
        config.model.path,
        torch_dtype="auto",
        quantization_config=bnb_config,  # None if not 4-bit
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model.path, use_fast=True)
    _ensure_pad_tokens(tokenizer, llm)
    tokenizer.padding_side = "right"  # ty: ignore[invalid-assignment]

    model, collator = architectures.build_planner_model(
        config.gnn, llm, tokenizer,
        disable_graph_token_rope=config.model.disable_graph_token_rope,
        freeze_llm=config.trainer.freeze_llm,
    )

    # Which token positions carry the graph channel during training forwards:
    # 'full_sequence' (historical) injects answer-side node mentions too — a channel
    # generation never has; 'prompt_only' clamps maps at answer_start to match
    # generation exactly; 'exclude_supervised' (e12) subtracts the loss-target
    # positions (from trainer.loss_target's index column) so no supervised token
    # carries its own node's channel — needed when the supervised block is inside
    # the prompt (loss_target='edge_list'), where prompt_only doesn't reach.
    valid_scopes = ("full_sequence", "prompt_only", "exclude_supervised")
    if config.data.injection_scope not in valid_scopes:
        raise ValueError(
            f"data.injection_scope must be one of {valid_scopes}, "
            f"got {config.data.injection_scope!r}")
    collator.injection_scope = config.data.injection_scope
    if config.data.injection_scope == "exclude_supervised":
        if config.trainer.loss_target not in _LOSS_TARGET_COLUMN:
            raise ValueError(
                "injection_scope='exclude_supervised' requires a masked loss_target "
                f"('responses' or 'edge_list'), got {config.trainer.loss_target!r} — "
                "with loss_target='all' every position is supervised and the "
                "exclusion would empty the injection map."
            )
        collator.supervised_positions_key = _LOSS_TARGET_COLUMN[config.trainer.loss_target]
        print(f"[data] injection_scope=exclude_supervised "
              f"(excluding {collator.supervised_positions_key})")
    else:
        print(f"[data] injection_scope={config.data.injection_scope}")

    # --- Multistage init: weight-only carry-over from a prior stage (NOT HF resume). --
    # Load PE first (lives outside the LoRA adapter), then attach the carried adapter.
    is_graph_arch = config.gnn.arch in (
        "rpearl_llm",
        "rpearl_gt_llm",
        "gt_llm",
        "graph_mask_llm",
        "learnable_graph_mask",
    )
    if (config.trainer.init_pe_from or config.trainer.init_lora_from) and not is_graph_arch:
        raise ValueError(
            "init_pe_from / init_lora_from are only supported for graph architectures."
        )
    # Load the pretrained PE encoder first; a later init_pe_from carry (stage 2/3)
    # overwrites the whole pe_model, including this submodule.
    if config.gnn.arch == "learnable_graph_mask" and config.gnn.pe_gt_from:
        model_loaders.load_navigator_pe_into(
            model, config.gnn.pe_gt_from, config.gnn.semantic_gt_from)
    if config.trainer.init_pe_from:
        model_loaders.load_pe_weights_into(
            model, config.trainer.init_pe_from, config.gnn.arch
        )
    attach_existing_adapter = config.trainer.init_lora_from is not None
    if attach_existing_adapter:
        model = PeftModel.from_pretrained(
            model, config.trainer.init_lora_from, is_trainable=not config.trainer.freeze_lora
        )
        if config.trainer.gradient_checkpointing and not config.trainer.freeze_lora:
            # Replicate TRL's get_peft_model behavior so LoRA gradients flow back
            # through the frozen base under gradient checkpointing.
            model.enable_input_require_grads()
        print(
            f"[multistage] attached {'frozen' if config.trainer.freeze_lora else 'trainable'} "
            f"LoRA adapter from {config.trainer.init_lora_from}"
        )

    train_dataset, eval_dataset = data.load_and_split_dataset(config.data, tokenizer)

    # LoRA config; for multimodal bases, exclude vision/audio towers (see architectures.peft_tower_exclude).
    tower_exclude = architectures.peft_tower_exclude(model)
    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        bias="none",
        target_modules=list(config.lora.target_modules),
        exclude_modules=tower_exclude,
        task_type="CAUSAL_LM",
    )
    if tower_exclude:
        print(
            f"[peft] multimodal base detected — excluding LoRA targets matching "
            f"{tower_exclude!r} (vision/audio towers)"
        )

    optim = "adamw_bnb_8bit" if config.trainer.bit4 else "adamw_torch_fused"

    # SFT trainer configuration. Computed defaults first; then trainer.sft (a free-form
    # dict) is merged LAST, so ANY SFTConfig/TrainingArguments field can be set from a
    # config or the CLI without a dedicated code path:
    #   +trainer.sft.lr_scheduler_type=cosine +trainer.sft.logging_steps=5
    sft_kwargs = dict(
        dataset_num_proc=config.trainer.dataset_num_proc,
        dataloader_num_workers=config.trainer.dataloader_num_workers,
        packing=False,  # packing combines multiple examples into a single input_id. Keep disabled to avoid graph contamination.
        max_length=config.data.max_seq_length,
        per_device_train_batch_size=config.trainer.per_device_train_batch_size,
        per_device_eval_batch_size=config.trainer.per_device_eval_batch_size,
        gradient_accumulation_steps=config.trainer.gradient_accumulation_steps,
        warmup_steps=config.trainer.warmup_steps,
        num_train_epochs=config.trainer.epochs,
        max_steps=config.trainer.max_steps,
        learning_rate=config.trainer.learning_rate,
        weight_decay=config.trainer.weight_decay,
        lr_scheduler_type="linear",
        logging_steps=15,
        gradient_checkpointing=config.trainer.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # precision
        fp16=_fp16_supported(),
        bf16=_bf16_supported(),
        # misc
        seed=3407,
        output_dir=output_dir,
        report_to=config.trainer.report_to,
        run_name=config.wandb.run_name,
        optim=optim,
        remove_unused_columns=False,
        # Checkpointing / Validation: step-based when max_steps is set (dev), else epoch-based.
        save_strategy="steps" if config.trainer.max_steps > 0 else "epoch",
        save_steps=max(1, config.trainer.max_steps // 2) if config.trainer.max_steps > 0 else 500,
        save_total_limit=3,
        eval_strategy="steps" if config.trainer.max_steps > 0 else "epoch",
        eval_steps=max(1, config.trainer.max_steps // 2) if config.trainer.max_steps > 0 else 0.5,
        do_eval=True,
    )
    sft_overrides = omegaconf.OmegaConf.to_container(config.trainer.sft, resolve=True) or {}
    if sft_overrides:
        print(f"[trainer] SFTConfig overrides from trainer.sft: {sft_overrides}")
    sft_args = SFTConfig(**{**sft_kwargs, **sft_overrides})

    if is_graph_arch:
        # gnn_config is dumped into train_config.json (metadata top-level, params under
        # "gnn") and read back by loaders at eval, so the on-disk key NAMES must stay stable. These keys map 1:1 onto config.gnn fields;
        # the remapped/cross-section ones (architecture, base_model, text_edge_list) and
        # the arch-conditional groups are spelled out.
        _direct = (
            "pe_hidden_channels", "pe_num_layers", "d_model", "num_samples", "dropout",
            "k_pe", "use_layer_norm", "eps", "pe_gain_init", "use_pe_norm", "pe_node_features",
        )
        gnn_config = {
            "architecture": config.gnn.arch,
            "base_model": config.model.path,
            "text_edge_list": config.data.text_edge_list,
            "injection_scope": config.data.injection_scope,
            # Two-group LR: structural (GT / PE) params train at structural_lr_mult × base LR
            # (GraphSFTTrainer.create_optimizer). Default 1.0 = no boost.
            "structural_lr_mult": config.gnn.structural_lr_mult,
            **{k: config.gnn[k] for k in _direct},
            **({k: config.gnn[k] for k in ("k_gt", "gt_num_layers", "gt_heads")}
               if config.gnn.arch in ("rpearl_gt_llm", "gt_llm", "learnable_graph_mask") else {}),
            # learnable_graph_mask navigator: record both sources so eval rebuilds the NavigatorPE.
            **({"pe_gt_from": config.gnn.pe_gt_from, "semantic_gt_from": config.gnn.semantic_gt_from}
               if config.gnn.arch == "learnable_graph_mask" and config.gnn.pe_gt_from else {}),
            # graph_mask_llm / learnable_graph_mask adjacency (A) + fold + scope rebuild params.
            **({k: config.gnn[k] for k in ("mask_k_hops", "mask_symmetrize", "mask_use_edges",
                                           "mask_buggy_causal_fold", "mask_layer_scope")}
               if config.gnn.arch in ("graph_mask_llm", "learnable_graph_mask") else {}),
            # learnable_graph_mask extra params (read back by loaders for eval).
            **({k: config.gnn[k] for k in ("mask_alpha", "mask_psi_scale")}
               if config.gnn.arch == "learnable_graph_mask" else {}),
        }
        trainer = GraphSFTTrainer(
            model=model,
            data_collator=collator,
            processing_class=tokenizer,
            # No fresh adapter when one was carried forward (it's already attached)
            # or when freeze_llm requests a PE-only run on the raw base.
            peft_config=lora_config
            if (not config.trainer.freeze_llm and not attach_existing_adapter)
            else None,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=sft_args,
            gnn_config=gnn_config,
            freeze_pe=config.trainer.freeze_pe,
            loss_target=config.trainer.loss_target,
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
                "architecture": config.gnn.arch,
                "base_model": config.model.path,
                "text_edge_list": config.data.text_edge_list,
            },
            loss_target=config.trainer.loss_target,
        )

    # Log all training config parameters to wandb
    if wandb.run is not None:
        wandb.config.update(
            omegaconf.OmegaConf.to_container(config, resolve=True),
            allow_val_change=True,
        )

    eval_samples_by_graph = data.load_eval_samples_by_graph(
        config.eval.data, config.eval.num_graphs
    )
    trainer.add_callback(
        callbacks.EvalCallback(
            eval_samples_by_graph,
            tokenizer=tokenizer,
            use_icl=config.eval.use_icl,
            include_edge_list=(config.data.text_edge_list == "present"),
            eval_epoch_interval=config.eval.epoch_interval,
        )
    )

    # Per-component grad norms, GT output magnitude, gate value, injection count.
    if config.trainer.gradient_debug:
        trainer.add_callback(callbacks.GradientDebugCallback())

    cross_eval_dir = os.path.join(sft_args.output_dir, "eval_logs", "cross_eval")

    # no_train: evaluate the untrained base model zero-shot instead of training. Uses
    # the in-memory base model over the train-time eval set (no graph-file provenance
    # is carried for that capped set, so the JSONs record graph_file=null).
    if config.trainer.no_train:
        print("[no_train] Skipping optimization — evaluating the base model zero-shot.")
        evaluate.evaluate_model(
            trainer.model,
            tokenizer,
            eval_samples_by_graph,
            output_dir=cross_eval_dir,
            text_edge_list=config.data.text_edge_list,
            use_icl=config.eval.use_icl,
            architecture=config.gnn.arch,
            checkpoint_label=sft_args.output_dir,
        )
    else:
        trainer.train()

    # Save model artifacts
    trainer.save_model()
    tokenizer.save_pretrained(sft_args.output_dir)

    # Post-training cross-eval. Reload the just-saved checkpoint from disk (not the
    # in-memory model) so this doubles as a save→load round-trip check of the adapter
    # + gnn weights — the boundary an in-memory eval would never exercise.
    if config.eval.post_train_graphs:
        samples_by_graph, graph_file_by_name = data.load_samples_by_graph(
            config.eval.post_train_graphs
        )
        eval_model, eval_tokenizer, _ = checkpoint.load_checkpoint(
            sft_args.output_dir, four_bit=config.trainer.bit4, device=config.trainer.device
        )
        evaluate.evaluate_model(
            eval_model,
            eval_tokenizer,
            samples_by_graph,
            graph_file_by_name,
            output_dir=cross_eval_dir,
            text_edge_list=config.data.text_edge_list,
            use_icl=config.eval.use_icl,
            architecture=config.gnn.arch,
            checkpoint_label=sft_args.output_dir,
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
        google/gemma-4-31B-it → gemma-4-31b
        google/gemma-4-12B-it → gemma-4-12b
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


def _setup_wandb(wandb_project: str, wandb_run_name: str, wandb_tag: str,
                 report_to: str = "wandb") -> str:
    """Create the tracked wandb run (its id names the run dir).

    ``report_to != 'wandb'`` (e.g. smoke runs) creates NO remote run — a local
    random id names the dir instead, and ``wandb.run`` stays None so every
    downstream ``if wandb.run is not None`` guard no-ops.
    """
    if report_to != "wandb":
        import uuid
        return uuid.uuid4().hex[:8]
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
    if config.trainer.save_name is not None:
        subdir = f"{config.trainer.save_name}_{wandb_run_id}"
    else:
        model_slug = _model_short_name(config.model.path)
        subdir = (
            f"{config.name}_{config.gnn.arch}_{model_slug}_r{config.lora.r}"
            + ("_4bit" if config.trainer.bit4 else "")
            + f"_{wandb_run_id}"
        )
    return str(os.path.join(config.trainer.checkpoint_dir, subdir))


# ----------------------------
# Config (Hydra)
# ----------------------------
# All fields, defaults, and per-field docs live in experiments/base_config.yaml;
# the composed OmegaConf object is consumed directly (no static dataclass).
def _validate_config(config: omegaconf.DictConfig) -> None:
    """Coerce/validate composed fields (replaces the former TrainConfig.__post_init__)."""
    config.gnn.eps = float(config.gnn.eps)
    if config.gnn.arch == "learnable_graph_mask":
        if not 0.0 <= float(config.gnn.mask_alpha) < 1.0:
            raise ValueError(
                f"gnn.mask_alpha must be in [0, 1) (alpha=1 kills the GT gradient), "
                f"got {config.gnn.mask_alpha}")
        if config.gnn.mask_layer_scope not in ("all", "dense"):
            raise ValueError(
                f"gnn.mask_layer_scope must be 'all' or 'dense', got {config.gnn.mask_layer_scope!r}")
        if config.gnn.mask_psi_scale not in ("cosine", "inv_sqrt_d"):
            raise ValueError(
                f"gnn.mask_psi_scale must be 'cosine' or 'inv_sqrt_d', got {config.gnn.mask_psi_scale!r}")
        if config.gnn.mask_psi_scale == "cosine" and not config.gnn.mask_use_edges:
            raise ValueError(
                "gnn.mask_psi_scale='cosine' with gnn.mask_use_edges=false is degenerate "
                "(constant mask, zero GT gradient). Use mask_psi_scale='inv_sqrt_d' or keep edges.")
    if config.gnn.pe_gt_from or config.gnn.semantic_gt_from:
        if config.gnn.arch != "learnable_graph_mask":
            raise ValueError(
                "gnn.pe_gt_from / semantic_gt_from are only supported for arch='learnable_graph_mask'.")
        if config.gnn.semantic_gt_from and not config.gnn.pe_gt_from:
            raise ValueError(
                "gnn.semantic_gt_from requires gnn.pe_gt_from (the navigator needs both). "
                "Set pe_gt_from alone for a GT-only Ψ producer.")
    if config.trainer.loss_target not in ("all", "responses", "edge_list"):
        raise ValueError(
            f"loss_target must be 'all', 'responses', or 'edge_list', got {config.trainer.loss_target!r}"
        )
    if config.trainer.loss_target == "edge_list" and config.data.text_edge_list != "present":
        raise ValueError(
            "loss_target='edge_list' requires text_edge_list='present'."
        )


# ----------------------------
# CLI (Hydra)
# ----------------------------
@hydra.main(
    version_base=None,
    config_path="../../../experiments",
    config_name="base_config",
)
def main(config: omegaconf.DictConfig) -> None:
    """Compose a run from the config groups, e.g. `... architecture=gt_llm gt=L5_d4096`."""
    _validate_config(config)
    train_model(config)


if __name__ == "__main__":
    main()
