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
from prism.models import gnn_llm
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


# Architectures whose graph channel is a bare Ψ producer (gt.build_psi_producer), so
# they accept the notebook's pretrained navigator weights via gnn.pe_gt_from /
# gnn.semantic_gt_from: the mask (Ψ Ψᵀ attention bias) and WIRE (Ψ as q/k rotation
# angles). Everything else consumes Ψ through pe_proj into the LLM hidden space and has
# no navigator wiring.
_PSI_PRODUCER_ARCHS = ("learnable_graph_mask", "wire_llm")


def train_model(config: omegaconf.DictConfig):
    # Silence the AccumulateGrad stream-mismatch advisory. It is a performance note
    # (redundant syncing, and CUDA-graph capture we never do), not a correctness one —
    # accumulation still lands in the right param.grad. The cross-iteration node it
    # reports comes from accelerate's multi-device dispatch under device_map="auto";
    # our own retentions are dropped in CompositeWireGraphLLM._arm. getattr so a torch
    # without the toggle still runs.
    _silence = getattr(torch.autograd.graph,
                       "set_warn_on_accumulate_grad_stream_mismatch", None)
    if _silence is not None:
        _silence(False)
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
    valid_scopes = ("full_sequence", "prompt_only", "exclude_supervised",
                    "decode_consistent")
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

    # GNN edge weighting over the scene-graph adjacency: "gaussian" (historical
    # Gaussian affinity from Euclidean distances) or "binary" (no edge_weight —
    # plain SBM adjacency). Threaded to the train collator here and to every eval
    # path below; recorded in train_config.json.
    valid_edge_weights = ("gaussian", "binary")
    if config.data.edge_weights not in valid_edge_weights:
        raise ValueError(
            f"data.edge_weights must be one of {valid_edge_weights}, "
            f"got {config.data.edge_weights!r}")
    collator.edge_weights = config.data.edge_weights
    print(f"[data] edge_weights={config.data.edge_weights}")

    # --- Multistage init: weight-only carry-over from a prior stage (NOT HF resume). --
    # Load PE first (lives outside the LoRA adapter), then attach the carried adapter.
    is_graph_arch = config.gnn.arch in (
        "rpearl_llm",
        "rpearl_gt_llm",
        "gt_llm",
        "graph_mask_llm",
        "learnable_graph_mask",
        "wire_llm",
    )
    if (config.trainer.init_pe_from or config.trainer.init_lora_from) and not is_graph_arch:
        raise ValueError(
            "init_pe_from / init_lora_from are only supported for graph architectures."
        )
    # Load the pretrained PE encoder first; a later init_pe_from carry (stage 2/3)
    # overwrites the whole pe_model, including this submodule.
    if config.gnn.arch in _PSI_PRODUCER_ARCHS and config.gnn.pe_gt_from:
        model_loaders.load_navigator_pe_into(
            model, config.gnn.pe_gt_from, config.gnn.semantic_gt_from)
    if config.trainer.init_pe_from:
        model_loaders.load_pe_weights_into(
            model, config.trainer.init_pe_from, config.gnn.arch
        )
    attach_existing_adapter = config.trainer.init_lora_from is not None
    if config.trainer.freeze_lora and not attach_existing_adapter:
        # freeze_lora is implemented via PeftModel.from_pretrained(is_trainable=...), so it
        # has no effect on a FRESH adapter. Silently training the LoRA of a run configured
        # to freeze it is exactly the kind of misconfigured baseline we must not produce.
        raise ValueError(
            "trainer.freeze_lora=true requires trainer.init_lora_from: it is applied when "
            "an EXISTING adapter is attached (is_trainable=False). With a fresh adapter "
            "there is nothing to freeze and the LoRA would train regardless.")
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
            # LOAD-BEARING at reload: selects the MagNet probe backbone, whose weights
            # are a different state_dict shape than the GCN one — a missing key would
            # fail the rebuild loudly rather than silently, but record it anyway.
            "directed",
            # LOAD-BEARING at train time: GraphSFTTrainer reads it off THIS dict, so a
            # missing key meant `.get(..., True)` won and the projection ran regardless of
            # the config — MEASURED on run 3hasg0sn, which passed enforce_beta_bound=false
            # and still projected all 5 MagChebConv layers to sum_k ||H_k||_2 = 1.
            "enforce_beta_bound",
            # LOAD-BEARING at reload: 'log' vs 'linear' is a DIFFERENT function of the
            # same weights (multiplicative gate vs raw covariance), so a missing key would
            # score a log-trained checkpoint under the linear bias.
            "mask_bias_mode",
            # LOAD-BEARING at reload: selects the GT's input_proj (Φ(X + P; T)). The
            # pe_model load is strict=False, so a missing key would silently drop it.
            "fuse_node_features",
        )
        gnn_config = {
            "architecture": config.gnn.arch,
            "base_model": config.model.path,
            "text_edge_list": config.data.text_edge_list,
            # Cross-section (config.model), and LOAD-BEARING at reload: loaders pass it
            # to every arch that honours it. It was previously absent from this dict, so
            # `gnn_cfg.get("disable_graph_token_rope", False)` always won and a run
            # trained with identity-RoPE was silently evaluated WITH normal RoPE.
            "disable_graph_token_rope": config.model.disable_graph_token_rope,
            # Prompt-format provenance: whether the targets teach SPINE tool calling and
            # how many few-shot examples the prompts carried (eval must match).
            "spine_tools": config.data.spine_tools,
            "icl_examples": config.data.icl_examples,
            "injection_scope": config.data.injection_scope,
            "edge_weights": config.data.edge_weights,
            # Two-group LR: structural (GT / PE) params train at structural_lr_mult × base LR
            # (GraphSFTTrainer.create_optimizer). Default 1.0 = no boost.
            "structural_lr_mult": config.gnn.structural_lr_mult,
            **{k: config.gnn[k] for k in _direct},
            **({k: config.gnn[k] for k in ("k_gt", "gt_num_layers", "gt_heads")}
               if config.gnn.arch in ("rpearl_gt_llm", "gt_llm", "learnable_graph_mask",
                                      "wire_llm") else {}),
            # wire_llm: every switch the rotation depends on, so eval rebuilds it exactly.
            # wire_omega_seed is provenance only — omega itself is checkpointed.
            **({k: config.gnn[k] for k in (
                "wire_layer_scope", "wire_sigma_init", "wire_freeze_sigma",
                "wire_omega_seed", "wire_rotate_nope_planes", "wire_max_angle",
                "wire_decode", "wire_vanilla", "wire_vanilla_omega_init")}
               if config.gnn.arch == "wire_llm" else {}),
            # wire_composite: the composite-graph arm is a DIFFERENT function of the same
            # weights (Ψ over the token cycle + scene + anchor, RoPE replaced on the
            # block), so every switch its graph depends on is recorded too.
            **({k: config.gnn[k] for k in (
                "wire_composite", "wire_magnet_r", "wire_context_window",
                "wire_cycle_weight", "wire_cycle_causal", "wire_crosslink_weight",
                "wire_crosslink_bidirectional", "wire_anchor_weight", "learn_r")}
               if config.gnn.arch == "wire_llm" and config.gnn.get("wire_composite")
               else {}),
            # Navigator Ψ producer (learnable_graph_mask / wire_llm): record BOTH sources.
            # pe_gt_from is provenance only (the Ψ topology is a standalone GT either way);
            # semantic_gt_from is load-bearing — it is what makes the eval rebuild pick the
            # legacy two-stage gt.TwoStagePE instead of the PE-only Ψ, so dropping it here
            # would silently reload an old run as a different function.
            **({"pe_gt_from": config.gnn.pe_gt_from, "semantic_gt_from": config.gnn.semantic_gt_from}
               if config.gnn.arch in _PSI_PRODUCER_ARCHS and config.gnn.pe_gt_from else {}),
            # Ψ-producer probe pooling (WHERE E_q is taken). LOAD-BEARING at reload and
            # invisible in the weights: pe_pool="gt" is a DIFFERENT function of the SAME
            # state_dict, so an eval rebuild falling back to gt.build_psi_producer's
            # "pe" default would score a different model with a clean, silent load.
            **({"pe_pool": config.gnn.pe_pool}
               if config.gnn.arch in _PSI_PRODUCER_ARCHS else {}),
            # graph_mask_llm / learnable_graph_mask adjacency (A) + fold + scope rebuild params.
            **({k: config.gnn[k] for k in ("mask_k_hops", "mask_symmetrize", "mask_use_edges",
                                           "mask_buggy_causal_fold", "mask_layer_scope")}
               if config.gnn.arch in ("graph_mask_llm", "learnable_graph_mask") else {}),
            # learnable_graph_mask extra params (read back by loaders for eval).
            **({k: config.gnn[k] for k in ("mask_alpha", "mask_psi_scale")}
               if config.gnn.arch == "learnable_graph_mask" else {}),
            # mask_composite: the covariance arm is a DIFFERENT function of the same
            # weights (beta * C_tok over the composite token block instead of the Psi
            # Psi^T node-pair gate), so every switch its graph and its probe statistic
            # depend on is recorded too. learn_r/directed/pe_pool ride the blocks above.
            **({k: config.gnn[k] for k in (
                "mask_composite", "mask_cycle_size", "mask_beta_init", "mask_magnet_r",
                "mask_cycle_weight", "mask_cycle_causal", "mask_crosslink_weight",
                "mask_anchor_enabled", "mask_anchor_weight", "mask_decode_refresh",
                "learn_r", "hidden_norm",
                "phase", "shift", "probe_distribution", "fixed_seed_mode",
                "fixed_seed_value", "center_second_moment", "cache_pe")}
               if config.gnn.arch == "learnable_graph_mask" and config.gnn.get("mask_composite")
               else {}),
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
                "spine_tools": config.data.spine_tools,
                "icl_examples": config.data.icl_examples,
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
            edge_weights=config.data.edge_weights,
            injection_scope=config.data.injection_scope,
        )
    )

    # Per-component grad norms, GT output magnitude, gate value, injection count.
    if config.trainer.gradient_debug:
        trainer.add_callback(callbacks.GradientDebugCallback())

    # Magnetic composite arms: watch the spectral-injectivity margin delta = dist(2rc, Z).
    # A collision is direction-destroying and leaves NO signature in the loss, and with a
    # learnable r the margin is a sawtooth of period 1/(2c) that one Adam step can cross —
    # so this is the only instrument for it. c is the token-cycle length, which is not
    # recoverable from the module: it comes from whichever arm's config owns it, never a
    # guess. Diagnostic only; it never writes r.
    _cycle_c = (config.gnn.get("mask_cycle_size") if config.gnn.get("mask_composite")
                else config.gnn.get("wire_context_window") if config.gnn.get("wire_composite")
                else None)
    if _cycle_c and config.gnn.get("directed"):
        trainer.add_callback(callbacks.ChargeDegeneracyCallback(cycle_length=int(_cycle_c)))

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
            edge_weights=config.data.edge_weights,
            injection_scope=config.data.injection_scope,
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
            edge_weights=config.data.edge_weights,
            injection_scope=config.data.injection_scope,
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
        if config.gnn.mask_layer_scope not in ("all", "dense", "dense_top_half", "dense_first"):
            raise ValueError(
                f"gnn.mask_layer_scope must be one of ('all', 'dense', 'dense_top_half', "
                f"'dense_first'), got {config.gnn.mask_layer_scope!r}")
        if config.gnn.mask_psi_scale not in ("cosine", "inv_sqrt_d"):
            raise ValueError(
                f"gnn.mask_psi_scale must be 'cosine' or 'inv_sqrt_d', got {config.gnn.mask_psi_scale!r}")
        if config.gnn.mask_psi_scale == "cosine" and not config.gnn.mask_use_edges:
            raise ValueError(
                "gnn.mask_psi_scale='cosine' with gnn.mask_use_edges=false is degenerate "
                "(constant mask, zero GT gradient). Use mask_psi_scale='inv_sqrt_d' or keep edges.")
        if config.gnn.get("mask_composite"):
            # The three preconditions MagCompGraphLLM.__init__ asserts, checked at compose
            # time so a mis-set value fails before the LLM is even loaded. pe_pool='gt' is
            # what puts T inside E_q; directed/learn_r are what give the token cycle a phase.
            for key, want in (("directed", True), ("learn_r", True), ("pe_pool", "gt")):
                if config.gnn.get(key) != want:
                    raise ValueError(
                        f"gnn.mask_composite=true requires gnn.{key}={want!r}, got "
                        f"{config.gnn.get(key)!r}: C = E_q[Φ'Φ'ᵀ] − ΨΨᵀ over "
                        "Φ' = T(Φ(q; L̄^(r), H)) needs the magnetic backbone (directed), "
                        "its charge as a free parameter (learn_r), and the blocks INSIDE "
                        "the probe expectation (pe_pool='gt').")
            if int(config.gnn.mask_cycle_size) < 2:
                raise ValueError(
                    f"gnn.mask_cycle_size must be >= 2, got {config.gnn.mask_cycle_size}")
            if not 0.0 < float(config.gnn.mask_magnet_r) < 0.25:
                raise ValueError(
                    "gnn.mask_magnet_r must be in (0, 0.25) (sigmoid-reparameterized; the "
                    f"endpoints have no logit), got {config.gnn.mask_magnet_r}")
            if int(config.gnn.get("mask_decode_refresh", 8)) < 0:
                raise ValueError(
                    "gnn.mask_decode_refresh must be >= 0 (tokens between decode-time "
                    "composite rebuilds; 0 freezes the graph at the prompt), got "
                    f"{config.gnn.get('mask_decode_refresh')}")
    if config.gnn.arch == "wire_llm":
        if config.gnn.wire_layer_scope not in ("all", "dense", "dense_top_half", "dense_first"):
            raise ValueError(
                f"gnn.wire_layer_scope must be one of ('all', 'dense', 'dense_top_half', "
                f"'dense_first'), got {config.gnn.wire_layer_scope!r}")
        # "error" stays ACCEPTED (checkpoints and configs predating decode-time key
        # rotation recorded it); WireGraphLLM normalises it to "rotate" and warns.
        if config.gnn.wire_decode not in gnn_llm.WIRE_DECODE_MODES + tuple(
                gnn_llm.WIRE_DECODE_LEGACY):
            raise ValueError(
                f"gnn.wire_decode must be one of {gnn_llm.WIRE_DECODE_MODES} "
                f"(legacy: {tuple(gnn_llm.WIRE_DECODE_LEGACY)}), "
                f"got {config.gnn.wire_decode!r}")
        if float(config.gnn.wire_sigma_init) <= 0:
            raise ValueError(
                f"gnn.wire_sigma_init must be > 0, got {config.gnn.wire_sigma_init}")
        if float(config.gnn.wire_max_angle) <= 0:
            raise ValueError(
                f"gnn.wire_max_angle must be > 0, got {config.gnn.wire_max_angle}")
        if config.gnn.wire_vanilla_omega_init not in gnn_llm.WIRE_VANILLA_OMEGA_INITS:
            raise ValueError(
                "gnn.wire_vanilla_omega_init must be one of "
                f"{gnn_llm.WIRE_VANILLA_OMEGA_INITS}, got "
                f"{config.gnn.wire_vanilla_omega_init!r}")
        if config.gnn.pe_node_features != "random":
            raise ValueError(
                "arch='wire_llm' requires gnn.pe_node_features='random' "
                f"(word-embedding feature prep is not wired). Got {config.gnn.pe_node_features!r}.")
    if config.gnn.get("pe_pool", "pe") != "pe" and config.gnn.arch not in _PSI_PRODUCER_ARCHS:
        raise ValueError(
            f"gnn.pe_pool={config.gnn.pe_pool!r} is read ONLY by gt.build_psi_producer, "
            f"i.e. by {_PSI_PRODUCER_ARCHS}; arch={config.gnn.arch!r} builds its GT "
            "elsewhere and would silently ignore it (a run that looks configured for "
            "Ψ = E_q[Φ(Φ(q; S, H), G, T)] but trains Ψ = Φ(E_q[Φ(q; S, H)], G, T)).")
    if config.gnn.pe_gt_from or config.gnn.semantic_gt_from:
        if config.gnn.arch not in _PSI_PRODUCER_ARCHS:
            raise ValueError(
                "gnn.pe_gt_from / semantic_gt_from are only supported for the Ψ-consuming "
                f"architectures {_PSI_PRODUCER_ARCHS}, got arch={config.gnn.arch!r}.")
        if config.gnn.semantic_gt_from and not config.gnn.pe_gt_from:
            raise ValueError(
                "gnn.semantic_gt_from requires gnn.pe_gt_from (the LEGACY two-stage Ψ "
                "producer gt.TwoStagePE needs both). For a current run set pe_gt_from "
                "alone: Ψ is the navigator's PE stage (gt.NavigatorPE, Ψ = PE_GT(graph)); "
                "the Semantic GT is the AGT head and lives in gt.NavigatorGT.")
    if (config.data.injection_scope == "decode_consistent"
            and config.gnn.arch not in ("graph_mask_llm", "learnable_graph_mask")):
        raise ValueError(
            "injection_scope='decode_consistent' is only wired for the mask archs "
            "(graph_mask_llm / learnable_graph_mask) — the additive family needs the "
            f"q/kv split (design note §2.3, not built). Got {config.gnn.arch!r}.")
    if config.data.injection_scope == "decode_consistent" and config.gnn.get("mask_composite"):
        raise ValueError(
            "injection_scope='decode_consistent' is not wired for gnn.mask_composite: "
            "MaskDecodeInjector arms a per-NODE-PAIR bias row, and this arm's bias is "
            "indexed by sequence POSITION in a composite graph that is fixed at prefill "
            "(extending the cycle mid-rollout would change every token's C_tok row). Use "
            "data.injection_scope='prompt_only'.")
    if (config.data.injection_scope == "decode_consistent"
            and config.gnn.mask_layer_scope == "all"):
        raise ValueError(
            "injection_scope='decode_consistent' requires a dense-family mask_layer_scope: "
            "at decode, sliding-window layers crop their KV cache so the per-step bias "
            "row cannot be applied there — training them with the bias would recreate "
            "the train/decode asymmetry this mode exists to remove.")
    if config.trainer.loss_target not in ("all", "responses", "edge_list"):
        raise ValueError(
            f"loss_target must be 'all', 'responses', or 'edge_list', got {config.trainer.loss_target!r}"
        )
    if config.trainer.loss_target == "edge_list" and config.data.text_edge_list != "present":
        raise ValueError(
            "loss_target='edge_list' requires text_edge_list='present'."
        )
    if config.data.spine_tools not in ("present", "none"):
        raise ValueError(
            f"data.spine_tools must be 'present' or 'none', got {config.data.spine_tools!r}")
    # Train/eval prompt policy: always reported (measured, not assumed) because the two
    # sides are deliberately allowed to differ. The standard configuration TRAINS tool-free
    # and DEPLOYS with the SPINE API live; the seam handles it (bare route -> [answer(...)]).
    eval_tools = not evaluate._spine_tools_disabled()
    print(f"[prompt-policy] train: spine_tools={config.data.spine_tools} "
          f"icl_examples={config.data.icl_examples}  |  eval: tools="
          f"{'on' if eval_tools else 'off'} use_icl={config.eval.use_icl}")
    # Only the genuinely lossy directions warn. Trained WITH tool targets but evaluated
    # with tools off: the model emits action lists into a prompt that forbids them and a
    # simulator that no-ops them. ICL either way: the few-shot layout also relocates the
    # scene graph (system message vs query turn), so the eval prompt has a shape the model
    # never saw during training.
    if config.data.spine_tools == "present" and not eval_tools:
        print("WARNING: trained with SPINE tool targets but PRISM_DISABLE_SPINE_TOOLS is "
              "set — eval forbids and no-ops the actions the targets teach.")
    if (config.data.icl_examples > 0) != bool(config.eval.use_icl):
        print(f"WARNING: ICL mismatch — training icl_examples={config.data.icl_examples} "
              f"but eval.use_icl={config.eval.use_icl}; the few-shot layout moves the scene "
              f"graph between the system message and the query turn.")


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
