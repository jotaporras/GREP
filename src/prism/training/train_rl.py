"""e16 RL training entrypoint: GRPO over nav tasks with graph-conditioned
vLLM rollouts.

Mirrors ``train_v3``'s scaffolding (hydra config, wandb run naming, run-dir
construction) but is a separate entrypoint per the task spec — the SFT→RL
interface change (prompt-only dataset, reward functions, group sampling,
rollout engine) is too drastic to thread through ``train_model``.

Two init modes (``trainer.rl.init_checkpoint``):
- set   → resume the policy from an SFT run dir via ``checkpoint.load_checkpoint``
          (nf4 reload keeps the LoRA adapter unmerged — continued training is
          exactly what that path preserves);
- unset → from scratch: base LLM + ``architectures.build_planner_model`` +
          fresh LoRA, with the ``gnn.pe_gt_from`` navigator carry as in e14.

Run: ``python -m prism.training.train_rl --config-name=e16_rl_config``
"""
from __future__ import annotations

import os

import hydra
import omegaconf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from prism.training import _trl_compat  # noqa: F401 — must precede trl.trainer imports

from trl import GRPOConfig

from prism.data import rl_dataset
from prism.models import architectures, inference
from prism.models.vllm_graph import engine as vg_engine
from prism.training import rewards, train_v3
from prism.training.trainers_rl import GraphGRPOTrainer, MaskGRPOTrainer

MASK_ARCHS = {"graph_mask_llm", "learnable_graph_mask"}


def train_rl(config: omegaconf.DictConfig) -> None:
    rl_cfg = config.trainer.rl
    run_id = train_v3._setup_wandb(
        config.wandb.project, config.wandb.run_name, config.wandb.tag,
        report_to=config.trainer.report_to)
    output_dir = train_v3._construct_output_dir(config, run_id)
    print(f"[train_rl] output dir: {output_dir}")

    # The in-process vLLM worker pins itself to cuda:0 of the visible set, so
    # on a two-GPU host the policy lives on the OTHER device (plaza: engine on
    # 0, trainer on 1). Single-GPU hosts share device 0.
    policy_device = int(rl_cfg.get("policy_device", 0))

    init_checkpoint = rl_cfg.get("init_checkpoint")
    is_mask_arch = False
    if init_checkpoint:
        from prism.eval import checkpoint as ckpt_mod

        gnn_config = _gnn_config_from_checkpoint(init_checkpoint)
        # The checkpoint records its SFT-time structural_lr_mult (e.g. e13f:
        # 0.012); the RL tower LR is THIS run's knob, so the live config wins.
        gnn_config["structural_lr_mult"] = float(config.gnn.structural_lr_mult)
        # Free-form arch overrides on top of the checkpoint's recorded config —
        # e.g. e17: +trainer.rl.gnn_overrides.post_fusion=true enables the
        # zero-init residual injection on a warm-started mask checkpoint. The
        # merged dict is what save_run_dir records, so the new fields persist.
        gnn_overrides = _freeform(rl_cfg.get("gnn_overrides"))
        gnn_config.update(gnn_overrides)
        is_mask_arch = gnn_config.get("architecture") in MASK_ARCHS
        # Additive archs also need the engine-policy fields (and the same call
        # fails loud on unservable archs) — mask archs skip vLLM entirely.
        policy = (None if is_mask_arch
                  else vg_engine.checkpoint_engine_policy(init_checkpoint))
        model, tokenizer, is_gnn = ckpt_mod.load_checkpoint(
            init_checkpoint, four_bit=config.trainer.bit4, device=policy_device)
        if not is_gnn:
            raise ValueError(
                f"{init_checkpoint} is a plain-LLM checkpoint; the graph RL "
                "trainer needs a graph checkpoint. Plain-LLM RL uses "
                "stock trl GRPOTrainer with use_vllm=True (the control arm).")
        if gnn_config.get("post_fusion", False) and not getattr(
                model, "_post_fusion", False):
            # Warm start from a PRE-post-fusion checkpoint (e17 v1): the loader
            # rebuilt the plain mask arch from the recorded config; attach the
            # fresh zero-init residual pathway now (bitwise no-op at init).
            model.enable_post_fusion(
                layer_scope=gnn_config.get("post_fusion_layer_scope",
                                           "dense_top_half"),
                d_gt=gnn_config["d_model"])
        if hasattr(model.llm, "_prism_peft_handle"):
            # Under nf4 the checkpoint's LoRA stays ATTACHED inside the inner
            # .llm (merging would round it into the nf4 grid — see loaders,
            # which stashes the PeftModel handle). transformers'
            # validate_quantization_for_training only recognizes adapters on
            # the TOP-LEVEL model, so mark the wrapper as PEFT-loaded: it
            # genuinely carries a trainable adapter.
            model._hf_peft_config_loaded = True
        peft_config = None
    else:
        bnb = None
        if config.trainer.bit4:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        llm = AutoModelForCausalLM.from_pretrained(
            config.model.path, torch_dtype="auto", device_map={"": policy_device},
            quantization_config=bnb)
        tokenizer = AutoTokenizer.from_pretrained(config.model.path)
        train_v3._ensure_pad_tokens(tokenizer, llm)
        model, _collator = architectures.build_planner_model(
            config.gnn, llm, tokenizer,
            disable_graph_token_rope=config.model.disable_graph_token_rope,
            freeze_llm=False)
        if config.gnn.pe_gt_from:
            from prism.models import loaders
            loaders.load_navigator_pe_into(
                model, config.gnn.pe_gt_from, config.gnn.get("semantic_gt_from"))
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=config.lora.r, lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=list(config.lora.target_modules),
            exclude_modules=architectures.peft_tower_exclude(model),
            task_type="CAUSAL_LM")
        gnn_config = _assemble_gnn_config(config)
        policy = {"identity_rope": bool(config.model.disable_graph_token_rope),
                  "pe_inject_value": True}

    if is_mask_arch:
        # Mask archs have no vLLM analog — rollouts run through the policy
        # itself (MaskGRPOTrainer), so no engine, no serving dir, no sync.
        rollout_llm = rollout_wrapper = None
    else:
        # Rollout engine over the checkpoint's servable weights. GPU placement /
        # memory split is env-driven (PRISM_VLLM_GPU_UTIL; CUDA_VISIBLE_DEVICES
        # for the process) — the plaza two-GPU topology is settled at PoC time.
        serving = (vg_engine.materialize_serving_dir(init_checkpoint, is_gnn=True)
                   if init_checkpoint else config.model.path)
        # Free-form engine passthrough (mirrors trainer.rl.grpo): any vLLM engine
        # kwarg is CLI-settable — e.g. quantization=bitsandbytes +
        # load_format=bitsandbytes for in-flight nf4 on 48 GB cards.
        engine_kwargs = _freeform(rl_cfg.get("engine"))
        # The trainer syncs the policy as a LoRARequest (see GraphGRPOTrainer.
        # _sync_policy_to_engine) — the engine must accept adapters of the
        # policy's rank.
        engine_kwargs.setdefault("enable_lora", True)
        engine_kwargs.setdefault("max_lora_rank", int(config.lora.r))
        rollout_llm, rollout_wrapper = vg_engine.build_graph_llm(
            serving,
            identity_rope=policy["identity_rope"],
            pe_inject_value=policy["pe_inject_value"],
            **engine_kwargs)

    core = inference._core_graph_model(model)
    include_edges = gnn_config.get("text_edge_list") == "present"
    dataset = rl_dataset.load_rl_dataset(
        rl_cfg.data_files, tokenizer, include_edges=include_edges,
        use_icl=False, icl_examples=0)
    print(f"[train_rl] {len(dataset)} prompts from {rl_cfg.data_files}")

    reward_weights = omegaconf.OmegaConf.to_container(rl_cfg.reward_weights) \
        if rl_cfg.get("reward_weights") else None
    reward_funcs = rewards.make_reward_funcs(reward_weights)

    grpo_kwargs = dict(
        output_dir=output_dir,
        learning_rate=config.trainer.learning_rate,
        per_device_train_batch_size=config.trainer.per_device_train_batch_size,
        gradient_accumulation_steps=config.trainer.gradient_accumulation_steps,
        num_train_epochs=config.trainer.epochs,
        max_steps=config.trainer.max_steps,
        beta=0.0,
        use_vllm=False,
        remove_unused_columns=False,
        report_to=config.trainer.report_to,
        num_generations=rl_cfg.num_generations,
        max_completion_length=rl_cfg.max_completion_length,
        max_prompt_length=None,
        temperature=rl_cfg.temperature,
        logging_steps=1,
        # Reentrant checkpointing silently drops gradients to the armed Ψ
        # tensor (captured state, not a checkpoint input) — the tower would
        # never learn. The trainer fails loud on the reentrant setting.
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    # Free-form passthrough merged LAST (mirrors trainer.sft): any GRPOConfig
    # field is CLI-settable without a schema change.
    grpo_kwargs.update(_freeform(rl_cfg.get("grpo")))
    args = GRPOConfig(**grpo_kwargs)

    if is_mask_arch:
        trainer = MaskGRPOTrainer(
            model,
            gnn_config=gnn_config,
            args=args,
            rollout_batch_size=rl_cfg.get("rollout_batch_size", 16),
            train_dataset=dataset,
            reward_funcs=reward_funcs,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
    else:
        trainer = GraphGRPOTrainer(
            model,
            gnn_config=gnn_config,
            rollout_llm=rollout_llm,
            rollout_wrapper=rollout_wrapper,
            args=args,
            sync_every=rl_cfg.get("sync_every", 1),
            train_dataset=dataset,
            reward_funcs=reward_funcs,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
    # The e16 research question is BEFORE-vs-AFTER RL, all else equal: save the
    # untrained policy (same tower init, zero-init LoRA) as a sibling run dir so
    # both ends evaluate through the identical checkpoint/eval path.
    init_dir = os.path.join(output_dir, "init")
    trainer.save_model(init_dir)
    tokenizer.save_pretrained(init_dir)
    print(f"[train_rl] init checkpoint saved: {init_dir}")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[train_rl] saved: {output_dir}")


def _freeform(v) -> dict:
    """Free-form config dict → plain dict (an EMPTY DictConfig is falsy, so
    ``v or {}`` would hand ``to_container`` a plain dict and raise)."""
    if v is None:
        return {}
    return omegaconf.OmegaConf.to_container(v) if isinstance(
        v, omegaconf.DictConfig) else dict(v)


def _gnn_config_from_checkpoint(path: str) -> dict:
    """The recorded config, re-nested the way ``save_run_dir`` expects (flat)."""
    from prism.models import loaders
    return loaders.load_gnn_config(path)


def _assemble_gnn_config(config: omegaconf.DictConfig) -> dict:
    """Flat run config for ``save_run_dir`` in the from-scratch case — the same
    key layout ``loaders.load_gnn_config`` flattens a saved run dir into."""
    gnn = omegaconf.OmegaConf.to_container(config.gnn, resolve=True)
    gnn.pop("arch", None)
    return {
        "architecture": config.gnn.arch,
        "base_model": config.model.path,
        "text_edge_list": config.data.text_edge_list,
        "injection_scope": config.data.injection_scope,
        "edge_weights": config.data.edge_weights,
        "spine_tools": config.data.spine_tools,
        "icl_examples": config.data.icl_examples,
        "disable_graph_token_rope": bool(config.model.disable_graph_token_rope),
        **gnn,
    }


@hydra.main(config_path="../../../experiments", config_name="e16_rl_config",
            version_base=None)
def main(config: omegaconf.DictConfig) -> None:
    train_rl(config)


if __name__ == "__main__":
    main()
