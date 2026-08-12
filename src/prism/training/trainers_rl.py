"""GRPO trainer with graph-conditioned vLLM rollouts (e16).

``GraphGRPOTrainer`` subclasses ``trl.GRPOTrainer`` (0.27) and replaces exactly
two seams, keeping trl's loss/advantage machinery intact:

1. **Rollouts** — ``_generate_single_turn`` routes through the ``vllm_graph``
   engine instead of trl's stock colocate engine (which cannot know about the
   runtime-registered model class or the Ψ multimodal transport). One request
   per incoming prompt (trl repeats each prompt ``num_generations`` times),
   Ψ built driver-side by the POLICY model's own tower under ``no_grad`` and
   cached per prompt — the cache is cleared at every new optimizer step so
   rollouts always sample under the CURRENT tower.
2. **Loss-side forward** — ``_get_per_token_logps_and_entropies`` rebuilds Ψ
   per micro-batch THROUGH THE LIVE TOWER (differentiable — this is v2's
   point: the GRPO loss backpropagates into the PE weights) and arms it on the
   policy before delegating, so the log-probs are computed under the SAME
   semantics the rollout sampled from (completion queries attend Ψ-carrying
   prompt keys; completion keys carry no Ψ).

The whole graph side trains: PEFT freezes every non-LoRA parameter when it
wraps the policy, so ``__init__`` re-enables ``structural_parameters()`` /
``base_lr_parameters()`` / ``pe_norm`` exactly as ``GraphSFTTrainer`` does,
and ``create_optimizer`` reuses the shared two-group-LR builder
(``structural_lr_mult``).

Weight sync: the engine serves the FIXED base weights (bf16 merged dir, or
in-flight nf4 on 48 GB cards); after each optimizer step the current PEFT
adapter is saved and re-attached as a fresh ``LoRARequest`` (id bump — vLLM
caches adapters by id). The tower needs NO engine-side sync: the engine never
holds tower weights — Ψ ships per request, and per-step cache invalidation
means each rollout's Ψ comes from the tower as of the latest optimizer step.

Restrictions (fail loud): additive archs only, ``disable_graph_token_rope``
unsupported (trl's forward passes no position_ids), ``beta`` must be 0 (no ref
model — a graph-consistent ref forward would need its own frozen tower copy).
Gradient checkpointing requires ``use_reentrant=False`` (reentrant recompute
silently drops gradients to the armed Ψ tensor, which is captured state, not a
checkpoint input) — ``train_rl`` sets that; the chunked-logps guard below
covers the remaining hazard.
"""
from __future__ import annotations

import os
import tempfile

import torch

from prism.training import _trl_compat  # noqa: F401 — must precede trl.trainer imports

from trl import GRPOTrainer

from prism.data import utils as data_utils
from prism.models.gnn_llm import core_graph_model
from prism.models.vllm_graph.psi import build_psi_transport
from prism.training.run_dir import save_run_dir
from prism.training.trainers import create_two_group_optimizer


class GraphGRPOTrainer(GRPOTrainer):
    def __init__(self, model, *, gnn_config: dict, rollout_llm, rollout_wrapper,
                 args, sync_every: int = 1, **kwargs):
        core = core_graph_model(model)
        if core._disable_graph_token_rope:
            raise ValueError(
                "disable_graph_token_rope checkpoints are unsupported in RL: "
                "trl's policy forward passes no position_ids, so identity-RoPE "
                "would silently not apply on the loss side.")
        if args.beta != 0.0:
            raise ValueError(
                "beta must be 0: the trl ref model would compute Ψ-free "
                "log-probs (it is a plain copy without the graph channel), "
                "making the KL term compare across different semantics.")
        if args.use_vllm:
            raise ValueError(
                "use_vllm must be False: this trainer owns its engine "
                "(trl's colocate engine cannot serve the registered graph model).")
        if args.gradient_checkpointing and (
                args.gradient_checkpointing_kwargs or {}).get("use_reentrant", True):
            raise ValueError(
                "gradient checkpointing needs use_reentrant=False: the armed Ψ "
                "tensor is captured state, not a checkpoint input, and the "
                "reentrant recompute silently drops its gradients — the tower "
                "would never learn. Set gradient_checkpointing_kwargs="
                "{'use_reentrant': False}.")

        super().__init__(model, args=args, **kwargs)
        # PEFT froze every non-LoRA parameter when trl wrapped the policy;
        # re-enable the graph side exactly as GraphSFTTrainer does — training
        # the tower on the RL reward is the point of e16.
        for p in core.structural_parameters():
            p.requires_grad = True
        base_lr_fn = getattr(core, "base_lr_parameters", None)
        for p in (base_lr_fn() if callable(base_lr_fn) else []):
            p.requires_grad = True
        if core.pe_norm is not None:
            for p in core.pe_norm.parameters():
                p.requires_grad = True
        self.gnn_config = gnn_config
        self.rollout_llm = rollout_llm
        self.rollout_wrapper = rollout_wrapper
        self.sync_every = sync_every
        self._core = core
        self._edge_weights = gnn_config.get("edge_weights", "binary")
        # Ψ source per prompt: the dataset's scene_graph_dict, keyed by the
        # VERBATIM prompt text (prompts are deterministic and unique per
        # (graph, task)). Parsing the graph back out of the prompt is not a
        # fallback — edgeless templates never serialize it.
        self._scene_by_prompt = {
            row["prompt"]: row["scene_graph_dict"]
            for row in self.train_dataset
        }
        # prompt → (prompt_ids, transport, pyg_graph, injection_map). The
        # transport is a no-grad snapshot for rollouts, valid only within one
        # optimizer step (cleared in _generate_single_turn on step change);
        # the graph + injection map are step-invariant and reused by the
        # loss-side live rebuild.
        self._transport_cache: dict[str, tuple] = {}
        self._cache_step = -1
        self._last_synced_step = -1
        # LoRA hot-swap state (see _sync_policy_to_engine).
        self._lora_version = 0
        self._lora_request = None
        self._lora_sync_root = tempfile.mkdtemp(prefix="grpo_lora_sync_")

    def create_optimizer(self):
        """Two-group LR shared with the SFT trainer: tower at
        ``structural_lr_mult`` × base, LoRA at base. The policy is PEFT-wrapped,
        so structural params resolve on the unwrapped core."""
        opt = create_two_group_optimizer(self, self._core.structural_parameters())
        return opt if opt is not None else super().create_optimizer()

    # ------------------------------------------------------------------ Ψ

    def _transport_for_prompt(self, prompt: str):
        """Cache entry ``(prompt_ids, transport, pyg_graph, injection_map)``
        for one prompt text. The transport snapshot is valid because entries
        never outlive the optimizer step that built them."""
        hit = self._transport_cache.get(prompt)
        if hit is not None:
            return hit
        prompt_ids = self.processing_class(
            prompt, add_special_tokens=False)["input_ids"]
        scene = self._scene_by_prompt.get(prompt)
        if scene is None:
            raise ValueError(
                "rollout prompt not found in the RL dataset — Ψ is built from "
                "the dataset's scene_graph_dict, so prompts must reach the "
                "engine verbatim. A miss is a data/collation bug, not a "
                "fallback case.")
        pyg_graph = data_utils.scene_graph_dict_to_pyg(
            scene, edge_weights=self._edge_weights)
        transport, injection_map = build_psi_transport(
            self._core, self.processing_class, prompt_ids, pyg_graph)
        entry = (prompt_ids, transport, pyg_graph, injection_map)
        self._transport_cache[prompt] = entry
        return entry

    # ------------------------------------------------------------- rollouts

    def _generate_single_turn(self, prompts: list):
        from vllm import SamplingParams

        if self.state.global_step != self._cache_step:
            # New optimizer step ⇒ the tower moved ⇒ every cached transport is
            # stale. Rebuild below so rollouts sample under the current tower.
            self._transport_cache.clear()
            self._cache_step = self.state.global_step
        if self.state.global_step != self._last_synced_step and (
                self.state.global_step % self.sync_every == 0):
            self._sync_policy_to_engine()
            self._last_synced_step = self.state.global_step

        generation_kwargs = {
            "n": 1,
            "repetition_penalty": self.repetition_penalty,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": 0.0 if self.min_p is None else self.min_p,
            "max_tokens": self.max_completion_length,
            "logprobs": 0,
        }
        if self.args.generation_kwargs is not None:
            generation_kwargs.update(self.args.generation_kwargs)
        sampling_params = SamplingParams(**generation_kwargs)

        reqs = []
        for prompt in prompts:
            prompt_ids, transport, _, _ = self._transport_for_prompt(prompt)
            reqs.append({
                "prompt_token_ids": list(prompt_ids),
                "multi_modal_data": {"image": {"graph_embeds": transport.unsqueeze(0)}},
            })
        outputs = self.rollout_llm.generate(
            reqs, sampling_params, use_tqdm=False,
            lora_request=self._lora_request)

        prompt_ids = [list(o.prompt_token_ids) for o in outputs]
        completion_ids = [list(out.token_ids) for o in outputs for out in o.outputs]
        logprobs = [
            [next(iter(lp.values())).logprob for lp in out.logprobs]
            for o in outputs for out in o.outputs
        ]
        return prompt_ids, completion_ids, logprobs, {}

    # ----------------------------------------------------------- loss side

    def _live_psi_for_rows(self, input_ids, attention_mask, entries):
        """[rows, S, hidden] Ψ rebuilt through the LIVE tower — differentiable,
        so the GRPO loss backpropagates into the PE weights. Placed at each
        row's left-padded prompt positions, mirroring the rollout transport."""
        B, S = input_ids.shape
        emb_layer = self._core.llm.get_input_embeddings()
        psi = None
        for b, (pids, _t, pyg_graph, injection_map) in enumerate(entries):
            ids = torch.tensor([pids], device=input_ids.device)
            embeddings = emb_layer(ids)
            row = self._core.build_pe_signal(
                embeddings, [pyg_graph], [injection_map])[0]  # [n, hidden]
            if psi is None:
                psi = torch.zeros(B, S, row.shape[-1],
                                  device=input_ids.device, dtype=row.dtype)
            start = int(attention_mask[b].nonzero()[0])
            psi[b, start:start + row.shape[0]] = row
        return psi

    def _entries_for_rows(self, input_ids, attention_mask, logits_to_keep):
        """Per-row cache entries recovered from the rows THEMSELVES: the prompt
        segment (everything left of the completion window, minus left-padding)
        keys into the Ψ cache. Order/slicing independent — trl calls the logps
        function both on the full generation batch and on grad-accum
        micro-batch slices, so positional alignment is not a contract."""
        by_pids = {tuple(e[0]): e for e in self._transport_cache.values()}
        prompt_cols = input_ids.shape[1] - int(logits_to_keep or 0)
        entries = []
        for b in range(input_ids.shape[0]):
            mask = attention_mask[b, :prompt_cols].bool()
            toks = tuple(input_ids[b, :prompt_cols][mask].tolist())
            e = by_pids.get(toks)
            if e is None:
                raise ValueError(
                    "loss-side row's prompt tokens not found in the Ψ cache — "
                    "the generate/loss tokenization contract changed.")
            entries.append(e)
        return entries

    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask,
                                           logits_to_keep, batch_size=None,
                                           **kwargs):
        if not self._transport_cache:
            return super()._get_per_token_logps_and_entropies(
                model, input_ids, attention_mask, logits_to_keep,
                batch_size=batch_size, **kwargs)
        entries = self._entries_for_rows(
            input_ids, attention_mask, logits_to_keep)
        B = input_ids.shape[0]
        chunk = batch_size or B
        if chunk < B and getattr(self._core.llm, "is_gradient_checkpointing", False):
            raise ValueError(
                "chunked logps (batch_size < batch) with gradient checkpointing "
                "would recompute earlier chunks' attention against the LAST "
                "chunk's armed Ψ during backward — silently wrong gradients. "
                "Disable chunking or gradient checkpointing.")
        outs_logps, outs_ents = [], []
        for start in range(0, B, chunk):
            rows = slice(start, start + chunk)
            # Armed for this chunk's forward; the model's own lifecycle keeps it
            # armed through a gradient-checkpointed backward (see
            # GraphAugmentedLLM.forward's finally) — do NOT disarm here.
            self._core._pe_signal = self._live_psi_for_rows(
                input_ids[rows], attention_mask[rows], entries[start:start + chunk])
            logps, ents = super()._get_per_token_logps_and_entropies(
                model, input_ids[rows], attention_mask[rows], logits_to_keep,
                batch_size=None, **kwargs)
            outs_logps.append(logps)
            outs_ents.append(ents)
        logps = torch.cat(outs_logps, dim=0)
        ents = (torch.cat(outs_ents, dim=0)
                if outs_ents and outs_ents[0] is not None else None)
        return logps, ents

    def training_step(self, model, inputs, num_items_in_batch=None):
        """On split-GPU hosts the policy computes on ``policy_device`` while
        the Trainer's ``_tr_loss`` accumulator lives on accelerate's device
        (the engine's GPU); transformers 5.14 raises on the mismatch. Moving
        the detached scalar is free and side-effect-less."""
        loss = super().training_step(model, inputs, num_items_in_batch)
        tr_loss = getattr(self, "_tr_loss", None)
        if tr_loss is not None and loss.device != tr_loss.device:
            loss = loss.to(tr_loss.device)
        return loss

    # ---------------------------------------------------------- weight sync

    def _sync_policy_to_engine(self) -> None:
        """LoRA-only hot-swap: save the current adapter and bump the LoRA id.

        The engine serves the FIXED base weights (bf16 merged dir, or in-flight
        nf4 — the same base the nf4 policy trains over); the policy delta rides
        as a ``LoRARequest`` on every rollout. Merging into the engine's base
        is impossible under bnb (packed params) and wasteful otherwise. The
        tower needs no sync at all: the engine never holds tower weights — Ψ
        ships per request, rebuilt each optimizer step from the live tower."""
        import shutil

        from peft import PeftModel

        # trl's peft_config path wraps the WHOLE policy (PeftModel(graph
        # model)); the from-checkpoint path keeps the adapter on the inner
        # ``.llm``. getattr on a PeftModel delegates into the base model, so
        # resolve by isinstance, never by attribute fallback.
        if isinstance(self.model, PeftModel):
            peft_model = self.model
        else:
            inner = getattr(self.model, "llm", None)
            if not isinstance(inner, PeftModel):
                raise RuntimeError(
                    "policy carries no PEFT adapter to sync — LoRA-only sync is "
                    "the contract (full-weight sync has no engine-side path).")
            peft_model = inner
        version = self._lora_version + 1
        adapter_dir = os.path.join(self._lora_sync_root, f"v{version}")
        os.makedirs(adapter_dir, exist_ok=True)
        peft_model.save_pretrained(adapter_dir)
        # PEFT nests per-adapter subdirs only in multi-adapter setups; vLLM
        # wants the dir that directly holds adapter_config.json.
        if not os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
            sub = os.path.join(adapter_dir, "default")
            if os.path.exists(os.path.join(sub, "adapter_config.json")):
                adapter_dir = sub
            else:
                raise RuntimeError(f"no adapter_config.json under {adapter_dir}")
        from vllm.lora.request import LoRARequest
        self._lora_request = LoRARequest(f"policy-v{version}", version, adapter_dir)
        prev = os.path.join(self._lora_sync_root, f"v{self._lora_version}")
        self._lora_version = version
        if os.path.isdir(prev):
            shutil.rmtree(prev)

    # ---------------------------------------------------------------- save

    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir or self.args.output_dir
        save_run_dir(self.model, self.gnn_config, output_dir)
        if getattr(self.model, "peft_config", None) is not None:
            super().save_model(output_dir, _internal_call)
