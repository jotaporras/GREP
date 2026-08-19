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
from prism.models.gnn_llm import (
    BatchedMaskDecodeInjector,
    _MaskDecodeRowState,
    build_injection_map,
    core_graph_model,
    decode_style_query_map,
    find_last_graph_scope,
    mask_node_values,
    node_token_variants,
    pointer_candidate_pairs,
    tok2node_vector,
)
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
        if args.gradient_checkpointing:
            # Reentrant recompute drops gradients to the armed Ψ tensor
            # (captured state, not a checkpoint input) — the tower would never
            # learn. Default to non-reentrant; refuse an explicit True.
            gc_kwargs = dict(args.gradient_checkpointing_kwargs or {})
            if gc_kwargs.get("use_reentrant") is True:
                raise ValueError(
                    "use_reentrant=True gradient checkpointing silently drops "
                    "the Ψ tower's gradients; use use_reentrant=False.")
            gc_kwargs["use_reentrant"] = False
            args.gradient_checkpointing_kwargs = gc_kwargs

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
        # Grads are live here (backward ran in super, optimizer.step has not) —
        # capture the tower's grad norm so wandb SHOWS the PE weights learning.
        sq = 0.0
        for p in self._core.structural_parameters():
            if p.grad is not None:
                sq += float(p.grad.detach().float().pow(2).sum())
        self._tower_grad_norm = sq ** 0.5
        tr_loss = getattr(self, "_tr_loss", None)
        if tr_loss is not None and loss.device != tr_loss.device:
            loss = loss.to(tr_loss.device)
        return loss

    def log(self, logs: dict, *args, **kwargs):
        if getattr(self, "_tower_grad_norm", None) is not None:
            logs["tower_grad_norm"] = self._tower_grad_norm
        logs["pe_gain_tanh"] = float(
            torch.tanh(self._core.pe_gain.detach()).float().mean())
        super().log(logs, *args, **kwargs)

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
        from peft import PeftModel

        output_dir = output_dir or self.args.output_dir
        save_run_dir(self.model, self.gnn_config, output_dir)
        if isinstance(self.model, PeftModel):
            # trl's peft_config path: the WHOLE policy is PEFT-wrapped and the
            # HF Trainer machinery saves the adapter.
            super().save_model(output_dir, _internal_call)
        else:
            # Warm-start path: the LoRA layers live INSIDE the inner .llm (nf4
            # keeps them unmerged; the llm is not a PeftModel instance) — the
            # loader stashed the PeftModel handle for exactly this re-save.
            # Trainer.save_model would call save_pretrained on the WRAPPER,
            # which has no adapter of its own.
            handle = getattr(core_graph_model(self.model).llm,
                             "_prism_peft_handle", None)
            if handle is None:
                raise RuntimeError(
                    "no adapter to save: policy is neither PEFT-wrapped nor a "
                    "warm-started checkpoint with an attached LoRA.")
            handle.save_pretrained(output_dir)


class MaskGRPOTrainer(GRPOTrainer):
    """GRPO trainer for the MASK archs (``learnable_graph_mask``): HF rollouts.

    The mask family's graph channel is an attention-score bias with a
    decode-time per-step extension — nothing vLLM's paged attention can carry —
    so rollouts run through the POLICY ITSELF via ``model.generate``, batched
    across the whole generation group by :class:`BatchedMaskDecodeInjector`
    (the batch-1 eval injector's semantics, one row state per sequence). That
    kills two problems at once: no engine and no weight sync exist (rollouts
    are exactly on-policy by construction), and no sequential per-completion
    generation.

    Loss side reimplements trl's small logps loop verbatim (its fixed signature
    cannot carry graph kwargs) and routes each chunk through
    ``LearnableGraphMaskLLM.forward`` with the SAME ``decode_consistent``
    query/key maps the SFT collator builds — so RL trains under semantics
    byte-identical to the SFT checkpoint it warm-starts from, and the maps are
    exactly what the rollout injector exposed step-by-step (parity-tested in
    e13c). Gradients flow loss → bias → Ψ → GT (``structural_parameters``).

    Fail-loud restrictions: ``injection_scope`` must be ``decode_consistent``;
    identity-RoPE checkpoints unsupported (per-row position rewrite);
    ``beta=0``; ``use_vllm=False``.
    """

    def __init__(self, model, *, gnn_config: dict, args,
                 rollout_batch_size: int = 16, **kwargs):
        core = core_graph_model(model)
        if gnn_config.get("injection_scope") != "decode_consistent":
            raise ValueError(
                "mask-arch RL requires injection_scope='decode_consistent' — "
                "the rollout injector and the loss-side maps implement exactly "
                f"that contract. Got {gnn_config.get('injection_scope')!r}.")
        if core._disable_graph_token_rope:
            raise ValueError(
                "disable_graph_token_rope checkpoints are unsupported in "
                "mask RL: the batched injector cannot rewrite position_ids "
                "per row.")
        if args.beta != 0.0:
            raise ValueError(
                "beta must be 0: the trl ref model would compute mask-free "
                "log-probs (plain copy without the graph channel).")
        if args.use_vllm:
            raise ValueError(
                "use_vllm must be False: mask archs have no vLLM analog — "
                "rollouts run through the policy's own generate().")
        if args.gradient_checkpointing:
            gc_kwargs = dict(args.gradient_checkpointing_kwargs or {})
            if gc_kwargs.get("use_reentrant") is True:
                raise ValueError(
                    "use_reentrant=True gradient checkpointing silently drops "
                    "the armed structural bias' gradients; use "
                    "use_reentrant=False.")
            gc_kwargs["use_reentrant"] = False
            args.gradient_checkpointing_kwargs = gc_kwargs

        super().__init__(model, args=args, **kwargs)
        # PEFT froze every non-LoRA parameter; the GT training on the RL
        # reward is the point — re-enable it (and the base-LR graph-side
        # params, e.g. the post-fusion modules).
        for p in core.structural_parameters():
            p.requires_grad = True
        base_lr_fn = getattr(core, "base_lr_parameters", None)
        for p in (base_lr_fn() if callable(base_lr_fn) else []):
            p.requires_grad = True
        self._core = core
        self._ensure_fp32_tower()
        # Gemma-it ends assistant turns with its turn-end token (gemma-4:
        # "<turn|>" = 106; gemma-3 called it "<end_of_turn>"), NOT
        # tokenizer.eos ("<eos>"). trl keys termination stats, the completion
        # mask, and our rollout truncation on self.eos_token_id, so left at
        # <eos> every completion counts as unterminated full-length. Token
        # NAMES drift across Gemma generations — derive the id from what the
        # chat template actually emits after the message content, restricted
        # to the model's declared stop ids.
        gen_cfg = getattr(core.llm, "generation_config", None)
        stops = getattr(gen_cfg, "eos_token_id", None)
        stops = set(stops if isinstance(stops, (list, tuple))
                    else [] if stops is None else [stops])
        stops.add(self.processing_class.eos_token_id)
        sentinel = ""
        tail = self.processing_class.apply_chat_template(
            [{"role": "user", "content": sentinel}],
            tokenize=False).rsplit(sentinel, 1)[1]
        tail_ids = self.processing_class(
            tail, add_special_tokens=False)["input_ids"]
        turn_end = next((i for i in tail_ids if i in stops), None)
        if turn_end is None:
            raise ValueError(
                f"could not find a turn-end token: none of the chat "
                f"template's post-content ids {tail_ids} is in the model's "
                f"stop set {sorted(stops)}.")
        self.eos_token_id = turn_end
        self.gnn_config = gnn_config
        self.rollout_batch_size = int(rollout_batch_size)
        self._edge_weights = gnn_config.get("edge_weights", "binary")
        self._scene_by_prompt = {
            row["prompt"]: row["scene_graph_dict"]
            for row in self.train_dataset
        }
        # prompt → (prompt_ids, pyg_graph, prompt_injection_map,
        #           node_token_seqs, node_values). node_values snapshot the
        #           LIVE tower, so entries are valid for one optimizer step.
        self._prompt_cache: dict[str, tuple] = {}
        self._cache_step = -1
        # full-row tokens → (query_map, key_map) in UNPADDED coordinates
        # (loss-side maps; token content fully determines them). Cleared with
        # the prompt cache — same lifetime, keeps memory bounded.
        self._row_map_cache: dict[tuple, tuple] = {}
        # full-row tokens → pointer candidate triples (e17-E; token-determined).
        self._ptr_pair_cache: dict[tuple, list] = {}

    def create_optimizer(self):
        opt = create_two_group_optimizer(self, self._core.structural_parameters())
        return opt if opt is not None else super().create_optimizer()

    # ------------------------------------------------------------------ cache

    def _entry_for_prompt(self, prompt: str):
        hit = self._prompt_cache.get(prompt)
        if hit is not None:
            return hit
        prompt_ids = self.processing_class(
            prompt, add_special_tokens=False)["input_ids"]
        scene = self._scene_by_prompt.get(prompt)
        if scene is None:
            raise ValueError(
                "rollout prompt not found in the RL dataset — the mask is "
                "built from the dataset's scene_graph_dict, so prompts must "
                "reach generation verbatim. A miss is a data/collation bug.")
        pyg_graph = data_utils.scene_graph_dict_to_pyg(
            scene, edge_weights=self._edge_weights)
        node_token_seqs = node_token_variants(
            pyg_graph.node_names, self.processing_class)
        scope_start = find_last_graph_scope(prompt_ids, self.processing_class)
        injection_map = build_injection_map(
            prompt_ids, node_token_seqs, scope_start=scope_start)
        device = next(self._core.parameters()).device
        node_values = mask_node_values(self._core, pyg_graph, device)
        # Post-fusion / pointer-fusion: snapshot raw Ψ for decode-step arming
        # (rollouts run no-grad; the loss-side forward recomputes Ψ WITH grad).
        psi = None
        if (getattr(self._core, "_post_fusion", False)
                or getattr(self._core, "_pointer_fusion", False)):
            with torch.no_grad():
                psi = self._core.pe_model(pyg_graph).float()
        entry = (prompt_ids, pyg_graph, injection_map, node_token_seqs,
                 node_values, psi)
        self._prompt_cache[prompt] = entry
        return entry

    @staticmethod
    def _offset_map(m: dict, off: int) -> dict:
        return {nid: [(s + off, e + off) for s, e in spans]
                for nid, spans in m.items()}

    def _ensure_fp32_tower(self):
        """The GT runs fp32 (the ``build_structural_mask`` contract); graph
        features are fp32 tensors. The HF stack casts leftover fp32 modules of
        a quantized policy to bf16 (GRPOConfig defaults ``bf16 = not fp16``),
        which crashes ``mask_node_values`` with a float/bf16 matmul mismatch.
        ``Module.float()`` casts ``param.data`` in place, so the optimizer's
        parameter references survive — safe to re-assert any time."""
        pe = getattr(self._core, "pe_model", None)
        if pe is not None and any(p.dtype != torch.float32
                                  for p in pe.parameters()):
            pe.float()
        # Same contract for every enabled fusion pathway (pf/glora/ptr/xf):
        # fp32 modules, hooks cast to the hidden-state dtype at read time.
        fp32_fn = getattr(self._core, "ensure_fp32_fusion", None)
        if callable(fp32_fn):
            fp32_fn()

    # --------------------------------------------------------------- rollouts

    def _generate_single_turn(self, prompts: list):
        self._ensure_fp32_tower()
        if self.state.global_step != self._cache_step:
            # New optimizer step ⇒ the tower moved ⇒ cached node_values are
            # stale. Rebuild so rollouts sample under the current tower.
            self._prompt_cache.clear()
            self._row_map_cache.clear()
            self._ptr_pair_cache.clear()
            self._cache_step = self.state.global_step

        core = self._core
        device = next(core.parameters()).device
        # generate() must run CACHED: with gradient checkpointing enabled on a
        # train-mode model, transformers drops past_key_values, every decode
        # step becomes a multi-token forward, and the decode injector (which
        # only fires on q_len==1) never engages — rollouts would sample with
        # NO mask. Disable GC + dropout for the sampling pass, restore after.
        was_training = core.llm.training
        was_gc = getattr(core.llm, "is_gradient_checkpointing", False)
        if was_gc:
            core.llm.gradient_checkpointing_disable()
        core.llm.eval()
        try:
            all_prompt_ids, all_completion_ids = self._rollout_chunks(
                prompts, core, device)
        finally:
            if was_training:
                core.llm.train()
            if was_gc:
                core.llm.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
        # logprobs=None: trl's non-vLLM contract — old/ref logps, when needed,
        # are recomputed through _get_per_token_logps_and_entropies, i.e.
        # under the SAME mask semantics the rollout sampled with.
        return all_prompt_ids, all_completion_ids, None, {}

    def _rollout_chunks(self, prompts, core, device):
        all_prompt_ids, all_completion_ids = [], []
        for start in range(0, len(prompts), self.rollout_batch_size):
            chunk = prompts[start:start + self.rollout_batch_size]
            entries = [self._entry_for_prompt(p) for p in chunk]
            max_len = max(len(e[0]) for e in entries)
            batch_ids = torch.full((len(chunk), max_len), self.pad_token_id,
                                   dtype=torch.long, device=device)
            attn = torch.zeros((len(chunk), max_len), dtype=torch.long,
                               device=device)
            padded_maps, row_states, graphs, psi_by_row = [], [], [], []
            for b, (pids, g, imap, seqs, node_values, psi) in enumerate(entries):
                off = max_len - len(pids)
                batch_ids[b, off:] = torch.tensor(pids, device=device)
                attn[b, off:] = 1
                pmap = self._offset_map(imap, off)
                padded_maps.append(pmap)
                graphs.append(g)
                psi_by_row.append(psi)
                row_states.append(_MaskDecodeRowState(
                    node_values, tok2node_vector(pmap, max_len, device), seqs))
            post_fusion = getattr(core, "_post_fusion", False)
            pointer_fusion = getattr(core, "_pointer_fusion", False)
            injector = BatchedMaskDecodeInjector(
                core, row_states, max_len,
                psi_by_row=(psi_by_row if post_fusion or pointer_fusion
                            else None))
            handle = core.llm.register_forward_pre_hook(
                injector.pre_hook, with_kwargs=True)
            try:
                with torch.no_grad():
                    core._struct_bias = core.build_structural_mask(
                        max_len, graphs, padded_maps, device)
                    if post_fusion:
                        # generate() bypasses the wrapper forward, so the
                        # prefill residual signal is armed manually here (the
                        # decode steps are armed by the injector).
                        core._pf_signal = core.build_pf_signal(
                            max_len, graphs, padded_maps, device)
                    if getattr(core, "_graph_lora", False):
                        # Per-graph, static across decode — armed once.
                        core._glora_A = core.build_glora_signal(graphs, device)
                    if getattr(core, "_cross_fusion", False):
                        core._xf_kv = core.build_xf_kv(graphs, device)
                    out = core.llm.generate(
                        input_ids=batch_ids, attention_mask=attn,
                        max_new_tokens=self.max_completion_length,
                        do_sample=True, temperature=self.temperature,
                        top_p=self.top_p, top_k=self.top_k,
                        min_p=self.min_p,
                        repetition_penalty=self.repetition_penalty,
                        pad_token_id=self.pad_token_id,
                        use_cache=True,
                    )
            finally:
                core._struct_bias = None
                handle.remove()
                core._decode_bias_row = None
                core._pf_signal = None
                core._pf_decode_vec = None
                core._glora_A = None
                core._xf_kv = None
                core._ptr_state = None
                core._ptr_decode_cand = None
            completions = out[:, max_len:]
            if completions.shape[1] > 1 and injector.decode_steps == 0:
                raise RuntimeError(
                    "decode injector never fired during generate() — "
                    "generation ran uncached (multi-token forwards only), so "
                    "completions were sampled WITHOUT the structural mask.")
            for b, (pids, *_rest) in enumerate(entries):
                ids = completions[b].tolist()
                if self.eos_token_id in ids:
                    ids = ids[:ids.index(self.eos_token_id) + 1]
                # generate() right-pads finished rows; strip trailing pads for
                # rows that ended without EOS too.
                while ids and ids[-1] == self.pad_token_id and \
                        ids[-1] != self.eos_token_id:
                    ids.pop()
                all_prompt_ids.append(list(pids))
                all_completion_ids.append(ids)
        return all_prompt_ids, all_completion_ids

    # -------------------------------------------------------------- loss side

    def _maps_for_row(self, toks: list, prompt_len: int, entry):
        key = tuple(toks)
        hit = self._row_map_cache.get(key)
        if hit is not None:
            return hit
        _pids, _g, _imap, node_token_seqs, _nv, _psi = entry
        scope_start = find_last_graph_scope(toks, self.processing_class)
        full_map = build_injection_map(toks, node_token_seqs,
                                       scope_start=scope_start)
        query_map = decode_style_query_map(full_map, prompt_len, toks,
                                           node_token_seqs)
        self._row_map_cache[key] = (query_map, full_map)
        return query_map, full_map

    def _ptr_pairs_for_row(self, toks: list, prompt_len: int, entry) -> list:
        """Teacher-forced pointer candidates (e17-E) for one unpadded row."""
        key = tuple(toks)
        hit = self._ptr_pair_cache.get(key)
        if hit is None:
            node_token_seqs = entry[3]
            hit = pointer_candidate_pairs(toks, prompt_len, node_token_seqs)
            self._ptr_pair_cache[key] = hit
        return hit

    def _entries_for_rows(self, input_ids, attention_mask, logits_to_keep):
        """Per-row cache entries + full-row token lists, recovered from the
        rows themselves (prompt tokens key the cache — order/slicing
        independent, trl slices micro-batches)."""
        by_pids = {tuple(e[0]): e for e in self._prompt_cache.values()}
        prompt_cols = input_ids.shape[1] - int(logits_to_keep or 0)
        out = []
        for b in range(input_ids.shape[0]):
            pmask = attention_mask[b, :prompt_cols].bool()
            ptoks = tuple(input_ids[b, :prompt_cols][pmask].tolist())
            e = by_pids.get(ptoks)
            if e is None:
                raise ValueError(
                    "loss-side row's prompt tokens not found in the prompt "
                    "cache — the generate/loss tokenization contract changed.")
            cmask = attention_mask[b, prompt_cols:].bool()
            ctoks = input_ids[b, prompt_cols:][cmask].tolist()
            out.append((e, list(ptoks) + ctoks, len(ptoks),
                        prompt_cols - len(ptoks)))
        return out

    def _get_per_token_logps_and_entropies(self, model, input_ids,
                                           attention_mask, logits_to_keep,
                                           batch_size=None,
                                           compute_entropy=False, **kwargs):
        from trl.trainer.utils import entropy_from_logits, selective_log_softmax

        if not self._prompt_cache:
            return super()._get_per_token_logps_and_entropies(
                model, input_ids, attention_mask, logits_to_keep,
                batch_size=batch_size, compute_entropy=compute_entropy,
                **kwargs)
        rows = self._entries_for_rows(input_ids, attention_mask, logits_to_keep)
        B = input_ids.shape[0]
        chunk = batch_size or B
        if chunk < B and getattr(self._core.llm, "is_gradient_checkpointing",
                                 False):
            raise ValueError(
                "chunked logps (batch_size < batch) with gradient "
                "checkpointing would recompute earlier chunks' attention "
                "against the LAST chunk's armed structural bias during "
                "backward — silently wrong gradients. Disable chunking or "
                "gradient checkpointing.")
        all_logps, all_ents = [], []
        pointer_fusion = getattr(self._core, "_pointer_fusion", False)
        for start in range(0, B, chunk):
            sl = slice(start, start + chunk)
            q_maps, k_maps, graphs, ptr_cands = [], [], [], []
            for entry, toks, prompt_len, off in rows[sl]:
                qm, km = self._maps_for_row(toks, prompt_len, entry)
                q_maps.append(self._offset_map(qm, off))
                k_maps.append(self._offset_map(km, off))
                graphs.append(entry[1])
                if pointer_fusion:
                    ptr_cands.append([
                        (s + off, n, t) for s, n, t in
                        self._ptr_pairs_for_row(toks, prompt_len, entry)])
            # Mirrors trl 0.27's inner loop exactly (slice, temperature,
            # selective_log_softmax) — reimplemented because its fixed
            # signature cannot carry the graph kwargs the mask forward needs.
            logits = model(
                input_ids=input_ids[sl], attention_mask=attention_mask[sl],
                graphs=graphs, injection_maps=q_maps,
                key_injection_maps=k_maps,
                **({"pointer_candidates": ptr_cands} if pointer_fusion else {}),
                logits_to_keep=logits_to_keep + 1, use_cache=False,
            ).logits
            logits = logits[:, :-1, :][:, -logits_to_keep:, :]
            logits = logits / self.temperature
            completion_ids = input_ids[sl][:, -logits_to_keep:]
            all_logps.append(selective_log_softmax(logits, completion_ids))
            if compute_entropy:
                with torch.no_grad():
                    all_ents.append(entropy_from_logits(logits))
        logps = torch.cat(all_logps, dim=0)
        ents = torch.cat(all_ents, dim=0) if compute_entropy else None
        return logps, ents

    # ------------------------------------------------------------- telemetry

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        sq = 0.0
        for p in self._core.structural_parameters():
            if p.grad is not None:
                sq += float(p.grad.detach().float().pow(2).sum())
        self._tower_grad_norm = sq ** 0.5
        tr_loss = getattr(self, "_tr_loss", None)
        if tr_loss is not None and loss.device != tr_loss.device:
            loss = loss.to(tr_loss.device)
        return loss

    def log(self, logs: dict, *args, **kwargs):
        if getattr(self, "_tower_grad_norm", None) is not None:
            logs["tower_grad_norm"] = self._tower_grad_norm
        super().log(logs, *args, **kwargs)

    # ------------------------------------------------------------------ save

    def save_model(self, output_dir=None, _internal_call=False):
        from peft import PeftModel

        output_dir = output_dir or self.args.output_dir
        save_run_dir(self.model, self.gnn_config, output_dir)
        if isinstance(self.model, PeftModel):
            # trl's peft_config path: the WHOLE policy is PEFT-wrapped and the
            # HF Trainer machinery saves the adapter.
            super().save_model(output_dir, _internal_call)
        else:
            # Warm-start path: the LoRA layers live INSIDE the inner .llm (nf4
            # keeps them unmerged; the llm is not a PeftModel instance) — the
            # loader stashed the PeftModel handle for exactly this re-save.
            # Trainer.save_model would call save_pretrained on the WRAPPER,
            # which has no adapter of its own.
            handle = getattr(core_graph_model(self.model).llm,
                             "_prism_peft_handle", None)
            if handle is None:
                raise RuntimeError(
                    "no adapter to save: policy is neither PEFT-wrapped nor a "
                    "warm-started checkpoint with an attached LoRA.")
            handle.save_pretrained(output_dir)
