"""GRPO trainer with graph-conditioned vLLM rollouts (e16).

``GraphGRPOTrainer`` subclasses ``trl.GRPOTrainer`` (0.27) and replaces exactly
two seams, keeping trl's loss/advantage machinery intact:

1. **Rollouts** — ``_generate_single_turn`` routes through the ``vllm_graph``
   engine instead of trl's stock colocate engine (which cannot know about the
   runtime-registered model class or the Ψ multimodal transport). One request
   per incoming prompt (trl repeats each prompt ``num_generations`` times),
   Ψ built driver-side by the POLICY model's own tower and cached per prompt.
2. **Loss-side forward** — ``_get_per_token_logps_and_entropies`` arms the same
   prompt-side Ψ on the policy model before delegating, so the log-probs are
   computed under the SAME semantics the rollout sampled from (completion
   queries attend Ψ-carrying prompt keys; completion keys carry no Ψ) — the
   training-consistent forward, which the vLLM cache reproduces exactly.

Weight sync: the engine serves the FIXED base weights (bf16 merged dir, or
in-flight nf4 on 48 GB cards); after each optimizer step the current PEFT
adapter is saved and re-attached as a fresh ``LoRARequest`` (id bump — vLLM
caches adapters by id). The adapter IS the entire trainable state (Ψ tower
frozen), so the sync payload is ~the adapter file. Cadence is the first knob
in the compute-optimization plan.

v1 restrictions (fail loud): additive archs only, ``disable_graph_token_rope``
unsupported (trl's forward passes no position_ids), Ψ tower FROZEN (enables the
per-prompt Ψ cache and LoRA-only sync; unfreezing is v2), ``beta`` must be 0
(no ref model — a graph-consistent ref forward is v2).
"""
from __future__ import annotations

import os
import re
import tempfile
from ast import literal_eval

import torch

from prism.training import _trl_compat  # noqa: F401 — must precede trl.trainer imports

from trl import GRPOTrainer

from prism.data import utils as data_utils
from prism.models.gnn_llm import core_graph_model
from prism.models.vllm_graph.psi import build_psi_transport
from prism.training.run_dir import save_run_dir


class GraphGRPOTrainer(GRPOTrainer):
    def __init__(self, model, *, gnn_config: dict, rollout_llm, rollout_wrapper,
                 args, sync_every: int = 1, **kwargs):
        core = core_graph_model(model)
        if core._disable_graph_token_rope:
            raise ValueError(
                "disable_graph_token_rope checkpoints are unsupported in RL v1: "
                "trl's policy forward passes no position_ids, so identity-RoPE "
                "would silently not apply on the loss side.")
        if args.beta != 0.0:
            raise ValueError(
                "beta must be 0 in RL v1: the trl ref model would compute "
                "Ψ-free log-probs (it is a plain copy without the graph channel), "
                "making the KL term compare across different semantics.")
        if args.use_vllm:
            raise ValueError(
                "use_vllm must be False: this trainer owns its engine "
                "(trl's colocate engine cannot serve the registered graph model).")
        # v1: freeze the Ψ tower — validity of the per-prompt Ψ cache and the
        # LoRA-only weight sync both depend on it.
        for p in core.structural_parameters():
            p.requires_grad = False
        if core.pe_norm is not None:
            for p in core.pe_norm.parameters():
                p.requires_grad = False

        super().__init__(model, args=args, **kwargs)
        self.gnn_config = gnn_config
        self.rollout_llm = rollout_llm
        self.rollout_wrapper = rollout_wrapper
        self.sync_every = sync_every
        self._core = core
        self._edge_weights = gnn_config.get("edge_weights", "binary")
        self._transport_cache: dict[str, tuple] = {}
        self._batch_transports: list[torch.Tensor] | None = None
        self._last_synced_step = -1
        # LoRA hot-swap state (see _sync_policy_to_engine).
        self._lora_version = 0
        self._lora_request = None
        self._lora_sync_root = tempfile.mkdtemp(prefix="grpo_lora_sync_")

    # ------------------------------------------------------------------ Ψ

    def _transport_for_prompt(self, prompt: str):
        """(prompt_ids, transport) for one prompt text, cached — valid because
        the Ψ tower is frozen and prompts are deterministic."""
        hit = self._transport_cache.get(prompt)
        if hit is not None:
            return hit
        prompt_ids = self.processing_class(
            prompt, add_special_tokens=False)["input_ids"]
        matches = re.findall(r"[Ss]cene graph: ?(.*})", prompt, re.DOTALL)
        if not matches:
            raise ValueError(
                "RL prompt carries no parseable scene graph — Ψ cannot be built. "
                "rl_dataset embeds 'Scene graph:{...}' in every prompt; a prompt "
                "without one is a data bug, not a fallback case.")
        pyg_graph = data_utils.scene_graph_dict_to_pyg(
            literal_eval(matches[-1]), edge_weights=self._edge_weights)
        transport, _ = build_psi_transport(
            self._core, self.processing_class, prompt_ids, pyg_graph)
        self._transport_cache[prompt] = (prompt_ids, transport)
        return prompt_ids, transport

    # ------------------------------------------------------------- rollouts

    def _generate_single_turn(self, prompts: list):
        from vllm import SamplingParams

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

        reqs, transports = [], []
        for prompt in prompts:
            prompt_ids, transport = self._transport_for_prompt(prompt)
            reqs.append({
                "prompt_token_ids": list(prompt_ids),
                "multi_modal_data": {"image": {"graph_embeds": transport.unsqueeze(0)}},
            })
            transports.append(transport)
        outputs = self.rollout_llm.generate(
            reqs, sampling_params, use_tqdm=False,
            lora_request=self._lora_request)

        prompt_ids = [list(o.prompt_token_ids) for o in outputs]
        completion_ids = [list(out.token_ids) for o in outputs for out in o.outputs]
        logprobs = [
            [next(iter(lp.values())).logprob for lp in out.logprobs]
            for o in outputs for out in o.outputs
        ]
        # Consumed by the loss-side forward (row-aligned with this batch).
        self._batch_transports = transports
        return prompt_ids, completion_ids, logprobs, {}

    # ----------------------------------------------------------- loss side

    def _psi_for_rows(self, input_ids, attention_mask, transports):
        """[rows, S, hidden] Ψ aligned to left-padded prompt positions."""
        B, S = input_ids.shape
        hidden = transports[0].shape[-1] - 1
        psi = torch.zeros(B, S, hidden, device=input_ids.device)
        for b, transport in enumerate(transports):
            start = int(attention_mask[b].nonzero()[0])
            n = transport.shape[0]
            psi[b, start:start + n] = transport[:, :hidden].to(input_ids.device)
        return psi

    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask,
                                           logits_to_keep, batch_size=None,
                                           **kwargs):
        transports = self._batch_transports
        if transports is None:
            return super()._get_per_token_logps_and_entropies(
                model, input_ids, attention_mask, logits_to_keep,
                batch_size=batch_size, **kwargs)
        B = input_ids.shape[0]
        if len(transports) != B:
            raise ValueError(
                f"Ψ/batch misalignment: {len(transports)} cached transports for a "
                f"batch of {B} rows — the generate/loss batching contract changed.")
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
            self._core._pe_signal = self._psi_for_rows(
                input_ids[rows], attention_mask[rows], transports[start:start + chunk])
            logps, ents = super()._get_per_token_logps_and_entropies(
                model, input_ids[rows], attention_mask[rows], logits_to_keep,
                batch_size=None, **kwargs)
            outs_logps.append(logps)
            outs_ents.append(ents)
        logps = torch.cat(outs_logps, dim=0)
        ents = (torch.cat(outs_ents, dim=0)
                if outs_ents and outs_ents[0] is not None else None)
        return logps, ents

    # ---------------------------------------------------------- weight sync

    def _sync_policy_to_engine(self) -> None:
        """LoRA-only hot-swap: save the current adapter and bump the LoRA id.

        The engine serves the FIXED base weights (bf16 merged dir, or in-flight
        nf4 — the same base the nf4 policy trains over); the policy delta rides
        as a ``LoRARequest`` on every rollout. Merging into the engine's base
        is impossible under bnb (packed params) and wasteful otherwise — the
        adapter payload is the entire trainable state precisely because the Ψ
        tower is frozen (v1 contract, enforced in ``__init__``)."""
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
                    "policy carries no PEFT adapter to sync — LoRA-only RL is "
                    "the v1 contract (full-weight sync has no engine-side path).")
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
