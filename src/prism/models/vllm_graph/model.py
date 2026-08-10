"""The registered graph-conditioned Gemma-4 wrapper model.

``GraphGemma4ForCausalLM`` wraps vLLM's stock ``Gemma4ForCausalLM``: the
multimodal runner hands it the Ψ transport rows batch-aligned via
``embed_input_ids(..., multimodal_embeddings, is_multimodal)``; the wrapper
scatters them into a packed ``[num_tokens, hidden]`` Ψ (plus the span-mask
column) and arms them for the patched attention forwards
(:mod:`prism.models.vllm_graph.attention`). Token embeddings themselves are the
stock ones — Ψ only enters inside attention, post-RoPE, exactly like the HF
``GraphAugmentedLLM`` path.

State lifecycle: armed in ``embed_input_ids``, consumed during ``forward``,
cleared in its ``finally`` — mirroring ``GraphAugmentedLLM.forward``'s disarm.
"""
from __future__ import annotations

import torch
from torch import nn

from vllm import ModelRegistry
from vllm.config import VllmConfig
from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM as _StockGemma4
from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsMultiModal
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from prism.models.vllm_graph.attention import install_psi_injection
from prism.models.vllm_graph.processor import (
    GraphDummyInputsBuilder,
    GraphMultiModalProcessor,
    GraphProcessingInfo,
)

GRAPH_ARCH_NAME = "GraphGemma4ForCausalLM"


@MULTIMODAL_REGISTRY.register_processor(
    GraphMultiModalProcessor,
    info=GraphProcessingInfo,
    dummy_inputs=GraphDummyInputsBuilder,
)
class GraphGemma4ForCausalLM(nn.Module, SupportsMultiModal, SupportsLoRA):
    # Composes the stock class's HF-repo mapping (google/gemma-4-* repos nest
    # text weights under ``model.language_model.``; the stock mapper folds that
    # to ``model.``) with this wrapper's ``language_model.`` nesting. Rules
    # apply SEQUENTIALLY to the mutated key, so no ``""`` catch-all: after rule
    # 1 rewrites, the key no longer startswith ``model.`` and rule 2 is inert.
    # The bitsandbytes loader keys its quantization targets off THIS mapper
    # (loader-side bookkeeping), so it must produce final names on its own —
    # merely composing with the inner class's load-time mapping is not enough.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr=dict(_StockGemma4.hf_to_vllm_mapper.orig_to_new_substr),
        orig_to_new_prefix={
            "model.language_model.": "language_model.model.",  # HF hub layout
            "model.": "language_model.model.",  # merged/fixture layout
            "lm_head.": "language_model.lm_head.",
        },
    )
    # The bitsandbytes loader (and LoRA) resolve fused-projection layout from
    # the top-level model class — mirror the wrapped stock class exactly.
    packed_modules_mapping = _StockGemma4.packed_modules_mapping

    @classmethod
    def get_placeholder_str(cls, modality, i):
        return None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Gemma4ForCausalLM"],
        )
        # Ψ / span mask for the CURRENT packed token batch; armed in
        # embed_input_ids, read by the patched attention forwards, cleared
        # after forward.
        self._psi_packed: torch.Tensor | None = None
        self._span_mask_packed: torch.Tensor | None = None
        # Engine-level policy, set by engine.build_graph_llm from the
        # checkpoint's train_config.json.
        self._identity_rope: bool = False
        self._pe_inject_value: bool = True
        self.dbg = {"embed_calls": 0, "psi_armed": 0, "attn_hit": 0, "attn_skip_shape": 0}
        install_psi_injection(self)

    def embed_multimodal(self, **kwargs):
        g = kwargs.get("graph_embeds")
        if g is None:
            return []
        if isinstance(g, torch.Tensor):
            return tuple(g[i] for i in range(g.shape[0]))
        return tuple(torch.as_tensor(t) for t in g)

    def embed_input_ids(self, input_ids, multimodal_embeddings=None, *, is_multimodal=None):
        embeds = self.language_model.embed_input_ids(input_ids)
        self._psi_packed = None
        self._span_mask_packed = None
        self.dbg["embed_calls"] += 1
        if multimodal_embeddings is not None and len(multimodal_embeddings) > 0:
            self.dbg["psi_armed"] += 1
            rows = torch.cat(list(multimodal_embeddings), dim=0)
            n_flagged = int(is_multimodal.sum().item())
            if n_flagged != rows.shape[0]:
                raise ValueError(
                    f"psi rows ({rows.shape[0]}) != flagged positions ({n_flagged})")
            hidden = embeds.shape[-1]
            if rows.shape[-1] != hidden + 1:
                raise ValueError(
                    f"psi transport width {rows.shape[-1]} != hidden+1 ({hidden + 1}); "
                    "last column must be the injected-span mask (see vllm_graph.psi)")
            rows = rows.to(dtype=embeds.dtype, device=embeds.device)
            psi = torch.zeros_like(embeds)
            psi[is_multimodal] = rows[:, :hidden]
            span = torch.zeros(embeds.shape[0], dtype=torch.bool, device=embeds.device)
            span[is_multimodal] = rows[:, hidden] > 0.5
            self._psi_packed = psi
            self._span_mask_packed = span
        return embeds

    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None, **kwargs):
        try:
            return self.language_model(
                input_ids, positions, intermediate_tensors, inputs_embeds)
        finally:
            self._psi_packed = None
            self._span_mask_packed = None

    def compute_logits(self, *args, **kwargs):
        return self.language_model.compute_logits(*args, **kwargs)

    def load_weights(self, weights):
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_language_model(self):
        return self.language_model


def register_graph_gemma4() -> None:
    """Register the wrapper under ``GRAPH_ARCH_NAME`` (idempotent).

    Requires an in-process engine (``VLLM_ENABLE_V1_MULTIPROCESSING=0``) so the
    runtime registration is visible to the model runner; a packaged vLLM plugin
    entry point replaces this for multiprocess/server deployments.
    """
    ModelRegistry.register_model(GRAPH_ARCH_NAME, GraphGemma4ForCausalLM)
