"""vLLM multimodal plumbing that carries the Ψ transport tensor into the engine.

vLLM's data parser only speaks image/video/audio, so the ``[seq_len, hidden+1]``
transport tensor rides under the "image" modality label (terratorch does the
same for its non-image rasters) — cosmetic only. The processor emits the REAL
prompt token ids plus one full-prompt placeholder range, so the engine computes
token embeddings itself and the runner hands the transport rows back to the
wrapper model batch-aligned (``is_multimodal`` mask).

Lifted from ``notebooks/2026-08-07 fable-vllm-graph-demo.ipynb``. The ``apply``
override is pinned to the vLLM 0.26 base-class signature; revisit on any bump.
"""
from __future__ import annotations

import torch
from transformers import BatchFeature

from vllm.inputs import MultiModalInput, mm_input
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    PlaceholderRange,
)
from vllm.multimodal.parse import DictEmbeddingItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
)

from prism.models.vllm_graph.psi import transport_dim

_FIELDS = {"graph_embeds": "image"}


def _graph_fields_config(hf_inputs=None):
    return {name: MultiModalFieldConfig.batched(mod) for name, mod in _FIELDS.items()}


class GraphDataParser(MultiModalDataParser):
    def _parse_image_data(self, data):
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="image",
                required_fields=set(_FIELDS),
                fields_factory=_graph_fields_config,
            )
        return super()._parse_image_data(data)


class GraphProcessingInfo(BaseProcessingInfo):
    def get_data_parser(self):
        return GraphDataParser()

    def get_hf_processor(self, **kwargs):
        # The served checkpoint is text-only (no preprocessor_config.json) and
        # the Ψ transport never touches HF processing — GraphDataParser handles
        # it — so the base class's AutoProcessor lookup would fail spuriously.
        return self.get_tokenizer()

    def get_supported_mm_limits(self):
        return {"image": 1}


class GraphDummyInputsBuilder(BaseDummyInputsBuilder):
    def get_dummy_text(self, mm_counts):
        return ""

    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options=None):
        # Profile at FULL seq_len: a runtime item larger than the profiled one
        # makes the v1 scheduler retry forever (vllm#26223 family).
        hidden = self.info.get_hf_config().get_text_config().hidden_size
        return {"image": {"graph_embeds": torch.zeros(1, seq_len, transport_dim(hidden))}}


class GraphMultiModalProcessor(BaseMultiModalProcessor):
    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        return _graph_fields_config(hf_inputs)

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        return []

    def apply(self, inputs, timing_ctx) -> MultiModalInput:
        _, passthrough = self._get_hf_mm_data(inputs.mm_data_items)
        g = torch.as_tensor(passthrough["graph_embeds"])
        if g.ndim == 2:
            g = g.unsqueeze(0)
        rows = g.shape[1]

        mm_kwargs = MultiModalKwargsItems.from_hf_inputs(
            BatchFeature({"graph_embeds": g}, tensor_type="pt"),
            self._get_mm_fields_config(None, {}),
        )
        mm_hashes = inputs.get_mm_hashes(self.info.model_id)

        prompt = inputs.prompt
        if isinstance(prompt, str):
            # Profiling path (dummy text is ""): synthesize a full-length prompt.
            prompt_ids = [self.info.get_tokenizer().eos_token_id] * rows
        else:
            prompt_ids = list(prompt)
            if len(prompt_ids) != rows:
                raise ValueError(
                    f"graph mm rows ({rows}) != prompt length ({len(prompt_ids)}); "
                    "psi and prompt_token_ids must come from the same tokenization"
                )

        return mm_input(
            prompt_token_ids=prompt_ids,
            mm_kwargs=mm_kwargs,
            mm_hashes=mm_hashes,
            mm_placeholders={"image": [PlaceholderRange(offset=0, length=rows)]},
        )
