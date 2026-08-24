"""SPINE-compatible LLM clients backed by the vLLM engine.

Both clients subclass the HF inference clients and override ONLY
``_generate_tokens`` — every other eval semantic (compact-prompt translation,
ICL policy, token budgets, decode, SPINE-JSON inverse translation) is inherited
verbatim, so backend choice cannot drift the prompts or the scoring.

Generation is greedy (``temperature=0``), matching ``inference.DECODE_KWARGS``.
"""
from __future__ import annotations

import torch
from torch import nn

from prism.models import inference
from prism.models.vllm_graph.psi import build_psi_transport


class _CpuAnchor(nn.Module):
    """Parameter-bearing stand-in for the HF model: the base client only reads
    ``next(model.parameters()).device`` to place tokenized inputs."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1), requires_grad=False)


def _greedy_params(max_new_tokens: int):
    from vllm import SamplingParams
    return SamplingParams(temperature=0, max_tokens=max_new_tokens)


class VLLMInMemoryLLM(inference.InMemoryLLM):
    """Plain-LLM SPINE client over a stock vLLM engine."""

    def __init__(self, engine, tokenizer, include_edges: bool, include_tools: bool,
                 icl_examples: int, response_format: str = "think_route"):
        super().__init__(_CpuAnchor(), tokenizer, include_edges=include_edges,
                         include_tools=include_tools, icl_examples=icl_examples,
                         response_format=response_format)
        self.engine = engine

    def _generate_tokens(self, input_ids, attention_mask, msg, max_new_tokens):
        out = self.engine.generate(
            [{"prompt_token_ids": input_ids[0].tolist()}],
            _greedy_params(max_new_tokens),
        )[0].outputs[0]
        return torch.tensor([list(out.token_ids)])


class VLLMGraphInMemoryLLM(inference.GraphAugmentedInMemoryLLM):
    """Graph-conditioned SPINE client: Ψ built driver-side by the checkpoint's
    Ψ producer, shipped per request through the engine's multimodal channel.

    ``psi_producer`` is the ``GraphAugmentedLLM`` from
    ``psi_producer.load_psi_producer`` (or a fully-loaded HF graph model —
    anything whose ``build_pe_signal`` chain carries the trained weights).
    Additive archs only; the engine build enforces that upstream.
    """

    def __init__(self, psi_producer, tokenizer, engine, include_edges: bool,
                 include_tools: bool, icl_examples: int, permutation=None,
                 edge_weights: str = "gaussian",
                 injection_scope: str = "full_sequence",
                 response_format: str = "think_route"):
        if injection_scope == "decode_consistent":
            raise ValueError(
                "decode_consistent is a mask-arch policy (MaskDecodeInjector); "
                "additive checkpoints served through vLLM cannot carry it.")
        super().__init__(psi_producer, tokenizer, include_edges=include_edges,
                         include_tools=include_tools, icl_examples=icl_examples,
                         permutation=permutation, edge_weights=edge_weights,
                         injection_scope=injection_scope,
                         response_format=response_format)
        self.engine = engine

    def _generate_tokens(self, input_ids, attention_mask, pyg_graphs, max_new_tokens):
        prompt_ids = input_ids[0].tolist()
        robot_loc = pyg_graphs[-1].robot_location if pyg_graphs else None
        print(f"[spine-llm] graph_found={bool(pyg_graphs)}, n_graphs={len(pyg_graphs)}, "
              f"robot_location={robot_loc}, backend=vllm")

        if not pyg_graphs:
            out = self.engine.generate(
                [{"prompt_token_ids": prompt_ids}], _greedy_params(max_new_tokens),
            )[0].outputs[0]
            return torch.tensor([list(out.token_ids)])

        producer = inference._core_graph_model(self.model)
        transport, _ = build_psi_transport(
            producer, self.tokenizer, prompt_ids, pyg_graphs[-1],
            permutation=self.permutation)
        req = {
            "prompt_token_ids": prompt_ids,
            "multi_modal_data": {"image": {"graph_embeds": transport.unsqueeze(0)}},
        }
        out = self.engine.generate([req], _greedy_params(max_new_tokens))[0].outputs[0]
        return torch.tensor([list(out.token_ids)])
