import re
from ast import literal_eval
from typing import List, Dict

import torch

from spine import models as spine_models

from prism.data import compact_prompt
from prism.data import utils
from prism.models.gnn_llm import (
    GraphAugmentedLLM,
    GraphMaskLLM,
    build_injection_map,
    find_last_graph_scope,
    node_token_variants,
)
from prism.models.composite_graph_llm import (
    CompositeGraphLLM,
    InjectedCompositeGraphLLM,
)


# Decode is deterministic GREEDY for confirmatory eval (reproducible, no seed needed).
# NOTE: the prior `temperature=0.01, min_p=0.1` were dead config — without `do_sample=True`
# transformers ignores them and runs greedy, so this makes the actual behavior explicit
# (results unchanged). Sampling params live here, in one place, if ever re-enabled.
DECODE_KWARGS = {"do_sample": False, "use_cache": True}


def _core_graph_model(model):
    """Peel PEFT wrappers to reach the CompositeGraphLLM / GraphAugmentedLLM / GraphMaskLLM core.

    PEFT-wrapped models fail isinstance checks; unwrapping ensures the correct injection branch runs.
    LoRA adapters remain live inside the graph model's .llm (PEFT patches it in place).
    """
    inner = model
    for _ in range(5):
        if isinstance(inner, (CompositeGraphLLM, GraphAugmentedLLM, GraphMaskLLM)):
            return inner
        nxt = getattr(inner, "base_model", None)
        if nxt is None or nxt is inner:
            nxt = getattr(inner, "model", None)
        if nxt is None or nxt is inner:
            break
        inner = nxt
    return inner


class InMemoryLLM(spine_models.InMemoryLLM):
    """SPINE-compatible LLM client for plain (non-graph-augmented) models.

    Subclasses ``spine.models.InMemoryLLM`` to inherit the shared SPINE client
    contract (``format_prompt``); overrides ``query_llm`` to route through the
    compact prompt translation (SPINE system prompt + ICL dropped; scene graph
    compacted; edge bullets present iff ``include_edges``) and inverse-translate
    the compact output back to SPINE JSON.
    """

    def __init__(self, model, tokenizer, include_edges: bool):
        # Reuse SPINE's __init__ for model/tokenizer/device; device is taken from
        # the model's own parameters rather than the SPINE "cuda" default.
        super().__init__(model, tokenizer, device=next(model.parameters()).device)
        self.include_edges = include_edges

    def _decode(self, outputs) -> str:
        # clean_up_tokenization_spaces=False: WordPiece post-process corrupts BPE (Llama) plan text.
        return self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    def _generate_tokens(self, input_ids, attention_mask, msg, max_new_tokens):
        """Abstracts token generation. In the base case, it's just calling `model.generate`"""
        outputs = self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, **DECODE_KWARGS,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        # Strip the input prefix — keep only newly generated tokens.
        return outputs[:, input_ids.shape[-1]:]

    def query_llm(self, msg: List[Dict], max_new_tokens: int = 2048):
        llm_msg = compact_prompt.spine_to_compact_messages(msg, include_edges=self.include_edges)
        input = self.tokenizer.apply_chat_template(
            llm_msg, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        input_ids = input["input_ids"].to(self.device)
        attention_mask = input["attention_mask"].to(self.device)

        print(f"[spine-llm] client={type(self).__name__}, prompt_tokens={input_ids.shape[1]}")

        with torch.no_grad():
            outputs = self._generate_tokens(input_ids, attention_mask, msg, max_new_tokens)

        compact_response = self._decode(outputs)
        print(f"[spine-llm] raw_output (first 500 chars): {compact_response[:500]}")
        # Inverse-translate compact output back to SPINE JSON for SPINE's parser and grader.
        planner_response = compact_prompt.compact_output_to_spine_json(compact_response)
        return planner_response, True


class GraphAugmentedInMemoryLLM(InMemoryLLM):
    """SPINE-compatible client for GraphAugmentedLLM / CompositeGraphLLM inference.

    GNN always parses the ORIGINAL message for full structural edges; ``include_edges``
    toggles only the LLM-facing text (enabling "PE + text edges" vs "PE only" ablation).
    Compact output is inverse-translated back to SPINE JSON. Falls back to plain LLM
    generation when no graph is found in the prompt.
    """

    def __init__(self, model, tokenizer, include_edges: bool, permutation=None):
        super().__init__(model, tokenizer, include_edges=include_edges)
        self.permutation = permutation

    def _parse_all_pyg_graphs(self, msg: List[Dict]) -> List:
        """Extract all scene graphs from SPINE message list and convert to PyG Data objects."""
        graphs = []
        for m in msg:
            if m.get("role") != "user":
                continue
            for match in re.finditer(r"[Ss]cene graph: ?(.*})", m.get("content", ""), re.DOTALL):
                try:
                    scene_graph_dict = literal_eval(match.group(1))
                    graphs.append(utils.scene_graph_dict_to_pyg(scene_graph_dict))
                except Exception:
                    continue
        return graphs

    def query_llm(self, msg: List[Dict], max_new_tokens: int = 2048):
        # Parse PyG graphs from ORIGINAL message (full connectivity, unaffected by include_edges).
        pyg_graphs = self._parse_all_pyg_graphs(msg)
        llm_msg = compact_prompt.spine_to_compact_messages(msg, include_edges=self.include_edges)

        input = self.tokenizer.apply_chat_template(
            llm_msg, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        input_ids = input["input_ids"].to(self.device)
        attention_mask = input["attention_mask"].to(self.device)

        print(f"[spine-llm] client={type(self).__name__}, prompt_tokens={input_ids.shape[1]}")

        with torch.no_grad():
            outputs = self._generate_tokens(input_ids, attention_mask, pyg_graphs, max_new_tokens)

        compact_response = self._decode(outputs)
        print(f"[spine-llm] raw_output (first 500 chars): {compact_response[:500]}")
        # Inverse-translate compact output back to SPINE JSON.
        planner_response = compact_prompt.compact_output_to_spine_json(compact_response)
        return planner_response, True

    def _generate_tokens(self, input_ids, attention_mask, pyg_graphs, max_new_tokens):
        """Inject R-PEARL PE into token embeddings and run model.generate."""
        robot_loc = pyg_graphs[-1].robot_location if pyg_graphs else None
        print(f"[spine-llm] graph_found={bool(pyg_graphs)}, n_graphs={len(pyg_graphs)}, robot_location={robot_loc}")

        # Unwrap any PEFT wrapper to the graph model so attribute access and the
        # architecture branch below hit the real model, not the PeftModel shell.
        graph_model = _core_graph_model(self.model)

        if not pyg_graphs:
            outputs = graph_model.llm.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens, **DECODE_KWARGS,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            return outputs[:, input_ids.shape[-1]:]

        # Compute base embeddings once (plain X — Ψ is injected inside attention).
        embeddings = (
            graph_model.llm.get_input_embeddings()(input_ids)
            .clone()
            .to(input_ids.device)
        )

        pyg_graph = pyg_graphs[-1]
        input_ids_list = input_ids[0].tolist()
        # Standalone + space-preceded tokenizations per node (100% injection).
        node_token_seqs = node_token_variants(pyg_graph.node_names, self.tokenizer)

        # Scope to last (query) graph block; prevents ICL-example nodes from matching query labels.
        scope_start = find_last_graph_scope(input_ids_list, self.tokenizer)
        print(f"[spine-llm] injection scope_start={scope_start} / {len(input_ids_list)} tokens")

        injection_map = build_injection_map(input_ids_list, node_token_seqs, scope_start=scope_start)

        if isinstance(graph_model, GraphMaskLLM):
            # Build and arm [1, 1, seq, seq] additive adjacency bias; cleared in finally.
            graph_model._struct_bias = graph_model.build_structural_mask(
                input_ids.shape[1], [pyg_graph], [injection_map], input_ids.device)
            try:
                outputs = graph_model.llm.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens, **DECODE_KWARGS,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                return outputs[:, input_ids.shape[-1]:]
            finally:
                graph_model._struct_bias = None

        if isinstance(graph_model, InjectedCompositeGraphLLM):
            # GT code injected post-RoPE into q/k/v per layer via prepare_generation.
            # Injection skips cached decode steps (prompt only); disarmed in finally.
            inputs_embeds = graph_model.prepare_generation(
                input_ids, [pyg_graph], [injection_map], permutation=self.permutation)
            hook_handle = None
            try:
                # Decode-time graph extension: pre-hook extends composite graph and computes
                # new token's kernel row before each decode attention step.
                if getattr(graph_model, "c_bias", False):
                    c = input_ids.shape[1]
                    aug = graph_model._composite_graph(
                        pyg_graph, injection_map, c, inputs_embeds.device,
                        permutation=self.permutation)
                    nts = [(i, seq) for i, variants in enumerate(node_token_seqs)
                           for seq in variants]
                    graph_model.decode_setup(aug, nts, c, c + max_new_tokens,
                                             device=inputs_embeds.device)

                    def _extend_hook(module, args, kwargs):
                        ids = kwargs.get("input_ids")
                        if ids is not None and ids.dim() == 2 and ids.shape[1] == 1:
                            graph_model.decode_extend(int(ids[0, -1]))
                    hook_handle = graph_model.llm.register_forward_pre_hook(
                        _extend_hook, with_kwargs=True)
                return graph_model.llm.generate(
                    inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens, **DECODE_KWARGS,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            finally:
                if hook_handle is not None:
                    hook_handle.remove()
                graph_model.decode_disarm()
                graph_model._pe_signal = None
                graph_model._pe_C = None
                graph_model._pe_Psi = None
                graph_model._pe_c_row = None
                graph_model._pe_taps = None

        if isinstance(graph_model, CompositeGraphLLM):
            # Build composite graph (token cycle + scene + cross-links), fuse embeddings via GT + gate.
            inputs_embeds = graph_model._fuse_embeddings(
                input_ids, [pyg_graph], [injection_map], permutation=self.permutation)
            return graph_model.llm.generate(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens, **DECODE_KWARGS,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Base R-PEARL: build_pe_signal places Ψ at scoped node spans (tanh gate);
        # attention layers add it post-RoPE. Injection skips cached decode steps; disarmed in finally.
        psi = graph_model.build_pe_signal(
            embeddings, [pyg_graph], [injection_map], permutation=self.permutation)
        graph_model._pe_signal = psi
        gen_kwargs = {}
        # Identity-RoPE (position_id 0) for graph-token spans, matching training.
        if getattr(graph_model, "_disable_graph_token_rope", False):
            gen_kwargs["position_ids"] = graph_model.graph_token_position_ids(
                [injection_map], embeddings.shape[1], embeddings.device)
        try:
            return graph_model.llm.generate(
                inputs_embeds=embeddings, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens, **DECODE_KWARGS,
                pad_token_id=self.tokenizer.eos_token_id, **gen_kwargs,
            )
        finally:
            graph_model._pe_signal = None
