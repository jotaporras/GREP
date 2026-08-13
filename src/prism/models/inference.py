import re
from ast import literal_eval
from typing import List, Dict

import torch

from spine import models as spine_models

from prism.data import compact_prompt
from prism.data import utils
from prism.models.gnn_llm import (
    MaskDecodeInjector,
    GraphAugmentedLLM,
    GraphMaskLLM,
    LearnableGraphMaskLLM,
    MagCompGraphLLM,
    WireGraphLLM,
    build_injection_map,
    find_last_graph_scope,
    node_token_variants,
)


# Decode is deterministic GREEDY for confirmatory eval (reproducible, no seed needed).
# NOTE: the prior `temperature=0.01, min_p=0.1` were dead config — without `do_sample=True`
# transformers ignores them and runs greedy, so this makes the actual behavior explicit
# (results unchanged). Sampling params live here, in one place, if ever re-enabled.
DECODE_KWARGS = {"do_sample": False, "use_cache": True}

# Generation budget. SPINE mode costs far more output: the model has to work through
# the tool tutorial, ratify each hop, and then still write the action list — and a
# generation cut off before the plan line yields an EMPTY plan
# (``compact_output_to_spine_json``), i.e. a lost sample rather than a wrong one.
# Measured on a 24-node graph, tool-free answers land in a few hundred tokens while
# ratifying answers ran past 1k, so SPINE gets 4x the tool-free budget.
MAX_NEW_TOKENS = 2048
SPINE_TOKEN_MULTIPLIER = 4


# Moved to gnn_llm (spine-free) so the RL trainer can import it; kept under the
# historical name for the existing call sites in this module and evaluate.py.
from prism.models.gnn_llm import core_graph_model as _core_graph_model  # noqa: E402


def _identity_rope_kwargs(graph_model, injection_map, seq_len, device) -> dict:
    """``{"position_ids": ...}`` when the checkpoint trained with identity-RoPE, else ``{}``.

    Duck-typed on ``_disable_graph_token_rope`` so the additive family and the learned
    mask share one call site each. Only the PROMPT is covered: transformers advances
    position_ids per decode step, so generated tokens get natural positions (the mask
    family's ``MaskDecodeInjector`` re-zeroes the query-tagged steps, restoring parity
    for ``decode_consistent``).

    Fails loud if the last prompt position is zeroed: ``_update_model_kwargs_for_
    generation`` derives every generated position as ``position_ids[..., -1:] + n``, so a
    zeroed final prompt token would silently restart the whole rollout at position 1.
    """
    if not getattr(graph_model, "_disable_graph_token_rope", False):
        return {}
    position_ids = graph_model.graph_token_position_ids([injection_map], seq_len, device)
    if seq_len > 1 and int(position_ids[0, -1]) == 0:
        raise RuntimeError(
            "identity-RoPE zeroed the FINAL prompt position: transformers derives each "
            "generated position from the last one, so every generated token would be "
            "numbered from 1. The prompt must not end inside a node mention "
            f"(seq_len={seq_len}).")
    return {"position_ids": position_ids}


class InMemoryLLM(spine_models.InMemoryLLM):
    """SPINE-compatible LLM client for plain (non-graph-augmented) models.

    Subclasses ``spine.models.InMemoryLLM`` to inherit the shared SPINE client
    contract (``format_prompt``); overrides ``query_llm`` to route through the
    compact prompt translation (verbose SPINE system prompt replaced by the compact
    one; scene graph compacted; edge bullets present iff ``include_edges``; the
    first ``icl_examples`` SPINE few-shot examples kept and compacted; SPINE API
    documented and action-list plans preserved iff ``include_tools``) and
    inverse-translate the compact output back to SPINE JSON.
    """

    def __init__(self, model, tokenizer, include_edges: bool, include_tools: bool,
                 icl_examples: int):
        # Reuse SPINE's __init__ for model/tokenizer/device; device is taken from
        # the model's own parameters rather than the SPINE "cuda" default.
        super().__init__(model, tokenizer, device=next(model.parameters()).device)
        self.include_edges = include_edges
        # Tool policy must match the simulator the planning loop runs
        # (evaluate._spine_tools_disabled -> _NoToolsGraphSim), and the ICL count must
        # match what the SPINE header actually carries; evaluate.py sets both.
        self.include_tools = include_tools
        self.icl_examples = icl_examples
        # SPINE mode gets SPINE_TOKEN_MULTIPLIER x the tool-free budget (see the constants).
        self.max_new_tokens = MAX_NEW_TOKENS * (SPINE_TOKEN_MULTIPLIER if include_tools else 1)

    def _decode(self, outputs) -> str:
        # clean_up_tokenization_spaces=False: WordPiece post-process corrupts BPE plan text.
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

    def query_llm(self, msg: List[Dict], max_new_tokens: int = None):
        # None -> the policy budget resolved at construction (4x under SPINE).
        max_new_tokens = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        llm_msg = compact_prompt.spine_to_compact_messages(
            msg, include_edges=self.include_edges, include_tools=self.include_tools,
            icl_examples=self.icl_examples)
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
    """SPINE-compatible client for GraphAugmentedLLM / graph-mask inference.

    GNN always parses the ORIGINAL message for full structural edges; ``include_edges``
    toggles only the LLM-facing text (enabling "PE + text edges" vs "PE only" ablation).
    Compact output is inverse-translated back to SPINE JSON. Falls back to plain LLM
    generation when no graph is found in the prompt.
    """

    # "gaussian" | "binary" — MUST match the train-time data.edge_weights policy
    # (see scene_graph_dict_to_pyg). Class-level default (historical Gaussian
    # affinity) so partially-constructed instances (tests) resolve it too.
    edge_weights = "gaussian"
    # Train-time data.injection_scope; "decode_consistent" arms MaskDecodeInjector
    # during generation (mask archs). Every other value generates as before
    # (prompt-only prefill wiring, no decode injection). Class-level default so
    # partially-constructed instances (tests) resolve it too.
    injection_scope = "full_sequence"

    def __init__(self, model, tokenizer, include_edges: bool, include_tools: bool,
                 icl_examples: int, permutation=None,
                 edge_weights: str = "gaussian",
                 injection_scope: str = "full_sequence"):
        super().__init__(model, tokenizer, include_edges=include_edges,
                         include_tools=include_tools, icl_examples=icl_examples)
        self.permutation = permutation
        self.edge_weights = edge_weights
        self.injection_scope = injection_scope

    def _parse_all_pyg_graphs(self, msg: List[Dict]) -> List:
        """Extract all scene graphs from SPINE message list and convert to PyG Data objects."""
        graphs = []
        for m in msg:
            if m.get("role") != "user":
                continue
            for match in re.finditer(r"[Ss]cene graph: ?(.*})", m.get("content", ""), re.DOTALL):
                try:
                    scene_graph_dict = literal_eval(match.group(1))
                    graphs.append(utils.scene_graph_dict_to_pyg(
                        scene_graph_dict, edge_weights=self.edge_weights))
                except Exception:
                    continue
        return graphs

    def query_llm(self, msg: List[Dict], max_new_tokens: int = None):
        # None -> the policy budget resolved at construction (4x under SPINE).
        max_new_tokens = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        # Parse PyG graphs from ORIGINAL message (full connectivity, unaffected by include_edges).
        pyg_graphs = self._parse_all_pyg_graphs(msg)
        llm_msg = compact_prompt.spine_to_compact_messages(
            msg, include_edges=self.include_edges, include_tools=self.include_tools,
            icl_examples=self.icl_examples)

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

        pyg_graph = pyg_graphs[-1]
        input_ids_list = input_ids[0].tolist()
        # Standalone + space-preceded tokenizations per node (100% injection).
        node_token_seqs = node_token_variants(pyg_graph.node_names, self.tokenizer)

        # Scope to last (query) graph block; prevents ICL-example nodes from matching query labels.
        scope_start = find_last_graph_scope(input_ids_list, self.tokenizer)
        print(f"[spine-llm] injection scope_start={scope_start} / {len(input_ids_list)} tokens")

        injection_map = build_injection_map(input_ids_list, node_token_seqs, scope_start=scope_start)

        if isinstance(graph_model, WireGraphLLM):
            # WIRE: the signal is armed over the prompt and stays armed for the whole
            # rollout; the attention hook re-rotates the cached prompt keys each step and
            # leaves generated positions at r=0 (prompt-only decode, same as the additive
            # and mask branches below). generate_with_graph also disarms in a finally, so
            # a stale signal can never leak into the next sample.
            if self.injection_scope == "decode_consistent":
                raise NotImplementedError(
                    "injection_scope='decode_consistent' is not wired for WireGraphLLM: "
                    "extending the graph channel to GENERATED node mentions needs the "
                    "mask family's q/kv split (MaskDecodeInjector), which has no rotation "
                    "analogue. Evaluate this checkpoint with a prompt-only scope, or "
                    "retrain with data.injection_scope='prompt_only'.")
            outputs = graph_model.generate_with_graph(
                input_ids=input_ids, graphs=[pyg_graph], injection_maps=[injection_map],
                permutation=self.permutation, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens, **DECODE_KWARGS,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            return outputs[:, input_ids.shape[-1]:]
        if isinstance(graph_model, (GraphMaskLLM, LearnableGraphMaskLLM)):
            # Build and arm [1, 1, seq, seq] additive (adjacency or learned-relative-PE) bias;
            # both classes share build_structural_mask + _struct_bias; cleared in finally.
            # Arming happens inside the try so a construction-time failure (e.g. OOM
            # building Ψ on a large graph) can never leave _struct_bias or the hook
            # attached across samples.
            hook_handle = None
            try:
                # `permutation` reaches BOTH the Ψ producer and the adjacency inside
                # build_structural_mask, so --permutation-seed actually relabels the graph
                # the mask is computed on (it was silently ignored here before).
                graph_model._struct_bias = graph_model.build_structural_mask(
                    input_ids.shape[1], [pyg_graph], [injection_map], input_ids.device,
                    permutation=self.permutation,
                    # MagCompGraphLLM's text layer IS the token sequence, so its bias
                    # needs the ids; the Psi Psi^T mask is a function of the graph alone
                    # and its signature has no room for them.
                    **({"input_ids": input_ids, "attention_mask": attention_mask}
                       if isinstance(graph_model, MagCompGraphLLM) else {}))
                # decode_consistent checkpoints extend the channel to generated
                # mentions: a forward pre-hook arms a per-step bias row (span-end
                # assignment, design note §2.2). Other scopes keep the historical
                # prompt-only behavior.
                if self.injection_scope == "decode_consistent":
                    injector = MaskDecodeInjector(
                        graph_model, pyg_graph, injection_map,
                        input_ids.shape[1], node_token_seqs,
                        permutation=self.permutation)
                    hook_handle = graph_model.llm.register_forward_pre_hook(
                        injector.pre_hook, with_kwargs=True)
                outputs = graph_model.llm.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens, **DECODE_KWARGS,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **_identity_rope_kwargs(graph_model, injection_map,
                                            input_ids.shape[1], input_ids.device),
                )
                return outputs[:, input_ids.shape[-1]:]
            finally:
                graph_model._struct_bias = None
                if hook_handle is not None:
                    hook_handle.remove()
                    graph_model._decode_bias_row = None

        # Base R-PEARL: build_pe_signal places Ψ at scoped node spans (tanh gate);
        # attention layers add it post-RoPE. Injection skips cached decode steps; disarmed in finally.
        # Base embeddings (plain X — Ψ is injected inside attention). Built HERE, not above
        # the branches: the WIRE and mask paths never read them, and an unconditional
        # [1, seq, hidden] clone of the embedding table is real memory on every eval sample.
        embeddings = (
            graph_model.llm.get_input_embeddings()(input_ids)
            .clone()
            .to(input_ids.device)
        )
        psi = graph_model.build_pe_signal(
            embeddings, [pyg_graph], [injection_map], permutation=self.permutation)
        graph_model._pe_signal = psi
        # Identity-RoPE (position_id 0) for graph-token spans, matching training.
        gen_kwargs = _identity_rope_kwargs(
            graph_model, injection_map, embeddings.shape[1], embeddings.device)
        try:
            return graph_model.llm.generate(
                inputs_embeds=embeddings, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens, **DECODE_KWARGS,
                pad_token_id=self.tokenizer.eos_token_id, **gen_kwargs,
            )
        finally:
            graph_model._pe_signal = None
