import re
from ast import literal_eval
from typing import List, Dict

import torch

from prism.data import utils
from prism.data.data import remove_edge_list
from prism.models.gnn_llm import (
    AugmentedGraphLLM,
    GraphAugmentedLLM,
    build_injection_map,
    find_last_graph_scope,
)


def _core_graph_model(model):
    """Peel PEFT wrappers to the AugmentedGraphLLM / GraphAugmentedLLM core.

    The eval model is loaded PEFT-wrapped (PeftModel -> LoraModel -> graph model),
    so ``isinstance(model, AugmentedGraphLLM)`` is False on the wrapper and the
    wrong injection branch runs (then fails on a missing ``pe_proj``). Unwrap to
    the underlying graph model first. The LoRA adapters live inside the graph
    model's ``.llm`` submodule (PEFT patches it in place), so calling the core's
    methods/``.llm`` still runs the adapted LLM.
    """
    inner = model
    for _ in range(5):
        if isinstance(inner, (AugmentedGraphLLM, GraphAugmentedLLM)):
            return inner
        nxt = getattr(inner, "base_model", None)
        if nxt is None or nxt is inner:
            nxt = getattr(inner, "model", None)
        if nxt is None or nxt is inner:
            break
        inner = nxt
    return inner


class InMemoryLLM:
    """SPINE-compatible LLM client for plain (non-graph-augmented) models.

    Tokenizes, generates, and decodes. No graph logic.
    """

    def __init__(self, model, tokenizer, device=None, strip_edges: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if device is not None else next(model.parameters()).device
        self.strip_edges = strip_edges

    def format_prompt(self, base_request: str, graph_as_json: str) -> List[Dict]:
        return [{"role": "user", "content": f"task: {base_request}. scene graph {graph_as_json}"}]

    def _decode(self, outputs) -> str:
        # clean_up_tokenization_spaces=False: the cleanup step is a WordPiece
        # post-process that is destructive for BPE (Llama) — it strips spaces
        # before punctuation and corrupts the generated plan text. Off here also
        # silences the transformers warning.
        return self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    def _generate_tokens(self, input_ids, attention_mask, msg, max_new_tokens):
        """Abstracts token generation. In the base case, it's just calling `model.generate`"""
        outputs = self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, use_cache=True, temperature=0.01, min_p=0.1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        # Strip the input prefix — keep only newly generated tokens.
        return outputs[:, input_ids.shape[-1]:]

    def query_llm(self, msg: List[Dict], max_new_tokens: int = 2048):
        if self.strip_edges:
            msg = [
                {**m, "content": remove_edge_list(m["content"])} if m["role"] == "user" else m
                for m in msg
            ]
        input = self.tokenizer.apply_chat_template(
            msg, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        input_ids = input["input_ids"].to(self.device)
        attention_mask = input["attention_mask"].to(self.device)

        print(f"[spine-llm] client={type(self).__name__}, strip_edges={self.strip_edges}, prompt_tokens={input_ids.shape[1]}")

        with torch.no_grad():
            outputs = self._generate_tokens(input_ids, attention_mask, msg, max_new_tokens)

        planner_response = self._decode(outputs)
        print(f"[spine-llm] raw_output (first 500 chars): {planner_response[:500]}")
        return planner_response, True


class GraphAugmentedInMemoryLLM(InMemoryLLM):
    """SPINE-compatible LLM client that runs full GraphAugmentedLLM (GNN + LoRA) inference.

    Parses the scene graph from the SPINE prompt text, builds a PyG graph,
    computes GNN-augmented embeddings, and generates via the LoRA-modified LLM.
    Falls back to plain LLM generation when no graph is found in the prompt.

    When ``strip_edges=True`` (matching ``text_edge_list=none`` in the training
    config), the edge list (object_connections / region_connections) is removed
    from the text the LLM sees so graph connectivity is communicated exclusively
    via R-PEARL PE injection.  When ``strip_edges=False`` (``text_edge_list=present``),
    edges remain in the text and PE is additive.  Graph parsing always runs on the
    original unmodified message so the GNN retains complete edge information
    regardless of this setting.

    Extra ICL examples are omitted by the caller (SPINE with use_icl=False),
    keeping only the system prompt and one ICL example. PE injection operates
    over the full input_ids without per-message offset correction.
    """

    def __init__(self, model, tokenizer, device=None, strip_edges: bool = False, permutation=None):
        super().__init__(model, tokenizer, device, strip_edges)
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
        # Always parse PyG graphs from the original message so the GNN has full
        # connectivity regardless of text_edge_list setting.
        pyg_graphs = self._parse_all_pyg_graphs(msg)

        llm_msg = (
            [
                {**m, "content": remove_edge_list(m["content"])} if m["role"] == "user" else m
                for m in msg
            ]
            if self.strip_edges
            else msg
        )

        input = self.tokenizer.apply_chat_template(
            llm_msg, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        input_ids = input["input_ids"].to(self.device)
        attention_mask = input["attention_mask"].to(self.device)

        print(f"[spine-llm] client={type(self).__name__}, prompt_tokens={input_ids.shape[1]}")

        with torch.no_grad():
            outputs = self._generate_tokens(input_ids, attention_mask, pyg_graphs, max_new_tokens)

        planner_response = self._decode(outputs)
        print(f"[spine-llm] raw_output (first 500 chars): {planner_response[:500]}")
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
                max_new_tokens=max_new_tokens, use_cache=True, temperature=0.01, min_p=0.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            return outputs[:, input_ids.shape[-1]:]

        # Compute base embeddings once
        embeddings = (
            graph_model.llm.get_input_embeddings()(input_ids)
            .clone()
            .to(input_ids.device)
        )

        pyg_graph = pyg_graphs[-1]
        input_ids_list = input_ids[0].tolist()
        node_token_seqs = [
            self.tokenizer.encode(name, add_special_tokens=False)
            for name in pyg_graph.node_names
        ]

        # Scope injection to the last (query) graph block: only token positions
        # at/after the final "scene graph:" marker are eligible. Otherwise the
        # query graph's node labels would also match identical strings inside
        # earlier ICL-example graphs and inject PE into the wrong regions.
        scope_start = find_last_graph_scope(input_ids_list, self.tokenizer)
        print(f"[spine-llm] injection scope_start={scope_start} / {len(input_ids_list)} tokens")

        injection_map = build_injection_map(input_ids_list, node_token_seqs, scope_start=scope_start)

        if isinstance(graph_model, AugmentedGraphLLM):
            # M9: assemble the augmented graph (cycle + scene + cross-links) from the
            # last graph's scene PyG + scoped injection map, run M4→M6→M7, and feed
            # the fused token embeddings to the RoPE-disabled LLM.
            inputs_embeds = graph_model._fuse_embeddings(
                input_ids, [pyg_graph], [injection_map], permutation=self.permutation)
            return graph_model.llm.generate(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens, use_cache=True, temperature=0.01, min_p=0.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        pe = graph_model.pe_proj(graph_model.pe_model(pyg_graph, permutation=self.permutation))  # [n, hidden_size]
        pe = pe * graph_model.pe_gain

        for node_idx, spans in injection_map.items():
            for start, end in spans:
                end = min(end, input_ids.shape[1])
                embeddings[0, start:end] = embeddings[0, start:end] + pe[node_idx]

        return graph_model.llm.generate(
            inputs_embeds=embeddings, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, use_cache=True, temperature=0.01, min_p=0.1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
