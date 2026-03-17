import re
from ast import literal_eval
from typing import List, Dict

import torch

from prism.data import utils
from prism.data.data import remove_edge_list
from prism.models.gnn_llm import build_injection_map


class InMemoryLLM:
    """SPINE-compatible LLM client for plain (non-graph-augmented) models.

    Tokenizes, generates, and decodes. No graph logic.
    """

    def __init__(self, model, tokenizer, device="cuda", strip_edges: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.strip_edges = strip_edges

    def format_prompt(self, base_request: str, graph_as_json: str) -> List[Dict]:
        return [{"role": "user", "content": f"task: {base_request}. scene graph {graph_as_json}"}]

    def _decode(self, outputs) -> str:
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()

    def _generate_tokens(self, input_ids, attention_mask, msg, max_new_tokens):
        """Abstracts token generation. In the base case, it's just calling `model.generate`"""
        outputs = self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, use_cache=True, temperature=0.01, min_p=0.1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        # Strip the input prefix — keep only newly generated tokens.
        return outputs[:, input_ids.shape[-1]:]

    def query_llm(self, msg: List[Dict], max_new_tokens: int = 256):
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
    """

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

    def query_llm(self, msg: List[Dict], max_new_tokens: int = 256):
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

        # Find the last user message containing a scene graph (the real task).
        # PE injection should be scoped to the task message + history only,
        # matching training where no ICL/system prefix exists.
        task_msg_idx = 0
        for i, m in enumerate(msg):
            if m.get("role") == "user" and re.search(r"[Ss]cene graph:", m.get("content", "")):
                task_msg_idx = i
        prefix_len = 0
        if task_msg_idx > 0:
            prefix_len = len(self.tokenizer.apply_chat_template(
                llm_msg[:task_msg_idx], tokenize=True, add_generation_prompt=False
            ))

        print(f"[spine-llm] client={type(self).__name__}, prompt_tokens={input_ids.shape[1]}, pe_scope_offset={prefix_len}")

        with torch.no_grad():
            outputs = self._generate_tokens(input_ids, attention_mask, pyg_graphs, max_new_tokens, prefix_len=prefix_len)

        planner_response = self._decode(outputs)
        print(f"[spine-llm] raw_output (first 500 chars): {planner_response[:500]}")
        return planner_response, True

    def _generate_tokens(self, input_ids, attention_mask, pyg_graphs, max_new_tokens, prefix_len=0):
        """Inject R-PEARL PE into token embeddings and run model.generate."""
        robot_loc = pyg_graphs[-1].robot_location if pyg_graphs else None
        print(f"[spine-llm] graph_found={bool(pyg_graphs)}, n_graphs={len(pyg_graphs)}, robot_location={robot_loc}")

        if not pyg_graphs:
            return super()._generate_tokens(input_ids, attention_mask, None, max_new_tokens)

        # Compute base embeddings once
        embeddings = (
            self.model.llm.get_input_embeddings()(input_ids)
            .clone()
            .to(input_ids.device)
        )

        # Inject GNN positional encodings for the real task graph (last one).
        # Scope the injection map to only the task message + history tokens
        # (skip ICL examples and system prompt) to match training, where
        # input_ids contain only the single-conversation training example.
        pyg_graph = pyg_graphs[-1]
        input_ids_list = input_ids[0].tolist()
        node_token_seqs = [
            self.tokenizer.encode(name, add_special_tokens=False)
            for name in pyg_graph.node_names
        ]
        task_ids = input_ids_list[prefix_len:]
        injection_map_local = build_injection_map(task_ids, node_token_seqs)
        injection_map = {
            nid: [(s + prefix_len, e + prefix_len) for s, e in spans]
            for nid, spans in injection_map_local.items()
        }
        pe = self.model.pe_proj(self.model.pe_model(pyg_graph))  # [n, hidden_size]

        # Rescale PE to match embedding norm. pe_proj ends with LayerNorm which
        # forces output to norm ≈ sqrt(d_model) ≈ 64, while LLM embeddings
        # (pre-RMSNorm) sit at ~0.5. Without rescaling, PE overwhelms embeddings.
        target_norm = embeddings.norm(dim=-1).mean()
        pe = torch.nn.functional.normalize(pe, dim=-1) * target_norm

        for node_idx, spans in injection_map.items():
            for start, end in spans:
                end = min(end, input_ids.shape[1])
                embeddings[0, start:end] = embeddings[0, start:end] + pe[node_idx]

        return self.model.llm.generate(
            inputs_embeds=embeddings, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, use_cache=True, temperature=0.01, min_p=0.1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
