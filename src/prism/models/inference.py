import re
from ast import literal_eval
from typing import List, Dict

import torch

from prism.data import utils
from prism.models.gnn_llm import build_injection_map


class InMemoryLLM:
    """SPINE-compatible LLM client for plain (non-graph-augmented) models.

    Tokenizes, generates, and decodes. No graph logic.
    """

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def format_prompt(self, base_request: str, graph_as_json: str) -> List[Dict]:
        return [{"role": "user", "content": f"task: {base_request}. scene graph {graph_as_json}"}]

    def _decode(self, outputs) -> str:
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()

    def _generate_tokens(self, input_ids, msg, max_new_tokens):
        """Abstracts token generation. In the base case, it's just calling `model.generate`"""
        outputs = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens, use_cache=True, temperature=0.01, min_p=0.1,
        )
        # Strip the input prefix — keep only newly generated tokens.
        return outputs[:, input_ids.shape[-1]:]

    def query_llm(self, msg: List[Dict], max_new_tokens: int = 256):
        input_ids = self.tokenizer.apply_chat_template(
            msg, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )["input_ids"].to(self.device)

        print(f"[spine-llm] client={type(self).__name__}, prompt_tokens={input_ids.shape[1]}")

        with torch.no_grad():
            outputs = self._generate_tokens(input_ids, msg, max_new_tokens)

        planner_response = self._decode(outputs)
        print(f"[spine-llm] raw_output (first 500 chars): {planner_response[:500]}")
        return planner_response, True


class GraphAugmentedInMemoryLLM(InMemoryLLM):
    """SPINE-compatible LLM client that runs full GraphAugmentedLLM (GNN + LoRA) inference.

    Parses the scene graph from the SPINE prompt text, builds a PyG graph,
    computes GNN-augmented embeddings, and generates via the LoRA-modified LLM.
    Falls back to plain LLM generation when no graph is found in the prompt.
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

    def _generate_tokens(self, input_ids, msg, max_new_tokens):
        """
        Generate model tokens by injecting graph embeddings and calling `model.generate`.
        """
        pyg_graphs = self._parse_all_pyg_graphs(msg)

        robot_loc = pyg_graphs[-1].robot_location if pyg_graphs else None
        print(f"[spine-llm] graph_found={bool(pyg_graphs)}, n_graphs={len(pyg_graphs)}, robot_location={robot_loc}")

        if not pyg_graphs:
            return super()._generate_tokens(input_ids, msg, max_new_tokens)

        # Compute base embeddings once
        embeddings = (
            self.model.llm.get_input_embeddings()(input_ids)
            .clone()
            .to(input_ids.device)
        )

        # Additively apply GNN positional encodings for each scene graph
        input_ids_list = input_ids[0].tolist()
        for pyg_graph in pyg_graphs:
            node_token_seqs = [
                self.tokenizer.encode(name, add_special_tokens=False)
                for name in pyg_graph.node_names
            ]
            injection_map = build_injection_map(input_ids_list, node_token_seqs)
            pe = self.model.pe_proj(self.model.pe_model(pyg_graph))  # [n, hidden_size]
            for node_idx, spans in injection_map.items():
                for start, end in spans:
                    end = min(end, input_ids.shape[1])
                    embeddings[0, start:end] = embeddings[0, start:end] + pe[node_idx]

        return self.model.generate(
            inputs_embeds=embeddings,
            max_new_tokens=max_new_tokens, use_cache=True, temperature=0.01, min_p=0.1,
        )
