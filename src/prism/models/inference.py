import re
from ast import literal_eval
from typing import List, Dict

import torch
import torch_geometric.utils as pyg_utils

from prism.data import utils


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
        out = self.tokenizer.batch_decode(outputs)
        return out[0].split("end_header_id|>")[-1].split("<|eot_id|>")[0]

    def _generate_tokens(self, input_ids, msg, max_new_tokens):
        """Abstracts token generation. In the base case, it's just calling `model.generate`"""
        return self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens, use_cache=True, temperature=0.01, min_p=0.1,
        )

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

    def _parse_pyg_graph(self, msg: List[Dict]):
        """Extract scene graph from SPINE message list and convert to PyG Data object."""
        for m in reversed(msg):
            if m.get("role") == "user":
                match = re.search(r"[Ss]cene graph: ?(.*})", m.get("content", ""))
                if match:
                    scene_graph_dict = literal_eval(match.group(1))
                    nx_graph, _ = utils.safe_parse_graph(scene_graph_dict)
                    node_names = list(nx_graph.nodes)
                    coords = torch.tensor(
                        [nx_graph.nodes[n]["coords"] for n in node_names], dtype=torch.float32
                    )
                    pyg_graph = pyg_utils.from_networkx(nx_graph)
                    pyg_graph.coords = coords
                    pyg_graph.x = torch.zeros((coords.size(0), 1), dtype=torch.float32)
                    pyg_graph.node_names = node_names
                    pyg_graph.node_types = [nx_graph.nodes[n]["type"] for n in node_names]
                    pyg_graph.robot_location = scene_graph_dict.get("robot_location")
                    return pyg_graph
        return None

    def _generate_tokens(self, input_ids, msg, max_new_tokens):
        """
        Generate model tokens by injecting graph embeddings and calling `model.generate`.
        """
        pyg_graph = self._parse_pyg_graph(msg)
        robot_loc = pyg_graph.robot_location if pyg_graph is not None else None
        print(f"[spine-llm] graph_found={pyg_graph is not None}, robot_location={robot_loc}")

        if pyg_graph is None:
            return super()._generate(input_ids, msg, max_new_tokens)

        augmented_embeds = self.model._augment_embeddings(input_ids, [pyg_graph])
        return self.model.generate(
            inputs_embeds=augmented_embeds,
            max_new_tokens=max_new_tokens, use_cache=True, temperature=0.01, min_p=0.1,
        )
