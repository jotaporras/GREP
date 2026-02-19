import re
from ast import literal_eval
from typing import List, Dict

import torch
import torch_geometric.utils as pyg_utils

from prism.data.utils import safe_parse_graph


class GraphAugmentedInMemoryLLM:
    """SPINE-compatible LLM client that runs full GraphAugmentedLLM (GNN + LoRA) inference.

    Parses the scene graph from the SPINE prompt text, builds a PyG graph,
    computes GNN-augmented embeddings, and generates via the LoRA-modified LLM.
    """

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def format_prompt(self, base_request: str, graph_as_json: str) -> List[Dict]:
        return [{"role": "user", "content": f"task: {base_request}. scene graph {graph_as_json}"}]

    def _parse_pyg_graph(self, msg: List[Dict]):
        """Extract scene graph from SPINE message list and convert to PyG Data object."""
        for m in reversed(msg):
            if m.get("role") == "user":
                match = re.search(r"[Ss]cene graph: ?(.*})", m.get("content", ""))
                if match:
                    scene_graph_dict = literal_eval(match.group(1))
                    nx_graph, _ = safe_parse_graph(scene_graph_dict)
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

    def query_llm(self, msg: List[Dict], max_new_tokens: int = 4048):
        pyg_graph = self._parse_pyg_graph(msg)

        input_ids = self.tokenizer.apply_chat_template(
            msg, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )["input_ids"].to(self.device)

        with torch.no_grad():
            if pyg_graph is not None:
                augmented_embeds = self.model._augment_embeddings(input_ids, [pyg_graph])
                outputs = self.model.generate(
                    inputs_embeds=augmented_embeds,
                    max_new_tokens=max_new_tokens, use_cache=True, temperature=0.01, min_p=0.1,
                )
            else:
                outputs = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens, use_cache=True, temperature=0.01, min_p=0.1,
                )

        out = self.tokenizer.batch_decode(outputs)
        planner_response = out[0].split("end_header_id|>")[-1].split("<|eot_id|>")[0]
        return planner_response, True
