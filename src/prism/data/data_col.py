import re
from ast import literal_eval
from typing import Optional

import torch
import torch_geometric.utils as pyg_utils
from transformers.data.data_collator import DataCollatorForLanguageModeling

from prism.data.utils import safe_parse_graph


class DataCollatorForGraphAugmentedLLM(DataCollatorForLanguageModeling):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, features, return_tensors: Optional[str] = None):
        """Attach parsed PyG graphs for each conversation example."""
        messages = []
        pyg_graphs = []
        conversations = []
        sanitized_examples = []

        for example in features:
            pattern = r"[Ss]cene graph:"
            # Graph is in the first message
            prompt = self.tokenizer.decode(example['input_ids'])
            if re.search(pattern=pattern, string=prompt):
                scene_graph_text = re.findall(pattern + r" ?(.*})", prompt)[0]
                scene_graph_dict = literal_eval(scene_graph_text)  # handles the single quotes safely
            else:
                raise ValueError(f"No scene graph found in prompt: {prompt}")

            try:
                nx_graph, _ = safe_parse_graph(scene_graph_dict)
                node_names = list(nx_graph.nodes)
                coords = torch.tensor(
                    [nx_graph.nodes[node]["coords"] for node in node_names],
                    dtype=torch.float32,
                )

                pyg_graph = pyg_utils.from_networkx(nx_graph)
                pyg_graph.coords = coords
                pyg_graph.x = torch.zeros((coords.size(0), 1), dtype=torch.float32)
                pyg_graph.edge_index = pyg_graph.edge_index
                pyg_graph.node_names = node_names
                pyg_graph.node_types = [nx_graph.nodes[node]["type"] for node in node_names]
                pyg_graph.robot_location = scene_graph_dict.get("robot_location")
                pyg_graph.raw_scene_graph = scene_graph_dict
                pyg_graphs.append(pyg_graph)

                # Sanitize input IDs and attention masks.
                pattern = r"'object_connections':"
                decoded = self.tokenizer.decode(example['input_ids'])
                cleaned = re.sub(pattern + r' ?.*,', '', decoded)
                encoded = self.tokenizer(cleaned, return_tensors="pt")
                example['input_ids'] = encoded['input_ids'].squeeze().tolist()
                example['attention_mask'] = encoded['attention_mask'].squeeze().tolist()

                # Sanitize conversations and messages for later reinstallation.
                """
                if 'conversations' in example.keys() and messages in example.keys():
                    conv, mes = example['conversations'], example['messages']
                    conv[0]['content'] = re.sub(pattern + r' ?.*,', '', conv[0]['content'])
                    mes[0]['content'] = re.sub(pattern + r' ?.*,', '', mes[0]['content'])
                    conversations.append(conv)
                    messages.append(mes)
                """

                sanitized_examples.append(
                    {
                        k: v
                        for k, v in example.items()
                        if k not in {"conversations", "scene_graph", "messages", "text"}
                    }
                )
            except Exception as e:
                print(f"Error parsing scene graph: {e}")
        # Call the parent collator to get the tensors (on sanitized examples so that it doesn't try to tensorize the scene graph/text)
        batch = super().__call__(sanitized_examples)
        batch["graphs"] = pyg_graphs
        if conversations and messages:
            batch['conversations'] = conversations
            batch['messages'] = messages
        return batch
