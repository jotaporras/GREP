import re
from ast import literal_eval
from typing import Optional

from torch_geometric.data import Batch
from transformers.data.data_collator import DataCollatorForLanguageModeling

from prism.data import utils
from prism.models.gnn_llm import build_injection_map


def remove_edge_list(decoded: str) -> str:
    """Remove the edge list (object_connections and region_connections) from
    a decoded prompt string containing a scene graph.

    Parameters
    ----------
    decoded : str
        The full decoded prompt text that contains a scene graph with
        ``'object_connections': ...`` and ``'region_connections': ...`` entries.

    Returns
    -------
    str
        The prompt with both connection lists removed.
    """
    pattern = r"'object_connections': .+?, 'region_connections': .+?, (?='robot_location'|\})"
    return re.sub(pattern, "", decoded)


class DataCollatorForGraphAugmentedLLM(DataCollatorForLanguageModeling):
    """Base collator for graph-augmented LLM training.

    Subclasses override ``_extract_graph`` to parse domain-specific graph
    formats.  This base handles batching ``graphs`` and ``injection_maps``
    into the batch dict.
    """

    def _extract_graph(self, example):
        """Parse a single training example and return graph + injection map.

        Subclasses must override this method.

        Parameters
        ----------
        example : dict
            A single training example containing at minimum ``input_ids``
            and ``attention_mask``.

        Returns
        -------
        tuple
            ``(pyg_graph, injection_map, input_ids, attention_mask)`` where
            ``injection_map`` is ``dict[int, list[tuple[int, int]]]``.
        """
        raise NotImplementedError("Subclasses must implement _extract_graph")

    def __call__(self, features, return_tensors: Optional[str] = None):
        """Attach parsed PyG graphs and injection maps for each example."""
        pyg_graphs = []
        injection_maps = []
        sanitized_examples = []

        for example in features:
            pyg_graph, injection_map, input_ids, attention_mask = self._extract_graph(example)
            pyg_graphs.append(pyg_graph)
            injection_maps.append(injection_map)

            example['input_ids'] = input_ids
            example['attention_mask'] = attention_mask

            sanitized_examples.append(
                {
                    k: v
                    for k, v in example.items()
                    if k not in {"conversations", "scene_graph", "messages", "text"}
                }
            )

        batch = super().__call__(sanitized_examples)
        batch["graphs"] = Batch.from_data_list(pyg_graphs)
        batch["injection_maps"] = injection_maps
        return batch


class SpineDataCollator(DataCollatorForGraphAugmentedLLM):
    """SPINE scene-graph collator.

    Parses SPINE scene graphs from prompt text, builds PyG graphs, and
    computes injection maps via ``build_injection_map``.
    """

    def __init__(self, *args, text_edge_list: str = "present", **kwargs):
        super().__init__(*args, **kwargs)
        if text_edge_list not in ("present", "none"):
            raise ValueError(f"text_edge_list must be 'present' or 'none', got {text_edge_list!r}")
        self.text_edge_list = text_edge_list

    def _extract_graph(self, example):
        """Parse SPINE scene graph from example and return graph + injection map."""
        pattern = r"[Ss]cene graph:"

        # Use full messages text to avoid truncation issues from max_seq_length.
        if 'messages' in example:
            full_text = self.tokenizer.apply_chat_template(example['messages'], tokenize=False)
        else:
            full_text = self.tokenizer.decode(example['input_ids'])

        if re.search(pattern=pattern, string=full_text):
            scene_graph_text = re.findall(pattern + r" ?(.*})", full_text)[0]
            scene_graph_dict = literal_eval(scene_graph_text)
        else:
            raise ValueError(f"No scene graph found in prompt")

        pyg_graph = utils.scene_graph_dict_to_pyg(scene_graph_dict)

        # Sanitize input IDs and attention masks.
        decoded = self.tokenizer.decode(example['input_ids'])
        cleaned = remove_edge_list(decoded) if self.text_edge_list == "none" else decoded
        encoded = self.tokenizer(cleaned, return_tensors="pt")
        input_ids = encoded['input_ids'].squeeze().tolist()
        attention_mask = encoded['attention_mask'].squeeze().tolist()

        # Compute injection map.
        node_token_seqs = [
            self.tokenizer.encode(name, add_special_tokens=False)
            for name in pyg_graph.node_names
        ]
        injection_map = build_injection_map(input_ids, node_token_seqs)

        return pyg_graph, injection_map, input_ids, attention_mask
