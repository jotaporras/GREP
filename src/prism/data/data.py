import ast
import json
import re
from typing import Optional, no_type_check

import datasets
from torch_geometric.data import Batch
from transformers.data.data_collator import DataCollatorForLanguageModeling

from prism.data import compact_prompt
from prism.data import utils
from prism.models.gnn_llm import build_injection_map, find_last_graph_scope, node_token_variants


def preprocess_dataset(
    ds: datasets.Dataset,
    tokenizer,
    architecture: str,
    text_edge_list: str,
) -> datasets.Dataset:
    """Prepare a raw JSON dataset for training.

    Applies three transforms in order:
    1. Rename ``conversations`` → ``messages``.
    2. Translate to the compact format via ``spine_to_compact_messages``: graph
       archs get the node-only block (GNN supplies edges); the ``llm`` baseline
       gets the same block WITH edge bullets (``include_edges=True``), since it
       has no GNN. ``text_edge_list`` no longer gates the LLM text.
    3. Tokenize with the chat template, keeping ``conversations`` and
       ``messages`` columns for the collator, then filter out examples that
       have no assistant turn.
    """
    @no_type_check
    def _tokenize(example):
        tokenized = tokenizer.apply_chat_template(
            example["messages"], tokenize=True, return_dict=True
        )
        tokenized["conversations"] = example["conversations"]
        tokenized["messages"] = example["messages"]
        return tokenized

    def _parse_scene_graph(example):
        full_text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        m = re.search(r"[Ss]cene graph:", full_text)
        start = full_text.index("{", m.end())
        tail = full_text[start:]
        try:
            sg, _ = json.JSONDecoder().raw_decode(tail)
        except json.JSONDecodeError:
            # Some rollouts serialize the scene graph as a Python repr
            # (single-quoted dict). Fall back to a safe literal parse over
            # the balanced-braces slice.
            depth, end = 0, None
            for i, ch in enumerate(tail):
                if ch == "{": depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end is None:
                raise
            sg = ast.literal_eval(tail[:end])
        all_names = [n["name"] for n in sg["objects"] + sg["regions"]]
        seen, duplicates = set(), set()
        for name in all_names:
            if name in seen:
                duplicates.add(name)
            seen.add(name)
        if duplicates:
            print(f"WARNING!!! Duplicate node labels found: {sorted(duplicates)}")
        example["scene_graph_dict"] = sg
        return example
    
    ds = ds.map(lambda e: {"messages": e["conversations"]})

    is_graph_arch = architecture in ("rpearl_llm", "rpearl_gt_llm", "gt_llm", "composite_graph_gt")
    if is_graph_arch:
        # Parse the GNN's scene graph from the ORIGINAL messages first, so the
        # compact translation below (which only rewrites the LLM-facing text)
        # leaves the graph the GNN ingests untouched.
        ds = ds.map(_parse_scene_graph)
        # Translate the verbose SPINE text to the compact format the LLM is
        # trained/evaluated on. The compact block already carries node names +
        # the bracketed edge lists, so `text_edge_list` stripping is moot here.
        def _translate_to_compact(example):
            example["messages"] = compact_prompt.spine_to_compact_messages(example["conversations"])
            return example
        ds = ds.map(_translate_to_compact)
    else:
        # Plain-LLM baseline: the SAME compact format as the graph archs, but the
        # scene-graph block carries the edge bullets (`include_edges=True`) since
        # there is no GNN to supply connectivity — the LLM must read it from text.
        # `text_edge_list` no longer gates the LLM text: edges are always present
        # in the compact block.
        def _translate_to_compact_with_edges(example):
            example["messages"] = compact_prompt.spine_to_compact_messages(
                example["conversations"], include_edges=True
            )
            return example
        ds = ds.map(_translate_to_compact_with_edges)
    ds = ds.map(_tokenize)
    ds = ds.filter(lambda e: any(m.get("role") == "assistant" for m in e["messages"]))
    return ds


def remove_edge_list(decoded: str) -> str:
    """Remove the edge list (object_connections and region_connections) from
    a decoded prompt string containing a scene graph.

    Handles both single-quoted Python repr (training data) and double-quoted
    multiline JSON (SPINE ``GraphHandler.to_json_str`` with ``indent=2``).

    Parameters
    ----------
    decoded : str
        The full decoded prompt text that contains a scene graph with
        ``object_connections`` and ``region_connections`` entries.

    Returns
    -------
    str
        The prompt with both connection lists removed.
    """
    # Training data: single-quoted, single-line Python repr
    decoded = re.sub(
        r"'object_connections': .+?, 'region_connections': .+?, (?='robot_location'|\})",
        "", decoded,
    )
    # SPINE eval: double-quoted, multiline JSON (json.dumps with indent=2).
    # Keys are separated by ,\n<indent> rather than ", " so we use ,\s* between them.
    # Trailing comma is optional (absent when region_connections is the last key).
    decoded = re.sub(
        r'"object_connections":\s*.+?,\s*"region_connections":\s*.+?,?\s*(?="robot_location"|\})',
        "", decoded, flags=re.DOTALL,
    )
    return decoded


class SpineDataCollator(DataCollatorForLanguageModeling):
    """SPINE scene-graph collator.

    Expects ``scene_graph_dict`` to already be parsed in each example
    (by ``preprocess_dataset``).  Converts to PyG graphs, computes
    injection maps, and batches them alongside the padded token tensors.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _extract_graph(self, example):
        """Build PyG graph and injection map from a preprocessed example."""
        pyg_graph = utils.scene_graph_dict_to_pyg(example["scene_graph_dict"])
        # Standalone + space-preceded tokenizations per node so every list / edge
        # mention binds (100% injection); see node_token_variants.
        node_token_seqs = node_token_variants(pyg_graph.node_names, self.tokenizer)
        # Scope injection to the last (query) graph block, matching eval
        # (GraphAugmentedInMemoryLLM) and R10. Without this, training cross-links
        # the query graph's labels to their mentions across the *whole* prompt —
        # including the ICL examples — so the composite-graph structure the model
        # learns diverges from the scoped structure it sees at inference (and the
        # gap widens with more ICL examples).
        scope_start = find_last_graph_scope(example["input_ids"], self.tokenizer)
        injection_map = build_injection_map(
            example["input_ids"], node_token_seqs, scope_start=scope_start
        )
        return pyg_graph, injection_map

    _NON_TENSOR_KEYS = {"conversations", "scene_graph", "scene_graph_dict", "messages", "text", "full_text"}

    def __call__(self, features, return_tensors: Optional[str] = None):
        """Attach parsed PyG graphs and injection maps for each example."""
        pyg_graphs = []
        injection_maps = []
        sanitized_examples = []

        for example in features:
            pyg_graph, injection_map = self._extract_graph(example)
            pyg_graphs.append(pyg_graph)
            injection_maps.append(injection_map)

            sanitized_examples.append(
                {k: v for k, v in example.items() if k not in self._NON_TENSOR_KEYS}
            )

        batch = super().__call__(sanitized_examples)
        batch["graphs"] = Batch.from_data_list(pyg_graphs)
        batch["injection_maps"] = injection_maps
        return batch
