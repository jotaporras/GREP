"""Utilities for parsing scene graphs from conversation transcripts."""

from __future__ import annotations

import re
import ast
from typing import Any, Dict, List, Mapping

from datasets import Dataset

SCENE_GRAPH_PATTERN = re.compile(r"[Ss]cene graph:\s*(\{.*)", re.DOTALL)
EXPECTED_GRAPH_KEYS = (
    "objects",
    "regions",
    "object_connections",
    "region_connections",
    "robot_location",
)

def _parse_scene_graph_dictionary_from_conversation(
    conversation
) -> Dict[str, Any]:
    """Extract the scene graph dictionary from a single conversation.

    Args:
        conversation: The dictionary containing the conversation.

    Returns:
        Parsed scene-graph dictionary with the expected topology keys.

    Raises:
        ValueError: If no scene graph is present or the payload cannot be parsed.
    """

    if "conversations" not in conversation:
        raise ValueError("Conversation payload is missing the 'conversations' key.")

    turns = conversation["conversations"]
    if not turns:
        raise ValueError("Conversation payload has no turns to inspect.")

    first_turn = turns[0]

    prompt = first_turn["content"]
    match = SCENE_GRAPH_PATTERN.search(prompt)
    if match is None:
        raise ValueError("Scene graph marker not found in first turn content.")

    # The second group contains the graph dict.
    scene_graph_text = match.group(1).strip()

    try:
        graph_dict = ast.literal_eval(scene_graph_text)
    except (SyntaxError, ValueError) as exc:
        raise ValueError("Scene graph literal could not be parsed.") from exc

    missing_keys = [key for key in EXPECTED_GRAPH_KEYS if key not in graph_dict]
    if missing_keys:
        raise ValueError(
            f"Scene graph is missing required keys: {', '.join(missing_keys)}."
        )

    return graph_dict


def find_undefined_nodes(scene_graph_dict: Mapping[str, Any]) -> List[str]:
    """Return nodes referenced in edges that are not declared in the graph."""

    defined_nodes = {obj["name"] for obj in scene_graph_dict["objects"]}
    defined_nodes.update(region["name"] for region in scene_graph_dict["regions"])

    undefined_nodes: List[str] = []

    for source, target in scene_graph_dict["object_connections"]:
        if source not in defined_nodes and source not in undefined_nodes:
            undefined_nodes.append(source)
        if target not in defined_nodes and target not in undefined_nodes:
            undefined_nodes.append(target)

    for source, target in scene_graph_dict["region_connections"]:
        if source not in defined_nodes and source not in undefined_nodes:
            undefined_nodes.append(source)
        if target not in defined_nodes and target not in undefined_nodes:
            undefined_nodes.append(target)

    return undefined_nodes


def add_scene_graph_feature(
    dataset: Dataset,
) -> Dataset:
    """Attach a scene graph feature to every example in the dataset and filter it.

    Args:
        dataset: HuggingFace ``Dataset`` containing conversation transcripts.
        feature_name: Name of the feature that will store the parsed graph.

    Returns:
        Dataset with a new feature populated by parsed scene graphs and filtered so
        that only examples with fully defined graphs remain.
    """

    def _add_feature(example: Dict[str, Any]) -> Dict[str, Any]:
        example["scene_graph"] = _parse_scene_graph_dictionary_from_conversation(example)
        return example

    def _has_defined_nodes(example: Mapping[str, Any]) -> bool:
        """Checks if the scene graph has no undefined nodes (sometimes LLMs yield incomplete node lists.)"""
        undefined_nodes = find_undefined_nodes(example["scene_graph"])
        return len(undefined_nodes) == 0

    processed_dataset = dataset.map(
        _add_feature,
        desc="Parsing scene graphs",
    ).filter(
        _has_defined_nodes,
        desc="Filtering undefined nodes in scene_graph",
    )
    return processed_dataset

__all__ = [
    "add_scene_graph_feature",
    "find_undefined_nodes",
    "_parse_scene_graph_dictionary_from_conversation",
]


