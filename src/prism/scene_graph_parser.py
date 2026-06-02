"""Utilities for parsing scene graphs from conversation transcripts."""

from __future__ import annotations

import re
import ast
import math
from statistics import median
from typing import Any, Dict, List, Mapping

import networkx as nx
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


def build_scene_affinity_graph(
    scene_graph_dict: Mapping[str, Any],
    sigma_mode: str = "median",
    keep_raw_distance_feature: bool = True,
) -> nx.Graph:
    """Build the weighted scene graph G_Sc for the composite-graph pipeline.

    Topology comes from the same object/region connection lists the rest of the
    parser uses; node labels stay the canonical underscore-joined names. Every
    edge gets a Gaussian heat-kernel affinity (E1)

        weight = exp(-d^2 / (2 * sigma^2)),   sigma = median edge distance,

    with the raw Euclidean distance kept under ``distance_m``. The median
    bandwidth makes affinities comparable across graph scales and keeps closer
    edges strictly stronger. Degenerate graphs (no edges, or sigma == 0) fall
    back to weight 1.

    Args:
        scene_graph_dict: Parsed scene graph with objects/regions (each with a
            ``coords`` list) and object/region connection lists.
        sigma_mode: Bandwidth rule for the kernel. Only ``"median"`` is used.
        keep_raw_distance_feature: Keep the raw meters under ``distance_m``.

    Returns:
        Undirected ``nx.Graph`` (directedness matches the source connections).
        ``distance_m`` and affinity ``weight`` are stored per edge; node
        ``coords`` are preserved for the validator (M10).
    """

    coords = {
        node["name"]: node["coords"]
        for node in (*scene_graph_dict["objects"], *scene_graph_dict["regions"])
    }

    G = nx.Graph()
    for name, c in coords.items():
        G.add_node(name, coords=c)

    distances: List[float] = []
    for key, edge_type in (("object_connections", "object"), ("region_connections", "region")):
        for source, target in scene_graph_dict[key]:
            if source in coords and target in coords:
                a = coords[source]
                b = coords[target]
                dist = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
                G.add_edge(source, target, type=edge_type, distance_m=dist)
                distances.append(dist)

    if sigma_mode != "median":
        raise ValueError(f"Unsupported sigma_mode {sigma_mode!r}; only 'median' is locked (E1).")
    sigma = median(distances) if distances else 0.0

    for _, _, attrs in G.edges(data=True):
        d = attrs["distance_m"]
        attrs["weight"] = math.exp(-(d ** 2) / (2.0 * sigma ** 2)) if sigma > 0 else 1.0
        if not keep_raw_distance_feature:
            del attrs["distance_m"]

    return G


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
    "build_scene_affinity_graph",
    "find_undefined_nodes",
    "_parse_scene_graph_dictionary_from_conversation",
]


