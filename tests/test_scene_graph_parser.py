from __future__ import annotations

import pytest  # type: ignore[import-not-found]
from datasets import Dataset

#sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from prism.scene_graph_parser import (  # noqa: E402  pylint: disable=wrong-import-position
    _parse_scene_graph_dictionary_from_conversation,
    add_scene_graph_feature,
    find_undefined_nodes,
)


EXAMPLE_CONVERSATION = {
    "conversations": [
        {
            "content": "Scene graph: {'objects': [{'name': 'house_1', 'coords': [-1, -1]}, {'name': 'shed_1', 'coords': [1, 3]}], 'regions': [{'name': 'road_1', 'coords': [0, 0]}], 'object_connections': [['house_1', 'road_1']], 'region_connections': [['road_1', 'road_1']], 'robot_location': 'road_1'}"
        }
    ]
}


CONVERSATION_WITH_UNDEFINED_NODE = {
    "conversations": [
        {
            "content": "Scene graph: {'objects': [{'name': 'house_1', 'coords': [0, 0]}], 'regions': [{'name': 'road_1', 'coords': [0, 1]}], 'object_connections': [['ghost_house', 'road_1']], 'region_connections': [['road_1', 'phantom_field']], 'robot_location': 'ghost_house'}"
        }
    ]
}


def test_scene_graph_contains_expected_keys() -> None:
    graph = _parse_scene_graph_dictionary_from_conversation(EXAMPLE_CONVERSATION)

    assert set(graph.keys()) == {
        "objects",
        "regions",
        "object_connections",
        "region_connections",
        "robot_location",
    }


def test_dataset_mapping_adds_graph_feature() -> None:
    dataset = Dataset.from_list([EXAMPLE_CONVERSATION, EXAMPLE_CONVERSATION])

    mapped_dataset = add_scene_graph_feature(dataset)

    assert "scene_graph" in mapped_dataset.column_names
    assert mapped_dataset.num_rows == 2
    assert mapped_dataset[0]["scene_graph"]["robot_location"] == "road_1"


def test_add_scene_graph_feature_raises_on_missing_marker() -> None:
    dataset = Dataset.from_list(
        [
            EXAMPLE_CONVERSATION,
            {"conversations": [{"content": "No graph here."}]},
        ]
    )

    with pytest.raises(ValueError):
        add_scene_graph_feature(dataset)


def test_find_undefined_nodes_returns_empty_list_when_all_nodes_defined() -> None:
    graph = _parse_scene_graph_dictionary_from_conversation(EXAMPLE_CONVERSATION)

    assert find_undefined_nodes(graph) == []


def test_find_undefined_nodes_returns_missing_node_names() -> None:
    graph = _parse_scene_graph_dictionary_from_conversation(CONVERSATION_WITH_UNDEFINED_NODE)

    assert find_undefined_nodes(graph) == ["ghost_house", "phantom_field"]


def test_add_scene_graph_feature_filters_undefined_nodes() -> None:
    dataset = Dataset.from_list([EXAMPLE_CONVERSATION, CONVERSATION_WITH_UNDEFINED_NODE])

    filtered_dataset = add_scene_graph_feature(dataset)

    assert filtered_dataset.num_rows == 1
    assert filtered_dataset[0]["scene_graph"]["robot_location"] == "road_1"


