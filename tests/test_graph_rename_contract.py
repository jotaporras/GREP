"""Tests for the rename-map populate contract.

Guards the defect these tests were written for: the LLM used to return the whole
populated graph and drifted on the rename over large graphs, emitting
`region_connections` that referenced names absent from the node lists. The old
check ran through SPINE's GraphHandler, which swallowed the KeyError and left a
partially loaded graph, so the corruption reached disk and surfaced downstream
as empty scene graphs and 0-node PyG crashes.

No model weights are loaded: every test drives the parsing/validation path with
a canned response string.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prism.data import graph_gen

BASE_GRAPH = {
    "graph": {
        "objects": [
            {"name": "object_1", "coords": [78.0, 9.1], "description": "__FILL__"},
            {"name": "object_2", "coords": [52.4, -56.2], "description": ""},
        ],
        "regions": [
            {"name": "region_1", "coords": [0, 0], "description": ""},
            {"name": "region_2", "coords": [5.7, -8.3], "description": ""},
            {"name": "region_3", "coords": [19.3, -6.5], "description": ""},
        ],
        "object_connections": [["object_1", "region_3"], ["object_2", "region_2"]],
        "region_connections": [["region_1", "region_2"], ["region_2", "region_3"]],
        "robot_location": "region_1",
    },
    "tasks": [],
    "_metadata": {"n_communities": 1, "n_tasks": 1},
}

RENAME = {
    "object_1": "fuel_tank_1",
    "object_2": "gate_1",
    "region_1": "clearing_1",
    "region_2": "comm_bunker_1",
    "region_3": "fuel_depot_1",
}

DESCRIPTIONS = {"object_1": "rusted"}

TASK = {
    "task": "Reach the area holding the fuel tank and report the edges and route.",
    "answer": r"(?i)\bclearing_1\b.*\bfuel_depot_1\b",
    "init_node": "clearing_1",
    "acceptance_criterion": (
        "A correct answer reaches fuel_depot_1 via comm_bunker_1, stating the edge "
        "fuel_depot_1 <=> fuel_tank_1."
    ),
}


def response(rename=None, descriptions=None, tasks=None) -> str:
    return json.dumps(
        {
            "rename": RENAME if rename is None else rename,
            "descriptions": DESCRIPTIONS if descriptions is None else descriptions,
            "tasks": [TASK] if tasks is None else tasks,
        }
    )


@pytest.fixture
def gen():
    return graph_gen.TaskGraphGen(client=object())


class TestRenameMapValidation:
    def test_accepts_a_well_formed_map(self):
        graph_gen.validate_rename_map(RENAME, BASE_GRAPH)

    def test_rejects_a_missing_node(self):
        partial = {k: v for k, v in RENAME.items() if k != "region_2"}
        with pytest.raises(ValueError, match="does not cover the base nodes"):
            graph_gen.validate_rename_map(partial, BASE_GRAPH)

    def test_rejects_an_invented_node(self):
        with pytest.raises(ValueError, match="does not cover the base nodes"):
            graph_gen.validate_rename_map({**RENAME, "region_9": "survey_3"}, BASE_GRAPH)

    def test_rejects_a_name_collision(self):
        collided = {**RENAME, "region_3": "comm_bunker_1"}
        with pytest.raises(ValueError, match="not injective"):
            graph_gen.validate_rename_map(collided, BASE_GRAPH)

    def test_rejects_an_empty_name(self):
        with pytest.raises(ValueError, match="empty/non-string"):
            graph_gen.validate_rename_map({**RENAME, "region_3": "  "}, BASE_GRAPH)

    def test_rejects_a_leftover_placeholder(self):
        with pytest.raises(ValueError, match="placeholder"):
            graph_gen.validate_rename_map({**RENAME, "region_3": "region_3"}, BASE_GRAPH)


class TestDescriptionValidation:
    def test_rejects_a_description_for_a_non_fill_node(self):
        with pytest.raises(ValueError, match="__FILL__ nodes"):
            graph_gen.validate_descriptions(
                {**DESCRIPTIONS, "object_2": "locked"}, BASE_GRAPH
            )

    def test_rejects_an_unfilled_placeholder(self):
        with pytest.raises(ValueError, match="__FILL__ nodes"):
            graph_gen.validate_descriptions({}, BASE_GRAPH)


class TestApplyRename:
    def test_topology_and_coords_come_from_the_base_graph(self):
        graph = graph_gen.apply_rename(BASE_GRAPH, RENAME, DESCRIPTIONS)

        assert graph["region_connections"] == [
            ["clearing_1", "comm_bunker_1"],
            ["comm_bunker_1", "fuel_depot_1"],
        ]
        assert graph["object_connections"] == [
            ["fuel_tank_1", "fuel_depot_1"],
            ["gate_1", "comm_bunker_1"],
        ]
        assert graph["robot_location"] == "clearing_1"
        assert [r["coords"] for r in graph["regions"]] == [
            r["coords"] for r in BASE_GRAPH["graph"]["regions"]
        ]
        assert len(graph["objects"]) == 2 and len(graph["regions"]) == 3

    def test_only_fill_descriptions_are_written(self):
        graph = graph_gen.apply_rename(BASE_GRAPH, RENAME, DESCRIPTIONS)

        by_name = {o["name"]: o for o in graph["objects"]}
        assert by_name["fuel_tank_1"]["description"] == "rusted"
        assert by_name["gate_1"]["description"] == ""
        assert all(r["description"] == "" for r in graph["regions"])


class TestGraphRefBackstop:
    def test_rejects_an_edge_with_three_endpoints(self):
        graph = graph_gen.apply_rename(BASE_GRAPH, RENAME, DESCRIPTIONS)
        graph["region_connections"].append(["clearing_1", "comm_bunker_1", "extra_1"])

        with pytest.raises(AssertionError, match="endpoints"):
            graph_gen.assert_graph_refs_resolve(graph)

    def test_rejects_an_edge_to_a_nonexistent_node(self):
        graph = graph_gen.apply_rename(BASE_GRAPH, RENAME, DESCRIPTIONS)
        graph["region_connections"].append(["clearing_1", "gear_shed_1"])

        with pytest.raises(AssertionError, match="unknown node"):
            graph_gen.assert_graph_refs_resolve(graph)


class TestParseResponse:
    def test_builds_the_downstream_schema(self, gen):
        out = gen.parse_response(response(), json.dumps(BASE_GRAPH), n_tasks=1)

        assert set(out) == {"graph", "tasks", "description"}
        assert set(out["graph"]) == {
            "objects",
            "regions",
            "object_connections",
            "region_connections",
            "robot_location",
        }
        assert out["tasks"] == [TASK]

    def test_caps_the_task_count(self, gen):
        out = gen.parse_response(
            response(tasks=[TASK, TASK, TASK]), json.dumps(BASE_GRAPH), n_tasks=2
        )
        assert len(out["tasks"]) == 2

    def test_a_corrupt_rename_map_is_rejected(self, gen):
        # The failure the old contract produced as a dangling edge: the model
        # emits a name for a node that is not in the skeleton.
        corrupt = {**RENAME}
        corrupt.pop("region_3")
        corrupt["region_33"] = "gear_shed_1"

        with pytest.raises(ValueError, match="does not cover the base nodes"):
            gen.parse_response(response(rename=corrupt), json.dumps(BASE_GRAPH))

    def test_a_task_naming_a_nonexistent_node_is_rejected(self, gen):
        stale = {**TASK, "acceptance_criterion": "A correct answer reaches survey_3."}

        with pytest.raises(ValueError, match="not in the graph"):
            gen.parse_response(response(tasks=[stale]), json.dumps(BASE_GRAPH))

    def test_a_task_starting_outside_the_graph_is_rejected(self, gen):
        stale = {**TASK, "init_node": "calibration_3"}

        with pytest.raises(ValueError, match="init_node"):
            gen.parse_response(response(tasks=[stale]), json.dumps(BASE_GRAPH))

    def test_an_object_as_init_node_is_rejected(self, gen):
        stale = {**TASK, "init_node": "fuel_tank_1"}

        with pytest.raises(ValueError, match="not a region"):
            gen.parse_response(response(tasks=[stale]), json.dumps(BASE_GRAPH))
