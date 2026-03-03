"""Tests for GraphSim.take_action and SPINE plan parsing."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from spine.mapping.graph_util import GraphHandler
from spine.spine import SPINE, ValidPlanFeedback
from prism.data.graph_sim import GraphSim


# ---------------------------------------------------------------------------
# Fixture graph
# ---------------------------------------------------------------------------

FIXTURE_GRAPH = {
    "objects": [
        {"name": "sail_boat_1", "coords": [1, 0], "description": "A weathered sail boat"},
    ],
    "regions": [
        {"name": "field_1", "coords": [0, 0], "description": "A grassy field"},
        {"name": "field_2", "coords": [5, 0], "description": "A rocky field"},
        {"name": "field_3", "coords": [10, 0], "description": "A wooded field"},
    ],
    "object_connections": [["sail_boat_1", "field_1"]],
    "region_connections": [["field_1", "field_2"], ["field_2", "field_3"]],
}

INIT_NODE = "field_1"


def make_graph_handler() -> GraphHandler:
    gh = GraphHandler("")
    gh.reset(graph_as_dict=FIXTURE_GRAPH, current_location=INIT_NODE)
    return gh


def make_graph_sim() -> GraphSim:
    gh = make_graph_handler()
    sim = GraphSim(gh)
    return sim


# ---------------------------------------------------------------------------
# Dummy LLM client (never actually queried in parsing tests)
# ---------------------------------------------------------------------------

class _DummyClient:
    def query_llm(self, msg):
        return "", False


def make_spine() -> SPINE:
    gh = make_graph_handler()
    return SPINE(graph=gh, client=_DummyClient())


# ---------------------------------------------------------------------------
# A. SPINE parsing tests
# ---------------------------------------------------------------------------

class TestSpineParsing:
    def test_parse_explore_region(self):
        spine = make_spine()
        (function, arg), success = spine._try_parse_command("explore_region(field_1, 3)")
        assert success
        assert function == "explore_region"
        ok, parsed = spine.try_parse_exploration_arg(arg)
        assert ok
        assert parsed == ("field_1", 3.0)

    def test_parse_map_region(self):
        spine = make_spine()
        (function, arg), success = spine._try_parse_command("map_region(field_1)")
        assert success
        assert function == "map_region"
        assert arg == "field_1"

    def test_parse_inspect(self):
        spine = make_spine()
        (function, arg), success = spine._try_parse_command("inspect(sail_boat_1, is it damaged?)")
        assert success
        assert function == "inspect"
        ok, parsed = spine.try_parse_inspection_arg(arg)
        assert ok
        assert parsed == ("sail_boat_1", "is it damaged?")

    def test_parse_goto(self):
        spine = make_spine()
        (function, arg), success = spine._try_parse_command("goto(field_2)")
        assert success
        assert function == "goto"
        assert arg == "field_2"

    def test_parse_answer(self):
        spine = make_spine()
        (function, arg), success = spine._try_parse_command("answer(yes there is a boat)")
        assert success
        assert function == "answer"
        assert arg == "yes there is a boat"

    def test_parse_invalid_action(self):
        spine = make_spine()
        plan, feedback = spine.extract_plan(["fly(field_1)"])
        assert feedback.success is False


# ---------------------------------------------------------------------------
# B. GraphSim.take_action tests
# ---------------------------------------------------------------------------

class TestGraphSimTakeAction:
    def test_explore_region_tuple_arg(self):
        sim = make_graph_sim()
        result = sim.take_action("explore_region", ("field_1", 3.0))
        assert isinstance(result, bool)

    def test_map_region_string_arg(self):
        sim = make_graph_sim()
        result = sim.take_action("map_region", "field_1")
        assert isinstance(result, bool)

    def test_inspect_tuple_arg(self):
        sim = make_graph_sim()
        # sail_boat_1 is in partial graph (all nodes present after reset)
        sim.take_action("inspect", ("sail_boat_1", "is it damaged?"))
        assert "description" in sim.partial_graph.graph.nodes["sail_boat_1"]

    def test_goto_updates_location(self):
        sim = make_graph_sim()
        sim.take_action("goto", "field_2")
        assert sim.partial_graph.current_location == "field_2"

    def test_explore_reveals_description(self):
        sim = make_graph_sim()
        sim.take_action("explore_region", ("field_1", 3.0))
        node_data = sim.partial_graph.graph.nodes["field_1"]
        assert "description" in node_data
        assert node_data["description"] == sim.graph.graph.nodes["field_1"]["description"]

    def test_explore_records_updator(self):
        sim = make_graph_sim()
        sim.take_action("explore_region", ("field_1", 3.0))
        updates = sim.updator.form_updates()
        assert "field_1" in updates

    def test_explore_discovers_missing_neighbor(self):
        sim = make_graph_sim()
        # Manually remove field_2 from partial graph to simulate unexplored area
        sim.partial_graph.graph.remove_node("field_2")
        sim.removed_nodes.append("field_2")

        sim.take_action("explore_region", ("field_1", 3.0))

        assert "field_2" in sim.partial_graph.graph.nodes
