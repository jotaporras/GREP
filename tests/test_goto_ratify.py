"""e19 SPINE closed loop: goto path ratification + per-turn have_updates.

Invariants locked here (SPINE goto semantics: graph search over the OBSERVED
map, robot follows the route until an edge is missing in the true graph):

- A goto whose believed route exists in the true graph is silent: no
  interruption, the location advances; a multi-hop goto also reports the
  traversed route (grounded pathfinding is the tool's contribution).
- A goto whose believed route uses an edge that does NOT exist in the
  ground-truth graph FAILS at that edge: the robot stops at the last node it
  reached, the bogus edge is retracted from the observed map, corrective
  feedback lands in the updator, and ``have_updates`` interrupts the plan so
  the model gets a replan turn.
- goto to an unknown region, or with no route in the observed map, is rejected
  outright; goto to the current location is a silent no-op.
- ``PRISM_GOTO_RATIFY=0`` restores the legacy (teleporting) behavior.
- The repeat-action nag never fires for goto while ratifying (valid routes ARE
  goto-chains).
- ``have_updates`` is reset per run_planning turn: a rejection in turn 1 must
  not make turn 2 break after its first action (the stickiness bug).
- End-to-end: an episode whose first plan uses a hallucinated edge produces a
  rejection turn, then a corrected plan terminates via ``answer``.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

from spine.spine import SPINE

from prism.data.graph_sim import GraphSim
from prism.data.planning_sim import PlanningSim
from test_sim import FIXTURE_GRAPH, INIT_NODE, make_graph_sim


# field_1 -- field_2 -- field_3 (no direct field_1 -- field_3 edge)


def test_valid_goto_chain_is_silent_and_advances():
    sim = make_graph_sim()
    assert sim.take_action("goto", "field_2") is False
    assert sim.take_action("goto", "field_3") is False
    assert sim.partial_graph.current_location == "field_3"
    assert "rejected" not in sim.get_updator().form_updates()


def test_multihop_goto_walks_true_path_and_reports_traversal():
    sim = make_graph_sim()
    assert sim.take_action("goto", "field_3") is False
    assert sim.partial_graph.current_location == "field_3"
    feedback = sim.get_updator().form_updates()
    assert "traversed field_1 -> field_2 -> field_3" in feedback
    assert "rejected" not in feedback and "failed" not in feedback


def test_goto_over_fake_edge_fails_en_route_and_interrupts():
    import numpy as np
    sim = make_graph_sim()
    sim.corrupt_with_fake_edges(1, np.random.default_rng(0))  # fake f1--f3
    assert sim.take_action("goto", "field_3") is True
    # believed shortest path used the fake direct edge; robot never left f1
    assert sim.partial_graph.current_location == INIT_NODE
    assert not sim.partial_graph.graph.has_edge("field_1", "field_3")
    feedback = sim.get_updator().form_updates()
    assert "failed en route" in feedback
    assert "field_1" in feedback and "field_3" in feedback


def test_goto_with_no_route_in_map_is_rejected():
    sim = make_graph_sim()
    sim.partial_graph.graph.remove_edge("field_1", "field_2")
    assert sim.take_action("goto", "field_3") is True
    assert sim.partial_graph.current_location == INIT_NODE
    assert "no route" in sim.get_updator().form_updates()


def test_unknown_region_is_rejected():
    sim = make_graph_sim()
    assert sim.take_action("goto", "atlantis_9") is True
    assert sim.partial_graph.current_location == INIT_NODE
    assert "rejected" in sim.get_updator().form_updates()


def test_goto_current_location_is_silent_noop():
    sim = make_graph_sim()
    assert sim.take_action("goto", INIT_NODE) is False
    assert sim.partial_graph.current_location == INIT_NODE
    assert "rejected" not in sim.get_updator().form_updates()


def test_kill_switch_restores_legacy_teleport(monkeypatch):
    monkeypatch.setenv("PRISM_GOTO_RATIFY", "0")
    sim = make_graph_sim()
    assert sim.take_action("goto", "field_3") is False
    assert sim.partial_graph.current_location == "field_3"


def test_no_goto_nag_while_ratifying():
    sim = make_graph_sim()
    for target in ("field_2", "field_3", "field_2"):
        sim.take_action("goto", target)
    assert "multiple times" not in sim.get_updator().form_updates()


def test_corrupt_with_fake_edges_touches_only_partial_graph():
    import numpy as np
    sim = make_graph_sim()
    picks = sim.corrupt_with_fake_edges(2, np.random.default_rng(0))
    # only 2-hop non-adjacent region pair in the fixture: field_1 -- field_3
    assert picks == [("field_1", "field_3")]
    assert sim.partial_graph.graph.has_edge("field_1", "field_3")
    assert not sim.graph.graph.has_edge("field_1", "field_3")
    # the prompt string the planner sees includes the fake edge
    assert "field_3" in sim.partial_graph.as_json_str


def test_corrupt_preferred_pairs_used_first_and_validated():
    import numpy as np
    sim = make_graph_sim()
    # invalid preferred pairs (true edge exists / unknown node) are skipped;
    # the valid one is taken without touching the random pool
    picks = sim.corrupt_with_fake_edges(
        1, np.random.default_rng(0),
        preferred_pairs=[("field_1", "field_2"),      # real edge — skip
                         ("field_1", "nowhere"),      # unknown node — skip
                         ("field_3", "field_1")])     # valid — normalized
    assert picks == [("field_1", "field_3")]
    assert sim.partial_graph.graph.has_edge("field_1", "field_3")
    assert not sim.graph.graph.has_edge("field_1", "field_3")


def test_corrupt_all_preferred_invalid_falls_back_to_random_pool():
    import numpy as np
    sim = make_graph_sim()
    picks = sim.corrupt_with_fake_edges(
        1, np.random.default_rng(0),
        preferred_pairs=[("field_1", "field_2")])     # real edge — skip
    assert picks == [("field_1", "field_3")]          # from the random pool


def test_rejected_goto_retracts_fake_edge_from_observed_map():
    import numpy as np
    sim = make_graph_sim()
    sim.corrupt_with_fake_edges(1, np.random.default_rng(0))
    assert sim.take_action("goto", "field_3") is True
    assert not sim.partial_graph.graph.has_edge("field_1", "field_3")
    feedback = sim.get_updator().form_updates()
    assert "remove_connections" in feedback
    assert "failed en route" in feedback


class _ScriptedClient:
    """query_llm stub that plays back canned SPINE-JSON responses in order."""

    def __init__(self, responses):
        self.responses = list(responses)

    def query_llm(self, msg, max_new_tokens=None):
        return self.responses.pop(0), True


def _spine_json(plan):
    return json.dumps({
        "primary_goal": "reach field_3",
        "relevant_graph": ["field_1", "field_2", "field_3"],
        "reasoning": "scripted",
        "plan": plan,
    })


def test_run_planning_rejection_then_recovery():
    import numpy as np
    sim = make_graph_sim()
    # corrupt the observed map with the fake field_1--field_3 shortcut so the
    # first believed route fails en route (an uncorrupted multi-hop goto now
    # simply walks the true path)
    sim.corrupt_with_fake_edges(1, np.random.default_rng(0))
    client = _ScriptedClient([
        # believed shortcut: the fake direct field_1 -> field_3 edge
        _spine_json(["goto(field_3)", "answer(field_1 -> field_3)"]),
        # corrected route after the rejection feedback
        _spine_json(["goto(field_2)", "goto(field_3)",
                     "answer(field_1 -> field_2 -> field_3)"]),
    ])
    planner = SPINE(graph=sim.partial_graph, client=client)
    result = PlanningSim(debug=False).run_planning(
        llm_planner=planner, task="go to field_3", graph_data_gen=sim,
        max_iterations=5)

    assert result.terminated_by == "answer"
    assert len(result.trace) == 2
    # turn 1 broke at the failed goto, before reaching answer
    assert result.trace[0].actions_executed == [("goto", "field_3")]
    assert "failed en route" in result.trace[1].planner_input
    # per-turn have_updates reset: turn 2 executed its FULL corrected plan
    assert [a for a, _ in result.trace[1].actions_executed] == [
        "goto", "goto", "answer"]
    assert sim.partial_graph.current_location == "field_3"
