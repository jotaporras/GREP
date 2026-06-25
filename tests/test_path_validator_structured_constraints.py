"""Gap-filler tests for prism.eval.path_validator's STRUCTURED-task grading — the
constraint-enforcement branches of `validate_structured` that the existing path suites
(test_path_validator, test_path_metrics) derive but never grade end-to-end:

  * waypoint enforcement   — a valid A->B route that SKIPS a required `via` waypoint must
                             score structured_correct=False (waypoints_ok gate).
  * avoid-set enforcement  — a valid A->B route that PASSES THROUGH an avoided node must
                             score structured_correct=False (avoid_ok gate).
  * positionality (edges)  — `kind == "edges"`: correct only when the goal is named AND its
                             containment edge is stated; naming the goal alone is not enough.
  * directed graph branch  — validate_path(directed=True) respects edge orientation.
  * start==goal cost guard — the shortest==0 div-by-zero guard (characterization).

The Gemma judge / path-rescue model is hard-disabled (GREP_JUDGE=0, GREP_PATH_RESCUE=0) so
every assertion below is the deterministic regex+NetworkX layer only — no model is loaded.
Oracles are hand-computed from small graphs, independent of the implementation.

Run: conda run -n GREP-PRISM python tests/test_path_validator_structured_constraints.py
"""
import math
import os
import sys

sys.path.insert(0, "src")

# Disable the model-backed judge AND path rescue so structured grading is purely
# regex+NetworkX (deterministic, no weights). path_validator reads these per-call.
os.environ["GREP_JUDGE"] = "0"
os.environ["GREP_PATH_RESCUE"] = "0"

from prism.eval import path_validator as P


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------

def _diamond_graph():
    """Two disjoint 2-hop routes start_1 -> goal_1, with NO direct start_1-goal_1 edge:
          start_1 - mid_1 - goal_1     (the "good" route, through mid_1)
          start_1 - bad_1 - goal_1     (the "bad" route, through bad_1)
    so a constraint can force one route over the other."""
    return {
        "regions": [
            {"name": "start_1"}, {"name": "mid_1"},
            {"name": "goal_1"}, {"name": "bad_1"},
        ],
        "objects": [],
        "region_connections": [
            ["start_1", "mid_1"], ["mid_1", "goal_1"],
            ["start_1", "bad_1"], ["bad_1", "goal_1"],
        ],
        "object_connections": [],
    }


def _containment_graph():
    """room_1 -- room_2 (regions); box_1 is an object contained in room_2."""
    return {
        "regions": [{"name": "room_1"}, {"name": "room_2"}],
        "objects": [{"name": "box_1"}],
        "object_connections": [["box_1", "room_2"]],
        "region_connections": [["room_1", "room_2"]],
    }


# ----------------------------------------------------------------------------
# Waypoint enforcement (kind == "path")
# ----------------------------------------------------------------------------

def test_structured_waypoint_satisfied_route_passes():
    """Route through the required waypoint mid_1 -> waypoints_ok, structured_correct."""
    g = _diamond_graph()
    m = P.validate_structured(
        "plan: start_1 -> mid_1 -> goal_1", g,
        init_node="start_1", answer="goal_1",
        criterion="reach goal_1 through mid_1", task="navigate to goal_1",
        full_response="plan: start_1 -> mid_1 -> goal_1")
    assert m is not None and m["kind"] == "path"
    assert m["goal"] == "goal_1"
    assert m["waypoints_ok"] is True
    assert m["full_path_valid"] is True
    assert m["structured_correct"] is True


def test_structured_waypoint_skipped_route_fails_despite_valid_path():
    """A VALID start_1 -> goal_1 route that detours via bad_1 (not the required mid_1)
    must fail: the path is graph-valid and reaches the goal, but waypoints_ok=False
    pulls structured_correct down. This is the branch that makes 'via X' meaningful."""
    g = _diamond_graph()
    m = P.validate_structured(
        "plan: start_1 -> bad_1 -> goal_1", g,
        init_node="start_1", answer="goal_1",
        criterion="reach goal_1 through mid_1", task="navigate to goal_1",
        full_response="plan: start_1 -> bad_1 -> goal_1")
    assert m["full_path_valid"] is True       # graph-valid walk to the goal...
    assert m["start_goal_ok"] is True
    assert m["waypoints_ok"] is False         # ...but it never visits mid_1
    assert m["structured_correct"] is False


# ----------------------------------------------------------------------------
# Avoid-set enforcement (kind == "path")
# ----------------------------------------------------------------------------

def test_structured_avoid_respected_route_passes():
    g = _diamond_graph()
    m = P.validate_structured(
        "plan: start_1 -> mid_1 -> goal_1", g,
        init_node="start_1", answer="goal_1",
        criterion="reach goal_1 without using bad_1", task="navigate to goal_1",
        full_response="plan: start_1 -> mid_1 -> goal_1")
    assert m is not None and m["kind"] == "path"
    assert m["avoid_ok"] is True
    assert m["structured_correct"] is True


def test_structured_avoid_violated_route_fails_despite_valid_path():
    """A valid route that passes through the avoided node bad_1 -> avoid_ok=False ->
    structured_correct=False, even though it is a graph-valid walk to the goal."""
    g = _diamond_graph()
    m = P.validate_structured(
        "plan: start_1 -> bad_1 -> goal_1", g,
        init_node="start_1", answer="goal_1",
        criterion="reach goal_1 without using bad_1", task="navigate to goal_1",
        full_response="plan: start_1 -> bad_1 -> goal_1")
    assert m["full_path_valid"] is True
    assert m["start_goal_ok"] is True
    assert m["avoid_ok"] is False
    assert m["structured_correct"] is False


# ----------------------------------------------------------------------------
# Positionality grading (kind == "edges")
# ----------------------------------------------------------------------------

def test_structured_positionality_requires_stated_containment_edge():
    """Edges-kind: stating the containment edge box_1 <-> room_2 AND naming the goal
    region passes."""
    g = _containment_graph()
    m = P.validate_structured(
        "box_1 <-> room_2", g,
        init_node="room_1", answer="room_2",
        criterion="state that box_1 is located in room_2", task="where is box_1?",
        full_response="box_1 <-> room_2")
    assert m is not None and m["kind"] == "edges"
    assert m["goal"] == "room_2"
    assert m["required_edges_present"] is True
    assert m["structured_correct"] is True


def test_structured_positionality_naming_goal_without_edge_fails():
    """Naming the goal region but NOT stating the containment edge must fail: a
    positionality answer has to assert WHERE the object is, not just echo the region."""
    g = _containment_graph()
    m = P.validate_structured(
        "the object is somewhere in room_2", g,   # names room_2, states no edge
        init_node="room_1", answer="room_2",
        criterion="state that box_1 is located in room_2", task="where is box_1?",
        full_response="the object is somewhere in room_2")
    assert m["kind"] == "edges"
    assert m["required_edges_present"] is False
    assert m["structured_correct"] is False


# ----------------------------------------------------------------------------
# Directed-graph branch of validate_path
# ----------------------------------------------------------------------------

def test_validate_path_directed_respects_edge_orientation():
    """object_connections [[a, b]] in a DiGraph yields a->b only. The forward route is
    valid; the reverse route b->a has no edge (edge_validity 0)."""
    g = {
        "objects": [{"name": "a", "coords": [0, 0]}, {"name": "b", "coords": [1, 0]}],
        "regions": [],
        "object_connections": [["a", "b"]],
    }
    fwd = P.validate_path("a -> b", g, directed=True)
    assert fwd["edge_validity_rate"] == 1.0
    assert fwd["full_path_valid"] is True

    rev = P.validate_path("b -> a", g, directed=True)
    assert rev["nodes_exist_rate"] == 1.0       # both nodes exist...
    assert rev["edge_validity_rate"] == 0.0     # ...but a->b is one-way
    assert rev["full_path_valid"] is False

    # Sanity contrast: undirected treats the reverse as valid.
    assert P.validate_path("b -> a", g, directed=False)["full_path_valid"] is True


# ----------------------------------------------------------------------------
# start == goal: shortest-path-length 0 div-by-zero guard (characterization)
# ----------------------------------------------------------------------------

def test_validate_path_start_equals_goal_loop_uses_optimality_guard():
    """A redundant loop a -> b -> a returns to the start. shortest_path_length(a,a)=0,
    so both the weighted and hop optimality fall into the `else 1.0` div-by-zero guard.
    This pins the guard's value (1.0): note it labels a wasteful non-trivial loop as
    'optimal', which is the documented degenerate-endpoint behavior, not a metric the
    eval relies on for start==goal tasks (none are emitted)."""
    g = {
        "objects": [{"name": "a", "coords": [0, 0]}, {"name": "b", "coords": [1, 0]}],
        "regions": [],
        "object_connections": [["a", "b"]],
    }
    m = P.validate_path("a -> b -> a", g, start="a", goal="a")
    assert m["full_path_valid"] is True
    assert math.isclose(m["edge_validity_rate"], 1.0)
    assert m["cost_optimality"] == 1.0          # else-guard, NOT emitted/shortest
    assert m["hop_optimality"] == 1.0


# ----------------------------------------------------------------------------
# Standalone runner (pytest is absent from the conda env)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    passed, failed = 0, []
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                passed += 1
                print(f"{name}: PASS")
            except Exception as e:  # noqa: BLE001 — report, don't abort the suite
                failed.append((name, f"{type(e).__name__}: {e}"))
                print(f"{name}: FAIL — {type(e).__name__}: {e}")
    print(f"\n{passed} passed, {len(failed)} failed")
    for name, err in failed:
        print(f"  FAIL {name}: {err}")
    sys.exit(1 if failed else 0)
