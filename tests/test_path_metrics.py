"""Tests for the eval path metrics added on top of ``path_validator`` (R4 / M10):

  * ``eval/valid_path_rate``      — A→B route with no hallucinated nodes/edges,
  * ``eval/path_optimality_rate`` — emitted hops ÷ shortest-path hops (valid paths),
  * ``eval/hallucination_rate``   — parsed plan nodes absent from the graph.

The first two reuse ``validate_path`` (regex + NetworkX); optimality is the new
hop-count analogue of the distance-weighted ``cost_optimality``. ``valid_path_ab`` /
``path_expected`` / ``hallucination_rate`` and the hop-null are added at the
``evaluate_sample`` chokepoint so they see the post-override ``start_goal_ok`` (a
reach-an-object route legitimately ends one hop past the goal region).

Also guards that the legacy ``grep/path_*`` aggregates stay byte-identical and that
the new ``eval/*`` aggregates use the wider denominator (failures count).

All deterministic, CPU, no model — node names follow the domain ``name_N`` form so
``path_validator``'s ``_NODE_ID`` regex matches them.
"""
import sys

sys.path.insert(0, "src")

from pytest import approx

from prism.eval import path_validator as pv


# A small connected graph (undirected). Two equally-short 2-hop routes from
# start_1 to goal_1 (via mid_1 or via alt_1) — so "many paths count as a solution".
#   start_1 - mid_1 - goal_1
#   start_1 - alt_1 - goal_1
#   drill_1 is an object contained in goal_1 (object_connection).
GRAPH = {
    "regions": [
        {"name": "start_1", "coords": [0, 0]},
        {"name": "mid_1", "coords": [1, 0]},
        {"name": "goal_1", "coords": [2, 0]},
        {"name": "alt_1", "coords": [1, 1]},
    ],
    "objects": [{"name": "drill_1", "coords": [2, 0]}],
    "region_connections": [
        ["start_1", "mid_1"], ["mid_1", "goal_1"],
        ["start_1", "alt_1"], ["alt_1", "goal_1"],
    ],
    "object_connections": [["drill_1", "goal_1"]],
}


# --------------------------------------------------------------------------
# validate_path: hop_optimality (the new hop-count metric) + node/edge validity
# --------------------------------------------------------------------------
def test_hop_optimality_is_one_for_a_shortest_route():
    m = pv.validate_path("start_1 -> mid_1 -> goal_1", GRAPH, start="start_1", goal="goal_1")
    assert m["full_path_valid"] is True
    assert m["start_goal_ok"] is True
    assert m["hop_optimality"] == approx(1.0)  # 2 emitted hops / 2 shortest hops


def test_hop_optimality_penalises_a_longer_valid_route():
    # A valid but wandering walk start_1 → goal_1: 4 hops vs shortest 2 ⇒ 2.0.
    m = pv.validate_path(
        "start_1 -> mid_1 -> goal_1 -> alt_1 -> goal_1", GRAPH, start="start_1", goal="goal_1")
    assert m["full_path_valid"] is True
    assert m["hop_optimality"] == approx(2.0)


def test_hallucinated_node_breaks_validity_and_shows_in_exist_rate():
    m = pv.validate_path("start_1 -> ghost_9 -> goal_1", GRAPH, start="start_1", goal="goal_1")
    assert m["full_path_valid"] is False
    assert m["nodes_exist_rate"] == approx(2 / 3)  # ghost_9 absent
    assert m["hop_optimality"] is None


def test_hallucinated_edge_breaks_validity_even_when_nodes_exist():
    # start_1 and goal_1 both exist but are NOT adjacent.
    m = pv.validate_path("start_1 -> goal_1", GRAPH, start="start_1", goal="goal_1")
    assert m["nodes_exist_rate"] == approx(1.0)
    assert m["edge_validity_rate"] == approx(0.0)
    assert m["full_path_valid"] is False
    assert m["hop_optimality"] is None


# --------------------------------------------------------------------------
# _augment_eval_metrics: derived fields read the FINAL start_goal_ok
# --------------------------------------------------------------------------
def _base_metrics(**over):
    m = {
        "parsed_nodes": ["start_1", "mid_1", "goal_1"], "num_parsed": 3,
        "nodes_exist_rate": 1.0, "edge_validity_rate": 1.0,
        "full_path_valid": True, "start_goal_ok": True,
        "cost_optimality": 1.0, "hop_optimality": 1.0,
    }
    m.update(over)
    return m


def test_augment_valid_path_for_navigation_goal():
    m = pv._augment_eval_metrics(_base_metrics(), goal="goal_1")
    assert m["path_expected"] is True
    assert m["valid_path_ab"] is True
    assert m["hallucination_rate"] == approx(0.0)
    assert m["hop_optimality"] == approx(1.0)


def test_augment_trusts_overridden_start_goal_ok_reach_object():
    # Reach-an-object route: parsed ends at the object, goal is the region, but
    # validate_structured already set start_goal_ok=True. _augment must respect it.
    m = _base_metrics(parsed_nodes=["start_1", "mid_1", "goal_1", "drill_1"], num_parsed=4,
                      start_goal_ok=True, kind="path")
    pv._augment_eval_metrics(m, goal="goal_1")
    assert m["valid_path_ab"] is True
    assert m["hop_optimality"] == approx(1.0)  # not nulled


def test_augment_nulls_optimality_when_endpoints_wrong():
    m = pv._augment_eval_metrics(_base_metrics(start_goal_ok=False), goal="goal_1")
    assert m["valid_path_ab"] is False
    assert m["hop_optimality"] is None  # meaningless for a route that misses the goal


def test_augment_excludes_positionality_edges_tasks():
    m = pv._augment_eval_metrics(_base_metrics(kind="edges"), goal="goal_1")
    assert m["path_expected"] is False   # positionality: not a route task
    assert m["valid_path_ab"] is False


def test_augment_hallucination_rate_counts_invalid_edges():
    # Edge hallucination = 1 - edge_validity_rate (invalid hops / total hops),
    # independent of nodes_exist_rate.
    m = pv._augment_eval_metrics(_base_metrics(edge_validity_rate=0.5), goal="goal_1")
    assert m["hallucination_rate"] == approx(0.5)


def test_augment_no_route_has_none_hallucination():
    m = pv._augment_eval_metrics(_base_metrics(num_parsed=0, full_path_valid=False), goal="goal_1")
    assert m["hallucination_rate"] is None
    assert m["valid_path_ab"] is False


def test_augment_single_node_route_has_none_hallucination():
    # One node ⇒ no hop ⇒ edge hallucination is undefined.
    m = pv._augment_eval_metrics(
        _base_metrics(num_parsed=1, edge_validity_rate=0.0), goal="goal_1")
    assert m["hallucination_rate"] is None


# --------------------------------------------------------------------------
# evaluate_sample end-to-end (structured ⇒ deterministic, no Gemma judge)
# --------------------------------------------------------------------------
def test_evaluate_sample_navigability_is_valid_and_optimal():
    m = pv.evaluate_sample(
        "Navigate from start_1 to goal_1.", "start_1 -> mid_1 -> goal_1", GRAPH,
        init_node="start_1", answer="goal_1")
    assert m["structured"] is True
    assert m["path_expected"] is True
    assert m["valid_path_ab"] is True
    assert m["hop_optimality"] == approx(1.0)
    assert m["hallucination_rate"] == approx(0.0)


def test_evaluate_sample_reach_object_route_counts_as_valid():
    # Route ends one hop past the goal region at the contained object — the
    # reach-an-object override must keep it valid end-to-end through evaluate_sample.
    m = pv.evaluate_sample(
        "Navigate from start_1 to goal_1.", "start_1 -> mid_1 -> goal_1 -> drill_1", GRAPH,
        init_node="start_1", answer="goal_1")
    assert m["valid_path_ab"] is True
    assert m["hop_optimality"] is not None


# --------------------------------------------------------------------------
# aggregate_path_metrics: denominators + legacy byte-identity
# --------------------------------------------------------------------------
def test_aggregate_new_metric_denominators():
    samples = [
        {"path_metrics": {"num_parsed": 2, "path_expected": True, "valid_path_ab": True,
                          "hop_optimality": 1.0, "hallucination_rate": 0.0,
                          "nodes_exist_rate": 1.0, "edge_validity_rate": 1.0,
                          "full_path_valid": True, "start_goal_ok": True}},
        {"path_metrics": {"num_parsed": 2, "path_expected": True, "valid_path_ab": False,
                          "hop_optimality": None, "hallucination_rate": 0.5,
                          "nodes_exist_rate": 0.5, "edge_validity_rate": 0.5,
                          "full_path_valid": False, "start_goal_ok": False}},
        # path expected but NO route emitted ⇒ counts as a valid_path failure.
        {"path_metrics": {"num_parsed": 0, "path_expected": True, "valid_path_ab": False,
                          "hop_optimality": None, "hallucination_rate": None}},
        # positionality task: not path_expected; single node ⇒ no hop ⇒ no edge
        # hallucination (None), so it does not contribute to the hallucination mean.
        {"path_metrics": {"num_parsed": 1, "path_expected": False, "valid_path_ab": False,
                          "hop_optimality": None, "hallucination_rate": None,
                          "nodes_exist_rate": 1.0, "edge_validity_rate": 0.0,
                          "full_path_valid": False, "start_goal_ok": False}},
    ]
    agg = pv.aggregate_path_metrics(samples)
    assert agg["num_path_expected"] == 3
    assert agg["valid_path_rate"] == approx(1 / 3)          # only sample 1 valid
    assert agg["path_optimality_rate"] == approx(1.0)        # only sample 1 has a hop ratio
    assert agg["hallucination_rate"] == approx((0.0 + 0.5) / 2)  # edge-routed samples 1,2


def test_aggregate_legacy_keys_unchanged():
    # Legacy grep/path_* keys average only over routed samples (num_parsed>0) and
    # must keep their historical values regardless of the new eval/* additions.
    samples = [
        {"path_metrics": {"num_parsed": 2, "edge_validity_rate": 1.0, "nodes_exist_rate": 1.0,
                          "full_path_valid": True, "start_goal_ok": True, "cost_optimality": 1.0,
                          "path_from_reasoning": False, "path_rescued": False}},
        {"path_metrics": {"num_parsed": 3, "edge_validity_rate": 0.5, "nodes_exist_rate": 1.0,
                          "full_path_valid": False, "start_goal_ok": False, "cost_optimality": None,
                          "path_from_reasoning": True, "path_rescued": False}},
        {"path_metrics": {"num_parsed": 0}},  # no route ⇒ excluded from legacy aggregates
    ]
    agg = pv.aggregate_path_metrics(samples)
    assert agg["edge_validity_rate"] == approx(0.75)
    # nodes_exist_rate / full_path_valid_rate / start_goal_ok_rate were dropped from
    # the reported aggregate (redundant with hallucination_rate / valid_path_rate);
    # the per-sample primitives still feed those derived metrics.
    assert "nodes_exist_rate" not in agg
    assert "full_path_valid_rate" not in agg
    assert "start_goal_ok_rate" not in agg
    assert agg["cost_optimality"] == approx(1.0)   # mean over non-None only
    assert agg["num_with_path"] == 2
    assert agg["num_from_reasoning"] == 1
    assert agg["num_rescued"] == 0


def test_aggregate_empty_is_empty_dict():
    assert pv.aggregate_path_metrics([]) == {}
