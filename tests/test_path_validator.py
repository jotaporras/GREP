"""Contract tests for prism.eval.path_validator — the deterministic regex + NetworkX
route-scoring layer that both eval endpoints (evaluate.py, scalability_evaluation.py)
route every sample through.

CS-only verification: this exercises the deterministic plumbing — graph construction,
route parsing, NetworkX scoring arithmetic, target derivation, verdict combination, and
run-level aggregation. The Gemma judge is an external boundary; where a judge-routed code
path is exercised it is stubbed with an inline deterministic fake (the worst-case carve-out:
the model is an opaque callable; we assert only that the harness *handles* its output, never
that the output is accurate).

Oracles are independent of the implementation: graphs/edges/distances are hand-computed,
verdict/aggregate arithmetic is closed-form, and regex behavior is asserted against
constructed inputs — never against a paraphrase of the function body.
"""
import math
import sys

sys.path.insert(0, "src")

from prism.eval import path_validator as P


# ----------------------------------------------------------------------------
# Fixtures — tiny hand-built scene graphs
# ----------------------------------------------------------------------------

def _triangle_graph():
    """Three objects a,b,c with edges a-b, b-c, a-c and coords chosen so the
    Euclidean edge weights are exactly: a-b=1, b-c=sqrt(2), a-c=1."""
    return {
        "objects": [
            {"name": "a", "coords": [0.0, 0.0]},
            {"name": "b", "coords": [0.0, 1.0]},
            {"name": "c", "coords": [1.0, 0.0]},
        ],
        "regions": [],
        "object_connections": [["a", "b"], ["b", "c"], ["a", "c"]],
        "region_connections": [],
    }


# ----------------------------------------------------------------------------
# build_graph
# ----------------------------------------------------------------------------

def test_build_graph_nodes_edges_and_euclidean_weights():
    """Nodes from objects+regions; edges carry distance_m = Euclidean(coords)."""
    G = P.build_graph(_triangle_graph())
    assert set(G.nodes) == {"a", "b", "c"}
    assert set(map(frozenset, G.edges)) == {
        frozenset(("a", "b")), frozenset(("b", "c")), frozenset(("a", "c"))
    }
    assert G["a"]["b"]["distance_m"] == 1.0
    assert math.isclose(G["b"]["c"]["distance_m"], math.sqrt(2))
    assert G["a"]["c"]["distance_m"] == 1.0


def test_build_graph_missing_coords_weight_is_one():
    """When either endpoint lacks coords, the edge weight falls back to 1.0."""
    g = {
        "objects": [{"name": "a", "coords": None}, {"name": "b", "coords": [3.0, 4.0]}],
        "regions": [],
        "object_connections": [["a", "b"]],
    }
    G = P.build_graph(g)
    assert G["a"]["b"]["distance_m"] == 1.0


def test_build_graph_skips_edges_with_unknown_endpoint():
    """An edge naming a node that wasn't declared is dropped, not crashed on."""
    g = {
        "objects": [{"name": "a", "coords": [0, 0]}],
        "regions": [],
        "object_connections": [["a", "ghost"]],
    }
    G = P.build_graph(g)
    assert set(G.nodes) == {"a"}
    assert G.number_of_edges() == 0


def test_build_graph_directed_flag_builds_digraph():
    import networkx as nx
    assert P.build_graph(_triangle_graph(), directed=True).is_directed()
    assert not P.build_graph(_triangle_graph(), directed=False).is_directed()
    assert isinstance(P.build_graph(_triangle_graph(), directed=True), nx.DiGraph)


# ----------------------------------------------------------------------------
# parse_path
# ----------------------------------------------------------------------------

def test_parse_path_arrow_chain():
    assert P.parse_path("route: a -> b -> c done") == ["a", "b", "c"]


def test_parse_path_default_takes_longest_chain():
    """With two arrow chains, the default picks the one with the most hops."""
    text = "first x -> y. then a -> b -> c -> d."
    assert P.parse_path(text) == ["a", "b", "c", "d"]


def test_parse_path_prefer_last_takes_final_chain():
    """prefer_last takes the chain stated LAST even if an earlier one is longer."""
    text = "early a -> b -> c -> d. final p -> q."
    assert P.parse_path(text, prefer_last=True) == ["p", "q"]


def test_parse_path_neutralizes_undirected_edge_arrows():
    """`u <-> v` edge statements must not be mistaken for a `->` route hop."""
    # Only the genuine route a -> b should be parsed; the <-> edge is neutralised.
    assert P.parse_path("edges: m <-> n. route: a -> b") == ["a", "b"]


def test_parse_path_goto_actions():
    assert P.parse_path("goto(a), goto(b), goto(c)") == ["a", "b", "c"]


def test_parse_path_valid_nodes_filter():
    assert P.parse_path("a -> ghost -> c", valid_nodes={"a", "c"}) == ["a", "c"]


def test_parse_path_empty_and_nonstring():
    assert P.parse_path("") == []
    assert P.parse_path(None) == []
    assert P.parse_path("no route here") == []
    assert P.parse_path(123) == []


# ----------------------------------------------------------------------------
# validate_path — NetworkX scoring arithmetic
# ----------------------------------------------------------------------------

def test_validate_path_optimal_direct_edge():
    """a -> c uses the direct edge; cost & hop optimality are exactly 1.0."""
    m = P.validate_path("a -> c", _triangle_graph(), start="a", goal="c")
    assert m["parsed_nodes"] == ["a", "c"]
    assert m["nodes_exist_rate"] == 1.0
    assert m["edge_validity_rate"] == 1.0
    assert m["full_path_valid"] is True
    assert m["start_goal_ok"] is True
    assert m["cost_optimality"] == 1.0
    assert m["hop_optimality"] == 1.0


def test_validate_path_suboptimal_detour_cost_and_hop():
    """a -> b -> c: emitted cost 1+sqrt(2), shortest a->c is the direct edge (1.0);
    cost_optimality = (1+sqrt2)/1; emitted 2 hops vs shortest 1 hop -> hop_optimality 2."""
    m = P.validate_path("a -> b -> c", _triangle_graph(), start="a", goal="c")
    assert m["full_path_valid"] is True
    assert math.isclose(m["cost_optimality"], 1.0 + math.sqrt(2))
    assert m["hop_optimality"] == 2.0


def test_validate_path_hallucinated_node_lowers_exist_rate():
    m = P.validate_path("a -> ghost", _triangle_graph())
    assert m["parsed_nodes"] == ["a", "ghost"]
    assert m["nodes_exist_rate"] == 0.5
    assert m["full_path_valid"] is False
    assert m["cost_optimality"] is None  # not valid -> never computed


def test_validate_path_nonadjacent_real_nodes_invalid_edge():
    """Two real nodes with no edge between them: exist-rate 1, edge-rate 0, invalid."""
    g = {
        "objects": [{"name": "a", "coords": [0, 0]}, {"name": "z", "coords": [9, 9]}],
        "regions": [],
        "object_connections": [],  # a and z are not connected
    }
    m = P.validate_path("a -> z", g)
    assert m["nodes_exist_rate"] == 1.0
    assert m["edge_validity_rate"] == 0.0
    assert m["full_path_valid"] is False


def test_validate_path_no_route_returns_default_dict():
    m = P.validate_path("the model said nothing useful", _triangle_graph())
    assert m["parsed_nodes"] == []
    assert m["num_parsed"] == 0
    assert m["full_path_valid"] is False
    assert m["path_from_reasoning"] is False
    assert m["path_rescued"] is False


def test_validate_path_reasoning_fallback_no_model_call():
    """Empty plan but a route in reasoning_text -> recovered, flagged from_reasoning.
    This fallback is a pure regex re-scan; no judge/model is involved."""
    m = P.validate_path(
        "I will plan now.", _triangle_graph(),
        start="a", goal="c", reasoning_text="my final route is a -> b -> c",
    )
    assert m["parsed_nodes"] == ["a", "b", "c"]
    assert m["path_from_reasoning"] is True
    assert m["path_rescued"] is False


def test_validate_path_start_goal_ok_logic():
    m = P.validate_path("a -> b -> c", _triangle_graph(), start="a", goal="b")
    # goal mismatch (path ends at c, goal is b)
    assert m["start_goal_ok"] is False


# ----------------------------------------------------------------------------
# parse_edges
# ----------------------------------------------------------------------------

def test_parse_edges_all_three_forms_and_quotes():
    """`u <-> v`, `[u, v]`, `(u, v)` (quoted or not) all parse to frozenset pairs.
    Node ids must match the grid-id pattern (lowercase + _<int> tail)."""
    text = "bay_1 <-> bay_2 and ['bay_3', 'bay_4'] and (bay_5, bay_6)"
    edges = P.parse_edges(text)
    assert frozenset(("bay_1", "bay_2")) in edges
    assert frozenset(("bay_3", "bay_4")) in edges
    assert frozenset(("bay_5", "bay_6")) in edges


def test_parse_edges_drops_self_loops():
    assert P.parse_edges("bay_1 <-> bay_1") == set()


def test_parse_edges_valid_nodes_filter():
    edges = P.parse_edges("bay_1 <-> bay_2", valid_nodes={"bay_1"})
    assert edges == set()  # bay_2 not allowed -> edge dropped


# ----------------------------------------------------------------------------
# _ordered_subseq / _strip_regex / is_yes_no_task
# ----------------------------------------------------------------------------

def test_ordered_subseq():
    assert P._ordered_subseq(["a", "c"], ["a", "b", "c", "d"]) is True
    assert P._ordered_subseq(["c", "a"], ["a", "b", "c", "d"]) is False
    assert P._ordered_subseq([], ["a"]) is True


def test_strip_regex_removes_boundaries_and_lookarounds():
    out = P._strip_regex(r"\bbay_1\b(?i)(?=foo)")
    assert "bay_1" in out
    assert r"\b" not in out
    assert "(?i)" not in out


def test_is_yes_no_task():
    assert P.is_yes_no_task("q", "yes") is True
    assert P.is_yes_no_task("q", "the answer is no") is True
    assert P.is_yes_no_task("q", "bay_3") is False
    assert P.is_yes_no_task("q", None) is False


# ----------------------------------------------------------------------------
# derive_targets — endpoint/kind resolution
# ----------------------------------------------------------------------------

def _two_room_graph():
    """room_1 -- room_2 (regions); box_1 hosted in room_2; hall_1 a spare region."""
    return {
        "regions": [{"name": "room_1"}, {"name": "room_2"}, {"name": "hall_1"}],
        "objects": [{"name": "box_1"}],
        "object_connections": [["box_1", "room_2"]],
        "region_connections": [["room_1", "room_2"], ["room_1", "hall_1"]],
    }


def test_derive_targets_reachability_is_path_kind():
    goal, waypoints, avoid, required_edges, kind = P.derive_targets(
        _two_room_graph(), init_node="room_1",
        answer="room_2", criterion="room_2 is reachable from room_1", task="can you reach room_2?",
    )
    assert goal == "room_2"
    assert kind == "path"
    assert waypoints == []
    assert avoid == []
    assert required_edges == []


def test_derive_targets_positionality_is_edges_kind_with_required_edge():
    goal, waypoints, avoid, required_edges, kind = P.derive_targets(
        _two_room_graph(), init_node="room_1",
        answer="room_2", criterion="state that box_1 is located in room_2",
        task="where is box_1?",
    )
    assert goal == "room_2"
    assert kind == "edges"
    assert frozenset(("box_1", "room_2")) in required_edges


def test_derive_targets_avoid_clause():
    goal, waypoints, avoid, required_edges, kind = P.derive_targets(
        _two_room_graph(), init_node="room_1",
        answer="room_2",
        criterion="reach room_2 without using hall_1", task="navigate to room_2",
    )
    assert goal == "room_2"
    assert "hall_1" in avoid
    assert kind == "path"


def test_derive_targets_no_graph_goal_returns_none():
    """A count/other task whose answer & criterion name no graph node -> goal None."""
    goal, *_ = P.derive_targets(
        _two_room_graph(), init_node="room_1",
        answer="42", criterion="how many objects are there", task="count the objects",
    )
    assert goal is None


# ----------------------------------------------------------------------------
# validate_structured + _augment_eval_metrics
# ----------------------------------------------------------------------------

def test_validate_structured_reachability_valid_route():
    """A valid constrained walk room_1 -> room_2 to the goal scores structured_correct."""
    g = _two_room_graph()
    m = P.validate_structured(
        "plan: room_1 -> room_2", g,
        init_node="room_1", answer="room_2",
        criterion="room_2 is reachable from room_1", task="reach room_2",
        full_response="plan: room_1 -> room_2",
    )
    assert m is not None
    assert m["kind"] == "path"
    assert m["goal"] == "room_2"
    assert m["full_path_valid"] is True
    assert m["start_goal_ok"] is True
    assert m["structured_correct"] is True


def test_validate_structured_reachability_wrong_destination_fails():
    g = _two_room_graph()
    m = P.validate_structured(
        "plan: room_1 -> hall_1", g,
        init_node="room_1", answer="room_2",
        criterion="room_2 is reachable from room_1", task="reach room_2",
        full_response="plan: room_1 -> hall_1",
    )
    assert m["structured_correct"] is False


def test_validate_structured_returns_none_when_no_goal():
    g = _two_room_graph()
    m = P.validate_structured(
        "whatever", g, init_node="room_1",
        answer="42", criterion="count the objects", task="how many objects",
        full_response="whatever",
    )
    assert m is None


def test_augment_eval_metrics_hallucination_and_path_expected():
    """hallucination_rate = 1 - edge_validity_rate for >=2-node routes; path_expected
    excludes positionality (kind == 'edges')."""
    m = {"kind": "path", "edge_validity_rate": 0.25, "num_parsed": 3,
         "full_path_valid": False, "start_goal_ok": False, "hop_optimality": 0.9}
    out = P._augment_eval_metrics(dict(m), goal="room_2")
    assert out["path_expected"] is True
    assert math.isclose(out["hallucination_rate"], 0.75)
    assert out["valid_path_ab"] is False
    assert out["hop_optimality"] is None  # nulled for non-valid A->B path

    edges_kind = P._augment_eval_metrics(
        {"kind": "edges", "edge_validity_rate": 1.0, "num_parsed": 1}, goal="room_2")
    assert edges_kind["path_expected"] is False
    assert edges_kind["hallucination_rate"] is None  # <2 nodes -> no hop to grade


# ----------------------------------------------------------------------------
# evaluate_sample — judge-skip branches (no model needed)
# ----------------------------------------------------------------------------

def test_evaluate_sample_structured_skips_judge():
    """A structural task is graded deterministically; judge is never invoked."""
    g = _two_room_graph()
    pm = P.evaluate_sample(
        "reach room_2", "room_1 -> room_2", g,
        init_node="room_1", answer="room_2",
        acceptance_criterion="room_2 is reachable from room_1",
    )
    assert pm["structured"] is True
    assert pm["judge_used"] is False
    assert pm["llm_judge_pass"] is None
    assert pm["structured_correct"] is True


def test_evaluate_sample_nonstructured_no_criterion_skips_judge():
    """No acceptance_criterion + non yes/no answer -> judge skipped, pass None."""
    g = _triangle_graph()
    pm = P.evaluate_sample(
        "go a to c", "a -> c", g, init_node="a", answer="c",
    )
    assert pm["structured"] is False
    assert pm["judge_used"] is False
    assert pm["llm_judge_pass"] is None


# ----------------------------------------------------------------------------
# Judge boundary — handling-only (stubbed model, worst-case carve-out)
# ----------------------------------------------------------------------------

def test_judge_acceptance_handles_pass_fail_output(monkeypatch=None):
    """The judge model is opaque: we stub it to return fixed bytes and assert ONLY
    that judge_acceptance parses PASS->True / FAIL->False. No accuracy claim."""
    P._JUDGE["loaded"] = True
    try:
        P._JUDGE["gen"] = lambda prompt, max_new_tokens=4: "PASS\n"
        assert P.judge_acceptance("q", "ans", acceptance_criterion="crit") is True
        P._JUDGE["gen"] = lambda prompt, max_new_tokens=4: "FAIL"
        assert P.judge_acceptance("q", "ans", acceptance_criterion="crit") is False
    finally:
        P._JUDGE["loaded"] = False
        P._JUDGE["gen"] = None


def test_judge_acceptance_none_when_no_reference():
    """No criterion and no answer regex -> returns None without touching the model."""
    assert P.judge_acceptance("q", "ans") is None


def test_write_path_with_judge_output_feeds_back_through_parse():
    """The judge's recovered route string is consumed by parse_path. We stub the
    model to emit a canonical route and assert the handling (parse) recovers it."""
    P._JUDGE["loaded"] = True
    try:
        P._JUDGE["gen"] = lambda prompt, max_new_tokens=256: "room_1 -> room_2"
        route = P.write_path_with_judge("blah", {"room_1", "room_2"})
        assert P.parse_path(route, prefer_last=True) == ["room_1", "room_2"]
    finally:
        P._JUDGE["loaded"] = False
        P._JUDGE["gen"] = None


def test_write_path_with_judge_empty_when_no_judge():
    P._JUDGE["loaded"] = True
    try:
        P._JUDGE["gen"] = None
        assert P.write_path_with_judge("anything", {"room_1"}) == ""
    finally:
        P._JUDGE["loaded"] = False
        P._JUDGE["gen"] = None


# ----------------------------------------------------------------------------
# combine_verdict — pure two-verdict truth table
# ----------------------------------------------------------------------------

def test_combine_verdict_false_positive():
    """RegEx keyword hit but judge rejects -> false_positive, subjective False."""
    v = P.combine_verdict(regex_correct=True, regex_keyword=True,
                          judge_pass=False, acceptance_criterion_present=True)
    assert v["objective_keyword"] is True
    assert v["subjective_correct"] is False
    assert v["false_positive"] is True
    assert v["false_negative"] is False
    assert v["judged"] is True


def test_combine_verdict_false_negative():
    v = P.combine_verdict(regex_correct=False, regex_keyword=False,
                          judge_pass=True, acceptance_criterion_present=True)
    assert v["subjective_correct"] is True
    assert v["false_negative"] is True
    assert v["false_positive"] is False


def test_combine_verdict_not_judged_when_pass_none():
    v = P.combine_verdict(regex_correct=True, regex_keyword=True,
                          judge_pass=None, acceptance_criterion_present=True)
    assert v["judged"] is False
    assert v["subjective_correct"] is None
    assert v["false_positive"] is False
    assert v["false_negative"] is False


def test_combine_verdict_not_judged_without_criterion():
    """judge_pass present but no acceptance_criterion -> still unjudged (disjoint inputs)."""
    v = P.combine_verdict(regex_correct=True, regex_keyword=True,
                          judge_pass=True, acceptance_criterion_present=False)
    assert v["judged"] is False
    assert v["subjective_correct"] is None


# ----------------------------------------------------------------------------
# aggregate_path_metrics — run-level arithmetic
# ----------------------------------------------------------------------------

def test_aggregate_path_metrics_means_and_rates():
    """Hand-computed aggregate over 3 samples (2 with a route, 1 without)."""
    samples = [
        {"path_metrics": {
            "num_parsed": 3, "edge_validity_rate": 1.0, "cost_optimality": 1.0,
            "path_from_reasoning": True, "path_rescued": False,
            "path_expected": True, "valid_path_ab": True,
            "hop_optimality": 1.0, "hallucination_rate": 0.0}},
        {"path_metrics": {
            "num_parsed": 2, "edge_validity_rate": 0.5, "cost_optimality": None,
            "path_from_reasoning": False, "path_rescued": True,
            "path_expected": True, "valid_path_ab": False,
            "hop_optimality": None, "hallucination_rate": 0.5}},
        {"path_metrics": {
            "num_parsed": 0, "path_expected": False}},
    ]
    agg = P.aggregate_path_metrics(samples)
    # Legacy family (over the 2 routes parsed):
    assert math.isclose(agg["edge_validity_rate"], 0.75)   # (1.0 + 0.5)/2
    assert agg["cost_optimality"] == 1.0                   # only one non-None value
    assert agg["num_with_path"] == 2
    assert agg["num_from_reasoning"] == 1
    assert agg["num_rescued"] == 1
    # eval/* family (over the full list):
    assert agg["valid_path_rate"] == 0.5                   # 1 valid of 2 expected
    assert agg["num_path_expected"] == 2
    assert agg["path_optimality_rate"] == 1.0              # only one hop_optimality
    assert math.isclose(agg["hallucination_rate"], 0.25)   # (0.0 + 0.5)/2


def test_aggregate_path_metrics_empty_input():
    assert P.aggregate_path_metrics([]) == {}
    assert P.aggregate_path_metrics([{"path_metrics": None}]) == {}


# ----------------------------------------------------------------------------
# Standalone runner (works under pytest and as a plain script)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"{name}: PASS")
    print("done")
