"""Reward-function unit tests (pure CPU, no spine/vllm/trl)."""
import pytest

from prism.training.rewards import (
    DEFAULT_REWARD_WEIGHTS, grade_completion, make_reward_funcs, parse_completion,
)

GRAPH = {
    "regions": [
        {"name": "kitchen", "coords": [0, 0, 0]},
        {"name": "hallway", "coords": [1, 0, 0]},
        {"name": "garage", "coords": [2, 0, 0]},
        {"name": "office", "coords": [3, 0, 0]},
    ],
    "objects": [],
    "region_connections": [["kitchen", "hallway"], ["hallway", "garage"],
                           ["garage", "office"]],
    "object_connections": [],
    "robot_location": "kitchen",
}

GOOD = ('{"primary_goal": "reach office", "relevant_graph": "kitchen-hallway-garage-office", '
        '"reasoning": "walk the chain", "plan": "kitchen -> hallway -> garage -> office"}')
BAD_EDGE = ('{"primary_goal": "g", "relevant_graph": "r", "reasoning": "x", '
            '"plan": "kitchen -> office"}')


def test_good_plan_scores_high():
    g = grade_completion(GOOD, GRAPH, "kitchen", r"(?i)\boffice\b")
    assert g["full_path_valid"] == 1.0
    assert g["edge_validity_rate"] == 1.0
    assert g["nodes_exist_rate"] == 1.0
    assert g["format_ok"] == 1.0
    assert g["keyword"] == 1.0


def test_invalid_edge_penalized():
    g = grade_completion(BAD_EDGE, GRAPH, "kitchen", r"(?i)\boffice\b")
    assert g["full_path_valid"] == 0.0
    assert g["edge_validity_rate"] == 0.0  # kitchen->office is not an edge
    assert g["nodes_exist_rate"] == 1.0    # both nodes exist though


def test_garbage_never_raises():
    g = grade_completion("total garbage %%%", GRAPH, "kitchen", r"x")
    # format_ok is 1.0 even here: compact_output_to_spine_json is deliberately
    # tolerant and wraps ANY bare text as [answer(...)] with the four keys — a
    # constant offset that cancels in GRPO's group-relative advantages. The
    # discriminating components must all be zero.
    for name in ("full_path_valid", "edge_validity_rate", "nodes_exist_rate",
                 "cost_optimality", "keyword"):
        assert g[name] == 0.0
    assert isinstance(parse_completion(""), dict)


def test_reward_funcs_weighted_and_named():
    funcs = make_reward_funcs({"full_path_valid": 2.0, "edge_validity_rate": 0.0})
    names = {f.__name__ for f in funcs}
    assert "reward_full_path_valid" in names
    assert "reward_edge_validity_rate" not in names  # weight 0 dropped
    fpv = next(f for f in funcs if f.__name__ == "reward_full_path_valid")
    out = fpv(prompts=["p"], completions=[GOOD], scene_graph_dict=[GRAPH],
              init_node=["kitchen"], answer_regex=[r"(?i)\boffice\b"])
    assert out == [2.0]


def test_clean_path_is_binary():
    # Fully real path -> 1.0; any hallucinated edge forfeits the whole bonus
    # even though the goal node is reached (the e17 dominant failure mode).
    assert grade_completion(GOOD, GRAPH, "kitchen",
                            r"(?i)\boffice\b")["clean_path"] == 1.0
    bad = grade_completion(BAD_EDGE, GRAPH, "kitchen", r"(?i)\boffice\b")
    assert bad["clean_path"] == 0.0
    # Garbage (no parsed path) must not collect the bonus either.
    assert grade_completion("total garbage %%%", GRAPH, "kitchen",
                            r"x")["clean_path"] == 0.0


def test_default_weights_cover_all_components():
    g = grade_completion(GOOD, GRAPH, "kitchen", r"(?i)\boffice\b")
    assert set(DEFAULT_REWARD_WEIGHTS) == set(g)
