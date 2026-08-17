"""Verifiable rewards for e16 RL training — pure functions, no LLM judge.

Each completion is parsed exactly the way eval parses model output
(``compact_prompt.compact_output_to_spine_json``) and graded by the same
regex/NetworkX machinery (``path_validator.validate_path``), so the reward is
the eval metric, not a proxy of it. Per docs/metrics.md, the shaped scalar is
for optimization only — the raw components are logged separately in wandb so
the causally-meaningful free-generation accuracy stays visible.

trl contract: a reward function receives ``prompts``, ``completions``, and
every non-``prompt`` dataset column as a keyword list (here ``scene_graph_dict``,
``answer_regex``, ``init_node``), and returns ``list[float]``.
"""
from __future__ import annotations

import json
import re

from prism.data import compact_prompt
from prism.eval import path_validator

# Shaped-reward composition; overridable via trainer.rl.reward_weights.
DEFAULT_REWARD_WEIGHTS = {
    "full_path_valid": 1.0,
    # e17 RCA (2026-08-16): 76% of held-out failures were paths that reached
    # the goal with 1-3 hallucinated edges (mostly exact 2-hop shortcuts).
    # Linear partial credit made a single skipped hop nearly free, so v2
    # doubles the dense edge term AND adds clean_path — a binary all-edges-
    # and-nodes-real bonus that any hallucination forfeits entirely.
    "edge_validity_rate": 0.6,
    "clean_path": 1.0,
    "nodes_exist_rate": 0.2,
    "cost_optimality": 0.3,
    "format_ok": 0.2,
    "keyword": 0.5,
}


def parse_completion(completion: str) -> dict:
    """Model output -> the SPINE answer dict eval grades (empty dict on garbage)."""
    try:
        return json.loads(compact_prompt.compact_output_to_spine_json(completion))
    except (ValueError, TypeError):
        return {}


def grade_completion(completion: str, scene_graph_dict: dict, init_node: str,
                     answer_regex: str) -> dict:
    """All raw reward components for one completion. Never raises."""
    parsed = parse_completion(completion)
    plan = str(parsed.get("plan", ""))
    format_ok = all(k in parsed for k in
                    ("primary_goal", "relevant_graph", "reasoning", "plan"))
    keyword = bool(re.search(answer_regex, plan, re.IGNORECASE)) if plan else False
    metrics = path_validator.validate_path(
        plan, scene_graph_dict, start=init_node,
        reasoning_text=str(parsed.get("reasoning", "")))
    edge_validity = float(metrics.get("edge_validity_rate") or 0.0)
    nodes_exist = float(metrics.get("nodes_exist_rate") or 0.0)
    num_parsed = int(metrics.get("num_parsed") or 0)
    return {
        "full_path_valid": float(metrics.get("full_path_valid") or 0.0),
        "edge_validity_rate": edge_validity,
        "clean_path": float(
            num_parsed > 0 and edge_validity == 1.0 and nodes_exist == 1.0),
        "nodes_exist_rate": nodes_exist,
        "cost_optimality": float(metrics.get("cost_optimality") or 0.0),
        "format_ok": float(format_ok),
        "keyword": float(keyword),
    }


def make_reward_funcs(weights: dict | None = None):
    """One trl reward function per component (weighted), so trl's per-function
    wandb logging exposes each raw component alongside the weighted sum it
    optimizes. Components with weight 0 are dropped entirely.
    """
    weights = {**DEFAULT_REWARD_WEIGHTS, **(weights or {})}

    def component_fn(name: str, weight: float):
        def fn(prompts, completions, scene_graph_dict, init_node, answer_regex,
               **kwargs):
            return [
                weight * grade_completion(c, g, n, a)[name]
                for c, g, n, a in zip(
                    completions, scene_graph_dict, init_node, answer_regex)
            ]
        fn.__name__ = f"reward_{name}"
        return fn

    return [component_fn(name, w) for name, w in weights.items() if w != 0.0]
