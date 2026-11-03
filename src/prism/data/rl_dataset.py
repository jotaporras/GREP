"""RL prompt dataset built from the nav graph/task files (``data_gen_NNN.json``).

Each row is one (graph, task) pair rendered as the EXACT prompt the eval
harness would show the model — the SPINE base prompt (``evaluate.
_fixed_get_base_prompt``) compacted by ``compact_prompt.spine_to_compact_messages``
and chat-templated — so the RL policy trains on the distribution it is
evaluated on. Reward-relevant fields ride along as passthrough columns for the
trl reward functions; the scene graph also stays parseable from the prompt
text itself, which is what the rollout engine's Ψ construction reads.

Imports the spine package (via ``prism.eval.evaluate``) — cluster/plaza envs
only, like the eval stack it mirrors.
"""
from __future__ import annotations

import glob
import json
import os

from datasets import Dataset

from prism.data import compact_prompt


def _task_rows(graph_file: str, keep_robot_location: bool) -> list[dict]:
    """One row per task. The robot is placed at the task's ``init_node``
    unless the task carries ``keep_robot_location`` (e22 named start) or
    ``keep_robot_location`` is set for the whole set (the robot's own sweep
    keeps ``plaza_1``); then the graph's ``robot_location`` stays."""
    with open(graph_file) as f:
        payload = json.load(f)
    graph = payload["graph"]
    name = os.path.splitext(os.path.basename(graph_file))[0]
    rows = []
    for task in payload["tasks"]:
        keep = keep_robot_location or task.get("keep_robot_location", False)
        robot_location = graph["robot_location"] if keep else task["init_node"]
        rows.append({
            "graph_name": name,
            "scene_graph_dict": {**graph, "robot_location": robot_location},
            "task": task["task"],
            "answer_regex": task["answer"],
            "init_node": task["init_node"],
        })
    return rows


def load_rl_dataset(
    graphs: str,
    tokenizer,
    *,
    include_edges: bool,
    use_icl: bool = False,
    icl_examples: int = 0,
    keep_robot_location: bool = False,
) -> Dataset:
    """``graphs``: a dir of ``data_gen_*.json`` files or a glob. Returns a
    Dataset with columns ``prompt`` (templated text), ``scene_graph_dict``,
    ``answer_regex``, ``init_node``, ``graph_name``, ``task``.

    Prompt policy mirrors eval: tools per ``PRISM_DISABLE_SPINE_TOOLS`` (e16
    trains tool-free, matching the e14 arms), ICL per ``use_icl`` /
    ``icl_examples``, edge bullets per ``include_edges`` (the checkpoint's
    ``text_edge_list`` policy). ``keep_robot_location`` (e22): keep every
    graph's own ``robot_location`` instead of moving the robot to each task's
    ``init_node`` (tasks flagged ``keep_robot_location`` always keep it).
    """
    # Deferred: pulls in the spine package.
    from prism.eval import evaluate

    pattern = graphs if any(ch in graphs for ch in "*?[") else os.path.join(
        graphs, "data_gen_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no data_gen files match {pattern!r}")

    include_tools = not evaluate._spine_tools_disabled()
    rows = []
    for f in files:
        for row in _task_rows(f, keep_robot_location):
            spine_msg = evaluate._fixed_get_base_prompt(
                row["task"], row["scene_graph_dict"], use_icl=use_icl)
            llm_msg = compact_prompt.spine_to_compact_messages(
                spine_msg, include_edges=include_edges,
                include_tools=include_tools, icl_examples=icl_examples)
            row["prompt"] = tokenizer.apply_chat_template(
                llm_msg, tokenize=False, add_generation_prompt=True)
            rows.append(row)
    return Dataset.from_list(rows)
