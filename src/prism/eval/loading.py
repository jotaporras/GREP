"""Disk I/O for PRISM eval data.

Kept separate from `prism.eval.evaluate` so the scoring library stays
pure in-memory. Callers that own a CLI / sbatch / config-driven entry
point (the standalone driver and the trainer's post-train hook) come
through here to turn a path on disk into `{graph_name: [EvalSample]}` plus
a parallel `{graph_name: graph_file_path}` map.

Single public function: `load_samples_by_graph(target)`.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Tuple

from prism.eval import evaluate


def load_samples_by_graph(target: str) -> Tuple[Dict[str, List[evaluate.EvalSample]], Dict[str, str]]:
    """Resolve `target` (file, directory, or glob) and load each graph JSON.

    Returns `(samples_by_graph, graph_file_by_name)` where both dicts are
    keyed by the graph file stem (e.g. ``"data_gen_004"``):

    * ``samples_by_graph[stem]`` — list of `EvalSample`s for that graph,
      ready to hand to `evaluate.eval_model_multiple_graphs`.
    * ``graph_file_by_name[stem]`` — the source path the samples came from,
      for callers that need to record provenance in their output JSON.

    Raises `SystemExit` if `target` resolves to zero matching files.
    """
    graph_files = _resolve_graph_files(target)

    samples_by_graph: Dict[str, List[evaluate.EvalSample]] = {}
    graph_file_by_name: Dict[str, str] = {}
    for gf in graph_files:
        stem = os.path.splitext(os.path.basename(gf))[0]
        with open(gf) as f:
            payload = json.load(f)
        samples_by_graph[stem] = evaluate.construct_eval_samples_from_dict(
            payload["graph"], payload["tasks"], graph_name=stem,
        )
        graph_file_by_name[stem] = gf
    return samples_by_graph, graph_file_by_name


def _resolve_graph_files(target: str) -> List[str]:
    """Expand `target` (single file, directory of JSONs, or glob) into a sorted list."""
    if os.path.isdir(target):
        files = sorted(glob.glob(os.path.join(target, "*.json")))
    elif any(ch in target for ch in ("*", "?", "[")):
        files = sorted(glob.glob(target))
    else:
        files = [target]
    if not files:
        raise SystemExit(f"No graph JSON files found at {target}")
    return files
