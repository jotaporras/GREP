"""Append fixed-endpoint long-hop Navigability tasks to already-populated graphs.

Extends a frozen test set (e.g. the v2 n60 test graphs) for a corpus whose new
train graphs carry long-hop tasks: each source graph keeps its original tasks
untouched and gains --n-longhop new ones whose (init, goal) region pair is
sampled uniformly over shortest-path lengths [3, diameter]. Endpoints are fixed
in Python; the LLM only phrases the task, and its output is rejected unless it
honours them — so the hop-length distribution cannot drift.

Outputs land as data_gen_{start-index+i:03d}.json (+ graph_gen twin) in
--dst-dir, renumbered after the new train graphs. The old->new file mapping is
written to <dst-dir>/appended_test_manifest.json and the sampled endpoints to
<dst-dir>/longhop_manifest.json (same shape as the populate path's manifest).

Requires the same LLM backend env as the populate phase (PRISM_LLM_BACKEND,
PRISM_HF_MODEL, ...). Idempotent: a destination file that already carries the
expected task count is skipped, so an interrupted run resumes for free.

Usage:
    python scripts/append_longhop_tasks.py \
        --src-dir  $V2/gen/nav_n60_gemma_data/split/test_graphs \
        --dst-dir  $V3/gen/nav_n60_gemma_data/populated_graphs \
        --start-index 52 --n-longhop 2 --seed 4242
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from prism.data import graph_gen

PROMPT_TEMPLATE = """You are extending an existing scene graph's task list for an LLM-planner benchmark.

Below is a populated scene graph (final node names — there is NO rename step) and its existing tasks. Generate EXACTLY {n_new} additional Navigability task(s) with FIXED endpoints:

{constraint_lines}

Rules for each new task:
- "init_node" must be EXACTLY the required start region id.
- The task must ask for the full multi-hop route to the required destination and end with a clause such as "give the full path and its connecting edges".
- Phrase the destination semantically (by a contained object, its description, or its name theme) rather than by node id; if sibling nodes share the destination's name prefix and nothing disambiguates it, you MAY name the destination id in the task text.
- Do not add waypoint or avoid constraints.
- Do not duplicate any existing task.
- "answer" is a regex matching the start then the destination id, e.g. "(?i)\\bstart_region_1\\b.*\\bgoal_region_1\\b". Wrap every node id in \\b word boundaries (written \\\\b inside the JSON string). Never anchor with ^ or $, never encode the full path, never key on yes/no words.
- "acceptance_criterion" is ONE sentence naming, by node id, the start region and the destination region and requiring a valid route between them (any valid walk is acceptable — do not spell out an example path or name intermediate regions).

Scene graph:
{graph_json}

Existing tasks (do not repeat):
{existing_tasks}

Return ONLY valid JSON, no extra text:
{{"tasks": [{{"task": "...", "answer": "...", "init_node": "...", "acceptance_criterion": "..."}}]}}
"""

FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")


def build_prompt(data: dict, constraints: list) -> str:
    lines = []
    for k, c in enumerate(constraints, 1):
        lines.append(
            f"- New task {k}: start (init_node) = \"{c['init']}\", "
            f"destination = \"{c['goal']}\" (optimal route: {c['hops']} hops)."
        )
    existing = "\n".join(f"- {t['task']}" for t in data["tasks"])
    return PROMPT_TEMPLATE.format(
        n_new=len(constraints),
        constraint_lines="\n".join(lines),
        graph_json=json.dumps(data["graph"], indent=1),
        existing_tasks=existing,
    )


def parse_tasks(response: str, n_new: int, graph: dict, constraints: list) -> list:
    obj = json.loads(FENCE.sub("", response.strip()))
    tasks = obj["tasks"]
    if len(tasks) != n_new:
        raise ValueError(f"expected {n_new} tasks, got {len(tasks)}")
    graph_gen.validate_task_refs(tasks, graph)
    graph_gen.validate_longhop_tasks(tasks, constraints)
    return tasks


def update_manifest(path: Path, key: str, value) -> None:
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest[key] = value
    path.write_text(json.dumps(manifest, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-dir", type=Path, required=True,
                    help="Directory of populated data_gen_*.json graphs to extend.")
    ap.add_argument("--dst-dir", type=Path, required=True,
                    help="populated_graphs/ dir of the target corpus.")
    ap.add_argument("--start-index", type=int, required=True,
                    help="First output index (files are renumbered from here).")
    ap.add_argument("--n-longhop", type=int, default=2)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--attempts", type=int, default=8)
    ap.add_argument("--reasoning-effort", type=str, default="low")
    args = ap.parse_args()

    src_files = sorted(args.src_dir.glob("data_gen_*.json"))
    if not src_files:
        raise SystemExit(f"No data_gen_*.json in {args.src_dir}")
    args.dst_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    gen = graph_gen.TaskGraphGen()

    failures = []
    for i, src in enumerate(src_files):
        idx = args.start_index + i
        stem = f"data_gen_{idx:03d}"
        out_path = args.dst_dir / f"{stem}.json"
        data = json.loads(src.read_text())
        n_expected = len(data["tasks"]) + args.n_longhop

        if out_path.exists():
            existing = json.loads(out_path.read_text())
            if len(existing.get("tasks", [])) == n_expected:
                print(f"{src.name} -> {out_path.name}: already extended, skipping")
                continue

        constraints = graph_gen.sample_longhop_constraints(
            data["graph"], args.n_longhop, rng
        )
        print(f"{src.name} -> {out_path.name}: constraints {constraints}")

        new_tasks = None
        for attempt in range(1, args.attempts + 1):
            try:
                response = gen.client.query_gpt(
                    query=build_prompt(data, constraints),
                    reasoning_effort=args.reasoning_effort,
                )
                new_tasks = parse_tasks(
                    response, args.n_longhop, data["graph"], constraints
                )
                break
            except Exception as ex:
                print(f"  attempt {attempt}/{args.attempts} rejected: {ex}")
        if new_tasks is None:
            failures.append(src.name)
            continue

        data["tasks"] = data["tasks"] + new_tasks
        tmp = out_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(out_path)
        graph_twin = args.dst_dir / f"graph_gen_{idx:03d}.json"
        graph_twin.write_text(json.dumps(data["graph"], indent=2))

        update_manifest(
            args.dst_dir / "appended_test_manifest.json", f"{stem}.json", src.name
        )
        update_manifest(args.dst_dir / "longhop_manifest.json", stem, constraints)

    if failures:
        raise SystemExit(
            f"{len(failures)} graph(s) failed all attempts: {failures} — "
            f"re-run to retry just those."
        )
    print(f"Done: extended {len(src_files)} graphs into {args.dst_dir}")


if __name__ == "__main__":
    main()
