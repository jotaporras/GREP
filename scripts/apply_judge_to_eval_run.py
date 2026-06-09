"""Retro-apply the M10 LLM judge (+ path validation) to a finished eval run.

When an eval run completes while the Gemma judge weights are unavailable (no HF
auth / no GPU), every acceptance_criterion sample falls back to RegEx/NetworkX
only and the subjective column is empty. This script re-opens those result JSONs
and fills the judge scoring back in *without re-running the model* — it replays
each recorded planner answer through the SAME grading framework a live eval uses
(``prism.eval.path_validator``):

  * It recreates the route the planner was trying to emit from the recorded
    ``plan`` text and grades it against the scene graph with ``path_validator``
    (regex + NetworkX: node existence, edge validity, full-path validity,
    start/goal, cost optimality; plus the deterministic edge/path check for
    structural positionality / reachability / navigability tasks).
  * For acceptance_criterion / yes-no tasks it then runs the Gemma judge
    (``path_validator.judge_acceptance``) and folds the verdict in with
    ``path_validator.combine_verdict`` — exactly as
    ``evaluate.eval_model_single_graph`` does on a live run.

All original text is preserved. The script only ADDS / refreshes the judge-side
fields (``path_metrics``, ``llm_judge_pass``, ``subjective_correct``,
``false_positive``, ``false_negative``, ``structured``) and the summary
aggregates (``subjective_accuracy``, ``num_judged``, ``num_false_pos``,
``num_false_neg``, ``path_metrics``). The objective RegEx/NetworkX verdicts
(``correct``, ``plan_keyword``, ``accuracy`` …) are left untouched.

The graph, ``init_node`` and ``acceptance_criterion`` are not stored in the
result JSON, so the source dataset (the populated-graph file the run scored) is
needed. By default it is read from each result's ``path`` / ``eval_data`` field;
pass ``--dataset`` to point at it explicitly (a file, directory or glob) when
that recorded path has since moved.

Usage:
    # re-grade one result file, write <name>.judged.json next to it
    python scripts/apply_judge_to_eval_run.py results/perm_41/e5_..._100.json

    # a whole directory of result files, in place, with an explicit dataset dir
    python scripts/apply_judge_to_eval_run.py results/perm_41/ \
        --dataset data/gen/nav100_n30_gemma_data/populated_graphs --in-place

    # override the judge model (default is path_validator.GEMMA_JUDGE_MODEL)
    python scripts/apply_judge_to_eval_run.py run.json --model google/gemma-4-E4B-it
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

# Make the in-repo package importable when run from a bare checkout (mirrors
# scripts/fix_answer_regexes.py); a real install / PYTHONPATH=src also works.
_REPO_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if os.path.isdir(_REPO_SRC) and _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)


def _resolve(target: str) -> list:
    """Expand a file / directory / glob into a sorted list of JSON paths."""
    if os.path.isdir(target):
        files = sorted(glob.glob(os.path.join(target, "*.json")))
    elif any(ch in target for ch in ("*", "?", "[")):
        files = sorted(glob.glob(target))
    else:
        files = [target]
    if not files:
        raise SystemExit(f"No JSON files found at {target}")
    return files


def _dataset_index(targets: list) -> dict:
    """Build ``{stem: {"graph": dict, "by_task": {task: taskdict}, "ordered": [taskdict]}}``.

    ``stem`` is the dataset file's basename (e.g. ``eval_graph_unique_100``), the
    same identifier ``eval`` stamps onto each sample's ``graph_name``. Each
    ``taskdict`` keeps the dataset's ``task`` / ``answer`` / ``init_node`` /
    ``acceptance_criterion`` fields — everything ``path_validator.evaluate_sample``
    needs but the result JSON does not carry.
    """
    index: dict = {}
    for ds in targets:
        for gf in _resolve(ds):
            with open(gf) as f:
                payload = json.load(f)
            # Skip JSONs that aren't populated-graph datasets — a directory/glob
            # may also sweep in result files (which carry "samples", not "tasks").
            if not (isinstance(payload, dict) and "graph" in payload and "tasks" in payload):
                continue
            stem = os.path.splitext(os.path.basename(gf))[0]
            tasks = payload["tasks"]
            index[stem] = {
                "graph": payload["graph"],
                "by_task": {t["task"]: t for t in tasks},
                "ordered": tasks,
            }
    if not index:
        raise SystemExit(
            f"No populated-graph datasets (with 'graph'+'tasks') found in: {targets}")
    return index


def _pick_task(index: dict, sample: dict, result_stem: str):
    """Match one recorded result sample back to its dataset entry.

    Returns ``(graph_dict, task_dict)`` or ``None``. Prefers the sample's own
    ``graph_name``, then the result file's stem; matches the task by text,
    falling back to positional ``idx`` within that graph.
    """
    stem = sample.get("graph_name") or result_stem
    ds = index.get(stem)
    if ds is None and len(index) == 1:
        ds = next(iter(index.values()))
    if ds is None:  # last resort: any dataset that contains this exact task text
        ds = next((d for d in index.values() if sample.get("task") in d["by_task"]), None)
    if ds is None:
        return None
    task = ds["by_task"].get(sample.get("task"))
    if task is None:
        i = sample.get("idx")
        if isinstance(i, int) and 0 <= i < len(ds["ordered"]):
            task = ds["ordered"][i]
    return (ds["graph"], task) if task is not None else None


def _path_metrics(planner_response, graph_dict, task, path_validator):
    """Recreate + grade the planner's path via ``path_validator.evaluate_sample``.

    Mirrors ``evaluate._sample_path_metrics``: the route is pulled from the
    recorded ``plan`` field; the full response is passed for the judge. Never
    raises — returns ``None`` on failure.
    """
    try:
        plan = planner_response.get("plan") if isinstance(planner_response, dict) else planner_response
        return path_validator.evaluate_sample(
            task["task"],
            "" if plan is None else str(plan),
            graph_dict,
            init_node=task.get("init_node"),
            acceptance_criterion=task.get("acceptance_criterion"),
            answer=task.get("answer"),
            full_response="" if planner_response is None else str(planner_response),
        )
    except Exception as e:
        print(f"[apply-judge] path-metric computation failed: {type(e).__name__}: {e}")
        return None


def _regrade_sample(sample: dict, graph_dict, task, path_validator) -> dict:
    """Add judge + path-validation fields to one sample in place; return stats.

    Replays the recorded planner ``response`` (recreate path -> NetworkX grade ->
    Gemma judge -> ``combine_verdict``), mirroring ``eval_model_single_graph``.
    The objective RegEx fields already on the sample (``correct``,
    ``plan_keyword``) are reused, never recomputed, so existing scoring is kept.
    """
    ac_present = bool(task.get("acceptance_criterion"))
    stats = {"judged": False, "ac": ac_present, "false_positive": False, "false_negative": False}
    planner_response = sample.get("response")
    if planner_response is None:  # crashed / unparseable sample — nothing to grade
        sample["path_metrics"] = None
        sample["structured"] = False
        sample["llm_judge_pass"] = None
        sample.setdefault("subjective_correct", None)
        sample.setdefault("false_positive", False)
        sample.setdefault("false_negative", False)
        return stats

    pm = _path_metrics(planner_response, graph_dict, task, path_validator)
    structured = bool(pm and pm.get("structured"))
    judge_pass = (pm or {}).get("llm_judge_pass")

    if structured:
        # Structural task: graded deterministically by NetworkX; no LLM judge and
        # no subjective column (matches eval_model_single_graph).
        v = {"subjective_correct": None, "false_positive": False, "false_negative": False}
    else:
        v = path_validator.combine_verdict(
            regex_correct=bool(sample.get("correct")),
            regex_keyword=bool(sample.get("plan_keyword")),
            judge_pass=judge_pass,
            acceptance_criterion_present=ac_present,
        )

    sample["path_metrics"] = pm
    sample["structured"] = structured
    sample["llm_judge_pass"] = judge_pass
    sample["subjective_correct"] = v["subjective_correct"]
    sample["false_positive"] = v["false_positive"]
    sample["false_negative"] = v["false_negative"]

    stats["judged"] = judge_pass is not None
    stats["false_positive"] = v["false_positive"]
    stats["false_negative"] = v["false_negative"]
    return stats


def _aggregate_path_metrics(samples: list) -> dict:
    """Mean M10 path metrics over samples that produced a parseable route.

    Mirrors ``evaluate._aggregate_path_metrics`` so the re-graded summary block
    matches what a live eval would have written.
    """
    pms = [r.get("path_metrics") for r in samples]
    pms = [p for p in pms if p and p.get("num_parsed", 0) > 0]
    if not pms:
        return {}

    def _mean(key):
        vals = [p[key] for p in pms if p.get(key) is not None]
        return (sum(vals) / len(vals)) if vals else None

    def _rate(key):
        return sum(1 for p in pms if p.get(key)) / len(pms)

    agg = {
        "edge_validity_rate": _mean("edge_validity_rate"),
        "nodes_exist_rate": _mean("nodes_exist_rate"),
        "full_path_valid_rate": _rate("full_path_valid"),
        "start_goal_ok_rate": _rate("start_goal_ok"),
        "cost_optimality": _mean("cost_optimality"),
        "num_with_path": len(pms),
    }
    structured = [p for p in pms if p.get("structured")]
    if structured:
        def _srate(key):
            return sum(1 for p in structured if p.get(key)) / len(structured)
        agg.update({
            "structured_pass_rate": _srate("structured_correct"),
            "waypoints_ok_rate": _srate("waypoints_ok"),
            "avoid_ok_rate": _srate("avoid_ok"),
            "required_edges_rate": _srate("required_edges_present"),
            "num_structured": len(structured),
        })
    judged = [p["llm_judge_pass"] for p in pms if p.get("llm_judge_pass") is not None]
    if judged:
        agg["llm_judge_accuracy"] = sum(judged) / len(judged)
    return agg


def _reaggregate(result: dict) -> None:
    """Refresh the run-level subjective / path-metric aggregates in place.

    Objective aggregates (``accuracy``, ``num_correct`` …) are left as the run
    recorded them; only the judge-derived summary fields are (re)written.
    """
    samples = result.get("samples", [])
    judged = [r for r in samples if r.get("llm_judge_pass") is not None]
    num_judged = len(judged)
    result["subjective_accuracy"] = (
        sum(1 for r in judged if r["llm_judge_pass"]) / num_judged if num_judged else None
    )
    result["num_judged"] = num_judged
    result["num_false_pos"] = sum(1 for r in samples if r.get("false_positive"))
    result["num_false_neg"] = sum(1 for r in samples if r.get("false_negative"))
    result["path_metrics"] = _aggregate_path_metrics(samples)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", help="Result JSON file, directory, or glob to re-grade.")
    ap.add_argument("--dataset", action="append", default=[],
                    help="Source populated-graph file/dir/glob (repeatable). "
                         "Default: each result's recorded 'path'/'eval_data'.")
    ap.add_argument("--in-place", action="store_true",
                    help="Overwrite the result files. Default writes a sibling copy.")
    ap.add_argument("--suffix", default=".judged",
                    help="Filename suffix for non-in-place output (default: .judged).")
    ap.add_argument("--model", default=None,
                    help="Judge HF model id (sets GREP_JUDGE_MODEL before loading).")
    args = ap.parse_args()

    if args.model:
        os.environ["GREP_JUDGE_MODEL"] = args.model

    # Deferred so --model takes effect before the module reads the env, and so
    # --help stays fast. Only path_validator is needed (networkx + lazy Gemma);
    # the heavy spine/model stack is intentionally NOT imported.
    from prism.eval import path_validator

    print(f"[apply-judge] judge model: {path_validator.GEMMA_JUDGE_MODEL}")

    result_files = _resolve(args.results)
    # A global --dataset is indexed once; otherwise each result resolves its own.
    shared_index = _dataset_index(args.dataset) if args.dataset else None

    grand = {"files": 0, "samples": 0, "matched": 0, "judged": 0, "ac": 0,
             "fp": 0, "fn": 0, "unmatched": 0}

    for rf in result_files:
        with open(rf) as f:
            result = json.load(f)
        samples = result.get("samples")
        if not samples:
            print(f"[apply-judge] {rf}: no 'samples' — skipped.")
            continue

        if shared_index is not None:
            index = shared_index
        else:
            ds_path = result.get("path") or result.get("eval_data")
            if not ds_path or not os.path.exists(ds_path):
                print(f"[apply-judge] {rf}: source dataset '{ds_path}' not found; "
                      f"pass --dataset to point at it. Skipped.")
                continue
            index = _dataset_index([ds_path])

        result_stem = os.path.splitext(os.path.basename(
            result.get("name") or os.path.basename(rf)))[0]

        n_match = n_judge = n_ac = n_fp = n_fn = n_unmatched = 0
        for sample in samples:
            picked = _pick_task(index, sample, result_stem)
            if picked is None:
                n_unmatched += 1
                continue
            graph_dict, task = picked
            st = _regrade_sample(sample, graph_dict, task, path_validator)
            n_match += 1
            n_ac += int(st["ac"])
            n_judge += int(st["judged"])
            n_fp += int(st["false_positive"])
            n_fn += int(st["false_negative"])

        _reaggregate(result)

        out = rf if args.in_place else (
            f"{os.path.splitext(rf)[0]}{args.suffix}{os.path.splitext(rf)[1]}")
        with open(out, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        subj = result.get("subjective_accuracy")
        subj_s = f"{subj:.1%}" if subj is not None else "n/a"
        warn = ""
        if n_ac and n_judge == 0:
            warn = "  ⚠ judge produced 0 verdicts (weights/auth still unavailable?)"
        if n_unmatched:
            warn += f"  ⚠ {n_unmatched} sample(s) unmatched to a dataset task"
        print(f"[apply-judge] {os.path.basename(rf)}: matched {n_match}/{len(samples)}, "
              f"ac={n_ac}, judged={n_judge}, FP={n_fp}, FN={n_fn}, SubjAcc={subj_s} "
              f"-> {out}{warn}")

        grand["files"] += 1
        grand["samples"] += len(samples)
        grand["matched"] += n_match
        grand["judged"] += n_judge
        grand["ac"] += n_ac
        grand["fp"] += n_fp
        grand["fn"] += n_fn
        grand["unmatched"] += n_unmatched

    print(f"\n[apply-judge] DONE  files={grand['files']}  samples={grand['samples']}  "
          f"matched={grand['matched']}  ac={grand['ac']}  judged={grand['judged']}  "
          f"FP={grand['fp']}  FN={grand['fn']}  unmatched={grand['unmatched']}")
    if grand["ac"] and grand["judged"] == 0:
        print("[apply-judge] WARNING: acceptance_criterion tasks were present but the "
              "judge scored none — check that the Gemma weights / HF auth are available "
              "(see path_validator.GEMMA_JUDGE_MODEL).")


if __name__ == "__main__":
    main()
