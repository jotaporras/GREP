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
  * When the recorded ``plan`` carries no route, the route is searched in the
    model's full *reasoning* too (``path_from_reasoning``), taking the path it
    commits to LAST. If that also finds nothing, the Gemma judge rewrites the
    route in ``a -> b -> c`` notation — biased toward the planner's final route —
    and the NetworkX diagnostics are re-run on it (``path_rescued``). This is the
    same one-way reasoning-check + rescue a live ``evaluate.py`` run applies; the
    Gemma rescue is disabled with ``GREP_PATH_RESCUE=0``.
  * For acceptance_criterion / yes-no tasks it then runs the Gemma judge
    (``path_validator.judge_acceptance``) and folds the verdict in with
    ``path_validator.combine_verdict`` — exactly as
    ``evaluate.eval_model_single_graph`` does on a live run.

With ``--gemma-regrade`` the script ALSO writes a second, parallel reading
(``*{gemma-suffix}.json``, default ``*.gemma.json``) in which the Gemma judge
recovers each sample's intended route from the full response and EVERY path /
structured metric and the objective ``correct`` are regraded on that recovered
route — not just when the regex found nothing. ``formatted`` / ``plan_keyword``
still describe the raw response (they are not path-derived); each ``path_metrics``
block is stamped with ``path_source`` (``gemma_judge`` / ``regex_fallback``) and
the raw ``gemma_route``, and the result carries ``"reading": "gemma_path_regrade"``.
The original regex/reasoning ``*{suffix}.json`` reading is left intact alongside.

The output is byte-for-byte what a live ``evaluate.py`` run on the current dataset
would have written, given the recorded planner responses. Every per-sample verdict
and run-level metric is recomputed with the **same functions and inputs** the live
library uses — ``_construct_eval_result`` (JSON-shape + keyword regex),
``path_validator.evaluate_sample`` (recreate path → NetworkX grade → Gemma judge),
the structured / ``combine_verdict`` branch of ``eval_model_single_graph``, and the
aggregates of ``eval_model_multiple_graphs``:

  * per sample — ``formatted``, ``plan_keyword``, ``correct`` (objective verdict),
    ``structured``, ``path_metrics``, ``llm_judge_pass``, ``subjective_correct``,
    ``false_positive``, ``false_negative`` (and ``answer_key`` re-stamped from the
    dataset regex, exactly as the library does);
  * run — ``num_total``, ``num_correct``, ``accuracy`` (objective keyword rate),
    ``num_formatted``, ``num_keyword``, ``num_errors``, ``subjective_accuracy``,
    ``num_judged``, ``num_false_pos``, ``num_false_neg``, ``path_metrics``.

Only what cannot be recomputed without re-running the model is taken verbatim from
the recording: the planner ``response``, ``interaction_trace``, ``terminated_by``,
``error``/``traceback``, and run metadata (``name``, ``path``, ``permutation``,
``elapsed_s`` …). Hard-crash samples (``error`` set — the library's ``except``
branch) keep their objective-False verdict and are not graded.

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
import copy
import glob
import json
import os
import re
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


def _path_metrics(planner_response, graph_dict, task, path_validator,
                  plan_text_override=None):
    """Recreate + grade the planner's path via ``path_validator.evaluate_sample``.

    Mirrors ``evaluate._sample_path_metrics``: the route is pulled from the
    recorded ``plan`` field; the full response is passed for the judge. Never
    raises — returns ``None`` on failure.

    ``plan_text_override`` (the Gemma-recovered route, for the ``--gemma-regrade``
    reading) replaces the plan text as the path source, so every path/structured
    metric is graded on that route instead of the recorded plan. The full response
    is still passed unchanged for the judge and the reasoning fallback.
    """
    try:
        if plan_text_override is not None:
            plan_text = plan_text_override
        else:
            plan = planner_response.get("plan") if isinstance(planner_response, dict) else planner_response
            plan_text = "" if plan is None else str(plan)
        return path_validator.evaluate_sample(
            task["task"],
            plan_text,
            graph_dict,
            init_node=task.get("init_node"),
            acceptance_criterion=task.get("acceptance_criterion"),
            answer=task.get("answer"),
            full_response="" if planner_response is None else str(planner_response),
        )
    except Exception as e:
        print(f"[apply-judge] path-metric computation failed: {type(e).__name__}: {e}")
        return None


def _gemma_recover_path(planner_response, graph_dict, task, path_validator):
    """The route the Gemma judge recovers from the FULL recorded response.

    Unlike the live one-way rescue (which fires only when the regex finds nothing),
    this asks the judge to recreate the planner's intended route in ``a -> b -> c``
    notation for EVERY sample, so the ``--gemma-regrade`` reading can grade the
    whole response on the judge's path. Returns "" on any failure / no route.
    """
    try:
        goal, *_ = path_validator.derive_targets(
            graph_dict, init_node=task.get("init_node"), answer=task.get("answer"),
            criterion=task.get("acceptance_criterion"), task=task.get("task"))
        nodes = {n["name"] for n in (*graph_dict.get("regions", []),
                                     *graph_dict.get("objects", []))}
        full = "" if planner_response is None else str(planner_response)
        return path_validator.write_path_with_judge(
            full, nodes, task=task.get("task"),
            start=task.get("init_node"), goal=goal) or ""
    except Exception as e:
        print(f"[apply-judge] gemma path recovery failed: {type(e).__name__}: {e}")
        return ""


def _eval_result(parsed_answer, answer_key):
    """``(formatted, plan_keyword)`` — a faithful copy of
    ``evaluate._construct_eval_result``'s scoring.

    ``formatted`` is True iff the planner answer carries all four required keys;
    ``plan_keyword`` is True iff the dataset ``answer`` regex matches the
    stringified ``plan``. Any error (None / malformed response, bad regex) yields
    ``(False, False)`` — and, exactly as the library, a failure in the keyword
    search also discards ``formatted`` (both are returned from the same ``try``).
    """
    try:
        formatted = all(k in parsed_answer for k in
                        ("primary_goal", "relevant_graph", "reasoning", "plan"))
        plan_keyword = bool(re.search(answer_key, str(parsed_answer["plan"]), re.IGNORECASE))
        return formatted, plan_keyword
    except Exception:
        return False, False


def _regrade_sample(sample: dict, graph_dict, task, path_validator,
                    *, gemma_path: bool = False) -> dict:
    """Recompute every per-sample verdict in place, identically to a live run.

    Reproduces ``eval_model_single_graph``'s per-sample logic exactly: re-derive
    ``formatted``/``plan_keyword`` against the current dataset regex
    (``_construct_eval_result``), recreate + grade the path and run the judge
    (``path_validator.evaluate_sample``), then take the structured verdict or
    ``combine_verdict`` for the objective ``correct`` and the separate subjective
    columns. Nothing objective is reused from the recording — it is all rederived.

    With ``gemma_path=True`` (the ``--gemma-regrade`` reading) the Gemma judge
    recovers the planner's intended route from the full response, and ALL path /
    structured metrics and the objective ``correct`` are graded on that route
    instead of the recorded plan. ``formatted`` / ``plan_keyword`` still describe
    the raw response (they are not path-derived). ``path_metrics`` is stamped with
    ``path_source`` (``gemma_judge`` / ``regex_fallback``) and ``gemma_route``.
    """
    ac_present = bool(task.get("acceptance_criterion"))
    stats = {"judged": False, "ac": ac_present, "structured": False,
             "judge_eligible": False, "judge_fallback": False,
             "false_positive": False, "false_negative": False,
             "path_rescued": False, "path_from_reasoning": False,
             "gemma_route": False}

    if sample.get("error") is not None:
        # Hard crash — eval_model_single_graph's `except` branch: objective verdict
        # False, no path metrics, no judge. (A None response WITHOUT an error is
        # NOT a crash; the library still grades it below with response_text="".)
        sample["formatted"] = False
        sample["plan_keyword"] = False
        sample["correct"] = False
        sample["structured"] = False
        sample["path_metrics"] = None
        sample["llm_judge_pass"] = None
        sample["subjective_correct"] = None
        sample["false_positive"] = False
        sample["false_negative"] = False
        return stats

    answer = task.get("answer")
    planner_response = sample.get("response")
    # _construct_eval_result against the CURRENT dataset answer regex (the
    # `answer_key` is re-stamped to it, as the library stores eval_sample.answer).
    formatted, plan_keyword = _eval_result(planner_response, answer)
    sample["answer_key"] = answer
    sample["formatted"] = formatted
    sample["plan_keyword"] = plan_keyword

    # --gemma-regrade: recover the route with the judge and grade the path on it.
    # Only override when the recovered route actually parses to a path; otherwise
    # fall back to the normal plan/reasoning grading so a NONE/empty reply can't
    # erase a route the plan really carried.
    plan_override = None
    gemma_route = ""
    if gemma_path:
        gemma_route = _gemma_recover_path(planner_response, graph_dict, task, path_validator)
        if gemma_route and path_validator.parse_path(gemma_route, prefer_last=True):
            plan_override = gemma_route
            stats["gemma_route"] = True

    pm = _path_metrics(planner_response, graph_dict, task, path_validator,
                       plan_text_override=plan_override)
    if pm is not None and gemma_path:
        pm["path_source"] = "gemma_judge" if plan_override is not None else "regex_fallback"
        pm["gemma_route"] = gemma_route
    structured = bool(pm and pm.get("structured"))
    judge_pass = (pm or {}).get("llm_judge_pass")

    if structured:
        # Structural task: objective verdict is the deterministic NetworkX edge/path
        # check; the Gemma judge is not run and there is no subjective column.
        objective_correct = bool(pm.get("structured_correct"))
        subjective_correct = None
        false_positive = false_negative = False
    else:
        # Non-structural: combine_verdict's two disjoint scores. objective_correct
        # is the RegEx is_correct (formatted AND keyword); the judge moves only the
        # subjective column.
        v = path_validator.combine_verdict(
            regex_correct=formatted and plan_keyword,
            regex_keyword=plan_keyword,
            judge_pass=judge_pass,
            acceptance_criterion_present=ac_present,
        )
        objective_correct = v["objective_correct"]
        subjective_correct = v["subjective_correct"]
        false_positive = v["false_positive"]
        false_negative = v["false_negative"]

    sample["path_metrics"] = pm
    sample["structured"] = structured
    sample["correct"] = objective_correct
    sample["llm_judge_pass"] = judge_pass
    sample["subjective_correct"] = subjective_correct
    sample["false_positive"] = false_positive
    sample["false_negative"] = false_negative

    stats["structured"] = structured
    # Judge-eligible = a non-structural task the judge SHOULD run on (AC or yes/no;
    # ``judge_used`` is evaluate_sample's own should_judge flag). Structural tasks
    # are graded deterministically by NetworkX and never reach the judge.
    stats["judge_eligible"] = (not structured) and (
        bool(pm and pm.get("judge_used")) or ac_present)
    # Mirrors evaluate.eval_model_single_graph's n_judge_fallback: a non-structural
    # AC task the judge could not score (model unavailable / returned no verdict).
    stats["judge_fallback"] = (not structured) and ac_present and judge_pass is None
    stats["judged"] = judge_pass is not None
    stats["false_positive"] = false_positive
    stats["false_negative"] = false_negative
    stats["path_rescued"] = bool(pm and pm.get("path_rescued"))
    stats["path_from_reasoning"] = bool(pm and pm.get("path_from_reasoning"))
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
        # Routes the plan didn't carry but the model stated in its reasoning
        # (recovered deterministically by regex, no model call).
        "num_from_reasoning": sum(1 for p in pms if p.get("path_from_reasoning")),
        # Routes recovered by the Gemma path rescue (regex found none in plan or
        # reasoning; judge rewrote it in `a -> b -> c` and NetworkX re-graded it).
        "num_rescued": sum(1 for p in pms if p.get("path_rescued")),
    }
    # Only the --gemma-regrade reading stamps path_source; add this field there so
    # the standard reading stays byte-for-byte like a live evaluate.py run.
    if any(p.get("path_source") for p in pms):
        agg["num_gemma_path"] = sum(1 for p in pms if p.get("path_source") == "gemma_judge")
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


def _objective_keyword(sample: dict) -> bool:
    """The per-sample ``objective_keyword`` that feeds ``accuracy``.

    Matches ``eval_model_single_graph``'s ``total_correct += int(v["objective_keyword"])``:
    for a structural task it is ``structured_correct`` (== the recomputed
    ``correct``); otherwise it is the RegEx ``plan_keyword``.
    """
    if sample.get("structured"):
        return bool(sample.get("correct"))
    return bool(sample.get("plan_keyword"))


def _reaggregate(result: dict) -> None:
    """Recompute ALL run-level metrics in place, identically to the live driver.

    Objective aggregates mirror ``eval_model_multiple_graphs`` (and ``accuracy``
    = objective-keyword rate from ``eval_model_single_graph``); subjective and
    path aggregates mirror the same. Run metadata (``name``, ``permutation``,
    ``elapsed_s`` …) is left untouched — it is not a metric.
    """
    samples = result.get("samples", [])
    n = len(samples)
    # Objective (RegEx/NetworkX) — read only RegEx/structured fields.
    result["num_total"] = n
    result["num_correct"] = sum(1 for r in samples if r.get("correct"))
    result["num_formatted"] = sum(1 for r in samples if r.get("formatted"))
    result["num_keyword"] = sum(1 for r in samples if r.get("plan_keyword"))
    result["num_errors"] = sum(1 for r in samples if r.get("error") is not None)
    result["accuracy"] = (sum(1 for r in samples if _objective_keyword(r)) / n) if n else 0.0
    # Subjective (Gemma judge) — judge verdict only, over judged samples only.
    judged = [r for r in samples if r.get("llm_judge_pass") is not None]
    num_judged = len(judged)
    result["subjective_accuracy"] = (
        sum(1 for r in judged if r["llm_judge_pass"]) / num_judged if num_judged else None
    )
    result["num_judged"] = num_judged
    result["num_false_pos"] = sum(1 for r in samples if r.get("false_positive"))
    result["num_false_neg"] = sum(1 for r in samples if r.get("false_negative"))
    result["path_metrics"] = _aggregate_path_metrics(samples)


def _process_result(result, index, result_stem, path_validator, *, gemma_path=False) -> dict:
    """Regrade every matched sample of one result in place, then reaggregate.

    The single per-file engine used for BOTH readings: the standard regex/reasoning
    grade (``gemma_path=False``) and the ``--gemma-regrade`` reading
    (``gemma_path=True``, every path graded on the judge-recovered route). Returns
    the per-file counters used for the summary line.
    """
    c = dict(match=0, judge=0, ac=0, fp=0, fn=0, unmatched=0, struct=0,
             eligible=0, fallback=0, rescued=0, reasoned=0, gemma_routes=0)
    for sample in result.get("samples", []):
        picked = _pick_task(index, sample, result_stem)
        if picked is None:
            c["unmatched"] += 1
            continue
        graph_dict, task = picked
        st = _regrade_sample(sample, graph_dict, task, path_validator, gemma_path=gemma_path)
        c["match"] += 1
        c["ac"] += int(st["ac"]); c["judge"] += int(st["judged"])
        c["struct"] += int(st["structured"]); c["eligible"] += int(st["judge_eligible"])
        c["fallback"] += int(st["judge_fallback"]); c["fp"] += int(st["false_positive"])
        c["fn"] += int(st["false_negative"]); c["rescued"] += int(st["path_rescued"])
        c["reasoned"] += int(st["path_from_reasoning"])
        c["gemma_routes"] += int(st.get("gemma_route", False))
    _reaggregate(result)
    return c


def _print_file_summary(rf_base, n_samples, c, result, out, *, gemma=False):
    """One per-file summary line, shared by both readings."""
    subj = result.get("subjective_accuracy")
    subj_s = f"{subj:.1%}" if subj is not None else "n/a"
    warn = ""
    if c["fallback"]:
        warn += (f"  ⚠ {c['fallback']} judge-eligible task(s) scored no verdict — "
                 f"check Gemma weights / HF auth")
    elif c["eligible"] == 0 and c["struct"] and not gemma:
        warn += (f"  (all {c['struct']} structural; NetworkX-graded, judge not used "
                 f"for scoring by design — Gemma loads only if a path rescue fires)")
    if c["unmatched"]:
        warn += f"  ⚠ {c['unmatched']} sample(s) unmatched to a dataset task"
    path_bits = (f"gemma_path={c['gemma_routes']}" if gemma
                 else f"path_from_reasoning={c['reasoned']}, path_rescued={c['rescued']}")
    tag = "[apply-judge|gemma]" if gemma else "[apply-judge]"
    print(f"{tag} {rf_base}: matched {c['match']}/{n_samples}, "
          f"acc={result.get('accuracy', 0.0):.0%} "
          f"(correct={result.get('num_correct', 0)}/{result.get('num_total', 0)}), "
          f"ac={c['ac']}, structural={c['struct']}, judge_eligible={c['eligible']}, "
          f"judged={c['judge']}, {path_bits}, FP={c['fp']}, FN={c['fn']}, "
          f"SubjAcc={subj_s} -> {out}{warn}")


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
    ap.add_argument("--gemma-regrade", action="store_true",
                    help="Also emit a SECOND reading that regrades every sample's path "
                         "metrics and correctness on the route the Gemma judge recovers "
                         "from the full response (written with --gemma-suffix).")
    ap.add_argument("--gemma-suffix", default=".gemma",
                    help="Filename suffix for the --gemma-regrade reading (default: .gemma).")
    args = ap.parse_args()

    if args.model:
        os.environ["GREP_JUDGE_MODEL"] = args.model

    # Deferred so --model takes effect before the module reads the env, and so
    # --help stays fast. Only path_validator is needed (networkx + lazy Gemma);
    # the heavy spine/model stack is intentionally NOT imported.
    from prism.eval import path_validator

    print(f"[apply-judge] judge model: {path_validator.GEMMA_JUDGE_MODEL}")

    result_files = _resolve(args.results)
    # Never re-grade our own outputs. A directory/glob sweep would otherwise pick up
    # the *{suffix}.json files this script writes and re-run the Gemma judge on them
    # every invocation (and pile up *.judged.judged.json) — an effective infinite
    # loop. Drop them, UNLESS the user explicitly named a single judged file.
    own_suffixes = (f"{args.suffix}.json", f"{args.gemma_suffix}.json")
    explicit_file = os.path.isfile(args.results) and not any(
        ch in args.results for ch in ("*", "?", "["))
    if not explicit_file:
        kept = [f for f in result_files if not f.endswith(own_suffixes)]
        skipped = len(result_files) - len(kept)
        if skipped:
            print(f"[apply-judge] skipping {skipped} already-graded "
                  f"'*{own_suffixes[0]}' / '*{own_suffixes[1]}' file(s) "
                  f"to avoid re-grading our own output.")
        result_files = kept
    if not result_files:
        raise SystemExit(f"No result files to grade at {args.results} "
                         f"(all were already-graded outputs of this script).")
    # A global --dataset is indexed once; otherwise each result resolves its own.
    shared_index = _dataset_index(args.dataset) if args.dataset else None

    grand = {"files": 0, "samples": 0, "matched": 0, "judged": 0, "ac": 0,
             "structured": 0, "eligible": 0, "fallback": 0,
             "fp": 0, "fn": 0, "unmatched": 0, "rescued": 0, "reasoned": 0,
             "gemma_routes": 0}

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

        # Snapshot the raw recording BEFORE the standard regrade mutates verdicts,
        # so the Gemma reading starts from the same recorded responses.
        raw = copy.deepcopy(result) if args.gemma_regrade else None

        c = _process_result(result, index, result_stem, path_validator)
        out = rf if args.in_place else (
            f"{os.path.splitext(rf)[0]}{args.suffix}{os.path.splitext(rf)[1]}")
        with open(out, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        _print_file_summary(os.path.basename(rf), len(samples), c, result, out)

        grand["files"] += 1
        grand["samples"] += len(samples)
        for gk, ck in (("matched", "match"), ("judged", "judge"), ("ac", "ac"),
                       ("structured", "struct"), ("eligible", "eligible"),
                       ("fallback", "fallback"), ("fp", "fp"), ("fn", "fn"),
                       ("unmatched", "unmatched"), ("rescued", "rescued"),
                       ("reasoned", "reasoned")):
            grand[gk] += c[ck]

        # Second reading: regrade the whole response on the route the Gemma judge
        # recovers, written to a separate *{gemma_suffix}.json file.
        if args.gemma_regrade:
            gc = _process_result(raw, index, result_stem, path_validator, gemma_path=True)
            raw["reading"] = "gemma_path_regrade"
            gout = f"{os.path.splitext(rf)[0]}{args.gemma_suffix}{os.path.splitext(rf)[1]}"
            with open(gout, "w") as f:
                json.dump(raw, f, indent=2, ensure_ascii=False)
            _print_file_summary(os.path.basename(rf), len(samples), gc, raw, gout, gemma=True)
            grand["gemma_routes"] += gc["gemma_routes"]

    gemma_bit = f"  gemma_path={grand['gemma_routes']}" if args.gemma_regrade else ""
    print(f"\n[apply-judge] DONE  files={grand['files']}  samples={grand['samples']}  "
          f"matched={grand['matched']}  ac={grand['ac']}  structural={grand['structured']}  "
          f"judge_eligible={grand['eligible']}  judged={grand['judged']}  "
          f"path_from_reasoning={grand['reasoned']}  path_rescued={grand['rescued']}{gemma_bit}  "
          f"FP={grand['fp']}  FN={grand['fn']}  unmatched={grand['unmatched']}")
    if grand["fallback"]:
        print(f"[apply-judge] WARNING: {grand['fallback']} non-structural acceptance_criterion "
              "task(s) could not be judged — check that the Gemma weights / HF auth are "
              "available (see path_validator.GEMMA_JUDGE_MODEL).")
    elif grand["eligible"] == 0 and grand["ac"]:
        print("[apply-judge] NOTE: every acceptance_criterion task is graph-structural "
              "(positionality / reachability / navigability). These are graded "
              "deterministically by NetworkX and DO NOT use the LLM judge — so the Gemma "
              "model is intentionally never loaded, exactly as a live evaluate.py run. "
              "judged=0 here is correct, not a failure.")


if __name__ == "__main__":
    main()
