#!/usr/bin/env python3
"""Format offline-generated planner responses into the eval-result JSON schema.

The data generator wrote one chat-message file per (graph, task) under
``generated_plans/sample_<GGG>_<TTT>.json`` (``GGG`` = graph index, matching
``populated_graphs/data_gen_<GGG>.json``; ``TTT`` = task index within that
graph). Each file is the full few-shot prompt; its LAST assistant message is the
model's response to that graph's task ``TTT``.

This script turns those raw responses into the same result format that a live
``prism.eval`` run writes (e.g. ``results/92mpnd3s_no_spine/data_gen_005.json``):
one ``results/<out>/data_gen_<GGG>.json`` per graph, with a ``samples`` list of
10 graded samples.

No planner model is loaded — each recorded response is *replayed* through the
real SPINE parser + ``PlanningSim`` + ``path_validator`` grading stack via a
stub client, so ``response``, ``interaction_trace``, ``terminated_by``,
``path_metrics`` (and ``gemma_regrade``) are produced by the exact same code a
live eval uses. The Gemma judge / regrade honor the usual env switches:
``GREP_JUDGE`` (judge on by default), ``GREP_GEMMA_REGRADE`` (adds the
``gemma_regrade`` block, off by default), ``GREP_PATH_RESCUE``, and
``PRISM_DISABLE_SPINE_TOOLS``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import traceback as traceback_mod
from dataclasses import asdict

# --- make the repo importable when run as a plain script ---------------------
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "src"),
           os.path.join(os.path.dirname(_REPO), "SPINE", "src")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from prism.eval import evaluate            # noqa: E402  (also installs SPINE prompt patch)
from prism.eval import path_validator      # noqa: E402
from prism.data import graph_sim, planning_sim  # noqa: E402
from spine import spine                    # noqa: E402
from spine.mapping import graph_util       # noqa: E402

_DEFAULT_GEN = ("data_store/revised/gen/nav100_n30_gemma_data")
_SAMPLE_RE = re.compile(r"sample_(\d{3})_(\d{3})\.json$")


class _ReplayClient:
    """Stands in for ``inference.InMemoryLLM``: returns a fixed assistant message.

    SPINE only ever calls ``client.query_llm(msg) -> (text, success)``; the
    ``msg`` (prompt) is irrelevant here because the response is pre-generated.
    """

    def __init__(self, content: str):
        self._content = content

    def query_llm(self, msg):
        return self._content, True


def _final_assistant_content(messages) -> str:
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "assistant":
            return m.get("content", "")
    return ""


def _final_user_task(messages) -> str:
    """The last ``task: ...`` request (sanity check only).

    Skips trailing ``Feedback:``/``updates:`` turns so a JSON-retry exchange at
    the end of a file doesn't get mistaken for the task.
    """
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "user":
            mt = re.match(r"\s*task:\s*(.+)", m.get("content", ""), re.DOTALL)
            if mt:
                return mt.group(1).strip()
    return ""


def grade_graph(eval_samples, contents, *, use_icl: bool):
    """Replay+grade one graph's samples — mirrors ``evaluate.eval_model_single_graph``
    but with a replay client instead of a live model. Returns ``(accuracy, samples)``.
    """
    graph_handler = graph_util.GraphHandler("")
    graph_sim_cls = (evaluate._NoToolsGraphSim
                     if evaluate._spine_tools_disabled() else graph_sim.GraphSim)
    graph_simulation = graph_sim_cls(graph_handler)
    planner_sim = planning_sim.PlanningSim(debug=False)

    total_correct = 0
    sample_results = []

    for i, (es, content) in enumerate(zip(eval_samples, contents)):
        planner_response = None
        planning_result = None
        try:
            client = _ReplayClient(content)
            llm_planner = spine.SPINE(
                graph=graph_simulation.partial_graph, client=client, use_icl=use_icl)
            graph_simulation.reset(graph_as_dict=es.graph, current_location=es.init_node)
            llm_planner.graph = graph_simulation.partial_graph

            planning_result = planner_sim.run_planning(
                llm_planner=llm_planner, task=es.task,
                graph_data_gen=graph_simulation, max_iterations=10)
            planner_response = planning_result.response

            result, _ = evaluate._construct_eval_result(planner_response, es.answer)
            pm = evaluate._sample_path_metrics(planner_response, es)
            structured = bool(pm and pm.get("structured"))
            judge_pass = (pm or {}).get("llm_judge_pass")
            if structured:
                sc = bool(pm.get("structured_correct"))
                v = {"objective_correct": sc, "objective_keyword": sc,
                     "subjective_correct": None, "false_positive": False,
                     "false_negative": False, "judged": False}
            else:
                v = path_validator.combine_verdict(
                    regex_correct=result.is_correct(),
                    regex_keyword=result.plan_keyword,
                    judge_pass=judge_pass,
                    acceptance_criterion_present=bool(es.acceptance_criterion),
                )

            sample_dict = {
                "graph_name": es.graph_name,
                "idx": i,
                "task": es.task,
                "answer_key": es.answer,
                "response": planner_response,
                "interaction_trace": [asdict(s) for s in planning_result.trace] if planning_result else [],
                "terminated_by": planning_result.terminated_by if planning_result else None,
                "formatted": result.formatted,
                "plan_keyword": result.plan_keyword,
                "correct": v["objective_correct"],
                "structured": structured,
                "subjective_correct": v["subjective_correct"],
                "false_positive": v["false_positive"],
                "false_negative": v["false_negative"],
                "llm_judge_pass": judge_pass,
                "error": None,
                "traceback": None,
                "path_metrics": pm,
            }
            if evaluate._gemma_regrade_enabled():
                sample_dict["gemma_regrade"] = evaluate._gemma_regrade_block(
                    planner_response, es, result)
            sample_results.append(sample_dict)
            total_correct += int(v["objective_keyword"])

        except Exception as e:  # match the live crash branch's shape
            tb_str = traceback_mod.format_exc()
            print(f"  [crash] {es.graph_name} idx {i}: {type(e).__name__}: {e}")
            sample_results.append({
                "graph_name": es.graph_name,
                "idx": i,
                "task": es.task,
                "answer_key": es.answer,
                "response": None,
                "interaction_trace": [asdict(s) for s in planning_result.trace] if planning_result else [],
                "terminated_by": planning_result.terminated_by if planning_result else "exception",
                "formatted": False,
                "plan_keyword": False,
                "correct": False,
                "structured": False,
                "subjective_correct": None,
                "false_positive": False,
                "false_negative": False,
                "llm_judge_pass": None,
                "error": f"{type(e).__name__}: {e}",
                "traceback": tb_str,
                "path_metrics": None,
            })

    accuracy = total_correct / len(eval_samples) if eval_samples else 0.0
    return accuracy, sample_results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-dir", default=_DEFAULT_GEN,
                    help="Dataset dir holding populated_graphs/ and generated_plans/ "
                         f"(default: {_DEFAULT_GEN})")
    ap.add_argument("--out-dir", default="results/gemma_31b_30n",
                    help="Where to write data_gen_<GGG>.json (default: results/gemma_31b_30n)")
    ap.add_argument("--checkpoint", default=None,
                    help="Value for the result's 'checkpoint' field "
                         "(default: the generated_plans dir).")
    ap.add_argument("--architecture", default="llm",
                    help="Value for the 'architecture' field (default: llm).")
    ap.add_argument("--text-edge-list", default="present",
                    help="Value for the 'text_edge_list' field (default: present).")
    ap.add_argument("--use-icl", action="store_true",
                    help="Set use_icl on the SPINE replay planner (cosmetic; the "
                         "prompt is ignored on replay).")
    args = ap.parse_args()

    gen_dir = args.gen_dir if os.path.isabs(args.gen_dir) else os.path.join(_REPO, args.gen_dir)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(_REPO, args.out_dir)
    graphs_dir = os.path.join(gen_dir, "populated_graphs")
    plans_dir = os.path.join(gen_dir, "generated_plans")
    os.makedirs(out_dir, exist_ok=True)

    checkpoint = args.checkpoint or plans_dir

    graph_files = sorted(glob.glob(os.path.join(graphs_dir, "data_gen_*.json")))
    if not graph_files:
        sys.exit(f"No data_gen_*.json under {graphs_dir}")

    print(f"gen-dir : {gen_dir}")
    print(f"out-dir : {out_dir}")
    print(f"graphs  : {len(graph_files)}")
    if evaluate._gemma_regrade_enabled():
        print("GREP_GEMMA_REGRADE=on -> samples will carry a gemma_regrade block")
    print()

    grand_total = grand_correct = 0
    missing = 0

    for gf in graph_files:
        stem = os.path.splitext(os.path.basename(gf))[0]          # data_gen_005
        ggg = stem.rsplit("_", 1)[-1]                              # 005
        with open(gf) as f:
            data = json.load(f)
        graph_dict = data["graph"]
        tasks = data["tasks"]
        eval_samples = evaluate.construct_eval_samples_from_dict(graph_dict, tasks, stem)

        contents, kept = [], []
        for i, es in enumerate(eval_samples):
            sp = os.path.join(plans_dir, f"sample_{ggg}_{i:03d}.json")
            if not os.path.exists(sp):
                print(f"  [missing] {os.path.basename(sp)} — skipping task {i}")
                missing += 1
                continue
            with open(sp) as f:
                messages = json.load(f)
            recorded_task = _final_user_task(messages)
            if recorded_task and recorded_task[:40] != es.task[:40]:
                print(f"  [warn] {os.path.basename(sp)} task mismatch:\n"
                      f"         file : {recorded_task[:70]!r}\n"
                      f"         graph: {es.task[:70]!r}")
            contents.append(_final_assistant_content(messages))
            kept.append(es)

        if not kept:
            print(f"{stem}: no samples found — skipped")
            continue

        accuracy, samples = grade_graph(kept, contents, use_icl=args.use_icl)
        num_correct = sum(1 for s in samples if s.get("correct"))

        log_data = {
            "checkpoint": checkpoint,
            "graph_file": os.path.relpath(gf, _REPO),
            "architecture": args.architecture,
            "text_edge_list": args.text_edge_list,
            "accuracy": accuracy,
            "num_samples": len(samples),
            "num_correct": num_correct,
            "samples": samples,
        }
        out_file = os.path.join(out_dir, f"{stem}.json")
        with open(out_file, "w") as f:
            json.dump(log_data, f, indent=2, default=str)
        print(f"{stem}: {num_correct}/{len(samples)} correct ({accuracy:.1%}) -> {out_file}")

        grand_total += len(samples)
        grand_correct += num_correct

    print(f"\nDone. {grand_correct}/{grand_total} correct across {len(graph_files)} graphs"
          + (f"; {missing} task file(s) missing." if missing else "."))


if __name__ == "__main__":
    main()
