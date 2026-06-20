"""Canonical eval library for PRISM planning eval.

Pure in-memory contract: callers pass an already-loaded `(model, tokenizer)`
and already-parsed graph/task dicts. File I/O and checkpoint loading live
in the experiment scripts and in `train_v2`.

No defaults on any policy argument. Every caller is expected to pass
`include_edge_list`, `use_icl`, and `permutation` explicitly. This is
deliberate — silent defaults inside the canonical library hide behavior
changes from the experiment scripts and the trainer.

Booleans inside the library, human-readable strings at the boundary: the
CLI / config layers keep `text_edge_list: "present" | "none"` for wandb
and stdout log readability; scripts convert to `include_edge_list: bool`
just before calling into here.

Imports follow the repo's qualified-module convention (`graph_util.GraphHandler`,
not `GraphHandler`).

Public surface:

- `EvalSample`                       — namedtuple `(task, answer, graph, init_node, graph_name)`.
- `GraphEvalResultSummary`           — per-graph aggregate result record.
- `construct_eval_samples_from_dict` — shape-conversion helper.
- `eval_model_single_graph`          — single-graph scoring loop.
- `eval_model_multiple_graphs`       — multi-graph driver with cross-cutting policy.
- `print_summary_table`              — stdout formatter over a list of results.
"""
from __future__ import annotations

import json
import os
import re
import time
import traceback as traceback_mod
from collections import namedtuple
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from spine import spine
from spine.mapping import graph_util
from spine.prompts import examples
import spine.prompts.prompts as spine_prompts

# Fix operator-precedence bug in SPINE's get_base_prompt_update_graph.
# TODO: This monkey-patch is import-time global state and should not live in
# the eval library long-term. It exists here for byte-compatible parity with
# the legacy `run_eval.py`; every current eval path inherited it transitively.
_orig_get_base_prompt = spine_prompts.get_base_prompt_update_graph


# R10: ICL is a 2/5 switch, and the e7/composite-graph model is trained with
# TWO in-context examples. SPINE's stock get_base_prompt uses five
# (EXAMPLE_1..EXAMPLE_5), which roughly triples the prompt length — and hence
# the composite-graph cycle length (one node per token) — versus training. With
# RoPE disabled (M8) the model is acutely sensitive to that distribution shift
# and degenerates into repeated-token output. Pin eval to the same two examples
# the model trained on so train and eval ICL counts match.
_ICL_EXAMPLES_2 = examples.EXAMPLE_1 + examples.EXAMPLE_2


def _fixed_get_base_prompt(request, scene_graph, use_icl=True):
    # use_icl=True -> exactly the 2 training ICL examples; use_icl=False -> 1
    # (EXAMPLE_1), preserving the prior minimal-prompt behavior for that flag.
    sys_prompt = spine_prompts.SYS_PROMPT
    # Eval-only system-prompt additions. The latent-connections note is always
    # present; the no-tool directive only when tool calling is disabled. Build a
    # fresh dict either way so the shared SYS_PROMPT object is never mutated.
    content = sys_prompt["content"] + _LATENT_CONNECTIONS_NOTE
    if _spine_tools_disabled():
        # Tell the model what `_NoToolsGraphSim` already enforces: the API actions
        # are a planning notation only — nothing executes, no feedback comes back.
        content += _NO_TOOL_CALL_DIRECTIVE
    sys_prompt = {"role": sys_prompt["role"], "content": content}
    header = [sys_prompt] + (_ICL_EXAMPLES_2 if use_icl else examples.EXAMPLE_1)
    return header + [
        {"role": "user",
         "content": f"{request}\nAdvice: \n- Recall the scene may be incomplete. \n- Carefully explain your reasoning in a step-by-step manner.\n- Reason over connections, coordinates, and semantic relationships between objects and regions in the scene.\n\n"
                    f"Scene graph:{scene_graph}"}
    ]


spine_prompts.get_base_prompt_update_graph = _fixed_get_base_prompt

from prism.models import gnn_llm
from prism.models import inference
from prism.models import utils as model_utils
from prism.data import graph_sim
from prism.data import planning_sim
from prism.eval import path_validator


# ----------------------------------------------------------------------------
# Tool-calling toggle
# ----------------------------------------------------------------------------
# SPINE's "tools" are the environment-interrogating actions the planner emits in
# its plan: they reveal new graph structure / descriptions and drive the
# explore->feedback loop. The terminal `answer` action is NOT a tool — it is
# detected by `PlanningSim.run_planning` itself, independent of `take_action`.
_SPINE_TOOL_ACTIONS = frozenset(
    {"map_region", "explore_region", "extend_map", "inspect", "goto"}
)


def _spine_tools_disabled() -> bool:
    """True when PRISM_DISABLE_SPINE_TOOLS opts the eval out of tool calling."""
    return os.environ.get("PRISM_DISABLE_SPINE_TOOLS", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _gemma_regrade_enabled() -> bool:
    """True when GREP_GEMMA_REGRADE asks for the second Gemma-path reading.

    When on, every sample additionally carries a ``gemma_regrade`` block: its path
    metrics and objective correctness graded on the route the Gemma judge recovers
    from the full response. The ORIGINAL regex/reasoning scores stay at the top
    level untouched — both readings are kept side by side.
    """
    return os.environ.get("GREP_GEMMA_REGRADE", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


# Appended to SYS_PROMPT (by `_fixed_get_base_prompt`) on EVERY eval, regardless
# of the tool toggle. GREP-PRISM injects the graph's connectivity into the
# model's latent space (the GNN/GREP pathway), so the planner can reason over
# relationships that may not be spelled out in the textual scene graph. This note
# makes that latent access explicit in the prompt.
_LATENT_CONNECTIONS_NOTE = (
    "\n\nNote: the graph's connections are available to you in latent space. You "
    "can reason over the relationships and paths between nodes from this latent "
    "access, even where they are not written out in the textual scene graph."
)


# Appended to SYS_PROMPT (by `_fixed_get_base_prompt`) only when tools are
# disabled. It reframes the API as a planning vocabulary so the model's prompt
# matches the runtime contract `_NoToolsGraphSim` enforces: no action executes
# and no observation ever returns. Written as advice in SPINE's own register
# (plan over the actions, then commit to answer()).
_NO_TOOL_CALL_DIRECTIVE = (
    "\n\nTOOL CALLING IS DISABLED FOR THIS TASK — READ CAREFULLY:\n"
    "- The API actions (goto, explore_region, map_region, inspect, extend_map) are "
    "available ONLY as a vocabulary to lay out and explain your plan. They are NOT "
    "executed.\n"
    "- You will receive NO updates, observations, descriptions, or feedback in "
    "response to any action. Never wait for, assume, or refer to a result from any "
    "step — there is none.\n"
    "- Treat the scene graph you are given as the complete and only information you "
    "will ever have. Reason over it directly; do not plan to discover anything new.\n"
    "- Still express your reasoning as a plan over these actions, then commit to a "
    "final answer(). Your answer() must stand entirely on its own, justified by the "
    "given graph alone and not by any executed step."
)


class _NoToolsGraphSim(graph_sim.GraphSim):
    """GraphSim that drops every SPINE tool action.

    Disabling tool calling without crippling the planner's ability to reason: a
    tool action becomes a no-op that reveals nothing and returns "no updates", so
    `run_planning` neither breaks early nor advances the observed graph. Because
    the loop keeps running until the planner emits `answer` (or hits
    `max_iterations`), the model retains its full multi-turn reasoning budget —
    it may take several turns and answer — it simply has no tools to call and must
    reason from the graph it was already given. `answer` falls through to the base
    implementation unchanged.
    """

    def take_action(self, action, argument) -> bool:
        if action in _SPINE_TOOL_ACTIONS:
            return False
        return super().take_action(action, argument)


# ----------------------------------------------------------------------------
# Public data structures
# ----------------------------------------------------------------------------

EvalSample = namedtuple(
    "EvalSample",
    ["task", "answer", "graph", "init_node", "graph_name", "acceptance_criterion"],
)
# acceptance_criterion is optional: only e6-style datasets carry it. When present
# it enables the M10 Gemma judge; otherwise M10 falls back to regex/NetworkX.
EvalSample.__new__.__defaults__ = (None,)
"""

Evaluation sample task specification: An eval task is given by a natural-languaget ask specification,
associated to a graph, and a starting node. 

`graph_name` is a stable identifier (typically the source-file stem, e.g.
"data_gen_004"). It is stamped onto every per-sample result dict so that
debugging tools can trace a sample back to its source graph without
relying on the surrounding output filename or metadata block.
"""


@dataclass
class GraphEvalResultSummary:
    """Aggregate results from evaluating one model over one graph's samples.

    Returned by `eval_model_multiple_graphs` (keyed by graph name). `samples`
    is the raw per-sample result list produced by `eval_model_single_graph`;
    the other fields are cached aggregates so the summary table doesn't have
    to re-walk it.
    """
    name: str
    num_total: int
    num_correct: int                       # objective (RegEx/NetworkX) correct count
    accuracy: float                        # objective accuracy (judge-free)
    subjective_accuracy: Optional[float]   # separate Gemma-judge accuracy over judged samples (None if none judged)
    num_judged: int                        # samples the judge actually scored
    num_formatted: int
    num_keyword: int
    num_false_pos: int                     # RegEx correct, judge rejected (lowers subjective)
    num_false_neg: int                     # RegEx wrong, judge accepted (raises subjective)
    num_errors: int
    elapsed_s: float
    n_nodes: int
    use_icl: bool
    permutation: Optional[dict]
    samples: List[dict] = field(default_factory=list)
    # M10 (R4) path-validity aggregates over this graph's samples. Empty when
    # no sample produced a parseable route.
    path_metrics: dict = field(default_factory=dict)
    # GREP_GEMMA_REGRADE second reading: objective accuracy / counts / path metrics
    # recomputed on the Gemma-recovered routes. None when the regrade is off — the
    # ORIGINAL aggregates above are always the headline.
    gemma: Optional[dict] = None


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------


def eval_model_single_graph(
    model,
    tokenizer,
    eval_samples: List[EvalSample],
    *,
    include_edge_list: bool,
    use_icl: bool,
    permutation,
) -> Tuple[float, List[Dict]]:
    """Run the planning-simulation loop over `eval_samples` related to the same grpah

    All samples in `eval_samples` are assumed to share the same underlying
    graph dict. For multi-graph evaluation use `eval_model_multiple_graphs`.

    Args:
        model:              Already-loaded planner model (plain LLM or
                            GraphAugmentedLLM, optionally PEFT-wrapped).
        tokenizer:          Matching tokenizer.
        eval_samples:       List of `EvalSample` (>= 1 element).
        include_edge_list:  True to include the textual edge list in the
                            planner prompt, False to strip it. Must match
                            how the checkpoint was trained. CLI / config
                            layers should keep this as the human-readable
                            string `"present"`/`"none"` for wandb and convert
                            at the boundary.
        use_icl:            True/False — required concrete bool. The library
                            does not delegate to SPINE's internal default;
                            callers must decide.
        permutation:        `prism.models.utils.Permutation` or `None`.
                            When set on a plain LLM the graph dict is
                            permuted per-sample; ignored for graph-augmented
                            models. `None` here is a real value
                            ("no permutation"), not a delegated default.

    Returns:
        `(accuracy, sample_results)` where `accuracy` is the OBJECTIVE
        RegEx/NetworkX keyword accuracy — computed from RegEx fields only, never
        from the judge. The separate Gemma judge score is carried per-sample
        (`subjective_correct` — None when unjudged, `false_positive`,
        `false_negative`) and aggregated into `subjective_accuracy` (over judged
        samples only, judge verdict only) by `eval_model_multiple_graphs`. The two
        scores share no inputs: `false_positive`/`false_negative` are diagnostics
        comparing them and feed neither. A sample the judge cannot score is omitted
        from the subjective accuracy (and warned about), never copied from RegEx.
        `sample_results` is a list of dicts with one entry per sample.
    """
    graph_handler = graph_util.GraphHandler("")
    graph_sim_cls = _NoToolsGraphSim if _spine_tools_disabled() else graph_sim.GraphSim
    graph_simulation = graph_sim_cls(graph_handler)

    strip_edges = not include_edge_list
    is_gnn = _is_graph_augmented(model)
    if is_gnn:
        client = inference.GraphAugmentedInMemoryLLM(
            model=model, 
            tokenizer=tokenizer,
            strip_edges=strip_edges, 
            permutation=permutation,
        )
    else:
        client = inference.InMemoryLLM(
            model=model, 
            tokenizer=tokenizer, 
            strip_edges=strip_edges,
        )


    llm_planner = spine.SPINE(
        graph=graph_simulation.partial_graph,
        client=client,
        use_icl=use_icl,
    )

    planner = planning_sim.PlanningSim(debug=False)

    total_correct = 0          # objective (RegEx/NetworkX) keyword marks -> headline accuracy
    n_judge_fallback = 0       # AC-tasks not judged (judge couldn't run) -> subjective==objective
    sample_results: List[Dict] = []

    for i, eval_sample in enumerate(eval_samples):
        graph_dict = eval_sample.graph
        init_node = eval_sample.init_node
        task = eval_sample.task
        answer = eval_sample.answer

        planner_response = None
        planning_result = None
        try:
            if permutation is not None and not is_gnn:
                graph_dict = model_utils.permute_graph_dict(graph_dict, seed=permutation.seed)
            graph_simulation.reset(graph_as_dict=graph_dict, current_location=init_node)
            llm_planner.graph = graph_simulation.partial_graph

            planning_result = planner.run_planning(
                llm_planner=llm_planner,
                task=task,
                graph_data_gen=graph_simulation,
                max_iterations=10,
            )
            planner_response = planning_result.response

            result, formatted_answer = _construct_eval_result(planner_response, answer)

            if result.formatted:
                print(formatted_answer)
            else:
                print(f"incorrect formatting\n{formatted_answer}")

            print(f"correct answer: {result.plan_keyword}")

            pm = _sample_path_metrics(planner_response, eval_sample)
            structured = bool(pm and pm.get("structured"))
            judge_pass = (pm or {}).get("llm_judge_pass")
            if structured:
                # Structural task (positionality / reachability / navigability):
                # the objective verdict is the deterministic NetworkX edge/path
                # check; the Gemma judge is not run. No subjective column, no
                # false-positive/negative diagnostics.
                sc = bool(pm.get("structured_correct"))
                v = {
                    "objective_correct": sc, "objective_keyword": sc,
                    "subjective_correct": None, "false_positive": False,
                    "false_negative": False, "judged": False,
                }
            else:
                # Legacy / non-structural (yes-no, count): two completely separate
                # graders. `objective_*` is the pure RegEx/NetworkX score (the judge
                # never touches it). `subjective_*` is the Gemma judge's score where
                # it ran. The judge moves the subjective column only: down on a false
                # positive (RegEx correct, judge wrong), up on a false negative.
                ac_present = bool(eval_sample.acceptance_criterion)
                if ac_present and judge_pass is None:
                    n_judge_fallback += 1
                v = path_validator.combine_verdict(
                    regex_correct=result.is_correct(),
                    regex_keyword=result.plan_keyword,
                    judge_pass=judge_pass,
                    acceptance_criterion_present=ac_present,
                )
            sample_dict = {
                "graph_name": eval_sample.graph_name,
                "idx": i,
                "task": task,
                "answer_key": answer,
                "response": planner_response,
                "interaction_trace": [asdict(s) for s in planning_result.trace] if planning_result else [],
                "terminated_by": planning_result.terminated_by if planning_result else None,
                "formatted": result.formatted,
                "plan_keyword": result.plan_keyword,
                # `correct` is the OBJECTIVE RegEx/NetworkX verdict (judge-free).
                # `subjective_correct` is the separate judge-based verdict.
                # `false_positive`/`false_negative` flag where the judge disagreed
                # with RegEx. `llm_judge_pass` is the raw judge verdict (None unless
                # yes/no or an acceptance_criterion exists).
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
            # GREP_GEMMA_REGRADE: attach a second reading (scores regraded on the
            # Gemma-recovered route) alongside the original scores above.
            if _gemma_regrade_enabled():
                sample_dict["gemma_regrade"] = _gemma_regrade_block(
                    planner_response, eval_sample, result)
            sample_results.append(sample_dict)
            if v["false_positive"]:
                print("⚑ FALSE POSITIVE: RegEx/NetworkX marked this correct but the Gemma judge "
                      "rejected it — subjective score reduced (objective score unchanged).")
            if v["false_negative"]:
                print("⚑ FALSE NEGATIVE: RegEx/NetworkX marked this wrong but the Gemma judge "
                      "accepted it — subjective score raised (objective score unchanged).")
            total_correct += int(v["objective_keyword"])

        except Exception as e:
            tb_str = traceback_mod.format_exc()
            print("!" * 80)
            print(f"[EVAL CRASH] Sample {i}/{len(eval_samples)} FAILED with unhandled exception!")
            print(f"[EVAL CRASH] Task: {task}")
            print(f"[EVAL CRASH] Error: {type(e).__name__}: {e}")
            print(tb_str)
            print("!" * 80)
            result = _EvalResult(formatted=False, plan_keyword=False)

            sample_results.append({
                "graph_name": eval_sample.graph_name,
                "idx": i,
                "task": task,
                "answer_key": answer,
                "response": None,
                "interaction_trace": [asdict(s) for s in planning_result.trace] if planning_result else [],
                "terminated_by": planning_result.terminated_by if planning_result else "exception",
                "formatted": False,
                "plan_keyword": False,
                "correct": False,           # objective — sample crashed
                "structured": False,
                "subjective_correct": None,  # not judged
                "false_positive": False,
                "false_negative": False,
                "llm_judge_pass": None,
                "error": f"{type(e).__name__}: {e}",
                "traceback": tb_str,
                "path_metrics": None,
            })

        print("\n=====\n")

    if n_judge_fallback:
        print(f"[eval] WARNING: {n_judge_fallback}/{len(eval_samples)} acceptance_criterion "
              f"task(s) could not be judged by Gemma ({path_validator.GEMMA_JUDGE_MODEL}) and were scored by "
              f"RegEx/NetworkX only. Ensure the judge weights/HF auth are available so the "
              f"acceptance criteria are enforced.")
    accuracy = total_correct / len(eval_samples) if eval_samples else 0.0
    return accuracy, sample_results


def eval_model_multiple_graphs(
    model,
    tokenizer,
    graph_samples: Dict[str, List[EvalSample]],
    *,
    include_edge_list: bool,
    use_icl: bool,
    permutation,
    on_graph_done: Optional[Callable[[str, GraphEvalResultSummary], None]],
) -> Dict[str, GraphEvalResultSummary]:
    """Evaluate one model over many graphs and return per-graph aggregates.
    

    `graph_samples` maps graph names (typically file stems) to the list of eval samples
    corresponding to that graph.

    `use_icl` is forwarded as-is to every graph's `eval_model_single_graph`
    call. No auto / force-on policy: the caller's word is final.
    """
    results: Dict[str, GraphEvalResultSummary] = {}
    for name, samples in graph_samples.items():
        if not samples:
            continue
        n_nodes = _graph_node_count(samples[0].graph)

        t0 = time.time()
        accuracy, sample_results = eval_model_single_graph(
            model,
            tokenizer,
            samples,
            include_edge_list=include_edge_list,
            use_icl=use_icl,
            permutation=permutation,
        )
        elapsed = time.time() - t0

        # Objective (RegEx/NetworkX) aggregates — read only RegEx fields.
        num_correct = sum(r["correct"] for r in sample_results)
        num_formatted = sum(r["formatted"] for r in sample_results)
        num_keyword = sum(r["plan_keyword"] for r in sample_results)
        num_errors = sum(1 for r in sample_results if r["error"] is not None)
        # Subjective (Gemma judge) aggregates — read only the judge verdict, over
        # the judged samples ONLY. No RegEx value is mixed in.
        judged = [r for r in sample_results if r["llm_judge_pass"] is not None]
        num_judged = len(judged)
        subjective_accuracy = (
            sum(1 for r in judged if r["llm_judge_pass"]) / num_judged if num_judged else None
        )
        num_false_pos = sum(1 for r in sample_results if r.get("false_positive"))
        num_false_neg = sum(1 for r in sample_results if r.get("false_negative"))

        # GREP_GEMMA_REGRADE second reading: same objective/subjective/path
        # aggregates, computed from each sample's `gemma_regrade` block (scores on
        # the Gemma-recovered route). None unless the regrade ran.
        gemma_blocks = [r["gemma_regrade"] for r in sample_results if r.get("gemma_regrade")]
        gemma = None
        if gemma_blocks:
            g_judged = [g for g in gemma_blocks if g.get("llm_judge_pass") is not None]
            gemma = {
                "num_total": len(gemma_blocks),
                "num_correct": sum(1 for g in gemma_blocks if g["correct"]),
                "accuracy": sum(1 for g in gemma_blocks if g["objective_keyword"]) / len(gemma_blocks),
                "num_judged": len(g_judged),
                "subjective_accuracy": (
                    sum(1 for g in g_judged if g["llm_judge_pass"]) / len(g_judged)
                    if g_judged else None),
                "num_false_pos": sum(1 for g in gemma_blocks if g.get("false_positive")),
                "num_false_neg": sum(1 for g in gemma_blocks if g.get("false_negative")),
                "path_metrics": _aggregate_path_metrics(gemma_blocks),
            }

        result = GraphEvalResultSummary(
            name=name,
            num_total=len(sample_results),
            num_correct=num_correct,
            accuracy=accuracy,
            subjective_accuracy=subjective_accuracy,
            num_judged=num_judged,
            num_formatted=num_formatted,
            num_keyword=num_keyword,
            num_false_pos=num_false_pos,
            num_false_neg=num_false_neg,
            num_errors=num_errors,
            elapsed_s=elapsed,
            n_nodes=n_nodes,
            use_icl=use_icl,
            permutation=permutation.to_dict() if permutation is not None else None,
            samples=sample_results,
            path_metrics=_aggregate_path_metrics(sample_results),
            gemma=gemma,
        )
        results[name] = result

        if on_graph_done is not None:
            on_graph_done(name, result)

    return results


def print_summary_table(results: List[GraphEvalResultSummary]) -> None:
    """Print a formatted accuracy / counts / time table to stdout."""
    if not results:
        print("(no results to summarise)")
        return

    name_width = max(len(r.name) for r in results)
    name_width = max(name_width, len("Eval File"))

    header = (
        f"{'Eval File':<{name_width}}  "
        f"{'Tasks':>5}  "
        f"{'Correct':>7}  "
        f"{'Acc(obj)':>8}  "
        f"{'SubjAcc':>8}  "
        f"{'FP':>3}  "
        f"{'FN':>3}  "
        f"{'Formatted':>9}  "
        f"{'Keyword':>7}  "
        f"{'Errors':>6}  "
        f"{'Time (s)':>8}"
    )
    sep = "-" * len(header)

    print(f"\n{sep}")
    print("EVALUATION SUITE SUMMARY  (Acc(obj)=RegEx/NetworkX; SubjAcc=Gemma judge; FP/FN=judge↔RegEx disagreements)")
    print(sep)
    print(header)
    print(sep)

    total_tasks = total_correct = total_formatted = total_keyword = total_errors = 0
    total_false_pos = total_false_neg = total_judged = 0
    total_subj_hits = 0.0
    total_time = 0.0

    def _pct(v):
        return f"{v:>8.1%}" if v is not None else f"{'n/a':>8}"

    for r in results:
        total_tasks += r.num_total
        total_correct += r.num_correct
        total_formatted += r.num_formatted
        total_keyword += r.num_keyword
        total_false_pos += r.num_false_pos
        total_false_neg += r.num_false_neg
        total_judged += r.num_judged
        if r.subjective_accuracy is not None:
            total_subj_hits += r.subjective_accuracy * r.num_judged
        total_errors += r.num_errors
        total_time += r.elapsed_s

        print(
            f"{r.name:<{name_width}}  "
            f"{r.num_total:>5}  "
            f"{r.num_correct:>7}  "
            f"{r.accuracy:>8.1%}  "
            f"{_pct(r.subjective_accuracy)}  "
            f"{r.num_false_pos:>3}  "
            f"{r.num_false_neg:>3}  "
            f"{r.num_formatted:>9}  "
            f"{r.num_keyword:>7}  "
            f"{r.num_errors:>6}  "
            f"{r.elapsed_s:>8.1f}"
        )

    print(sep)
    overall_acc = total_correct / total_tasks if total_tasks else 0.0
    overall_subj = (total_subj_hits / total_judged) if total_judged else None
    print(
        f"{'TOTAL':<{name_width}}  "
        f"{total_tasks:>5}  "
        f"{total_correct:>7}  "
        f"{overall_acc:>8.1%}  "
        f"{_pct(overall_subj)}  "
        f"{total_false_pos:>3}  "
        f"{total_false_neg:>3}  "
        f"{total_formatted:>9}  "
        f"{total_keyword:>7}  "
        f"{total_errors:>6}  "
        f"{total_time:>8.1f}"
    )
    print(sep)
    if total_false_pos or total_false_neg:
        net = (overall_subj - overall_acc) if overall_subj is not None else 0.0
        print(f"  ⚑ Subjective column (judge-only, over {total_judged} judged): {total_false_pos} false "
              f"positive(s) (−) and {total_false_neg} false negative(s) (+); net {net:+.1%} vs the "
              f"judge-free objective column.")

    # GREP_GEMMA_REGRADE block — original vs Gemma-recovered-path objective accuracy.
    g_results = [r for r in results if r.gemma]
    if g_results:
        print("\nGEMMA-REGRADE  (objective accuracy on the Gemma-recovered path vs the original)")
        print(f"{'Eval File':<{name_width}}  {'Acc(orig)':>9}  {'Acc(gemma)':>10}  "
              f"{'Correct(o)':>10}  {'Correct(g)':>10}")
        g_tasks = g_orig = g_gem = 0
        for r in g_results:
            g = r.gemma
            g_tasks += r.num_total
            g_orig += r.num_correct
            g_gem += g["num_correct"]
            print(f"{r.name:<{name_width}}  {r.accuracy:>9.1%}  {g['accuracy']:>10.1%}  "
                  f"{r.num_correct:>10}  {g['num_correct']:>10}")
        oa = g_orig / g_tasks if g_tasks else 0.0
        ga = g_gem / g_tasks if g_tasks else 0.0
        print(f"{'TOTAL':<{name_width}}  {oa:>9.1%}  {ga:>10.1%}  {g_orig:>10}  {g_gem:>10}")
        print(sep)

    # M10 (R4) path-validity block — only printed when some graph yielded routes.
    pm_results = [r for r in results if r.path_metrics]
    if pm_results:
        def _fmt(v):
            return f"{v:.2f}" if isinstance(v, (int, float)) else "  —"
        print("\nPATH-VALIDITY (M10)")
        print(f"{'Eval File':<{name_width}}  {'edge_val':>8}  {'nodes_ex':>8}  "
              f"{'full_valid':>10}  {'start_goal':>10}  {'cost_opt':>8}  {'judge':>6}")
        for r in pm_results:
            p = r.path_metrics
            print(f"{r.name:<{name_width}}  {_fmt(p.get('edge_validity_rate')):>8}  "
                  f"{_fmt(p.get('nodes_exist_rate')):>8}  {_fmt(p.get('full_path_valid_rate')):>10}  "
                  f"{_fmt(p.get('start_goal_ok_rate')):>10}  {_fmt(p.get('cost_optimality')):>8}  "
                  f"{_fmt(p.get('llm_judge_accuracy')):>6}")
        print(sep)


# ----------------------------------------------------------------------------
# Private helpers
# ----------------------------------------------------------------------------

def construct_eval_samples_from_dict(
    graph_dict: dict,
    tasks_list: List[dict],
    graph_name: str,
) -> List[EvalSample]:
    """Convert a parsed `{"graph": ..., "tasks": [...]}` payload into `EvalSample`s.

    Pure in-memory; no I/O. Callers (scripts, train_v2) `json.load` the file,
    decide what the graph's stable identifier is (typically the filename
    stem), and pass `data["graph"]`, `data["tasks"]`, and that identifier in.
    Every sample produced here will carry `graph_name` so per-sample result
    dicts can be traced back to their source graph.
    """
    return [
        EvalSample(
            task=t["task"],
            answer=t["answer"],
            graph=graph_dict,
            init_node=t["init_node"],
            graph_name=graph_name,
            acceptance_criterion=t.get("acceptance_criterion"),
        )
        for t in tasks_list
    ]

class _EvalResult:
    """Two-bit eval verdict: response is JSON-formatted, plan contains the keyword."""

    def __init__(self, formatted: bool, plan_keyword: bool):
        self.formatted = formatted
        self.plan_keyword = plan_keyword

    def is_correct(self) -> bool:
        return self.formatted and self.plan_keyword


def _has_correct_keys(answer: Dict[str, str]) -> bool:
    return (
        "primary_goal" in answer
        and "relevant_graph" in answer
        and "reasoning" in answer
        and "plan" in answer
    )


def _construct_eval_result(parsed_answer: Dict[str, str], answer_key: str) -> Tuple[_EvalResult, dict]:
    """Construct an eval result from a parsed planner answer: JSON-shape correctness and keyword match."""
    is_formatted = False
    has_plan_keyphrase = False
    try:
        is_formatted = _has_correct_keys(parsed_answer)
        has_plan_keyphrase = bool(
            re.search(answer_key, str(parsed_answer["plan"]), re.IGNORECASE)
        )
        return _EvalResult(formatted=is_formatted, plan_keyword=has_plan_keyphrase), parsed_answer
    except Exception as e:
        import traceback as _tb
        if not parsed_answer:
            print(f"[eval] _construct_eval_result received empty response — SPINE likely exhausted retries "
                  f"(all attempts produced unparseable JSON). exception: {e}")
        else:
            print(f"[eval] _construct_eval_result exception: {e}\n{_tb.format_exc()}")
        return _EvalResult(False, False), parsed_answer


def _sample_path_metrics(planner_response, eval_sample: EvalSample) -> Optional[dict]:
    """M10 (R4) per-sample path validation; never raises (returns None on failure).

    Pulls the route out of the planner's ``plan`` field, validates it against the
    sample's graph (regex + NetworkX), and runs the Gemma judge only if the
    sample carries an ``acceptance_criterion``.
    """
    try:
        plan = planner_response.get("plan") if isinstance(planner_response, dict) else planner_response
        return path_validator.evaluate_sample(
            eval_sample.task,
            "" if plan is None else str(plan),
            eval_sample.graph,
            init_node=eval_sample.init_node,
            acceptance_criterion=eval_sample.acceptance_criterion,
            answer=eval_sample.answer,
            full_response="" if planner_response is None else str(planner_response),
        )
    except Exception as e:
        print(f"[eval] path-metric computation failed: {type(e).__name__}: {e}")
        return None


def _objective_verdict(pm, result: "_EvalResult", eval_sample: EvalSample):
    """The ``(v, structured, judge_pass)`` triple for a path-metrics dict.

    Mirrors the per-sample verdict logic in ``eval_model_single_graph`` but with no
    side effects (no ``n_judge_fallback`` bump), so it can grade the Gemma-regrade
    path metrics with the SAME rules as the original.
    """
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
            acceptance_criterion_present=bool(eval_sample.acceptance_criterion),
        )
    return v, structured, judge_pass


def _gemma_regrade_block(planner_response, eval_sample: EvalSample,
                         result: "_EvalResult") -> dict:
    """The ``gemma_regrade`` sub-record: the sample's scores regraded on the
    Gemma-recovered route, kept ALONGSIDE the original top-level scores.

    The path metrics come from ``path_validator.gemma_regrade_path_metrics`` — the
    SAME shared function the retro-grader (``apply_judge_to_eval_run.py``) uses — so
    the two readings are byte-identical. It stamps ``path_source``
    (``gemma_judge`` / ``regex_fallback``) and the raw ``gemma_route``.
    """
    pm_g = path_validator.gemma_regrade_path_metrics(
        planner_response, eval_sample.graph,
        init_node=eval_sample.init_node, answer=eval_sample.answer,
        acceptance_criterion=eval_sample.acceptance_criterion, task=eval_sample.task)
    v_g, structured_g, judge_pass_g = _objective_verdict(pm_g, result, eval_sample)
    return {
        "correct": v_g["objective_correct"],
        "objective_keyword": v_g["objective_keyword"],
        "structured": structured_g,
        "subjective_correct": v_g["subjective_correct"],
        "false_positive": v_g["false_positive"],
        "false_negative": v_g["false_negative"],
        "llm_judge_pass": judge_pass_g,
        "path_metrics": pm_g,
    }


# Canonical aggregation now lives in `path_validator` (model-free) so the offline
# retro-grader shares one source. Kept under the original private name for the call
# sites in this module and the eval callback.
_aggregate_path_metrics = path_validator.aggregate_path_metrics


def _is_graph_augmented(model) -> bool:
    """True if `model` is (or wraps) a graph-augmented LLM, including under PEFT.

    Covers the legacy `GraphAugmentedLLM` (PE injection), the parameter-free
    `GraphMaskLLM` (structural attention mask), and the M9 `CompositeGraphLLM`
    (composite-graph fusion).
    """
    graph_types = (gnn_llm.GraphAugmentedLLM, gnn_llm.GraphMaskLLM, gnn_llm.CompositeGraphLLM)
    if isinstance(model, graph_types):
        return True
    inner = getattr(getattr(model, "base_model", None), "model", None)
    return isinstance(inner, graph_types)


def _graph_node_count(graph: dict) -> int:
    return len(graph.get("objects", {})) + len(graph.get("regions", {}))
