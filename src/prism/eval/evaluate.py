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
    header = [spine_prompts.SYS_PROMPT] + (_ICL_EXAMPLES_2 if use_icl else examples.EXAMPLE_1)
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
    num_correct: int
    accuracy: float
    num_formatted: int
    num_keyword: int
    num_errors: int
    elapsed_s: float
    n_nodes: int
    use_icl: bool
    permutation: Optional[dict]
    samples: List[dict] = field(default_factory=list)
    # M10 (R4) path-validity aggregates over this graph's samples. Empty when
    # no sample produced a parseable route.
    path_metrics: dict = field(default_factory=dict)


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
        `(accuracy, sample_results)` where `accuracy` is fraction with the
        `plan_keyword` flag set, and `sample_results` is a list of dicts
        with one entry per sample (see schema in source).   
    """
    graph_handler = graph_util.GraphHandler("")
    graph_simulation = graph_sim.GraphSim(graph_handler)

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

    total_correct = 0
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
            sample_results.append({
                "graph_name": eval_sample.graph_name,
                "idx": i,
                "task": task,
                "answer_key": answer,
                "response": planner_response,
                "interaction_trace": [asdict(s) for s in planning_result.trace] if planning_result else [],
                "terminated_by": planning_result.terminated_by if planning_result else None,
                "formatted": result.formatted,
                "plan_keyword": result.plan_keyword,
                # `correct` is the authoritative RegEx/NetworkX verdict. The Gemma
                # judge is advisory only (never the true judge): its per-sample
                # verdict is surfaced separately as `llm_judge_pass` (None unless
                # the answer is yes/no or an acceptance_criterion exists).
                "correct": result.is_correct(),
                "llm_judge_pass": (pm or {}).get("llm_judge_pass"),
                "error": None,
                "traceback": None,
                "path_metrics": pm,
            })

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
                "correct": False,
                "llm_judge_pass": None,
                "error": f"{type(e).__name__}: {e}",
                "traceback": tb_str,
                "path_metrics": None,
            })

        print("\n=====\n")
        total_correct += result.plan_keyword

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

        num_correct = sum(r["correct"] for r in sample_results)
        num_formatted = sum(r["formatted"] for r in sample_results)
        num_keyword = sum(r["plan_keyword"] for r in sample_results)
        num_errors = sum(1 for r in sample_results if r["error"] is not None)

        result = GraphEvalResultSummary(
            name=name,
            num_total=len(sample_results),
            num_correct=num_correct,
            accuracy=accuracy,
            num_formatted=num_formatted,
            num_keyword=num_keyword,
            num_errors=num_errors,
            elapsed_s=elapsed,
            n_nodes=n_nodes,
            use_icl=use_icl,
            permutation=permutation.to_dict() if permutation is not None else None,
            samples=sample_results,
            path_metrics=_aggregate_path_metrics(sample_results),
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
        f"{'Acc':>7}  "
        f"{'Formatted':>9}  "
        f"{'Keyword':>7}  "
        f"{'Errors':>6}  "
        f"{'Time (s)':>8}"
    )
    sep = "-" * len(header)

    print(f"\n{sep}")
    print("EVALUATION SUITE SUMMARY")
    print(sep)
    print(header)
    print(sep)

    total_tasks = total_correct = total_formatted = total_keyword = total_errors = 0
    total_time = 0.0

    for r in results:
        total_tasks += r.num_total
        total_correct += r.num_correct
        total_formatted += r.num_formatted
        total_keyword += r.num_keyword
        total_errors += r.num_errors
        total_time += r.elapsed_s

        print(
            f"{r.name:<{name_width}}  "
            f"{r.num_total:>5}  "
            f"{r.num_correct:>7}  "
            f"{r.accuracy:>7.1%}  "
            f"{r.num_formatted:>9}  "
            f"{r.num_keyword:>7}  "
            f"{r.num_errors:>6}  "
            f"{r.elapsed_s:>8.1f}"
        )

    print(sep)
    overall_acc = total_correct / total_tasks if total_tasks else 0.0
    print(
        f"{'TOTAL':<{name_width}}  "
        f"{total_tasks:>5}  "
        f"{total_correct:>7}  "
        f"{overall_acc:>7.1%}  "
        f"{total_formatted:>9}  "
        f"{total_keyword:>7}  "
        f"{total_errors:>6}  "
        f"{total_time:>8.1f}"
    )
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


def _aggregate_path_metrics(sample_results: List[dict]) -> dict:
    """Mean M10 path metrics over samples that produced a parseable route."""
    pms = [r.get("path_metrics") for r in sample_results]
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
    judged = [p["llm_judge_pass"] for p in pms if p.get("llm_judge_pass") is not None]
    if judged:
        agg["llm_judge_accuracy"] = sum(judged) / len(judged)
    return agg


def _is_graph_augmented(model) -> bool:
    """True if `model` is (or wraps) a graph-augmented LLM, including under PEFT.

    Covers both the legacy `GraphAugmentedLLM` (PE injection) and the M9
    `CompositeGraphLLM` (composite-graph fusion).
    """
    graph_types = (gnn_llm.GraphAugmentedLLM, gnn_llm.CompositeGraphLLM)
    if isinstance(model, graph_types):
        return True
    inner = getattr(getattr(model, "base_model", None), "model", None)
    return isinstance(inner, graph_types)


def _graph_node_count(graph: dict) -> int:
    return len(graph.get("objects", {})) + len(graph.get("regions", {}))
