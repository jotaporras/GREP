"""Neighbour-naming probe (e18): can a checkpoint NAME the regions adjacent to a node?

The navigation eval scores routes; this probe isolates the one capability the
e17 failure analysis says is missing (docs/2026-08-21 e18_direction_discussion.md
§2): *given a node, produce its neighbours' names*. For every region ``r`` of
every graph file given, the model is asked — through the SAME prompt/client
stack the navigation eval uses (compact prompt, no edge list for graph archs,
SPINE JSON inverse-translation) — to list the regions directly connected to
``r``. The plan text is scored as a SET against the graph:

* ``first_ok``      — the first region named is a true neighbour (the quantity
                      the decision-step analysis is about);
* ``exact``         — named set == neighbour set;
* ``precision`` / ``recall`` over the named set;
* ``sibling_err``   — a named non-neighbour that shares a type prefix with a
                      true neighbour (e.g. ``sub_dock_2`` for ``sub_dock_1``):
                      the sibling-confusion mode of the n60 analysis;
* ``hallucinated``  — named non-neighbours with no sibling excuse.

This is a standalone script (not a ``scalability_evaluation`` task file) because
``path_validator.derive_targets`` reads node ids out of the answer regex and
grades a *route* to the last one — a neighbour-set answer has no route.

Usage (betty, one GPU)::

    python scripts/neighbour_probe.py --checkpoint <run_dir or checkpoint-N> \
        --graphs <split>/test_graphs --output results/e18_probe/<run>.json

``--text-edge-list`` follows the navigation eval's resolution (train_config);
pass ``present`` for the plain-LLM control. Output is one JSON with per-query
records and the aggregate, printed as a one-line summary at the end.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from prism.data import data
from prism.eval import checkpoint
from prism.eval import evaluate
from prism.models import inference

_ID_TAIL = re.compile(r"_\d+$")

TASK = ("You are at {r}. Which regions are directly connected to {r}? List EVERY "
        "region that shares an edge with {r}, and no other region. Give the names "
        "as a comma-separated list in the plan.")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--graphs", required=True,
                   help="A graph JSON ({graph, tasks}) file, a directory, or a glob.")
    p.add_argument("--output", required=True, help="JSON results path.")
    p.add_argument("--four-bit", action="store_true", default=False)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--text-edge-list", choices=["present", "none"], default=None)
    p.add_argument("--use-icl", choices=["true", "false"], default="false")
    p.add_argument("--max-graphs", type=int, default=-1,
                   help="Cap on graph files probed (-1 = all).")
    p.add_argument("--max-new-tokens", type=int, default=256)
    return p.parse_args(argv)


def _type_prefix(name: str) -> str:
    return _ID_TAIL.sub("", name)


def neighbours(graph: dict) -> dict[str, set[str]]:
    regions = [r["name"] for r in graph["regions"]]
    adj = {r: set() for r in regions}
    for a, b in graph["region_connections"]:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def named_regions(text: str, regions: list[str], exclude: str) -> list[str]:
    """Region names in ``text`` ordered by first occurrence (``exclude`` dropped)."""
    hits = []
    for name in regions:
        if name == exclude:
            continue
        m = re.search(rf"\b{re.escape(name)}\b", text)
        if m:
            hits.append((m.start(), name))
    return [n for _, n in sorted(hits)]


def score(named: list[str], truth: set[str], regions: list[str]) -> dict:
    pred = set(named)
    truth_prefixes = {_type_prefix(n) for n in truth}
    wrong = pred - truth
    sibling = {n for n in wrong if _type_prefix(n) in truth_prefixes}
    return {
        "first_ok": bool(named) and named[0] in truth,
        "exact": pred == truth,
        "precision": (len(pred & truth) / len(pred)) if pred else 0.0,
        "recall": (len(pred & truth) / len(truth)) if truth else 1.0,
        "n_named": len(pred),
        "n_truth": len(truth),
        "sibling_err": len(sibling),
        "hallucinated": len(wrong - sibling),
        "missed": sorted(truth - pred),
        "wrong": sorted(wrong),
    }


def plan_text(planner_response: str) -> str:
    """The ``plan`` field of the SPINE JSON the client returns (raw string if the
    model's output did not translate into JSON with a plan)."""
    try:
        parsed = json.loads(planner_response)
    except json.JSONDecodeError:
        return planner_response
    if isinstance(parsed, dict) and "plan" in parsed:
        return str(parsed["plan"])
    return planner_response


def build_client(model, tokenizer, is_gnn: bool, *, include_edge_list: bool,
                 use_icl: bool, edge_weights: str, injection_scope: str):
    include_tools = not evaluate._spine_tools_disabled()
    icl_examples = evaluate._compact_icl_examples(use_icl)
    if is_gnn:
        return inference.GraphAugmentedInMemoryLLM(
            model=model, tokenizer=tokenizer, include_edges=include_edge_list,
            include_tools=include_tools, icl_examples=icl_examples, permutation=None,
            edge_weights=edge_weights, injection_scope=injection_scope)
    return inference.InMemoryLLM(
        model=model, tokenizer=tokenizer, include_edges=include_edge_list,
        include_tools=include_tools, icl_examples=icl_examples)


def probe_graph(client, graph: dict, *, use_icl: bool, max_new_tokens: int) -> list[dict]:
    adj = neighbours(graph)
    regions = [r["name"] for r in graph["regions"]]
    records = []
    for r in regions:
        g = dict(graph)
        g["robot_location"] = r
        msg = evaluate._fixed_get_base_prompt(TASK.format(r=r), json.dumps(g), use_icl)
        response, _ = client.query_llm(msg, max_new_tokens=max_new_tokens)
        plan = plan_text(response)
        named = named_regions(plan, regions, exclude=r)
        rec = {"node": r, "truth": sorted(adj[r]), "named": named, "plan": plan}
        rec.update(score(named, adj[r], regions))
        records.append(rec)
    return records


def aggregate(records: list[dict]) -> dict:
    n = len(records)
    mean = lambda k: (sum(float(x[k]) for x in records) / n) if n else float("nan")
    return {
        "n_queries": n,
        "first_ok": mean("first_ok"),
        "exact": mean("exact"),
        "precision": mean("precision"),
        "recall": mean("recall"),
        "sibling_err_per_query": mean("sibling_err"),
        "hallucinated_per_query": mean("hallucinated"),
        "mean_degree": mean("n_truth"),
    }


def main(argv=None):
    args = _parse_args(argv)
    use_icl = args.use_icl == "true"
    ckpt = os.path.abspath(args.checkpoint.rstrip("/"))
    is_gnn = checkpoint.is_gnn_checkpoint(ckpt)
    text_edge_list = checkpoint.resolve_text_edge_list(ckpt, is_gnn, args.text_edge_list)
    edge_weights = checkpoint.resolve_edge_weights(ckpt)
    injection_scope = checkpoint.resolve_injection_scope(ckpt)
    print(f"[probe] checkpoint={ckpt} is_gnn={is_gnn} text_edge_list={text_edge_list} "
          f"edge_weights={edge_weights} injection_scope={injection_scope}")

    samples_by_graph, graph_file_by_name = data.load_samples_by_graph(args.graphs)
    names = sorted(graph_file_by_name)
    if args.max_graphs > 0:
        names = names[:args.max_graphs]

    model, tokenizer, _ = checkpoint.load_checkpoint(ckpt, four_bit=args.four_bit,
                                                     device=args.device)
    client = build_client(model, tokenizer, is_gnn,
                          include_edge_list=(text_edge_list == "present"),
                          use_icl=use_icl, edge_weights=edge_weights,
                          injection_scope=injection_scope)

    per_graph = {}
    all_records = []
    for name in names:
        with open(graph_file_by_name[name]) as f:
            graph = json.load(f)["graph"]
        recs = probe_graph(client, graph, use_icl=use_icl,
                           max_new_tokens=args.max_new_tokens)
        per_graph[name] = {"records": recs, "aggregate": aggregate(recs)}
        all_records.extend(recs)
        a = per_graph[name]["aggregate"]
        print(f"[probe] {name}: first_ok={a['first_ok']:.2f} exact={a['exact']:.2f} "
              f"P={a['precision']:.2f} R={a['recall']:.2f} "
              f"sib/q={a['sibling_err_per_query']:.2f} hall/q={a['hallucinated_per_query']:.2f}")

    summary = aggregate(all_records)
    out = {
        "checkpoint": ckpt, "graphs": args.graphs, "is_gnn": is_gnn,
        "text_edge_list": text_edge_list, "injection_scope": injection_scope,
        "use_icl": use_icl, "task_template": TASK,
        "aggregate": summary, "per_graph": per_graph,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[probe] TOTAL n={summary['n_queries']} first_ok={summary['first_ok']:.3f} "
          f"exact={summary['exact']:.3f} P={summary['precision']:.3f} "
          f"R={summary['recall']:.3f} sib/q={summary['sibling_err_per_query']:.3f} "
          f"hall/q={summary['hallucinated_per_query']:.3f} -> {args.output}")


if __name__ == "__main__":
    main()
