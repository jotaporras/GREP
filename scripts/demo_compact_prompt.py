"""Demo: the compact plain-text prompt format fed to the GREP-PRISM models.

Shows exactly what training and eval prompts become once the verbose SPINE JSON
(long system prompt + few-shot ICL examples + full scene-graph JSON) is translated
to the compact format the LLM consumes:

  * the verbose SPINE system prompt is always DROPPED and replaced by the short
    compact-format system prompt: intro + tool policy + the ``<think>…</think>``
    contract (``compact_prompt.compact_system_prompt``);
  * the scene graph becomes a compact ``Scene graph:`` block (bulleted node-name
    lists + robot location). Graph-augmented archs OMIT edges (the GNN supplies
    connectivity) and use the latent-connectivity intro; the plain-LLM baseline
    (``include_edges=True``) instead lists ``• Region Edges:`` / ``• Object Edges:``
    and uses an edge-aware intro that points at them — see the PLAIN-LLM section;
  * SPINE TOOL CALLING (``include_tools=True``) documents the SPINE API in the system
    prompt, demands every hop be ratified with ``map_region`` before ``answer()``, and
    keeps each target's action-list plan (``[goto(x), answer(a -> b)]``). With
    ``include_tools=False`` the prompt instead forbids tool calls, tells the model to
    walk adjacencies by thought in latent space, and the plan is a bare arrow route;
  * ICL (``icl_examples > 0``) keeps that many of the leading SPINE few-shot examples
    and compacts them — plans, ``updates:`` turns and replans intact, which is what
    demonstrates tool calling. With ICL the query graph is NOT hoisted into the system
    message: it opens the query ``user`` turn so it stays the LAST graph block and PE
    injection (``find_last_graph_scope``) still scopes to the query graph;
  * tasks then stack as ``user``/``assistant`` pairs in the same conversation per
    graph; the assistant target wraps reasoning in
    ``<think>Relevant graph: …\\n\\nReasoning: …</think>`` followed by the bare plan.

Every SPINE / SPINE+ICL example below is drawn from the DATA — the raw logged rollouts
under ``generated_plans/`` still carry SPINE's system prompt and its five ICL examples,
so they are fed to ``compact_prompt`` verbatim rather than to a hand-written showcase.

TRAINING and DEPLOYMENT run different policies, and the demo shows both. The shipped
configuration TRAINS tool-free and zero-shot (``data.spine_tools=none``,
``data.icl_examples=0``) and DEPLOYS with the SPINE API live, still zero-shot
(``PRISM_DISABLE_SPINE_TOOLS`` unset, ``eval.use_icl=false``). The seam absorbs the tool
gap: the model emits the bare route it was trained on and
``compact_output_to_spine_json`` wraps it as ``[answer(route)]`` for the planner. ICL is
fully supported on both sides and simply switched off by default — the SPINE + ICL
sections below always render it so the layout stays visible.

The ``User:``/``Assistant:`` turn delimiters are produced by the tokenizer's chat
template (native role special tokens), NOT literal text — so with ``--tokenizer``
the demo renders the byte-for-byte prompt via ``apply_chat_template``; without it,
it prints an illustrative ``Role:``-labelled view instead of the byte-exact
template output.

Pure pre-processing; does NOT modify anything under data/.

Run:  PYTHONPATH=src python scripts/demo_compact_prompt.py            # shipped policy
      PYTHONPATH=src python scripts/demo_compact_prompt.py --tokenizer meta-llama/Llama-3.2-3B-Instruct
      PYTHONPATH=src python scripts/demo_compact_prompt.py --icl-examples 2   # few-shot deploy
      PYTHONPATH=src python scripts/demo_compact_prompt.py --train-spine-tools --train-icl-examples 2
"""

import argparse
import json
from pathlib import Path

from prism.data.compact_prompt import (
    assemble_training_conversation,
    compact_output_to_spine_json,
    format_eval_messages,
    format_training_messages,
    icl_demos_from_rollouts,
    render,
    spine_to_compact_messages,
    strip_icl,
    try_load_json,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "n_30" / "gen" / "nav100_n30_gemma_data"
PLAN_DIR = DATA_DIR / "generated_plans"
DEFAULT_PLAN = PLAN_DIR / "sample_000_000.json"
DEFAULT_GRAPH = DATA_DIR / "populated_graphs" / "data_gen_000.json"
# A DIFFERENT graph's rollouts, used as data-drawn few-shot demos for the query graph.
ICL_GRAPH_INDEX = "001"

# A REAL two-turn SPINE rollout, captured verbatim from a live eval run so the tool
# section can show tool calls actually working. The logged training corpus answers
# directly ([answer(...)]), so replaying it would show a SPINE section containing no
# SPINE calls; this is the same graph (data_gen_000), just a task where the model chose
# to ratify before committing.
#
# `reasoning` / `plan` are the model's own text, unedited. `updates` is what GraphSim
# returned after executing the turn's first action. The demo re-wraps these into SPINE's
# logged 4-key JSON shape and runs them through the LIVE translator, so the printed
# conversation is produced by the pipeline rather than written out here.
#
# The section RENDERS this at ``include_edges=False`` — the GT+LLM arch, where the scene
# graph block carries node names only and connectivity arrives through the graph channel.
# The capture itself came from the edge-list arm, because a trained GT checkpoint could
# not be run on the capture machine (see the provenance string). What is being shown is
# the tool-call mechanism and the prompt shape, which are identical either way; the one
# thing NOT demonstrated is where the model's adjacency knowledge came from.
RECORDED_SPINE_ROLLOUT = {
    "model": "google/gemma-4-E2B-it",
    "provenance": ("live eval run, MPS bf16, zero-shot, tools ON, graph data_gen_000 task 4; "
                   "captured on the plain-LLM arm (edges in text, no GNN) and rendered here "
                   "as the GT+LLM arm reads it (no edge list)"),
    "task": "Route the robot from the starting area to the battery bank and list the path.",
    "turns": [
        {
            "relevant_graph": "hub_1, battery_bank_1",
            "reasoning": (
                "The robot starts at hub_1 and needs to reach battery_bank_1. I need to find a "
                "path through the connected regions.\n"
                "1. hub_1 is connected to mess_hall_1, corridor_1, lobby_1, lounge_1, hallway_1, "
                "archive_1.\n"
                "2. battery_bank_1 is connected to generator_room_1 and maintenance_bay_1.\n"
                "3. I need to find a path from the hub_1 cluster to the battery_bank_1 cluster.\n"
                "4. lobby_1 is connected to office_1 and clean_room_1.\n"
                "5. office_1 is connected to power_plant_1.\n"
                "6. power_plant_1 is connected to generator_room_1.\n"
                "7. generator_room_1 is connected to battery_bank_1.\n"
                "8. A potential path is hub_1 -> lobby_1 -> office_1 -> power_plant_1 -> "
                "generator_room_1 -> battery_bank_1. I must confirm these edges via map_region "
                "calls if they are not explicitly known. I will start by mapping the starting "
                "region to confirm immediate neighbors."),
            "plan": "[map_region(hub_1)]",
            "note": "ratifies the start region before committing to any hop",
            "updates": "update_node_attributes({name: hub_1, description: })",
        },
        {
            "relevant_graph": "hub_1, battery_bank_1",
            "reasoning": (
                "I have mapped hub_1. I need to re-evaluate the path to battery_bank_1. The "
                "previous plan was to go through lobby_1 -> office_1 -> power_plant_1 -> "
                "generator_room_1 -> battery_bank_1. I will now attempt to navigate directly to "
                "the goal region, battery_bank_1, relying on the graph search to find the path "
                "through the confirmed or existing edges."),
            "plan": ("[goto(battery_bank_1), answer(hub_1 -> lobby_1 -> office_1 -> "
                     "power_plant_1 -> generator_room_1 -> battery_bank_1)]"),
            "note": "hop confirmed, so it commits: goto the goal, then the terminal answer()",
            "updates": None,
        },
    ],
    # From path_validator on that run's final route.
    "metrics": {
        "structured_correct": True, "edge_validity_rate": 1.0, "hallucination_rate": 0.0,
        "full_path_valid": True, "hop_optimality": 2.5,
        "parsed_nodes": ["hub_1", "lobby_1", "office_1", "power_plant_1",
                         "generator_room_1", "battery_bank_1"],
    },
}


def _rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _roles(messages) -> str:
    return " ".join(m["role"][0].upper() for m in messages)  # e.g. "S U A U A"


def _show(messages, tokenizer, add_generation_prompt: bool) -> None:
    label = "apply_chat_template" if tokenizer is not None else "illustrative Role: view"
    if add_generation_prompt:
        label += " (+generation prompt)"
    print(f"\n-- prompt the model receives [{label}] --")
    print(render(messages, tokenizer=tokenizer, add_generation_prompt=add_generation_prompt))


def _make_counter(tokenizer):
    if tokenizer is not None:
        return "tokens", lambda s: len(tokenizer.encode(s, add_special_tokens=False))
    return "chars", len


def _reduction_summary(before_msgs, after_msgs, tokenizer) -> None:
    label, count = _make_counter(tokenizer)
    b = count(render(before_msgs, tokenizer=tokenizer))
    a = count(render(after_msgs, tokenizer=tokenizer))
    saved = 100.0 * (b - a) / b if b else 0.0
    print(f"\n  context ({label}):  before={b:,}  after={a:,}  saved={saved:.1f}%")


def _verbose_baseline(graph_dict, task):
    """Illustrative 'before' baseline: the full graph dict restated per task."""
    return [{"role": "user", "content": f"task: {task}\nScene graph:{graph_dict}"}]


def _plan_lines(messages):
    """The text each assistant turn emits after </think> — i.e. the plan format."""
    return [m["content"].split("</think>", 1)[-1].strip()
            for m in messages if m["role"] == "assistant"]


def _rollouts_for_graph(graph_idx: str):
    """Every logged rollout for one graph, in task order (real data, ICL prefix intact)."""
    return [try_load_json(p) for p in sorted(PLAN_DIR.glob(f"sample_{graph_idx}_*.json"))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-sample", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--graph-sample", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--tokenizer", type=str, default="")
    # Two policies, because the live pipeline runs two: TRAINING (data.spine_tools /
    # data.icl_examples) and DEPLOYMENT (PRISM_DISABLE_SPINE_TOOLS / eval.use_icl). The
    # defaults are the shipped configuration — trained tool-free and zero-shot, deployed
    # with the SPINE API live and still zero-shot.
    parser.add_argument("--train-spine-tools", dest="train_tools", action="store_true",
                        default=False,
                        help="train WITH the SPINE API + action-list targets (default: off)")
    parser.add_argument("--train-icl-examples", type=int, default=0,
                        help="few-shot examples in TRAINING prompts (default 0)")
    parser.add_argument("--spine-tools", dest="deploy_tools", action="store_true", default=True,
                        help="deploy with the SPINE API + action-list plans (default)")
    parser.add_argument("--no-spine-tools", dest="deploy_tools", action="store_false",
                        help="deploy with no tool calls: adjacency reasoning, bare routes")
    parser.add_argument("--icl-examples", type=int, default=0,
                        help="few-shot examples in DEPLOYED prompts (0=none, -1=all; default 0)")
    args = parser.parse_args()

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    train_tools, train_icl = args.train_tools, args.train_icl_examples
    tools, icl = args.deploy_tools, args.icl_examples
    print(f"policy  TRAIN: include_tools={train_tools} icl_examples={train_icl}   "
          f"DEPLOY: include_tools={tools} icl_examples={icl}   "
          f"(include_edges toggled per section)")
    if train_tools != tools:
        print("  train/deploy tool policies differ by design: the model learns bare routes and\n"
              "  compact_output_to_spine_json wraps them as [answer(route)] for the planner.")

    payload = try_load_json(args.graph_sample)
    graph_dict, tasks = payload["graph"], payload["tasks"]
    task = tasks[args.task_index]["task"]

    # --- 1. Training prompt: a real rollout -> compact (TRAIN policy) ---------------
    # Zero-shot training goes through the builder; with few-shot training the live path
    # (data.preprocess_dataset) prepends evaluate.SPINE_ICL_EXAMPLES and translates, which
    # is what spine_to_compact_messages does to a rollout that still carries its ICL.
    conversation = try_load_json(args.plan_sample)
    training_messages = (
        format_training_messages(conversation, include_edges=False, include_tools=train_tools)
        if train_icl == 0 else
        spine_to_compact_messages(conversation, include_edges=False,
                                  include_tools=train_tools, icl_examples=train_icl)
    )
    _rule(f"TRAINING PROMPT  (from {args.plan_sample.name}; "
          f"spine_tools={'present' if train_tools else 'none'}, icl_examples={train_icl})")
    _show(training_messages, tokenizer, add_generation_prompt=False)
    _reduction_summary(strip_icl(conversation), training_messages, tokenizer)

    # --- 2. Eval prompt: graph + task -> compact (open assistant turn) ------------
    eval_messages = format_eval_messages(
        graph_dict, task, include_edges=False, include_tools=tools)
    _rule(f"EVAL PROMPT  (DEPLOY policy; from {args.graph_sample.name}, task {args.task_index})")
    _show(eval_messages, tokenizer, add_generation_prompt=True)  # mirrors query_llm

    # --- 3. SPINE TOOL CALLING: a real two-turn rollout, in compact form ----------
    # The logged corpus answers directly ([answer(...)]), so replaying it here would show
    # a "SPINE" section with no SPINE calls in it. Instead this replays RECORDED_SPINE_
    # ROLLOUT: a real receding-horizon exchange captured from a live eval run — the model
    # ratifies with map_region, GraphSim executes it and returns an `updates:` turn, and
    # only then does the model commit to a route. The transcript is fed through the LIVE
    # translator in SPINE's own logged shape, so what prints is what the pipeline
    # produces, not a hand-written sample.
    raw = conversation
    rec = RECORDED_SPINE_ROLLOUT
    spine_convo = [
        {"role": "system", "content": "<SPINE system prompt — dropped by the translator>"},
        {"role": "user",
         "content": f"{rec['task']}\nAdvice: \n- Recall the scene may be incomplete.\n\n"
                    f"Scene graph:{graph_dict}"},
    ]
    for t, turn in enumerate(rec["turns"]):
        spine_convo.append({"role": "assistant", "content": json.dumps({
            "primary_goal": rec["task"], "relevant_graph": turn["relevant_graph"],
            "reasoning": turn["reasoning"], "plan": turn["plan"]})})
        if turn.get("updates"):                    # what GraphSim returned after executing
            spine_convo.append({"role": "user", "content": turn["updates"]})
    compact_rollout = spine_to_compact_messages(
        spine_convo, include_edges=False, include_tools=True, icl_examples=0)

    _rule(f"SPINE TOOL CALLING  (recorded rollout: {rec['model']}, "
          f"{len(rec['turns'])} planning turns, tools ON, GT+LLM arch: NO edge list)")
    sys_on = next(m["content"] for m in compact_rollout if m["role"] == "system")
    sys_off = next(m["content"] for m in spine_to_compact_messages(
        raw, include_edges=False, include_tools=False, icl_examples=0) if m["role"] == "system")
    assert "SPINE tool API" not in sys_off and "map_region" not in sys_off
    assert "SPINE tool API" in sys_on and "map_region" in sys_on
    # This is the deployed graph-augmented prompt: node names only, connectivity latent.
    assert "• Region Edges:" not in sys_on and "• Object Edges:" not in sys_on
    assert "latent space" in sys_on
    print(f"  task: {rec['task']}")
    print(f"  provenance: {rec['provenance']}\n")
    print("  Prompt shape: the scene-graph block lists node names only — no Region/Object")
    print("  Edges — and the intro tells the model its connectivity is in latent space, which")
    print("  is what the GT+LLM model actually receives. map_region is therefore the way to")
    print("  CONFIRM an edge in the world, not the way to learn the graph.\n")
    print("  How a SPINE call actually runs in the compact prompt. Only the FIRST action of")
    print("  a plan executes; the simulator's result comes back as a bare `updates:` user")
    print("  turn, and the model replans over the same task:\n")
    for t, turn in enumerate(rec["turns"], start=1):
        print(f"    turn {t}  assistant -> {turn['plan']}")
        print(f"             {turn['note']}")
        if turn.get("updates"):
            print(f"             GraphSim executed it and returned:")
            print(f"               updates -> {turn['updates']}")
    pm = rec["metrics"]
    print(f"\n  graded on the final route: structured_correct={pm['structured_correct']} "
          f"edge_validity={pm['edge_validity_rate']} hallucination={pm['hallucination_rate']} "
          f"hops={len(pm['parsed_nodes'])}")
    print("  Every hop is a real graph edge — the ratify-then-commit rule is what the tool")
    print("  section buys, and it is why the route can be trusted rather than regex-matched.")
    # The seam preserves the action list in BOTH directions: forward it survives into the
    # compact assistant turn, inverse it goes back to SPINE unwrapped (never buried in answer()).
    print("\n  seam check — inverse translation of each turn's compact output:")
    for t, m in enumerate([m for m in compact_rollout if m["role"] == "assistant"], start=1):
        plan = json.loads(compact_output_to_spine_json(m["content"]))["plan"]
        print(f"    turn {t}: compact_output_to_spine_json -> plan={plan[:110]!r}")
    # Contrast: the tool-free arm on the same pipeline, and what the corpus targets contain.
    no_tools = spine_to_compact_messages(
        raw, include_edges=False, include_tools=False, icl_examples=0)
    print(f"\n  same pipeline, TOOLS OFF (shown on the corpus rollout {args.plan_sample.name}, "
          f"whose target answers\n  directly): no tool section in the prompt, and the plan is a "
          f"bare route\n    {_plan_lines(no_tools)[-1][:110]!r}")
    graph_idx = args.plan_sample.stem.split("_")[1]
    rollouts = _rollouts_for_graph(graph_idx)
    target_plans = [_plan_lines(spine_to_compact_messages(
        r, include_edges=False, include_tools=True, icl_examples=0))[-1] for r in rollouts]
    n_tooling = sum(1 for p in target_plans if not p.startswith("[answer("))
    print(f"\n  corpus check (graph {graph_idx}): {n_tooling}/{len(target_plans)} logged targets "
          f"call a non-answer SPINE action; the rest answer directly — which is why the\n"
          f"  tool-calling behavior above comes from a live run, not from the training corpus.")
    print("\n  the full compact conversation the model saw and produced, turn by turn:")
    _show(compact_rollout, tokenizer, add_generation_prompt=False)

    # --- 3b. SPINE + ICL: the rollout's own few-shot examples, compacted ----------
    with_icl = spine_to_compact_messages(
        raw, include_edges=False, include_tools=tools, icl_examples=max(icl, 2))
    _rule(f"SPINE + ICL  (icl_examples={max(icl, 2)}: leading SPINE examples kept and compacted)")
    sys_msgs = [m for m in with_icl if m["role"] == "system"]
    graph_turns = [i for i, m in enumerate(with_icl) if "Scene graph:" in m["content"]]
    assert len(sys_msgs) == 1, "expected exactly one system message"
    assert "Scene graph:" not in sys_msgs[0]["content"], \
        "with ICL the system message carries the prompt only — graphs stay in user turns"
    assert graph_turns[-1] == max(i for i, m in enumerate(with_icl)
                                 if m["role"] == "user" and "Scene graph:" in m["content"]), \
        "the query graph must be the LAST block (find_last_graph_scope anchor)"
    assert "Agent Role: You are an excellent graph planner" not in render(with_icl), \
        "verbose SPINE system prompt leaked"
    print(f"  input  roles: [{_roles(raw)[:60]}…]  ({len(raw)} SPINE turns)")
    print(f"  output roles: [{_roles(with_icl)}]  ({len(with_icl)} turns)")
    print(f"  'Scene graph:' blocks at message indices {graph_turns}: "
          f"{len(graph_turns) - 1} ICL demo graph(s) then the query graph (last).")
    print("  Each demo keeps its own plans, `updates:` turns and replans — that is what")
    print("  demonstrates tool calling to the model.")
    print(f"\n  demo plans: {[p[:70] for p in _plan_lines(with_icl)[:3]]}")
    _reduction_summary(raw, with_icl, tokenizer)
    _show(with_icl, tokenizer, add_generation_prompt=False)

    # --- 3c. PLAIN-LLM COMPACT: edges in the block + edge-aware system prompt ------
    # The plain-LLM baseline has no GNN, so it consumes the SAME compact format but
    # WITH connectivity written into the scene-graph block (`• Region Edges:` /
    # `• Object Edges:`) and an edge-aware system prompt that points at those edges
    # (no latent-space claim). `include_edges=True` is exactly what the training
    # (data.py) and eval (InMemoryLLM.query_llm) paths pass for the `llm` arch, in
    # all three settings (training, in-training eval, scalability eval). Same input
    # rollout as above; the only change is include_edges.
    compact_llm = spine_to_compact_messages(
        raw, include_edges=True, include_tools=tools, icl_examples=0)
    compact_gnn = spine_to_compact_messages(
        raw, include_edges=False, include_tools=tools, icl_examples=0)
    _rule("PLAIN-LLM COMPACT  (include_edges=True: edges in block + edge-aware system prompt)")
    llm_sys = next(m["content"] for m in compact_llm if m["role"] == "system")
    gnn_sys = next(m["content"] for m in compact_gnn if m["role"] == "system")
    assert "• Region Edges:" in llm_sys and "• Object Edges:" in llm_sys, "edge bullets missing"
    assert "latent" not in llm_sys, "plain-LLM prompt must not claim latent connectivity"
    assert "latent space" in gnn_sys and "• Region Edges:" not in gnn_sys, "graph-aug path changed"
    print("  Same compact pipeline as graph-aug (SPINE system dropped, graph hoisted),")
    print("  but the block now carries the edges and the system prompt cites them:")
    print("\n  -- intro paragraph, GRAPH-AUGMENTED (include_edges=False) --")
    print("   " + gnn_sys.split("\n\n")[0])
    print("\n  -- intro paragraph, PLAIN-LLM (include_edges=True) --")
    print("   " + llm_sys.split("\n\n")[0])
    print("\n  -- scene-graph block (plain-LLM): node bullets + Region/Object Edges --")
    block = llm_sys.split("Scene graph:\n", 1)[1] if "Scene graph:\n" in llm_sys else llm_sys
    print("   Scene graph:\n   " + block.replace("\n", "\n   "))
    _show(compact_llm, tokenizer, add_generation_prompt=False)

    # --- 4. Multi-task over ONE graph: graph in system, tasks stacked -------------
    all_tasks = [t["task"] for t in tasks]
    multi_eval = format_eval_messages(
        graph_dict, all_tasks, include_edges=False, include_tools=tools)
    roles_e = [m["role"] for m in multi_eval]
    _rule(f"MULTI-TASK EVAL  ({len(all_tasks)} tasks stacked under one shared graph)")
    print(f"  {len(multi_eval)} turns: 1 system(graph) + {roles_e.count('user')} stacked user tasks; "
          f"'Scene graph:' block appears {sum(c['content'].count('Scene graph:') for c in multi_eval)}x (once, in system).")
    _reduction_summary([m for t in all_tasks for m in _verbose_baseline(graph_dict, t)], multi_eval, tokenizer)

    # --- 5. Few-shot demos DRAWN FROM THE DATA (a different graph's rollouts) -----
    # icl_demos_from_rollouts parses real logged rollouts into demos, so an eval prompt
    # can be few-shot with in-distribution examples instead of SPINE's canned ones.
    icl_rollouts = _rollouts_for_graph(ICL_GRAPH_INDEX)[:2]
    if icl_rollouts:
        demos = icl_demos_from_rollouts(icl_rollouts, include_tools=tools)
        icl_eval = format_eval_messages(
            graph_dict, task, include_edges=False, include_tools=tools, icl_demos=demos)
        n_demo_turns = sum(len(d["turns"]) for d in demos)
        _rule(f"EVAL PROMPT + DATA-DRAWN ICL  ({len(demos)} demo graph(s) / {n_demo_turns} demo "
              f"task(s) from sample_{ICL_GRAPH_INDEX}_*.json)")
        blocks = [i for i, m in enumerate(icl_eval) if "Scene graph:" in m["content"]]
        assert "Scene graph:" not in icl_eval[0]["content"], "system message must be prompt-only"
        assert blocks[-1] == len(icl_eval) - 1, "query graph must open the LAST turn"
        print(f"  roles: [{_roles(icl_eval)}]; graph blocks at {blocks} "
              f"(demo graphs first, query graph last = the injection anchor).")
        _show(icl_eval, tokenizer, add_generation_prompt=True)

    # --- 6. Multi-task training: several rollouts for ONE graph -------------------
    if len(rollouts) > 1:
        multi_train = assemble_training_conversation(
            rollouts, include_edges=False, include_tools=train_tools)
        roles = [m["role"] for m in multi_train]
        _rule(f"MULTI-TASK TRAINING  ({len(rollouts)} rollouts, graph {graph_idx}, tasks stacked under one graph)")
        print(f"  {len(multi_train)} turns ({roles.count('system')} system(graph) / {roles.count('user')} user / "
              f"{roles.count('assistant')} assistant); 'Scene graph:' block appears "
              f"{sum(m['content'].count('Scene graph:') for m in multi_train)}x (once, in system).")
        _reduction_summary([m for r in rollouts for m in strip_icl(r)], multi_train, tokenizer)

        # Rendered view: how multiple tasks per graph actually look stacked.
        few = assemble_training_conversation(
            rollouts[:3], include_edges=False, include_tools=train_tools)
        n_tasks = sum(1 for m in few if m["role"] == "user")
        _rule(f"MULTIPLE TASKS PER GRAPH — rendered ({n_tasks} tasks stacked under one shared graph)")
        print(f"  one system(graph) message, then {n_tasks} (user task, assistant answer) pairs in the same conversation:")
        _show(few, tokenizer, add_generation_prompt=False)

    print()


if __name__ == "__main__":
    main()
