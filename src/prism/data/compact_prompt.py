"""Compact plain-text prompt formatter.

Converts the verbose SPINE JSON prompts (long system prompt + full scene-graph
JSON with coordinates and every edge) into the compact plain-text format the
GREP-PRISM models actually need to consume. The graph topology is dropped from
the text because it is already delivered to the model through the GNN pathway
(``scene_graph_dict_to_pyg`` -> R-PEARL / composite graph -> injection map that
binds node-name token spans to graph nodes); the text only needs the task, the
node *names* (so the injection map has tokens to bind to) and, for training, the
target answer.

The output is a chat ``messages`` list — the same shape the training pipeline
(``preprocess_dataset`` -> ``apply_chat_template``) and the eval client
(``GraphAugmentedInMemoryLLM.query_llm`` -> ``apply_chat_template``) already
consume. The ``User:`` / ``Assistant:`` turn delimiters are NOT literal text:
they are supplied by ``tokenizer.apply_chat_template`` as the model's native role
special tokens (for Llama-3, ``<|start_header_id|>user<|end_header_id|>`` … ),
and the open assistant turn at eval comes from ``add_generation_prompt=True``.
This module only fills the per-turn *content*.

A short compact-format system prompt (``COMPACT_SYSTEM_PROMPT``) plus the scene
graph form a leading ``system`` message; tasks then stack as ``user``/``assistant``
pairs in the same conversation per graph::

    [system]  <compact system prompt: <think>…</think> contract + latent-connectivity note>

              Scene graph:
              • Region nodes: hub_1, quarters_1, ...
              • Object nodes: food_dispenser_1, ...
              • Robot location: hub_1

    [user]    <task>

    [assistant]  <think>Relevant graph: hub_1, mess_hall_1, food_dispenser_1

                 Reasoning: <reasoning chain></think><plan, answer() unwrapped>

    [user]    <next task>           # stacks: graph already in the system message
    [assistant]  <think>…</think><plan>

Edges in the block are conditional (``include_edges``). Graph-augmented
architectures omit them — their GNN already ingests connectivity from the full
SPINE JSON — so the block carries node names only. The plain-LLM baseline uses
the SAME compact format but WITH ``• Region Edges:`` / ``• Object Edges:``
bullets, since it has no GNN to supply connectivity and must read the edges from
text. The single ``Scene graph:`` system block is the only
``find_last_graph_scope`` anchor, so PE injection scopes over every stacked task
and answer; putting it in the system message also keeps it out of the loss.
``build_conversation`` / ``assemble_training_conversation`` construct these; the
single-task helpers are thin wrappers.

Two further policy args, both required (no library-level defaults), mirror
``include_edges``:

``include_tools`` — SPINE tool calling. True inserts the SPINE API tutorial
(``_SPINE_TOOLS_SECTION``) into the system prompt, has the route checked by ``goto``
before ``answer()`` reports it, and keeps the assistant plan as a SPINE
action list (``[goto(x), answer(a -> b)]``). False adds NOTHING: that arm is the
pre-SPINE prompt exactly — intro + answer contract, plan unwrapped to the bare route
— so the tool-free baseline never drifts. WITHIN eval it must agree with the simulator:
``evaluate._spine_tools_disabled()`` (``PRISM_DISABLE_SPINE_TOOLS``) swaps
``GraphSim`` for ``_NoToolsGraphSim`` and is the same switch that sets this flag.
ACROSS train and eval it may differ, and by default does: the deployed configuration
trains tool-free (``data.spine_tools=none``) and evaluates with the API live, which
the seam absorbs — :func:`compact_output_to_spine_json` wraps the bare route the
model emits as ``[answer(route)]`` for the planner.

``icl_examples`` (``spine_to_compact_messages``) — how many of the leading SPINE
few-shot examples to KEEP and compact (0 = drop them all, the historical
behavior; -1 = keep all). When ICL is kept the scene graph is NOT hoisted to the
system message: it stays inline at the head of the query ``user`` turn, so the
query block remains the LAST ``Scene graph: •`` match and ``find_last_graph_scope``
still scopes PE injection to the query graph rather than an ICL example's graph.
``build_conversation(..., icl_demos=…)`` lays out constructed conversations the
same way.

This module is pure pre-processing: it never touches anything under ``data/``.
"""

import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Union


def try_load_json(file: Union[str, Path]):
    """Load a sample JSON, tolerating the trailing-string corruption seen in
    some rollouts (a few junk lines appended after the JSON value).

    Mirrors ``prism.data.utils.try_load_json`` but kept local so this
    plain-text formatter has no dependency on the heavy data stack
    (torch_geometric / spine) that ``utils`` imports at module load.
    """
    with open(file) as f:
        content = f.read()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return json.loads("".join(content.split("\n")[:-4]))


def strip_icl(msgs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Strip the few-shot ICL prefix from a logged SPINE rollout.

    A rollout is ``[system] + ICL example turns + real task turns``. Every ICL
    example and the real task start with a ``user`` message whose content begins
    with ``task:``; the real task is the *last* such message. Returns
    ``[system] + msgs[real_task_idx:]``.

    Mirrors ``prism.data.utils.strip_icl`` (kept local; see ``try_load_json``).
    """
    if not (isinstance(msgs, list) and msgs):
        raise ValueError("rollout must be a non-empty list of messages")
    task_idx = next(
        (
            i
            for i in range(len(msgs) - 1, -1, -1)
            if msgs[i].get("role") == "user"
            and msgs[i].get("content", "").lstrip().lower().startswith("task:")
        ),
        None,
    )
    if task_idx is None:
        raise ValueError("no `task:` user message found in rollout")
    if not any(m.get("role") == "assistant" for m in msgs[task_idx + 1:]):
        raise ValueError("no assistant turn after the real task")
    head = [msgs[0]] if msgs[0].get("role") == "system" else []
    return head + msgs[task_idx:]


def _node_names(entries: List[dict]) -> str:
    """Comma-join the ``name`` field of a list of node entries."""
    return ", ".join(e["name"] for e in entries)


def _edges(pairs: List) -> str:
    """Comma-join undirected edges as ``u <=> v`` from a list of ``[u, v]`` pairs.

    Used for the ``• Region Edges:`` / ``• Object Edges:`` bullets in the
    plain-LLM compact block (``include_edges=True``): ``region_connections`` are
    region<=>region borders and ``object_connections`` bind each object to its
    region. Malformed (non-pair) entries are skipped rather than raising.
    """
    return ", ".join(f"{p[0]} <=> {p[1]}" for p in pairs if len(p) == 2)


# Scene-graph intro paragraph — the only part that varies with ``include_edges``.
# Graph-augmented archs (edges omitted from the block) get the latent-connectivity
# note; the plain-LLM baseline is pointed at the edge bullets instead.
#
# The latent note is the ONE deliberate change to the tool-free prompt: everything
# else in that arm is byte-identical to the pre-SPINE original, so the tool-free
# baseline stays comparable. It states that the connectivity is reachable through the
# node names and then spends most of its length on VERIFICATION and recovery — check the
# route, drop a step that does not hold, take a longer supported path over a short
# unsupported one — rather than on insisting the model already knows the whole graph.
_INTRO_LATENT = (
    "You are a navigation planner for a mobile robot. Below is a scene graph listing the "
    "environment's regions, its objects, and the robot's starting location. The connections "
    "between these nodes — which regions border one another, and which region each object is "
    "in — are available to you in latent space; reason over reachability and paths from that "
    "latent access even though the connecting edges are not written out here.\n\n"
    "About that latent access, and how to check your work with it:\n"
    "• Read the connections off the node names themselves rather than off a written list. "
    "Recover the neighbors of the robot's location, walk outward node by node toward the "
    "goal, and read off a connected route.\n"
    "• Verify the route before you give it. Every node you name must be one that appears in "
    "the lists below, and every step between consecutive nodes must be one your latent "
    "reading of the graph supports.\n"
    "• If a step does not hold up, do not keep it. Correct that step, or look for an "
    "alternative route to the same goal — a longer path you can support is better than a "
    "short one you cannot.\n"
    "• Naming similarity is not adjacency, and neither is nearness in coordinates. Two nodes "
    "sounding related is not a connection between them, and a route assembled from names "
    "that merely look like they belong together is the most common way to get this wrong.\n"
    "• Work the same way for objects: find the region the object sits in, then route to that "
    "region. The object itself is not a step in the path.\n"
    "• If no route you can support gets all the way to the goal, give the part you can "
    "support and say where it stops, rather than filling the gap with a step you cannot.\n"
    "• State the route you settle on plainly, with every intermediate node in order."
)

_INTRO_WITH_EDGES = (
    "You are a navigation planner for a mobile robot. Below is a scene graph listing the "
    "environment's regions, its objects, the robot's starting location, and the connections "
    "between these nodes — which regions border one another (listed under Region Edges), and "
    "which region each object is in (listed under Object Edges). Reason over reachability and "
    "paths using these listed edges."
)


# ``include_tools=True``: the SPINE API is live. This section is a TUTORIAL, not a
# reminder — the deployed model is trained tool-free (``data.spine_tools=none``), so it
# learns the action-list FORMAT from this text. It is deliberately not framed as a new
# capability: ``goto`` is all over the training targets and the rest of the actions appear
# in the ICL examples, so the model already knows these calls. What it needs is the
# syntax, the argument rules, and when each action is the right one.
#
# SOURCE OF TRUTH: ``spine/prompts/api.py`` (spliced verbatim into SPINE's own system
# prompt at ``spine/prompts/base.py:62``). Both of its groups are reproduced here:
#   - the PLANNING API, the 8 actions enforced by ``VALID_ACTIONS``
#     (``spine/spine.py:16-27``) — anything else is rejected with feedback + a replan;
#   - the GRAPH UPDATE API, which the model only ever RECEIVES, emitted by
#     ``GraphUpdate.form_updates()`` (``spine/spine_util.py:84-156``).
# Argument validation mirrors ``spine/spine.py:28-33, 202-315`` (region vs object node,
# existence + reachability, numeric coords) so a plan is not spent on a rejection.
# ``navigation_update(...)`` / ``add_node(...)`` / ``add_connection(...)`` are named
# because SPINE really emits them (``spine_util.py:144``, ``:193-198`` via
# ``get_add_connection_update_str`` at ``:161-165``) even though ``api.py`` never declares
# them; the model would otherwise meet undocumented calls in its update stream. Corpus
# audit: ``navigation_update`` occurs in real n_10 rollouts, and it is ADVISORY text
# (``form_updates`` wraps ``freeform_updates``), not a graph mutation — the prompt says so,
# because reading it as a graph fact would corrupt the model's map.
# The receding-horizon model is what ``planning_sim.PlanningSim`` actually runs. Routes are
# checked with ``goto``, not ``map_region``: both sit in ``NAVIGATION_ACTIONS`` and so both
# require reachability (``spine.py:243-265``), which makes ``map_region`` circular as a
# ratifier — it presupposes the path it would be testing. ``goto`` runs the graph search and
# fails loudly with the closest bridging pair. The corpus agrees: across n_30 targets
# ``goto`` appears 103 times and ``map_region`` 0; across n_10, 224 vs 11.
#
# ``include_tools=False`` has NO counterpart section: that arm is the original
# pre-SPINE prompt (intro + answer contract), unchanged.
_SPINE_TOOLS_SECTION = (
    "PLANNING API — the only eight actions you may put in a plan. Anything else is rejected "
    "and you have to replan:\n"
    "• goto(region_node: str) -> None: navigate to region_node. A graph search finds the most "
    "efficient path there, so call it on the goal region, not on each intermediate hop. If no "
    "path exists the plan is rejected, and the feedback names the closest pair of nodes "
    "bridging the two disconnected parts of the graph.\n"
    "• map_region(region_node: str) -> List[str]: navigate to region_node and reveal its "
    "neighboring nodes (objects and regions) and the region's own description. It does NOT reveal "
    "object attributes, and it cannot add connections. Returns graph updates.\n"
    "• extend_map(x_coordinate: int, y_coordinate: int) -> List[str]: try to add a region node at "
    "those coordinates. Use it when the goal is far away (over about 10 meters). If that spot is "
    "not physically feasible — an obstacle, say — the closest feasible region is added instead. "
    "Returns graph updates.\n"
    "• explore_region(region_node: str, exploration_radius_meters: float = 3) -> List[str]: "
    "explore within that radius around region_node. Only call it once you are close to your goal, "
    "inside the radius. Returns graph updates.\n"
    "• inspect(object_node: str, vlm_query: str) -> List[str]: ask a vision-language model about "
    "an object. Callable on ANY object node without navigating there first — proximity is handled "
    "for you — and it is the only way to get object-level attributes. Keep the query concise. "
    "Returns graph updates.\n"
    "• replan() -> None: a placeholder meaning \"I will update the plan once I have the new "
    "information\". It is never executed directly; use it to close a plan whose later steps depend "
    "on nodes you have not discovered yet, instead of guessing a stand-in node.\n"
    "• answer(answer: str) -> None: answer the instruction. Terminal — it ends the task. Never "
    "write \"I will replan\" inside an answer, and always name the relevant objects or locations "
    "you identified.\n"
    "• clarify(question: str) -> None: ask for clarification. Use it when the instruction never "
    "identifies what to act on, and it is then your ONLY action for that turn.\n\n"
    "ARGUMENT RULES — break one and the plan is rejected with feedback instead of executed:\n"
    "• goto, map_region and explore_region take a REGION node; inspect takes an OBJECT node. "
    "Passing a region to inspect is rejected, with map_region or explore_region suggested "
    "instead.\n"
    "• For all four, the node must already exist in the graph AND be reachable from the robot's "
    "current location — inspect included, even though it needs no navigation of its own. If it "
    "is not reachable, extend_map toward the bridging pair named in the feedback first.\n"
    "• extend_map takes numeric coordinates (x, y); explore_region takes (region_node, radius).\n"
    "• clarify is the whole plan for its turn when you use it.\n\n"
    "GRAPH UPDATE API — you RECEIVE these, you never call them. After an action runs you get a "
    "user turn reporting what changed, written with these functions: add_nodes(...), "
    "remove_nodes(...), add_connections(...), remove_connections(...), "
    "update_robot_location(region_node), update_node_attributes(...), and no_updates() when "
    "nothing changed. Read them as facts about the graph and fold them into your next plan. "
    "Singular add_node(...) and add_connection(...) mean the same as their plural forms.\n"
    "One entry in that stream is NOT a graph fact: navigation_update(...) carries free-form "
    "advice about your own planning — for example that you are calling replan too many times "
    "in a row, or should try explore_region rather than extend_map. Treat it as a correction "
    "to your behaviour, not as a change to the graph.\n\n"
    "How your plan runs, step by step: you write a plan as a bracketed, comma-separated list of "
    "these calls. Execution is receding-horizon — only the FIRST action in the list is actually "
    "taken. You then receive the update turn described above and write a fresh plan for the same "
    "task over the updated graph. The task ends the moment your plan reaches answer(), so a plan "
    "may be a single action or a full sequence ending in answer().\n\n"
    "A worked example. Suppose the robot is in start_region and the goal is goal_region, by way "
    "of middle_region. Read the route off the graph, then let goto check it for you:\n"
    "<think>Relevant graph: start_region, middle_region, goal_region\n\n"
    "Reasoning: start_region borders middle_region, which borders goal_region, so the route "
    "holds. goto will take the graph-search path there and reject the plan if it does "
    "not.</think>[goto(goal_region), answer(start_region -> middle_region -> goal_region)]\n"
    "If instead the plan comes back rejected for want of a path, the feedback names the closest "
    "bridging pair; use extend_map on that gap, or route around it, and answer only once a "
    "connected route stands.\n\n"
    "FIRST, decide whether the instruction is plannable at all. It IS plannable if you can point "
    "at what to act on: a named node, an object, coordinates, or a description that picks out one "
    "— \"the area containing the medkit\" and \"a tool near gallery_1\" both name a target. It is "
    "NOT plannable if the instruction never says what or where — \"go get the thing from over "
    "there\" identifies nothing. In that case do not guess a goal and do not explore on spec: emit "
    "clarify(...) as the whole plan for that turn, for example [clarify(Which object should I "
    "fetch, and from which region?)], and stop there.\n"
    "A target you have to DISCOVER first is still plannable, and is not a reason to clarify: "
    "\"the keycard that is not in the graph yet\" or \"whichever room it turns out to be in\" "
    "tells you exactly what to look for. Search for it with map_region or explore_region and "
    "close that plan with replan(). Reach for clarify only when nothing at all is identified.\n\n"
    "When you MUST call a tool:\n"
    "• Put goto(goal_region) in front of the answer() that reports the route. goto runs a graph "
    "search over the real graph, so it succeeds only if a path exists and is rejected with "
    "corrective feedback if none does — that is what checks the route you worked out, and it "
    "costs one action rather than one per hop. NEVER assert an edge you have neither read off "
    "the scene graph nor had accepted by goto; a hallucinated edge is a failed task.\n"
    "• map_region is for LEARNING a region you cannot see into — it reveals a region's "
    "neighbors and description. It is not the way to test a route: it needs the region to be "
    "reachable already, so a map_region that succeeds only tells you what you were trying to "
    "find out. Use it to discover, and goto to confirm.\n"
    "• If a node you need is absent, or a region has no confirmed connection toward the goal, "
    "call explore_region or extend_map first; do not invent the missing link. If a node you need "
    "does not exist yet, you may name it in the Relevant graph line as "
    "unobserved_node(description) — that is prose for your own reasoning, never a call.\n"
    "• NEVER substitute a node that happens to be in the graph for one the task says is missing. "
    "If the task is about something unrecorded, acting on a similarly named node that already "
    "exists is a wrong answer, and passing it to inspect or goto is a rejected plan.\n"
    "• When a later step depends on a node you have not discovered yet, plan only as far as your "
    "knowledge actually reaches and close that plan with replan() — for example "
    "[explore_region(gallery_1, 3), replan()]. You will be given the discovery in the next "
    "updates: turn and can plan the rest then. Do not stretch a plan past the last thing you "
    "know.\n"
    "• answer() repeats only ratified hops, in order.\n"
    "• Keep the reasoning short enough that the plan line always gets written. An unfinished "
    "answer counts as no answer at all."
)


def _answer_contract(include_tools: bool) -> str:
    """The <think>…</think> output contract. Part 1 is shared; part 2 is the plan
    format, which is a SPINE action list with tools on and a bare arrow route with
    tools off (``_format_assistant`` / ``compact_output_to_spine_json`` are the exact
    inverse of this choice).

    The tools-off text is the pre-SPINE original, verbatim — that arm must not drift.
    """
    if include_tools:
        final = (
            "2. Immediately after </think>, give the plan ONLY as a SPINE action list in square "
            "brackets: the actions in execution order, comma-separated, ending in the terminal "
            "answer() once the route has been ratified — for example, "
            "[goto(goal_region), answer(start_region -> middle_region -> goal_region)]. "
            "The answer() argument carries the route itself: its "
            "nodes in order joined by arrows. Put nothing else there — no further reasoning, no "
            "labels, and no second <think> block."
        )
    else:
        final = (
            "2. Immediately after </think>, give the final plan only: the route the robot "
            "follows, its nodes in order joined by arrows (for example, "
            "start_region -> middle_region -> goal_region). Put nothing else there — no further "
            "reasoning, no labels, and no second <think> block."
        )
    return (
        "Answer in two parts, in this exact order:\n"
        "1. One <think> … </think> block that holds ALL of your reasoning. Begin it with "
        '"Relevant graph:" followed by the specific nodes this task depends on — never leave '
        'this blank — then "Reasoning:" followed by your step-by-step path-finding. Every bit '
        "of thinking goes inside this block, never after it.\n"
        + final
        + "\n\nSeveral tasks may be asked about this same scene graph in turn; answer each one "
        "independently in this same format."
    )


def compact_system_prompt(include_edges: bool, include_tools: bool) -> str:
    """The short compact-format system prompt.

    Prepended to the scene-graph block in the leading system message (both training
    and eval). ``include_edges`` selects the intro paragraph (latent connectivity vs
    listed edges).

    ``include_tools=False`` reproduces the PRE-SPINE prompt exactly — intro + answer
    contract, no tool section of any kind — so the tool-free arm is byte-identical to
    the original apart from the deliberately expanded latent-space note in
    ``_INTRO_LATENT``. ``include_tools=True`` inserts the SPINE tutorial between them
    and switches the contract's plan format to the action list.
    """
    intro = _INTRO_WITH_EDGES if include_edges else _INTRO_LATENT
    contract = _answer_contract(include_tools)
    if not include_tools:
        return f"{intro}\n\n{contract}"
    return f"{intro}\n\n{_SPINE_TOOLS_SECTION}\n\n{contract}"


# Tool-free instantiations, kept as module constants for the A/B baseline and for
# tests that assert which intro variant a rendered prompt used. Set either to ""
# via ``_system_content``'s empty-prompt path to run the graph-only baseline.
COMPACT_SYSTEM_PROMPT = compact_system_prompt(include_edges=False, include_tools=False)
COMPACT_SYSTEM_PROMPT_WITH_EDGES = compact_system_prompt(include_edges=True, include_tools=False)


def _graph_block(graph_dict: dict, include_edges: bool) -> str:
    """The scene-graph block, emitted once as a leading system message.

    Keeps the ``Scene graph:`` header so it stays the single
    ``find_last_graph_scope`` injection anchor (``gnn_llm.py``). Regions/Objects
    are bulleted node-name lists plus the robot's starting node.

    With ``include_edges=False`` (graph-augmented archs) NO edges are written:
    the GNN already ingests connectivity from the full SPINE scene-graph JSON, so
    edges in text would be redundant. With ``include_edges=True`` (the plain-LLM
    baseline, which has no GNN) the block additionally lists ``• Region Edges:``
    (region<=>region borders) and ``• Object Edges:`` (object<=>region) so the
    model can read connectivity from text.
    """
    lines = [
        "Scene graph:",
        f"• Region nodes: {_node_names(graph_dict['regions'])}",
        f"• Object nodes: {_node_names(graph_dict['objects'])}",
        f"• Robot location: {graph_dict['robot_location']}",
    ]
    if include_edges:
        lines.append(f"• Region Edges: {_edges(graph_dict.get('region_connections', []))}")
        lines.append(f"• Object Edges: {_edges(graph_dict.get('object_connections', []))}")
    return "\n".join(lines)


def _system_content(graph_dict, include_edges: bool, include_tools: bool) -> str:
    """Leading system-message content: the system prompt + the scene-graph block.

    The prompt precedes the ``Scene graph:`` block, so ``find_last_graph_scope``
    still anchors on the block and PE injection is unaffected. With
    ``compact_system_prompt(...) == ""`` this is the graph block alone (A/B baseline).
    ``include_edges`` adds the edge bullets to the block AND selects the plain-LLM
    intro, which points at those edges instead of claiming latent connectivity;
    ``include_tools`` selects the SPINE-API vs the no-tool-calls section.

    ``graph_dict=None`` yields the prompt alone — the ICL layout, where the graph
    stays inline in the query ``user`` turn so it remains the LAST graph block.
    """
    prompt = compact_system_prompt(include_edges=include_edges, include_tools=include_tools)
    if graph_dict is None:
        return prompt
    block = _graph_block(graph_dict, include_edges=include_edges)
    return f"{prompt}\n\n{block}" if prompt else block


def _task_turn_content(graph_dict, task: str, include_edges: bool) -> str:
    """A ``user`` turn carrying its own scene graph: the block FIRST, then the task.

    Used for ICL demos and, in the ICL layout, for the query turn. Block-first keeps
    the task's (and the following answer's) node mentions at/after the block, so
    ``find_last_graph_scope``'s injection scope still covers them.
    """
    block = _graph_block(graph_dict, include_edges=include_edges)
    return f"{block}\n\n{task.strip()}"


def _icl_demo_messages(icl_demos, include_edges: bool) -> List[Dict[str, str]]:
    """Render few-shot demos as compact ``user``/``assistant`` turns.

    Each demo is ``{"graph": dict, "turns": [{"task": str, "assistant": str}, ...]}``
    (see :func:`icl_demos_from_rollouts`): its FIRST user turn carries that demo's own
    compact scene-graph block, later turns of the same demo are bare tasks — the same
    stacking rule the query section uses. Demo graphs therefore all precede the query
    graph, which keeps the query block last for injection scoping.
    """
    out: List[Dict[str, str]] = []
    for demo in icl_demos:
        for i, turn in enumerate(demo["turns"]):
            content = (
                _task_turn_content(demo["graph"], turn["task"], include_edges)
                if i == 0
                else turn["task"].strip()
            )
            out.append({"role": "user", "content": content})
            if turn.get("assistant"):
                out.append({"role": "assistant", "content": turn["assistant"]})
    return out


def build_conversation(
    graph_dict: dict,
    turns: List[Dict[str, str]],
    include_edges: bool,
    include_tools: bool,
    icl_demos=(),
) -> List[Dict[str, str]]:
    """Assemble a chat ``messages`` list for ONE shared graph.

    ``turns`` is an ordered list of ``{"task": str, "assistant": Optional[str]}``.
    The scene graph is emitted EXACTLY ONCE, above the first task; every later task
    is a bare ``user`` turn, so tasks (and their answers) simply stack in the same
    conversation per graph. Each turn with a non-empty ``"assistant"`` adds an
    assistant turn after its user turn.

    Stating the graph once above the first task is what lets tasks stack and keeps
    the compact format correct for multiple tasks: that single block is the only
    ``find_last_graph_scope`` injection anchor, so PE injection scopes over the whole
    conversation and every task's / answer's node mentions bind to the one graph.

    WHERE that block goes depends on ``icl_demos``:
    - no demos: in the leading ``system`` message, above the first task (which also
      keeps the graph out of the loss — only assistant turns are trained), and
      consistent with eval, where new tasks are appended after each answer;
    - with demos: the system message holds the prompt only, the demos follow (each
      carrying its OWN graph block), and the query graph opens the first query
      ``user`` turn — so the query block is still the LAST one and injection cannot
      scope to an ICL example's graph.

    Roles are real message turns (not literal ``User:``/``Assistant:`` text): the
    chat template supplies the delimiters as native special tokens, and the SFT
    masker (``train_on_responses_only``) trains on each assistant turn in turn.
    """
    if not turns:
        raise ValueError("build_conversation requires at least one turn")
    icl_demos = list(icl_demos)
    messages: List[Dict[str, str]] = [{
        "role": "system",
        "content": _system_content(
            None if icl_demos else graph_dict,
            include_edges=include_edges,
            include_tools=include_tools,
        ),
    }]
    messages.extend(_icl_demo_messages(icl_demos, include_edges=include_edges))
    for i, turn in enumerate(turns):
        content = (
            _task_turn_content(graph_dict, turn["task"], include_edges)
            if icl_demos and i == 0
            else turn["task"].strip()
        )
        messages.append({"role": "user", "content": content})
        if turn.get("assistant"):
            messages.append({"role": "assistant", "content": turn["assistant"]})
    return messages


def format_eval_messages(
    graph_dict: dict, tasks, include_edges: bool, include_tools: bool, icl_demos=()
) -> List[Dict[str, str]]:
    """Build a compact eval prompt as a chat ``messages`` list (no answers).

    ``tasks`` may be a single task string (one ``user`` turn) or a list of task
    strings for the SAME graph (multiple ``user`` turns, graph stated once). At
    eval the open assistant turn comes from
    ``apply_chat_template(..., add_generation_prompt=True)``; for multi-task eval
    the driver appends each model answer as an assistant turn and the next task
    as a (graph-free) user turn — :func:`append_followup_task` does this.

    Parameters
    ----------
    graph_dict : dict
        Scene graph with ``"regions"`` and ``"objects"`` lists of ``{"name": ...}``.
    tasks : str | list[str]
        One task, or several tasks over the one graph.
    include_edges : bool
        Whether to write the ``• Region Edges:`` / ``• Object Edges:`` bullets
        into the scene-graph block (plain-LLM baseline) or omit them
        (graph-augmented archs, whose GNN supplies connectivity). Required: this
        is a policy decision the caller must make, never defaulted here.
    include_tools : bool
        Whether the SPINE tool API is live (API section + action-list plan format)
        or disabled (reason-by-thought, bare-route plan). Required, same rule; must
        agree with the eval simulator (``evaluate._spine_tools_disabled``).
    icl_demos : sequence
        Few-shot demos to place ahead of the query — see :func:`icl_demos_from_rollouts`.
        Empty (default) keeps the historical zero-shot layout with the graph hoisted
        into the system message.
    """
    task_list = [tasks] if isinstance(tasks, str) else list(tasks)
    return build_conversation(
        graph_dict,
        [{"task": t} for t in task_list],
        include_edges=include_edges,
        include_tools=include_tools,
        icl_demos=icl_demos,
    )


def append_followup_task(messages: List[Dict[str, str]], task: str) -> List[Dict[str, str]]:
    """Append a graph-free follow-up ``user`` turn for the same graph.

    Used by a multi-task eval driver between planning calls: the graph node lists
    already live earlier in ``messages``, so a follow-up task restates only the
    task. Returns a new list (does not mutate the input).
    """
    return messages + [{"role": "user", "content": task.strip()}]


def format_training_messages(
    conversation: List[Dict[str, str]],
    include_edges: bool,
    include_tools: bool,
    icl_demos=(),
) -> List[Dict[str, str]]:
    """Build a compact training example (one SPINE rollout) as ``messages``.

    ``conversation`` is a logged rollout (``[system] + ICL turns + real task``);
    :func:`assemble_training_conversation` handles the general multi-rollout case.
    Equivalent to ``assemble_training_conversation([conversation], …)``.
    ``include_edges`` / ``include_tools`` / ``icl_demos`` are threaded straight
    through (see :func:`assemble_training_conversation`).
    """
    return assemble_training_conversation(
        [conversation],
        include_edges=include_edges,
        include_tools=include_tools,
        icl_demos=icl_demos,
    )


def _parse_rollout(convo: List[Dict[str, str]], include_tools: bool):
    """One logged SPINE rollout -> ``(scene_graph_dict, turn)``.

    ``strip_icl``-s the rollout, then reads the task and its scene graph off the real
    user turn and the answer (reasoning / ``relevant_graph`` / ``plan``) off the
    assistant turn. Shared by :func:`assemble_training_conversation` and
    :func:`icl_demos_from_rollouts` so a demo and a target are built identically.
    """
    msgs = strip_icl(convo)
    user_turn = next(m for m in reversed(msgs) if m["role"] == "user")
    assistant_turn = next(m for m in reversed(msgs) if m["role"] == "assistant")
    sg = _extract_scene_graph_dict(user_turn["content"])
    turn = {
        "task": _extract_task(user_turn["content"]),
        "assistant": _format_assistant(assistant_turn["content"], include_tools=include_tools),
    }
    return sg, turn


def icl_demos_from_rollouts(
    rollouts: List[List[Dict[str, str]]], include_tools: bool
) -> List[Dict]:
    """Build few-shot demos from REAL logged rollouts, one demo per graph.

    Each rollout is parsed by :func:`_parse_rollout`; rollouts sharing a graph (same
    node set) are merged into one demo, so a demo shows several tasks stacked under
    its graph exactly as the query section does. The result is what
    ``build_conversation(..., icl_demos=…)`` renders ahead of the query — i.e. the
    few-shot examples are drawn from the dataset (with tool-calling plans intact when
    ``include_tools``), not from a hand-written showcase.
    """
    demos: List[Dict] = []
    names_seen: List[set] = []
    for convo in rollouts:
        sg, turn = _parse_rollout(convo, include_tools=include_tools)
        names = {n["name"] for n in sg["regions"] + sg["objects"]}
        if names in names_seen:
            demos[names_seen.index(names)]["turns"].append(turn)
        else:
            names_seen.append(names)
            demos.append({"graph": sg, "turns": [turn]})
    return demos


def assemble_training_conversation(
    rollouts: List[List[Dict[str, str]]],
    include_edges: bool,
    include_tools: bool,
    icl_demos=(),
) -> List[Dict[str, str]]:
    """Merge per-task SPINE rollouts for ONE graph into a multi-task example.

    Each element of ``rollouts`` is a single-task logged rollout (e.g. the
    ``generated_plans/sample_GGG_*.json`` files for one graph ``GGG``). Each is
    ``strip_icl``-ed, its task and scene graph parsed from the real user turn and
    its answer (reasoning / ``Relevant Nodes:`` / ``Plan:``) from the assistant
    turn. The result is one conversation with the graph stated once followed by a
    ``(user task, assistant answer)`` pair per rollout.

    All rollouts must describe the SAME graph; mismatched node sets raise (fail
    loud) so a mixed-graph batch can't silently corrupt the shared graph block.

    ``include_edges`` / ``include_tools`` are required policy args passed straight to
    :func:`build_conversation`: the first writes the edge bullets into the shared
    scene-graph block (plain-LLM baseline) or omits them (graph-augmented archs); the
    second keeps the SPINE action list in each target plan (and documents the API in
    the system prompt) or unwraps it to a bare route. Neither is defaulted here.
    ``icl_demos`` (from :func:`icl_demos_from_rollouts`) prepends few-shot examples.
    """
    if not rollouts:
        raise ValueError("assemble_training_conversation requires >= 1 rollout")
    graph_dict = None
    graph_names = None
    turns: List[Dict[str, str]] = []
    for convo in rollouts:
        sg, turn = _parse_rollout(convo, include_tools=include_tools)
        names = {n["name"] for n in sg["regions"] + sg["objects"]}
        if graph_dict is None:
            graph_dict, graph_names = sg, names
        elif names != graph_names:
            raise ValueError(
                "assemble_training_conversation got rollouts for different graphs "
                f"(node sets differ by {names ^ graph_names}); each call is one graph."
            )
        turns.append(turn)
    return build_conversation(
        graph_dict,
        turns,
        include_edges=include_edges,
        include_tools=include_tools,
        icl_demos=icl_demos,
    )


def render(messages, tokenizer=None, add_generation_prompt: bool = False) -> str:
    """Render a ``messages`` list to the string the model actually receives.

    With ``tokenizer`` (the accurate path), defers to
    ``tokenizer.apply_chat_template`` so role delimiters appear as the model's
    native special tokens. Without one, falls back to a human-readable
    ``Role:``-labelled view — illustrative only; the real prompt uses the
    template's special tokens, not these ASCII labels.
    """
    if tokenizer is not None:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    parts = [f"{m['role'].capitalize()}:\n{m['content']}" for m in messages]
    if add_generation_prompt:
        parts.append("Assistant:")
    return "\n\n".join(parts)


def _extract_task(user_content: str) -> str:
    """Pull the task text from a user message.

    Strips an optional leading ``task:`` prefix (training/ICL turns have it; the
    eval query turn built by ``_fixed_get_base_prompt`` is ``"{request}\\nAdvice:
    …\\nScene graph:{dict}"`` with no prefix), then cuts at the earliest of
    ``Advice:`` / ``scene graph:`` (case-insensitive), which follow the task.
    """
    body = user_content
    m = re.match(r"\s*task:\s*", body, flags=re.IGNORECASE)
    if m:
        body = body[m.end():]
    cut = re.search(r"\n\s*advice:|scene graph:", body, flags=re.IGNORECASE)
    if cut:
        body = body[: cut.start()]
    return body.strip()


def _extract_scene_graph_dict(text: str) -> dict:
    """Parse the scene graph embedded after a ``Scene graph:`` marker.

    Mirrors the robust parse in ``data.py:_parse_scene_graph``: try a JSON
    ``raw_decode`` from the first ``{``, falling back to ``ast.literal_eval``
    over the balanced-brace slice for rollouts that serialize the graph as a
    single-quoted Python repr. (Kept local to avoid touching the live pipeline;
    could later be DRY'd into ``prism.data.utils``.)
    """
    m = re.search(r"[Ss]cene graph:", text)
    start = text.index("{", m.end())
    tail = text[start:]
    try:
        sg, _ = json.JSONDecoder().raw_decode(tail)
    except json.JSONDecodeError:
        depth, end = 0, None
        for i, ch in enumerate(tail):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            raise
        sg = ast.literal_eval(tail[:end])
    return sg


def _format_assistant(assistant_content: str, include_tools: bool) -> str:
    """Render the assistant JSON answer as the compact target content.

    The think block holds the reasoning scaffolding, led by the relevant graph:
    ``<think>Relevant graph: …\n\nReasoning: …</think>``. It uses the
    ``<think>…</think>`` convention Ollama parses; Llama-3.1 has no native thinking
    template, so this SFT target is what teaches the model to emit the markers.
    The plan text comes immediately after the closing tag (no label). No
    ``Assistant:`` prefix — the role delimiter is supplied by the chat template.

    ``include_tools=True`` keeps the SPINE plan as the action list the rollout logged
    (``[goto(x), answer(a -> b)]``) so the target teaches tool calling; False unwraps
    ``[answer(…)]`` to the bare route (:func:`_unwrap_plan`). Either way it matches
    the plan format ``_answer_contract`` asks for.
    """
    # strict=False tolerates literal control chars inside JSON strings, matching
    # SPINE's own try_parse (some rollouts embed raw newlines/tabs in reasoning).
    answer = json.loads(_strip_code_fence(assistant_content), strict=False)
    # Rollouts aren't always typed consistently: relevant_graph / plan / reasoning
    # may be a list instead of a string. Coerce so the compact target builds for
    # every rollout (this runs inside preprocess_dataset on real training data).
    reasoning = _as_text(answer["reasoning"]).strip()
    relevant = _as_text(answer["relevant_graph"]).strip()
    plan = _as_text(answer["plan"]).strip() if include_tools else _unwrap_plan(
        _as_text(answer["plan"]))
    return (
        f"<think>Relevant graph: {relevant}\n\n"
        f"Reasoning: {reasoning}</think>{plan}"
    )


def _as_text(value) -> str:
    """Coerce a SPINE answer field to text: lists join with ', ', else ``str()``."""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return value if isinstance(value, str) else str(value)


def _strip_code_fence(content: str) -> str:
    """Remove an optional ```json ... ``` markdown fence around the answer."""
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
        content = re.sub(r"\n?```$", "", content.strip())
    return content


def _unwrap_plan(plan: str) -> str:
    """Unwrap ``[answer(<inner>)]`` to ``<inner>``.

    The inner prose already contains the ``a -> b`` route. Non-``answer``
    plans (e.g. ``[goto(x), map_region(x)]``) are returned with only the outer
    brackets stripped.
    """
    body = plan.strip()
    if body.startswith("[") and body.endswith("]"):
        body = body[1:-1].strip()
    m = re.fullmatch(r"answer\((.*)\)", body, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return body


# ---------------------------------------------------------------------------
# Live translator: SPINE messages <=> compact text at the LLM seam.
#
# These are the functions the training (`data.preprocess_dataset`) and eval
# (`models.inference.GraphAugmentedInMemoryLLM.query_llm`) paths call. The GNN
# pathway is untouched: it keeps parsing the full SPINE scene-graph JSON. Only
# the *text* delivered to the LLM is compacted (forward), and the model's
# compact *text* is mapped back to a SPINE-JSON string (inverse) so the existing
# SPINE parser + grader work unchanged.
# ---------------------------------------------------------------------------


def _compact_user_content(content: str) -> str:
    """Translate a graph-free SPINE user turn to compact form.

    - ``task:`` / request turn -> the task text only.
    - Anything else (``updates: …`` / ``Feedback: …``) -> kept verbatim (already
      short). The scene-graph-bearing query turn is handled by
      :func:`spine_to_compact_messages`, which hoists the graph to a system
      message, so this is only ever called on graph-free turns.
    """
    stripped = content.lstrip()
    if re.match(r"task:", stripped, flags=re.IGNORECASE):
        return _extract_task(content)
    return content


def spine_to_compact_messages(
    messages: List[Dict[str, str]],
    include_edges: bool,
    include_tools: bool,
    icl_examples: int,
) -> List[Dict[str, str]]:
    """FORWARD translator: a SPINE ``messages`` list -> compact ``messages``.

    ``include_edges`` adds ``• Region Edges:`` / ``• Object Edges:`` bullets to
    every compact scene-graph block — used by the plain-LLM baseline, which has no
    GNN and must read connectivity from text. Graph-augmented archs leave it False
    (their GNN supplies connectivity from the original SPINE JSON).

    ``include_tools`` selects the system-prompt tool policy (SPINE API + path
    ratification vs no tool calls) and whether assistant plans keep their action
    list; it must match the eval simulator (``evaluate._spine_tools_disabled``).

    ``icl_examples`` is how many of the LEADING SPINE few-shot examples to keep and
    compact — SPINE's header order is EXAMPLE_1, EXAMPLE_2, … so ``2`` reproduces the
    two examples ``evaluate._fixed_get_base_prompt`` sends. ``0`` drops them all (the
    historical behavior) and ``-1`` keeps every one present.

    CAVEAT for ``include_tools=False`` with ICL kept: a demo plan keeps whatever actions
    the rollout logged (``_unwrap_plan`` only unwraps a pure ``[answer(…)]``), and
    SPINE's canned EXAMPLE_1/EXAMPLE_2 DO call ``goto`` / ``map_region``. A tool-free run
    with ICL therefore shows demos that contradict the no-tool-calls instruction — pair
    ``include_tools=False`` with ``icl_examples=0``, or with demos whose plans are pure
    answers (:func:`icl_demos_from_rollouts` over answer-only rollouts).

    The verbose SPINE system prompt is ALWAYS dropped and replaced by the short
    compact one (``_system_content``). An "example" is delimited by a graph-bearing
    ``user`` turn: it runs up to (excluding) the next one, so its receding-horizon
    ``updates:`` / ``Feedback:`` turns and replans travel with it — which is what
    demonstrates tool calling. The LAST graph-bearing turn is the real query (each
    ICL example precedes it; planning turns follow it). ``strip_icl`` is NOT reused:
    the SPINE eval query turn has no ``task:`` prefix, so the scene-graph marker is
    the reliable boundary.

    Layout, per :func:`build_conversation`: with no ICL kept, the query graph is
    hoisted into the leading ``system`` message
    (``[system(prompt + graph)] + [user task, assistant answer]*``). With ICL kept,
    the system message holds the prompt only and every graph — the demos' and the
    query's — stays inline at the head of its own ``user`` turn, so the query block
    remains the LAST ``Scene graph: •`` match for ``find_last_graph_scope``.

    Per surviving turn: graph-bearing ``user`` turns -> block + task text; other
    ``user`` turns -> :func:`_compact_user_content`; ``assistant`` 4-key SPINE-JSON
    answers -> the ``<think>…</think>plan`` form via :func:`_format_assistant` (other
    assistant content kept verbatim).
    """
    graph_idxs = [
        i for i, m in enumerate(messages)
        if m.get("role") == "user" and re.search(r"[Ss]cene graph:", m.get("content", ""))
    ]
    keep_icl = 0
    if graph_idxs:
        n_icl = len(graph_idxs) - 1  # every graph turn before the query opens one example
        keep_icl = n_icl if icl_examples < 0 else min(icl_examples, n_icl)
        # The kept examples are the LEADING ones (SPINE header order EXAMPLE_1, EXAMPLE_2,
        # …), then the query segment; the SPINE system prompt and any examples in between
        # are dropped. Each example runs from its graph turn to the next one.
        messages = (
            messages[graph_idxs[0]:graph_idxs[keep_icl]] + messages[graph_idxs[n_icl]:]
            if keep_icl
            else messages[graph_idxs[n_icl]:]
        )

    out: List[Dict[str, str]] = []
    seen_graphs = 0
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        if role == "user":
            content = m.get("content", "")
            if re.search(r"[Ss]cene graph:", content):
                sg = _extract_scene_graph_dict(content)
                task = _extract_task(content)
                if keep_icl:
                    out.append({"role": "user", "content": _task_turn_content(
                        sg, task, include_edges=include_edges)})
                elif seen_graphs == 0:
                    out.append({"role": "system", "content": _system_content(
                        sg, include_edges=include_edges, include_tools=include_tools)})
                    out.append({"role": "user", "content": task})
                else:  # unreachable with keep_icl == 0 (only the query graph survives)
                    out.append({"role": "user", "content": task})
                seen_graphs += 1
            else:
                out.append({"role": "user", "content": _compact_user_content(content)})
            continue
        # assistant
        content = m.get("content", "")
        try:
            content = _format_assistant(content, include_tools=include_tools)
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError, ValueError):
            pass  # not a parseable 4-key SPINE answer — keep the turn verbatim
        out.append({"role": "assistant", "content": content})
    if keep_icl:
        # Prompt-only system message; each graph (demos' and query's) stays inline.
        out.insert(0, {"role": "system", "content": _system_content(
            None, include_edges=include_edges, include_tools=include_tools)})
    return out


def compact_output_to_spine_json(text: str) -> str:
    """INVERSE translator: a model's compact output -> a SPINE-JSON string.

    Parses ``<think>Relevant graph: R\\n\\nReasoning: Y</think>Z`` (tolerant of a
    missing think block / labels) and returns a JSON string with the four keys
    the SPINE parser and grader expect.

    A bare route ``Z`` (the tool-free plan format) is re-wrapped as ``[answer(Z)]``
    — the exact inverse of :func:`_unwrap_plan` — which SPINE's
    ``preprocess_cmd_str`` / ``_try_parse_command`` round-trip through commas,
    arrows, and parens. A plan that is ALREADY a SPINE action list
    (``[goto(x), answer(a -> b)]``, what the model emits under ``include_tools``) is
    passed through untouched: re-wrapping it would bury the actions inside a single
    ``answer()`` and the planning loop would never execute a tool. Detection is
    flag-free so the seam stays tolerant of a model that answers in either format.

    TRUNCATION: an opened ``<think>`` with no closing tag means generation was cut
    off before the plan line was ever written, so the plan is EMPTY. It is not the
    reasoning. Dumping an unfinished think block into ``answer(...)`` would hand the
    grader a wall of prose whose node mentions the answer-regex can match — scoring a
    truncated generation "correct" off text it happened to contain. Correctness must
    come from a plan the model actually committed to, so there is nothing to grade
    here and the empty plan says exactly that. The partial reasoning is still
    returned under ``reasoning`` for diagnosis.

    ``primary_goal`` is synthesized empty: the compact format drops it and the
    grader only checks key *presence* (``_has_correct_keys``).
    """
    think = re.search(r"<think>(.*?)</think>(.*)", text, flags=re.DOTALL)
    truncated = False
    if think:
        inner, plan = think.group(1), think.group(2)
    elif re.search(r"<think>", text):
        # Opened but never closed: truncated mid-reasoning, no plan was emitted.
        inner, plan, truncated = text.split("<think>", 1)[1], "", True
    else:
        # No think block at all: treat everything as the plan, nothing as reasoning.
        inner, plan = "", text

    rel = re.search(r"Relevant graph:\s*(.*?)(?:\n\s*\n|Reasoning:|$)", inner, flags=re.IGNORECASE | re.DOTALL)
    rea = re.search(r"Reasoning:\s*(.*)", inner, flags=re.IGNORECASE | re.DOTALL)
    relevant_graph = rel.group(1).strip() if rel else ""
    reasoning = rea.group(1).strip() if rea else inner.strip()

    plan = "" if truncated else plan.strip()
    if not plan:
        plan_out = ""                                   # nothing committed to, nothing to grade
    elif _is_action_list(plan):
        plan_out = plan
    else:
        plan_out = f"[answer({plan})]"
    return json.dumps({
        "primary_goal": "",
        "relevant_graph": relevant_graph,
        "reasoning": reasoning,
        "plan": plan_out,
    })


# SPINE actions a plan may open with — the full planning API, mirroring
# ``spine.spine.VALID_ACTIONS`` (spine/spine.py:16-27). Kept as a literal because this
# module is deliberately free of the heavy spine/torch imports; parity with the real
# set is asserted in tests/test_spine_tools_icl.py so the two cannot drift.
_SPINE_ACTIONS = (
    "goto", "map_region", "explore_region", "extend_map", "inspect", "replan",
    "answer", "clarify",
)


def _is_action_list(plan: str) -> bool:
    """True when ``plan`` is already a bracketed SPINE action list.

    Requires the outer brackets AND a leading ``action(`` call, so a bracketed piece
    of prose (or a bare ``a -> b`` route) is still wrapped in ``answer()``.
    """
    body = plan.strip()
    if not (body.startswith("[") and body.endswith("]")):
        return False
    m = re.match(r"\s*([a-zA-Z_]\w*)\s*\(", body[1:-1])
    return bool(m) and m.group(1) in _SPINE_ACTIONS
