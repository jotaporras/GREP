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
    """Comma-join undirected edges as ``u <-> v`` from a list of ``[u, v]`` pairs.

    Used for the ``• Region Edges:`` / ``• Object Edges:`` bullets in the
    plain-LLM compact block (``include_edges=True``): ``region_connections`` are
    region<->region borders and ``object_connections`` bind each object to its
    region. Malformed (non-pair) entries are skipped rather than raising.
    """
    return ", ".join(f"{p[0]} <-> {p[1]}" for p in pairs if len(p) == 2)


# Shared <think>…</think> output contract — identical across both system-prompt
# variants below; only the scene-graph intro paragraph differs.
_COMPACT_ANSWER_CONTRACT = (
    "Answer in two parts, in this exact order:\n"
    "1. One <think> … </think> block that holds ALL of your reasoning. Begin it with "
    '"Relevant graph:" followed by the specific nodes this task depends on — never leave '
    'this blank — then "Reasoning:" followed by your step-by-step path-finding. Every bit '
    "of thinking goes inside this block, never after it.\n"
    "2. Immediately after </think>, give the final plan only: the route the robot follows, "
    "its nodes in order joined by arrows (for example, "
    "start_region -> middle_region -> goal_region). Put nothing else there — no further "
    "reasoning, no labels, and no second <think> block.\n\n"
    "Several tasks may be asked about this same scene graph in turn; answer each one "
    "independently in this same format."
)


# Short, compact-format system prompt prepended to the scene-graph block in the
# leading system message (both training and eval). States the <think>…</think>
# output contract. Used by the GRAPH-AUGMENTED archs: edges are NOT in the block
# (the GNN supplies connectivity), so the intro says connectivity is available in
# latent space. Set to "" to run the graph-only baseline for an A/B.
COMPACT_SYSTEM_PROMPT = (
    "You are a navigation planner for a mobile robot. Below is a scene graph listing the "
    "environment's regions, its objects, and the robot's starting location. The connections "
    "between these nodes — which regions border one another, and which region each object is "
    "in — are available to you in latent space; reason over reachability and paths from that "
    "latent access even though the connecting edges are not written out here.\n\n"
    + _COMPACT_ANSWER_CONTRACT
)


# Plain-LLM variant (``include_edges=True`` / no GNN): the edges ARE written out
# in the block as ``• Region Edges:`` / ``• Object Edges:``, and there is no
# latent pathway — so the intro points the model at those listed edges instead of
# a latent-space claim. The answer contract is identical to the graph-aug prompt.
COMPACT_SYSTEM_PROMPT_WITH_EDGES = (
    "You are a navigation planner for a mobile robot. Below is a scene graph listing the "
    "environment's regions, its objects, the robot's starting location, and the connections "
    "between these nodes — which regions border one another (listed under Region Edges), and "
    "which region each object is in (listed under Object Edges). Reason over reachability and "
    "paths using these listed edges.\n\n"
    + _COMPACT_ANSWER_CONTRACT
)


def _graph_block(graph_dict: dict, include_edges: bool = False) -> str:
    """The scene-graph block, emitted once as a leading system message.

    Keeps the ``Scene graph:`` header so it stays the single
    ``find_last_graph_scope`` injection anchor (``gnn_llm.py``). Regions/Objects
    are bulleted node-name lists plus the robot's starting node.

    With ``include_edges=False`` (graph-augmented archs) NO edges are written:
    the GNN already ingests connectivity from the full SPINE scene-graph JSON, so
    edges in text would be redundant. With ``include_edges=True`` (the plain-LLM
    baseline, which has no GNN) the block additionally lists ``• Region Edges:``
    (region<->region borders) and ``• Object Edges:`` (object<->region) so the
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


def _system_content(graph_dict: dict, include_edges: bool = False) -> str:
    """Leading system-message content: the system prompt + the scene-graph block.

    The prompt precedes the ``Scene graph:`` block, so ``find_last_graph_scope``
    still anchors on the block and PE injection is unaffected. With
    ``COMPACT_SYSTEM_PROMPT == ""`` this is the graph block alone (A/B baseline).
    ``include_edges`` adds the edge bullets to the block AND selects the plain-LLM
    system prompt (``COMPACT_SYSTEM_PROMPT_WITH_EDGES``), which points at those
    edges instead of claiming latent connectivity.
    """
    prompt = COMPACT_SYSTEM_PROMPT_WITH_EDGES if include_edges else COMPACT_SYSTEM_PROMPT
    block = _graph_block(graph_dict, include_edges=include_edges)
    return f"{prompt}\n\n{block}" if prompt else block


def build_conversation(
    graph_dict: dict, turns: List[Dict[str, str]], include_edges: bool = False
) -> List[Dict[str, str]]:
    """Assemble a chat ``messages`` list for ONE shared graph.

    ``turns`` is an ordered list of ``{"task": str, "assistant": Optional[str]}``.
    The scene graph is emitted EXACTLY ONCE as a leading ``system`` message, above
    the first task; every task is then a bare ``user`` turn, so tasks (and their
    answers) simply stack in the same conversation per graph. Each turn with a
    non-empty ``"assistant"`` adds an assistant turn after its user turn.

    Putting the graph in the system message (above the first task) is what lets
    tasks stack and keeps the compact format correct for multiple tasks: that
    single block is the only ``find_last_graph_scope`` injection anchor, so PE
    injection scopes over the whole conversation and every task's / answer's node
    mentions bind to the one graph. It also keeps the graph out of the loss
    (only assistant turns are trained), and stays consistent with eval where new
    tasks are appended after each answer.

    Roles are real message turns (not literal ``User:``/``Assistant:`` text): the
    chat template supplies the delimiters as native special tokens, and the SFT
    masker (``train_on_responses_only``) trains on each assistant turn in turn.
    """
    if not turns:
        raise ValueError("build_conversation requires at least one turn")
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": _system_content(graph_dict, include_edges=include_edges)}
    ]
    for turn in turns:
        messages.append({"role": "user", "content": turn["task"].strip()})
        if turn.get("assistant"):
            messages.append({"role": "assistant", "content": turn["assistant"]})
    return messages


def format_eval_messages(graph_dict: dict, tasks) -> List[Dict[str, str]]:
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
    """
    task_list = [tasks] if isinstance(tasks, str) else list(tasks)
    return build_conversation(graph_dict, [{"task": t} for t in task_list])


def append_followup_task(messages: List[Dict[str, str]], task: str) -> List[Dict[str, str]]:
    """Append a graph-free follow-up ``user`` turn for the same graph.

    Used by a multi-task eval driver between planning calls: the graph node lists
    already live earlier in ``messages``, so a follow-up task restates only the
    task. Returns a new list (does not mutate the input).
    """
    return messages + [{"role": "user", "content": task.strip()}]


def format_training_messages(conversation: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Build a compact training example (one SPINE rollout) as ``messages``.

    ``conversation`` is a logged rollout (``[system] + ICL turns + real task``);
    :func:`assemble_training_conversation` handles the general multi-rollout case.
    Equivalent to ``assemble_training_conversation([conversation])``.
    """
    return assemble_training_conversation([conversation])


def assemble_training_conversation(
    rollouts: List[List[Dict[str, str]]]
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
    """
    if not rollouts:
        raise ValueError("assemble_training_conversation requires >= 1 rollout")
    graph_dict = None
    graph_names = None
    turns: List[Dict[str, str]] = []
    for convo in rollouts:
        msgs = strip_icl(convo)
        user_turn = next(m for m in reversed(msgs) if m["role"] == "user")
        assistant_turn = next(m for m in reversed(msgs) if m["role"] == "assistant")
        sg = _extract_scene_graph_dict(user_turn["content"])
        names = {n["name"] for n in sg["regions"] + sg["objects"]}
        if graph_dict is None:
            graph_dict, graph_names = sg, names
        elif names != graph_names:
            raise ValueError(
                "assemble_training_conversation got rollouts for different graphs "
                f"(node sets differ by {names ^ graph_names}); each call is one graph."
            )
        turns.append({
            "task": _extract_task(user_turn["content"]),
            "assistant": _format_assistant(assistant_turn["content"]),
        })
    return build_conversation(graph_dict, turns)


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


def _format_assistant(assistant_content: str) -> str:
    """Render the assistant JSON answer as the compact target content.

    The think block holds the reasoning scaffolding, led by the relevant graph:
    ``<think>Relevant graph: …\n\nReasoning: …</think>``. It uses the
    ``<think>…</think>`` convention Ollama parses; Llama-3.1 has no native thinking
    template, so this SFT target is what teaches the model to emit the markers.
    The plan text comes immediately after the closing tag (no label). No
    ``Assistant:`` prefix — the role delimiter is supplied by the chat template.
    """
    # strict=False tolerates literal control chars inside JSON strings, matching
    # SPINE's own try_parse (some rollouts embed raw newlines/tabs in reasoning).
    answer = json.loads(_strip_code_fence(assistant_content), strict=False)
    # Rollouts aren't always typed consistently: relevant_graph / plan / reasoning
    # may be a list instead of a string. Coerce so the compact target builds for
    # every rollout (this runs inside preprocess_dataset on real training data).
    reasoning = _as_text(answer["reasoning"]).strip()
    relevant = _as_text(answer["relevant_graph"]).strip()
    plan = _unwrap_plan(_as_text(answer["plan"]))
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
# Live translator: SPINE messages <-> compact text at the LLM seam.
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
    messages: List[Dict[str, str]], include_edges: bool = False
) -> List[Dict[str, str]]:
    """FORWARD translator: a SPINE ``messages`` list -> compact ``messages``.

    ``include_edges`` adds ``• Region Edges:`` / ``• Object Edges:`` bullets to
    the hoisted scene-graph system block — used by the plain-LLM baseline, which
    has no GNN and must read connectivity from text. Graph-augmented archs leave
    it False (their GNN supplies connectivity from the original SPINE JSON).

    Drops BOTH the verbose SPINE system prompt and any few-shot ICL examples
    (dropping ICL keeps train/eval symmetric), then hoists the query's scene graph
    into a leading ``system`` message — prefixed with the short compact-format
    system prompt (``_system_content``) — so the format is
    ``[system(prompt + scene graph)] + [user task, assistant answer]*`` and tasks
    stack.

    The real task is the LAST ``user`` turn carrying a ``Scene graph:`` block
    (each ICL example precedes it with its own graph; receding-horizon planning
    turns — ``updates:`` / replans — follow it). We keep that turn onward, lift
    its graph to the system message and reduce it to the task text, and translate
    the rest. ``strip_icl`` is NOT reused: the SPINE eval query turn has no
    ``task:`` prefix, so the scene-graph marker is the reliable boundary.

    Per surviving turn: the graph-bearing query ``user`` turn -> system(graph) +
    user(task); other ``user`` turns -> :func:`_compact_user_content`;
    ``assistant`` 4-key SPINE-JSON answers -> the ``<think>…</think>plan`` form
    via :func:`_format_assistant` (other assistant content kept verbatim).
    """
    graph_idxs = [
        i for i, m in enumerate(messages)
        if m.get("role") == "user" and re.search(r"[Ss]cene graph:", m.get("content", ""))
    ]
    if graph_idxs:
        messages = messages[graph_idxs[-1]:]  # last (query) graph onward; drops ICL + system

    out: List[Dict[str, str]] = []
    graph_done = False
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        if role == "user":
            content = m.get("content", "")
            if not graph_done and re.search(r"[Ss]cene graph:", content):
                out.append({"role": "system", "content": _system_content(
                    _extract_scene_graph_dict(content), include_edges=include_edges)})
                out.append({"role": "user", "content": _extract_task(content)})
                graph_done = True
            else:
                out.append({"role": "user", "content": _compact_user_content(content)})
            continue
        # assistant
        content = m.get("content", "")
        try:
            content = _format_assistant(content)
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError, ValueError):
            pass  # not a parseable 4-key SPINE answer — keep the turn verbatim
        out.append({"role": "assistant", "content": content})
    return out


def compact_output_to_spine_json(text: str) -> str:
    """INVERSE translator: a model's compact output -> a SPINE-JSON string.

    Parses ``<think>Relevant graph: R\\n\\nReasoning: Y</think>Z`` (tolerant of a
    missing think block / labels) and returns a JSON string with the four keys
    the SPINE parser and grader expect. The plan ``Z`` is re-wrapped as
    ``[answer(Z)]`` — the exact inverse of :func:`_unwrap_plan` — which SPINE's
    ``preprocess_cmd_str`` / ``_try_parse_command`` round-trip through commas,
    arrows, and parens, and whose ``str()`` the answer-regex still matches.
    ``primary_goal`` is synthesized empty: the compact format drops it and the
    grader only checks key *presence* (``_has_correct_keys``).
    """
    think = re.search(r"<think>(.*?)</think>(.*)", text, flags=re.DOTALL)
    if think:
        inner, plan = think.group(1), think.group(2)
    else:
        # No think block: treat everything as the plan, nothing as reasoning.
        inner, plan = "", text

    rel = re.search(r"Relevant graph:\s*(.*?)(?:\n\s*\n|Reasoning:|$)", inner, flags=re.IGNORECASE | re.DOTALL)
    rea = re.search(r"Reasoning:\s*(.*)", inner, flags=re.IGNORECASE | re.DOTALL)
    relevant_graph = rel.group(1).strip() if rel else ""
    reasoning = rea.group(1).strip() if rea else inner.strip()

    plan = plan.strip()
    return json.dumps({
        "primary_goal": "",
        "relevant_graph": relevant_graph,
        "reasoning": reasoning,
        "plan": f"[answer({plan})]",
    })
