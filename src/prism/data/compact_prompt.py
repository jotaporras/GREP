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

Eval messages (from ``populated_graphs/data_gen_*.json``) — one ``user`` turn::

    <task>

    Scene graph:
    • Region nodes: hub_1, quarters_1, ...
    • Object nodes: food_dispenser_1, ...
    • Region edges: [hub_1, mess_hall_1], ...
    • Object edges: [food_dispenser_1, mess_hall_1], ...
    • Robot location: hub_1

Training messages (from ``generated_plans/sample_*.json``) — add an ``assistant``
turn whose content wraps the reasoning in thinking tokens::

    <think>Relevant graph: hub_1, mess_hall_1, food_dispenser_1

    Reasoning: <reasoning chain></think><answer() unwrapped to its inner prose>

Multiple tasks over one graph share a single conversation: the ``Scene graph:``
block is emitted ONCE (first user turn) and later tasks are graph-free user
turns. Besides the obvious context saving, this keeps PE injection correct across
tasks — the single ``Scene graph:`` block is the only ``find_last_graph_scope``
anchor, so injection scopes over every task and answer in the conversation (the
verbose format restates the graph per task and scopes to the last task only).
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


def _edge_list(edges: List[List[str]]) -> str:
    """Render connection pairs compactly as ``[a, b], [c, d], ...``."""
    return ", ".join(f"[{a}, {b}]" for a, b in edges)


def _graph_block(graph_dict: dict) -> str:
    """The scene-graph block stated once per conversation.

    Keeps the ``Scene graph:`` header so it stays the single
    ``find_last_graph_scope`` injection anchor (``gnn_llm.py``). Regions/Objects
    are bulleted node-name lists; edges are the compacted bracket-pair form of
    ``region_connections`` / ``object_connections`` (no coordinates, no JSON
    indentation — far smaller than the verbose dict). Robot location is the
    starting node. This is the LLM-facing text only; the GNN still ingests the
    full SPINE scene-graph JSON unchanged.
    """
    return (
        "Scene graph:\n"
        f"• Region nodes: {_node_names(graph_dict['regions'])}\n"
        f"• Object nodes: {_node_names(graph_dict['objects'])}\n"
        f"• Region edges: {_edge_list(graph_dict['region_connections'])}\n"
        f"• Object edges: {_edge_list(graph_dict['object_connections'])}\n"
        f"• Robot location: {graph_dict['robot_location']}"
    )


def build_conversation(graph_dict: dict, turns: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Assemble a multi-task chat ``messages`` list for ONE shared graph.

    ``turns`` is an ordered list of ``{"task": str, "assistant": Optional[str]}``.
    The graph's ``Regions:`` / ``Objects:`` node lists are emitted EXACTLY ONCE,
    appended to the first user turn; every later user turn carries only its task,
    because the graph already sits in the conversation history. Each turn with a
    non-empty ``"assistant"`` adds an assistant turn after its user turn.

    Stating the node lists once is what makes the compact format correct for
    multiple tasks over the same graph: the single graph block is the only
    injection-scope anchor (``find_last_graph_scope``), so PE injection scopes
    over the whole conversation and every task's / answer's node mentions bind to
    the one graph — unlike the verbose format, which restates the graph per task
    and so scopes injection to the last task only.

    Roles are real message turns (not literal ``User:``/``Assistant:`` text): the
    chat template supplies the delimiters as native special tokens, and the SFT
    masker (``train_on_responses_only``) trains on each assistant turn in turn.
    """
    if not turns:
        raise ValueError("build_conversation requires at least one turn")
    messages: List[Dict[str, str]] = []
    for i, turn in enumerate(turns):
        task = turn["task"].strip()
        content = f"{task}\n\n{_graph_block(graph_dict)}" if i == 0 else task
        messages.append({"role": "user", "content": content})
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
    """Translate one SPINE user turn's content to compact form.

    - Turn with an embedded scene graph -> ``{task}\\n\\n{scene-graph block}``.
    - ``task:`` / request turn without a graph -> the task text only.
    - Anything else (``updates: …`` / ``Feedback: …``) -> kept verbatim (already
      short, and carries no graph to compact).
    """
    if re.search(r"[Ss]cene graph:", content):
        task = _extract_task(content)
        graph_dict = _extract_scene_graph_dict(content)
        return f"{task}\n\n{_graph_block(graph_dict)}"
    stripped = content.lstrip()
    if re.match(r"task:", stripped, flags=re.IGNORECASE):
        return _extract_task(content)
    return content


def spine_to_compact_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """FORWARD translator: a SPINE ``messages`` list -> compact ``messages``.

    Per turn: ``system`` is dropped (the compact format carries no system text,
    on both train and eval, so they stay consistent); ``user`` turns are
    compacted by :func:`_compact_user_content`; ``assistant`` turns that are the
    4-key SPINE-JSON answer become the ``<think>…</think>plan`` form via
    :func:`_format_assistant` (any other assistant content is kept verbatim).

    A ``Scene graph:`` block is emitted wherever the source turn had one, so
    multi-graph ICL keeps its own blocks and ``find_last_graph_scope`` still
    resolves to the last (query) graph — matching the GNN's ``pyg_graphs[-1]``.
    """
    out: List[Dict[str, str]] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        if role == "assistant":
            content = m.get("content", "")
            try:
                content = _format_assistant(content)
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError, ValueError):
                pass  # not a parseable 4-key SPINE answer — keep the turn verbatim
            out.append({"role": "assistant", "content": content})
        else:
            out.append({"role": "user", "content": _compact_user_content(m.get("content", ""))})
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
