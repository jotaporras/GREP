"""Validate a generated plan/route against the scene graph.

Two layers:

1. **Regex + NetworkX (always on).** Parse a route out of the planner's text
   (``a -> b -> c`` arrows or ``goto(a), goto(b)`` actions), exact-match the node
   strings against ``G_Sc.nodes``, and score it with NetworkX:
   ``nodes_exist_rate``, ``edge_validity_rate``, ``full_path_valid``,
   ``start_goal_ok``, ``cost_optimality`` (emitted weighted cost ÷ shortest path
   by ``distance_m``). Malformed/empty input is invalid, never raises.

   **Reasoning check + Gemma path rescue.** The route is searched first in the
   planner's plan, then — when that finds nothing — in its full *reasoning* (the
   model often states the route there), taking the route it commits to LAST. If
   reasoning yields nothing either, the Gemma judge (``write_path_with_judge``)
   recreates the route strictly in ``a -> b -> c`` notation, biased toward the
   final route the planner commits to. The SAME NetworkX diagnostics are re-run on
   whichever route is found. Both fallbacks are one-way: an empty/hallucinated
   route scores as no/invalid path and can never inflate a verdict.
   ``path_from_reasoning`` / ``path_rescued`` flag the source. The Gemma rescue
   (only) is disabled with ``GREP_PATH_RESCUE=0``.

2. **LLM judge (separate, subjective score).** When a task carries an
   ``acceptance_criterion`` (or is a yes/no task), an LLM-as-judge (Gemma 4 E4B,
   ``GEMMA_JUDGE_MODEL``, default ``google/gemma-4-E4B-it``) grades the response
   against that rubric. For acceptance_criterion tasks the judge runs *every*
   evaluation turn (validation and test). The judge score and the RegEx score are
   kept **completely separate** — computed from disjoint inputs, neither reading the
   other: the RegEx/NetworkX accuracy is judge-free, and the *subjective* accuracy is
   the judge's verdict over judged samples only. ``combine_verdict`` computes both
   verdicts plus the ``false_positive``/``false_negative`` diagnostics. If the judge
   cannot load (no weights/auth) those samples are simply absent from the subjective
   score (and the caller warns) — never copied from RegEx. With no
   ``acceptance_criterion`` and a non-yes/no answer, the judge is skipped entirely.

The graph is built with the same coords→Euclidean ``distance_m`` convention as
``scene_graph_parser`` / ``data.utils``; directedness follows the source
(undirected scene graphs by default).
"""

from __future__ import annotations

import math
import os
import re
from typing import Dict, List, Optional

import networkx as nx

# Gemma 4 E4B judge. Defaults to the official gated "google/gemma-4-E4B-it" repo
# (requires an HF auth token with access granted); set GREP_JUDGE_MODEL to override.
GEMMA_JUDGE_MODEL = os.environ.get("GREP_JUDGE_MODEL", "google/gemma-4-E4B-it")

# Route formats: "a -> b -> c" (spec form) or PRISM "goto(a), goto(b)" actions.
_ARROW = re.compile(r"\s*->\s*")
_GOTO = re.compile(r"goto\(\s*([A-Za-z0-9_]+)\s*\)")
_NODE_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def build_graph(graph_dict: dict, directed: bool = False) -> nx.Graph:
    """Build a NetworkX graph from a scene-graph dict.

    Nodes carry ``coords``; edges carry ``distance_m`` (Euclidean between coords,
    or 1.0 when coords are unavailable). ``directed`` builds a ``DiGraph`` so the
    validator can mirror a directed source.
    """
    G = nx.DiGraph() if directed else nx.Graph()
    coords = {}
    for node in (*graph_dict.get("objects", []), *graph_dict.get("regions", [])):
        G.add_node(node["name"], coords=node.get("coords"))
        coords[node["name"]] = node.get("coords")
    for key in ("object_connections", "region_connections"):
        for edge in graph_dict.get(key, []):
            u, v = edge[0], edge[1]
            if u in G and v in G:
                a, b = coords.get(u), coords.get(v)
                dist = (math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
                        if a is not None and b is not None else 1.0)
                G.add_edge(u, v, distance_m=dist)
    return G


_ARROW_CHAIN = re.compile(rf"{_NODE_TOKEN.pattern}(?:\s*->\s*{_NODE_TOKEN.pattern})+")


def parse_path(
    text: str, valid_nodes: Optional[set] = None, *, prefer_last: bool = False
) -> List[str]:
    """Extract the ordered route from planner text.

    Pulls an ``a -> b -> c`` chain (the spec form), then ``goto(...)`` actions.
    Undirected edge statements ``u <=> v`` use ``<=>``, not ``->``, so they never
    collide with a route hop even when a response carries both an edge list and a
    path. By default the LONGEST chain is taken; ``prefer_last``
    instead takes the chain stated LAST — the model's final committed route, after
    any earlier exploratory revisions (used when scanning free-form reasoning).
    When ``valid_nodes`` is given, tokens are filtered to exact matches. Returns []
    on malformed/empty input.
    """
    if not text or not isinstance(text, str):
        return []
    nodes: List[str] = []
    if "->" in text:
        chains = _ARROW_CHAIN.findall(text)
        if chains:
            best = chains[-1] if prefer_last else max(chains, key=lambda c: c.count("->"))
            nodes = _NODE_TOKEN.findall(best)
    elif "goto(" in text:
        nodes = _GOTO.findall(text)
    if valid_nodes is not None:
        nodes = [n for n in nodes if n in valid_nodes]
    return nodes


def validate_path(
    generated_text: str,
    graph_dict: dict,
    *,
    start: Optional[str] = None,
    goal: Optional[str] = None,
    directed: bool = False,
    reasoning_text: Optional[str] = None,
    rescue_response: Optional[str] = None,
    task: Optional[str] = None,
) -> Dict:
    """Validate a generated route against the graph (regex + NetworkX). Never raises.

    Returns a metrics dict: ``parsed_nodes``, ``num_parsed``, ``nodes_exist_rate``,
    ``edge_validity_rate``, ``full_path_valid``, ``start_goal_ok``,
    ``cost_optimality`` (None unless ``full_path_valid`` and ≥2 nodes),
    ``path_from_reasoning``, ``path_rescued``.

    When the regex finds no route in ``generated_text``, two fallbacks fire:

    1. **Reasoning scan (always on).** Re-scans ``reasoning_text``, taking the
       route stated LAST (final committed path).
    2. **Gemma rescue (gated).** If ``rescue_response`` is given and
       ``GREP_PATH_RESCUE`` is on, the judge rewrites the route in ``a -> b -> c``.

    The same NetworkX diagnostics grade whichever route is found; both fallbacks
    are one-way. ``path_from_reasoning`` / ``path_rescued`` flag the source.
    """
    G = build_graph(graph_dict, directed=directed)
    node_set = set(G.nodes)
    # Parse without filtering first so hallucinated nodes are visible in the rate.
    parsed = parse_path(generated_text)
    from_reasoning = False
    rescued = False
    if not parsed and reasoning_text:
        # Reasoning scan: take the route the model commits to last (no model call).
        parsed = parse_path(reasoning_text, prefer_last=True)
        from_reasoning = bool(parsed)
    if not parsed and rescue_response and _path_rescue_enabled():
        # Gemma rescue: rewrite the route in `a -> b -> c` and re-run NetworkX.
        rescued_route = write_path_with_judge(
            rescue_response, node_set, task=task, start=start, goal=goal)
        parsed = parse_path(rescued_route, prefer_last=True)
        rescued = bool(parsed)
    result = {
        "parsed_nodes": parsed,
        "num_parsed": len(parsed),
        "nodes_exist_rate": 0.0,
        "edge_validity_rate": 0.0,
        "full_path_valid": False,
        "start_goal_ok": False,
        "cost_optimality": None,
        "hop_optimality": None,
        "path_from_reasoning": from_reasoning,
        "path_rescued": rescued,
    }
    if not parsed:
        return result

    exists = [n in node_set for n in parsed]
    result["nodes_exist_rate"] = sum(exists) / len(parsed)

    pairs = list(zip(parsed[:-1], parsed[1:]))
    edge_ok = [G.has_edge(u, v) for u, v in pairs] if pairs else []
    result["edge_validity_rate"] = (sum(edge_ok) / len(edge_ok)) if edge_ok else 0.0
    result["full_path_valid"] = bool(all(exists) and (not pairs or all(edge_ok)))
    result["start_goal_ok"] = bool(
        (start is None or parsed[0] == start) and (goal is None or parsed[-1] == goal)
    )

    if result["full_path_valid"] and len(parsed) >= 2:
        emitted = sum(G[u][v]["distance_m"] for u, v in pairs)
        try:
            shortest = nx.shortest_path_length(G, parsed[0], parsed[-1], weight="distance_m")
            result["cost_optimality"] = (emitted / shortest) if shortest > 0 else 1.0
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            result["cost_optimality"] = None
        # Hop-count optimality: emitted hops ÷ shortest-path hops (unweighted BFS).
        # Any optimal route scores 1.0; nulled by evaluate_sample for invalid A→B paths.
        try:
            hops = nx.shortest_path_length(G, parsed[0], parsed[-1])
            result["hop_optimality"] = (len(pairs) / hops) if hops > 0 else 1.0
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            result["hop_optimality"] = None
    return result


# --------------------------------------------------------------------------
# Conditional LLM-as-judge (Gemma 4 E4B) — used only when acceptance_criterion set
# --------------------------------------------------------------------------

_JUDGE = {"gen": None, "loaded": False}


def _load_judge():
    """Lazily load the Gemma 4 E4B judge once, returning a ``generate(prompt)->str``
    callable (or None if the model can't be loaded).

    Gemma 4 E4B is an image-text-to-text model, so we drive it text-only through
    its chat template + ``generate`` rather than the text-generation pipeline.
    """
    if _JUDGE["loaded"]:
        return _JUDGE["gen"]
    _JUDGE["loaded"] = True
    # Hard off-switch: GREP_JUDGE=0 skips the model load (disables both acceptance
    # judge and path rescue, since both route through here).
    if os.environ.get("GREP_JUDGE", "1").strip().lower() not in ("1", "true", "yes", "on"):
        print("[path_validator] Gemma judge disabled (GREP_JUDGE=0); regex/NetworkX only.")
        _JUDGE["gen"] = None
        return None
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(GEMMA_JUDGE_MODEL)
        model = AutoModelForImageTextToText.from_pretrained(
            GEMMA_JUDGE_MODEL,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        model.eval()

        tokenizer = getattr(processor, "tokenizer", processor)
        has_chat = bool(getattr(tokenizer, "chat_template", None))

        def _generate(prompt: str, max_new_tokens: int = 4) -> str:
            if has_chat:  # instruct checkpoints
                messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
                inputs = processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt",
                ).to(model.device)
            else:  # base/completion checkpoints (no chat template)
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            input_len = inputs["input_ids"].shape[-1]
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            return tokenizer.decode(
                out[0][input_len:], skip_special_tokens=True,
                clean_up_tokenization_spaces=False,  # destructive for BPE (see inference._decode)
            )

        _JUDGE["gen"] = _generate
    except Exception as e:  # gated/missing model, offline, OOM, etc.
        print(f"[path_validator] Gemma judge unavailable ({type(e).__name__}: {e}); "
              f"falling back to regex/NetworkX only.")
        _JUDGE["gen"] = None
    return _JUDGE["gen"]


# A yes/no answer is one whose ground-truth `answer` regex contains "yes" or "no".
_YESNO_ANSWER = re.compile(r"\b(yes|no)\b", re.I)


def is_yes_no_task(task: Optional[str], answer: Optional[str]) -> bool:
    """True when the ground-truth ``answer`` regex contains 'yes' or 'no'.
    ``task`` is unused but kept for call-site clarity."""
    return bool(answer and _YESNO_ANSWER.search(answer))


def judge_acceptance(
    task: str,
    response: str,
    *,
    acceptance_criterion: Optional[str] = None,
    answer_regex: Optional[str] = None,
) -> Optional[bool]:
    """Grade a response with the Gemma 4 E4B judge against a reference.

    Reference precedence: ``acceptance_criterion`` if given, else the regex
    ground-truth ``answer_regex`` (so yes/no tasks without a criterion are still
    judged — correcting regex false positives). Returns True/False, or None when
    no reference is available or the judge can't be loaded (caller then keeps the
    regex/NetworkX verdict).
    """
    if acceptance_criterion:
        reference = f"Acceptance criterion: {acceptance_criterion}"
    elif answer_regex:
        reference = (
            "The ground-truth answer matches this regular expression "
            f"(use it to decide what a correct answer is): {answer_regex}"
        )
    else:
        return None
    generate = _load_judge()
    if generate is None:
        return None
    prompt = (
        "You are grading a robot planner's answer to a question. The automated "
        "regex check can produce false positives, so judge the meaning, not just "
        "keyword presence. Reply with exactly PASS or FAIL.\n"
        f"Task: {task}\n"
        f"{reference}\n"
        f"Planner answer: {response}\n"
        "Does the planner's answer actually satisfy the question? Verdict:"
    )
    try:
        verdict = generate(prompt, max_new_tokens=4).strip().upper()
        return verdict.startswith("PASS")
    except Exception as e:
        print(f"[path_validator] Gemma judge call failed ({type(e).__name__}: {e}).")
        return None


# --------------------------------------------------------------------------
# Gemma path rescue — recreate the route in `a -> b -> c` notation when the
# RegEx could not detect one, then re-run the NetworkX diagnostics on it.
# --------------------------------------------------------------------------
# A route can be many hops of long grid ids, so it needs far more than the
# 4-token PASS/FAIL verdict budget.
_RESCUE_MAX_NEW_TOKENS = 256


def _path_rescue_enabled() -> bool:
    """True unless GREP_PATH_RESCUE explicitly disables the Gemma path rescue."""
    return os.environ.get("GREP_PATH_RESCUE", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def write_path_with_judge(
    response: str,
    valid_nodes: Optional[set] = None,
    *,
    task: Optional[str] = None,
    start: Optional[str] = None,
    goal: Optional[str] = None,
) -> str:
    """Re-express the planner's intended route in the canonical ``a -> b -> c``
    arrow notation using the Gemma judge — the path counterpart of
    ``judge_acceptance``'s one-way rescue.

    Invoked ONLY when neither the plan nor the reasoning regex found a route, to
    recover a path the model expressed off-spec. The judge reads the full reasoning
    and is biased toward the route the planner commits to LAST (its final route,
    not an earlier discarded attempt). The reply is fed straight back through
    ``parse_path`` + the NetworkX diagnostics, so an empty/hallucinated answer just
    scores as no/invalid path — it can rescue, never inflate. Returns "" when the
    judge can't be loaded or there is nothing to read.
    """
    generate = _load_judge()
    if generate is None or not response:
        return ""
    node_list = ", ".join(sorted(valid_nodes)) if valid_nodes else ""
    ends = ""
    if start:
        ends += f"\nThe route starts at node: {start}"
    if goal:
        ends += f"\nThe route ends at node: {goal}"
    prompt = (
        "You are reading a robot planner's reasoning and answer, and must recover "
        "the single route it was trying to express. Read its full reasoning, not "
        "just its final line. The planner often explores and REVISES its route while "
        "thinking, so when it states more than one route, take the LAST one it "
        "commits to — the route nearest the END of its answer — not an earlier "
        "discarded attempt.\n"
        "Output ONLY that route, as a chain of node ids in exactly this form:\n"
        "node_a -> node_b -> node_c\n"
        "Use ONLY ids from the allowed list, copied exactly, with consecutive nodes "
        "connected in the graph. Output nothing else — no prose, no explanation. If "
        "the answer expresses no route at all, output NONE.\n"
        f"Allowed node ids: {node_list}\n"
        f"Task: {task or ''}"
        f"{ends}\n"
        f"Planner reasoning and answer: {response}\n"
        "Final route:"
    )
    try:
        return generate(prompt, max_new_tokens=_RESCUE_MAX_NEW_TOKENS) or ""
    except Exception as e:
        print(f"[path_validator] Gemma path-writer call failed ({type(e).__name__}: {e}).")
        return ""


# --------------------------------------------------------------------------
# Deterministic structured grading (edges + paths) — replaces the judge for
# positionality / reachability / navigability
# --------------------------------------------------------------------------
# Structural tasks are graded against the graph itself, not a yes/no regex or the
# LLM judge: a correct answer must STATE the relevant edges (``u <=> v``) and, for
# reachability/navigability, give a valid route to the goal. The goal, waypoints
# and avoid-set are read from the task's ``answer`` regex + ``acceptance_criterion``
# (which already name the node ids), so no new data schema is needed.

# A node id: lowercase type prefix + one-or-more ``_<int>`` tails (grid ids like
# ``bay_3_26_1`` carry several).
_NODE_ID = re.compile(r"[a-z][a-z_]*(?:_\d+)+")
# An edge may be stated as ``u <=> v`` (spec form) or in SPINE's native pair forms
# ``[u, v]`` / ``(u, v)`` that the planner emits in its ``answer(...)`` — and the
# pair nodes are often quoted (``['u', 'v']``). Accept all.
_Q = r"['\"]?"
_EDGE_STMTS = (
    re.compile(rf"({_NODE_ID.pattern})\s*<=>\s*({_NODE_ID.pattern})"),
    re.compile(rf"\[\s*{_Q}({_NODE_ID.pattern}){_Q}\s*,\s*{_Q}({_NODE_ID.pattern}){_Q}\s*\]"),
    re.compile(rf"\(\s*{_Q}({_NODE_ID.pattern}){_Q}\s*,\s*{_Q}({_NODE_ID.pattern}){_Q}\s*\)"),
)
_VIA = re.compile(
    r"(?:\bvia\b|\bthrough\b|passing\s+through|going\s+through|by\s+way\s+of|crossing)"
    # stop before any clause that merely *describes* the route ("stating ...",
    # "and lists ...", an example "a -> b" walk) so the example path isn't swept
    # in as extra waypoints.
    r"(.*?)(?:[.;]|->|\bwithout\b|\bavoid|\bstating\b|\broute\b|\band\s+lists?\b|$)", re.I)
_AVOID = re.compile(
    r"(?:without\s+(?:using|passing\s+through|going\s+through|entering)|does\s+not\s+use|"
    r"avoid(?:s|ing)?|excluding|not\s+via|not\s+using)\b(.*?)(?:[.,;]|$)", re.I)
# A path task (reachability/navigability) vs a positionality task (edges only).
# `\breach` (no trailing boundary) matches reach / reaches / reachable /
# reachability — the criterion for a reachability task usually says "reachable"
# or "reachability", which `\breach\b` missed, mis-binning it as `edges`.
_PATH_CUE = re.compile(
    r"\broute\b|\bpath\b|\bnavigat|\breach|\bget\s+(?:to|from)\b|\bmove\s+(?:to|from|directly)|"
    r"\bdirectly\s+connected\b|\bone\s+(?:move|hop|step)\b|\btravel", re.I)


def _strip_regex(s: Optional[str]) -> str:
    """Drop regex boundaries / inline flags / lookarounds so node ids read clean."""
    return re.sub(r"\\b|\(\?[a-z]*\)|\(\?<?[!=][^)]*\)", " ", s or "")


def _ordered_subseq(sub: List[str], seq: List[str]) -> bool:
    """True iff every element of ``sub`` appears in ``seq`` in order."""
    it = iter(seq)
    return all(any(x == s for x in it) for s in sub)


def parse_edges(text: str, valid_nodes: Optional[set] = None) -> set:
    """Undirected edges stated in ``text`` (``u <=> v``, ``[u, v]`` or ``(u, v)``)
    as frozenset pairs. Self-loops (``u == v``) are dropped."""
    out = set()
    for pat in _EDGE_STMTS:
        for u, v in pat.findall(text or ""):
            if u != v and (valid_nodes is None or (u in valid_nodes and v in valid_nodes)):
                out.add(frozenset((u, v)))
    return out


def derive_targets(graph_dict: dict, *, init_node, answer, criterion, task):
    """Resolve (goal, waypoints, avoid, required_edges, kind) from task fields.

    ``kind`` is ``"path"`` (reachability/navigability — grade a route to the goal)
    or ``"edges"`` (positionality — grade the stated containment edges). Endpoints
    come from the ``answer`` regex when it names an ordered route, else from the
    criterion; each referenced object's containment edge becomes a required edge.
    Returns ``goal=None`` when no graph target can be resolved (e.g. a count task).
    """
    regions = {r["name"] for r in graph_dict.get("regions", [])}
    objects = {o["name"] for o in graph_dict.get("objects", [])}
    host = {}
    for a, b in graph_dict.get("object_connections", []):
        if a in objects and b in regions:
            host[a] = b
        elif b in objects and a in regions:
            host[b] = a
    nodes = regions | objects
    blob = f"{criterion or ''} || {task or ''}"

    def ids(s):
        seen, out = set(), []
        for t in _NODE_ID.findall(s or ""):
            if t in nodes and t not in seen:
                seen.add(t)
                out.append(t)
        return out

    # The answer regex authoritatively names endpoints/waypoints; never treat them
    # as avoided (a criterion can mention an endpoint in a "without using ..." clause).
    ans_ids = [t for t in ids(_strip_regex(answer)) if t != init_node]
    avoid = {t for span in _AVOID.findall(blob) for t in _NODE_ID.findall(span) if t in nodes}
    avoid -= set(ans_ids) | ({init_node} if init_node else set())
    waypoints = [t for span in _VIA.findall(blob) for t in _NODE_ID.findall(span)
                 if t in nodes and t != init_node and t not in avoid]
    waypoints = list(dict.fromkeys(waypoints))  # de-dup, keep order

    # Union of answer-named ids and criterion/task ids (positionality answers may
    # name only the region; the object comes from the criterion).
    crit_ids = [t for t in ids(blob)
                if t != init_node and t not in avoid and t not in waypoints]
    ordered = ans_ids + [t for t in crit_ids if t not in ans_ids]
    region_refs = [t for t in ordered if t in regions]
    object_refs = [t for t in ordered if t in objects]

    goal = region_refs[-1] if region_refs else (host.get(object_refs[0]) if object_refs else None)
    waypoints = [w for w in waypoints if w != goal]  # the goal is never a waypoint
    required_edges = [frozenset((host[o], o)) for o in object_refs if o in host]
    kind = "path" if (_PATH_CUE.search(blob) or waypoints or len(region_refs) >= 2) else "edges"
    return goal, waypoints, sorted(avoid), required_edges, kind


def validate_structured(
    generated_text: str,
    graph_dict: dict,
    *,
    init_node: Optional[str],
    answer: Optional[str],
    criterion: Optional[str],
    task: Optional[str],
    directed: bool = False,
    full_response: Optional[str] = None,
) -> Optional[Dict]:
    """Deterministic edge/path verdict for a structural task. Never raises.

    Returns ``validate_path`` metrics extended with ``goal``, ``kind``,
    ``waypoints_ok``, ``avoid_ok``, ``required_edges_present``, ``structured_correct``
    — or ``None`` when no goal resolves (caller falls back to regex/judge).

    Reasoning scan runs for every ``kind``; Gemma rescue only for ``kind == "path"``.
    """
    goal, waypoints, avoid, required_edges, kind = derive_targets(
        graph_dict, init_node=init_node, answer=answer, criterion=criterion, task=task)
    if goal is None:
        return None
    m = validate_path(
        generated_text, graph_dict, start=init_node, goal=goal, directed=directed,
        reasoning_text=full_response,
        rescue_response=(full_response if kind == "path" else None), task=task)
    parsed = m["parsed_nodes"]
    stated = parse_edges(generated_text, set(build_graph(graph_dict, directed=directed).nodes))
    # A containment edge counts as present if stated (`u <=> v`) OR traversed
    # (a reach-object route ends goal_region→object, expressing containment implicitly).
    path_edges = {frozenset((parsed[k], parsed[k + 1])) for k in range(len(parsed) - 1)}
    m["goal"] = goal
    m["kind"] = kind
    m["waypoints_ok"] = _ordered_subseq(waypoints, parsed)
    m["avoid_ok"] = not (set(avoid) & set(parsed))
    m["required_edges"] = [sorted(e) for e in required_edges]
    m["required_edges_present"] = all(e in stated or e in path_edges for e in required_edges)
    if kind == "edges":  # positionality: name the goal and state its containment edge(s).
        goal_named = bool(re.search(rf"\b{re.escape(goal)}\b", generated_text or ""))
        # Containment positionality must state the edge; an edge-less structural
        # query (e.g. "northmost area") passes on correctly naming the goal.
        m["structured_correct"] = bool(
            goal_named and (m["required_edges_present"] if required_edges else True))
    else:  # reachability / navigability: a valid constrained walk to the goal.
        # `goal` is the destination REGION. A reach-an-object route legitimately ends
        # one hop past the goal at a contained object (goal_region -> object); both
        # endings count as reaching the goal (overrides validate_path's literal check).
        regions = {r["name"] for r in graph_dict.get("regions", [])}
        objects = {o["name"] for o in graph_dict.get("objects", [])}
        host = {}
        for a, b in graph_dict.get("object_connections", []):
            if a in objects and b in regions:
                host[a] = b
            elif b in objects and a in regions:
                host[b] = a
        start_ok = init_node is None or (bool(parsed) and parsed[0] == init_node)
        ends_at_goal = bool(parsed) and parsed[-1] == goal
        ends_at_goal_object = (
            len(parsed) >= 2 and parsed[-2] == goal and host.get(parsed[-1]) == goal)
        m["start_goal_ok"] = bool(start_ok and (ends_at_goal or ends_at_goal_object))
        m["structured_correct"] = bool(
            m["full_path_valid"] and m["start_goal_ok"] and m["waypoints_ok"]
            and m["avoid_ok"])
    return m


def _augment_eval_metrics(m: Dict, *, goal: Optional[str]) -> Dict:
    """Add eval/* per-sample metrics using the post-override ``start_goal_ok``.

    Adds:
      * ``path_expected`` — True for reachability/navigability tasks with a resolved
        goal; excludes positionality (``kind == "edges"``) and non-path tasks.
      * ``valid_path_ab`` — ``full_path_valid`` AND ``start_goal_ok`` (no waypoint/
        avoid constraints required).
      * ``hallucination_rate`` — fraction of route hops that are not real graph
        edges (``1 - edge_validity_rate``); captures both nonexistent nodes and
        invented edges between real nodes. ``None`` for routes with no hop (<2 nodes).
      * ``hop_optimality`` — nulled for non-valid A→B paths.
    """
    kind = m.get("kind")  # only set for structured tasks
    m["path_expected"] = bool(goal is not None and kind != "edges")
    # Edge hallucination: a route needs ≥2 nodes (one hop) to have any edge to grade.
    m["hallucination_rate"] = (
        (1.0 - m.get("edge_validity_rate", 0.0)) if m.get("num_parsed", 0) >= 2 else None)
    m["valid_path_ab"] = bool(
        m["path_expected"] and m.get("full_path_valid") and m.get("start_goal_ok"))
    if not m["valid_path_ab"]:
        m["hop_optimality"] = None
    return m


def evaluate_sample(
    task: str,
    response_text: str,
    graph_dict: dict,
    *,
    init_node: Optional[str] = None,
    goal: Optional[str] = None,
    acceptance_criterion: Optional[str] = None,
    answer: Optional[str] = None,
    full_response: Optional[str] = None,
    directed: bool = False,
) -> Dict:
    """Per-sample verdict: deterministic structured grade, else regex + judge.

    A structural task (positionality / reachability / navigability — one whose
    answer/criterion resolves to a graph goal) is graded deterministically by
    ``validate_structured``; ``structured`` is True and the Gemma judge is
    skipped. Otherwise ``init_node`` is the route start and the judge runs when
    the task has an ``acceptance_criterion`` OR is a yes/no task (grading against
    the criterion if present, else the regex ``answer``). ``llm_judge_pass`` is
    None when not run / judge unavailable; ``judge_used`` records the attempt.
    """
    structured = validate_structured(
        response_text, graph_dict, init_node=init_node, answer=answer,
        criterion=acceptance_criterion, task=task, directed=directed,
        full_response=full_response or response_text)
    if structured is not None:
        structured["structured"] = True
        structured["judge_used"] = False
        structured["llm_judge_pass"] = None
        return _augment_eval_metrics(structured, goal=structured.get("goal"))

    metrics = validate_path(
        response_text, graph_dict, start=init_node, goal=goal, directed=directed,
        reasoning_text=full_response or response_text,
        rescue_response=full_response or response_text, task=task)
    metrics["structured"] = False
    should_judge = bool(acceptance_criterion) or is_yes_no_task(task, answer)
    metrics["judge_used"] = should_judge
    metrics["llm_judge_pass"] = (
        judge_acceptance(
            task, full_response or response_text,
            acceptance_criterion=acceptance_criterion, answer_regex=answer,
        )
        if should_judge else None
    )
    return _augment_eval_metrics(metrics, goal=goal)


def gemma_regrade_path_metrics(
    planner_response,
    graph_dict: dict,
    *,
    init_node: Optional[str] = None,
    answer: Optional[str] = None,
    acceptance_criterion: Optional[str] = None,
    task: Optional[str] = None,
    directed: bool = False,
) -> Optional[Dict]:
    """GREP_GEMMA_REGRADE path metrics — shared by live eval and retro-grader for
    byte-identical results.

    Asks the Gemma judge to recover the route from the full recorded response for
    EVERY sample, then grades via ``evaluate_sample``. ``path_source`` is
    ``'gemma_judge'`` when the judge's route parsed, else ``'regex_fallback'``
    (ordinary plan/reasoning grading). ``gemma_route`` carries the raw judge output.
    Never raises.
    """
    full = "" if planner_response is None else str(planner_response)
    plan = planner_response.get("plan") if isinstance(planner_response, dict) else planner_response
    plan_text = "" if plan is None else str(plan)
    try:
        goal, *_ = derive_targets(
            graph_dict, init_node=init_node, answer=answer,
            criterion=acceptance_criterion, task=task)
        nodes = {n["name"] for n in (*graph_dict.get("regions", []),
                                     *graph_dict.get("objects", []))}
        route = write_path_with_judge(
            full, nodes, task=task, start=init_node, goal=goal) or ""
    except Exception as e:
        print(f"[path_validator] gemma path recovery failed ({type(e).__name__}: {e}).")
        route = ""
    use_route = bool(route and parse_path(route, prefer_last=True))
    pm = evaluate_sample(
        task, route if use_route else plan_text, graph_dict,
        init_node=init_node, acceptance_criterion=acceptance_criterion,
        answer=answer, full_response=full, directed=directed)
    if pm is not None:
        pm["path_source"] = "gemma_judge" if use_route else "regex_fallback"
        pm["gemma_route"] = route
    return pm


def combine_verdict(
    *,
    regex_correct: bool,
    regex_keyword: bool,
    judge_pass: Optional[bool],
    acceptance_criterion_present: bool,
) -> Dict:
    """Produce two separate per-sample verdicts from disjoint inputs.

    * **objective** — RegEx/NetworkX args only (``regex_correct``/``regex_keyword``).
    * **subjective** — ``judge_pass`` only; ``None`` when not judged.

    ``false_positive`` / ``false_negative`` compare the two verdicts and feed neither.

    Returns: ``objective_correct``, ``objective_keyword`` (RegEx-only);
    ``subjective_correct``, ``subjective_keyword`` (judge-only, ``None`` if unjudged);
    ``false_positive``, ``false_negative``, ``judged``.
    """
    judged = acceptance_criterion_present and judge_pass is not None
    # --- objective: RegEx/NetworkX inputs only -----------------------------------
    objective_correct = bool(regex_correct)
    objective_keyword = bool(regex_keyword)
    # --- subjective: judge input only (None when not judged) ---------------------
    subjective_correct = bool(judge_pass) if judged else None
    subjective_keyword = bool(judge_pass) if judged else None
    # --- diagnostics: compare the two, feed neither score ------------------------
    false_positive = bool(judged and regex_keyword and not judge_pass)
    false_negative = bool(judged and (not regex_keyword) and judge_pass)
    return {
        "objective_correct": objective_correct,
        "objective_keyword": objective_keyword,
        "subjective_correct": subjective_correct,
        "subjective_keyword": subjective_keyword,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "judged": judged,
    }


# --------------------------------------------------------------------------
# Run-level aggregation — shared by evaluate.py, eval callback, and retro-grader.
# --------------------------------------------------------------------------
def aggregate_path_metrics(sample_results: List[dict]) -> dict:
    """Mean path-validation metrics over an eval run's per-sample ``path_metrics``.

    Two metric families with different denominators:

    * **Legacy** (``edge_validity_rate``, ``cost_optimality`` …, logged under
      ``grep/path_*``): averaged over samples with a parseable route
      (``num_parsed > 0``).
    * **eval/\\*** — computed over the full sample list (failures included):
        - ``valid_path_rate``      = #valid_path_ab / #path_expected
        - ``path_optimality_rate`` = mean ``hop_optimality`` over valid A→B paths
        - ``hallucination_rate``   = mean per-sample edge-hallucination (invalid hops /
          total hops) over routes with ≥2 nodes
    """
    all_pm = [r.get("path_metrics") for r in sample_results]
    all_pm = [p for p in all_pm if p]
    pms = [p for p in all_pm if p.get("num_parsed", 0) > 0]

    agg: dict = {}
    if pms:
        def _mean(key):
            vals = [p[key] for p in pms if p.get(key) is not None]
            return (sum(vals) / len(vals)) if vals else None

        agg.update({
            "edge_validity_rate": _mean("edge_validity_rate"),
            "cost_optimality": _mean("cost_optimality"),
            "num_with_path": len(pms),
            # Routes found in reasoning (not plan) by deterministic regex scan.
            "num_from_reasoning": sum(1 for p in pms if p.get("path_from_reasoning")),
            # Routes recovered by the Gemma path rescue (regex found none in plan or reasoning).
            "num_rescued": sum(1 for p in pms if p.get("path_rescued")),
        })
        # GREP_GEMMA_REGRADE only: keeps the original aggregate byte-identical for judge-free runs.
        if any(p.get("path_source") for p in pms):
            agg["num_gemma_path"] = sum(1 for p in pms if p.get("path_source") == "gemma_judge")
        # Deterministic structural aggregates (present when structured tasks ran).
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

    # --- eval/* metrics over the FULL list (failures included) -------------------
    expected = [p for p in all_pm if p.get("path_expected")]
    if expected:
        agg["valid_path_rate"] = sum(1 for p in expected if p.get("valid_path_ab")) / len(expected)
        agg["num_path_expected"] = len(expected)
    hop = [p["hop_optimality"] for p in all_pm if p.get("hop_optimality") is not None]
    if hop:
        agg["path_optimality_rate"] = sum(hop) / len(hop)
    halluc = [p["hallucination_rate"] for p in all_pm if p.get("hallucination_rate") is not None]
    if halluc:
        agg["hallucination_rate"] = sum(halluc) / len(halluc)
    return agg
