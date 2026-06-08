"""M10 — validate a generated plan/path against the scene graph (R4).

Two layers:

1. **Regex + NetworkX (always on).** Parse a route out of the planner's text
   (``a -> b -> c`` arrows or ``goto(a), goto(b)`` actions), exact-match the node
   strings against ``G_Sc.nodes``, and score it with NetworkX:
   ``nodes_exist_rate``, ``edge_validity_rate``, ``full_path_valid``,
   ``start_goal_ok``, ``cost_optimality`` (emitted weighted cost ÷ shortest path
   by ``distance_m``). Malformed/empty input is invalid, never raises.

2. **LLM judge (separate, subjective score).** When a task carries an
   ``acceptance_criterion`` (or is a yes/no task), an LLM-as-judge (Gemma 4 E2B,
   ``GEMMA_JUDGE_MODEL``, default ``google/gemma-4-E2B-it``) grades the response
   against that rubric. For acceptance_criterion tasks the judge runs *every*
   evaluation turn (validation and test). The judge score and the RegEx score are
   kept **completely separate** — computed from disjoint inputs, neither reading the
   other: the RegEx/NetworkX accuracy is judge-free, and the *subjective* accuracy is
   the judge's verdict over judged samples only. Relative to the objective baseline
   the judge moves the subjective column only — it LOWERS it on a false positive
   (RegEx correct, judge wrong) and RAISES it on a false negative (RegEx wrong, judge
   correct). ``combine_verdict`` computes both verdicts plus the
   ``false_positive``/``false_negative`` diagnostics. If the judge cannot load (no
   weights/auth) those samples are simply absent from the subjective score (and the
   caller warns) — never copied from RegEx. With no ``acceptance_criterion`` and a
   non-yes/no answer, the judge is skipped entirely.

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

# Gemma 4 E2B judge. Defaults to the official gated "google/gemma-4-E2B-it" repo
# (requires an HF auth token with access granted); set GREP_JUDGE_MODEL to override
# (e.g. the ungated "unsloth/gemma-4-E2B-it" mirror when no auth is configured).
GEMMA_JUDGE_MODEL = os.environ.get("GREP_JUDGE_MODEL", "google/gemma-4-E2B-it")

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


def parse_path(text: str, valid_nodes: Optional[set] = None) -> List[str]:
    """Extract the ordered route from planner text.

    Pulls the longest ``a -> b -> c`` chain (the spec form), then ``goto(...)``
    actions. Undirected edge statements ``u <-> v`` are neutralised first so the
    ``->`` inside ``<->`` is never mistaken for a route hop — a response may carry
    both an edge list and a path. When ``valid_nodes`` is given, tokens are
    filtered to exact matches. Returns [] on malformed/empty input.
    """
    if not text or not isinstance(text, str):
        return []
    nodes: List[str] = []
    route_text = text.replace("<->", " ").replace("<-", " ")  # drop edge arrows
    if "->" in route_text:
        chains = _ARROW_CHAIN.findall(route_text)
        if chains:
            best = max(chains, key=lambda c: c.count("->"))
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
) -> Dict:
    """Validate a generated route against the graph (regex + NetworkX). Never raises.

    Returns a metrics dict: ``parsed_nodes``, ``num_parsed``, ``nodes_exist_rate``,
    ``edge_validity_rate``, ``full_path_valid``, ``start_goal_ok``,
    ``cost_optimality`` (None unless ``full_path_valid`` and ≥2 nodes).
    """
    G = build_graph(graph_dict, directed=directed)
    node_set = set(G.nodes)
    # Parse without filtering first so hallucinated nodes are visible in the rate.
    parsed = parse_path(generated_text)
    result = {
        "parsed_nodes": parsed,
        "num_parsed": len(parsed),
        "nodes_exist_rate": 0.0,
        "edge_validity_rate": 0.0,
        "full_path_valid": False,
        "start_goal_ok": False,
        "cost_optimality": None,
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
    return result


# --------------------------------------------------------------------------
# Conditional LLM-as-judge (Gemma E2B) — used only when acceptance_criterion set
# --------------------------------------------------------------------------

_JUDGE = {"gen": None, "loaded": False}


def _load_judge():
    """Lazily load the Gemma E2B judge once, returning a ``generate(prompt)->str``
    callable (or None if the model can't be loaded).

    Gemma 3n is an image-text-to-text model, so we drive it text-only through its
    chat template + ``generate`` rather than the text-generation pipeline.
    """
    if _JUDGE["loaded"]:
        return _JUDGE["gen"]
    _JUDGE["loaded"] = True
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


# A yes/no answer is one whose ground-truth `answer` regex contains the literal
# word "yes" or "no" — exactly these are routed to the Gemma judge, since a bare
# regex match on such answers is the most false-positive-prone.
_YESNO_ANSWER = re.compile(r"\b(yes|no)\b", re.I)


def is_yes_no_task(task: Optional[str], answer: Optional[str]) -> bool:
    """True when the ground-truth ``answer`` regex contains 'yes' or 'no'.

    The presence of yes/no in the answer is the trigger for the Gemma judge
    (the ``task`` argument is unused but kept for call-site clarity).
    """
    return bool(answer and _YESNO_ANSWER.search(answer))


def judge_acceptance(
    task: str,
    response: str,
    *,
    acceptance_criterion: Optional[str] = None,
    answer_regex: Optional[str] = None,
) -> Optional[bool]:
    """Grade a response with the Gemma E2B judge against a reference.

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
# Deterministic structured grading (edges + paths) — replaces the judge for
# positionality / reachability / navigability
# --------------------------------------------------------------------------
# Structural tasks are graded against the graph itself, not a yes/no regex or the
# LLM judge: a correct answer must STATE the relevant edges (``u <-> v``) and, for
# reachability/navigability, give a valid route to the goal. The goal, waypoints
# and avoid-set are read from the task's ``answer`` regex + ``acceptance_criterion``
# (which already name the node ids), so no new data schema is needed.

# A node id: lowercase type prefix + one-or-more ``_<int>`` tails (grid ids like
# ``bay_3_26_1`` carry several).
_NODE_ID = re.compile(r"[a-z][a-z_]*(?:_\d+)+")
_EDGE_STMT = re.compile(rf"({_NODE_ID.pattern})\s*<->\s*({_NODE_ID.pattern})")
_VIA = re.compile(
    r"(?:\bvia\b|\bthrough\b|passing\s+through|going\s+through|by\s+way\s+of|crossing)"
    r"(.*?)(?:[.;]|\bwithout\b|\bavoid|$)", re.I)
_AVOID = re.compile(
    r"(?:without\s+(?:using|passing\s+through|going\s+through|entering)|does\s+not\s+use|"
    r"avoid(?:s|ing)?|excluding|not\s+via|not\s+using)\b(.*?)(?:[.,;]|$)", re.I)
# A path task (reachability/navigability) vs a positionality task (edges only).
_PATH_CUE = re.compile(
    r"\broute\b|\bpath\b|\bnavigat|\breach\b|\bget\s+from\b|\bmove\s+(?:to|from|directly)|"
    r"\bdirectly\s+connected\b|\bone\s+(?:move|hop|step)\b|\btravel", re.I)


def _strip_regex(s: Optional[str]) -> str:
    """Drop regex boundaries / inline flags / lookarounds so node ids read clean."""
    return re.sub(r"\\b|\(\?[a-z]*\)|\(\?<?[!=][^)]*\)", " ", s or "")


def _ordered_subseq(sub: List[str], seq: List[str]) -> bool:
    """True iff every element of ``sub`` appears in ``seq`` in order."""
    it = iter(seq)
    return all(any(x == s for x in it) for s in sub)


def parse_edges(text: str, valid_nodes: Optional[set] = None) -> set:
    """Undirected edges stated as ``u <-> v`` in ``text`` (as frozenset pairs)."""
    out = set()
    for u, v in _EDGE_STMT.findall(text or ""):
        if valid_nodes is None or (u in valid_nodes and v in valid_nodes):
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

    avoid = {t for span in _AVOID.findall(blob) for t in _NODE_ID.findall(span) if t in nodes}
    waypoints = [t for span in _VIA.findall(blob) for t in _NODE_ID.findall(span)
                 if t in nodes and t != init_node and t not in avoid]
    seen_wp = dict.fromkeys(waypoints)  # de-dup, keep order
    waypoints = list(seen_wp)

    # Union of answer-named ids (ordered, authoritative for path endpoints) and
    # criterion/task ids — so a positionality answer that names only the region
    # still picks up the contained object (and its containment edge) from the
    # criterion.
    ans_ids = [t for t in ids(_strip_regex(answer)) if t != init_node and t not in avoid]
    crit_ids = [t for t in ids(blob)
                if t != init_node and t not in avoid and t not in waypoints]
    ordered = ans_ids + [t for t in crit_ids if t not in ans_ids]
    region_refs = [t for t in ordered if t in regions]
    object_refs = [t for t in ordered if t in objects]

    goal = region_refs[-1] if region_refs else (host.get(object_refs[0]) if object_refs else None)
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
) -> Optional[Dict]:
    """Deterministic edge/path verdict for a structural task. Never raises.

    Returns the ``validate_path`` metrics extended with ``goal``, ``kind``,
    ``waypoints_ok``, ``avoid_ok``, ``required_edges_present`` and the boolean
    ``structured_correct`` — or ``None`` when the task isn't graph-structural
    (no resolvable goal), so the caller falls back to the regex/judge path.
    """
    goal, waypoints, avoid, required_edges, kind = derive_targets(
        graph_dict, init_node=init_node, answer=answer, criterion=criterion, task=task)
    if goal is None:
        return None
    m = validate_path(generated_text, graph_dict, start=init_node, goal=goal, directed=directed)
    parsed = m["parsed_nodes"]
    stated = parse_edges(generated_text, set(build_graph(graph_dict, directed=directed).nodes))
    m["goal"] = goal
    m["kind"] = kind
    m["waypoints_ok"] = _ordered_subseq(waypoints, parsed)
    m["avoid_ok"] = not (set(avoid) & set(parsed))
    m["required_edges"] = [sorted(e) for e in required_edges]
    m["required_edges_present"] = all(e in stated for e in required_edges)
    if kind == "edges":  # positionality: name the goal and state its containment edge(s).
        goal_named = bool(re.search(rf"\b{re.escape(goal)}\b", generated_text or ""))
        # Containment positionality must state the edge; an edge-less structural
        # query (e.g. "northmost area") passes on correctly naming the goal.
        m["structured_correct"] = bool(
            goal_named and (m["required_edges_present"] if required_edges else True))
    else:  # reachability / navigability: a valid constrained walk to the goal.
        m["structured_correct"] = bool(
            m["full_path_valid"] and m["start_goal_ok"] and m["waypoints_ok"]
            and m["avoid_ok"] and m["required_edges_present"])
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
    """M10 per-sample verdict: deterministic structured grade, else regex + judge.

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
        criterion=acceptance_criterion, task=task, directed=directed)
    if structured is not None:
        structured["structured"] = True
        structured["judge_used"] = False
        structured["llm_judge_pass"] = None
        return structured

    metrics = validate_path(response_text, graph_dict, start=init_node, goal=goal, directed=directed)
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
    return metrics


def combine_verdict(
    *,
    regex_correct: bool,
    regex_keyword: bool,
    judge_pass: Optional[bool],
    acceptance_criterion_present: bool,
) -> Dict:
    """Produce the two *completely separate* per-sample verdicts.

    The two scores are computed from **disjoint inputs** and never read each other:

    * **objective** — a function of the RegEx/NetworkX args ONLY
      (``regex_correct``/``regex_keyword``). The judge has zero effect on it.
    * **subjective** — a function of ``judge_pass`` ONLY. It is the judge's boolean
      where the judge ran (``acceptance_criterion`` present and a boolean returned),
      else ``None`` (not judged — it does NOT borrow the RegEx value).

    ``false_positive`` / ``false_negative`` are separate diagnostics that *compare*
    the two verdicts (judge disagreeing with RegEx); they feed neither score.

    Returns: ``objective_correct``/``objective_keyword`` (RegEx-only booleans),
    ``subjective_correct``/``subjective_keyword`` (judge-only, ``None`` if unjudged),
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
