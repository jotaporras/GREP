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


def parse_path(text: str, valid_nodes: Optional[set] = None) -> List[str]:
    """Extract the ordered route from planner text.

    Tries ``->`` arrows first (the spec form), then ``goto(...)`` actions. Each
    segment is reduced to its node token and, when ``valid_nodes`` is given,
    filtered to exact matches (sidestepping substring ambiguity). Returns [] on
    malformed/empty input.
    """
    if not text or not isinstance(text, str):
        return []
    nodes: List[str] = []
    if "->" in text:
        for seg in _ARROW.split(text):
            m = _NODE_TOKEN.search(seg.strip().strip(".,;:[]()"))
            if m:
                nodes.append(m.group(0))
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
    """M10 per-sample verdict: regex/NetworkX path metrics + (conditional) judge.

    ``init_node`` is the route start (``start_goal_ok``). The Gemma judge runs
    when the task has an ``acceptance_criterion`` OR is a yes/no task; it grades
    against the criterion if present, else the regex ``answer`` (the ground-truth
    pattern). Its boolean lands in ``llm_judge_pass`` (None when not run / judge
    unavailable); ``judge_used`` records whether judging was attempted.
    """
    metrics = validate_path(response_text, graph_dict, start=init_node, goal=goal, directed=directed)
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
