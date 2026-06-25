"""Deterministic-CS verification for the answer-scoring + summary plumbing in
``prism.eval.evaluate``.

CS-only scope: these are the judge-free, regex/shape-based scoring helpers and
the stdout summary formatter — pure functions of dicts/strings. No model is
constructed or called; nothing here touches a forward/backward pass.

Targets: ``_has_correct_keys``, ``_EvalResult``, ``_construct_eval_result``,
``construct_eval_samples_from_dict``, ``_graph_node_count``,
``print_summary_table``. Oracles are hand-built from the docstrings, independent
of the implementation bodies.

Run directly:  ``python tests/test_eval_scoring.py``
Or via pytest: ``pytest tests/test_eval_scoring.py``
"""
import sys
sys.path.insert(0, "src")

import io
from contextlib import redirect_stdout

from prism.eval import evaluate


# ---------------------------------------------------------------------------
# _has_correct_keys / _EvalResult — JSON-shape verdict
# ---------------------------------------------------------------------------

_FULL = {"primary_goal": "x", "relevant_graph": "y", "reasoning": "z", "plan": "p"}


def test_has_correct_keys_requires_all_four():
    """All four answer keys present -> True; any missing -> False."""
    assert evaluate._has_correct_keys(_FULL) is True
    for drop in ("primary_goal", "relevant_graph", "reasoning", "plan"):
        partial = {k: v for k, v in _FULL.items() if k != drop}
        assert evaluate._has_correct_keys(partial) is False


def test_eval_result_is_correct_is_conjunction():
    """is_correct() <=> formatted AND plan_keyword (full truth table)."""
    assert evaluate._EvalResult(True, True).is_correct() is True
    assert evaluate._EvalResult(True, False).is_correct() is False
    assert evaluate._EvalResult(False, True).is_correct() is False
    assert evaluate._EvalResult(False, False).is_correct() is False


# ---------------------------------------------------------------------------
# _construct_eval_result — shape correctness + keyword match
# ---------------------------------------------------------------------------

def test_construct_eval_result_keyword_case_insensitive():
    """A well-formed answer whose plan contains the key (any case) -> (True, True)."""
    ans = dict(_FULL, plan="First GOTO the Kitchen then stop")
    res, parsed = evaluate._construct_eval_result(ans, "kitchen")
    assert res.formatted is True
    assert res.plan_keyword is True
    assert res.is_correct() is True
    assert parsed is ans


def test_construct_eval_result_missing_keyword_is_formatted_only():
    """Well-formed but the key isn't in the plan -> formatted True, keyword False."""
    ans = dict(_FULL, plan="go to the bedroom")
    res, _ = evaluate._construct_eval_result(ans, "kitchen")
    assert res.formatted is True
    assert res.plan_keyword is False


def test_construct_eval_result_unformatted_when_keys_missing():
    """Missing required keys -> formatted False (keyword may still match)."""
    ans = {"plan": "go to the kitchen"}  # only 'plan'
    res, _ = evaluate._construct_eval_result(ans, "kitchen")
    assert res.formatted is False


def test_construct_eval_result_formatting_independent_of_answer_key():
    """CONTRACT: 'formatted' is a property of the answer's JSON shape and must NOT
    depend on the answer_key string. A well-formed answer whose answer_key happens
    to contain a regex metacharacter (unbalanced paren) must still report
    formatted=True.

    Current behaviour: re.search(answer_key, ...) raises re.error on the bad
    pattern, the blanket ``except`` discards the already-computed is_formatted, and
    the function returns _EvalResult(False, False) — silently mislabelling a
    correctly-formatted planner answer as malformed. This test pins that defect.
    """
    ans = dict(_FULL, plan="go to the couch on the left")
    res, _ = evaluate._construct_eval_result(ans, "couch (left")  # unbalanced '('
    assert res.formatted is True


# ---------------------------------------------------------------------------
# construct_eval_samples_from_dict — field mapping + optional AC
# ---------------------------------------------------------------------------

def test_construct_eval_samples_maps_fields_and_stamps_graph_name():
    """Each task -> one EvalSample carrying task/answer/init_node, the shared
    graph dict, the stamped graph_name, and acceptance_criterion (None if absent)."""
    graph = {"objects": {"a": {}}, "regions": {}}
    tasks = [
        {"task": "t0", "answer": "ans0", "init_node": "a"},
        {"task": "t1", "answer": "ans1", "init_node": "b",
         "acceptance_criterion": "must reach b"},
    ]
    samples = evaluate.construct_eval_samples_from_dict(graph, tasks, "data_gen_004")
    assert len(samples) == 2
    s0, s1 = samples
    assert (s0.task, s0.answer, s0.init_node) == ("t0", "ans0", "a")
    assert s0.graph_name == "data_gen_004"
    assert s0.graph is graph
    assert s0.acceptance_criterion is None         # not provided -> default None
    assert s1.acceptance_criterion == "must reach b"


# ---------------------------------------------------------------------------
# _graph_node_count
# ---------------------------------------------------------------------------

def test_graph_node_count_sums_objects_and_regions():
    """Node count = |objects| + |regions|; missing sections count as 0."""
    assert evaluate._graph_node_count(
        {"objects": {"a": {}, "b": {}}, "regions": {"r": {}}}) == 3
    assert evaluate._graph_node_count({"objects": {}, "regions": {}}) == 0
    assert evaluate._graph_node_count({}) == 0


# ---------------------------------------------------------------------------
# print_summary_table — stdout TOTAL arithmetic
# ---------------------------------------------------------------------------

def _summary(name, *, num_total, num_correct, accuracy):
    return evaluate.GraphEvalResultSummary(
        name=name, num_total=num_total, num_correct=num_correct, accuracy=accuracy,
        subjective_accuracy=None, num_judged=0, num_formatted=num_correct,
        num_keyword=num_correct, num_false_pos=0, num_false_neg=0, num_errors=0,
        elapsed_s=1.0, n_nodes=3, use_icl=True, permutation=None,
    )


def test_print_summary_table_total_row_micro_averages():
    """TOTAL objective accuracy is Σcorrect/Σtotal = 2/4 = 50.0%."""
    results = [
        _summary("g0", num_total=2, num_correct=1, accuracy=0.5),
        _summary("g1", num_total=2, num_correct=1, accuracy=0.5),
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        evaluate.print_summary_table(results)
    out = buf.getvalue()
    assert "TOTAL" in out
    assert "50.0%" in out
    assert "4" in out  # total tasks


def test_print_summary_table_empty_is_safe():
    """No results -> a clear message, no division by zero."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        evaluate.print_summary_table([])
    assert "no results" in buf.getvalue().lower()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"{name}: PASS")
    print("done")
