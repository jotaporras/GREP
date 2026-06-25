"""Contract tests for the `path_metrics` plumbing in the two eval endpoints:
`prism.eval.evaluate` and `prism.eval.scalability_evaluation`.

CS-only verification. `path_metrics` is built per-sample, aggregated per-graph and
per-run, and serialized to JSON by both endpoints. This file verifies that
deterministic plumbing:

  evaluate.py        : _sample_path_metrics (plan extraction + delegation + error->None)
                       _aggregate_multi_graph_eval (run-level path_metrics fold)
                       _write_cross_eval_json (path_metrics -> JSON round-trip)
  scalability_eval.py: _NumpyEncoder (numpy in path_metrics -> JSON scalars)
                       _write_cross_eval_result / _write_seeded_result / _write_seed_summary

These endpoint modules import heavy externals (`spine`, `datasets`, `torch_geometric`)
that are NOT installed here and that the helpers under test do not use at runtime — only
at import. We inject minimal stub modules for those *boundaries* so the real endpoint
module body loads; every assertion exercises the REAL helper, the REAL `path_validator`,
and the REAL `GraphEvalResultSummary`/`_NumpyEncoder`. The stubs are never called by a
test. (Per AGENTS.md we never install packages; this is the documented optional-dep
boundary pattern applied at import scope.)
"""
import json
import os
import sys
import types

sys.path.insert(0, "src")


# ----------------------------------------------------------------------------
# Boundary stubs — satisfy import-time references only.
# ----------------------------------------------------------------------------

def _mod(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    if "." in name:
        parent, _, child = name.rpartition(".")
        if parent in sys.modules:  # link to parent when it already exists
            setattr(sys.modules[parent], child, m)
    return m


def _install_boundary_stubs():
    import importlib
    # Prefer the REAL modules when the environment provides them (e.g. the conda
    # GREP-PRISM env has spine/datasets/torch_geometric). Only fall back to import-
    # time stubs when a genuine boundary is missing (e.g. a bare uv venv). This keeps
    # the endpoint modules loading their real code wherever possible.
    try:
        import spine.prompts.prompts  # noqa: F401
        import spine.prompts.examples  # noqa: F401
        import spine.mapping.graph_util  # noqa: F401
        import datasets  # noqa: F401
        from prism.models import gnn_llm, inference, loaders  # noqa: F401
        from prism.data import graph_sim, planning_sim, data  # noqa: F401
        return False  # all real boundaries present — no stubs installed
    except Exception:
        pass

    # Real (empty) namespace packages must exist before we attach stub submodules.
    for pkg in ("prism", "prism.models", "prism.data", "prism.eval"):
        importlib.import_module(pkg)
    # --- spine package tree (external, not installed) ---
    _mod("spine")
    _mod("spine.spine")  # provides spine.SPINE in real life; only used in fn bodies
    setattr(sys.modules["spine"], "spine", sys.modules["spine.spine"])
    _mod("spine.mapping")
    _mod("spine.mapping.graph_util", GraphHandler=object)
    _mod("spine.prompts")
    # EXAMPLE_1/EXAMPLE_2 are concatenated at import time -> must be lists.
    _mod("spine.prompts.examples", EXAMPLE_1=[], EXAMPLE_2=[])
    _mod("spine.prompts.prompts",
         get_base_prompt_update_graph=(lambda *a, **k: None),
         SYS_PROMPT={"role": "system", "content": "x"})

    # --- prism model/data submodules that pull torch_geometric / spine ---
    class _G:  # stand-ins for the graph-augmented model classes (isinstance checks only)
        pass
    _mod("prism.models.gnn_llm", GraphAugmentedLLM=_G, GraphMaskLLM=_G,
         CompositeGraphLLM=_G)
    _mod("prism.models.inference",
         GraphAugmentedInMemoryLLM=object, InMemoryLLM=object)

    class GraphSim:  # subclassed at import in evaluate.py (_NoToolsGraphSim)
        def take_action(self, action, argument):
            return False
    _mod("prism.data.graph_sim", GraphSim=GraphSim)
    _mod("prism.data.planning_sim", PlanningSim=object)

    # --- scalability-only boundaries ---
    _mod("datasets")
    _mod("prism.data.data", load_samples_by_graph=(lambda *a, **k: ({}, {})))
    _mod("prism.models.loaders",
         from_pretrained=(lambda *a, **k: (None, None)),
         graph_augmented_llm_from_pretrained=(lambda *a, **k: (None, None)))


_STUBBED = _install_boundary_stubs()
print(f"[test] boundary stubs installed: {_STUBBED} "
      f"({'real spine/datasets present' if not _STUBBED else 'fell back to stubs'})")

# Real modules under test (load real code when boundaries are present).
from prism.eval import evaluate
from prism.eval import scalability_evaluation as scal
from prism.eval import path_validator as P

# numpy is genuinely installed.
import numpy as np


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------

def _triangle_graph():
    return {
        "objects": [
            {"name": "a", "coords": [0.0, 0.0]},
            {"name": "b", "coords": [0.0, 1.0]},
            {"name": "c", "coords": [1.0, 0.0]},
        ],
        "regions": [],
        "object_connections": [["a", "b"], ["b", "c"], ["a", "c"]],
        "region_connections": [],
    }


def _summary(name, samples, *, num_total, num_correct, accuracy, path_metrics):
    """Build a GraphEvalResultSummary with only the fields the helpers read."""
    return evaluate.GraphEvalResultSummary(
        name=name, num_total=num_total, num_correct=num_correct, accuracy=accuracy,
        subjective_accuracy=None, num_judged=0, num_formatted=num_total,
        num_keyword=num_correct, num_false_pos=0, num_false_neg=0, num_errors=0,
        elapsed_s=1.0, n_nodes=3, use_icl=True, permutation=None,
        samples=samples, path_metrics=path_metrics,
    )


# ----------------------------------------------------------------------------
# evaluate._sample_path_metrics — per-sample path_metrics construction
# ----------------------------------------------------------------------------

def test_sample_path_metrics_extracts_plan_from_dict_and_delegates():
    """planner_response is a dict: the 'plan' value is scored, full_response is the
    str() of the whole response. Result must equal an independent direct call to
    path_validator.evaluate_sample with those exact extracted inputs."""
    g = _triangle_graph()
    es = evaluate.EvalSample(task="go a to c", answer="c", graph=g,
                             init_node="a", graph_name="g1")
    resp = {"plan": "a -> c", "reasoning": "trivial"}
    got = evaluate._sample_path_metrics(resp, es)
    oracle = P.evaluate_sample(
        "go a to c", "a -> c", g, init_node="a",
        acceptance_criterion=None, answer="c", full_response=str(resp))
    assert got == oracle
    assert got["parsed_nodes"] == ["a", "c"]
    assert got["full_path_valid"] is True


def test_sample_path_metrics_non_dict_response_scored_whole():
    """A non-dict planner_response is scored directly (plan == the response)."""
    g = _triangle_graph()
    es = evaluate.EvalSample(task="go a to c", answer="c", graph=g,
                             init_node="a", graph_name="g1")
    got = evaluate._sample_path_metrics("a -> c", es)
    assert got is not None
    assert got["parsed_nodes"] == ["a", "c"]


def test_sample_path_metrics_returns_none_on_error():
    """Contract: never propagates — returns None if scoring raises. We force the
    error by making the (boundary) scorer raise; the helper must swallow it."""
    g = _triangle_graph()
    es = evaluate.EvalSample(task="t", answer="c", graph=g, init_node="a",
                             graph_name="g1")
    orig = P.evaluate_sample
    try:
        def _boom(*a, **k):
            raise RuntimeError("scorer exploded")
        P.evaluate_sample = _boom
        assert evaluate._sample_path_metrics({"plan": "a -> c"}, es) is None
    finally:
        P.evaluate_sample = orig


# ----------------------------------------------------------------------------
# evaluate._aggregate_multi_graph_eval — run-level path_metrics fold
# ----------------------------------------------------------------------------

def test_aggregate_multi_graph_eval_folds_path_metrics_over_all_samples():
    """Run-level path_metrics must be aggregate_path_metrics over the CONCATENATED
    per-graph samples; num_correct = Σ round(acc·n); accuracy = Σcorrect/Σn."""
    pm_a = {"num_parsed": 2, "edge_validity_rate": 1.0, "cost_optimality": 1.0,
            "path_from_reasoning": False, "path_rescued": False,
            "path_expected": True, "valid_path_ab": True,
            "hop_optimality": 1.0, "hallucination_rate": 0.0}
    pm_b = {"num_parsed": 2, "edge_validity_rate": 0.0, "cost_optimality": None,
            "path_from_reasoning": True, "path_rescued": False,
            "path_expected": True, "valid_path_ab": False,
            "hop_optimality": None, "hallucination_rate": 1.0}
    s_a = [{"correct": True, "path_metrics": pm_a}]
    s_b = [{"correct": False, "path_metrics": pm_b}]
    r1 = _summary("g1", s_a, num_total=1, num_correct=1, accuracy=1.0, path_metrics={})
    r2 = _summary("g2", s_b, num_total=1, num_correct=0, accuracy=0.0, path_metrics={})

    out = evaluate._aggregate_multi_graph_eval({"g1": r1, "g2": r2}, step=3, epoch=1.5)

    assert out["step"] == 3 and out["epoch"] == 1.5
    assert out["num_graphs"] == 2
    assert out["num_samples"] == 2
    assert out["num_correct"] == 1            # round(1.0*1) + round(0.0*1)
    assert out["accuracy"] == 0.5
    assert out["samples"] == s_a + s_b
    assert out["per_graph"]["g1"] == {"accuracy": 1.0, "num_correct": 1, "num_total": 1}
    # The path_metrics block must equal the real aggregator over the concat samples.
    assert out["path_metrics"] == P.aggregate_path_metrics(s_a + s_b)
    # And that aggregate genuinely reflects path_metrics content (hand-checked):
    assert out["path_metrics"]["edge_validity_rate"] == 0.5   # (1.0 + 0.0)/2
    assert out["path_metrics"]["valid_path_rate"] == 0.5      # 1 of 2 expected
    assert out["path_metrics"]["num_from_reasoning"] == 1


# ----------------------------------------------------------------------------
# evaluate._write_cross_eval_json — path_metrics survives serialization
# ----------------------------------------------------------------------------

def test_write_cross_eval_json_roundtrips_path_metrics(tmp_path=None):
    out_dir = os.path.join(
        os.environ.get("TMPDIR", "/tmp"), "claude_eval_pm_test_evaluate")
    os.makedirs(out_dir, exist_ok=True)
    pm = {"edge_validity_rate": 0.75, "valid_path_rate": 0.5, "num_with_path": 2}
    samples = [{"idx": 0, "correct": True,
                "path_metrics": {"num_parsed": 2, "edge_validity_rate": 1.0}}]
    r = _summary("data_gen_004", samples, num_total=1, num_correct=1,
                 accuracy=1.0, path_metrics=pm)
    evaluate._write_cross_eval_json(
        out_dir, "data_gen_004", r, checkpoint="ckpt", graph_file="g.json",
        architecture="llm", text_edge_list="present")
    with open(os.path.join(out_dir, "data_gen_004.json")) as f:
        loaded = json.load(f)
    assert loaded["path_metrics"] == pm
    assert loaded["samples"][0]["path_metrics"]["edge_validity_rate"] == 1.0
    assert loaded["accuracy"] == 1.0 and loaded["num_correct"] == 1


# ----------------------------------------------------------------------------
# scalability_evaluation._NumpyEncoder — numpy in path_metrics -> JSON scalars
# ----------------------------------------------------------------------------

def test_numpy_encoder_converts_numpy_scalars_and_arrays():
    """path_metrics produced under permutation sweeps can carry numpy types; the
    encoder must emit native JSON numbers/lists, not crash."""
    pm = {"edge_validity_rate": np.float64(0.75),
          "num_with_path": np.int64(2),
          "parsed_nodes": np.array(["a", "b"])}
    s = json.dumps(pm, cls=scal._NumpyEncoder)
    back = json.loads(s)
    assert back["edge_validity_rate"] == 0.75
    assert isinstance(back["edge_validity_rate"], float)
    assert back["num_with_path"] == 2
    assert isinstance(back["num_with_path"], int)
    assert back["parsed_nodes"] == ["a", "b"]


# ----------------------------------------------------------------------------
# scalability writers — path_metrics / per-sample path_metrics serialization
# ----------------------------------------------------------------------------

def test_write_cross_eval_result_roundtrips_native_path_metrics():
    """Realistic case: aggregate_path_metrics emits pure-Python ints/floats, which
    round-trip cleanly through the no-seed cross_eval writer."""
    out_dir = os.path.join(
        os.environ.get("TMPDIR", "/tmp"), "claude_eval_pm_test_scal_cross")
    os.makedirs(out_dir, exist_ok=True)
    pm = {"edge_validity_rate": 0.5, "num_with_path": 2, "valid_path_rate": 1.0}
    samples = [{"idx": 0, "path_metrics": {"num_parsed": 2, "edge_validity_rate": 1.0}}]
    r = _summary("graphX", samples, num_total=1, num_correct=1, accuracy=1.0,
                 path_metrics=pm)
    out_file = scal._write_cross_eval_result(
        r, out_dir=out_dir, checkpoint="ckpt", graph_file="g.json",
        architecture="graph-augmented", text_edge_list="none")
    with open(out_file) as f:
        loaded = json.load(f)
    assert loaded["path_metrics"] == pm
    assert loaded["samples"][0]["path_metrics"]["num_parsed"] == 2


def test_no_seed_writers_stringify_numpy_ints_unlike_seeded_writer():
    """IFACE divergence (characterization, not a crash): the two no-seed cross_eval
    writers serialize with `default=str`, so a numpy INT in path_metrics is written
    as the STRING "1" (np.int64 is not an int subclass); a numpy FLOAT survives as a
    number (np.float64 IS a float subclass). The seeded writer uses _NumpyEncoder and
    preserves the int as a number. If numpy ever reaches path_metrics in cross_eval
    mode, the JSON type silently diverges between the two output layouts."""
    out_dir = os.path.join(
        os.environ.get("TMPDIR", "/tmp"), "claude_eval_pm_test_numpy_div")
    os.makedirs(out_dir, exist_ok=True)
    pm = {"edge_validity_rate": np.float64(0.5), "num_with_path": np.int64(1)}

    # --- no-seed scalability writer (default=str) ---
    r1 = _summary("gA", [], num_total=1, num_correct=1, accuracy=1.0, path_metrics=pm)
    f1 = scal._write_cross_eval_result(
        r1, out_dir=out_dir, checkpoint="c", graph_file="g.json",
        architecture="llm", text_edge_list="none")
    a = json.load(open(f1))["path_metrics"]
    assert a["edge_validity_rate"] == 0.5 and isinstance(a["edge_validity_rate"], float)
    assert a["num_with_path"] == "1"          # numpy int -> stringified by default=str

    # --- no-seed evaluate writer (also default=str) ---
    r2 = _summary("gB", [], num_total=1, num_correct=1, accuracy=1.0, path_metrics=pm)
    evaluate._write_cross_eval_json(
        out_dir, "gB", r2, checkpoint="c", graph_file="g.json",
        architecture="llm", text_edge_list="none")
    b = json.load(open(os.path.join(out_dir, "gB.json")))["path_metrics"]
    assert b["num_with_path"] == "1"          # same stringification

    # --- seeded writer (_NumpyEncoder) preserves the int as a number ---
    r3 = _summary("gC", [], num_total=1, num_correct=1, accuracy=1.0, path_metrics=pm)
    f3, _ = scal._write_seeded_result(
        r3, out_dir=out_dir, ckpt_name="ck", graph_file="g.json")
    # path_metrics isn't in the seeded trial record, but a per-sample numpy int is:
    r4 = _summary("gD", [{"path_metrics": {"num_parsed": np.int64(1)}}],
                  num_total=1, num_correct=1, accuracy=1.0, path_metrics={})
    f4, _ = scal._write_seeded_result(
        r4, out_dir=out_dir, ckpt_name="ck", graph_file="g.json")
    d = json.load(open(f4))["samples"][0]["path_metrics"]
    assert d["num_parsed"] == 1 and isinstance(d["num_parsed"], int)  # preserved


def test_write_seeded_result_keeps_per_sample_path_metrics():
    """The seeded trial file keeps the full samples (with per-sample path_metrics);
    the returned trial_record is the same object the summariser later aggregates."""
    out_dir = os.path.join(
        os.environ.get("TMPDIR", "/tmp"), "claude_eval_pm_test_scal_seed")
    os.makedirs(out_dir, exist_ok=True)
    samples = [{"idx": 0, "path_metrics": {"num_parsed": np.int64(3),
                                           "edge_validity_rate": np.float64(1.0)}}]
    r = _summary("graphY", samples, num_total=1, num_correct=1, accuracy=1.0,
                 path_metrics={"edge_validity_rate": 1.0})
    out_path, trial = scal._write_seeded_result(
        r, out_dir=out_dir, ckpt_name="ckptA", graph_file="g.json")
    assert os.path.basename(out_path) == "ckptA_graphY.json"
    with open(out_path) as f:
        loaded = json.load(f)
    assert loaded["samples"][0]["path_metrics"]["num_parsed"] == 3
    assert loaded["samples"][0]["path_metrics"]["edge_validity_rate"] == 1.0
    # trial_record carries samples through to the summary builder.
    assert trial["samples"] is samples
    assert trial["accuracy"] == 1.0


def test_write_seed_summary_strips_samples_keeps_trial_scalars():
    """The seed summary drops the heavy per-sample (and thus per-sample path_metrics)
    payload, keeping only the scalar trial projection."""
    out_dir = os.path.join(
        os.environ.get("TMPDIR", "/tmp"), "claude_eval_pm_test_scal_summary")
    os.makedirs(out_dir, exist_ok=True)
    trial = {"name": "graphY.json", "path": "g.json", "num_total": 1,
             "num_correct": 1, "accuracy": 1.0, "num_formatted": 1,
             "num_keyword": 1, "num_errors": 0, "elapsed_s": 1.0,
             "permutation": None,
             "samples": [{"path_metrics": {"num_parsed": 3}}]}
    path = scal._write_seed_summary(
        out_dir=out_dir, ckpt_name="ckptA", checkpoint="ckpt",
        graphs_arg="data/eval/*", permutation=None, trial_records=[trial])
    with open(path) as f:
        loaded = json.load(f)
    assert "samples" not in loaded["trials"][0]
    assert loaded["trials"][0]["accuracy"] == 1.0


# ----------------------------------------------------------------------------
# Standalone runner (works under pytest and as a plain script)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"{name}: PASS")
            except Exception as e:
                failures += 1
                import traceback
                print(f"{name}: FAIL — {type(e).__name__}: {e}")
                traceback.print_exc()
    print("done" if not failures else f"done ({failures} FAILED)")
