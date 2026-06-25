"""Contract tests for the post-hoc eval driver `prism.eval.scalability_evaluation`.

CS-only scope: the deterministic plumbing of the driver — train/eval-config
resolution (fail-loud), the two JSON output writers + their numpy encoder, and
the per-graph progress accumulator. The model-loading path (`_load_checkpoint` ->
`loaders.*`) and `evaluate.eval_model_multiple_graphs` are boundaries and are NOT
exercised; no model is built. Oracles are hand-written from each helper's
docstring/spec, never copied from its body.
"""
import sys
sys.path.insert(0, "src")

import io
import json
import os
from contextlib import redirect_stdout

import numpy as np

from prism.eval import scalability_evaluation as se
from prism.eval import evaluate
from prism.models import utils as model_utils


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------

def _result(**overrides) -> evaluate.GraphEvalResultSummary:
    """A fully-populated GraphEvalResultSummary; override any field by keyword."""
    base = dict(
        name="data_gen_004",
        num_total=10,
        num_correct=7,
        accuracy=0.7,
        subjective_accuracy=None,
        num_judged=0,
        num_formatted=9,
        num_keyword=8,
        num_false_pos=0,
        num_false_neg=0,
        num_errors=1,
        elapsed_s=12.5,
        n_nodes=5,
        use_icl=True,
        permutation=None,
        samples=[{"task": "go", "correct": True}],
        path_metrics={"path_valid_rate": 0.6},
    )
    base.update(overrides)
    return evaluate.GraphEvalResultSummary(**base)


# ----------------------------------------------------------------------------
# _NumpyEncoder
# ----------------------------------------------------------------------------

def test_numpy_encoder_emits_native_json_types():
    """np scalars/arrays must serialize to JSON numbers/lists, not strings."""
    payload = {
        "f32": np.float32(0.5),
        "i64": np.int64(7),
        "arr": np.array([1, 2, 3]),
    }
    decoded = json.loads(json.dumps(payload, cls=se._NumpyEncoder))
    assert decoded["f32"] == 0.5 and isinstance(decoded["f32"], float)
    assert decoded["i64"] == 7 and isinstance(decoded["i64"], int)
    assert decoded["arr"] == [1, 2, 3]


# ----------------------------------------------------------------------------
# _is_gnn_checkpoint
# ----------------------------------------------------------------------------

def test_is_gnn_checkpoint_detects_marker(tmp_path):
    """True iff gnn_config.json exists in the dir."""
    assert se._is_gnn_checkpoint(str(tmp_path)) is False
    (tmp_path / "gnn_config.json").write_text("{}")
    assert se._is_gnn_checkpoint(str(tmp_path)) is True


# ----------------------------------------------------------------------------
# _resolve_text_edge_list  (fail-loud config recovery)
# ----------------------------------------------------------------------------

def test_resolve_cli_override_wins_without_any_config(tmp_path):
    """An explicit override is returned verbatim, no files consulted."""
    assert se._resolve_text_edge_list(str(tmp_path), is_gnn=True, cli_override="none") == "none"
    assert se._resolve_text_edge_list(str(tmp_path), is_gnn=False, cli_override="present") == "present"


def test_resolve_reads_gnn_config(tmp_path):
    (tmp_path / "gnn_config.json").write_text(json.dumps({"text_edge_list": "none"}))
    assert se._resolve_text_edge_list(str(tmp_path), is_gnn=True, cli_override=None) == "none"


def test_resolve_reads_train_config_for_plain_llm(tmp_path):
    (tmp_path / "train_config.json").write_text(json.dumps({"text_edge_list": "present"}))
    assert se._resolve_text_edge_list(str(tmp_path), is_gnn=False, cli_override=None) == "present"


def test_resolve_gnn_missing_key_raises(tmp_path):
    """gnn_config.json without the key must fail loud, not assume 'present'."""
    (tmp_path / "gnn_config.json").write_text(json.dumps({"other": 1}))
    try:
        se._resolve_text_edge_list(str(tmp_path), is_gnn=True, cli_override=None)
        assert False, "expected KeyError"
    except KeyError as e:
        assert "text_edge_list" in str(e)


def test_resolve_plain_llm_missing_train_config_raises(tmp_path):
    """No train_config.json + no override -> FileNotFoundError (no silent guess)."""
    try:
        se._resolve_text_edge_list(str(tmp_path), is_gnn=False, cli_override=None)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError as e:
        assert "train_config.json" in str(e)


def test_resolve_plain_llm_missing_key_raises(tmp_path):
    (tmp_path / "train_config.json").write_text(json.dumps({"other": 1}))
    try:
        se._resolve_text_edge_list(str(tmp_path), is_gnn=False, cli_override=None)
        assert False, "expected KeyError"
    except KeyError as e:
        assert "text_edge_list" in str(e)


# ----------------------------------------------------------------------------
# _write_cross_eval_result
# ----------------------------------------------------------------------------

def test_write_cross_eval_shape_and_filename(tmp_path):
    """File is <name>.json with the keys eval_viewer.html / judge-eval consume."""
    res = _result()
    out = se._write_cross_eval_result(
        res, out_dir=str(tmp_path), checkpoint="/ckpt/run",
        graph_file="/data/g.json", architecture="llm", text_edge_list="none",
    )
    assert out == str(tmp_path / "data_gen_004.json")
    doc = json.loads(open(out).read())
    assert set(doc) == {
        "checkpoint", "graph_file", "architecture", "text_edge_list",
        "accuracy", "num_samples", "num_correct", "path_metrics", "samples",
    }
    # num_samples is sourced from num_total (the contract), not a separate field.
    assert doc["num_samples"] == res.num_total == 10
    assert doc["num_correct"] == 7
    assert doc["architecture"] == "llm" and doc["text_edge_list"] == "none"


def test_write_cross_eval_stringifies_numpy_scalars(tmp_path):
    """DIVERGENCE PROBE: cross_eval writer uses json default=str, so a numpy
    scalar in a serialized field becomes a JSON *string* — unlike the seeded
    writer below, which keeps it numeric via _NumpyEncoder. Same input, two
    representations. eval_viewer/judge-eval read `accuracy` expecting a number.
    """
    res = _result(accuracy=np.float32(0.5))
    out = se._write_cross_eval_result(
        res, out_dir=str(tmp_path), checkpoint="c",
        graph_file="g", architecture="llm", text_edge_list="none",
    )
    doc = json.loads(open(out).read())
    assert doc["accuracy"] == "0.5"          # stringified — the break
    assert isinstance(doc["accuracy"], str)


# ----------------------------------------------------------------------------
# _write_seeded_result
# ----------------------------------------------------------------------------

def test_write_seeded_result_shape_and_record(tmp_path):
    """File is <ckpt>_<name>.json; returned trial_record mirrors the file and
    carries `samples` (the summariser strips it later)."""
    res = _result()
    out_path, rec = se._write_seeded_result(
        res, out_dir=str(tmp_path), ckpt_name="run42", graph_file="/data/g.json",
    )
    assert out_path == str(tmp_path / "run42_data_gen_004.json")
    assert rec["name"] == "data_gen_004.json"
    assert rec["path"] == "/data/g.json"
    assert "samples" in rec
    on_disk = json.loads(open(out_path).read())
    assert on_disk == rec   # file and returned record agree


def test_write_seeded_result_keeps_numpy_numeric(tmp_path):
    """Counterpart to the cross_eval probe: numpy stays numeric here."""
    res = _result(accuracy=np.float32(0.5), num_correct=np.int64(7))
    out_path, rec = se._write_seeded_result(
        res, out_dir=str(tmp_path), ckpt_name="run", graph_file="g",
    )
    doc = json.loads(open(out_path).read())
    assert doc["accuracy"] == 0.5 and isinstance(doc["accuracy"], float)
    assert doc["num_correct"] == 7 and isinstance(doc["num_correct"], int)


# ----------------------------------------------------------------------------
# _write_seed_summary
# ----------------------------------------------------------------------------

def test_write_seed_summary_drops_samples_and_serializes_permutation(tmp_path):
    """Summary trials must NOT contain `samples`; permutation is .to_dict()'d."""
    _, rec = se._write_seeded_result(
        _result(), out_dir=str(tmp_path), ckpt_name="run", graph_file="g",
    )
    perm = model_utils.Permutation(42)
    path = se._write_seed_summary(
        out_dir=str(tmp_path), ckpt_name="run", checkpoint="/ckpt",
        graphs_arg="data/*.json", permutation=perm, trial_records=[rec],
    )
    doc = json.loads(open(path).read())
    assert path == str(tmp_path / "run_summary.json")
    assert doc["pattern"] == "data/*.json"
    assert doc["permutation"] == perm.to_dict()          # {seed:42, num_nodes:None, permutation:None}
    assert doc["permutation"]["seed"] == 42
    assert len(doc["trials"]) == 1
    assert "samples" not in doc["trials"][0]              # the stripping invariant
    assert doc["trials"][0]["name"] == "data_gen_004.json"


def test_write_seed_summary_permutation_none(tmp_path):
    """permutation=None -> JSON null, not an AttributeError on .to_dict()."""
    _, rec = se._write_seeded_result(
        _result(), out_dir=str(tmp_path), ckpt_name="run", graph_file="g",
    )
    path = se._write_seed_summary(
        out_dir=str(tmp_path), ckpt_name="run", checkpoint="/ckpt",
        graphs_arg="g", permutation=None, trial_records=[rec],
    )
    assert json.loads(open(path).read())["permutation"] is None


# ----------------------------------------------------------------------------
# _make_progress_printer  (accumulator bookkeeping)
# ----------------------------------------------------------------------------

def test_progress_printer_accumulates_running_overall():
    """Running overall = sum(correct)/sum(total) across calls; index increments."""
    by_graph = {"a": [], "b": []}
    printer = se._make_progress_printer(by_graph)

    buf = io.StringIO()
    with redirect_stdout(buf):
        printer("a", _result(name="a", num_total=10, num_correct=5, accuracy=0.5))
        printer("b", _result(name="b", num_total=10, num_correct=9, accuracy=0.9))
    lines = buf.getvalue().strip().splitlines()

    assert lines[0].startswith("[1/2] a:")
    assert "overall: 50.0% (5/10)" in lines[0]
    assert lines[1].startswith("[2/2] b:")
    assert "overall: 70.0% (14/20)" in lines[1]   # (5+9)/(10+10)


def test_progress_printer_zero_total_no_zerodiv():
    """A zero-sample graph first must yield overall 0.0%, not divide-by-zero."""
    printer = se._make_progress_printer({"a": []})
    buf = io.StringIO()
    with redirect_stdout(buf):
        printer("a", _result(name="a", num_total=0, num_correct=0, accuracy=0.0))
    assert "overall: 0.0% (0/0)" in buf.getvalue()


if __name__ == "__main__":
    import tempfile

    class _TP:
        """Minimal tmp_path shim so the file runs standalone (no pytest)."""
        def __init__(self, d): self._d = d
        def __truediv__(self, name): return _P(os.path.join(self._d, name))

    class _P(str):
        def write_text(self, s):
            with open(self, "w") as f:
                f.write(s)

    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            n = fn.__code__.co_argcount
            if n == 0:
                fn()
            else:
                with tempfile.TemporaryDirectory() as d:
                    fn(_TP(d))
            print(f"{name}: PASS")
    print("done")
