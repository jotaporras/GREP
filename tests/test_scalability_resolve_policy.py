"""Gap-filler contract tests for prism.eval.scalability_evaluation — the post-hoc eval
driver named in the verification request. The existing suite (test_eval_path_metrics_plumbing)
covers the OUTPUT writers + _NumpyEncoder; this file covers the INPUT side that nothing else
exercises:

  * _resolve_text_edge_list — recovers the train-time `text_edge_list` policy so eval matches
    training. This is a fail-LOUD contract (AGENTS.md: no silent fallback): a checkpoint trained
    with `text_edge_list=none` must NEVER be silently evaluated with edge bullets re-added. CLI
    override wins; else read gnn_config.json (graph ckpt) / train_config.json (plain LLM); a
    missing file or missing key must RAISE, not guess.
  * _is_gnn_checkpoint — graph-vs-LLM checkpoint discrimination by gnn_config.json presence.
  * _make_progress_printer — the running cumulative-accuracy arithmetic (grand_correct/grand_total)
    shared by both eval modes; verified via captured stdout against a hand-computed oracle.

CS-only deterministic plumbing (file IO + arithmetic). No model is loaded: _load_checkpoint and
the scoring path are NOT touched here. Oracles are restated from the docstrings, not the bodies.

Run: conda run -n GREP-PRISM python tests/test_scalability_resolve_policy.py
"""
import contextlib
import io
import json
import os
import sys

sys.path.insert(0, "src")

from prism.eval import evaluate
from prism.eval import scalability_evaluation as scal


_SCRATCH = os.environ.get("TMPDIR", "/tmp")


def _ckpt_dir(tag):
    """Fresh empty checkpoint dir under the scratch space."""
    d = os.path.join(_SCRATCH, f"claude_scal_resolve_{tag}")
    os.makedirs(d, exist_ok=True)
    # start clean so a stale gnn_config/train_config from a prior run can't leak in
    for fn in ("gnn_config.json", "train_config.json", "adapter_config.json"):
        p = os.path.join(d, fn)
        if os.path.exists(p):
            os.remove(p)
    return d


def _write_json(d, name, payload):
    with open(os.path.join(d, name), "w") as f:
        json.dump(payload, f)


# ==========================================================================
# _is_gnn_checkpoint — gnn_config.json presence is the discriminator
# ==========================================================================
def test_is_gnn_checkpoint_true_only_with_gnn_config():
    d = _ckpt_dir("isgnn")
    assert scal._is_gnn_checkpoint(d) is False
    _write_json(d, "gnn_config.json", {"architecture": "gt_llm"})
    assert scal._is_gnn_checkpoint(d) is True


# ==========================================================================
# _resolve_text_edge_list — CLI override precedence (beats everything, no file read)
# ==========================================================================
def test_resolve_cli_override_wins_for_gnn_without_reading_file():
    """A CLI override is returned verbatim even when it contradicts gnn_config.json."""
    d = _ckpt_dir("cli_gnn")
    _write_json(d, "gnn_config.json", {"text_edge_list": "none"})
    assert scal._resolve_text_edge_list(d, is_gnn=True, cli_override="present") == "present"


def test_resolve_cli_override_wins_with_no_files_at_all():
    """Override short-circuits before any file access, so a bare dir still resolves."""
    d = _ckpt_dir("cli_bare")
    assert scal._resolve_text_edge_list(d, is_gnn=False, cli_override="none") == "none"


# ==========================================================================
# _resolve_text_edge_list — graph checkpoint: read gnn_config.json
# ==========================================================================
def test_resolve_gnn_reads_recorded_policy():
    for recorded in ("present", "none"):
        d = _ckpt_dir(f"gnn_{recorded}")
        _write_json(d, "gnn_config.json", {"architecture": "gt_llm", "text_edge_list": recorded})
        assert scal._resolve_text_edge_list(d, is_gnn=True, cli_override=None) == recorded


def test_resolve_gnn_missing_key_raises_keyerror():
    """gnn_config.json without 'text_edge_list' must fail loud (KeyError), never default."""
    d = _ckpt_dir("gnn_nokey")
    _write_json(d, "gnn_config.json", {"architecture": "gt_llm"})  # no text_edge_list
    raised = False
    try:
        scal._resolve_text_edge_list(d, is_gnn=True, cli_override=None)
    except KeyError as e:
        raised = "text_edge_list" in str(e)
    assert raised, "missing text_edge_list in gnn_config.json must raise KeyError"


# ==========================================================================
# _resolve_text_edge_list — plain-LLM checkpoint: read train_config.json
# ==========================================================================
def test_resolve_llm_reads_train_config_policy():
    d = _ckpt_dir("llm_present")
    _write_json(d, "train_config.json", {"architecture": "llm", "text_edge_list": "present"})
    assert scal._resolve_text_edge_list(d, is_gnn=False, cli_override=None) == "present"


def test_resolve_llm_missing_file_raises_filenotfound():
    """A plain-LLM checkpoint with no train_config.json and no CLI override must raise
    FileNotFoundError — the train-time edge policy is unknown and must not be guessed."""
    d = _ckpt_dir("llm_nofile")  # cleaned: no train_config.json
    raised = False
    try:
        scal._resolve_text_edge_list(d, is_gnn=False, cli_override=None)
    except FileNotFoundError as e:
        raised = "train_config.json" in str(e)
    assert raised, "absent train_config.json (no override) must raise FileNotFoundError"


def test_resolve_llm_missing_key_raises_keyerror():
    d = _ckpt_dir("llm_nokey")
    _write_json(d, "train_config.json", {"architecture": "llm"})  # no text_edge_list
    raised = False
    try:
        scal._resolve_text_edge_list(d, is_gnn=False, cli_override=None)
    except KeyError as e:
        raised = "text_edge_list" in str(e)
    assert raised, "train_config.json without text_edge_list must raise KeyError"


# ==========================================================================
# _make_progress_printer — running cumulative accuracy arithmetic (stdout oracle)
# ==========================================================================
def _summary(name, *, num_correct, num_total, accuracy, elapsed_s=1.0):
    """Minimal GraphEvalResultSummary carrying only fields the printer reads."""
    return evaluate.GraphEvalResultSummary(
        name=name, num_total=num_total, num_correct=num_correct, accuracy=accuracy,
        subjective_accuracy=None, num_judged=0, num_formatted=num_total,
        num_keyword=num_correct, num_false_pos=0, num_false_neg=0, num_errors=0,
        elapsed_s=elapsed_s, n_nodes=3, use_icl=True, permutation=None,
        samples=[], path_metrics={},
    )


def test_progress_printer_accumulates_overall_accuracy():
    """Across two graphs (1/2 then 3/4), the printer's running 'overall' must be the
    cumulative micro-average: after g1 -> 1/2=50.0%, after g2 -> 4/6=66.7%."""
    samples_by_graph = {"g1": [object()], "g2": [object()]}  # only len() is used (2 graphs)
    printer = scal._make_progress_printer(samples_by_graph)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        printer("g1", _summary("g1", num_correct=1, num_total=2, accuracy=0.5))
        printer("g2", _summary("g2", num_correct=3, num_total=4, accuracy=0.75))
    out = buf.getvalue().splitlines()

    # Line 1: this-graph 1/2 (50.0%), overall 1/2 (50.0%).
    assert "[1/2] g1:" in out[0]
    assert "1/2" in out[0] and "overall: 50.0% (1/2)" in out[0]
    # Line 2: this-graph 3/4 (75.0%), overall 4/6 = 66.7%.
    assert "[2/2] g2:" in out[1]
    assert "overall: 66.7% (4/6)" in out[1]


# ==========================================================================
# Standalone runner (pytest is absent from the conda env)
# ==========================================================================
if __name__ == "__main__":
    passed, failed = 0, []
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                passed += 1
                print(f"{name}: PASS")
            except Exception as e:  # noqa: BLE001 — report, don't abort the suite
                failed.append((name, f"{type(e).__name__}: {e}"))
                print(f"{name}: FAIL — {type(e).__name__}: {e}")
    print(f"\n{passed} passed, {len(failed)} failed")
    for name, err in failed:
        print(f"  FAIL {name}: {err}")
    sys.exit(1 if failed else 0)
