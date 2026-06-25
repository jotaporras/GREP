"""Deterministic-CS verification for ``prism.eval.render`` (the path-metrics figure
renderer) and its in-process driver ``evaluate.render_path_metrics_figure``.

CS-only scope: every symbol here is pure plumbing — run-name parsing, file
selection, per-graph metric averaging, sort order, and the matplotlib (Agg)
render path. No model is constructed or called; ``samples`` are plain dicts, the
exact shape ``GraphEvalResultSummary.samples`` carries.

Oracles are independent of the implementation: hand-computed means, hand-built
filename sets, and closed-form expected labels straight from the docstrings.

Run directly:  ``python tests/test_eval_render.py``
Or via pytest: ``pytest tests/test_eval_render.py``
"""
import sys
sys.path.insert(0, "src")

import tempfile
from pathlib import Path

from prism.eval import render
from prism.eval import evaluate


def _tmpdir(tmp_path) -> Path:
    """pytest passes a tmp_path fixture; the script runner passes None."""
    return Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())


# ---------------------------------------------------------------------------
# Run-name / label parsing (pure string functions, hand-computed oracles)
# ---------------------------------------------------------------------------

def test_graph_index_parses_int_and_defaults_negative():
    """``data_gen_NNN`` -> int(NNN) (leading zeros dropped); no match -> -1."""
    assert render._graph_index(Path("data_gen_004.json")) == 4
    assert render._graph_index(Path("data_gen_012.gemma.json")) == 12
    assert render._graph_index(Path("not_a_graph.json")) == -1


def test_graph_label_compacts_or_passes_through():
    """``data_gen_004`` -> ``G4``; an unindexed name returns itself."""
    assert render._graph_label("data_gen_004") == "G4"
    assert render._graph_label("weird_name") == "weird_name"


def test_split_run_extracts_wandb_id_suffix():
    """A trailing token that is lowercase-alnum, len 6-12, with >=1 letter AND
    >=1 digit is the wandb id; otherwise the whole name is the model."""
    assert render._split_run("e4_llm_x_xriur1bi") == ("e4_llm_x", "xriur1bi")
    # last token "gt": no digit, too short -> not an id.
    assert render._split_run("composite_graph_gt") == ("composite_graph_gt", None)


def test_split_run_rejects_non_id_suffixes():
    """All-digit, all-letter, or out-of-length suffixes are NOT wandb ids."""
    assert render._split_run("run_12345678")[1] is None   # no letter
    assert render._split_run("run_abcdefgh")[1] is None   # no digit
    assert render._split_run("run_a1")[1] is None         # too short (<6)
    assert render._split_run("run_" + "a1" * 7)[1] is None  # 14 chars (>12)
    # single token (no underscore) is never split, even if id-shaped.
    assert render._split_run("xriur1bi") == ("xriur1bi", None)


def test_run_label_matches_documented_format():
    """Docstring example: model 'composite_graph_gt' + id '3xk9p2af'
    -> 'COMPOSITE GRAPH GT (3xk9p2af)'."""
    assert render._run_label("composite_graph_gt_3xk9p2af") == \
        "COMPOSITE GRAPH GT (3xk9p2af)"
    # No id suffix -> bare upper-cased model name, no parens.
    assert render._run_label("composite_graph_gt") == "COMPOSITE GRAPH GT"


def test_arch_tag_badge_joins_present_parts_only():
    """Badge = arch · tag, dropping whichever part is absent."""
    assert render._arch_tag_badge("e4_llm_xriur1bi", "composite_graph_gt") == \
        "composite_graph_gt  ·  xriur1bi"
    assert render._arch_tag_badge("e4_llm_xriur1bi", None) == "xriur1bi"
    assert render._arch_tag_badge("plain_run", "llm") == "llm"
    assert render._arch_tag_badge("plain_run", None) == ""


# ---------------------------------------------------------------------------
# File selection (prefer .gemma.json, skip .judged.json) — real files on disk
# ---------------------------------------------------------------------------

def test_select_graph_files_prefers_gemma_skips_judged(tmp_path=None):
    """For one graph with both .json and .gemma.json present, the .gemma is
    chosen; a graph that exists only as .judged.json is dropped entirely."""
    d = _tmpdir(tmp_path)
    (d / "data_gen_001.json").write_text("{}")
    (d / "data_gen_001.gemma.json").write_text("{}")
    (d / "data_gen_002.judged.json").write_text("{}")  # judged-only -> skipped
    (d / "data_gen_003.json").write_text("{}")          # plain-only -> kept
    sel = render.select_graph_files(d)
    assert set(sel) == {"data_gen_001", "data_gen_003"}
    assert sel["data_gen_001"].name == "data_gen_001.gemma.json"
    assert sel["data_gen_003"].name == "data_gen_003.json"


# ---------------------------------------------------------------------------
# Per-graph metric averaging (hand-computed oracle)
# ---------------------------------------------------------------------------

def test_graph_metric_means_bools_average_none_dropped():
    """Booleans average to a [0,1] rate; all-None metrics drop out of the dict;
    the longest *valid* path is the max num_parsed over valid_path_ab samples."""
    samples = [
        {"correct": True,
         "path_metrics": {"edge_validity_rate": 1.0, "valid_path_ab": True,
                          "num_parsed": 3, "llm_judge_pass": None}},
        {"correct": False,
         "path_metrics": {"edge_validity_rate": 0.0, "valid_path_ab": False,
                          "llm_judge_pass": None}},
    ]
    means, longest = render.graph_metric_means_from_samples(samples)
    assert means["edge_validity_rate"] == 0.5    # mean(1.0, 0.0)
    assert means["valid_path_ab"] == 0.5         # mean(1, 0)
    assert means["correct"] == 0.5               # bool averaged, not crashed
    assert "llm_judge_pass" not in means          # entirely None -> absent
    assert longest == 3


def test_graph_metric_means_longest_falls_back_to_parsed_nodes():
    """When a valid sample has no usable num_parsed, longest uses len(parsed_nodes)."""
    samples = [
        {"path_metrics": {"valid_path_ab": True, "num_parsed": 0,
                          "parsed_nodes": ["a", "b", "c", "d"]}},
        # invalid path: contributes nothing to longest even with a long route.
        {"path_metrics": {"valid_path_ab": False, "num_parsed": 99}},
    ]
    _, longest = render.graph_metric_means_from_samples(samples)
    assert longest == 4


def test_graph_metric_means_empty_is_empty():
    """No samples -> empty mean dict and zero longest (no NaN, no crash)."""
    assert render.graph_metric_means_from_samples([]) == ({}, 0)


def test_collect_from_samples_sorts_by_numeric_graph_index():
    """Graphs sort by integer index, not lexically: data_gen_002 precedes
    data_gen_010 (string sort would invert these)."""
    by_graph = {
        "data_gen_010": [{"correct": True, "path_metrics": {}}],
        "data_gen_002": [{"correct": False, "path_metrics": {}}],
    }
    graphs, per_graph, longest = render.collect_from_samples(by_graph)
    assert graphs == ["data_gen_002", "data_gen_010"]
    assert set(per_graph) == {"data_gen_002", "data_gen_010"}
    assert set(longest) == {"data_gen_002", "data_gen_010"}


# ---------------------------------------------------------------------------
# Whole render path (matplotlib Agg) — smoke: writes a PNG / None on empty
# ---------------------------------------------------------------------------

def test_render_from_samples_writes_png(tmp_path=None):
    """A run with real samples produces a path_metrics_<run>.png file on disk."""
    out = _tmpdir(tmp_path)
    by_graph = {
        "data_gen_001": [
            {"correct": True,
             "path_metrics": {"edge_validity_rate": 1.0, "valid_path_ab": True,
                              "num_parsed": 2}},
        ],
    }
    p = render.render_from_samples("e4_llm_xriur1bi", by_graph, out,
                                   architecture="composite_graph_gt")
    assert p is not None
    assert Path(p).exists()
    assert Path(p).name == "path_metrics_e4_llm_xriur1bi.png"


def test_render_from_samples_none_when_no_graphs(tmp_path=None):
    """No graphs carrying samples -> returns None, writes nothing."""
    out = _tmpdir(tmp_path)
    assert render.render_from_samples("run", {}, out) is None


# ---------------------------------------------------------------------------
# evaluate.render_path_metrics_figure — in-process integration glue
# ---------------------------------------------------------------------------

def _summary(name, samples):
    return evaluate.GraphEvalResultSummary(
        name=name, num_total=len(samples), num_correct=0, accuracy=0.0,
        subjective_accuracy=None, num_judged=0, num_formatted=0, num_keyword=0,
        num_false_pos=0, num_false_neg=0, num_errors=0, elapsed_s=0.0,
        n_nodes=3, use_icl=True, permutation=None, samples=samples,
    )


def test_render_path_metrics_figure_writes_into_visuals(tmp_path=None):
    """Feeds live summaries to render; figure lands under <out_dir>/visuals/."""
    out = _tmpdir(tmp_path)
    results = {
        "data_gen_001": _summary("data_gen_001", [
            {"correct": True,
             "path_metrics": {"edge_validity_rate": 1.0, "valid_path_ab": True,
                              "num_parsed": 2}},
        ]),
    }
    p = evaluate.render_path_metrics_figure(results, str(out), "e4_llm_xriur1bi",
                                            architecture="composite_graph_gt")
    assert p is not None
    assert Path(p).exists()
    assert Path(p).parent.name == "visuals"


def test_render_path_metrics_figure_none_when_no_samples(tmp_path=None):
    """All-empty samples -> returns None (nothing to plot), no exception."""
    out = _tmpdir(tmp_path)
    results = {"data_gen_001": _summary("data_gen_001", [])}
    assert evaluate.render_path_metrics_figure(results, str(out), "run") is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"{name}: PASS")
    print("done")
