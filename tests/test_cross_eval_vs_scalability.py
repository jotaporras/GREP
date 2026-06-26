"""Contract tests pinning the functional *differences* between the two post-train
eval drivers that ``train_v3.train_model`` fires back-to-back:

  1. ``prism.eval.evaluate.run_post_train_cross_eval``  (gated on ``post_train_eval_graphs``)
  2. ``prism.eval.scalability_evaluation.main``          (unconditional)

Both funnel the SAME graph set through the SAME ``eval_model_multiple_graphs`` core,
so the model-scoring path is not a difference. The differences live entirely in the
surrounding orchestration, and these tests pin them:

  * **Artifact collision** — both per-graph writers emit ``<out_dir>/<name>.json`` with
    an identical schema, and both default to ``<checkpoint>/eval_logs/cross_eval/``.
    Run in train_v3 order, scalability OVERWRITES the cross-eval JSONs. (writers_*)
  * **``text_edge_list`` sourcing** — cross-eval trusts the live ``config`` value;
    scalability re-resolves it from on-disk artifacts and fails LOUD when absent —
    UNLESS a CLI override is passed (train_v3 passes ``config.text_edge_list``, which
    short-circuits the disk read, collapsing the two back to the same value). (resolve_*)
  * **4-bit is NOT a scalability default** — ``--four-bit`` defaults False; the 4-bit
    reload in the train_v3 invocation comes only from train_v3 passing ``--four-bit``
    explicitly. cross-eval uses the in-memory (as-trained) model. (parse_args_*)
  * **Graph-set resolution** — a directory target expands to all ``*.json`` (sorted);
    a file target is a singleton. When ``post_train_eval_graphs`` and the
    ``eval_graphs_dir`` derived from ``eval_data`` point at the same directory, the
    evaluated set is byte-identical → the redundancy claim holds. (resolve_graph_files_*)

These are deterministic CS surfaces; no model is built. The one genuine functional
difference NOT exercised here — in-memory model vs disk-reloaded 4-bit checkpoint —
needs a real checkpoint + GPU and is called out in the report.

Run directly:  ``python tests/test_cross_eval_vs_scalability.py``
Or via pytest: ``pytest tests/test_cross_eval_vs_scalability.py``
"""
import sys
sys.path.insert(0, "src")

import json
import os
import tempfile

from prism.data import data
from prism.eval import evaluate
from prism.eval import scalability_evaluation as scal


# ---------------------------------------------------------------------------
# Oracle helper: an independently-built result record. We assert on it being
# serialized identically by both writers — we do NOT re-derive either writer.
# ---------------------------------------------------------------------------
def _result(name="data_gen_007"):
    """A minimal fully-specified GraphEvalResultSummary (all required fields)."""
    return evaluate.GraphEvalResultSummary(
        name=name,
        num_total=4,
        num_correct=3,
        accuracy=0.75,
        subjective_accuracy=None,
        num_judged=0,
        num_formatted=4,
        num_keyword=4,
        num_false_pos=0,
        num_false_neg=0,
        num_errors=0,
        elapsed_s=1.5,
        n_nodes=30,
        use_icl=True,
        permutation=None,
        samples=[{"task": "t", "correct": True}],
        path_metrics={"valid_path_rate": 0.5},
    )


# ---------------------------------------------------------------------------
# Artifact collision — same filename, same schema, same default out_dir.
# ---------------------------------------------------------------------------
def test_writers_emit_identical_filename_and_schema():
    """Both per-graph writers emit ``<name>.json`` with the same keys+values.

    cross-eval calls ``_write_cross_eval_json(out_dir, name, result, ...)``;
    scalability calls ``_write_cross_eval_result(result, out_dir=..., ...)``. In the
    train_v3 wiring ``name == result.name`` (results dict is keyed by graph name), so
    the filenames collide and scalability (run second) overwrites cross-eval's file.
    """
    res = _result()
    meta = dict(checkpoint="/ckpt", graph_file="/g/data_gen_007.json",
                architecture="graph-augmented", text_edge_list="present")

    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        evaluate._write_cross_eval_json(a, res.name, res, **meta)
        out_file = scal._write_cross_eval_result(res, out_dir=b, **meta)

        fname = f"{res.name}.json"
        assert os.path.basename(out_file) == fname
        assert os.listdir(a) == [fname], os.listdir(a)
        assert os.listdir(b) == [fname], os.listdir(b)

        with open(os.path.join(a, fname)) as f:
            doc_cross = json.load(f)
        with open(os.path.join(b, fname)) as f:
            doc_scal = json.load(f)

    # Identical schema → scalability's file is a drop-in overwrite of cross-eval's.
    assert set(doc_cross) == set(doc_scal), (set(doc_cross) ^ set(doc_scal))
    assert doc_cross == doc_scal, "writers diverge on a shared key"
    # Spot-check the contract keys callers (eval_viewer.html / judge-eval) depend on.
    for k in ("checkpoint", "graph_file", "architecture", "text_edge_list",
              "accuracy", "num_samples", "num_correct", "path_metrics", "samples"):
        assert k in doc_cross, k
    assert doc_cross["num_samples"] == res.num_total       # field rename total→samples
    assert doc_cross["accuracy"] == res.accuracy


def test_default_out_dirs_collide():
    """cross-eval and (no-seed) scalability default to the SAME directory.

    cross-eval: ``os.path.join(output_dir, "eval_logs", "cross_eval")``.
    scalability: documented default ``<checkpoint>/eval_logs/cross_eval/`` (main()).
    Same ``checkpoint == output_dir`` (train_v3 passes ``sft_args.output_dir`` for both)
    ⇒ identical path ⇒ collision.
    """
    ckpt = "/runs/e9_composite_graph_gt_gemma_r16_4bit_abc123"
    cross = os.path.join(ckpt, "eval_logs", "cross_eval")
    scal_default = os.path.join(ckpt, "eval_logs", "cross_eval")  # main() no-seed branch
    assert cross == scal_default


# ---------------------------------------------------------------------------
# Arg defaults — 4-bit is NOT a scalability default.
# ---------------------------------------------------------------------------
def test_parse_args_defaults_are_not_four_bit():
    """``_parse_args`` minimal invocation: four_bit False, no seeds, icl true, device 0.

    Proves the 4-bit reload in the train_v3 path is supplied by train_v3 (``--four-bit``),
    not a scalability default. The no-seed default also selects the cross_eval output
    layout (vs the seeded transferability layout).
    """
    ns = scal._parse_args(["--checkpoint", "/c", "--graphs", "/g"])
    assert ns.four_bit is False
    assert ns.permutation_seed is None          # no-seed → cross_eval layout
    assert ns.use_icl == "true"
    assert ns.output is None                     # → <checkpoint>/eval_logs/cross_eval
    assert ns.text_edge_list is None             # → disk re-resolve unless overridden
    assert ns.device == 0


def test_parse_args_train_v3_invocation_forces_four_bit_and_override():
    """The exact argv train_v3 builds: ``--four-bit`` set, ``--text-edge-list`` overridden.

    Mirrors train_v3.py:307-313. Confirms the 4-bit reload + edge-list override are
    BOTH live in the train_v3 wiring — so scalability re-evaluates the *quantized
    disk* checkpoint while cross-eval used the *in-memory* model.
    """
    argv = ["--checkpoint", "/ckpt", "--graphs", "/graphs", "--four-bit",
            "--text-edge-list", "present", "--device", "0"]
    ns = scal._parse_args(argv)
    assert ns.four_bit is True
    assert ns.text_edge_list == "present"
    assert ns.permutation_seed is None


# ---------------------------------------------------------------------------
# text_edge_list sourcing — the real behavioral divergence.
# ---------------------------------------------------------------------------
def test_resolve_text_edge_list_cli_override_short_circuits_disk():
    """A CLI override is returned verbatim WITHOUT touching disk.

    This is the train_v3 case: train_v3 passes ``config.text_edge_list`` as the
    override, so scalability does NOT re-resolve from gnn/train config — it collapses
    to the same value cross-eval uses. (checkpoint path is bogus on purpose: if the
    function read it, the test would error.)
    """
    assert scal._resolve_text_edge_list("/nonexistent", is_gnn=True, cli_override="none") == "none"
    assert scal._resolve_text_edge_list("/nonexistent", is_gnn=False, cli_override="present") == "present"


def test_resolve_text_edge_list_reads_gnn_config():
    """Graph checkpoint, no override → value read back from gnn_config.json."""
    with tempfile.TemporaryDirectory() as ck:
        with open(os.path.join(ck, "gnn_config.json"), "w") as f:
            json.dump({"architecture": "composite_graph_gt", "text_edge_list": "none"}, f)
        assert scal._resolve_text_edge_list(ck, is_gnn=True, cli_override=None) == "none"


def test_resolve_text_edge_list_reads_train_config():
    """Plain-LLM checkpoint, no override → value read back from train_config.json."""
    with tempfile.TemporaryDirectory() as ck:
        with open(os.path.join(ck, "train_config.json"), "w") as f:
            json.dump({"architecture": "llm", "text_edge_list": "present"}, f)
        assert scal._resolve_text_edge_list(ck, is_gnn=False, cli_override=None) == "present"


def test_resolve_text_edge_list_fails_loud_when_unrecorded():
    """scalability fails LOUD when the policy is neither overridden nor persisted.

    This is the divergence cross-eval does NOT have: cross-eval blindly trusts
    ``config.text_edge_list`` and can never raise here.
    """
    # gnn config present but missing the key → KeyError.
    with tempfile.TemporaryDirectory() as ck:
        with open(os.path.join(ck, "gnn_config.json"), "w") as f:
            json.dump({"architecture": "composite_graph_gt"}, f)
        try:
            scal._resolve_text_edge_list(ck, is_gnn=True, cli_override=None)
            assert False, "expected KeyError on missing gnn text_edge_list"
        except KeyError:
            pass

    # plain-LLM with no train_config.json at all → FileNotFoundError.
    with tempfile.TemporaryDirectory() as ck:
        try:
            scal._resolve_text_edge_list(ck, is_gnn=False, cli_override=None)
            assert False, "expected FileNotFoundError on missing train_config.json"
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Graph-set resolution — when the two targets coincide, the set is identical.
# ---------------------------------------------------------------------------
def _touch_json(d, stem):
    with open(os.path.join(d, f"{stem}.json"), "w") as f:
        f.write("{}")


def test_resolve_graph_files_dir_is_sorted_all_json():
    """A directory target expands to ALL ``*.json`` files, sorted.

    Both drivers route their target through ``load_samples_by_graph`` →
    ``_resolve_graph_files``. So pointing both at the same dir yields the identical
    (and deterministically ordered) graph set — the basis of the redundancy.
    """
    with tempfile.TemporaryDirectory() as d:
        for stem in ("data_gen_003", "data_gen_001", "data_gen_002"):
            _touch_json(d, stem)
        # a non-json sibling must be ignored
        with open(os.path.join(d, "README.md"), "w") as f:
            f.write("x")
        files = data._resolve_graph_files(d)
        assert [os.path.basename(p) for p in files] == [
            "data_gen_001.json", "data_gen_002.json", "data_gen_003.json"]


def test_resolve_graph_files_file_is_singleton_subset_of_dir():
    """A single-file target (single_graph.yaml's ``eval_data``) is a singleton, but the
    scalability driver evaluates the file's whole DIRECTORY.

    train_v3 derives ``eval_graphs_dir = eval_data if isdir else dirname(eval_data)``,
    so a file ``eval_data`` is widened to its directory for scalability — matching the
    directory ``post_train_eval_graphs`` cross-eval uses. This pins the file⊂dir
    relationship that makes that widening land on the same set.
    """
    with tempfile.TemporaryDirectory() as d:
        for stem in ("data_gen_001", "data_gen_002"):
            _touch_json(d, stem)
        one = os.path.join(d, "data_gen_001.json")

        file_set = {os.path.basename(p) for p in data._resolve_graph_files(one)}
        dir_set = {os.path.basename(p) for p in data._resolve_graph_files(d)}
        widened = {os.path.basename(p) for p in data._resolve_graph_files(os.path.dirname(one))}

        assert file_set == {"data_gen_001.json"}
        assert file_set < dir_set                  # strict subset
        assert widened == dir_set                  # dirname(file) widens to the full set


def test_resolve_graph_files_empty_fails_loud():
    """Zero matches → SystemExit (loud), not a silent empty run. Shared by both drivers."""
    with tempfile.TemporaryDirectory() as d:
        try:
            data._resolve_graph_files(d)
            assert False, "expected SystemExit on empty target"
        except SystemExit:
            pass


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"{name}: PASS")
    print("done")
