"""Contract tests for the ORCHESTRATION GLUE of prism.eval.scalability_evaluation —
the parts of `main()` + `_parse_args` that nothing else in the suite exercises.

The module docstring promises "one script, two modes, switched by whether
`--permutation-seed` is given". The existing tests cover the leaf helpers in isolation
(test_eval_path_metrics_plumbing: writers + _NumpyEncoder; test_scalability_resolve_policy:
_resolve_text_edge_list / _is_gnn_checkpoint / _make_progress_printer). NONE of them verify
the routing decision itself:

  * which WRITER and which OUTPUT DIRECTORY each mode selects
      - no seeds  -> <checkpoint>/eval_logs/cross_eval/<graph>.json   (cross_eval shape)
      - seeds     -> <output>/perm_<seed>/<ckpt>_<graph>.json + <ckpt>_summary.json
  * the train/eval policy PLUMBED into the scorer:
      - include_edge_list == (text_edge_list == "present")
      - use_icl          == (--use-icl == "true")
      - one Permutation(seed) per seed, else None
  * _parse_args: required args, defaults, choices validation, nargs="+", store_true.

CS-only deterministic plumbing (argparse + file routing + dict/JSON construction). The model
load and the scoring path are the BOUNDARY: `_load_checkpoint` and
`evaluate.eval_model_multiple_graphs` are stubbed so no model is built. The stub for the
scorer also RECORDS the policy kwargs main() passed it, which is how we verify plumbing.
Oracles are restated from the docstring contract, never copied from main()'s body.

Run: conda run -n GREP-PRISM uv run python tests/test_scalability_main_routing.py
(pytest is not installed in the env; the __main__ runner mirrors the rest of the suite.)
"""
import json
import os
import sys

sys.path.insert(0, "src")

from prism.eval import evaluate
from prism.eval import scalability_evaluation as scal
from prism.models import utils as model_utils


_SCRATCH = os.environ.get("TMPDIR", "/tmp")


def _ckpt_dir(tag):
    """A bare (LLM-shaped, no gnn_config) checkpoint dir under scratch."""
    d = os.path.join(_SCRATCH, f"claude_scal_main_{tag}")
    os.makedirs(d, exist_ok=True)
    for fn in ("gnn_config.json", "train_config.json", "adapter_config.json"):
        p = os.path.join(d, fn)
        if os.path.exists(p):
            os.remove(p)
    return d


def _out_dir(tag):
    d = os.path.join(_SCRATCH, f"claude_scal_out_{tag}")
    os.makedirs(d, exist_ok=True)
    return d


def _summary(name, *, num_correct=1, num_total=2, accuracy=0.5):
    """A GraphEvalResultSummary the writers can serialize; samples carry one record so
    we can confirm the seeded file keeps samples and the summary strips them."""
    return evaluate.GraphEvalResultSummary(
        name=name, num_total=num_total, num_correct=num_correct, accuracy=accuracy,
        subjective_accuracy=None, num_judged=0, num_formatted=num_total,
        num_keyword=num_correct, num_false_pos=0, num_false_neg=0, num_errors=0,
        elapsed_s=1.0, n_nodes=3, use_icl=True, permutation=None,
        samples=[{"idx": 0, "correct": True}], path_metrics={"valid_path_rate": 1.0},
    )


class _Recorder:
    """Replaces the BOUNDARY: data.load_samples_by_graph, _load_checkpoint, and
    evaluate.eval_model_multiple_graphs. Records every policy kwarg the scorer is called
    with so the test can assert what main() plumbed through. No model is built."""

    def __init__(self, graph_names):
        self.graph_names = list(graph_names)
        self.calls = []  # one dict of recorded kwargs per scorer invocation

    def load_samples_by_graph(self, target):
        es = evaluate.EvalSample(task="t", answer="c", graph={"objects": []},
                                 init_node="a", graph_name="g")
        samples_by_graph = {n: [es] for n in self.graph_names}
        graph_file_by_name = {n: f"/data/{n}.json" for n in self.graph_names}
        return samples_by_graph, graph_file_by_name

    def load_checkpoint(self, checkpoint, four_bit, device):
        return object(), object(), False  # (model, tokenizer, is_gnn) — opaque stubs

    def eval_model_multiple_graphs(self, model, tokenizer, samples_by_graph, *,
                                   include_edge_list, use_icl, permutation, on_graph_done,
                                   edge_weights, injection_scope):
        self.calls.append({
            "include_edge_list": include_edge_list,
            "use_icl": use_icl,
            "permutation_seed": None if permutation is None else permutation.seed,
            "permutation_is_none": permutation is None,
            "edge_weights": edge_weights,
            "injection_scope": injection_scope,
        })
        return {n: _summary(n) for n in self.graph_names}


def _run_main(argv, recorder):
    """Drive scal.main() with stubbed boundaries and a fixed argv."""
    saved = (sys.argv, scal.data.load_samples_by_graph,
             scal._load_checkpoint, scal.evaluate.eval_model_multiple_graphs)
    try:
        sys.argv = ["scalability_evaluation"] + argv
        scal.data.load_samples_by_graph = recorder.load_samples_by_graph
        scal._load_checkpoint = recorder.load_checkpoint
        scal.evaluate.eval_model_multiple_graphs = recorder.eval_model_multiple_graphs
        scal.main()
    finally:
        (sys.argv, scal.data.load_samples_by_graph,
         scal._load_checkpoint, scal.evaluate.eval_model_multiple_graphs) = saved


# ==========================================================================
# main() — NO-SEED mode: cross_eval layout + default dir + policy plumbing
# ==========================================================================
def test_main_no_seed_writes_cross_eval_layout_and_default_dir():
    """No --permutation-seed: must write <checkpoint>/eval_logs/cross_eval/<graph>.json
    in the cross_eval shape (architecture/text_edge_list/samples/path_metrics), and must
    NOT create any perm_<seed> dir or *_summary.json."""
    ckpt = _ckpt_dir("noseed")
    rec = _Recorder(["g1"])
    _run_main(["--checkpoint", ckpt, "--graphs", "/data/g.json",
               "--text-edge-list", "present", "--use-icl", "false"], rec)

    expected = os.path.join(ckpt, "eval_logs", "cross_eval", "g1.json")
    assert os.path.exists(expected), f"cross_eval output not at default path: {expected}"
    log = json.load(open(expected))
    # cross_eval shape (oracle: docstring + _write_cross_eval_result contract).
    assert log["architecture"] == "llm"
    assert log["text_edge_list"] == "present"
    assert log["accuracy"] == 0.5 and log["num_correct"] == 1 and log["num_samples"] == 2
    assert "samples" in log and "path_metrics" in log
    # No seeded artifacts anywhere under the checkpoint.
    assert not os.path.exists(os.path.join(ckpt, "eval_logs", "cross_eval", "perm_0"))
    for f in os.listdir(os.path.join(ckpt, "eval_logs", "cross_eval")):
        assert not f.endswith("_summary.json"), f"unexpected seeded summary: {f}"


def test_main_no_seed_plumbs_policy_into_scorer():
    """include_edge_list must equal (text_edge_list=='present'); use_icl must equal
    (--use-icl=='true'); no seeds => permutation is None."""
    ckpt = _ckpt_dir("noseed_policy")
    rec = _Recorder(["g1"])
    _run_main(["--checkpoint", ckpt, "--graphs", "/data/g.json",
               "--text-edge-list", "none", "--use-icl", "true"], rec)
    assert len(rec.calls) == 1
    assert rec.calls[0]["include_edge_list"] is False   # text_edge_list == "none"
    assert rec.calls[0]["use_icl"] is True
    assert rec.calls[0]["permutation_is_none"] is True
    # No edge_weights key in the stub checkpoint's train_config.json => the
    # resolver returns the exact historical policy ("gaussian").
    assert rec.calls[0]["edge_weights"] == "gaussian"
    assert rec.calls[0]["injection_scope"] == "full_sequence"


# ==========================================================================
# main() — SEEDED mode: transferability layout + perm subdir + summary
# ==========================================================================
def test_main_seeded_writes_perm_subdir_trial_and_summary():
    """With --permutation-seed 7 and --output OUT: per-graph file lands at
    OUT/perm_7/<ckpt>_g1.json (keeping samples); a OUT/perm_7/<ckpt>_summary.json is
    written with trials whose samples are STRIPPED; permutation block records seed 7."""
    ckpt = _ckpt_dir("seeded")
    ckpt_name = os.path.basename(os.path.abspath(ckpt.rstrip("/")))
    out = _out_dir("seeded")
    rec = _Recorder(["g1"])
    _run_main(["--checkpoint", ckpt, "--graphs", "/data/g.json",
               "--text-edge-list", "present", "--permutation-seed", "7",
               "--output", out], rec)

    perm_dir = os.path.join(out, "perm_7")
    trial_file = os.path.join(perm_dir, f"{ckpt_name}_g1.json")
    summary_file = os.path.join(perm_dir, f"{ckpt_name}_summary.json")
    assert os.path.exists(trial_file), f"seeded trial file missing: {trial_file}"
    assert os.path.exists(summary_file), f"seeded summary missing: {summary_file}"

    trial = json.load(open(trial_file))
    assert "samples" in trial and trial["samples"][0]["idx"] == 0  # per-graph keeps samples
    assert trial["accuracy"] == 0.5 and trial["num_correct"] == 1

    summary = json.load(open(summary_file))
    assert len(summary["trials"]) == 1
    assert "samples" not in summary["trials"][0]                  # summary strips samples
    assert summary["trials"][0]["accuracy"] == 0.5
    assert summary["permutation"]["seed"] == 7                    # real Permutation.to_dict()

    # Seeded mode must NOT also emit the cross_eval default layout.
    assert not os.path.exists(os.path.join(ckpt, "eval_logs", "cross_eval", "g1.json"))


def test_main_seeded_passes_permutation_object_to_scorer():
    """The scorer receives a real Permutation seeded with the requested value."""
    ckpt = _ckpt_dir("seeded_perm")
    rec = _Recorder(["g1"])
    _run_main(["--checkpoint", ckpt, "--graphs", "/data/g.json",
               "--text-edge-list", "present", "--permutation-seed", "7",
               "--output", _out_dir("seeded_perm")], rec)
    assert rec.calls[0]["permutation_is_none"] is False
    assert rec.calls[0]["permutation_seed"] == 7


def test_main_multi_seed_one_dir_and_one_scorer_call_per_seed():
    """Two seeds => two perm_<seed> dirs and the scorer invoked once per seed, each with
    the matching Permutation. (Loop-control contract: seeds iterate, not collapse.)"""
    ckpt = _ckpt_dir("multiseed")
    ckpt_name = os.path.basename(os.path.abspath(ckpt.rstrip("/")))
    out = _out_dir("multiseed")
    rec = _Recorder(["g1"])
    _run_main(["--checkpoint", ckpt, "--graphs", "/data/g.json",
               "--text-edge-list", "present", "--permutation-seed", "7", "13",
               "--output", out], rec)
    assert os.path.exists(os.path.join(out, "perm_7", f"{ckpt_name}_summary.json"))
    assert os.path.exists(os.path.join(out, "perm_13", f"{ckpt_name}_summary.json"))
    assert [c["permutation_seed"] for c in rec.calls] == [7, 13]


def test_main_seeded_default_output_dir_is_results():
    """Without --output, seeded mode defaults the base dir to 'results' (relative to CWD),
    then nests perm_<seed>. We chdir into scratch so 'results/' lands there, not in repo."""
    ckpt = _ckpt_dir("seeded_default")
    ckpt_name = os.path.basename(os.path.abspath(ckpt.rstrip("/")))
    cwd_sandbox = _out_dir("seeded_default_cwd")
    rec = _Recorder(["g1"])
    saved_cwd = os.getcwd()
    try:
        os.chdir(cwd_sandbox)
        _run_main(["--checkpoint", ckpt, "--graphs", "/data/g.json",
                   "--text-edge-list", "present", "--permutation-seed", "5"], rec)
    finally:
        os.chdir(saved_cwd)
    assert os.path.exists(os.path.join(cwd_sandbox, "results", "perm_5",
                                       f"{ckpt_name}_summary.json"))


# ==========================================================================
# _parse_args — argument plumbing (required / defaults / choices / nargs)
# ==========================================================================
def _parse(argv):
    saved = sys.argv
    try:
        sys.argv = ["scalability_evaluation"] + argv
        return scal._parse_args()
    finally:
        sys.argv = saved


def test_parse_args_defaults():
    ns = _parse(["--checkpoint", "ck", "--graphs", "g"])
    assert ns.device == 0
    assert ns.four_bit is False
    assert ns.permutation_seed is None
    assert ns.output is None
    assert ns.use_icl == "true"          # default true (historical behavior)
    assert ns.text_edge_list is None     # no default; resolved later


def test_parse_args_permutation_seed_is_list_of_ints():
    ns = _parse(["--checkpoint", "ck", "--graphs", "g",
                 "--permutation-seed", "1", "2", "3"])
    assert ns.permutation_seed == [1, 2, 3]
    assert all(isinstance(s, int) for s in ns.permutation_seed)


def test_parse_args_four_bit_store_true():
    ns = _parse(["--checkpoint", "ck", "--graphs", "g", "--four-bit"])
    assert ns.four_bit is True


def test_parse_args_missing_required_raises():
    """--checkpoint and --graphs are required; argparse must exit (SystemExit), not return."""
    for argv in ([], ["--checkpoint", "ck"], ["--graphs", "g"]):
        raised = False
        try:
            _parse(argv)
        except SystemExit:
            raised = True
        assert raised, f"expected SystemExit for incomplete argv {argv!r}"


def test_parse_args_rejects_bad_choices():
    """choices=[...] on --text-edge-list and --use-icl must reject out-of-set values."""
    for argv in (["--checkpoint", "ck", "--graphs", "g", "--text-edge-list", "bogus"],
                 ["--checkpoint", "ck", "--graphs", "g", "--use-icl", "maybe"]):
        raised = False
        try:
            _parse(argv)
        except SystemExit:
            raised = True
        assert raised, f"expected SystemExit for bad choice in {argv!r}"


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
                import traceback
                failed.append((name, f"{type(e).__name__}: {e}"))
                print(f"{name}: FAIL — {type(e).__name__}: {e}")
                traceback.print_exc()
    print(f"\n{passed} passed, {len(failed)} failed")
    for name, err in failed:
        print(f"  FAIL {name}: {err}")
    sys.exit(1 if failed else 0)


def test_main_resolves_decode_consistent_from_train_config():
    """A checkpoint whose train_config.json records injection_scope=decode_consistent
    must thread that exact value into eval_model_multiple_graphs (the eval client
    arms MaskDecodeInjector off it)."""
    ckpt = _ckpt_dir("decode_scope")
    with open(os.path.join(ckpt, "train_config.json"), "w") as f:
        json.dump({"injection_scope": "decode_consistent",
                   "edge_weights": "binary"}, f)
    rec = _Recorder(["g1"])
    _run_main(["--checkpoint", ckpt, "--graphs", "/data/g.json",
               "--text-edge-list", "none", "--use-icl", "true"], rec)
    assert rec.calls[0]["injection_scope"] == "decode_consistent"
    assert rec.calls[0]["edge_weights"] == "binary"
