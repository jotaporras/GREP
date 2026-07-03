"""Tests for the ``text_edge_list`` -> ``include_edges`` policy.

``text_edge_list`` ("present"/"none") must toggle whether the textual edge list
appears in the LLM-facing prompt, for ALL architectures, at BOTH the training
text path and the eval text path, via a single bool
``include_edges = (text_edge_list == "present")``, with NO library-level defaults.

The single source of truth for whether the edge bullets appear is
``compact_prompt.spine_to_compact_messages(messages, include_edges=...)``; every
architecture (the five graph archs + the plain LLM), on both the training text
path (``data.preprocess_dataset``) and the eval text path
(``inference.InMemoryLLM`` / ``inference.GraphAugmentedInMemoryLLM``), must reach
the formatter through that one bool.

Two kinds of test live here:

1. PURE-TEXT MATRIX (executable anywhere — ``compact_prompt`` is torch-free):
   load a real conversation and assert the formatter writes the edge bullets and
   selects the WITH_EDGES system prompt IFF ``include_edges`` is True.

2. WIRING CHECKS (torch-free, via ``ast``): ``data.py`` / ``inference.py`` /
   ``evaluate.py`` / ``compact_prompt.py`` cannot be imported locally (they pull
   torch / datasets / transformers), so the wiring is verified by statically
   parsing the source — robust and import-free.

An optional model-level integration test (real tokenizer + ``preprocess_dataset``)
is guarded by ``pytest.importorskip`` so it is collectable everywhere and only
RUNS where torch + a tokenizer are available (the cluster). Mirrors the
conventions in ``tests/test_graph_mask.py``.
"""
import ast
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")

# Repo-anchored paths so the file works regardless of cwd (the ast checks read
# source straight from disk; nothing here imports torch).
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

from prism.data import compact_prompt  # noqa: E402  (torch-free; safe to import)

VAL_JSON = REPO_ROOT / "data/gen/e5_graph_oriented_data/split/formatted_all_new__val.json"
DATA_PY = SRC / "prism/data/data.py"
INFERENCE_PY = SRC / "prism/models/inference.py"
EVALUATE_PY = SRC / "prism/eval/evaluate.py"
COMPACT_PY = SRC / "prism/data/compact_prompt.py"

# The five graph architectures + the plain LLM baseline. Every one of these must
# thread the edge flag through ``spine_to_compact_messages``.
GRAPH_ARCHS = ("rpearl_llm", "rpearl_gt_llm", "gt_llm", "graph_mask_llm", "learnable_graph_mask")
ALL_ARCHS = GRAPH_ARCHS + ("llm",)


# ---------------------------------------------------------------------------
# ast helpers (torch-free static inspection)
# ---------------------------------------------------------------------------

def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _find_func(tree: ast.AST, name: str, cls: str = None):
    """Find a module-level FunctionDef, or a method ``name`` inside class ``cls``."""
    if cls is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == cls:
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef) and sub.name == name:
                        return sub
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _param_default_status(func: ast.FunctionDef, argname: str) -> str:
    """Return 'absent' | 'no_default' | 'has_default' for ``argname`` on ``func``."""
    a = func.args
    pos = a.args
    n_defaults = len(a.defaults)
    first_default = len(pos) - n_defaults
    for i, arg in enumerate(pos):
        if arg.arg == argname:
            return "has_default" if i >= first_default else "no_default"
    for arg, default in zip(a.kwonlyargs, a.kw_defaults):
        if arg.arg == argname:
            return "has_default" if default is not None else "no_default"
    return "absent"


def _callee_name(call: ast.Call):
    f = call.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return None


def _calls_to(scope: ast.AST, name: str):
    return [n for n in ast.walk(scope) if isinstance(n, ast.Call) and _callee_name(n) == name]


def _kw(call: ast.Call, key: str):
    for k in call.keywords:
        if k.arg == key:
            return k.value
    return None


def _is_name(node, ident: str) -> bool:
    return isinstance(node, ast.Name) and node.id == ident


def _is_const_true(node) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _find_if_on(func: ast.FunctionDef, ident: str):
    """The first ``if`` statement in ``func`` whose test mentions Name ``ident``."""
    for node in ast.walk(func):
        if isinstance(node, ast.If):
            if any(_is_name(n, ident) for n in ast.walk(node.test)):
                return node
    return None


# ---------------------------------------------------------------------------
# 1. PURE-TEXT MATRIX  (executable locally; compact_prompt is the source of truth)
# ---------------------------------------------------------------------------

def _load_conversations():
    assert VAL_JSON.exists(), f"missing eval data: {VAL_JSON}"
    data = json.loads(VAL_JSON.read_text())
    assert isinstance(data, list) and data, "val file must be a non-empty JSON list"
    # Keep only conversations that carry a scene-graph-bearing user turn (every
    # one does, but be explicit so the formatter has a graph to hoist).
    convs = [
        item["conversations"] for item in data
        if any(m.get("role") == "user" and re.search(r"[Ss]cene graph:", m.get("content", ""))
               for m in item["conversations"])
    ]
    assert convs, "no scene-graph-bearing conversation found in val file"
    return convs


def test_pure_text_matrix_edges_present_iff_include_edges():
    """On real conversations: ``• Region Edges:`` / ``• Object Edges:`` appear in
    the hoisted system message IFF ``include_edges`` is True, and the matching
    system-prompt variant is selected. Cover several conversations."""
    convs = _load_conversations()
    for conv in convs[:10]:
        for include_edges in (True, False):
            out = compact_prompt.spine_to_compact_messages(conv, include_edges=include_edges)
            assert out and out[0]["role"] == "system", "graph must hoist to a leading system message"
            sys_content = out[0]["content"]

            has_region = "Region Edges" in sys_content
            has_object = "Object Edges" in sys_content
            assert has_region is include_edges, (
                f"'Region Edges' present={has_region} but include_edges={include_edges}")
            assert has_object is include_edges, (
                f"'Object Edges' present={has_object} but include_edges={include_edges}")

            # Exactly the right system-prompt variant is chosen. The two prompts
            # diverge (neither is a substring of the other), so membership is clean.
            with_edges_used = compact_prompt.COMPACT_SYSTEM_PROMPT_WITH_EDGES in sys_content
            plain_used = compact_prompt.COMPACT_SYSTEM_PROMPT in sys_content
            assert with_edges_used is include_edges, (
                f"WITH_EDGES prompt used={with_edges_used} but include_edges={include_edges}")
            assert plain_used is (not include_edges), (
                f"plain prompt used={plain_used} but include_edges={include_edges}")


def test_pure_text_matrix_only_difference_is_edges_block():
    """The two variants differ ONLY by the presence of the edge bullets + the
    matching prompt; the node-name lines are identical, proving the toggle adds
    edges rather than rebuilding an unrelated block."""
    conv = _load_conversations()[0]
    on = compact_prompt.spine_to_compact_messages(conv, include_edges=True)[0]["content"]
    off = compact_prompt.spine_to_compact_messages(conv, include_edges=False)[0]["content"]
    # Node-name bullets identical across variants.
    for line in off.splitlines():
        if line.startswith("• Region nodes:") or line.startswith("• Object nodes:") \
                or line.startswith("• Robot location:"):
            assert line in on, f"node line changed by toggle: {line!r}"
    # Edge bullets appear only in the ON variant.
    assert "• Region Edges:" in on and "• Object Edges:" in on
    assert "• Region Edges:" not in off and "• Object Edges:" not in off


# ---------------------------------------------------------------------------
# 2. WIRING CHECKS  (torch-free: parse the source instead of importing torch)
# ---------------------------------------------------------------------------

def test_compact_prompt_no_false_defaults_on_include_edges():
    """The four formatter functions (and indeed every ``include_edges``-bearing
    function in the module) must declare ``include_edges`` with NO default."""
    tree = _parse(COMPACT_PY)
    named = ("_graph_block", "_system_content", "build_conversation", "spine_to_compact_messages")
    for fn in named:
        func = _find_func(tree, fn)
        assert func is not None, f"compact_prompt.{fn} not found"
        assert _param_default_status(func, "include_edges") == "no_default", (
            f"compact_prompt.{fn} must take include_edges with no default")
    # Stronger sweep: NO function anywhere in compact_prompt may default include_edges.
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            status = _param_default_status(node, "include_edges")
            assert status != "has_default", (
                f"compact_prompt.{node.name} defaults include_edges (no library defaults allowed)")


def test_data_preprocess_resolves_include_edges_from_text_edge_list():
    """``preprocess_dataset`` computes ``include_edges = (text_edge_list == 'present')``."""
    tree = _parse(DATA_PY)
    func = _find_func(tree, "preprocess_dataset")
    assert func is not None, "data.preprocess_dataset not found"
    assert "text_edge_list" in {a.arg for a in func.args.args}, \
        "preprocess_dataset must take a text_edge_list parameter"
    found = False
    for node in ast.walk(func):
        if isinstance(node, ast.Assign) and any(
                _is_name(t, "include_edges") for t in node.targets):
            v = node.value
            assert isinstance(v, ast.Compare) and _is_name(v.left, "text_edge_list") \
                and len(v.ops) == 1 and isinstance(v.ops[0], ast.Eq) \
                and isinstance(v.comparators[0], ast.Constant) \
                and v.comparators[0].value == "present", (
                    "include_edges must be (text_edge_list == 'present')")
            found = True
    assert found, "no `include_edges = (text_edge_list == 'present')` assignment in preprocess_dataset"


def test_data_preprocess_passes_flag_to_every_translate_call():
    """Every ``spine_to_compact_messages`` call in the (now arch-agnostic)
    ``preprocess_dataset`` passes the resolved ``include_edges`` bool — never a
    hardcoded True, never a bare call."""
    tree = _parse(DATA_PY)
    func = _find_func(tree, "preprocess_dataset")
    calls = _calls_to(func, "spine_to_compact_messages")
    assert calls, "preprocess_dataset has no spine_to_compact_messages call"
    for call in calls:
        v = _kw(call, "include_edges")
        assert v is not None, "call missing include_edges kwarg"
        assert _is_name(v, "include_edges"), "include_edges not passed as the resolved bool"
        assert not _is_const_true(v), "include_edges hardcoded True"


def test_inference_clients_take_include_edges_with_no_default():
    """Both clients accept ``include_edges`` with NO default and store it."""
    tree = _parse(INFERENCE_PY)
    for cls in ("InMemoryLLM", "GraphAugmentedInMemoryLLM"):
        init = _find_func(tree, "__init__", cls)
        assert init is not None, f"{cls}.__init__ not found"
        assert _param_default_status(init, "include_edges") == "no_default", \
            f"{cls}.__init__ must take include_edges with no default"
    # InMemoryLLM stores self.include_edges; GraphAugmentedInMemoryLLM inherits it
    # by passing include_edges into super().__init__.
    base_init = _find_func(tree, "__init__", "InMemoryLLM")
    stores = any(
        isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == "include_edges"
                and isinstance(t.value, ast.Name) and t.value.id == "self"
                for t in n.targets)
        for n in ast.walk(base_init)
    )
    assert stores, "InMemoryLLM.__init__ must store self.include_edges"
    sub_init = _find_func(tree, "__init__", "GraphAugmentedInMemoryLLM")
    super_calls = [c for c in ast.walk(sub_init) if isinstance(c, ast.Call)
                   and isinstance(c.func, ast.Attribute) and c.func.attr == "__init__"]
    assert any(_is_name(_kw(c, "include_edges"), "include_edges") for c in super_calls), \
        "GraphAugmentedInMemoryLLM must forward include_edges to super().__init__"


def test_inference_query_llm_passes_self_include_edges():
    """Both clients' ``query_llm`` call the formatter with
    ``include_edges=self.include_edges`` (no hardcoded True, no bare call)."""
    tree = _parse(INFERENCE_PY)
    for cls in ("InMemoryLLM", "GraphAugmentedInMemoryLLM"):
        q = _find_func(tree, "query_llm", cls)
        assert q is not None, f"{cls}.query_llm not found"
        calls = _calls_to(q, "spine_to_compact_messages")
        assert calls, f"{cls}.query_llm does not call spine_to_compact_messages"
        for call in calls:
            v = _kw(call, "include_edges")
            assert v is not None, f"{cls}.query_llm: include_edges kwarg missing"
            assert isinstance(v, ast.Attribute) and v.attr == "include_edges" \
                and isinstance(v.value, ast.Name) and v.value.id == "self", \
                f"{cls}.query_llm must pass include_edges=self.include_edges"
            assert not _is_const_true(v), f"{cls}.query_llm hardcodes include_edges=True"


def test_evaluate_constructs_both_clients_with_include_edge_list():
    """``evaluate.py`` builds InMemoryLLM and GraphAugmentedInMemoryLLM with
    ``include_edges=include_edge_list`` and no longer references ``strip_edges``."""
    tree = _parse(EVALUATE_PY)
    func = _find_func(tree, "eval_model_single_graph")
    assert func is not None, "eval_model_single_graph not found"
    for client in ("InMemoryLLM", "GraphAugmentedInMemoryLLM"):
        calls = _calls_to(func, client)
        assert calls, f"eval_model_single_graph does not construct {client}"
        for call in calls:
            v = _kw(call, "include_edges")
            assert _is_name(v, "include_edge_list"), \
                f"{client} must be built with include_edges=include_edge_list"
    # strip_edges must not appear anywhere in evaluate.py source.
    assert "strip_edges" not in EVALUATE_PY.read_text(), "strip_edges still referenced in evaluate.py"


def test_no_strip_edges_anywhere_in_src():
    """The identifier ``strip_edges`` must not occur anywhere under src/."""
    offenders = []
    for py in SRC.rglob("*.py"):
        if re.search(r"\bstrip_edges\b", py.read_text()):
            offenders.append(str(py))
    assert not offenders, f"strip_edges still present in: {offenders}"


# ---------------------------------------------------------------------------
# Optional model-level integration (real tokenizer + preprocess_dataset).
# Skipped where torch / a tokenizer are unavailable (locally); RUNS on cluster.
# ---------------------------------------------------------------------------

def test_preprocess_dataset_toggles_edges_end_to_end():
    """CLUSTER-ONLY end-to-end: with a real tokenizer, ``preprocess_dataset``
    produces compact messages whose system content carries the edge bullets IFF
    ``text_edge_list == 'present'`` — for BOTH a graph arch and the plain LLM.

    Skipped automatically wherever torch / datasets / transformers / a tokenizer
    are not present (e.g. the local workstation)."""
    import os
    import pytest  # local import so the module imports without pytest installed

    pytest.importorskip("torch")
    pytest.importorskip("datasets")
    pytest.importorskip("transformers")

    import datasets
    from transformers import AutoTokenizer
    from prism.data import data as data_mod

    tok_id = os.environ.get("GREP_TEST_TOKENIZER", "google/gemma-4-12B-it")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_id)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"tokenizer {tok_id!r} unavailable: {e}")
    if tokenizer.chat_template is None:
        pytest.skip(f"tokenizer {tok_id!r} has no chat template")

    raw = json.loads(VAL_JSON.read_text())[:2]
    base = datasets.Dataset.from_list([{"conversations": r["conversations"]} for r in raw])

    for arch in ("llm", "rpearl_llm"):
        for text_edge_list, want_edges in (("present", True), ("none", False)):
            ds = data_mod.preprocess_dataset(
                base, tokenizer, text_edge_list=text_edge_list)
            sys_msgs = [m["content"] for ex in ds for m in ex["messages"] if m["role"] == "system"]
            assert sys_msgs, f"{arch}/{text_edge_list}: no system message produced"
            for content in sys_msgs:
                assert ("Region Edges" in content) is want_edges, (
                    f"{arch}/{text_edge_list}: Region Edges present={'Region Edges' in content} "
                    f"want={want_edges}")
                assert ("Object Edges" in content) is want_edges, (
                    f"{arch}/{text_edge_list}: Object Edges mismatch")


if __name__ == "__main__":
    failures = []
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"{name}: PASS")
            except (ImportError, ModuleNotFoundError) as e:
                # The cluster-only integration test imports pytest/torch/etc, which
                # are absent locally; under __main__ that surfaces as an import
                # error -> SKIP (pytest.importorskip handles it cleanly on cluster).
                print(f"{name}: SKIP ({type(e).__name__}: {e})")
            except Exception as e:  # noqa: BLE001
                if e.__class__.__name__ in ("Skipped", "OutcomeException") or "importorskip" in repr(e):
                    print(f"{name}: SKIP ({e})")
                else:
                    failures.append(name)
                    print(f"{name}: FAIL -> {type(e).__name__}: {e}")
    print("done" if not failures else f"FAILURES: {failures}")
