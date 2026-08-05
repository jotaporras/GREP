"""Static smoke test for the ``disable_graph_token_rope`` wiring — no torch, no ML.

The flag is only meaningful if the SAME value reaches all four stations:

    sbatch override -> build_planner_model -> train_config.json -> loaders rebuild

A break at any one of them is silent: the run trains with identity-RoPE and evaluates
without it (or the reverse), and every number it produces is off a configuration that
was never actually run. The behavioural tests live next to the model
(``test_learnable_graph_mask.py``: the position_ids the wrapper builds;
``test_decode_style_mask.py``: decode parity; ``test_eval_inference_path.py``: a real
generate). This file guards only the plumbing between them, by AST/text so it stays
sub-second and drags in no dependencies.
"""

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
KEY = "disable_graph_token_rope"

ARCHITECTURES = REPO / "src" / "prism" / "models" / "architectures.py"
LOADERS = REPO / "src" / "prism" / "models" / "loaders.py"
TRAIN_V3 = REPO / "src" / "prism" / "training" / "train_v3.py"
INFERENCE = REPO / "src" / "prism" / "models" / "inference.py"
GNN_LLM = REPO / "src" / "prism" / "models" / "gnn_llm.py"
E14 = REPO / "scripts" / "e14_stage1to3_binary.sbatch"

# Every architecture wrapper that must accept the flag as a constructor kwarg. The mask
# entry is the one added here; the additive family is listed so a refactor that drops it
# from one of them fails loudly rather than quietly reverting those runs to normal RoPE.
FLAG_CONSUMERS = ("LearnableGraphMaskLLM", "GraphAugmentedLLM")


def _tree(path):
    return ast.parse(path.read_text())


def _calls(tree, func_name):
    """Every ast.Call to ``func_name`` (bare or attribute), anywhere in the tree."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
        if name == func_name:
            out.append(node)
    return out


def _kwargs(call):
    return {kw.arg for kw in call.keywords if kw.arg}


def test_constructors_accept_the_flag():
    """Both wrappers take it as a named constructor parameter (not **kwargs)."""
    tree = _tree(GNN_LLM)
    classes = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    for cls in FLAG_CONSUMERS:
        assert cls in classes, f"{cls} disappeared from gnn_llm.py"
        init = next(n for n in classes[cls].body
                    if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        params = {a.arg for a in init.args.args} | {a.arg for a in init.args.kwonlyargs}
        assert KEY in params, f"{cls}.__init__ no longer accepts {KEY}"


def test_build_planner_model_forwards_the_flag_to_every_consumer():
    """architectures.build_planner_model is the ONLY place training constructs these
    wrappers, so a missing kwarg here makes the config switch inert."""
    tree = _tree(ARCHITECTURES)
    for cls in FLAG_CONSUMERS:
        calls = _calls(tree, cls)
        assert calls, f"architectures.py no longer constructs {cls}"
        for call in calls:
            assert KEY in _kwargs(call), \
                f"architectures.py constructs {cls} without {KEY} — the flag would be inert"


def test_train_v3_records_the_flag_in_gnn_config():
    """gnn_config becomes train_config.json. Absent here, loaders' ``.get(KEY, False)``
    always wins at eval and an identity-RoPE checkpoint is scored without it."""
    src = TRAIN_V3.read_text()
    assert re.search(rf'"{KEY}":\s*config\.model\.{KEY}', src), \
        f"train_v3 does not record {KEY} in gnn_config"


def test_loaders_rebuild_reads_the_flag_for_every_consumer():
    """Eval rebuilds from train_config.json; each construction site must read the key
    back, or the reloaded model is a different function than the trained one."""
    tree = _tree(LOADERS)
    for cls in FLAG_CONSUMERS:
        calls = _calls(tree, cls)
        assert calls, f"loaders.py no longer rebuilds {cls}"
        for call in calls:
            assert KEY in _kwargs(call), f"loaders.py rebuilds {cls} without {KEY}"


def test_inference_gates_generation_on_the_same_attribute():
    """Generation must consult the model attribute (one helper, both branches) instead of
    assuming normal RoPE at decode."""
    src = INFERENCE.read_text()
    assert "_identity_rope_kwargs" in src
    assert src.count("_identity_rope_kwargs(") >= 3, \
        "expected one definition plus the mask and additive call sites"
    assert f'getattr(graph_model, "_{KEY}", False)' in src


def test_e14_script_exposes_the_flag_as_an_env_override():
    """The sbatch must default the variable AND pass it through; a hard-coded literal
    would silently pin every rerun to one arm."""
    src = E14.read_text()
    var = KEY.upper()
    assert re.search(rf"^{var}=\$\{{{var}:-", src, re.M), \
        f"{var} is not env-overridable in {E14.name}"
    assert f'model.{KEY}="${var}"' in src, \
        f"{E14.name} does not forward {var} to the Hydra override"
    assert f"model.{KEY}=true" not in src and f"model.{KEY}=false" not in src, \
        "a hard-coded literal override would shadow the variable"
