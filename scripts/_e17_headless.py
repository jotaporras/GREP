"""Make the nbconvert output of the e17 notebook runnable under plain `python`.

Applied by ``scripts/e17_pipeline.sh`` to the freshly converted script, so the notebook
stays the single source and the executable never drifts from it. Three edits:

  1. ``get_ipython()`` is undefined outside a kernel — the autoreload magic is a notebook
     convenience, so it is guarded rather than deleted (same fix e9_gnn_navigation.py
     carries in-tree).
  2. ``display`` is a kernel builtin; under `python` it becomes a print.
  3. ``suite``, the two epoch counts and the probe count become env-overridable, so a
     smoke run is `E17_EDGE_EPOCHS=1 ...` rather than an edited notebook.

Run: python scripts/_e17_headless.py scripts/e17_magnet_composite_graphs.py
"""
import re
import sys

HEADER = '''# --- headless shims (scripts/_e17_headless.py) -------------------------------------
import os as _os


def get_ipython():                      # defined only inside a kernel
    return None


def display(*args, **kwargs):           # a kernel builtin; print outside one
    for a in args:
        print(a)


def _env(name, default):
    """Env override for a notebook literal, so a smoke run needs no edit."""
    return type(default)(_os.environ.get(name, default))
# ----------------------------------------------------------------------------------
'''

SUBS = [
    # The magics: nbconvert emits get_ipython().run_line_magic(...); make them no-ops.
    (re.compile(r"^get_ipython\(\)\.run_line_magic\((.*)\)$", re.M),
     r"_ = None  # magic: \1"),
    # Env-overridable knobs.
    (re.compile(r"^suite     = 'suite2'$", re.M),
     "suite     = _env('E17_SUITE', 'suite2')"),
    (re.compile(r"^epochs = 150$", re.M), "epochs = _env('E17_EDGE_EPOCHS', 150)"),
    (re.compile(r"^epochs = 50$", re.M), "epochs = _env('E17_RES_EPOCHS', 50)"),
    (re.compile(r"^        num_samples=320,$", re.M),
     "        num_samples=_env('E17_NUM_SAMPLES', 320),"),
    (re.compile(r"^plans_per_graph = 10$", re.M),
     "plans_per_graph = _env('E17_PLANS_PER_GRAPH', 10)"),
    (re.compile(r"^train_samples, val_samples, test_samples = 100, 20, 20$", re.M),
     "train_samples, val_samples, test_samples = (_env('E17_TRAIN_SAMPLES', 100),\n"
     "                                            _env('E17_VAL_SAMPLES', 20),\n"
     "                                            _env('E17_TEST_SAMPLES', 20))"),
]


def main(path: str) -> None:
    src = open(path).read()
    if "headless shims" in src:
        print(f"[headless] {path} already patched")
        return
    applied = []
    for pattern, repl in SUBS:
        src, n = pattern.subn(repl, src)
        applied.append(n)
    # The shims must precede every use, and nbconvert puts `#!/usr/bin/env python` first.
    lines = src.split("\n")
    cut = 1 if lines and lines[0].startswith("#!") else 0
    src = "\n".join(lines[:cut]) + ("\n" if cut else "") + HEADER + "\n".join(lines[cut:])
    open(path, "w").write(src)
    print(f"[headless] {path}: magics={applied[0]} knobs={sum(applied[1:])}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "scripts/e17_magnet_composite_graphs.py")
