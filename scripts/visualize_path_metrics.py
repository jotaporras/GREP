#!/usr/bin/env python3
"""
CLI for the PRISM path-metrics figure.

Thin wrapper over :mod:`prism.eval.render` — all rendering logic lives in the
package so the eval drivers (``prism.eval.evaluate`` / ``scalability_evaluation``)
can render in-process from live samples, while this script renders a run's already
exported per-graph ``<graph>.json`` files from disk.

Usage:
    python scripts/visualize_path_metrics.py [RESULTS_DIR ...]

With no arguments every run under results/models is visualized and the figures
are written to results/e7_architecture_experiments.
"""

import sys
from pathlib import Path

# Make ``src/`` importable when run as a loose script (not an installed package).
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from prism.eval import render

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "results" / "models"
OUTPUT_DIR = PROJECT_ROOT / "results" / "e7_architecture_experiments"


def main():
    if sys.argv[1:]:
        dirs = [Path(a) for a in sys.argv[1:]]
    else:
        dirs = sorted(d for d in MODELS_DIR.iterdir() if d.is_dir())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for results_dir in dirs:
        if not results_dir.exists():
            print(f"ERROR: {results_dir} not found; skipping.")
            continue
        print(f"Processing {results_dir} ...")
        render.render_dir(results_dir, OUTPUT_DIR)


if __name__ == "__main__":
    main()
