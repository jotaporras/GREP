"""Populate a single eval-graph skeleton via the GPT-5.5 pipeline.

Wraps `prism.data.data_gen.DataGenerator.populate_graphs_and_tasks` so each
skeleton lands at a caller-chosen output path. `generate_data_spine.py` would
rename files to `data_gen_{idx:03d}.json` and erase the size suffix in the
skeleton filename (e.g. `eval_graph_unique_250.json`), which we need to keep
for the transferability eval pipeline.

Usage:
    python scripts/populate_eval_graph.py \
        --skeleton data/eval/e4_transferability_skeletons/eval_graph_unique_250.json \
        --output   data/eval/e4_transferability/eval_graph_unique_250.json
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prism.data.data_gen import DataGenerator


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skeleton", required=True, help="Path to a skeleton JSON from generate_eval_graphs.py.")
    ap.add_argument("--output", required=True, help="Where to write the populated eval JSON.")
    args = ap.parse_args()

    with open(args.skeleton) as f:
        base_graph = json.load(f)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    generator = DataGenerator(graph_unknown=[0])
    with tempfile.TemporaryDirectory() as tmpdir:
        generator.populate_graphs_and_tasks([json.dumps(base_graph)], log_dir=tmpdir)
        shutil.copy(Path(tmpdir) / "data_gen_000.json", args.output)

    print(f"Populated: {args.skeleton} -> {args.output}")


if __name__ == "__main__":
    main()
