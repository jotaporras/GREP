"""Run PRISM planning eval on a trained checkpoint against arbitrary test-graph files.

Built to evaluate the e3 models on the larger e4 test graphs, but works for any
checkpoint + any eval-graph JSON in the `{"graph": {...}, "tasks": [...]}` format.

Usage:
    python scripts/eval_checkpoint_on_graphs.py \
        --checkpoint outputs/e3_new_training_data/e3_llm_llama-3.1-8b_r16_4bit_0sy9j5rz \
        --graphs data/training_data_20260428/aggregate_20260428/split_20260428/test_graphs \
        [--out <dir>] [--text-edge-list present|none] [--no-4bit] [--device 0]

`--checkpoint` may point at a run dir (top-level final adapter) or any subdir
containing `adapter_config.json` (plain LLM) or `gnn_config.json` (graph-augmented).
`--graphs` may be a single JSON file or a directory of them.

Writes one results JSON per graph file to `--out` (default:
`<checkpoint>/eval_logs/cross_eval/`) and prints an accuracy summary table.
"""
import argparse
import json
import os
import sys
from glob import glob

# Repo `src/` layout — make `prism` importable when run from anywhere.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prism.eval import run_eval
from prism.models import loaders


def _is_gnn_checkpoint(path: str) -> bool:
    return os.path.exists(os.path.join(path, "gnn_config.json"))


def _load_checkpoint(path: str, four_bit: bool, device: int):
    """Load either a graph-augmented or plain-LLM checkpoint. Returns (model, tokenizer, is_gnn)."""
    if _is_gnn_checkpoint(path):
        model, tok = loaders.graph_augmented_llm_from_pretrained(
            path, load_in_4bit=four_bit, device=device
        )
        return model, tok, True
    if not os.path.exists(os.path.join(path, "adapter_config.json")):
        raise FileNotFoundError(
            f"{path} has neither gnn_config.json nor adapter_config.json — "
            "not a recognizable checkpoint dir."
        )
    model, tok = loaders.from_pretrained(path, load_in_4bit=four_bit, device=device)
    return model, tok, False


def _load_graph_samples(graph_file: str):
    """Build EvalSamples from a `{"graph": ..., "tasks": [...]}` file (same format as
    train_v2._load_eval_samples)."""
    with open(graph_file) as f:
        data = json.load(f)
    graph_data = data["graph"]
    return [
        run_eval.EvalSample(
            task=entry["task"],
            answer=entry["answer"],
            graph=graph_data,
            init_node=entry["init_node"],
        )
        for entry in data["tasks"]
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True,
                    help="Checkpoint dir (run dir or subdir with adapter_config.json / gnn_config.json).")
    ap.add_argument("--graphs", required=True,
                    help="A test-graph JSON file or a directory of them.")
    ap.add_argument("--out", default=None,
                    help="Output dir for results JSON (default: <checkpoint>/eval_logs/cross_eval).")
    ap.add_argument("--text-edge-list", choices=["present", "none"], default=None,
                    help="Whether the textual edge list is shown to the planner. "
                         "Default: 'none' for graph-augmented checkpoints, 'present' otherwise "
                         "(matches the e3 training configs).")
    ap.add_argument("--no-4bit", action="store_true",
                    help="Disable 4-bit quantization (e3/e4 checkpoints were trained with bit4: true).")
    ap.add_argument("--device", type=int, default=-1,
                    help="Physical GPU index; -1 uses device_map='auto'.")
    args = ap.parse_args()

    ckpt = os.path.abspath(args.checkpoint.rstrip("/"))

    # Resolve graph files.
    if os.path.isdir(args.graphs):
        graph_files = sorted(glob(os.path.join(args.graphs, "*.json")))
    else:
        graph_files = [args.graphs]
    if not graph_files:
        ap.error(f"No graph JSON files found at {args.graphs}")

    out_dir = args.out or os.path.join(ckpt, "eval_logs", "cross_eval")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading checkpoint: {ckpt}")
    model, tokenizer, is_gnn = _load_checkpoint(ckpt, four_bit=not args.no_4bit, device=args.device)
    text_edge_list = args.text_edge_list or ("none" if is_gnn else "present")
    print(f"  architecture: {'graph-augmented' if is_gnn else 'plain LLM'}  |  "
          f"text_edge_list={text_edge_list}  |  4bit={not args.no_4bit}")
    print(f"  {len(graph_files)} graph file(s)  ->  {out_dir}\n")

    summary = []
    grand_correct = grand_total = 0

    for gf in graph_files:
        name = os.path.splitext(os.path.basename(gf))[0]
        samples = _load_graph_samples(gf)
        n = len(samples)
        print(f"== {name}: {n} tasks ==")

        sample_results = []
        graph_correct = 0

        for i, sample in enumerate(samples):
            _, results = run_eval.eval_model(
                model=model,
                tokenizer=tokenizer,
                eval_samples=[sample],
                text_edge_list=text_edge_list,
            )
            r = results[0]
            sample_results.append(r)

            if r["correct"]:
                graph_correct += 1
            graph_total = i + 1
            grand_correct_now = grand_correct + graph_correct
            grand_total_now = grand_total + graph_total

            status = "✓" if r["correct"] else "✗"
            graph_acc = graph_correct / graph_total
            overall_acc = grand_correct_now / grand_total_now
            print(
                f"  [{status}] {graph_total}/{n} this graph  "
                f"graph: {graph_acc:.0%} ({graph_correct}/{graph_total})  "
                f"overall: {overall_acc:.0%} ({grand_correct_now}/{grand_total_now})  "
                f"task: {sample.task[:50]}"
            )

        accuracy = graph_correct / n if n else 0.0
        grand_correct += graph_correct
        grand_total += n

        log_data = {
            "checkpoint": ckpt,
            "graph_file": gf,
            "architecture": "graph-augmented" if is_gnn else "llm",
            "text_edge_list": text_edge_list,
            "accuracy": accuracy,
            "num_samples": len(sample_results),
            "num_correct": graph_correct,
            "samples": sample_results,
        }
        out_file = os.path.join(out_dir, f"{name}.json")
        with open(out_file, "w") as f:
            json.dump(log_data, f, indent=2, default=str)
        print(f"  -> {accuracy:.1%} ({graph_correct}/{n})  saved: {out_file}\n")
        summary.append((name, graph_correct, n, accuracy))

    print("=" * 52)
    print(f"{'graph':<28}{'correct':>10}{'accuracy':>12}")
    print("-" * 52)
    tot_c = tot_n = 0
    for name, c, n, acc in summary:
        print(f"{name:<28}{f'{c}/{n}':>10}{acc:>11.1%}")
        tot_c += c
        tot_n += n
    print("-" * 52)
    print(f"{'TOTAL':<28}{f'{tot_c}/{tot_n}':>10}{(tot_c / tot_n if tot_n else 0):>11.1%}")
    print("=" * 52)


if __name__ == "__main__":
    main()
