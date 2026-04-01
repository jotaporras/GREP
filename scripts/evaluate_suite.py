"""Batch evaluation wrapper — run evaluate.py logic across multiple eval graphs.

Loads the model once, then evaluates every eval JSON matching a glob pattern
(sorted alphabetically). Prints a summary table at the end.

Usage:
    python scripts/evaluate_suite.py <checkpoint> --pattern "data/eval/eval_graph_unique_*.json" [--four-bit] [--device 0]

Examples:
    python scripts/evaluate_suite.py outputs/my_checkpoint --pattern "data/eval/eval_graph_unique_*.json" --four-bit
    python scripts/evaluate_suite.py outputs/my_checkpoint --pattern "data/eval/eval_graph_*.json" --output results/
"""
import argparse
import glob
import json
import os
import sys
import time

_early_parser = argparse.ArgumentParser(add_help=False)
_early_parser.add_argument("--device", type=int, default=0)
_early_args, _ = _early_parser.parse_known_args()
if _early_args.device >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_early_args.device)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prism.eval.run_eval import EvalSample, eval_model
from prism.models import loaders


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a PRISM checkpoint on multiple eval graphs and produce a summary table."
    )
    parser.add_argument("checkpoint", help="Path to checkpoint directory.")
    parser.add_argument(
        "--pattern",
        required=True,
        help='Glob pattern for eval JSON files, e.g. "data/eval/eval_graph_unique_*.json".',
    )
    parser.add_argument(
        "--four-bit",
        action="store_true",
        default=False,
        help="Load the model with 4-bit quantization.",
    )
    parser.add_argument(
        "--text-edge-list",
        choices=["present", "none"],
        default=None,
        help=(
            "Whether the edge list is included in the LLM text prompt. "
            "For graph-augmented checkpoints this is read from gnn_config.json; "
            "for plain LLM checkpoints you must set this explicitly (default: present)."
        ),
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU index to load the model on (default: 0). Set to -1 for device_map='auto'.",
    )
    parser.add_argument(
        "--use-icl",
        type=str,
        choices=["true", "false"],
        default=None,
        help="Whether to include ICL examples in the SPINE prompt. Default: auto.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Directory to save per-trial JSON results. One file per eval graph.",
    )
    return parser.parse_args()


def load_eval_samples(eval_data_path: str):
    with open(eval_data_path) as f:
        eval_config = json.load(f)
    return [
        EvalSample(
            task=t["task"],
            answer=t["answer"],
            graph=eval_config["graph"],
            init_node=t["init_node"],
        )
        for t in eval_config["tasks"]
    ]


def load_model(args):
    checkpoint = args.checkpoint
    gnn_config_path = os.path.join(checkpoint, "gnn_config.json")

    print(f"Loading checkpoint: {checkpoint}")
    if os.path.exists(gnn_config_path):
        print("Detected graph-augmented architecture (gnn_config.json found).")
        with open(gnn_config_path) as f:
            gnn_cfg = json.load(f)
        text_edge_list = gnn_cfg.get("text_edge_list", "present")
        if args.text_edge_list is not None and args.text_edge_list != text_edge_list:
            print(
                f"WARNING: --text-edge-list={args.text_edge_list!r} overrides "
                f"checkpoint value {text_edge_list!r}"
            )
            text_edge_list = args.text_edge_list
        model, tokenizer = loaders.graph_augmented_llm_from_pretrained(
            path=checkpoint, load_in_4bit=args.four_bit, device=args.device
        )
    else:
        print("Detected plain LLM architecture.")
        text_edge_list = args.text_edge_list if args.text_edge_list is not None else "present"
        model, tokenizer = loaders.from_pretrained(
            path=checkpoint, load_in_4bit=args.four_bit, device=args.device
        )

    return model, tokenizer, text_edge_list


def print_summary_table(trial_results):
    """Print a formatted table summarising every trial."""
    name_width = max(len(r["name"]) for r in trial_results)
    name_width = max(name_width, len("Eval File"))

    header = (
        f"{'Eval File':<{name_width}}  "
        f"{'Tasks':>5}  "
        f"{'Correct':>7}  "
        f"{'Acc':>7}  "
        f"{'Formatted':>9}  "
        f"{'Keyword':>7}  "
        f"{'Errors':>6}  "
        f"{'Time (s)':>8}"
    )
    sep = "-" * len(header)

    print(f"\n{sep}")
    print("EVALUATION SUITE SUMMARY")
    print(sep)
    print(header)
    print(sep)

    total_tasks = 0
    total_correct = 0
    total_formatted = 0
    total_keyword = 0
    total_errors = 0
    total_time = 0.0

    for r in trial_results:
        n = r["num_total"]
        c = r["num_correct"]
        fmt = r["num_formatted"]
        kw = r["num_keyword"]
        errs = r["num_errors"]
        t = r["elapsed_s"]
        acc = r["accuracy"]

        total_tasks += n
        total_correct += c
        total_formatted += fmt
        total_keyword += kw
        total_errors += errs
        total_time += t

        print(
            f"{r['name']:<{name_width}}  "
            f"{n:>5}  "
            f"{c:>7}  "
            f"{acc:>7.1%}  "
            f"{fmt:>9}  "
            f"{kw:>7}  "
            f"{errs:>6}  "
            f"{t:>8.1f}"
        )

    print(sep)
    overall_acc = total_correct / total_tasks if total_tasks else 0.0
    print(
        f"{'TOTAL':<{name_width}}  "
        f"{total_tasks:>5}  "
        f"{total_correct:>7}  "
        f"{overall_acc:>7.1%}  "
        f"{total_formatted:>9}  "
        f"{total_keyword:>7}  "
        f"{total_errors:>6}  "
        f"{total_time:>8.1f}"
    )
    print(sep)


def main():
    args = parse_args()

    eval_files = sorted(glob.glob(args.pattern))
    if not eval_files:
        print(f"No files matched pattern: {args.pattern}")
        sys.exit(1)

    print(f"Found {len(eval_files)} eval files matching '{args.pattern}':")
    for f in eval_files:
        print(f"  {f}")
    print()

    model, tokenizer, text_edge_list = load_model(args)
    print(f"text_edge_list={text_edge_list!r}\n")

    use_icl = {"true": True, "false": False}.get(args.use_icl)

    if args.output:
        os.makedirs(args.output, exist_ok=True)

    trial_results = []

    for eval_path in eval_files:
        trial_name = os.path.basename(eval_path)
        print(f"\n{'='*60}")
        print(f"  TRIAL: {trial_name}")
        print(f"{'='*60}")

        eval_samples = load_eval_samples(eval_path)
        print(f"  {len(eval_samples)} samples")

        t0 = time.time()
        accuracy, sample_results = eval_model(
            eval_samples=eval_samples,
            model=model,
            tokenizer=tokenizer,
            text_edge_list=text_edge_list,
            use_icl=use_icl,
        )
        elapsed = time.time() - t0

        num_correct = sum(r["correct"] for r in sample_results)
        num_formatted = sum(r["formatted"] for r in sample_results)
        num_keyword = sum(r["plan_keyword"] for r in sample_results)
        num_errors = sum(1 for r in sample_results if r["error"] is not None)

        print(f"\n  => {trial_name}: {num_correct}/{len(sample_results)} correct ({accuracy:.1%}) in {elapsed:.1f}s")

        trial_record = {
            "name": trial_name,
            "path": eval_path,
            "num_total": len(sample_results),
            "num_correct": num_correct,
            "accuracy": accuracy,
            "num_formatted": num_formatted,
            "num_keyword": num_keyword,
            "num_errors": num_errors,
            "elapsed_s": elapsed,
            "samples": sample_results,
        }
        trial_results.append(trial_record)

        if args.output:
            out_path = os.path.join(args.output, trial_name.replace(".json", "_results.json"))
            with open(out_path, "w") as f:
                json.dump(trial_record, f, indent=2)
            print(f"  Saved: {out_path}")

    print_summary_table(trial_results)

    if args.output:
        summary_path = os.path.join(args.output, "summary.json")
        summary = {
            "checkpoint": args.checkpoint,
            "pattern": args.pattern,
            "trials": [
                {k: v for k, v in r.items() if k != "samples"}
                for r in trial_results
            ],
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
