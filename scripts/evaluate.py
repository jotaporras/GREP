"""Standalone evaluation script for PRISM checkpoints.

Usage:
    python scripts/evaluate.py <checkpoint_path> [--four-bit] [--eval-data PATH] [--output PATH]

Examples:
    python scripts/evaluate.py outputs/e2_rpearl_improvements/e2_llm_llama-3.1-8b_r16_4bit_1x0xqg4q --four-bit
    python scripts/evaluate.py outputs/... --four-bit --eval-data data/eval/eval_1_multi_step.json --output results.json
"""
import argparse
import json
import os
import sys

# Allow running from the repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prism.eval.run_eval import EvalSample, eval_model
from prism.models import loaders


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a PRISM checkpoint on planning tasks.")
    parser.add_argument("checkpoint", help="Path to checkpoint directory.")
    parser.add_argument(
        "--four-bit",
        action="store_true",
        default=False,
        help="Load the model with 4-bit quantization.",
    )
    parser.add_argument(
        "--eval-data",
        default="data/eval/eval_1_multi_step.json",
        help="Path to eval JSON file (default: data/eval/eval_1_multi_step.json).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save per-sample JSON results.",
    )
    parser.add_argument(
        "--text-edge-list",
        choices=["present", "none"],
        default=None,
        help=(
            "Whether the edge list is included in the LLM text prompt. "
            "For graph-augmented checkpoints this is read automatically from gnn_config.json; "
            "for plain LLM checkpoints you must set this explicitly (default: present)."
        ),
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


def main():
    args = parse_args()

    checkpoint = args.checkpoint
    gnn_config_path = os.path.join(checkpoint, "gnn_config.json")

    print(f"Loading checkpoint: {checkpoint}")
    if os.path.exists(gnn_config_path):
        print("Detected graph-augmented architecture (gnn_config.json found).")
        with open(gnn_config_path) as f:
            gnn_cfg = json.load(f)
        # text_edge_list is written by train_v2.py; fall back to "present" for
        # older checkpoints that predate this key.
        text_edge_list = gnn_cfg.get("text_edge_list", "present")
        if args.text_edge_list is not None and args.text_edge_list != text_edge_list:
            print(
                f"WARNING: --text-edge-list={args.text_edge_list!r} overrides "
                f"checkpoint value {text_edge_list!r}"
            )
            text_edge_list = args.text_edge_list
        model, tokenizer = loaders.graph_augmented_llm_from_pretrained(
            path=checkpoint, load_in_4bit=args.four_bit
        )
    else:
        print("Detected plain LLM architecture.")
        text_edge_list = args.text_edge_list if args.text_edge_list is not None else "present"
        model, tokenizer = loaders.from_pretrained(
            path=checkpoint, load_in_4bit=args.four_bit
        )

    print(f"text_edge_list={text_edge_list!r}")
    print(f"Loading eval data: {args.eval_data}")
    eval_samples = load_eval_samples(args.eval_data)
    print(f"Running eval on {len(eval_samples)} samples...")

    accuracy, sample_results = eval_model(
        eval_samples=eval_samples, model=model, tokenizer=tokenizer,
        text_edge_list=text_edge_list,
    )

    print(f"\n{'='*60}")
    print(f"Accuracy: {accuracy:.4f} ({sum(r['correct'] for r in sample_results)}/{len(sample_results)})")
    print(f"{'='*60}\n")

    if args.output:
        output_data = {
            "checkpoint": checkpoint,
            "eval_data": args.eval_data,
            "accuracy": accuracy,
            "num_correct": sum(r["correct"] for r in sample_results),
            "num_total": len(sample_results),
            "samples": sample_results,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
