"""Standalone evaluation script for PRISM checkpoints.

Saves a named report to results/<checkpoint_name>_<eval_name>.json.

Usage:
    python scripts/evaluate.py <checkpoint_path> [--four-bit] [--eval-data PATH] [--output DIR]

Examples:
    python scripts/evaluate.py outputs/e2_rpearl_improvements/e2_llm_llama-3.1-8b_r16_4bit_1x0xqg4q --four-bit
    python scripts/evaluate.py outputs/... --four-bit --eval-data data/eval/eval_1_multi_step.json --output results/
"""
import argparse
import json
import os
import sys

# Isolate the target GPU BEFORE any torch/CUDA imports.  Without this, PyTorch
# initializes a CUDA context on every visible GPU at import time, consuming
# memory and blocking other processes from using those GPUs.  Each evaluate.py
# process sets its own CUDA_VISIBLE_DEVICES, so parallel runs on different GPUs
# (e.g. --device 0 and --device 1 in separate terminals) coexist safely.
_early_parser = argparse.ArgumentParser(add_help=False)
_early_parser.add_argument("--device", type=int, default=0)
_early_args, _ = _early_parser.parse_known_args()
if _early_args.device >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_early_args.device)

# Allow running from the repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prism.eval.run_eval import EvalSample, eval_model
from prism.models import loaders
from prism.models.utils import Permutation


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
        default="results",
        help="Directory to save named result files (default: results/).",
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
        help=(
            "Whether to include ICL examples in the SPINE prompt. "
            "Default: auto (off for GNN models, on for plain LLMs)."
        ),
    )
    parser.add_argument(
        "--permutation-seed",
        type=int,
        nargs="+",
        default=None,
        help="One or more random seeds for node-index permutation (equivariance experiment). Omit to disable.",
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
            path=checkpoint, load_in_4bit=args.four_bit, device=args.device
        )
    else:
        print("Detected plain LLM architecture.")
        text_edge_list = args.text_edge_list if args.text_edge_list is not None else "present"
        model, tokenizer = loaders.from_pretrained(
            path=checkpoint, load_in_4bit=args.four_bit, device=args.device
        )

    print(f"text_edge_list={text_edge_list!r}")
    print(f"Loading eval data: {args.eval_data}")
    eval_samples = load_eval_samples(args.eval_data)
    print(f"Running eval on {len(eval_samples)} samples...")

    use_icl = {"true": True, "false": False}.get(args.use_icl)
    ckpt_name = os.path.basename(os.path.normpath(checkpoint))
    eval_name = os.path.splitext(os.path.basename(args.eval_data))[0]

    permutations = [Permutation(s) for s in args.permutation_seed] if args.permutation_seed else [None]

    for permutation in permutations:
        seed_tag = f"perm_{permutation.seed}" if permutation is not None else None
        if seed_tag:
            print(f"\n{'#'*60}")
            print(f"  PERMUTATION SEED: {permutation.seed}")
            print(f"{'#'*60}")

        accuracy, sample_results = eval_model(
            eval_samples=eval_samples, model=model, tokenizer=tokenizer,
            text_edge_list=text_edge_list, use_icl=use_icl,
            permutation=permutation,
        )

        print(f"\n{'='*60}")
        print(f"Accuracy: {accuracy:.4f} ({sum(r['correct'] for r in sample_results)}/{len(sample_results)})")
        if permutation is not None:
            print(f"Permutation: {permutation}")
        print(f"{'='*60}\n")

        out_dir = os.path.join(args.output, seed_tag) if seed_tag else args.output
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{ckpt_name}_{eval_name}.json")

        output_data = {
            "checkpoint": checkpoint,
            "eval_data": args.eval_data,
            "accuracy": accuracy,
            "num_correct": sum(r["correct"] for r in sample_results),
            "num_total": len(sample_results),
            "permutation": permutation.to_dict() if permutation is not None else None,
            "samples": sample_results,
        }
        with open(out_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
