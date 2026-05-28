"""Post-hoc log a scalability_evaluation results JSON to an existing wandb run.

Usage:
    python scripts/log_eval_to_wandb.py \
        --run-id qmu8x2qu \
        --results results/e3_rpearl_llm_llama-3.1-8b_r16_4bit_qmu8x2qu_eval_1_multi_step.json
"""
import argparse
import json
import os

from dotenv import load_dotenv

import wandb


def main():
    parser = argparse.ArgumentParser(description="Log a scalability_evaluation results JSON to an existing wandb run.")
    parser.add_argument("--run-id", required=True, help="Wandb run ID to resume (e.g. qmu8x2qu).")
    parser.add_argument("--results", required=True, help="Path to JSON file written by prism.eval.scalability_evaluation.")
    parser.add_argument("--project", default="GREP-PRISM", help="Wandb project (default: GREP-PRISM).")
    parser.add_argument("--epoch", type=float, default=None, help="Epoch tag to include with the log (optional).")
    args = parser.parse_args()

    load_dotenv()

    with open(args.results) as f:
        data = json.load(f)

    accuracy = data["accuracy"]
    num_correct = data["num_correct"]
    num_total = data["num_total"]

    wandb.init(id=args.run_id, resume="must", project=args.project)

    log_payload = {"eval/accuracy": accuracy}
    if args.epoch is not None:
        log_payload["epoch"] = args.epoch
    wandb.log(log_payload)

    wandb.save(os.path.abspath(args.results), base_path=os.path.dirname(os.path.abspath(args.results)))
    wandb.finish()

    print(f"Logged eval/accuracy={accuracy:.4f} ({num_correct}/{num_total}) to run {args.run_id}.")


if __name__ == "__main__":
    main()
