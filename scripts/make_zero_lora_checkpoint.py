"""Build an UNTRAINED plain-LLM checkpoint dir from a trained one: same metadata, LoRA B = 0.

A LoRA layer adds B @ A x; with every lora_B tensor zero it adds nothing, so the
checkpoint evaluates as the base model. It keeps the source's train_config.json
(text_edge_list, response_format, icl_examples, ...), so cross_eval uses exactly
the prompt contract of the trained arm. e22 uses it as the untrained text_edges
baseline (base gemma-4-31B-it, edge list in the prompt, route_only).

Usage (CPU job):
  python scripts/make_zero_lora_checkpoint.py --src <trained text ckpt> --dst <new dir>
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ADAPTER = "adapter_model.safetensors"
SKIP = {"optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json",
        "training_args.bin", "gnn_weights.pt", ADAPTER}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    args = ap.parse_args()

    tc = json.loads((args.src / "train_config.json").read_text())
    if "gnn" in tc and tc.get("architecture") not in (None, "llm"):
        raise SystemExit(f"{args.src} is a graph checkpoint ({tc.get('architecture')}); "
                         "zeroing its LoRA would not remove the graph channel")
    args.dst.mkdir(parents=True, exist_ok=False)
    for f in args.src.iterdir():
        if f.is_file() and f.name not in SKIP:
            shutil.copy2(f, args.dst / f.name)

    state = load_file(args.src / ADAPTER)
    n_b = 0
    for k, v in state.items():
        if ".lora_B." in k:
            state[k] = torch.zeros_like(v)
            n_b += 1
    if n_b == 0:
        raise SystemExit(f"no lora_B tensors in {args.src / ADAPTER}")
    save_file(state, args.dst / ADAPTER)
    (args.dst / "ZERO_LORA.txt").write_text(
        f"Untrained: copy of {args.src} with all {n_b} lora_B tensors set to 0 "
        f"(scripts/make_zero_lora_checkpoint.py). Evaluates as the base model.\n")
    print(f"wrote {args.dst}: {n_b} lora_B tensors zeroed, {len(state)} tensors total")


if __name__ == "__main__":
    main()
