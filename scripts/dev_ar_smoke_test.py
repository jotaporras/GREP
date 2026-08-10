"""Smoke test: autoregressive generation with a tiny HF model on local hardware.

Usage: conda run -n GREP-PRISM-v3 python scripts/dev_ar_smoke_test.py [--model MODEL]
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt", default="The shortest path between two nodes in a graph")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to(device)
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - start

    n_new = out.shape[1] - inputs["input_ids"].shape[1]
    print(tokenizer.decode(out[0], skip_special_tokens=True))
    print(f"\n{n_new} tokens in {elapsed:.2f}s ({n_new / elapsed:.1f} tok/s)")


if __name__ == "__main__":
    main()
