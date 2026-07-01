"""Injection-ablation diagnostic (Step 1 of the no-edge-list investigation).

For a trained checkpoint, run teacher-forced forwards over held-out conversations
under different injection conditions and grade next-token prediction on the answer's
node mentions, split into decision / completion / repeat positions
(``prism.eval.injection_diag``). Conditions:

  train_style  — injection map over the FULL sequence: exactly what training saw
                 (answer-side node mentions carry the graph channel).
  prompt_only  — map clamped at answer_start: exactly what generation sees (only
                 prompt mentions carry the channel; decode steps receive nothing).
  no_injection — empty map (additive/mask archs) or graphs=None (postfusion/llm):
                 plain causal LLM.

How to read the decision-token row:
  train_style >> prompt_only ≈ no_injection  → a label-side (leak) channel was
      learned; it cannot exist at decode, explaining generation collapse.
  train_style ≈ prompt_only ≈ no_injection   → the graph channel was never engaged.
  prompt_only >> no_injection                → an eval-compatible readout exists;
      generation/search is the bottleneck, not the channel.

Usage (one checkpoint per invocation; see scripts/diag_injection_ablation.sbatch):
    python scripts/diag_injection_ablation.py \
        --checkpoint $ALELAB_DRIVE/GREP-PRISM/outputs/refactor_verify_fulltest/rv_gmask_noedges_ror8gtet \
        --val-file data/revised/gen/nav100_n30_gemma_data/split/formatted_all_new__val.json
"""
import argparse
import json
import os

import datasets
import torch
from torch_geometric.data import Batch

from prism.data import data as data_mod
from prism.data import utils as data_utils
from prism.eval import checkpoint as ckpt_mod
from prism.eval import injection_diag
from prism.models import gnn_llm


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Trained run dir (gnn_config.json or adapter_config.json).")
    p.add_argument("--val-file", default="data/revised/gen/nav100_n30_gemma_data/split/formatted_all_new__val.json",
                   help="Held-out conversations JSON (same format as training data).")
    p.add_argument("--text-edge-list", default=None, choices=["present", "none"],
                   help="Override the train-time policy recorded in the checkpoint.")
    p.add_argument("--max-examples", type=int, default=-1, help="Cap examples; -1 = all.")
    p.add_argument("--device", type=int, default=0, help="GPU index; -1 = device_map auto.")
    p.add_argument("--four-bit", action="store_true", help="Load the base LLM in 4-bit.")
    p.add_argument("--seed", type=int, default=1234,
                   help="torch seed re-set before EVERY forward so stochastic PE probes "
                        "(R-PEARL) are identical across conditions of the same example.")
    p.add_argument("--out", default=None,
                   help="Output JSON path (default: <checkpoint>/injection_diag.json).")
    return p.parse_args()


ADDITIVE_ARCHS = ("rpearl_llm", "rpearl_gt_llm", "gt_llm")
MASK_ARCHS = ("graph_mask_llm", "learnable_graph_mask")


def build_conditions(arch, pyg_graph, full_map, prompt_map):
    """(name, graphs, injection_maps, zero_gate) per condition, per architecture.

    ``no_injection`` must be logit-identical to the stock LLM while keeping the same
    forward code path as the other conditions:
    - additive archs: full map + ``zero_gate`` (pe_gain temporarily set to 0, so
      Ψ·tanh(0)=0). An empty map would break gt_llm's word_embeddings feature prep,
      which requires every node to have a mention span.
    - mask archs: empty map → all-zero structural bias (no gate parameter exists).
    - postfusion ignores injection maps (every token cross-attends all nodes), so
      train_style and prompt_only coincide; contrast is graph on / graphs=None.
    """
    graphs = Batch.from_data_list([pyg_graph])
    if arch == "postfusion_graph_llm":
        return [("train_style", graphs, [full_map], False), ("no_injection", None, None, False)]
    if arch in ADDITIVE_ARCHS:
        return [
            ("train_style", graphs, [full_map], False),
            ("prompt_only", graphs, [prompt_map], False),
            ("no_injection", graphs, [full_map], True),
        ]
    if arch in MASK_ARCHS:
        return [
            ("train_style", graphs, [full_map], False),
            ("prompt_only", graphs, [prompt_map], False),
            ("no_injection", graphs, [{}], False),
        ]
    raise ValueError(f"diagnostic not wired for architecture {arch!r}")


def main():
    args = parse_args()
    is_gnn = ckpt_mod.is_gnn_checkpoint(args.checkpoint)
    if is_gnn:
        with open(os.path.join(args.checkpoint, "gnn_config.json")) as f:
            arch = json.load(f)["architecture"]
    else:
        arch = "llm"
    text_edge_list = ckpt_mod.resolve_text_edge_list(args.checkpoint, is_gnn, args.text_edge_list)
    print(f"[diag] checkpoint={args.checkpoint}")
    print(f"[diag] arch={arch} text_edge_list={text_edge_list} val_file={args.val_file}")

    model, tokenizer, _ = ckpt_mod.load_checkpoint(args.checkpoint, four_bit=args.four_bit, device=args.device)
    device = next(model.parameters()).device

    ds = datasets.load_dataset("json", data_files=[args.val_file], split="train")
    ds = data_mod.preprocess_dataset(ds, tokenizer, architecture=arch, text_edge_list=text_edge_list)
    if args.max_examples > 0:
        ds = ds.select(range(min(args.max_examples, len(ds))))

    condition_names = ["no_injection"] if arch == "llm" else (
        ["train_style", "no_injection"] if arch == "postfusion_graph_llm"
        else ["train_style", "prompt_only", "no_injection"])
    totals = {
        cond: {s: {"n": 0, "correct": 0, "nll_sum": 0.0} for s in injection_diag.POSITION_SETS}
        for cond in condition_names
    }

    n_graded = 0
    for ex_idx, example in enumerate(ds):
        input_ids = torch.tensor([example["input_ids"]], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        answer_start = example["answer_start"]

        pyg_graph = data_utils.scene_graph_dict_to_pyg(example["scene_graph_dict"])
        node_token_seqs = gnn_llm.node_token_variants(pyg_graph.node_names, tokenizer)
        scope_start = gnn_llm.find_last_graph_scope(example["input_ids"], tokenizer)
        full_map = gnn_llm.build_injection_map(
            example["input_ids"], node_token_seqs, scope_start=scope_start)
        prompt_map = gnn_llm.clamp_injection_map(full_map, answer_start)
        position_sets = injection_diag.partition_answer_node_positions(full_map, answer_start)
        if not position_sets["decision"]:
            print(f"[diag] example {ex_idx}: no answer-side node mentions, skipped")
            continue

        if arch == "llm":
            conditions = [("no_injection", None, None, False)]
        else:
            conditions = build_conditions(arch, pyg_graph, full_map, prompt_map)

        for cond, graphs, maps, zero_gate in conditions:
            if zero_gate:
                saved_gain = model.pe_gain.data.clone()
                model.pe_gain.data.zero_()
            torch.manual_seed(args.seed)  # identical PE probes across conditions
            with torch.no_grad():
                if arch == "llm":
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                    graphs=graphs, injection_maps=maps)
            if zero_gate:
                model.pe_gain.data.copy_(saved_gain)
            for set_name, positions in position_sets.items():
                part = injection_diag.grade_positions(outputs.logits, input_ids, positions)
                injection_diag.merge_counts(totals[cond][set_name], part)
        n_graded += 1
        if (ex_idx + 1) % 10 == 0:
            print(f"[diag] {ex_idx + 1}/{len(ds)} examples done")

    results = {
        "checkpoint": args.checkpoint,
        "architecture": arch,
        "text_edge_list": text_edge_list,
        "val_file": args.val_file,
        "seed": args.seed,
        "n_examples_graded": n_graded,
        "conditions": {
            cond: {s: injection_diag.summarize(c) for s, c in sets.items()}
            for cond, sets in totals.items()
        },
    }

    print(f"\n[diag] === {arch} ({os.path.basename(args.checkpoint.rstrip('/'))}), "
          f"{n_graded} examples ===")
    print(f"{'condition':<14}" + "".join(
        f"{s + '.acc':>22}{s + '.nll':>18}" for s in injection_diag.POSITION_SETS))
    for cond in condition_names:
        row = f"{cond:<14}"
        for s in injection_diag.POSITION_SETS:
            m = results["conditions"][cond][s]
            row += f"{m['acc']:>14.4f} (n={m['n']})".rjust(22) + f"{m['mean_nll']:>18.3f}"
        print(row)

    out_path = args.out or os.path.join(args.checkpoint, "injection_diag.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[diag] wrote {out_path}")


if __name__ == "__main__":
    main()
