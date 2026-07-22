"""Capture attention matrices from a trained checkpoint for visualization.

Runs ONE teacher-forced forward over a held-out conversation (same preprocessing
as the injection diagnostic) and saves, for each requested layer, the attention
probabilities from that layer's self-attention — head-mean over the full
sequence plus per-head over the node-token submatrix — with enough metadata to
draw node-aligned views (tok2node, node names, answer_start, decoded tokens).

Mask checkpoints are captured under their NATIVE injection wiring (resolved from
train_config.json: full_sequence / prompt_only / decode_consistent). Attention
weights come from forward hooks on the target layers' self_attn modules; the
mask archs run eager attention by construction, and the plain-LLM path is forced
to eager here. Fails loudly if a hook sees no weights.

Usage:
    python scripts/capture_attention.py --checkpoint <run_dir> \
        --val-file data/revised/gen/nav100_n30_gemma_data/split/formatted_all_new__val.json \
        --example-idx 0 --layers 0,5 --out <path.pt>
"""
import argparse

import datasets
import torch
from torch_geometric.data import Batch

from prism.data import data as data_mod
from prism.data import utils as data_utils
from prism.eval import checkpoint as ckpt_mod
from prism.models import gnn_llm


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--val-file",
                   default="data/revised/gen/nav100_n30_gemma_data/split/formatted_all_new__val.json")
    p.add_argument("--example-idx", type=int, default=0)
    p.add_argument("--layers", default="0,5", help="comma-separated decoder layer indices")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    layer_ids = [int(x) for x in args.layers.split(",")]
    is_gnn = ckpt_mod.is_gnn_checkpoint(args.checkpoint)
    arch = ckpt_mod.load_gnn_config(args.checkpoint)["architecture"] if is_gnn else "llm"
    text_edge_list = ckpt_mod.resolve_text_edge_list(args.checkpoint, is_gnn, None)
    scope = ckpt_mod.resolve_injection_scope(args.checkpoint)
    print(f"[capture] arch={arch} text_edge_list={text_edge_list} scope={scope} layers={layer_ids}")

    model, tokenizer, _ = ckpt_mod.load_checkpoint(args.checkpoint, four_bit=False, device=args.device)
    device = next(model.parameters()).device
    llm = model.llm if hasattr(model, "llm") else model
    if arch == "llm":
        # Mask archs already run eager (set at wrap time); force it for the plain LLM
        # so the hooks below receive real attention probabilities.
        llm.config._attn_implementation = "eager"
        for layer in llm.model.layers:
            layer.self_attn.config._attn_implementation = "eager"

    ds = datasets.load_dataset("json", data_files=[args.val_file], split="train")
    ds = data_mod.preprocess_dataset(ds, tokenizer, text_edge_list=text_edge_list)
    example = ds[args.example_idx]
    input_ids = torch.tensor([example["input_ids"]], dtype=torch.long, device=device)
    answer_start = example["answer_start"]

    pyg_graph = data_utils.scene_graph_dict_to_pyg(example["scene_graph_dict"])
    node_token_seqs = gnn_llm.node_token_variants(pyg_graph.node_names, tokenizer)
    scope_start = gnn_llm.find_last_graph_scope(example["input_ids"], tokenizer)
    full_map = gnn_llm.build_injection_map(example["input_ids"], node_token_seqs,
                                           scope_start=scope_start)
    if scope == "prompt_only":
        q_map, k_map = gnn_llm.clamp_injection_map(full_map, answer_start), None
    elif scope == "decode_consistent":
        q_map = gnn_llm.decode_style_query_map(full_map, answer_start,
                                               example["input_ids"], node_token_seqs)
        k_map = full_map
    else:
        q_map, k_map = full_map, None

    captured = {}
    def hook(lid):
        def fn(module, h_args, h_kwargs, output):
            weights = output[1]
            if weights is None:
                raise RuntimeError(f"layer {lid}: attention weights are None (not eager?)")
            captured[lid] = weights.detach().float().cpu()  # [1, H, S, S]
        return fn
    handles = [llm.model.layers[lid].self_attn.register_forward_hook(hook(lid), with_kwargs=True)
               for lid in layer_ids]

    with torch.no_grad():
        if arch == "llm":
            model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
        elif k_map is not None:
            model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                  graphs=Batch.from_data_list([pyg_graph]), injection_maps=[q_map],
                  key_injection_maps=[k_map])
        else:
            model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                  graphs=Batch.from_data_list([pyg_graph]), injection_maps=[q_map])
    for h in handles:
        h.remove()

    S = input_ids.shape[1]
    tok2node = gnn_llm.tok2node_vector(full_map, S, "cpu")
    node_pos = (tok2node >= 0).nonzero(as_tuple=True)[0]
    layers_out = {}
    for lid, w in captured.items():
        w = w[0]                                   # [H, S, S]
        layers_out[lid] = {
            "head_mean": w.mean(0).to(torch.float16),                       # [S, S]
            "node_sub_per_head": w[:, node_pos][:, :, node_pos].to(torch.float16),
            "is_global": not bool(getattr(llm.model.layers[lid].self_attn, "is_sliding", False)),
        }

    torch.save({
        "checkpoint": args.checkpoint, "architecture": arch,
        "text_edge_list": text_edge_list, "injection_scope": scope,
        "example_idx": args.example_idx, "answer_start": answer_start,
        "input_ids": example["input_ids"],
        "tokens": [tokenizer.decode([t]) for t in example["input_ids"]],
        "tok2node": tok2node.tolist(), "node_pos": node_pos.tolist(),
        "node_names": list(pyg_graph.node_names),
        "edge_index": pyg_graph.edge_index.tolist(),
        "layers": layers_out,
    }, args.out)
    print(f"[capture] wrote {args.out} (S={S}, node tokens={len(node_pos)})")


if __name__ == "__main__":
    main()
