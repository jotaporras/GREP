"""Build a stage-3 PE-init folder from the pretrained edge-detector GNN.

The edge detector (notebooks/e9_gnn_navigation.ipynb) is saved as a *flat*
``GraphTransformer.state_dict()`` (its MLP edge-classifier head is NOT in the
file). ``trainer.init_pe_from`` does not consume that directly: it reads
``<dir>/gnn_weights.pt`` and expects a dict ``{"gt_model", "pe_proj", ...}``
(see prism.models.loaders.load_pe_weights_into).

This script rebuilds the GraphTransformer at the *notebook* hyperparameters,
strict-loads the edge-detector weights into it (a loud failure if any
shape-bearing hparam drifts, instead of the silent strict=False drop the
trainer would do), attaches a freshly-initialised ``pe_proj`` sized to the base
LLM's text hidden size, and saves the pair as ``gnn_weights.pt``.

Omitted on purpose: ``pe_gain`` / ``pe_norm`` are left out of the dict so the
training model keeps its config init (cold-start gate pe_gain_init, fresh
pe_norm) -- the loader treats both as optional.
"""

import argparse
import os

import torch
from torch import nn
from transformers import AutoConfig

from prism.models import gt as gt_module

# GraphTransformer hyperparameters that produced edge_detector_gt_final.pt.
# Authoritative source: the GNN-instantiation cell in e9_gnn_navigation.ipynb.
# These MUST reproduce the checkpoint's architecture exactly; the strict load
# below is the assertion that they do.
GT_HPARAMS = dict(
    num_layers=3,           # -> gnn.gt_num_layers
    pe_hidden_channels=256,
    pe_num_layers=5,
    d_model=1024,
    heads=8,                # -> gnn.gt_heads
    num_samples=320,
    dropout=0.1,
    k_pe=3,
    k_gt=2,                 # -> gnn.k_gt
    eps=1e-6,
    use_layer_norm=True,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gnn-checkpoint", required=True,
                        help="Flat GraphTransformer state_dict (edge_detector_gt_final.pt).")
    parser.add_argument("--model-path", required=True,
                        help="Base LLM whose text hidden_size sizes pe_proj (e.g. google/gemma-4-31B-it).")
    parser.add_argument("--out-dir", required=True,
                        help="Destination PE-init dir; gnn_weights.pt is written inside it.")
    args = parser.parse_args()  

    # Rebuild the GT and strict-load the detector weights. node_feature_dim is
    # left at its None default to match the notebook (gnn.pe_node_features=random
    # at train time -> node_feature_dim=None).
    gt = gt_module.GraphTransformer(**GT_HPARAMS)
    state = torch.load(args.gnn_checkpoint, map_location="cpu")
    gt.load_state_dict(state, strict=True)  # raises on any arch mismatch

    # Fresh pe_proj: nn.Linear(d_model, text_hidden) with bias, mirroring
    # GraphAugmentedLLM.__init__. Random init here == a fresh init at train time;
    # it exists only to satisfy the loader's strict pe_proj load and is trained
    # in stage 3.
    hidden = AutoConfig.from_pretrained(args.model_path).get_text_config().hidden_size
    pe_proj = nn.Linear(GT_HPARAMS["d_model"], hidden)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "gnn_weights.pt")
    torch.save({"gt_model": gt.state_dict(), "pe_proj": pe_proj.state_dict()}, out_path)
    print(f"[build] wrote {out_path} (pe_proj: {GT_HPARAMS['d_model']} -> {hidden})")


if __name__ == "__main__":
    main()
