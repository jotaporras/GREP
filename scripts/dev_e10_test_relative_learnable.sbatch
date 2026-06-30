#!/bin/bash
#SBATCH --job-name=e10-relative-pe
#SBATCH --output=/vast/home/j/jporras/sourcecode/GREP/slurm-%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G

# e10 relative-PE comparison. Runs TWO arches back-to-back under identical data/trainer
# conditions and the SAME adjacency A, so they are directly comparable:
#
#   1. graph_mask_llm      -- CORRECTED parameter-free adjacency mask (re-run of the prior
#                             baseline now that the SDPA boolean-mask causal-leak is fixed;
#                             prior e9 graph_mask runs trained with that leak).
#   2. learnable_graph_mask -- NEW learnable relative-PE mask: M = alpha*A + (1-alpha)*sim(Psi Psi^T),
#                             Psi from a standalone GraphTransformer; injected into Gemma's
#                             global/full-attention ("dense") layers only.
#
# Both: no_edge_list (connectivity must come from the mask), from scratch (not multistage).

BASE_CONFIG=e9_base_config
MODEL_PATH="google/gemma-4-12B-it"
CHECKPOINT_DIR="$ALELAB_DRIVE/GREP-PRISM/outputs/dev_e10/relative_pe"

# Loading env stuff
module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3

# Shared overrides: same data + trainer + experiment tag, and the SAME adjacency A
# (mask_k_hops/symmetrize/use_edges) so the baseline and the learnable mask differ ONLY
# in whether the edge bias is learned.
COMMON=(
    model.path="$MODEL_PATH"
    wandb.tag=e10_relative_pe
    trainer.checkpoint_dir="$CHECKPOINT_DIR"
    trainer.freeze_llm=false
    trainer.freeze_pe=false
    data.text_edge_list=none
    trainer.learning_rate=0.00025
    trainer.epochs=3
    gnn.mask_k_hops=1
    gnn.mask_symmetrize=true
    gnn.mask_use_edges=true
)

# --- Run 1: corrected parameter-free graph-mask baseline ---
srun python -m prism.training.train_v3 \
    --config-name="$BASE_CONFIG" \
    "${COMMON[@]}" \
    gnn.arch=graph_mask_llm \
    trainer.save_name="e10_relative_pe__graph_mask_baseline" \
    wandb.run_name="e10_relative_pe__graph_mask_baseline"

# --- Run 2: new learnable relative-PE mask (standalone GT producing Psi; dense layers) ---
srun python -m prism.training.train_v3 \
    --config-name="$BASE_CONFIG" \
    "${COMMON[@]}" \
    gnn.arch=learnable_graph_mask \
    gnn.d_model=1024 \
    gnn.dropout=0.1 \
    gnn.pe_hidden_channels=256 \
    gnn.pe_num_layers=2 \
    gnn.num_samples=40 \
    gnn.k_pe=2 \
    gnn.gt_num_layers=2 \
    gnn.gt_heads=8 \
    gnn.k_gt=2 \
    gnn.eps=1e-6 \
    gnn.use_layer_norm=true \
    gnn.pe_node_features=random \
    gnn.mask_alpha=0.7 \
    gnn.mask_layer_scope=dense \
    gnn.mask_psi_scale=cosine \
    gnn.structural_lr_mult=5.0 \
    trainer.save_name="e10_relative_pe__learnable" \
    wandb.run_name="e10_relative_pe__learnable"
