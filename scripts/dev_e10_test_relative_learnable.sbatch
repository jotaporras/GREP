#!/bin/bash
#SBATCH --job-name=e10-alpha-sweep
#SBATCH --output=/vast/home/j/jporras/sourcecode/GREP/slurm-%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#SBATCH --array=1-6

# DISPOSABLE: mask_alpha sweep for the learnable relative-PE mask, WITH the edge list in the
# prompt (data.text_edge_list=present). Mask M = alpha*A + (1-alpha)*sim(Psi Psi^T):
# small alpha => learnable term dominates; alpha=1 => pure adjacency (learnable term vanishes).
#
#   1  alpha=0.01   learnable_graph_mask  (learnable term dominates)
#   2  alpha=0.05   learnable_graph_mask
#   3  alpha=0.10   learnable_graph_mask
#   4  alpha=0.50   learnable_graph_mask  (balanced)
#   5  alpha=0.90   learnable_graph_mask  (adjacency-dominant)
#   6  alpha=1.00   PURE ADJACENCY -> graph_mask_llm (dense). The learnable arch rejects alpha=1
#                   (it zeroes the GT gradient); this is the alpha->1 limit.
#
# All: DENSE (global) layers, cosine scaling, structural_lr_mult=5, edges IN TEXT, full 10-graph eval.
#
#   sbatch scripts/dev_e10_test_relative_learnable.sbatch      # all 6
#   sbatch --array=4 scripts/dev_e10_test_relative_learnable.sbatch   # just alpha=0.5

BASE_CONFIG=e9_base_config
MODEL_PATH="google/gemma-4-12B-it"
CHECKPOINT_DIR="$ALELAB_DRIVE/GREP-PRISM/outputs/dev_e10/alpha_sweep"

# Loading env stuff
module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3

# Shared overrides. Edges are now IN THE TEXT (text_edge_list=present); full held-out eval.
COMMON=(
    model.path="$MODEL_PATH"
    wandb.tag=e10_alpha_sweep
    trainer.checkpoint_dir="$CHECKPOINT_DIR"
    trainer.freeze_llm=false
    trainer.freeze_pe=false
    data.text_edge_list=present
    trainer.learning_rate=0.00025
    trainer.epochs=3
    gnn.mask_k_hops=1
    gnn.mask_symmetrize=true
    gnn.mask_use_edges=true
    gnn.mask_layer_scope=dense
    eval.data=data/revised/gen/nav100_n30_gemma_data/split/test_graphs
    eval.num_graphs=-1
)

# Learnable-mask GT args (arms 1-3). Arm 4 (pure adjacency) uses graph_mask_llm and ignores these.
GT_ARGS=(
    gnn.arch=learnable_graph_mask
    gnn.d_model=1024
    gnn.dropout=0.1
    gnn.pe_hidden_channels=256
    gnn.pe_num_layers=2
    gnn.num_samples=40
    gnn.k_pe=2
    gnn.gt_num_layers=2
    gnn.gt_heads=8
    gnn.k_gt=2
    gnn.eps=1e-6
    gnn.use_layer_norm=true
    gnn.pe_node_features=random
    gnn.mask_psi_scale=cosine
    gnn.structural_lr_mult=5.0
)

case "$SLURM_ARRAY_TASK_ID" in
  1) NAME=alpha_0p01; ARM=("${GT_ARGS[@]}" gnn.mask_alpha=0.01) ;;
  2) NAME=alpha_0p05; ARM=("${GT_ARGS[@]}" gnn.mask_alpha=0.05) ;;
  3) NAME=alpha_0p10; ARM=("${GT_ARGS[@]}" gnn.mask_alpha=0.1) ;;
  4) NAME=alpha_0p50; ARM=("${GT_ARGS[@]}" gnn.mask_alpha=0.5) ;;
  5) NAME=alpha_0p90; ARM=("${GT_ARGS[@]}" gnn.mask_alpha=0.9) ;;
  6) NAME=alpha_1p00; ARM=(gnn.arch=graph_mask_llm) ;;   # pure adjacency (dense scope via COMMON)
  *) echo "Unknown SLURM_ARRAY_TASK_ID='$SLURM_ARRAY_TASK_ID' (expected 1-6)"; exit 1 ;;
esac

srun python -m prism.training.train_v3 \
    --config-name="$BASE_CONFIG" \
    "${COMMON[@]}" \
    "${ARM[@]}" \
    trainer.save_name="e10_alpha_sweep__${NAME}" \
    wandb.run_name="e10_alpha_sweep__${NAME}"
