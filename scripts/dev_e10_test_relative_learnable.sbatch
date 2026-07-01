#!/bin/bash
#SBATCH --job-name=e10-relative-pe
#SBATCH --output=/vast/home/j/jporras/sourcecode/GREP/slurm-%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#SBATCH --array=1-4

# e10 relative-PE comparison as a SLURM job ARRAY (one arm per array index), all under
# identical data/trainer conditions and the SAME adjacency A, so they are directly comparable:
#
#   1  graph_mask_fixed   graph_mask_llm, ALL layers, CORRECT masking (mask_buggy_causal_fold=false).
#   2  graph_mask_buggy   graph_mask_llm, ALL layers, PRE-FIX LEAK (mask_buggy_causal_fold=true) —
#                         near-bidirectional attention via the sliding layers; the masking the
#                         historical ~30% baseline actually trained/evaluated with.
#   3  adjacency_dense    graph_mask_llm, DENSE (global) layers only — pure adjacency, NOTHING
#                         learnable; the non-learnable control matching the learnable mask's scope.
#   4  learnable          learnable_graph_mask: M = alpha*A + (1-alpha)*sim(Psi Psi^T), dense layers.
#
# All: no_edge_list (connectivity from the mask), from scratch (not multistage).
#
#   Run all four:      sbatch scripts/dev_e10_test_relative_learnable.sbatch
#   Run a subset:      sbatch --array=2,3 scripts/dev_e10_test_relative_learnable.sbatch
#   Local single arm:  SLURM_ARRAY_TASK_ID=3 bash scripts/dev_e10_test_relative_learnable.sbatch

BASE_CONFIG=e9_base_config
MODEL_PATH="google/gemma-4-12B-it"
CHECKPOINT_DIR="$ALELAB_DRIVE/GREP-PRISM/outputs/dev_e10/relative_pe"

# Loading env stuff
module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3

# Shared overrides: same data + trainer + experiment tag, and the SAME adjacency A.
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
    # Eval on the WHOLE held-out set (all 10 graphs), not the num_graphs=5 default.
    eval.data=data/revised/gen/nav100_n30_gemma_data/split/test_graphs
    eval.num_graphs=-1
)

# GT / learnable-only args (arm 4).
GT_ARGS=(
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
    gnn.mask_alpha=0.7
    gnn.mask_psi_scale=cosine
    gnn.structural_lr_mult=5.0
)

case "$SLURM_ARRAY_TASK_ID" in
  1) NAME=graph_mask_fixed
     ARM=(gnn.arch=graph_mask_llm  gnn.mask_layer_scope=all   gnn.mask_buggy_causal_fold=false) ;;
  2) NAME=graph_mask_buggy
     ARM=(gnn.arch=graph_mask_llm  gnn.mask_layer_scope=all   gnn.mask_buggy_causal_fold=true) ;;
  3) NAME=adjacency_dense
     ARM=(gnn.arch=graph_mask_llm  gnn.mask_layer_scope=dense gnn.mask_buggy_causal_fold=false) ;;
  4) NAME=learnable
     ARM=(gnn.arch=learnable_graph_mask gnn.mask_layer_scope=dense gnn.mask_buggy_causal_fold=false "${GT_ARGS[@]}") ;;
  *) echo "Unknown SLURM_ARRAY_TASK_ID='$SLURM_ARRAY_TASK_ID' (expected 1-4)"; exit 1 ;;
esac

srun python -m prism.training.train_v3 \
    --config-name="$BASE_CONFIG" \
    "${COMMON[@]}" \
    "${ARM[@]}" \
    trainer.save_name="e10_relative_pe__${NAME}" \
    wandb.run_name="e10_relative_pe__${NAME}"
