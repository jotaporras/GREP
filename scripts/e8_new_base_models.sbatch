#!/bin/bash
#SBATCH --job-name=e8-train-eval
#SBATCH --output=/vast/home/j/jporras/sourcecode/GREP/slurm-%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#SBATCH --array=0-4%4

# e8 new base models — train + cross-eval, one self-contained config per task.
#
# === MODIFIED RERUN (2026-06-19) ===========================================
# Architecture refactor: removed the spectral-norm / Lipschitz-norm / β=1/F
# filter-norm machinery and the ad-hoc magnitude-matching rescales from the
# R-PEARL / GT / injection path (norms are now plain LayerNorm; PE strength is
# set by learnable tanh gates only). This rerun covers the GRAPH archs only —
# idx 1-4 (rpearl_llm{,_no_edges}, rpearl_gt_llm{,_no_edges}); the plain LLM
# baseline (idx 0) is unaffected by the refactor and is not rerun.
#   Submit:  sbatch --array=1-4 scripts/e8_new_base_models.sbatch
# ===========================================================================
#
# Five Gemma-base configs under experiments/e8_new_base_models/. Each yaml is
# fully self-contained (data, base_model, hyperparams); this script only selects
# the config, assigns a UNIQUE save_name / wandb run name, and hands off to
# scripts/train_and_eval.sh for the train -> locate-checkpoint -> eval pipeline.
#
#   idx  config                         architecture    text_edge_list
#   0    e8_llm                         llm             present (n/a)
#   1    e8_rpearl_llm                  rpearl_llm      present
#   2    e8_rpearl_llm_no_edges         rpearl_llm      none
#   3    e8_rpearl_llm_gt               rpearl_gt_llm   present
#   4    e8_rpearl_llm_gt_no_edges      rpearl_gt_llm   none
#   5    e8_rpearl_llm_gt_pegain0       rpearl_gt_llm   present (pe_gain_init=0 cold-start sanity)
#   6    e8_rpearl_llm_gt_norope_graph  rpearl_gt_llm   present (idx5 + RoPE off for graph tokens)
#   7    e8_rpearl_llm_gt_norope_pegain005 rpearl_gt_llm present (idx6 + pe_gain_init=0.05, Ψ on)
#   8    e8_rpearl_llm_gt_frozen_pe1    rpearl_gt_llm   present (LLM frozen, pe_gain=1.0 — PE-only probe)
#   9    e8_rpearl_llm_gt_frozen_minimal rpearl_gt_llm  present (idx8 + minimal/local: 1 GT layer, k_gt=1, pe 2L/k=1)
#   10   e8_rpearl_llm_gt_rmsnorm       rpearl_gt_llm   present (calibrated RMSNorm Ψ + zero-init gate, LLM unfrozen/LoRA)
#   11   e8_rpearl_llm_gt_semantic      rpearl_gt_llm   present (idx10 + semantic node features: mean word-embedding, 1 GNN pass)
#   12   e8_gt_llm                      gt_llm          present (pure GT over word-embeddings, NO R-PEARL)
#   13   e8_graph_mask_llm              graph_mask_llm  present (NO PE/GNN: attention mask forced to mirror adjacency, k=1)
#   14   e8_gt_llm_no_edges             gt_llm          none    (idx12 + no edge list, pe_gain_init=0 cold-start)
#   15   e8_graph_mask_llm_no_edges     graph_mask_llm  present (idx13 ablation: edgeless mask, mask_use_edges=false — no connectivity)
#   16   e8_gt_llm_no_edges_L3          gt_llm          none    (gt_llm_no_edges GT-depth sweep, L=3; idx14 is L=2)
#   17   e8_gt_llm_no_edges_L4          gt_llm          none    (gt_llm_no_edges GT-depth sweep, L=4)
#   18   e8_gt_llm_no_edges_L5          gt_llm          none    (gt_llm_no_edges GT-depth sweep, L=5)
#   19   e8_gt_llm_no_edges_L5_d2048    gt_llm          none    (L5 GT-WIDTH sweep, d_model=2048; idx18 is d=1024)
#   20   e8_gt_llm_no_edges_L5_d3072    gt_llm          none    (L5 GT-WIDTH sweep, d_model=3072)
#   21   e8_gt_llm_no_edges_L5_d4096    gt_llm          none    (L5 GT-WIDTH sweep, d_model=4096, ~519M GT params)
#   22   e8_llm_zeroshot                llm             present (NO TRAINING: base Gemma zero-shot, edges in prompt)
#   23   e8_llm_zeroshot_no_edges       llm             none    (NO TRAINING: base Gemma zero-shot, no edges in prompt)
#
# Unique save_names are required: the committed yamls all share name: e8 and the
# rpearl_*/_no_edges pairs share an architecture, so train_v2 auto-naming would
# collide. We also override wandb_run_name (the committed yamls have duplicated
# run names) so each W&B run is distinct.
#
# Data: data/revised/gen/nav100_n30_gemma_data/split (rsynced gemma dataset).
#
# Submit:
#   sbatch scripts/e8_new_base_models.sbatch              # full 5-config array
#   sbatch --array=1,3 scripts/e8_new_base_models.sbatch  # subset
#   TASK_ID=0 bash scripts/e8_new_base_models.sbatch      # local single-task run

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

export PROJECT=${PROJECT:-GREP-PRISM}
export ENTITY=${ENTITY:-alelab}
export WANDB_ENTITY=${WANDB_ENTITY:-$ENTITY}
export EXPERIMENT_TAG=${EXPERIMENT_TAG:-e8_new_base_models}
export ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3}
export CHECKPOINT_DIR=${CHECKPOINT_DIR:-/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/outputs/e8_new_base_models}

# Held-out test graphs from the rsynced gemma dataset.
export TEST_GRAPHS=${TEST_GRAPHS:-data/revised/gen/nav100_n30_gemma_data/split/test_graphs}

CFG_DIR=experiments/e8_new_base_models
CONFIGS=(
    "$CFG_DIR/e8_llm.yaml"
    "$CFG_DIR/e8_rpearl_llm.yaml"
    "$CFG_DIR/e8_rpearl_llm_no_edges.yaml"
    "$CFG_DIR/e8_rpearl_llm_gt.yaml"
    "$CFG_DIR/e8_rpearl_llm_gt_no_edges.yaml"
    "$CFG_DIR/e8_rpearl_llm_gt_pegain0.yaml"
    "$CFG_DIR/e8_rpearl_llm_gt_norope_graph.yaml"
    "$CFG_DIR/e8_rpearl_llm_gt_norope_pegain005.yaml"
    "$CFG_DIR/e8_rpearl_llm_gt_frozen_pe1.yaml"
    "$CFG_DIR/e8_rpearl_llm_gt_frozen_minimal.yaml"
    "$CFG_DIR/e8_rpearl_llm_gt_rmsnorm.yaml"
    "$CFG_DIR/e8_rpearl_llm_gt_semantic.yaml"
    "$CFG_DIR/e8_gt_llm.yaml"
    "$CFG_DIR/e8_graph_mask_llm.yaml"
    "$CFG_DIR/e8_gt_llm_no_edges.yaml"
    "$CFG_DIR/e8_graph_mask_llm_no_edges.yaml"
    "$CFG_DIR/e8_gt_llm_no_edges_L3.yaml"
    "$CFG_DIR/e8_gt_llm_no_edges_L4.yaml"
    "$CFG_DIR/e8_gt_llm_no_edges_L5.yaml"
    "$CFG_DIR/e8_gt_llm_no_edges_L5_d2048.yaml"
    "$CFG_DIR/e8_gt_llm_no_edges_L5_d3072.yaml"
    "$CFG_DIR/e8_gt_llm_no_edges_L5_d4096.yaml"
    "$CFG_DIR/e8_llm_zeroshot.yaml"
    "$CFG_DIR/e8_llm_zeroshot_no_edges.yaml"
)
SAVE_NAMES=(
    e8_llm
    e8_rpearl_llm
    e8_rpearl_llm_no_edges
    e8_rpearl_llm_gt
    e8_rpearl_llm_gt_no_edges
    e8_rpearl_llm_gt_pegain0
    e8_rpearl_llm_gt_norope_graph
    e8_rpearl_llm_gt_norope_pegain005
    e8_rpearl_llm_gt_frozen_pe1
    e8_rpearl_llm_gt_frozen_minimal
    e8_rpearl_llm_gt_rmsnorm
    e8_rpearl_llm_gt_semantic
    e8_gt_llm
    e8_graph_mask_llm
    e8_gt_llm_no_edges
    e8_graph_mask_llm_no_edges
    e8_gt_llm_no_edges_L3
    e8_gt_llm_no_edges_L4
    e8_gt_llm_no_edges_L5
    e8_gt_llm_no_edges_L5_d2048
    e8_gt_llm_no_edges_L5_d3072
    e8_gt_llm_no_edges_L5_d4096
    e8_llm_zeroshot
    e8_llm_zeroshot_no_edges
)

TASK_ID=${SLURM_ARRAY_TASK_ID:-${1:-${TASK_ID:-0}}}
TOTAL_TASKS=${#CONFIGS[@]}
if (( TASK_ID < 0 || TASK_ID >= TOTAL_TASKS )); then
    echo "Invalid array index ${TASK_ID}; expected 0..$((TOTAL_TASKS - 1))" >&2
    exit 1
fi

export CONFIG=${CONFIGS[TASK_ID]}
export SAVE_NAME=${SAVE_NAMES[TASK_ID]}
export WANDB_RUN_NAME=$SAVE_NAME

echo "[e8] task ${TASK_ID}/$((TOTAL_TASKS - 1)): ${CONFIG} -> ${SAVE_NAME}"
exec bash scripts/train_and_eval.sh
