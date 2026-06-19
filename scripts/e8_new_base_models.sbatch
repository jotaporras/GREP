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
)
SAVE_NAMES=(
    e8_llm
    e8_rpearl_llm
    e8_rpearl_llm_no_edges
    e8_rpearl_llm_gt
    e8_rpearl_llm_gt_no_edges
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
