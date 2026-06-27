#!/bin/bash
#SBATCH --job-name=rv-rerun
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --time=12:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#SBATCH --array=0-5%3
#
# refactor_verify — rerun the "best" e8 architectures (GT / RPEARL / GraphMask, each
# with and without the textual edge list) under the NEW nested Hydra config scheme, to
# verify the eval refactor end-to-end (train -> save -> RELOAD-from-disk -> evaluate_model).
#
# === SUBMIT AS A JOB ARRAY, NOT A for-LOOP ==================================
# This replaces the earlier `for c in ...; do CONFIG=$c sbatch ...; done` approach.
# That flooded the scheduler with 6+ independent jobs which got packed onto the same
# B200 nodes; the over-committed co-tenants were killed mid-eval (see reports/). A job
# array with a `%N` concurrency cap (here %3) throttles how many run at once and lets
# SLURM schedule the set as one managed unit.
#
#   idx  config                  arch            text_edge_list
#   0    rv_gt_l5d2048_edges     gt_llm          present
#   1    rv_gt_l5d2048_noedges   gt_llm          none
#   2    rv_rpearl_edges         rpearl_llm      present
#   3    rv_rpearl_noedges       rpearl_llm      none
#   4    rv_gmask_edges          graph_mask_llm  present  (real adjacency mask)
#   5    rv_gmask_noedges        graph_mask_llm  none     (real adjacency mask)
#
# Each config (experiments/refactor_verify/rv_*.yaml) inherits base_config and sets
# eval.post_train_graphs, so the refactored disk-reload cross-eval actually fires (it is a
# no-op when post_train_graphs is null). This script OWNS the wandb run name and the
# checkpoint save_name (one per task) via Hydra overrides, so neither falls back to the
# base_config default `wandb.run_name: e9_hydra` — every run is named for its arch+edge arm.
#
# Submit:
#   sbatch scripts/refactor_verify.sbatch                  # all 6, 3 at a time
#   sbatch --array=0,2,4 scripts/refactor_verify.sbatch    # subset
#   sbatch --array=2 scripts/refactor_verify.sbatch        # single arm (smoke the wrapper)
#   TASK_ID=2 bash scripts/refactor_verify.sbatch          # local single-task run (no SLURM)
#
# Mirrors the working e8/e9 pattern: `srun python -m prism.training.train_v3 ...` with
# SRUN_CPUS_PER_TASK exported (SLURM 22.05+ no longer propagates --cpus-per-task to steps)
# and srun required so the step gets the GPU cgroup/device binding.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-14}"

PROJECT=${PROJECT:-GREP-PRISM}
ENTITY=${ENTITY:-alelab}
export WANDB_ENTITY=${WANDB_ENTITY:-$ENTITY}
EXPERIMENT_TAG=${EXPERIMENT_TAG:-refactor_verify}
ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/outputs/refactor_verify}

# config-name (Hydra) per array task, and the descriptive name used for BOTH the wandb
# run and the checkpoint save_name (kept identical so W&B and the on-disk dir line up).
CONFIGS=(
    refactor_verify/rv_gt_l5d2048_edges
    refactor_verify/rv_gt_l5d2048_noedges
    refactor_verify/rv_rpearl_edges
    refactor_verify/rv_rpearl_noedges
    refactor_verify/rv_gmask_edges
    refactor_verify/rv_gmask_noedges
)
NAMES=(
    rv_gt_l5d2048_edges
    rv_gt_l5d2048_noedges
    rv_rpearl_edges
    rv_rpearl_noedges
    rv_gmask_edges
    rv_gmask_noedges
)

TASK_ID=${SLURM_ARRAY_TASK_ID:-${1:-${TASK_ID:-0}}}
TOTAL_TASKS=${#CONFIGS[@]}
if (( TASK_ID < 0 || TASK_ID >= TOTAL_TASKS )); then
    echo "Invalid array index ${TASK_ID}; expected 0..$((TOTAL_TASKS - 1))" >&2
    exit 1
fi
CONFIG=${CONFIGS[TASK_ID]}
NAME=${NAMES[TASK_ID]}

module load anaconda3
source activate "$ENV_NAME"

echo "[$(date --iso-8601=seconds)] task ${TASK_ID}/$((TOTAL_TASKS - 1)): CONFIG=${CONFIG} NAME=${NAME} CHECKPOINT_DIR=${CHECKPOINT_DIR}"
srun python -m prism.training.train_v3 --config-name="$CONFIG" \
    trainer.save_name="$NAME" \
    trainer.checkpoint_dir="$CHECKPOINT_DIR" \
    wandb.project="$PROJECT" \
    wandb.run_name="$NAME" \
    wandb.tag="$EXPERIMENT_TAG"
