#!/bin/bash
#SBATCH --job-name=rv-rerun
#SBATCH --output=slurm-%x-%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#
# refactor_verify — rerun a "best" e8 architecture under the NEW nested Hydra config
# scheme, to verify the eval refactor end-to-end (train -> save -> RELOAD-from-disk ->
# evaluate_model). One config per job. Submit one per arm (6 total) — see the loop at the
# bottom of the file.
#
# Each config (experiments/refactor_verify/rv_*.yaml) inherits base_config and sets
# eval.post_train_graphs, so the refactored disk-reload cross-eval actually fires (it is a
# no-op when post_train_graphs is null).
#
# Submit:
#   CONFIG=refactor_verify/rv_gt_l5d2048_edges sbatch scripts/refactor_verify.sbatch
#
# Mirrors the working e8/e9 pattern: `srun python -m prism.training.train_v3 ...` with
# SRUN_CPUS_PER_TASK exported (SLURM 22.05+ no longer propagates --cpus-per-task to steps)
# and srun required so the step gets the GPU cgroup binding.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-14}"

: "${CONFIG:?Set CONFIG=<hydra config-name>, e.g. refactor_verify/rv_gt_l5d2048_edges}"

PROJECT=${PROJECT:-GREP-PRISM}
ENTITY=${ENTITY:-alelab}
export WANDB_ENTITY=${WANDB_ENTITY:-$ENTITY}
ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/outputs/refactor_verify}

module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "[$(date --iso-8601=seconds)] CONFIG=$CONFIG CHECKPOINT_DIR=$CHECKPOINT_DIR"
srun python -m prism.training.train_v3 --config-name="$CONFIG" \
    trainer.checkpoint_dir="$CHECKPOINT_DIR" \
    wandb.project="$PROJECT"

# ---------------------------------------------------------------------------
# Submit all six arms (run from the repo root on betty):
#
#   for c in rv_gt_l5d2048_edges rv_gt_l5d2048_noedges \
#            rv_rpearl_edges     rv_rpearl_noedges \
#            rv_gmask_edges      rv_gmask_noedges; do
#       CONFIG="refactor_verify/$c" sbatch scripts/refactor_verify.sbatch
#   done
# ---------------------------------------------------------------------------
