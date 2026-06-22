#!/bin/bash
#SBATCH --job-name=e9-multistage
#SBATCH --output=/vast/home/j/jporras/sourcecode/GREP/slurm-%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#
# e9 multistage training: SFT (Stage 1) -> PE-only edge-list loss (Stage 2),
# then STOP for analysis. Stage 3 (joint, edges removed) is launched separately
# after you inspect Stage-2 results.
#
# Submit:
#   sbatch scripts/e9_multistage_training.sbatch                  # stages 1+2
#   RUN_STAGE3=1 sbatch scripts/e9_multistage_training.sbatch     # all three in one job
#
# Stage 3 alone (after analysis): run the exact command the Stage-1/2 driver
# PRINTS on completion (it calls train_v2 directly with --init_lora_from /
# --init_pe_from pointing at the existing Stage-2 dir). Note RUN_STAGE3=1 RETRAINS
# Stages 1 and 2 from scratch (no skip-if-exists), so use it only for a fresh
# end-to-end run, not to resume into Stage 3.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

export PROJECT=${PROJECT:-GREP-PRISM}
export ENTITY=${ENTITY:-alelab}
export WANDB_ENTITY=${WANDB_ENTITY:-$ENTITY}
export EXPERIMENT_TAG=${EXPERIMENT_TAG:-e9_multistage_training}
export ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3}
export CHECKPOINT_DIR=${CHECKPOINT_DIR:-/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/outputs/e9_multistage_training}

exec bash scripts/e9_multistage_training.sh
