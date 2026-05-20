#!/bin/bash
#SBATCH --job-name=e5-transfer
#SBATCH --output=/vast/home/j/jporras/sourcecode/GREP/slurm-%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#SBATCH --array=0-14%4

# Transferability + permutation-equivariance eval for the best E5 checkpoints.
#
# Each array task evaluates ONE (checkpoint, graph_size) cell — all 10 graphs of
# that size — across all three permutation seeds (model loaded once). The 15
# tasks cover:
#     3 checkpoints × 5 graph sizes  (10 graphs + 3 perm seeds run inside each)
#
# Sizes 500 and 1000 dropped: the populate model silently truncates the graph at
# those sizes (drops most regions, leaves dangling edges).
#
# Eval graphs (populated locally, 10 per size) must already live at
# data/eval/e5_transferability/eval_graph_unique_{10,30,50,100,250}_s*.json
# — produced by scripts/generate_e5_transferability_suite.sh and rsynced over.
#
# Per-task output:
#   results/e5_transferability/perm_{32,42,58}/<ckpt>_eval_graph_unique_<N>_s<seed>.json
#   results/e5_transferability/perm_{32,42,58}/<ckpt>_summary.json   (overwritten per task)
#
# Usage:
#   sbatch scripts/e5_transferability.sbatch
#   # Run a single cell (first ckpt, N=10) locally:
#   bash scripts/e5_transferability.sbatch 0

set -euo pipefail

PROJECT=${PROJECT:-GREP-PRISM}
ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v2}
OUTPUT_BASE=${OUTPUT_BASE:-/vast/home/j/jporras/sourcecode/GREP}
CHECKPOINT_BASE=${CHECKPOINT_BASE:-/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/outputs/e5_pe_sweep}
EVAL_DIR=${EVAL_DIR:-data/eval/e5_transferability}
RESULTS_DIR=${RESULTS_DIR:-results/e5_transferability}

mkdir -p "${OUTPUT_BASE}"

# Stable CWD. Under SLURM, $0 is the spooled script path — use SLURM_SUBMIT_DIR
# (the directory sbatch was invoked from). Local fallback uses script location.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# SLURM 22.05+ stopped propagating --cpus-per-task from sbatch to srun, which
# causes "CPU binding outside of job step allocation" on dgx-b200 nodes whose
# topology doesn't include cpu 0. Pass it explicitly.
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-14}"

log() {
    echo "[$(date --iso-8601=seconds)] $*"
}

# Best run per architecture from the e5 PE sweep — SAVE_NAME prefixes (the
# wandb-id suffix is glob-resolved below). EDIT after the e5 cross-eval finishes:
# inspect $ALELAB_DRIVE/GREP-PRISM/outputs/e5_pe_sweep/*/eval_logs/cross_eval/ (or notebooks/e5_pe_sweep.ipynb)
# and set the best RPEARL, best GT, and an e4ref baseline.
# The 7 e5 SAVE_NAMEs (see scripts/e5_train_and_eval.sbatch):
#   e5_cfg1_rpearl_pe4096_dm4096_L5   e5_cfg2_rpearl_pe4096_dm4096_L2
#   e5_cfg3_gt_pe4096_dm4096_L2       e5_cfg4_rpearl_pe256_dm4096_L2
#   e5_cfg4_gt_pe256_dm4096_L2        e5_e4ref_rpearl_pe256_dm1024_L5
#   e5_e4ref_gt_pe256_dm1024_L5
CHECKPOINTS=(
    e5_cfg1_rpearl_pe4096_dm4096_L5   # best RPEARL  (edit after cross-eval)
    e5_cfg3_gt_pe4096_dm4096_L2       # best GT      (edit after cross-eval)
    e5_e4ref_rpearl_pe256_dm1024_L5   # e4 baseline  (edit after cross-eval)
)
SIZES=(10 30 50 100 250)
PERM_SEEDS=(32 42 58)

N_CKPT=${#CHECKPOINTS[@]}
N_SIZE=${#SIZES[@]}
TOTAL_TASKS=$((N_CKPT * N_SIZE))

TASK_ID=${SLURM_ARRAY_TASK_ID:-${1:-0}}
if (( TASK_ID < 0 || TASK_ID >= TOTAL_TASKS )); then
    log "Invalid array index ${TASK_ID}; expected 0..$((TOTAL_TASKS - 1))."
    exit 1
fi

CKPT_IDX=$(( TASK_ID / N_SIZE ))
SIZE_IDX=$(( TASK_ID % N_SIZE ))

CKPT_PREFIX=${CHECKPOINTS[CKPT_IDX]}
SIZE=${SIZES[SIZE_IDX]}

# Resolve the SAVE_NAME prefix to a checkpoint dir (newest mtime), as in
# scripts/e5_train_and_eval.sbatch — the wandb-id suffix is unknown ahead of time.
CHECKPOINT=$(ls -dt "${CHECKPOINT_BASE}/${CKPT_PREFIX}"_*/ 2>/dev/null | head -1 || true)
CHECKPOINT="${CHECKPOINT%/}"
if [[ -z "$CHECKPOINT" || ! -d "$CHECKPOINT" ]]; then
    log "No checkpoint found matching ${CHECKPOINT_BASE}/${CKPT_PREFIX}_*"
    exit 1
fi

# All 10 graphs of this size. The "_s" delimiter keeps N=10 from matching N=100.
PATTERN="${EVAL_DIR}/eval_graph_unique_${SIZE}_s*.json"
N_GRAPHS=$(ls -1 ${PATTERN} 2>/dev/null | wc -l)
if (( N_GRAPHS == 0 )); then
    log "No eval graphs found at: $PATTERN — did you rsync from the workstation?"
    exit 1
fi

log "Task ${TASK_ID}/$((TOTAL_TASKS - 1)): ckpt=${CKPT_PREFIX}  size=${SIZE}  graphs=${N_GRAPHS}  perm_seeds=${PERM_SEEDS[*]}"
log "Resolved checkpoint: ${CHECKPOINT}"

CMD=(
    srun
    python scripts/evaluate_suite.py
    "$CHECKPOINT"
    --pattern "$PATTERN"
    --four-bit
    --permutation-seed "${PERM_SEEDS[@]}"
    --output "$RESULTS_DIR"
)

log "CMD: ${CMD[*]}"
"${CMD[@]}"

log "Done: ${CKPT_PREFIX} on N=${SIZE} (${N_GRAPHS} graphs)"
