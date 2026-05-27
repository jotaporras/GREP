#!/bin/bash
#SBATCH --job-name=e5-transfer
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%A_%a.out
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
# All artifacts live under the cluster artifact root, never in the code repo:
#   ARTIFACT_ROOT=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM
#
# Eval graphs (populated locally, 10 per size) must already live at
#   ${ARTIFACT_ROOT}/data/eval/e5_transferability/eval_graph_unique_{10,30,50,100,250}_s*.json
# — produced by scripts/generate_e5_transferability_suite.sh and rsynced over.
#
# Per-task output:
#   ${ARTIFACT_ROOT}/results/e5_transferability/perm_{32,42,58}/<ckpt>_eval_graph_unique_<N>_s<seed>.json
#   ${ARTIFACT_ROOT}/results/e5_transferability/perm_{32,42,58}/<ckpt>_summary.json  (overwritten per task)
#
# Usage:
#   sbatch scripts/e5_transferability.sbatch
#   # Run a single cell (first ckpt, N=10) locally:
#   bash scripts/e5_transferability.sbatch 0

set -euo pipefail

PROJECT=${PROJECT:-GREP-PRISM}
ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v2}

# Cluster artifact root — all data/outputs/results/logs live here, NEVER in the
# code repo (/vast/home/j/jporras/sourcecode/GREP holds only code).
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/vast/projects/aribeiro/alelab/jporras/GREP-PRISM}
OUTPUT_BASE=${OUTPUT_BASE:-${ARTIFACT_ROOT}}
CHECKPOINT_BASE=${CHECKPOINT_BASE:-${ARTIFACT_ROOT}/outputs/e5_pe_sweep}
EVAL_DIR=${EVAL_DIR:-${ARTIFACT_ROOT}/data/eval/e5_transferability}
RESULTS_DIR=${RESULTS_DIR:-${ARTIFACT_ROOT}/results/e5_transferability}

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

# One run per architecture from the e5 PE sweep, selected by cross-eval
# accuracy on the held-out test graphs, plus the text-only LLM baseline:
#   RPEARL — cfg4-rpearl    45% (9/20)
#   GT     — cfg4-gt        50% (10/20)
#   LLM    — text-only baseline (graph serialized into the prompt).
CHECKPOINTS=(
    e5_cfg4_rpearl_pe256_dm4096_L2_rbg6w5c4    # RPEARL, 45% cross-eval
    e5_cfg4_gt_pe256_dm4096_L2_nuohivct        # GT,     50% cross-eval
    e5_llm_r5pbg901                            # LLM     text-only baseline
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

CKPT_NAME=${CHECKPOINTS[CKPT_IDX]}
SIZE=${SIZES[SIZE_IDX]}

CHECKPOINT="${CHECKPOINT_BASE}/${CKPT_NAME}"
if [[ ! -d "$CHECKPOINT" ]]; then
    log "Checkpoint dir does not exist: $CHECKPOINT"
    exit 1
fi

# All 10 graphs of this size. The "_s" delimiter keeps N=10 from matching N=100.
PATTERN="${EVAL_DIR}/eval_graph_unique_${SIZE}_s*.json"
# Count matches without tripping `set -e`/`pipefail`: `ls` of an unmatched glob
# exits 2, which would silently kill the script before the friendly check below.
shopt -s nullglob
EVAL_FILES=( ${PATTERN} )
shopt -u nullglob
N_GRAPHS=${#EVAL_FILES[@]}
if (( N_GRAPHS == 0 )); then
    log "No eval graphs found at: $PATTERN — did you rsync from the workstation?"
    exit 1
fi

log "Task ${TASK_ID}/$((TOTAL_TASKS - 1)): ckpt=${CKPT_NAME}  size=${SIZE}  graphs=${N_GRAPHS}  perm_seeds=${PERM_SEEDS[*]}"

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

log "Done: ${CKPT_NAME} on N=${SIZE} (${N_GRAPHS} graphs)"
