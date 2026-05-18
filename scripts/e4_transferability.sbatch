#!/bin/bash
#SBATCH --job-name=e4-transfer
#SBATCH --output=/vast/home/j/jporras/sourcecode/GREP/slurm-%A_%a.out
#SBATCH --time=10:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#SBATCH --array=0-14%4

# Transferability + permutation-equivariance eval for the best E4 checkpoints.
#
# Each array task evaluates ONE (checkpoint, graph_size) cell across all three
# permutation seeds, mirroring the E3 transferability study in
# notebooks/e3_transferability_results.ipynb. The 15 tasks cover:
#     3 checkpoints × 5 graph sizes  (3 perm seeds run inside each task)
#
# Sizes 500 and 1000 dropped: GPT-5.5 silently truncates the populated graph
# at those sizes (drops most regions, leaves dangling edges).
#
# Eval graphs (populated locally with GPT-5.5) must already live at
# data/eval/e4_transferability/eval_graph_unique_{10,30,50,100,250}.json.
#
# Per-task output:
#   results/e4_transferability/perm_{32,42,58}/<ckpt>_eval_graph_unique_<N>.json
#   results/e4_transferability/perm_{32,42,58}/<ckpt>_summary.json   (overwritten per task)
#
# Usage:
#   sbatch scripts/e4_transferability.sbatch
#   # Run a single cell (LLM, N=10) locally:
#   bash scripts/e4_transferability.sbatch 0

set -euo pipefail

PROJECT=${PROJECT:-GREP-PRISM}
ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v2}
OUTPUT_BASE=${OUTPUT_BASE:-/vast/home/j/jporras/sourcecode/GREP}
CHECKPOINT_BASE=${CHECKPOINT_BASE:-outputs/e4_new_training_data_2_iclfix}
EVAL_DIR=${EVAL_DIR:-data/eval/e4_transferability}
RESULTS_DIR=${RESULTS_DIR:-results/e4_transferability}

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

# Best run from each architecture in the e4 iclfix sweep.
CHECKPOINTS=(
    e4_llm_llama-3.1-8b_r16_4bit_xriur1bi
    e4_rpearl_llm_llama-3.1-8b_r16_4bit_j9yjmx8j
    e4_rpearl_gt_llm_llama-3.1-8b_r16_4bit_33qmzmls
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
EVAL_FILE="${EVAL_DIR}/eval_graph_unique_${SIZE}.json"

if [[ ! -d "$CHECKPOINT" ]]; then
    log "Checkpoint dir does not exist: $CHECKPOINT"
    exit 1
fi
if [[ ! -f "$EVAL_FILE" ]]; then
    log "Eval graph does not exist: $EVAL_FILE — did you rsync from the workstation?"
    exit 1
fi

log "Task ${TASK_ID}/$((TOTAL_TASKS - 1)): ckpt=${CKPT_NAME}  size=${SIZE}  perm_seeds=${PERM_SEEDS[*]}"

CMD=(
    srun
    python scripts/evaluate_suite.py
    "$CHECKPOINT"
    --pattern "$EVAL_FILE"
    --four-bit
    --permutation-seed "${PERM_SEEDS[@]}"
    --output "$RESULTS_DIR"
)

log "CMD: ${CMD[*]}"
"${CMD[@]}"

log "Done: ${CKPT_NAME} on N=${SIZE}"
