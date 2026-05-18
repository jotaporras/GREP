#!/bin/bash
#SBATCH --job-name=e3-transfer
#SBATCH --output=/vast/home/j/jporras/sourcecode/GREP/slurm-%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#SBATCH --array=0-2

# Rerun of the April E3 transferability study on betty. One array task per
# checkpoint; each task is exactly the April invocation, with --output
# repointed at results/e3_transferability_betty/ so the original
# results/perm_*/ is preserved for direct A/B comparison.
#
# April pattern (from the user's notes):
#   uv run scripts/evaluate_suite.py <ckpt> \
#       --pattern "data/eval/transferability_data_v1/eval_graph_unique_*.json" \
#       --four-bit --device -1 --use-icl true \
#       --output results \
#       --permutation-seed 42 58 32
#
# Per-task output:
#   results/e3_transferability_betty/perm_{32,42,58}/<ckpt>_eval_graph_unique_<N>.json
#   results/e3_transferability_betty/perm_{32,42,58}/<ckpt>_summary.json
#
# Usage:
#   sbatch scripts/e3_transferability.sbatch
#   # Run a single checkpoint locally (e.g. LLM):
#   bash scripts/e3_transferability.sbatch 0

set -euo pipefail

PROJECT=${PROJECT:-GREP-PRISM}
ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v2}
OUTPUT_BASE=${OUTPUT_BASE:-/vast/home/j/jporras/sourcecode/GREP}
CHECKPOINT_BASE=${CHECKPOINT_BASE:-outputs/e3_new_training_data}
EVAL_DIR=${EVAL_DIR:-data/eval/transferability_data_v1}
RESULTS_DIR=${RESULTS_DIR:-results/e3_transferability_betty}

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

CHECKPOINTS=(
    e3_llm_llama-3.1-8b_r16_4bit_0sy9j5rz
    e3_rpearl_llm_llama-3.1-8b_r16_4bit_qmu8x2qu
    e3_rpearl_gt_llm_llama-3.1-8b_r16_4bit_bzqt4zxt
)

TASK_ID=${SLURM_ARRAY_TASK_ID:-${1:-0}}
if (( TASK_ID < 0 || TASK_ID >= ${#CHECKPOINTS[@]} )); then
    log "Invalid array index ${TASK_ID}; expected 0..$((${#CHECKPOINTS[@]} - 1))."
    exit 1
fi

CKPT_NAME=${CHECKPOINTS[TASK_ID]}
CHECKPOINT="${CHECKPOINT_BASE}/${CKPT_NAME}"

if [[ ! -d "$CHECKPOINT" ]]; then
    log "Checkpoint dir does not exist: $CHECKPOINT"
    exit 1
fi

log "Task ${TASK_ID}: ckpt=${CKPT_NAME}"
log "Eval glob: ${EVAL_DIR}/eval_graph_unique_*.json"
log "Output:    ${RESULTS_DIR}/perm_{32,42,58}/"

CMD=(
    srun
    python scripts/evaluate_suite.py
    "$CHECKPOINT"
    --pattern "${EVAL_DIR}/eval_graph_unique_*.json"
    --four-bit
    --device -1
    --use-icl true
    --output "$RESULTS_DIR"
    --permutation-seed 42 58 32
)

log "CMD: ${CMD[*]}"
"${CMD[@]}"

log "Done: ${CKPT_NAME}"
