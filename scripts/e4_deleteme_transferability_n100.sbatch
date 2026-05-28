#!/bin/bash
#SBATCH --job-name=e4-transfer-n100-deleteme
#SBATCH --output=/vast/home/j/jporras/sourcecode/GREP/slurm-%A_%a.out
#SBATCH --time=10:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#SBATCH --array=0-2

# Transferability eval on the 20-graph N=100 deleteme sample.
# One array task per checkpoint; each task evaluates all 20 graphs in the
# folder (loads model once) across all 3 permutation seeds.

set -euo pipefail

ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v2}
OUTPUT_BASE=${OUTPUT_BASE:-/vast/home/j/jporras/sourcecode/GREP}
CHECKPOINT_BASE=${CHECKPOINT_BASE:-outputs/e4_new_training_data_2_iclfix}
EVAL_DIR=${EVAL_DIR:-data/eval/dev_deleteme__e4_transferability_skeletons__deleteme}
RESULTS_DIR=${RESULTS_DIR:-results/e4_transferability_n100_deleteme}

mkdir -p "${OUTPUT_BASE}"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-14}"

log() { echo "[$(date --iso-8601=seconds)] $*"; }

CHECKPOINTS=(
    e4_llm_llama-3.1-8b_r16_4bit_xriur1bi
    e4_rpearl_llm_llama-3.1-8b_r16_4bit_j9yjmx8j
    e4_rpearl_gt_llm_llama-3.1-8b_r16_4bit_33qmzmls
)
PERM_SEEDS=(32 42 58)

TASK_ID=${SLURM_ARRAY_TASK_ID:-${1:-0}}
if (( TASK_ID < 0 || TASK_ID >= ${#CHECKPOINTS[@]} )); then
    log "Invalid array index ${TASK_ID}; expected 0..$((${#CHECKPOINTS[@]} - 1))."
    exit 1
fi

CKPT_NAME=${CHECKPOINTS[TASK_ID]}
CHECKPOINT="${CHECKPOINT_BASE}/${CKPT_NAME}"
PATTERN="${EVAL_DIR}/eval_graph_unique_100*.json"

if [[ ! -d "$CHECKPOINT" ]]; then
    log "Checkpoint dir does not exist: $CHECKPOINT"
    exit 1
fi

N_GRAPHS=$(ls -1 ${PATTERN} 2>/dev/null | wc -l)
if (( N_GRAPHS == 0 )); then
    log "No eval graphs found at: $PATTERN — did you rsync from the workstation?"
    exit 1
fi

log "Task ${TASK_ID}: ckpt=${CKPT_NAME}  graphs=${N_GRAPHS}  perm_seeds=${PERM_SEEDS[*]}"

CMD=(
    srun
    python -m prism.eval.scalability_evaluation
    --checkpoint "$CHECKPOINT"
    --graphs "$PATTERN"
    --four-bit
    --permutation-seed "${PERM_SEEDS[@]}"
    --output "$RESULTS_DIR"
)

log "CMD: ${CMD[*]}"
"${CMD[@]}"

log "Done: ${CKPT_NAME} on N=100 (${N_GRAPHS} graphs)"
