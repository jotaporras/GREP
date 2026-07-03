#!/bin/bash
#SBATCH --job-name=e4-sweep
#SBATCH --output=/vast/home/j/jporras/sourcecode/GREP/slurm-%A_%a.out
#SBATCH --time=7:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#SBATCH --array=0-6%7

set -euo pipefail

PROJECT=${PROJECT:-GREP-PRISM}
ENTITY=${ENTITY:-alelab}
EXPERIMENT_TAG=${EXPERIMENT_TAG:-e4_new_training_data_2_iclfix}
ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v2}
OUTPUT_BASE=${OUTPUT_BASE:-/vast/home/j/jporras/sourcecode/GREP}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-outputs/e4_new_training_data_2_iclfix}

mkdir -p "${OUTPUT_BASE}"

# Stable CWD. Under SLURM, $0 is the spooled script path — don't use it.
# SLURM_SUBMIT_DIR is the directory sbatch was invoked from (the repo root if
# you submit with `sbatch scripts/e4_run_sweep.sbatch`). Local fallback uses
# the script's location.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

log() {
    echo "[$(date --iso-8601=seconds)] $*"
}

# Sweep matrix (TASK_ID 0..6). seq_len fixed at 8192.
#   PE_DIM = pe_hidden_channels (0 = sentinel for LLM / use yaml default).
#   D_MODEL = PE output dim (0 = use yaml default of 1024).
# Tasks 5-6 test "PE output dim matching Llama hidden_size=4096" — the
# direct-injection reading of 'matching LoRA' (vs the rank reading at task 2/4).
BASE_YAMLS=(
    experiments/e4_new_training_data_2/e4_llm.yaml
    experiments/e4_new_training_data_2/e4_rpearl_llm_no_edges.yaml
    experiments/e4_new_training_data_2/e4_rpearl_llm_no_edges.yaml
    experiments/e4_new_training_data_2/e4_rpearl_llm_gt_no_edges.yaml
    experiments/e4_new_training_data_2/e4_rpearl_llm_gt_no_edges.yaml
    experiments/e4_new_training_data_2/e4_rpearl_llm_no_edges.yaml
    experiments/e4_new_training_data_2/e4_rpearl_llm_gt_no_edges.yaml
)
SEQ_LENS=(8192 8192 8192 8192 8192 8192 8192)
PE_DIMS=(   0  256   16  256   16  256  256)
D_MODELS=(  0    0    0    0    0 4096 4096)
WANDB_NAMES=(
    e4_llm_newdata2_iclfix_seq8192
    e4_rpearl_newdata2_iclfix_seq8192_pe256
    e4_rpearl_newdata2_iclfix_seq8192_pe16
    e4_rpearl_gt_newdata2_iclfix_seq8192_pe256
    e4_rpearl_gt_newdata2_iclfix_seq8192_pe16
    e4_rpearl_newdata2_iclfix_seq8192_pe256_dm4096
    e4_rpearl_gt_newdata2_iclfix_seq8192_pe256_dm4096
)

TASK_ID=${SLURM_ARRAY_TASK_ID:-${1:-0}}
TOTAL_TASKS=${#BASE_YAMLS[@]}

if (( TASK_ID < 0 || TASK_ID >= TOTAL_TASKS )); then
    log "Invalid array index ${TASK_ID}; expected 0..$((TOTAL_TASKS - 1))."
    exit 1
fi

BASE_YAML=${BASE_YAMLS[TASK_ID]}
SEQ_LEN=${SEQ_LENS[TASK_ID]}
PE_DIM=${PE_DIMS[TASK_ID]}
D_MODEL=${D_MODELS[TASK_ID]}
WANDB_NAME=${WANDB_NAMES[TASK_ID]}

TASK_NUMBER=$((TASK_ID + 1))
log "Launching task ${TASK_NUMBER}/${TOTAL_TASKS}: yaml=${BASE_YAML}, seq=${SEQ_LEN}, pe=${PE_DIM}, d_model=${D_MODEL}, name=${WANDB_NAME}"

CMD=(
    srun
    python -m prism.training.train_v2
    "$BASE_YAML"
)
CMD+=(--max_seq_length "$SEQ_LEN")
CMD+=(--wandb_run_name "$WANDB_NAME")
CMD+=(--wandb_tag "$EXPERIMENT_TAG")
CMD+=(--checkpoint_dir "$CHECKPOINT_DIR")
if (( PE_DIM > 0 )); then
    CMD+=(--pe_hidden_channels "$PE_DIM")
fi
if (( D_MODEL > 0 )); then
    CMD+=(--d_model "$D_MODEL")
fi

log "CMD: ${CMD[*]}"
"${CMD[@]}"

log "Completed task ${TASK_NUMBER}/${TOTAL_TASKS}: ${WANDB_NAME}"
