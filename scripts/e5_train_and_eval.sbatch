#!/bin/bash
#SBATCH --job-name=e5-train-eval
#SBATCH --output=/vast/home/j/jporras/sourcecode/GREP/slurm-%A_%a.out
#SBATCH --time=16:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#SBATCH --array=0-4%5

# e5 PE / architecture sweep — train + cross-eval, one config per array task.
#
# Five configs (RPEARL and Graph-Transformer) varying pe_hidden_channels and
# d_model against Llama-3.1-8B's hidden size (4096):
#
#   idx  cfg          arch    pe_hidden  d_model  pe_num_layers(L)  k_pe(K)  k_gt  gt_num_layers
#   0    cfg1         rpearl  4096       4096     5                 3        -     -
#   1    cfg2         rpearl  4096       4096     2                 3        -     -
#   2    cfg3         gt      4096       4096     2                 2        2     2
#   3    cfg4-rpearl  rpearl  256        4096     2                 3        -     -
#   4    cfg4-gt      gt      256        4096     2                 2        2     2
#
# configs 1-3 raise both pe_hidden_channels and d_model to 4096; config 4 keeps
# pe_hidden_channels at the yaml default (256) and only raises d_model.
#
# Each array task trains one config on the e5 graph-oriented dataset, then runs
# cross-eval on that dataset's held-out test graphs — both in the same task.
#
# Submit AFTER scripts/e5_generate_training_data.sh has produced
# data/gen/e5_graph_oriented_data/split/. Local single-task run:
#   bash scripts/e5_train_and_eval.sbatch <TASK_ID>

set -euo pipefail

PROJECT=${PROJECT:-GREP-PRISM}
ENTITY=${ENTITY:-alelab}
EXPERIMENT_TAG=${EXPERIMENT_TAG:-e5_pe_sweep}
ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v2}
OUTPUT_BASE=${OUTPUT_BASE:-/vast/home/j/jporras/sourcecode/GREP}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-outputs/e5_pe_sweep}

# e5 dataset produced by scripts/e5_generate_training_data.sh
DATA_SPLIT=${DATA_SPLIT:-data/gen/e5_graph_oriented_data/split}

mkdir -p "${OUTPUT_BASE}"

# Stable CWD (repo root). Under SLURM, $0 is the spooled script — don't use it;
# SLURM_SUBMIT_DIR is the directory `sbatch` was invoked from. Local fallback
# uses the script's location.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

# SLURM 22.05+ no longer propagates --cpus-per-task to srun steps automatically.
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-14}"

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

log() {
    echo "[$(date --iso-8601=seconds)] $*"
}

TRAIN_DATA="${DATA_SPLIT}/formatted_all_new__train.json"
VAL_DATA="${DATA_SPLIT}/formatted_all_new__val.json"
TEST_GRAPHS="${DATA_SPLIT}/test_graphs"

if [[ ! -f "$TRAIN_DATA" || ! -f "$VAL_DATA" || ! -d "$TEST_GRAPHS" ]]; then
    log "ERROR: e5 dataset not found under ${DATA_SPLIT}."
    log "       Run scripts/e5_generate_training_data.sh first."
    exit 1
fi

# Single held-out graph for the per-epoch in-training eval callback.
EVAL_DATA=$(ls "${TEST_GRAPHS}"/data_gen_*.json 2>/dev/null | sort | head -1 || true)
if [[ -z "$EVAL_DATA" ]]; then
    log "ERROR: no data_gen_*.json found in ${TEST_GRAPHS}."
    exit 1
fi

RPEARL_YAML=experiments/e4_new_training_data_2/e4_rpearl_llm_no_edges.yaml
GT_YAML=experiments/e4_new_training_data_2/e4_rpearl_llm_gt_no_edges.yaml

# ---- Per-task config arrays (5 tasks). GT-only keys are 0 for RPEARL tasks. ----
#                       0              1              2           3              4
BASE_YAMLS=(  "$RPEARL_YAML"  "$RPEARL_YAML"  "$GT_YAML"  "$RPEARL_YAML"  "$GT_YAML" )
SAVE_NAMES=(
    e5_cfg1_rpearl_pe4096_dm4096_L5
    e5_cfg2_rpearl_pe4096_dm4096_L2
    e5_cfg3_gt_pe4096_dm4096_L2
    e5_cfg4_rpearl_pe256_dm4096_L2
    e5_cfg4_gt_pe256_dm4096_L2
)
PE_HIDDEN=(   4096   4096   4096   256    256  )
D_MODELS=(    4096   4096   4096   4096   4096 )
PE_LAYERS=(   5      2      2      2      2    )
K_PE=(        3      3      2      3      2    )
K_GT=(        0      0      2      0      2    )   # 0 = not a GT task
GT_LAYERS=(   0      0      2      0      2    )

TASK_ID=${SLURM_ARRAY_TASK_ID:-${1:-0}}
TOTAL_TASKS=${#BASE_YAMLS[@]}

if (( TASK_ID < 0 || TASK_ID >= TOTAL_TASKS )); then
    log "Invalid array index ${TASK_ID}; expected 0..$((TOTAL_TASKS - 1))."
    exit 1
fi

BASE_YAML=${BASE_YAMLS[TASK_ID]}
SAVE_NAME=${SAVE_NAMES[TASK_ID]}
TASK_NUMBER=$((TASK_ID + 1))

log "Task ${TASK_NUMBER}/${TOTAL_TASKS}: ${SAVE_NAME}"
log "  yaml=${BASE_YAML} pe_hidden=${PE_HIDDEN[TASK_ID]} d_model=${D_MODELS[TASK_ID]} pe_num_layers=${PE_LAYERS[TASK_ID]} k_pe=${K_PE[TASK_ID]} k_gt=${K_GT[TASK_ID]} gt_num_layers=${GT_LAYERS[TASK_ID]}"

# ---- Stage 1: training ----
TRAIN_CMD=(
    srun
    python -m prism.training.train_v2
    "$BASE_YAML"
    --name e5
    --data "$TRAIN_DATA"
    --val_data "$VAL_DATA"
    --eval_data "$EVAL_DATA"
    --checkpoint_dir "$CHECKPOINT_DIR"
    --save_name "$SAVE_NAME"
    --wandb_project "$PROJECT"
    --wandb_run_name "$SAVE_NAME"
    --wandb_tag "$EXPERIMENT_TAG"
    --pe_hidden_channels "${PE_HIDDEN[TASK_ID]}"
    --d_model "${D_MODELS[TASK_ID]}"
    --pe_num_layers "${PE_LAYERS[TASK_ID]}"
    --k_pe "${K_PE[TASK_ID]}"
)
if (( GT_LAYERS[TASK_ID] > 0 )); then
    TRAIN_CMD+=(--k_gt "${K_GT[TASK_ID]}" --gt_num_layers "${GT_LAYERS[TASK_ID]}")
fi

log "TRAIN: ${TRAIN_CMD[*]}"
"${TRAIN_CMD[@]}"

# ---- Stage 2: locate the trained checkpoint ----
# train_v2 writes to {checkpoint_dir}/{save_name}_{wandb_run_id}; save_name is
# unique per task, so glob + newest mtime resolves it unambiguously.
CHECKPOINT=$(ls -dt "${CHECKPOINT_DIR}/${SAVE_NAME}"_*/ 2>/dev/null | head -1 || true)
CHECKPOINT="${CHECKPOINT%/}"
if [[ -z "$CHECKPOINT" || ! -d "$CHECKPOINT" ]]; then
    log "ERROR: no checkpoint found matching ${CHECKPOINT_DIR}/${SAVE_NAME}_*"
    exit 1
fi
log "Checkpoint: ${CHECKPOINT}"

# ---- Stage 3: cross-eval on the e5 held-out test graphs ----
# eval_checkpoint_on_graphs.py auto-detects architecture (rpearl vs gt vs llm)
# from the checkpoint and writes per-graph results to
# {checkpoint}/eval_logs/cross_eval/.
EVAL_CMD=(
    srun
    python scripts/eval_checkpoint_on_graphs.py
    --checkpoint "$CHECKPOINT"
    --graphs "$TEST_GRAPHS"
)
log "EVAL: ${EVAL_CMD[*]}"
"${EVAL_CMD[@]}"

log "Completed task ${TASK_NUMBER}/${TOTAL_TASKS}: ${SAVE_NAME} -> ${CHECKPOINT}/eval_logs/cross_eval/"
