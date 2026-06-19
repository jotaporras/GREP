#!/bin/bash
#SBATCH --job-name=e9-train-eval
#SBATCH --output=/vast/home/j/jporras/sourcecode/GREP/slurm-%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#SBATCH --array=0-4%4

# e9 gemma-4 31B  -- train + cross-eval, one self-contained config per task.
#
# Same 5 architectures as e8_new_base_models, but on google/gemma-4-31B-it
# (the ~32B dense gemma-4; tower-based Gemma4Config -- AutoModelForCausalLM
# loads the text tower only). seq/batch kept at e8 values (8192 / 1): the 31B
# text model in 4-bit (~15.5GB) + grad-checkpointed activations fits a B200
# with headroom. If the rpearl variants OOM, drop max_seq_length to 4096.
#
#   idx  config                         architecture    text_edge_list
#   0    e9_llm                         llm             present (n/a)
#   1    e9_rpearl_llm                  rpearl_llm      present
#   2    e9_rpearl_llm_no_edges         rpearl_llm      none
#   3    e9_rpearl_llm_gt               rpearl_gt_llm   present
#   4    e9_rpearl_llm_gt_no_edges      rpearl_gt_llm   none
#
# Submit:
#   sbatch scripts/e9_gemma31b.sbatch              # full 5-config array
#   sbatch --array=0 scripts/e9_gemma31b.sbatch    # canary (llm) first
#   TASK_ID=0 bash scripts/e9_gemma31b.sbatch      # local single-task run

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

export PROJECT=${PROJECT:-GREP-PRISM}
export ENTITY=${ENTITY:-alelab}
export WANDB_ENTITY=${WANDB_ENTITY:-$ENTITY}
export EXPERIMENT_TAG=${EXPERIMENT_TAG:-e8_new_base_models}
export ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3}
export CHECKPOINT_DIR=${CHECKPOINT_DIR:-/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/outputs/e9_gemma31b}

# Held-out test graphs from the rsynced gemma dataset.
export TEST_GRAPHS=${TEST_GRAPHS:-data/revised/gen/nav100_n30_gemma_data/split/test_graphs}

CFG_DIR=experiments/e9_gemma31b
CONFIGS=(
    "$CFG_DIR/e9_llm.yaml"
    "$CFG_DIR/e9_rpearl_llm.yaml"
    "$CFG_DIR/e9_rpearl_llm_no_edges.yaml"
    "$CFG_DIR/e9_rpearl_llm_gt.yaml"
    "$CFG_DIR/e9_rpearl_llm_gt_no_edges.yaml"
)
SAVE_NAMES=(
    e9_llm
    e9_rpearl_llm
    e9_rpearl_llm_no_edges
    e9_rpearl_llm_gt
    e9_rpearl_llm_gt_no_edges
)

TASK_ID=${SLURM_ARRAY_TASK_ID:-${1:-${TASK_ID:-0}}}
TOTAL_TASKS=${#CONFIGS[@]}
if (( TASK_ID < 0 || TASK_ID >= TOTAL_TASKS )); then
    echo "Invalid array index ${TASK_ID}; expected 0..$((TOTAL_TASKS - 1))" >&2
    exit 1
fi

export CONFIG=${CONFIGS[TASK_ID]}
export SAVE_NAME=${SAVE_NAMES[TASK_ID]}
export WANDB_RUN_NAME=$SAVE_NAME

echo "[e9] task ${TASK_ID}/$((TOTAL_TASKS - 1)): ${CONFIG} -> ${SAVE_NAME}"
exec bash scripts/train_and_eval.sh
