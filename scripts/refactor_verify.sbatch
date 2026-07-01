#!/bin/bash
#SBATCH --job-name=rv-rerun
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --time=12:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#SBATCH --array=0-5%3
#
#
# This slurm script reruns some of the e8 best runs  (GT / RPEARL / GraphMask, each
# with and without the textual edge list) to replicate after refactoring to support 
# Hydra and include eval into training. Eval is run based on checkpoints to ensure
# that loading is also functional.
#
#   idx  config                  arch            text_edge_list
#   0    rv_gt_l5d2048_edges     gt_llm          present
#   1    rv_gt_l5d2048_noedges   gt_llm          none
#   2    rv_rpearl_edges         rpearl_llm      present
#   3    rv_rpearl_noedges       rpearl_llm      none
#   4    rv_gmask_edges          graph_mask_llm  present  (real adjacency mask)
#   5    rv_gmask_noedges        graph_mask_llm  none     (real adjacency mask)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-14}"

PROJECT=${PROJECT:-GREP-PRISM}
ENTITY=${ENTITY:-alelab}
export WANDB_ENTITY=${WANDB_ENTITY:-$ENTITY}
EXPERIMENT_TAG=${EXPERIMENT_TAG:-refactor_verify_fulltest}
ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/outputs/refactor_verify_fulltest}
# Eval on the WHOLE held-out set (all 10 graphs), NOT the single data_gen_023.json the
# rv_*.yaml configs hardcode. num_graphs=-1 keeps every graph (data.py: <=0 => no cap).
TEST_GRAPHS=${TEST_GRAPHS:-data/revised/gen/nav100_n30_gemma_data/split/test_graphs}
EVAL_NUM_GRAPHS=${EVAL_NUM_GRAPHS:--1}

# config-name (Hydra) per array task, and the descriptive name used for BOTH the wandb
# run and the checkpoint save_name (kept identical so W&B and the on-disk dir line up).
CONFIGS=(
    refactor_verify/rv_gt_l5d2048_edges
    refactor_verify/rv_gt_l5d2048_noedges
    refactor_verify/rv_rpearl_edges
    refactor_verify/rv_rpearl_noedges
    refactor_verify/rv_gmask_edges
    refactor_verify/rv_gmask_noedges
)
NAMES=(
    rv_gt_l5d2048_edges
    rv_gt_l5d2048_noedges
    rv_rpearl_edges
    rv_rpearl_noedges
    rv_gmask_edges
    rv_gmask_noedges
)

TASK_ID=${SLURM_ARRAY_TASK_ID:-${1:-${TASK_ID:-0}}}
TOTAL_TASKS=${#CONFIGS[@]}
if (( TASK_ID < 0 || TASK_ID >= TOTAL_TASKS )); then
    echo "Invalid array index ${TASK_ID}; expected 0..$((TOTAL_TASKS - 1))" >&2
    exit 1
fi
CONFIG=${CONFIGS[TASK_ID]}
NAME=${NAMES[TASK_ID]}

module load anaconda3
source activate "$ENV_NAME"

echo "[$(date --iso-8601=seconds)] task ${TASK_ID}/$((TOTAL_TASKS - 1)): CONFIG=${CONFIG} NAME=${NAME} CHECKPOINT_DIR=${CHECKPOINT_DIR}"
srun python -m prism.training.train_v3 --config-name="$CONFIG" \
    trainer.save_name="$NAME" \
    trainer.checkpoint_dir="$CHECKPOINT_DIR" \
    wandb.project="$PROJECT" \
    wandb.run_name="$NAME" \
    wandb.tag="$EXPERIMENT_TAG" \
    eval.data="$TEST_GRAPHS" \
    eval.num_graphs="$EVAL_NUM_GRAPHS"
