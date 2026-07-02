#!/bin/bash
#SBATCH --job-name=diag-inj-ablation
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%A_%a.out
#SBATCH --time=4:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#SBATCH --array=0-5%3
#
# Injection-ablation diagnostic (Step 1 of the no-edge-list investigation): grade
# teacher-forced decision/completion/repeat token accuracy under
# train_style / prompt_only / no_injection conditions on the trained no-edge
# checkpoints. No training; forward passes only. See scripts/diag_injection_ablation.py
# for how to read the resulting table.
#
#   idx  checkpoint                             arch                  wandb run
#   0    rv_gmask_noedges_ror8gtet              graph_mask_llm        ror8gtet (gen acc 0.13)
#   1    rv_rpearl_noedges_8yav2y2h             rpearl_llm            8yav2y2h (gen acc 0.11)
#   2    rv_gt_l5d2048_noedges_e0bwztzo         gt_llm                e0bwztzo (gen acc 0.16)
#   3    e9_ms_stage3__no_edge_list_kfuu2djo    rpearl_gt_llm         kfuu2djo (gen acc 0.10, pe_gain 0.27)
#   4    e10_integ_rpe__no_edge_list_tyvhwlmx   learnable_graph_mask  tyvhwlmx (gen acc 0.14)
#   5    e9_llm_baseline_noedges__gemma31b      llm (floor reference) 4r9lf0me (gen acc 0.16)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3}
OUTPUTS=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/outputs
VAL_FILE=${VAL_FILE:-data/revised/gen/nav100_n30_gemma_data/split/formatted_all_new__val.json}

# Home dir sits at its quota and its model caches were purged; the hub cache
# lives on vast (shared across jobs, downloads once), datasets cache on
# node-local scratch.
export HF_HOME=/vast/projects/aribeiro/alelab/jporras/hf_cache
export HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf_datasets_${SLURM_JOB_ID:-manual}"

CHECKPOINTS=(
    "$OUTPUTS/refactor_verify_fulltest/rv_gmask_noedges_ror8gtet"
    "$OUTPUTS/refactor_verify_fulltest/rv_rpearl_noedges_8yav2y2h"
    "$OUTPUTS/refactor_verify_fulltest/rv_gt_l5d2048_noedges_e0bwztzo"
    "$OUTPUTS/e9_multistage_training/e9_ms_stage3/e9_ms_stage3__no_edge_list_kfuu2djo"
    "$OUTPUTS/dev_e10/e10_integ_rpe/e10_integ_rpe__no_edge_list_tyvhwlmx"
    "$OUTPUTS/e9_llm_baseline_noedges__gemma31b/e9_llm_baseline_noedges__gemma31b_4r9lf0me"
)

TASK_ID=${SLURM_ARRAY_TASK_ID:-${1:-0}}
CHECKPOINT=${CHECKPOINTS[TASK_ID]}

module load anaconda3
source activate "$ENV_NAME"

echo "[$(date --iso-8601=seconds)] diag task ${TASK_ID}: CHECKPOINT=${CHECKPOINT}"
srun python scripts/diag_injection_ablation.py \
    --checkpoint "$CHECKPOINT" \
    --val-file "$VAL_FILE"
