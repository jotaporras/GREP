#!/bin/bash
#SBATCH --job-name=e8-full-eval-n30
#SBATCH --output=/vast/home/j/jporras/sourcecode/GREP/slurm-%A_%a.out
#SBATCH --time=48:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#SBATCH --array=0-4%4

# Post-hoc FULL eval of five trained e8 checkpoints over ALL 100 graphs of the
# nav100_n30 gemma dataset (populated_graphs/, 10 tasks each = 1000 samples per
# model). One array task per checkpoint; architecture (LLM vs graph-augmented)
# and the train-time text_edge_list are auto-detected from each checkpoint by
# prism.eval.scalability_evaluation. All five were trained 4-bit, so eval loads
# 4-bit too (--four-bit).
#
#   idx  wandb id   checkpoint
#   0    yekuq6ge   e8_llm_yekuq6ge                  (plain LLM, edges in prompt)
#   1    6k3lflpf   e8_rpearl_llm_6k3lflpf           (R-PEARL,   edges present)
#   2    idqu6q6o   e8_graph_mask_llm_idqu6q6o       (graph-mask attn, edges present)
#   3    5clzd956   e8_graph_mask_llm_no_edges_5clzd956 (graph-mask, mask edges ablated; text edges present)
#   4    1g2bkh1h   e8_gt_llm_1g2bkh1h               (pure GT over word-embeddings)
#
# Results: <checkpoint>/eval_logs/<OUTPUT_SUBDIR>/<graph>.json (one per graph).
# OUTPUT_SUBDIR defaults to full_n30_eval to keep this 100-graph run SEPARATE
# from the 10-graph train-time cross_eval already under eval_logs/cross_eval/
# (the 10 test graphs are a subset of these 100, so the default dir would mix
# two different eval sets).
#
# Submit:
#   sbatch scripts/e8_full_eval_n30.sbatch                 # all five
#   sbatch --array=0,1,4 scripts/e8_full_eval_n30.sbatch   # subset
#   TASK_ID=0 bash scripts/e8_full_eval_n30.sbatch         # local single-task run

set -euo pipefail

# Stable CWD = repo root (so the repo-relative GRAPHS path resolves). Under SLURM
# $0 is the spooled script path — use SLURM_SUBMIT_DIR (the dir sbatch ran from);
# local fallback uses the script location.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

# SLURM 22.05+ no longer propagates --cpus-per-task to srun steps automatically.
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-14}"

ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3}
CKPT_BASE=${CKPT_BASE:-/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/outputs/e8_new_base_models}
# All 100 populated graphs (graph + 10 tasks each); test_graphs is a 10-graph subset.
GRAPHS=${GRAPHS:-data/revised/gen/nav100_n30_gemma_data/populated_graphs}
OUTPUT_SUBDIR=${OUTPUT_SUBDIR:-full_n30_eval}

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

log() { echo "[$(date --iso-8601=seconds)] $*"; }

# One checkpoint per array index (absolute dirs under the artifact root).
CHECKPOINTS=(
    "$CKPT_BASE/e8_llm_yekuq6ge"
    "$CKPT_BASE/e8_rpearl_llm_6k3lflpf"
    "$CKPT_BASE/e8_graph_mask_llm_idqu6q6o"
    "$CKPT_BASE/e8_graph_mask_llm_no_edges_5clzd956"
    "$CKPT_BASE/e8_gt_llm_1g2bkh1h"
)

TASK_ID=${SLURM_ARRAY_TASK_ID:-${1:-${TASK_ID:-0}}}
TOTAL_TASKS=${#CHECKPOINTS[@]}
if (( TASK_ID < 0 || TASK_ID >= TOTAL_TASKS )); then
    echo "Invalid array index ${TASK_ID}; expected 0..$((TOTAL_TASKS - 1))" >&2
    exit 1
fi

CHECKPOINT=${CHECKPOINTS[TASK_ID]}
OUTPUT_DIR="${CHECKPOINT}/eval_logs/${OUTPUT_SUBDIR}"

[[ -d "$CHECKPOINT" ]] || { log "ERROR: checkpoint dir not found: $CHECKPOINT"; exit 1; }
[[ -d "$GRAPHS" ]]     || { log "ERROR: graphs dir not found: $GRAPHS"; exit 1; }

log "[e8 full-eval] task ${TASK_ID}/$((TOTAL_TASKS - 1))"
log "  checkpoint: ${CHECKPOINT}"
log "  graphs:     ${GRAPHS}"
log "  output:     ${OUTPUT_DIR}"

CMD=(
    srun
    python -m prism.eval.scalability_evaluation
    --checkpoint "$CHECKPOINT"
    --graphs "$GRAPHS"
    --output "$OUTPUT_DIR"
    --four-bit
)

log "EVAL: ${CMD[*]}"
"${CMD[@]}"

log "Done: ${CHECKPOINT} -> ${OUTPUT_DIR}/<graph>.json"
