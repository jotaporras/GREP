#!/bin/bash
# train_and_eval.sh — de-facto train + cross-eval driver.
#
# Generalized from scripts/e5_train_and_eval.sbatch (renamed; see git history).
# Runs the standard two-stage pipeline for ONE self-contained training config:
#   1. train  : python -m prism.training.train_v2 $CONFIG [overrides]
#   2. locate : newest {CHECKPOINT_DIR}/{SAVE_NAME}_*  checkpoint dir
#   3. eval   : scripts/eval_checkpoint_on_graphs.py on $TEST_GRAPHS
#
# Driven entirely by environment variables so per-experiment sbatch array
# scripts can reuse it without duplicating the pipeline (see
# scripts/e8_new_base_models.sbatch). Standalone single-config dry run:
#
#   CONFIG=experiments/e8_new_base_models/e8_llm.yaml \
#   SAVE_NAME=e8_llm \
#   TEST_GRAPHS=data/revised/gen/nav100_n30_gemma_data/split/test_graphs \
#   CHECKPOINT_DIR=outputs/e8_new_base_models \
#   bash scripts/train_and_eval.sh
#
# Required env:
#   CONFIG        path (repo-relative) to a self-contained training yaml
#   SAVE_NAME     unique checkpoint/run name (train_v2 appends the wandb run id)
#   TEST_GRAPHS   directory of held-out test graphs for the cross-eval stage
# Optional env (override the corresponding yaml field when set):
#   CHECKPOINT_DIR, PROJECT, EXPERIMENT_TAG, WANDB_RUN_NAME, NAME,
#   DATA, VAL_DATA, EVAL_DATA, ENV_NAME, EXTRA_TRAIN_ARGS

set -euo pipefail

: "${CONFIG:?set CONFIG to a training yaml}"
: "${SAVE_NAME:?set SAVE_NAME}"
: "${TEST_GRAPHS:?set TEST_GRAPHS to a test_graphs directory}"

PROJECT=${PROJECT:-GREP-PRISM}
ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v2}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-$SAVE_NAME}

# Stable CWD (repo root). Under SLURM $0 is the spooled script — do not use it;
# SLURM_SUBMIT_DIR is the directory sbatch was invoked from. Local fallback uses
# the script location.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

# SLURM 22.05+ no longer propagates --cpus-per-task to srun steps automatically.
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-14}"

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

log() { echo "[$(date --iso-8601=seconds)] $*"; }

[[ -f "$CONFIG" ]]      || { log "ERROR: config not found: $CONFIG"; exit 1; }
[[ -d "$TEST_GRAPHS" ]] || { log "ERROR: test graphs dir not found: $TEST_GRAPHS"; exit 1; }

log "Config ${CONFIG} -> save_name ${SAVE_NAME}"

# ---- Stage 1: training (yaml + per-run overrides) ----
TRAIN_CMD=(
    srun
    python -m prism.training.train_v2
    "$CONFIG"
    --save_name "$SAVE_NAME"
    --wandb_project "$PROJECT"
    --wandb_run_name "$WANDB_RUN_NAME"
)
[[ -n "${EXPERIMENT_TAG:-}" ]] && TRAIN_CMD+=(--wandb_tag "$EXPERIMENT_TAG")
[[ -n "${CHECKPOINT_DIR:-}" ]] && TRAIN_CMD+=(--checkpoint_dir "$CHECKPOINT_DIR")
[[ -n "${NAME:-}" ]]          && TRAIN_CMD+=(--name "$NAME")
[[ -n "${DATA:-}" ]]          && TRAIN_CMD+=(--data "$DATA")
[[ -n "${VAL_DATA:-}" ]]      && TRAIN_CMD+=(--val_data "$VAL_DATA")
[[ -n "${EVAL_DATA:-}" ]]     && TRAIN_CMD+=(--eval_data "$EVAL_DATA")
[[ -n "${EXTRA_TRAIN_ARGS:-}" ]] && TRAIN_CMD+=(${EXTRA_TRAIN_ARGS})

log "TRAIN: ${TRAIN_CMD[*]}"
"${TRAIN_CMD[@]}"

# ---- Stage 2: locate the trained checkpoint ----
# train_v2 writes {checkpoint_dir}/{save_name}_{wandb_run_id}; SAVE_NAME is
# unique per run, so glob + newest mtime resolves it unambiguously. When
# CHECKPOINT_DIR is not overridden, read checkpoint_dir from the yaml.
CKPT_BASE=${CHECKPOINT_DIR:-$(python - "$CONFIG" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1])).get("checkpoint_dir", "outputs"))
PY
)}
CHECKPOINT=$(ls -dt "${CKPT_BASE}/${SAVE_NAME}"_*/ 2>/dev/null | head -1 || true)
CHECKPOINT="${CHECKPOINT%/}"
if [[ -z "$CHECKPOINT" || ! -d "$CHECKPOINT" ]]; then
    log "ERROR: no checkpoint found matching ${CKPT_BASE}/${SAVE_NAME}_*"
    exit 1
fi
log "Checkpoint: ${CHECKPOINT}"

# ---- Stage 3: cross-eval on the held-out test graphs ----
# eval_checkpoint_on_graphs.py auto-detects architecture (llm/rpearl/gt) from
# the checkpoint and writes results to {checkpoint}/eval_logs/cross_eval/.
EVAL_CMD=(
    srun
    python scripts/eval_checkpoint_on_graphs.py
    --checkpoint "$CHECKPOINT"
    --graphs "$TEST_GRAPHS"
)
log "EVAL: ${EVAL_CMD[*]}"
"${EVAL_CMD[@]}"

log "Done: ${SAVE_NAME} -> ${CHECKPOINT}/eval_logs/cross_eval/"
