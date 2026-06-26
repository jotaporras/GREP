#!/bin/bash
# e9_multistage_training.sh — chained multistage training driver.
#
#   Stage 1  SFT the LLM (LoRA), PE frozen, edges in text          (1 epoch)
#   Stage 2  freeze SFT'd LLM, train ONLY the PE on edge-list loss  (many epochs)
#   --- analyze Stage 2 here, then ---
#   Stage 3  unfreeze, train PE + LoRA jointly, edges removed       (1 epoch)
#
# Each stage is a SEPARATE `train_v3` invocation (fresh optimizer / epoch budget /
# freeze regime). Each stage is a Hydra config (experiments/e9_ms_stageN.yaml) inheriting
# experiments/base_config.yaml. Weights carry forward via trainer.init_lora_from /
# trainer.init_pe_from (weight-only, NOT HF resume). The Stage-2 checkpoint is
# self-contained (frozen adapter + trained PE), so Stage 3 inits BOTH from it.
#
# By default this runs Stages 1 and 2 then STOPS and prints the Stage-3 command, so
# you can inspect Stage-2 eval_logs first. Set RUN_STAGE3=1 to also run Stage 3.
#
# Usage:
#   bash scripts/e9_multistage_training.sh                 # stages 1+2, print stage-3 cmd
#   RUN_STAGE3=1 bash scripts/e9_multistage_training.sh    # all three
#   CHECKPOINT_DIR=/vast/.../outputs/e9_multistage_training bash scripts/e9_multistage_training.sh
#
# Optional env: PROJECT, EXPERIMENT_TAG, ENV_NAME, CHECKPOINT_DIR, EXTRA_TRAIN_ARGS.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-14}"

PROJECT=${PROJECT:-GREP-PRISM}
EXPERIMENT_TAG=${EXPERIMENT_TAG:-e9_multistage_training}
ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3}

# Checkpoint root: env override (cluster artifact root) else the base config's value.
CHECKPOINT_DIR=${CHECKPOINT_DIR:-$(python - experiments/base_config.yaml <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["trainer"]["checkpoint_dir"])
PY
)}

module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

log() { echo "[$(date --iso-8601=seconds)] $*"; }

# Resolve {CHECKPOINT_DIR}/{save_name}_{wandb_id} — save_name is unique per stage,
# so newest-mtime glob is unambiguous (mirrors scripts/train_and_eval.sh).
resolve_ckpt() {
    local save_name="$1" dir
    dir=$(ls -dt "${CHECKPOINT_DIR}/${save_name}"_*/ 2>/dev/null | head -1 || true)
    dir="${dir%/}"
    [[ -n "$dir" && -d "$dir" ]] || { log "ERROR: no checkpoint matching ${CHECKPOINT_DIR}/${save_name}_*"; exit 1; }
    echo "$dir"
}

run_stage() {  # run_stage <config-name> <save_name> [extra Hydra overrides...]
    local cfg="$1" save_name="$2"; shift 2
    local cmd=(srun python -m prism.training.train_v3 --config-name="$cfg"
        trainer.save_name="$save_name"
        wandb.project="$PROJECT"
        wandb.run_name="$save_name"
        wandb.tag="$EXPERIMENT_TAG"
        trainer.checkpoint_dir="$CHECKPOINT_DIR"
        "$@")
    [[ -n "${EXTRA_TRAIN_ARGS:-}" ]] && cmd+=(${EXTRA_TRAIN_ARGS})
    log "TRAIN: ${cmd[*]}"
    "${cmd[@]}"
}

# ---- Stage 1: SFT ----
log "===== Stage 1: SFT (LoRA, PE frozen) ====="
run_stage e9_ms_stage1 e9_ms_s1
STAGE1_DIR=$(resolve_ckpt e9_ms_s1)
log "Stage 1 checkpoint: $STAGE1_DIR"

# ---- Stage 2: PE-only on edge-list loss ----
log "===== Stage 2: PE-only (LLM+adapter frozen, edge-list loss) ====="
run_stage e9_ms_stage2 e9_ms_s2 trainer.init_lora_from="$STAGE1_DIR"
STAGE2_DIR=$(resolve_ckpt e9_ms_s2)
log "Stage 2 checkpoint: $STAGE2_DIR"

# ---- Stage 3: joint, edges removed ----
if [[ "${RUN_STAGE3:-0}" == "1" ]]; then
    log "===== Stage 3: joint PE+LoRA, edges removed ====="
    run_stage e9_ms_stage3 e9_ms_s3 \
        trainer.init_lora_from="$STAGE2_DIR" trainer.init_pe_from="$STAGE2_DIR"
    STAGE3_DIR=$(resolve_ckpt e9_ms_s3)
    log "Stage 3 checkpoint: $STAGE3_DIR"
    log "Done. All three stages complete."
else
    log "Stages 1+2 done. Inspect Stage-2 eval_logs, then run Stage 3 with:"
    cat <<EOF

  CHECKPOINT_DIR='${CHECKPOINT_DIR}' \\
  srun python -m prism.training.train_v3 --config-name=e9_ms_stage3 \\
      trainer.save_name=e9_ms_s3 wandb.project='${PROJECT}' wandb.run_name=e9_ms_s3 \\
      wandb.tag='${EXPERIMENT_TAG}' trainer.checkpoint_dir='${CHECKPOINT_DIR}' \\
      trainer.init_lora_from='${STAGE2_DIR}' trainer.init_pe_from='${STAGE2_DIR}'

  (or re-run this script with RUN_STAGE3=1 to chain all three.)
EOF
fi
