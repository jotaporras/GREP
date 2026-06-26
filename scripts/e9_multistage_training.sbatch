#!/bin/bash
#SBATCH --job-name=e9-multistage
#SBATCH --output=/vast/home/j/jporras/sourcecode/GREP/slurm-%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#
# e9 multistage training — SELF-CONTAINED batch script (no external driver).
#
#   Stage 1  SFT the LLM (LoRA), PE frozen, edges in text            (1 epoch)
#   Stage 2  freeze SFT'd LLM, train ONLY the PE on edge-list loss    (many epochs)
#   Stage 3  unfreeze, train PE + LoRA jointly, edges removed         (1 epoch)
#
# Each stage is a SEPARATE train_v3 run (fresh optimizer / epoch budget / freeze
# regime), driven by a Hydra config (experiments/e9_ms_stageN.yaml inheriting
# experiments/base_config.yaml). Weights carry forward via trainer.init_lora_from /
# trainer.init_pe_from (weight-only, NOT HF resume). The Stage-2 checkpoint is
# self-contained (frozen adapter + trained PE), so Stage 3 inits BOTH from it.
#
# By default runs Stages 1+2 then STOPS and prints the Stage-3 command (inspect the
# Stage-2 eval_logs first). Set RUN_STAGE3=1 to chain all three in one job.
#
# Submit:
#   sbatch scripts/e9_multistage_training.sbatch                # stages 1+2
#   RUN_STAGE3=1 sbatch scripts/e9_multistage_training.sbatch   # all three
#
# Each stage launches exactly like the working e8 pipeline (scripts/train_and_eval.sh):
# `srun python -m prism.training.train_v3 ...` with SRUN_CPUS_PER_TASK exported
# (SLURM 22.05+ no longer propagates --cpus-per-task to steps automatically). `srun`
# is required so the step gets the GPU cgroup/device binding — running python bare
# leaves the device unbound and CUDA reports "device busy or unavailable".

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

# SLURM 22.05+ no longer propagates --cpus-per-task to srun steps automatically.
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-14}"

PROJECT=${PROJECT:-GREP-PRISM}
ENTITY=${ENTITY:-alelab}
export WANDB_ENTITY=${WANDB_ENTITY:-$ENTITY}
EXPERIMENT_TAG=${EXPERIMENT_TAG:-e9_multistage_training}
ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/outputs/e9_multistage_training}

module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

log() { echo "[$(date --iso-8601=seconds)] $*"; }

# Resolve {CHECKPOINT_DIR}/{save_name}_{wandb_id}: save_name is unique per stage, so
# the newest-mtime glob is unambiguous.
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

# Each stage can be SKIPPED to reuse an existing checkpoint instead of recomputing
# it (the carried weights are the same either way):
#   SKIP_STAGE1=1  reuse a Stage-1 checkpoint, start at Stage 2
#   SKIP_STAGE2=1  reuse a Stage-2 checkpoint, start at Stage 3 (implies skipping 1)
# Each skipped stage's dir is located by resolve_ckpt (newest-mtime glob) UNLESS
# pinned explicitly via STAGE1_DIR=<dir> / STAGE2_DIR=<dir>. Skipping Stage 2 also
# skips Stage 1 (Stage 3 only needs the Stage-2 checkpoint), so STAGE1_DIR is never
# required in that case.
SKIP_STAGE2=${SKIP_STAGE2:-0}
SKIP_STAGE1=${SKIP_STAGE1:-0}
[[ "$SKIP_STAGE2" == "1" ]] && SKIP_STAGE1=1

# ---- Stage 2: PE-only on edge-list loss (LLM+adapter frozen) ----
if [[ "$SKIP_STAGE2" == "1" ]]; then
    log "===== Stages 1+2: SKIPPED (SKIP_STAGE2=1) — reusing existing Stage-2 checkpoint ====="
else
    # ---- Stage 1: SFT (LoRA, PE frozen) ----
    if [[ "$SKIP_STAGE1" == "1" ]]; then
        log "===== Stage 1: SKIPPED (SKIP_STAGE1=1) — reusing existing checkpoint ====="
    else
        log "===== Stage 1: SFT (LoRA, PE frozen) ====="
        run_stage e9_ms_stage1 e9_ms_s1
    fi
    STAGE1_DIR=${STAGE1_DIR:-$(resolve_ckpt e9_ms_s1)}
    log "Stage 1 checkpoint: $STAGE1_DIR"

    log "===== Stage 2: PE-only (LLM+adapter frozen, edge-list loss) ====="
    run_stage e9_ms_stage2 e9_ms_s2 trainer.init_lora_from="$STAGE1_DIR"
fi
STAGE2_DIR=${STAGE2_DIR:-$(resolve_ckpt e9_ms_s2)}
log "Stage 2 checkpoint: $STAGE2_DIR"

# ---- Stage 3: joint PE+LoRA, edges removed ----
if [[ "${RUN_STAGE3:-0}" == "1" ]]; then
    log "===== Stage 3: joint PE+LoRA, edges removed ====="
    run_stage e9_ms_stage3 e9_ms_s3 \
        trainer.init_lora_from="$STAGE2_DIR" trainer.init_pe_from="$STAGE2_DIR"
    log "Stage 3 checkpoint: $(resolve_ckpt e9_ms_s3)"
    log "Done."
else
    log "Stage 2 done. Inspect Stage-2 eval_logs, then submit Stage 3 as a batch job:"
    cat <<EOF

  SKIP_STAGE2=1 RUN_STAGE3=1 \\
  STAGE2_DIR='${STAGE2_DIR}' \\
  sbatch scripts/e9_multistage_training.sbatch

  (Submit as sbatch — NOT a bare srun on the login node, which lands on a CPU
   partition with no conda env. This reuses the Stage-2 checkpoint above and runs
   only Stage 3.)
EOF
fi
