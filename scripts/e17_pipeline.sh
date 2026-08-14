#!/usr/bin/env bash
# e17 pipeline: notebook -> MagGT checkpoint -> Stage 3 joint training.
#
#   bash scripts/e17_pipeline.sh                      # full run
#   E17_SMOKE=1 bash scripts/e17_pipeline.sh          # plumbing test, full strength
#
# The notebook is the single source: the script is REGENERATED from it every run
# (nbconvert) and patched for headless execution by scripts/_e17_headless.py, so the two
# cannot drift. §2 pretrains the MagGT on edge existence and §3 fine-tunes it on the
# covariance metric; both write $SUITE/mag_gt.pt, which Stage 3 loads via gnn.pe_gt_from.
#
# Stage 2 is NOT run: the leak-free encoder pretraining for this architecture is the
# notebook, and Stage 3 starts from it with a fresh LoRA. Supply INIT_LORA/INIT_PE below
# only when a Stage-2 run exists to carry forward.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO="$PWD"

# E17_SMOKE=1 shortens the RUN, never the CONFIGURATION: one sample per split, one plan
# each, one epoch per notebook stage, one gradient step, one eval graph. Everything that
# defines the model — M=320, d_model, the composite-graph weights, beta_init — stays at
# its experiments/ value, so a green smoke means the full-strength arm constructs, trains
# and generates. It does NOT mean anything learned: at one epoch both notebook loops
# validate at i=0, take that untrained snapshot as `best_state`, and restore it at the
# end, so the mag_gt.pt a smoke writes holds UNTRAINED weights. Each knob is still
# individually overridable — the assignments below only fill what the caller left unset.
if [ "${E17_SMOKE:-0}" = "1" ]; then
  : "${E17_TRAIN_SAMPLES:=1}" "${E17_VAL_SAMPLES:=1}" "${E17_TEST_SAMPLES:=1}"
  : "${E17_PLANS_PER_GRAPH:=1}" "${E17_EDGE_EPOCHS:=1}" "${E17_RES_EPOCHS:=1}"
  : "${E17_STAGE3_MAX_STEPS:=1}" "${E17_EVAL_GRAPHS:=1}"
  export E17_TRAIN_SAMPLES E17_VAL_SAMPLES E17_TEST_SAMPLES E17_PLANS_PER_GRAPH \
         E17_EDGE_EPOCHS E17_RES_EPOCHS
fi

SUITE="${E17_SUITE:-suite2}"
NOTEBOOK="notebooks/2026-08-10 e17_magnet_composite_graphs.ipynb"
SCRIPT="scripts/e17_magnet_composite_graphs.py"
CKPT="outputs/e17_mag_gt/${SUITE}/mag_gt.pt"
STAGE3_EPOCHS="${E17_STAGE3_EPOCHS:-3}"
# Smoke: E17_STAGE3_MAX_STEPS=1 runs ONE gradient step (train_v3 switches to step-based
# save/eval when max_steps > 0). 0 = epoch-based, i.e. the real run.
STAGE3_MAX_STEPS="${E17_STAGE3_MAX_STEPS:-0}"
# W&B must not block an unattended run; override with E17_WANDB_MODE=online.
export WANDB_MODE="${E17_WANDB_MODE:-offline}"
export E17_SUITE="$SUITE"

echo "=== [1/3] notebook -> script ($(date +%H:%M:%S))"
jupyter nbconvert --to script --output-dir=scripts \
        --output "$(basename "$SCRIPT" .py)" "$NOTEBOOK"
python scripts/_e17_headless.py "$SCRIPT"

echo "=== [2/3] pretrain the MagGT: §2 edges, §3 covariance -> ${CKPT}"
# Run FROM notebooks/ so the notebook's own ../data and ../outputs paths resolve.
( cd notebooks && python "../${SCRIPT}" )
test -f "$CKPT" || { echo "FATAL: ${CKPT} was not written"; exit 1; }
echo "    wrote: $(ls -la "$CKPT" | awk '{print $5" bytes"}')"

echo "=== [3/3] Stage 3: joint LoRA + graph channel, MagGT from ${CKPT}"
python -m prism.training.train_v3 --config-name=e17_ms_stage3 \
    model.path=google/gemma-4-31B-it \
    gnn.pe_gt_from="$CKPT" \
    gnn.semantic_gt_from=null \
    ${E17_NUM_SAMPLES:+gnn.num_samples=$E17_NUM_SAMPLES} \
    ${E17_EVAL_GRAPHS:+eval.num_graphs=$E17_EVAL_GRAPHS} \
    trainer.epochs="$STAGE3_EPOCHS" \
    trainer.max_steps="$STAGE3_MAX_STEPS" \
    trainer.checkpoint_dir="outputs/e17_mag_gt/${SUITE}" \
    trainer.save_name="e17_ms_stage3_${SUITE}" \
    ${INIT_LORA:+trainer.init_lora_from=$INIT_LORA} \
    ${INIT_PE:+trainer.init_pe_from=$INIT_PE} \
    wandb.run_name="e17_ms_stage3_${SUITE}" \
    "$@"

echo "=== done ($(date +%H:%M:%S)); Stage 3 under outputs/e17_mag_gt/${SUITE}"
