#!/usr/bin/env bash
# e17 pipeline: notebook -> MagE-GT checkpoint -> Stage 3 joint training.
#
#   bash scripts/e17_pipeline.sh                      # full run
#   E17_SMOKE=1 bash scripts/e17_pipeline.sh          # plumbing test, full strength
#   E17_STAGE1=1 bash scripts/e17_pipeline.sh         # (re)train the Stage-1 baseline
#
# THREE suites, because the MagE-GT and the LoRA have different lifecycles:
#   E17_MAGGT_LOAD_SUITE  (suite3)  the notebook READS this, and Stage 3 takes its MagE-GT
#   E17_MAGGT_SAVE_SUITE  (suite4)  the notebook WRITES here, so a rerun cannot clobber
#                                   the finished MagE-GT it is loading from
#   E17_LORA_SUITE        (suite3)  Stage 1 / Stage 3 checkpoints land here
#
# The notebook is the single source: the script is REGENERATED from it every run
# (nbconvert) and patched for headless execution by scripts/_e17_headless.py, so the two
# cannot drift. §2 pretrains the MagE-GT on edge existence and §3 fine-tunes it on the
# covariance metric; both write $SUITE/mag_gt.pt, which Stage 3 loads via gnn.pe_gt_from.
#
# Stage 1 is the BASELINE adapter: LoRA SFT with the PE frozen and the edges still in the
# text. It is trained ONCE and REUSED, so it is OFF by default (E17_STAGE1=1 retrains it).
# Either way Stage 3 starts from it: the adapter is discovered under the LoRA save dir and
# passed as trainer.init_lora_from. Stage 3 NEVER runs on an un-fine-tuned LoRA — if no
# baseline is found the pipeline stops rather than silently training from scratch. Stage 2 is NOT run: the
# leak-free encoder pretraining for this architecture is the notebook, so Stage 3 takes
# the MagE-GT from there and the adapter from Stage 1. Set INIT_LORA to override.
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
  : "${E17_STAGE3_MAX_STEPS:=1}" "${E17_EVAL_GRAPHS:=1}" "${E17_STAGE1_MAX_STEPS:=1}"
  export E17_TRAIN_SAMPLES E17_VAL_SAMPLES E17_TEST_SAMPLES E17_PLANS_PER_GRAPH \
         E17_EDGE_EPOCHS E17_RES_EPOCHS
fi

# The Stage-1 baseline: trained once and reused, so OFF unless explicitly requested.
E17_STAGE1="${E17_STAGE1:-0}"
# The MagE-GT is FINISHED and only read; the LoRA is still being written. Separate suites so
# a notebook rerun cannot overwrite the MagE-GT that Stage 3 depends on.
MAGGT_SAVE_SUITE="${E17_MAGGT_SAVE_SUITE:-suite4}"
MAGGT_LOAD_SUITE="${E17_MAGGT_LOAD_SUITE:-suite3}"
LORA_SUITE="${E17_LORA_SUITE:-suite3}"
SUITE="$LORA_SUITE"          # every LLM checkpoint path below is the LoRA suite
NOTEBOOK="notebooks/2026-08-10 e17_magnet_composite_graphs.ipynb"
SCRIPT="scripts/e17_magnet_composite_graphs.py"
# Stage 3's MagE-GT comes from the LOAD suite, not from whatever this launch's notebook
# wrote. Set E17_PE_GT_FROM to consume a freshly trained one instead.
CKPT="${E17_PE_GT_FROM:-outputs/e17_mag_gt/${MAGGT_LOAD_SUITE}/mag_gt.pt}"
STAGE3_EPOCHS="${E17_STAGE3_EPOCHS:-3}"
# §3 keeps TWO M=320 autograd graphs alive per sample (probe_covariance for C_tok, plus
# the detector's cached_pe), and c varies 270-808 across samples, so the caching allocator
# fragments badly: the OOM that killed a full run reported 14.5 GiB allocated against 8.0
# GiB reserved-but-unallocated. expandable_segments lets those blocks be reused.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
# BOTH cards for the LLM stages: trainer.device=-1 makes loaders use device_map="auto",
# which shards the 31B NF4 base across every VISIBLE GPU. Pinning to one card is what made
# eval OOM at 2500-token prompts, so leave both visible unless the caller says otherwise.
# (The notebook half is single-GPU regardless — it uses device='cuda', the first visible.)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
# W&B must not block an unattended run; override with E17_WANDB_MODE=online.
export WANDB_MODE="${E17_WANDB_MODE:-offline}"
export E17_SAVE_SUITE="$MAGGT_SAVE_SUITE"
export E17_LOAD_SUITE="$MAGGT_LOAD_SUITE"

echo "=== [1/4] notebook -> script ($(date +%H:%M:%S))"
jupyter nbconvert --to script --output-dir=scripts \
        --output "$(basename "$SCRIPT" .py)" "$NOTEBOOK"
uv run scripts/_e17_headless.py "$SCRIPT"

echo "=== [2/4] notebook: MagE-GT reads ${MAGGT_LOAD_SUITE}, writes ${MAGGT_SAVE_SUITE}"
# Run FROM notebooks/ so the notebook's own ../data and ../outputs paths resolve.
( cd notebooks && uv run "../${SCRIPT}" )
test -f "$CKPT" || { echo "FATAL: no MagE-GT at ${CKPT} for Stage 3 to load"; exit 1; }
echo "    Stage-3 MagE-GT: $CKPT ($(ls -la "$CKPT" | awk '{print $5}') bytes)"

if [ "$E17_STAGE1" = "1" ]; then
echo "=== [3/4] Stage 1: SFT the LoRA, PE frozen, edges in text ($(date +%H:%M:%S))"
# No bit4 override needed: e17_ms_stage1 now inherits the e17 architecture, so it composes
# as the same MagCompGraphLLM Stage 3 trains (pe_pool=gt, M=320, bit4=true) with the graph
# channel frozen at beta=0. The adapter therefore carries into Stage 3 with no mismatch.
uv run -m prism.training.train_v3 --config-name=e17_ms_stage1 \
    model.path=google/gemma-4-31B-it \
    trainer.device=-1 \
    ${E17_STAGE1_MAX_STEPS:+trainer.max_steps=$E17_STAGE1_MAX_STEPS} \
    ${E17_EVAL_GRAPHS:+eval.num_graphs=$E17_EVAL_GRAPHS} \
    trainer.checkpoint_dir="outputs/e17_magnetic_composite_graphs/${SUITE}" \
    trainer.save_name="e17_ms_stage1_${SUITE}" \
    wandb.run_name="e17_ms_stage1_${SUITE}"

else
  echo "=== [3/4] Stage 1 not retrained (E17_STAGE1=0); reusing the existing baseline"
fi

# Resolve the baseline adapter, whether THIS launch trained it or an earlier one did.
# train_v3 appends the W&B id to save_name, so the directory is only knowable by search:
# prefer this suite, then any suite under the same checkpoint root, newest first.
LORA_ROOT="outputs/e17_magnetic_composite_graphs"
if [ -z "${INIT_LORA:-}" ]; then
  INIT_LORA="$(ls -dt ${LORA_ROOT}/${SUITE}/e17_ms_stage1_* 2>/dev/null | head -1 || true)"
fi
if [ -z "${INIT_LORA:-}" ]; then
  INIT_LORA="$(ls -dt ${LORA_ROOT}/*/e17_ms_stage1_* 2>/dev/null | head -1 || true)"
fi
# Stage 3 must NEVER start from an un-fine-tuned LoRA: stop rather than train from scratch.
test -n "${INIT_LORA:-}" || {
  echo "FATAL: no Stage-1 baseline adapter found under ${LORA_ROOT}/*/e17_ms_stage1_*"
  echo "       train one first:  E17_STAGE1=1 bash scripts/e17_pipeline.sh"
  exit 1; }
test -f "${INIT_LORA}/adapter_model.safetensors" || {
  echo "FATAL: ${INIT_LORA} holds no adapter_model.safetensors — not a usable baseline"
  exit 1; }
echo "    Stage-1 baseline adapter: $INIT_LORA"

echo "=== [4/4] Stage 3: joint LoRA + graph channel, MagE-GT from ${CKPT}"
uv run -m prism.training.train_v3 --config-name=e17_ms_stage3 \
    model.path=google/gemma-4-31B-it \
    trainer.device=-1 \
    gnn.pe_gt_from="$CKPT" \
    gnn.semantic_gt_from=null \
    ${E17_NUM_SAMPLES:+gnn.num_samples=$E17_NUM_SAMPLES} \
    ${E17_EVAL_GRAPHS:+eval.num_graphs=$E17_EVAL_GRAPHS} \
    trainer.epochs="$STAGE3_EPOCHS" \
    trainer.checkpoint_dir="outputs/e17_magnetic_composite_graphs/${SUITE}" \
    trainer.save_name="e17_ms_stage3_${SUITE}" \
    ${INIT_LORA:+trainer.init_lora_from=$INIT_LORA} \
    ${INIT_PE:+trainer.init_pe_from=$INIT_PE} \
    wandb.run_name="e17_ms_stage3_${SUITE}" \
    "$@"

echo "=== done ($(date +%H:%M:%S)); Stage 3 under outputs/e17_magnetic_composite_graphs/${SUITE}"
