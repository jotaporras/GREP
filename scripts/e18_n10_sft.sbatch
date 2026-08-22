#!/bin/bash
#SBATCH --job-name=e18-n10
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%j.out
#SBATCH --time=02:30:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
# betty's SLURM cli_filter (2026-08-03) enforces EXACTLY 28 CPUs / 224 GB per
# GPU on dgx-b200.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#
# e18 — node-identity dev loop on the n~10 corpus (docs/2026-08-21
# e18_n10_identity_plan.md). One job = one ARM of the e17 mask recipe
# (e17_pf_sft.sbatch verbatim: 31B-it, stage-1 LoRA warm start, navigator-GT
# tower, mask_alpha 0, binary edges, decode_consistent, slr 0.012) with the
# e18 flags toggled, capped at MAX_STEPS optimizer steps so a run fits ~30 min
# of training, then: (1) held-out navigation eval on test_graphs, (2) the
# neighbour-naming probe on test_graphs and a few train graphs.
#
#   ARM=mask         control: the e17 mask, nothing new
#   ARM=mask_a       + decision gating (A; diagnostic — e13b decode_trail prior
#                      says the zero-shot hard version did nothing)
#   ARM=mask_b       + structural key channel (B; the candidate)
#   ARM=mask_ab      + both
#   ARM=mask_bind    + binding auxiliary loss only
#   ARM=mask_b_bind  + B + binding
#   ARM=mask_a_bind  + A + binding
#   ARM=mask_d       + soft edge tokens (D; graph-side upper bound — one learned
#                      embedding per directed edge spliced in after BOS)
#   ARM=text_edges   plain-LLM upper bound: edge list in text, fresh LoRA
#
# Submit:  ARM=mask_b sbatch scripts/e18_n10_sft.sbatch
#          ARM=mask MAX_STEPS=40 RUN_NAME=e18_calib sbatch scripts/e18_n10_sft.sbatch   # s/step calibration
# Override DATA_SPLIT for the n15 corpus.
#
# Full-scale n60_v3 (the e17 control cn7ub88q: 3 epochs = 936 steps, per-epoch
# built-in eval on 7 test graphs x 12 tasks, 72.6% at epoch 3, ~3 h):
#   ARM=mask_a MAX_STEPS=-1 EVAL_EPOCH_INTERVAL=1 PROBE_TEST_GRAPHS=3 \
#     DATA_SPLIT=$PROJ/data/n_60_vllm_v3/gen/nav_n60_gemma_data/split RUN_NAME=e18_n60_mask_a \
#     sbatch --time=08:00:00 scripts/e18_n10_sft.sbatch
# MAX_STEPS=-1 = no cap (trainer.epochs governs). PROBE_TEST_GRAPHS caps the
# test-graph probe (48 rooms x 7 graphs = 336 queries at ~12 s each otherwise).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

ALELAB=/vast/projects/aribeiro/alelab/jporras
PROJ=$ALELAB/GREP-PRISM
ENV_NAME=${ENV_NAME:-$ALELAB/envs/GREP-PRISM-v3}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-$PROJ/outputs/e18_identity}
PE_DIR=$PROJ/pretrained_pe
STAGE1_LORA=$PROJ/outputs/e9_multistage_training/e9_ms_stage1/e9_ms_stage1_sqgk4o3j/checkpoint-100

ARM="${ARM:?set ARM=mask|mask_a|mask_b|mask_ab|mask_bind|mask_b_bind|mask_a_bind|mask_d|text_edges (scripts/e18_arms.sh)}"
DATA_SPLIT=${DATA_SPLIT:-$PROJ/data/n_10_vllm/gen/nav_n10_gemma_data/split}
MAX_STEPS=${MAX_STEPS:-300}
RUN_NAME=${RUN_NAME:-e18_n10_${ARM}}
TAG=${TAG:-e18_identity}
PROBE_TRAIN_GRAPHS=${PROBE_TRAIN_GRAPHS:-3}
PROBE_TEST_GRAPHS=${PROBE_TEST_GRAPHS:--1}      # -1 = every test graph
EVAL_EPOCH_INTERVAL=${EVAL_EPOCH_INTERVAL:-999} # 1 = built-in generation eval at each epoch end (e17 harness)
PROBE_OUT=${PROBE_OUT:-$PROJ/results/e18_probe}

export WANDB_ENTITY=${WANDB_ENTITY:-alelab}
export HF_HOME=${HF_HOME:-$ALELAB/hf_cache}
export HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf_datasets_${SLURM_JOB_ID:-manual}"
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-28}"
# PATH prepend, not `conda activate`: compute-node batch shells have no conda
# (e17 jobs 7639051-53 died exit 127).
export PATH="$ENV_NAME/bin:$PATH"

test -f "$DATA_SPLIT/formatted_all_new__train.json" || { echo "missing $DATA_SPLIT/formatted_all_new__train.json"; exit 2; }

# Arm table lives in scripts/e18_arms.sh (shared with the plaza runner).
PE_GT="$PE_DIR/path_navigator_gt.pt"
source scripts/e18_arms.sh          # cwd is the repo root (cd above)
e18_arm_args "$ARM"

echo "[$(date --iso-8601=seconds)] $RUN_NAME arm=$ARM data=$DATA_SPLIT max_steps=$MAX_STEPS tag=$TAG"
echo "=== python: $(which python) ==="

# eval.epoch_interval=999 (default): no in-training generation eval (the
# step-based HF eval at max_steps//2 is the cheap val loss only); the held-out
# navigation eval runs once post-training from the saved checkpoint
# (post_train_graphs). EVAL_EPOCH_INTERVAL=1 adds the per-epoch generation eval
# the e17 n60 control used, so epoch-3 numbers are directly comparable.
srun python -m prism.training.train_v3 \
    --config-name=e9_base_config \
    model.path="google/gemma-4-31B-it" \
    "${ARM_ARGS[@]}" \
    data.train_files="$DATA_SPLIT/formatted_all_new__train.json" \
    data.val_files="$DATA_SPLIT/formatted_all_new__val.json" \
    data.edge_weights=binary \
    trainer.freeze_lora=false trainer.freeze_pe=false \
    trainer.learning_rate=0.00025 trainer.epochs=3 trainer.max_steps="$MAX_STEPS" \
    +trainer.sft.save_total_limit=1 \
    trainer.checkpoint_dir="$CHECKPOINT_DIR" \
    trainer.save_name="$RUN_NAME" \
    wandb.run_name="$RUN_NAME" wandb.tag="$TAG" \
    eval.data="$DATA_SPLIT/test_graphs" eval.num_graphs=-1 eval.epoch_interval="$EVAL_EPOCH_INTERVAL" \
    eval.post_train_graphs="$DATA_SPLIT/test_graphs"

# The run dir is <CHECKPOINT_DIR>/<RUN_NAME>_<wandb_id>/ (train_v3
# _construct_output_dir) — take the newest one matching this run name.
# Exactly <RUN_NAME>_<8-char wandb id>: a bare "${RUN_NAME}_*" also matches sibling
# arms (e18_n10_mask_* ⊃ e18_n10_mask_d_*, mask_b_* ⊃ mask_b_bind_*) and picks
# whichever was written last — 7731157 probed the mask_d checkpoint as "mask".
RUN_DIR=$(ls -dt "$CHECKPOINT_DIR/${RUN_NAME}"_???????? | head -1)
test -n "$RUN_DIR" || { echo "no run dir ${RUN_NAME}_<wandb id> under $CHECKPOINT_DIR"; exit 2; }
echo "[$(date --iso-8601=seconds)] run dir: $RUN_DIR"

# --- neighbour-naming probe: held-out graphs, then a few TRAIN graphs ---------
mkdir -p "$PROBE_OUT"
srun python scripts/neighbour_probe.py --checkpoint "$RUN_DIR" \
    --graphs "$DATA_SPLIT/test_graphs" --max-graphs "$PROBE_TEST_GRAPHS" \
    --output "$PROBE_OUT/${RUN_NAME}_test.json"
srun python scripts/neighbour_probe.py --checkpoint "$RUN_DIR" \
    --graphs "$DATA_SPLIT/train_graphs" --max-graphs "$PROBE_TRAIN_GRAPHS" \
    --output "$PROBE_OUT/${RUN_NAME}_train.json"

echo "[$(date --iso-8601=seconds)] $RUN_NAME DONE"
