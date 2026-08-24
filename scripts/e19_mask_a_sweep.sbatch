#!/bin/bash
#SBATCH --job-name=e19-sweep
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%j.out
#SBATCH --time=06:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
# betty's SLURM cli_filter (2026-08-03) enforces EXACTLY 28 CPUs / 224 GB per
# GPU on dgx-b200.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#
# e19 mask_a hparam sweep — one-factor-at-a-time star around the e18 mask_a
# recipe (9a6lwgfj), n60_v3, 3 epochs. Two seed replicas anchor the noise band
# (base seed is train_v3's hard-coded 3407); every other run changes exactly
# one knob. 6-epoch arms were considered and rejected (user, 2026-08-24).
#
#   SWEEP=base_rep1   seed 17 only (replica)
#   SWEEP=base_rep2   seed 1031 only (replica)
#   SWEEP=lr_low      lr 2.5e-4 -> 1e-4
#   SWEEP=lr_high     lr 2.5e-4 -> 5e-4
#   SWEEP=slr_005     structural_lr_mult 0.012 -> 0.05
#   SWEEP=slr_02      structural_lr_mult 0.012 -> 0.2
#   SWEEP=alpha_03    mask_alpha 0.0 -> 0.3
#   SWEEP=scope_top_half  mask_layer_scope dense -> dense_top_half (scope_all is
#                  INVALID: decode_consistent needs a dense-family scope — the
#                  sliding-window KV crop guard in train_v3 rejects it, job 7822683)
#   SWEEP=dg_6        decision_gain_init 3.0 -> 6.0
#
# Submit:  SWEEP=slr_005 sbatch scripts/e19_mask_a_sweep.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

ALELAB=/vast/projects/aribeiro/alelab/jporras
PROJ=$ALELAB/GREP-PRISM
ENV_NAME=${ENV_NAME:-$ALELAB/envs/GREP-PRISM-v3}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-$PROJ/outputs/e19_mask_a_sweep}
PE_DIR=$PROJ/pretrained_pe
STAGE1_LORA=$PROJ/outputs/e9_multistage_training/e9_ms_stage1/e9_ms_stage1_sqgk4o3j/checkpoint-100

SWEEP="${SWEEP:?set SWEEP=base_rep1|base_rep2|lr_low|lr_high|slr_005|slr_02|alpha_03|scope_top_half|dg_6}"
DATA_SPLIT=${DATA_SPLIT:-$PROJ/data/n_60_vllm_v3/gen/nav_n60_gemma_data/split}
RUN_NAME=${RUN_NAME:-e19_swp_${SWEEP}}
TAG=${TAG:-e19_mask_a_sweep}
PROBE_TRAIN_GRAPHS=${PROBE_TRAIN_GRAPHS:-3}
PROBE_TEST_GRAPHS=${PROBE_TEST_GRAPHS:-3}
PROBE_OUT=${PROBE_OUT:-$PROJ/results/e19_probe}

export WANDB_ENTITY=${WANDB_ENTITY:-alelab}
export HF_HOME=${HF_HOME:-$ALELAB/hf_cache}
export HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf_datasets_${SLURM_JOB_ID:-manual}"
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-28}"
# PATH prepend, not `conda activate`: compute-node batch shells have no conda.
export PATH="$ENV_NAME/bin:$PATH"
# Fragmentation guard (e19 v1 epoch-1 eval OOMs).
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

test -f "$DATA_SPLIT/formatted_all_new__train.json" || { echo "missing $DATA_SPLIT/formatted_all_new__train.json"; exit 2; }

PE_GT="$PE_DIR/path_navigator_gt.pt"
source scripts/e18_arms.sh          # cwd is the repo root (cd above)
e18_arm_args mask_a                 # the 9a6lwgfj recipe, verbatim

# One knob per run. Hydra rejects duplicate overrides, so knobs already inside
# ARM_ARGS are swapped IN PLACE via bash pattern substitution (the patterns
# match e18_arms.sh's literal spellings — keep them in sync).
LR=0.00025
SEED_ARGS=()
case "$SWEEP" in
  base_rep1) SEED_ARGS=(+trainer.sft.seed=17) ;;
  base_rep2) SEED_ARGS=(+trainer.sft.seed=1031) ;;
  lr_low)    LR=0.0001 ;;
  lr_high)   LR=0.0005 ;;
  slr_005)   ARM_ARGS=("${ARM_ARGS[@]/gnn.structural_lr_mult=0.012/gnn.structural_lr_mult=0.05}") ;;
  slr_02)    ARM_ARGS=("${ARM_ARGS[@]/gnn.structural_lr_mult=0.012/gnn.structural_lr_mult=0.2}") ;;
  alpha_03)  ARM_ARGS=("${ARM_ARGS[@]/gnn.mask_alpha=0.0/gnn.mask_alpha=0.3}") ;;
  scope_top_half) ARM_ARGS=("${ARM_ARGS[@]/gnn.mask_layer_scope=dense/gnn.mask_layer_scope=dense_top_half}") ;;
  dg_6)      ARM_ARGS=("${ARM_ARGS[@]/gnn.decision_gain_init=3.0/gnn.decision_gain_init=6.0}") ;;
  *) echo "unknown SWEEP=$SWEEP" >&2; exit 2 ;;
esac

echo "[$(date --iso-8601=seconds)] $RUN_NAME sweep=$SWEEP lr=$LR data=$DATA_SPLIT tag=$TAG"
echo "=== python: $(which python) ==="

srun python -m prism.training.train_v3 \
    --config-name=e9_base_config \
    model.path="google/gemma-4-31B-it" \
    "${ARM_ARGS[@]}" \
    ${SEED_ARGS[@]+"${SEED_ARGS[@]}"} \
    data.train_files="$DATA_SPLIT/formatted_all_new__train.json" \
    data.val_files="$DATA_SPLIT/formatted_all_new__val.json" \
    data.edge_weights=binary \
    trainer.freeze_lora=false trainer.freeze_pe=false \
    trainer.learning_rate="$LR" trainer.epochs=3 trainer.max_steps=-1 \
    +trainer.sft.save_total_limit=1 \
    trainer.checkpoint_dir="$CHECKPOINT_DIR" \
    trainer.save_name="$RUN_NAME" \
    wandb.run_name="$RUN_NAME" wandb.tag="$TAG" \
    eval.data="$DATA_SPLIT/test_graphs" eval.num_graphs=-1 eval.epoch_interval=1 \
    eval.post_train_graphs="$DATA_SPLIT/test_graphs"

RUN_DIR=$(ls -dt "$CHECKPOINT_DIR/${RUN_NAME}"_???????? | head -1)
test -n "$RUN_DIR" || { echo "no run dir ${RUN_NAME}_<wandb id> under $CHECKPOINT_DIR"; exit 2; }
echo "[$(date --iso-8601=seconds)] run dir: $RUN_DIR"

mkdir -p "$PROBE_OUT"
srun python scripts/neighbour_probe.py --checkpoint "$RUN_DIR" \
    --graphs "$DATA_SPLIT/test_graphs" --max-graphs "$PROBE_TEST_GRAPHS" \
    --output "$PROBE_OUT/${RUN_NAME}_test.json"
srun python scripts/neighbour_probe.py --checkpoint "$RUN_DIR" \
    --graphs "$DATA_SPLIT/train_graphs" --max-graphs "$PROBE_TRAIN_GRAPHS" \
    --output "$PROBE_OUT/${RUN_NAME}_train.json"

echo "[$(date --iso-8601=seconds)] $RUN_NAME DONE"
