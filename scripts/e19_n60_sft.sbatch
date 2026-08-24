#!/bin/bash
#SBATCH --job-name=e19-n60
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%j.out
#SBATCH --time=08:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
# betty's SLURM cli_filter (2026-08-03) enforces EXACTLY 28 CPUs / 224 GB per
# GPU on dgx-b200.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#
# e19 — hop-separated post-fusion channels on the full n60_v3 corpus
# (docs/2026-08-23 e19_hop_separated_fusion_design.md). One job = one ARM:
# the e18 mask_a (or mask_a_bind) recipe + per-hop fusion channels with the
# gates OPEN at init. Full 3-epoch runs (the cn7ub88q protocol: per-epoch
# built-in generation eval on the 7 test graphs), then the neighbour-naming
# probe.
#
#   ARM=hop_shift | hop_shift_k5 | hop_shift_bind        V1 (no-R-PEARL codebook GT)
#   ARM=hop_depth | hop_depth_bind | hop_depth_gain3     V2 (shared-GT block taps)
#
# Submit:  ARM=hop_shift sbatch scripts/e19_n60_sft.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

ALELAB=/vast/projects/aribeiro/alelab/jporras
PROJ=$ALELAB/GREP-PRISM
ENV_NAME=${ENV_NAME:-$ALELAB/envs/GREP-PRISM-v3}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-$PROJ/outputs/e19_hopfusion}
PE_DIR=$PROJ/pretrained_pe
STAGE1_LORA=$PROJ/outputs/e9_multistage_training/e9_ms_stage1/e9_ms_stage1_sqgk4o3j/checkpoint-100

ARM="${ARM:?set ARM=hop_shift|hop_shift_k5|hop_shift_bind|hop_depth|hop_depth_bind|hop_depth_gain3 (scripts/e19_arms.sh)}"
DATA_SPLIT=${DATA_SPLIT:-$PROJ/data/n_60_vllm_v3/gen/nav_n60_gemma_data/split}
MAX_STEPS=${MAX_STEPS:--1}                      # -1 = no cap (trainer.epochs governs)
RUN_NAME=${RUN_NAME:-e19_n60_${ARM}}
TAG=${TAG:-e19_hopfusion}
PROBE_TRAIN_GRAPHS=${PROBE_TRAIN_GRAPHS:-3}
PROBE_TEST_GRAPHS=${PROBE_TEST_GRAPHS:-3}       # cap: 48 rooms x graphs at ~12 s each
EVAL_EPOCH_INTERVAL=${EVAL_EPOCH_INTERVAL:-1}   # per-epoch generation eval (e17 n60 protocol)
PROBE_OUT=${PROBE_OUT:-$PROJ/results/e19_probe}

export WANDB_ENTITY=${WANDB_ENTITY:-alelab}
export HF_HOME=${HF_HOME:-$ALELAB/hf_cache}
export HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf_datasets_${SLURM_JOB_ID:-manual}"
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-28}"
# PATH prepend, not `conda activate`: compute-node batch shells have no conda
# (e17 jobs 7639051-53 died exit 127).
export PATH="$ENV_NAME/bin:$PATH"
# First fleet's epoch-1 generation evals OOMed on ~44 GiB allocs with ~44 GiB
# reserved-but-unallocated — fragmentation, not true pressure.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

test -f "$DATA_SPLIT/formatted_all_new__train.json" || { echo "missing $DATA_SPLIT/formatted_all_new__train.json"; exit 2; }

# Arm tables: e18_arms.sh supplies the mask_a base, e19_arms.sh the hop overlay.
PE_GT="$PE_DIR/path_navigator_gt.pt"
source scripts/e18_arms.sh          # cwd is the repo root (cd above)
source scripts/e19_arms.sh
e19_arm_args "$ARM"

echo "[$(date --iso-8601=seconds)] $RUN_NAME arm=$ARM data=$DATA_SPLIT max_steps=$MAX_STEPS tag=$TAG"
echo "=== python: $(which python) ==="

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
# _construct_output_dir); exactly <RUN_NAME>_<8-char id> so sibling arms with a
# shared prefix never match (the e18 7731157 lesson).
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
