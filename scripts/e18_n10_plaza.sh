#!/bin/bash
# e18 arm on plaza (2x A6000 48 GB, no scheduler) — the betty sbatch's twin for
# when the cluster queue is backed up. Same arm table (scripts/e18_arms.sh),
# same recipe, EXCEPT the base model is 4-bit NF4 (trainer.bit4=true): the 31B
# in bf16 is ~62 GB and does not fit one 48 GB card. So: a pipeline / probe
# smoke test and a first look at the arms, NOT a B200 s/step calibration.
#
#   GPU=0 ARM=mask MAX_STEPS=40 RUN_NAME=e18_n10_calib_plaza4b nohup scripts/e18_n10_plaza.sh > logs/<name>.log 2>&1 &
#
# Prereqs on plaza (data, copied from betty — never code):
#   data/n_10_vllm/gen/nav_n10_gemma_data/split/      (the n10 corpus)
#   outputs/e9_multistage_training/e9_ms_stage1/e9_ms_stage1_sqgk4o3j/checkpoint-100/{adapter_config.json,adapter_model.safetensors}
#   path_navigator_gt.pt                               (repo root; md5 c6fa43ca…)

set -euo pipefail
cd "$(dirname "$0")/.."
PROJ=$PWD

GPU=${GPU:-0}
ARM="${ARM:?set ARM=mask|mask_a|mask_b|mask_ab|mask_bind|mask_b_bind|mask_d|text_edges (scripts/e18_arms.sh)}"
DATA_SPLIT=${DATA_SPLIT:-$PROJ/data/n_10_vllm/gen/nav_n10_gemma_data/split}
MAX_STEPS=${MAX_STEPS:-300}
RUN_NAME=${RUN_NAME:-e18_n10_${ARM}_plaza4b}
TAG=${TAG:-e18_identity}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-$PROJ/outputs/e18_identity}
PROBE_TRAIN_GRAPHS=${PROBE_TRAIN_GRAPHS:-3}
PROBE_OUT=${PROBE_OUT:-$PROJ/results/e18_probe}
PE_GT=$PROJ/path_navigator_gt.pt
STAGE1_LORA=$PROJ/outputs/e9_multistage_training/e9_ms_stage1/e9_ms_stage1_sqgk4o3j/checkpoint-100

export WANDB_ENTITY=${WANDB_ENTITY:-alelab}
export CUDA_VISIBLE_DEVICES=$GPU          # trainer.device=0 then means this card
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "$HOME/venvs/grep-rl/bin/activate"

test -f "$DATA_SPLIT/formatted_all_new__train.json" || { echo "missing $DATA_SPLIT/formatted_all_new__train.json"; exit 2; }
test -f "$STAGE1_LORA/adapter_model.safetensors" || { echo "missing $STAGE1_LORA/adapter_model.safetensors"; exit 2; }
test -f "$PE_GT" || { echo "missing $PE_GT"; exit 2; }

source scripts/e18_arms.sh
e18_arm_args "$ARM"

echo "[$(date --iso-8601=seconds)] $RUN_NAME arm=$ARM gpu=$GPU data=$DATA_SPLIT max_steps=$MAX_STEPS tag=$TAG (4-bit)"
echo "=== python: $(which python) ==="

python -m prism.training.train_v3 \
    --config-name=e9_base_config \
    model.path="google/gemma-4-31B-it" \
    "${ARM_ARGS[@]}" \
    data.train_files="$DATA_SPLIT/formatted_all_new__train.json" \
    data.val_files="$DATA_SPLIT/formatted_all_new__val.json" \
    data.edge_weights=binary \
    trainer.bit4=true trainer.device=0 \
    trainer.freeze_lora=false trainer.freeze_pe=false \
    trainer.learning_rate=0.00025 trainer.epochs=3 trainer.max_steps="$MAX_STEPS" \
    +trainer.sft.save_total_limit=1 \
    trainer.checkpoint_dir="$CHECKPOINT_DIR" \
    trainer.save_name="$RUN_NAME" \
    wandb.run_name="$RUN_NAME" wandb.tag="$TAG" \
    eval.data="$DATA_SPLIT/test_graphs" eval.num_graphs=-1 eval.epoch_interval=999 \
    eval.post_train_graphs="$DATA_SPLIT/test_graphs"

RUN_DIR=$(ls -dt "$CHECKPOINT_DIR/${RUN_NAME}"_* | head -1)
echo "[$(date --iso-8601=seconds)] run dir: $RUN_DIR"

mkdir -p "$PROBE_OUT"
python scripts/neighbour_probe.py --checkpoint "$RUN_DIR" --four-bit --device 0 \
    --graphs "$DATA_SPLIT/test_graphs" \
    --output "$PROBE_OUT/${RUN_NAME}_test.json"
python scripts/neighbour_probe.py --checkpoint "$RUN_DIR" --four-bit --device 0 \
    --graphs "$DATA_SPLIT/train_graphs" --max-graphs "$PROBE_TRAIN_GRAPHS" \
    --output "$PROBE_OUT/${RUN_NAME}_train.json"

echo "[$(date --iso-8601=seconds)] $RUN_NAME DONE"
