#!/bin/bash
# e21 — plaza PREVIEW runs of the oracle_v2 cells (docs/2026-08-24
# e21_oracle_scale_design.md) while the betty pair (7834933/34) waits in queue.
#
# NOT a comparable replicate of the betty cells: the 31B does not fit a 48 GB
# A6000 in bf16, so these run QLoRA 4-bit (trainer.bit4=true) — treat the
# numbers as an early signal only. Run names carry a _plaza4bit suffix so they
# can never be confused with the bf16 fleet in wandb.
#
# Data paths are the plaza-local copies (scp'd from betty 2026-08-24):
#   ~/sourcecode/GREP-PRISM/data/n_60_oracle_v2/split          (train/val)
#   ~/sourcecode/GREP-PRISM/data/n_60_vllm_v3/test_graphs      (frozen eval)
#
# Usage (one GPU each; launch under nohup):
#   nohup bash scripts/e21_plaza_oracle_v2.sh hop_depth  0 > ~/e21_hop.log  2>&1 &
#   nohup bash scripts/e21_plaza_oracle_v2.sh text_edges 1 > ~/e21_text.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/.."

ARM="${1:?usage: e21_plaza_oracle_v2.sh <hop_depth|text_edges> <gpu>}"
GPU="${2:?usage: e21_plaza_oracle_v2.sh <hop_depth|text_edges> <gpu>}"

REPO=$HOME/sourcecode/GREP-PRISM
PE_GT=$REPO/path_navigator_gt.pt
STAGE1_LORA=$REPO/outputs/e9_multistage_training/e9_ms_stage1/e9_ms_stage1_sqgk4o3j/checkpoint-100
DATA_SPLIT=$REPO/data/n_60_oracle_v2/split
EVAL_GRAPHS=$REPO/data/n_60_vllm_v3/test_graphs
CHECKPOINT_DIR=$REPO/outputs/e21_plaza
RUN_NAME=e21_oracle_v2_${ARM}_plaza4bit

test -f "$DATA_SPLIT/formatted_all_new__train.json" || { echo "missing $DATA_SPLIT"; exit 2; }
test -d "$EVAL_GRAPHS" || { echo "missing $EVAL_GRAPHS"; exit 2; }
test -f "$PE_GT" || { echo "missing $PE_GT"; exit 2; }
test -d "$STAGE1_LORA" || { echo "missing $STAGE1_LORA"; exit 2; }

# shellcheck disable=SC1091
source "$HOME/venvs/grep-rl/bin/activate"
# train_v3's device_map pins to cuda:0 of the VISIBLE set, so select the card
# here and pass trainer.device=0.
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PRISM_DISABLE_SPINE_TOOLS=1
export WANDB_ENTITY=alelab

source scripts/e18_arms.sh
source scripts/e19_arms.sh
case "$ARM" in
  hop_depth)  e19_arm_args hop_depth ;;
  text_edges) e18_arm_args text_edges ;;
  *) echo "unknown ARM=$ARM"; exit 2 ;;
esac

echo "[$(date --iso-8601=seconds)] $RUN_NAME gpu=$GPU (4-bit preview)"

python -m prism.training.train_v3 \
    --config-name=e9_base_config \
    model.path="google/gemma-4-31B-it" \
    "${ARM_ARGS[@]}" \
    data.train_files="$DATA_SPLIT/formatted_all_new__train.json" \
    data.val_files="$DATA_SPLIT/formatted_all_new__val.json" \
    data.edge_weights=binary \
    data.response_format=route_only \
    data.spine_tools=none data.icl_examples=0 \
    trainer.freeze_lora=false trainer.freeze_pe=false \
    trainer.learning_rate=0.00025 trainer.epochs=3 trainer.max_steps=-1 \
    trainer.bit4=true trainer.device=0 \
    +trainer.sft.save_total_limit=1 \
    trainer.checkpoint_dir="$CHECKPOINT_DIR" \
    trainer.save_name="$RUN_NAME" \
    wandb.run_name="$RUN_NAME" wandb.tag=e21_oracle_scale \
    eval.data="$EVAL_GRAPHS" eval.num_graphs=-1 eval.epoch_interval=1 \
    eval.use_icl=false \
    eval.post_train_graphs="$EVAL_GRAPHS"

echo "[$(date --iso-8601=seconds)] $RUN_NAME DONE"
