#!/bin/bash
#SBATCH --job-name=e22-geo-sft
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
# betty's SLURM cli_filter (2026-08-03) enforces EXACTLY 28 CPUs / 224 GB per
# GPU on dgx-b200.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#
# e22 — geometric-scale SFT on the geo_v1 corpus (docs/2026-11-02
# e22_geometric_scale_design.md §5). One job = one cell:
#
#   ARM=hop_depth                    e22_geo_hop_depth     (v2c recipe)
#   ARM=text_edges                   e22_geo_text_edges    (edge list in prompt)
#   ARM=hop_depth MODEL=<id> STAGE1_LORA=<its stage-1 LoRA>
#                                    e22_geo_hop_depth_<model>  (deferred: no
#                                    sub-31B base has a stage-1 LoRA yet)
#
# KEEP_ROBOT_LOCATION=true keeps each eval graph's own robot_location in the
# prompt (eval.keep_robot_location; the robot sim's convention for set C).
# Tasks flagged keep_robot_location in the corpus keep it regardless.
#
# Recipe = e20_path_only_sft.sbatch DATASET=oracle_v2c (lr 2.5e-4, route_only,
# binary edge weights, no ICL, save_total_limit 1) with 4 epochs.
#
# Built-in per-epoch eval runs on ONE dir, EVAL_GRAPHS (default set A, the
# frozen n60_v3 test graphs, 84 questions). Sets B (geo_test) and C (hospital,
# scripts/hospital_eval_set.py) go through cross_eval on the saved checkpoint;
# checkpoint pick = best epoch on B (§5).
#
# MAX_SEQ_LENGTH: 130-region geo prompts exceed the 8192 base default
# (scripts/e22_corpus_audit.py prints the prompt-length distribution).
#
# Submit:  ARM=hop_depth  sbatch scripts/e22_geo_sft.sbatch
#          ARM=text_edges sbatch scripts/e22_geo_sft.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

ALELAB=/vast/projects/aribeiro/alelab/jporras
PROJ=$ALELAB/GREP-PRISM
ENV_NAME=${ENV_NAME:-$ALELAB/envs/GREP-PRISM-v3}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-$PROJ/outputs/e22_geometric_scale}
PE_DIR=$PROJ/pretrained_pe

ARM="${ARM:?set ARM=hop_depth|text_edges}"
DATASET="${DATASET:-geo_v1}"
MODEL="${MODEL:-google/gemma-4-31B-it}"
# The 31B stage-1 LoRA does not fit another base model: a non-default MODEL
# must bring its own STAGE1_LORA.
if [[ "$MODEL" == "google/gemma-4-31B-it" ]]; then
  STAGE1_LORA=${STAGE1_LORA:-$PROJ/outputs/e9_multistage_training/e9_ms_stage1/e9_ms_stage1_sqgk4o3j/checkpoint-100}
else
  STAGE1_LORA="${STAGE1_LORA:?MODEL=$MODEL needs its own STAGE1_LORA}"
fi
case "$DATASET" in
  # geo_v1 = BSP floor-plan skeletons, 40-130 regions, robot-scale long-hop
  # routes (e22_geo_v1_generate.sbatch + e21_merge_shards.py).
  geo_v1) DATA_SPLIT=${DATA_SPLIT:-$PROJ/data/n_geo_v1/split} ;;
  *) echo "unknown DATASET=$DATASET (geo_v1)"; exit 2 ;;
esac
EVAL_GRAPHS=${EVAL_GRAPHS:-$PROJ/data/n_60_vllm_v3/gen/nav_n60_gemma_data/split/test_graphs}

MAX_STEPS=${MAX_STEPS:--1}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-16384}
RESPONSE_FORMAT=route_only
MODEL_TAG=${MODEL_TAG:-$( [[ "$MODEL" == "google/gemma-4-31B-it" ]] && echo "" || echo "_${MODEL##*/}" )}
RUN_NAME=${RUN_NAME:-e22_geo_${ARM}${MODEL_TAG}}
TAG=${TAG:-e22_geometric_scale}
EVAL_EPOCH_INTERVAL=${EVAL_EPOCH_INTERVAL:-1}

export WANDB_ENTITY=${WANDB_ENTITY:-alelab}
export HF_HOME=${HF_HOME:-$ALELAB/hf_cache}
export HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf_datasets_${SLURM_JOB_ID:-manual}"
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-28}"
# PATH prepend, not `conda activate` (compute-node batch shells have no conda).
export PATH="$ENV_NAME/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# route_only forbids SPINE tool prompts at eval (train_v3 validates this pair).
export PRISM_DISABLE_SPINE_TOOLS=1

test -f "$DATA_SPLIT/formatted_all_new__train.json" || { echo "missing $DATA_SPLIT/formatted_all_new__train.json"; exit 2; }
test -d "$EVAL_GRAPHS" || { echo "missing $EVAL_GRAPHS"; exit 2; }

# Arm tables: e18_arms.sh supplies mask_a + text_edges, e19_arms.sh the hop overlay.
PE_GT="$PE_DIR/path_navigator_gt.pt"
source scripts/e18_arms.sh
source scripts/e19_arms.sh
case "$ARM" in
  hop_depth)  e19_arm_args hop_depth ;;
  text_edges) e18_arm_args text_edges ;;
  *) echo "unknown ARM=$ARM (hop_depth|text_edges)"; exit 2 ;;
esac

echo "[$(date --iso-8601=seconds)] $RUN_NAME arm=$ARM model=$MODEL data=$DATA_SPLIT eval=$EVAL_GRAPHS max_len=$MAX_SEQ_LENGTH tag=$TAG"
echo "=== python: $(which python) ==="

srun python -m prism.training.train_v3 \
    --config-name=e9_base_config \
    model.path="$MODEL" \
    "${ARM_ARGS[@]}" \
    data.train_files="$DATA_SPLIT/formatted_all_new__train.json" \
    data.val_files="$DATA_SPLIT/formatted_all_new__val.json" \
    data.edge_weights=binary \
    data.response_format="$RESPONSE_FORMAT" \
    data.max_seq_length="$MAX_SEQ_LENGTH" \
    data.spine_tools=none data.icl_examples=0 \
    trainer.freeze_lora=false trainer.freeze_pe=false \
    trainer.learning_rate=0.00025 trainer.epochs=4 trainer.max_steps="$MAX_STEPS" \
    +trainer.sft.save_total_limit=1 \
    trainer.checkpoint_dir="$CHECKPOINT_DIR" \
    trainer.save_name="$RUN_NAME" \
    wandb.run_name="$RUN_NAME" wandb.tag="$TAG" \
    eval.data="$EVAL_GRAPHS" eval.num_graphs=-1 eval.epoch_interval="$EVAL_EPOCH_INTERVAL" \
    eval.use_icl=false eval.keep_robot_location="${KEEP_ROBOT_LOCATION:-false}" \
    eval.post_train_graphs="$EVAL_GRAPHS"

RUN_DIR=$(ls -dt "$CHECKPOINT_DIR/${RUN_NAME}"_???????? | head -1)
test -n "$RUN_DIR" || { echo "no run dir ${RUN_NAME}_<wandb id> under $CHECKPOINT_DIR"; exit 2; }
echo "[$(date --iso-8601=seconds)] run dir: $RUN_DIR"
echo "[$(date --iso-8601=seconds)] $RUN_NAME DONE"
