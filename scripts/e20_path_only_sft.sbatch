#!/bin/bash
#SBATCH --job-name=e20-path-only
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%j.out
#SBATCH --time=10:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
# betty's SLURM cli_filter (2026-08-03) enforces EXACTLY 28 CPUs / 224 GB per
# GPU on dgx-b200.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#
# e20 — path-only prediction training (docs/2026-08-24 e20_path_only_pred_design.md).
# One job = one (ARM, DATASET) cell:
#
#   ARM=hop_depth      best e19 arch by eval/accuracy (73.8% — v6sy0uhi recipe)
#   ARM=text_edges     plain-LLM baseline (edge list in text, fresh LoRA)
#
#   DATASET=pathonly   e20a: route-only Gemma distillation on the v3 graphs
#   DATASET=oracle     e20b: 3x fresh graphs, NetworkX ground-truth routes
#
# Targets are bare routes (data.response_format=route_only): no <think>
# scaffold, no tools, no ICL. Eval prompts carry the same contract (the seam
# wraps the bare route back to [answer(...)], so grading is unchanged) and run
# on the SAME frozen n60_v3 test graphs as e18/e19 — accuracy numbers are
# directly comparable to hop_depth 73.8 / cn7ub88q 72.6 / mask_a 56.0.
#
# Protocol otherwise identical to e19_n60_sft.sbatch: lr 2.5e-4, 3 epochs,
# per-epoch built-in generation eval, binary edge weights, save_total_limit 1.
#
# Submit:  ARM=hop_depth  DATASET=pathonly sbatch scripts/e20_path_only_sft.sbatch
#          ARM=text_edges DATASET=pathonly sbatch scripts/e20_path_only_sft.sbatch
#          ARM=hop_depth  DATASET=oracle   sbatch scripts/e20_path_only_sft.sbatch
#          ARM=text_edges DATASET=oracle   sbatch scripts/e20_path_only_sft.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

ALELAB=/vast/projects/aribeiro/alelab/jporras
PROJ=$ALELAB/GREP-PRISM
ENV_NAME=${ENV_NAME:-$ALELAB/envs/GREP-PRISM-v3}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-$PROJ/outputs/e20_path_only_pred}
PE_DIR=$PROJ/pretrained_pe
STAGE1_LORA=$PROJ/outputs/e9_multistage_training/e9_ms_stage1/e9_ms_stage1_sqgk4o3j/checkpoint-100

ARM="${ARM:?set ARM=hop_depth|text_edges}"
DATASET="${DATASET:?set DATASET=pathonly|oracle}"
case "$DATASET" in
  pathonly) DATA_SPLIT=${DATA_SPLIT:-$PROJ/data/n_60_pathonly_v1/gen/nav_n60_gemma_data/split} ;;
  oracle)   DATA_SPLIT=${DATA_SPLIT:-$PROJ/data/n_60_oracle_v1/gen/nav_n60_gemma_data/split} ;;
  # CONTROL cell: the ORIGINAL v3 reasoning corpus. Pair with
  # RESPONSE_FORMAT=think_route to reproduce the e19 regime inside this exact
  # harness — the missing arm of the 2x2 (corpus x response_format) needed to
  # tell "route_only unlocked the baseline" from "the baseline always won".
  v3)       DATA_SPLIT=${DATA_SPLIT:-$PROJ/data/n_60_vllm_v3/gen/nav_n60_gemma_data/split} ;;
  # e21 cells: oracle579 = oracle v1 train subsampled to the pathonly corpus
  # size (579) — separates data-size from target-purity in the e20 oracle gain
  # (built by scripts/e21_subsample_split.py). oracle_v2 = the 500-graph
  # hop-stratified + grounding-diverse corpus (e21_n60_oracle_v2_generate).
  oracle579) DATA_SPLIT=${DATA_SPLIT:-$PROJ/data/n_60_oracle_v1/gen/nav_n60_gemma_data/split_579} ;;
  oracle_v2) DATA_SPLIT=${DATA_SPLIT:-$PROJ/data/n_60_oracle_v2/gen/nav_n60_gemma_data/split} ;;
  # oracle_v2b = v2 minus grounding directives (same skeleton seeds 601-1100,
  # sharded gen e21_n60_oracle_v2b_generate.sbatch + e21_merge_shards.py).
  oracle_v2b) DATA_SPLIT=${DATA_SPLIT:-$PROJ/data/n_60_oracle_v2b/split} ;;
  # oracle_v2c = v2b + restored "from the starting area" phrasing on 40% of the
  # long-hop tasks (e21_n60_oracle_v2c_generate.sbatch + e21_merge_shards.py).
  oracle_v2c) DATA_SPLIT=${DATA_SPLIT:-$PROJ/data/n_60_oracle_v2c_r2/split} ;;
  *) echo "unknown DATASET=$DATASET (pathonly|oracle|v3|oracle579|oracle_v2|oracle_v2b|oracle_v2c)"; exit 2 ;;
esac
# Comparable eval: ALWAYS the frozen n60_v3 test graphs (same 7 graphs / 84
# questions as e18/e19), regardless of which corpus trains the run.
EVAL_GRAPHS=${EVAL_GRAPHS:-$PROJ/data/n_60_vllm_v3/gen/nav_n60_gemma_data/split/test_graphs}

MAX_STEPS=${MAX_STEPS:--1}
RESPONSE_FORMAT=${RESPONSE_FORMAT:-route_only}
RUN_NAME=${RUN_NAME:-e20_${DATASET}_${ARM}_${RESPONSE_FORMAT}}
TAG=${TAG:-e20_path_only_pred}
EVAL_EPOCH_INTERVAL=${EVAL_EPOCH_INTERVAL:-1}
# route_only (default, e20) | think_route (the e19/historical format).
case "$RESPONSE_FORMAT" in route_only|think_route) ;; *) echo "unknown RESPONSE_FORMAT=$RESPONSE_FORMAT"; exit 2 ;; esac

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

echo "[$(date --iso-8601=seconds)] $RUN_NAME arm=$ARM dataset=$DATASET data=$DATA_SPLIT eval=$EVAL_GRAPHS tag=$TAG"
echo "=== python: $(which python) ==="

srun python -m prism.training.train_v3 \
    --config-name=e9_base_config \
    model.path="google/gemma-4-31B-it" \
    "${ARM_ARGS[@]}" \
    data.train_files="$DATA_SPLIT/formatted_all_new__train.json" \
    data.val_files="$DATA_SPLIT/formatted_all_new__val.json" \
    data.edge_weights=binary \
    data.response_format="$RESPONSE_FORMAT" \
    data.spine_tools=none data.icl_examples=0 \
    trainer.freeze_lora=false trainer.freeze_pe=false \
    trainer.learning_rate=0.00025 trainer.epochs=3 trainer.max_steps="$MAX_STEPS" \
    +trainer.sft.save_total_limit=1 \
    trainer.checkpoint_dir="$CHECKPOINT_DIR" \
    trainer.save_name="$RUN_NAME" \
    wandb.run_name="$RUN_NAME" wandb.tag="$TAG" \
    eval.data="$EVAL_GRAPHS" eval.num_graphs=-1 eval.epoch_interval="$EVAL_EPOCH_INTERVAL" \
    eval.use_icl=false \
    eval.post_train_graphs="$EVAL_GRAPHS"

RUN_DIR=$(ls -dt "$CHECKPOINT_DIR/${RUN_NAME}"_???????? | head -1)
test -n "$RUN_DIR" || { echo "no run dir ${RUN_NAME}_<wandb id> under $CHECKPOINT_DIR"; exit 2; }
echo "[$(date --iso-8601=seconds)] run dir: $RUN_DIR"
echo "[$(date --iso-8601=seconds)] $RUN_NAME DONE"
