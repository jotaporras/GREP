#!/bin/bash
#SBATCH --job-name=attn-capture
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%A_%a.out
#SBATCH --time=02:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#SBATCH --array=0-2
#
# Attention-matrix capture for the visualization notebook (layers 0 and 5;
# 5 = first global/masked layer). Same val example (idx 0) for all three.
#
#   idx  checkpoint                                    role
#   0    e9_llm_baseline__gemma31b_sfa1r99x            with-edges plain-LLM baseline
#   1    e13d_gt_decode_zx3s426b                       best no-edge learnable RPE mask (0.79)
#   2    e13c_gmask_decode_wg97eelk                    no-edge param-free adjacency mask (0.61)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3}
ALELAB=/vast/projects/aribeiro/alelab/jporras
OUT_DIR=$ALELAB/GREP-PRISM/outputs/attn_capture
mkdir -p "$OUT_DIR"

export HF_HOME=/vast/projects/aribeiro/alelab/jporras/hf_cache
export HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf_datasets_${SLURM_JOB_ID:-manual}"

module load anaconda3
source activate "$ENV_NAME"

TASK_ID=${SLURM_ARRAY_TASK_ID:-${1:-0}}
case "$TASK_ID" in
    0) CKPT=$ALELAB/GREP-PRISM/outputs/e9_llm_baseline/e9_llm_baseline__gemma31b_sfa1r99x; NAME=llm_baseline_withedges ;;
    1) CKPT=$ALELAB/GREP-PRISM/outputs/e13d_nav_decode/e13d_gt_decode_zx3s426b;            NAME=rpe_mask_decode_best ;;
    2) CKPT=$ALELAB/GREP-PRISM/outputs/e13c_decode_consistent/e13c_gmask_decode_wg97eelk;  NAME=gmask_decode ;;
    *) echo "bad task id $TASK_ID" >&2; exit 1 ;;
esac

srun python scripts/capture_attention.py \
    --checkpoint "$CKPT" \
    --example-idx 0 --layers 0,5 \
    --out "$OUT_DIR/attn_${NAME}.pt"

echo "DONE: $OUT_DIR/attn_${NAME}.pt"
