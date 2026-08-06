#!/bin/bash
#SBATCH --job-name=nav-n60-gemma31b-vllm
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#
# N~60 variant of nav_n100_vllm_generate.sbatch (same pipeline, smaller graphs):
# 6 communities x 8 nodes = 48 regions x 1.3 ~= 62 nodes.
# Data root: $ALELAB_DRIVE/GREP-PRISM/data/n_60_vllm
#
# Submit:  sbatch scripts/nav_n60_vllm_generate.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

: "${ALELAB_DRIVE:?ALELAB_DRIVE must be set (shared drive root)}"

ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-vllm}
export HF_HOME=${HF_HOME:-/vast/projects/aribeiro/alelab/jporras/hf_cache}

module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export PRISM_LLM_BACKEND=vllm
export VLLM_USE_FLASHINFER_SAMPLER=0
export PRISM_VLLM_MAX_LEN=32768
export PRISM_VLLM_TP=1
export DATA_ROOT="${ALELAB_DRIVE}/GREP-PRISM/data/n_60_vllm"
export ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-8}"
export GRAPH_COMMUNITIES=6
export GRAPH_NODES_PER_COMMUNITY=8
export GRAPH_LABEL=60

mkdir -p "$DATA_ROOT"

srun bash scripts/nav_n100_gemma_generate.sh 0 "google/gemma-4-31B-it" 33
