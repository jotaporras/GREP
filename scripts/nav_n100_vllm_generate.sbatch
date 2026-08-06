#!/bin/bash
#SBATCH --job-name=nav-n100-gemma31b-vllm
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#
# SLURM wrapper for scripts/nav_n100_gemma_generate.sh with the vLLM backend
# (GREP-T1 AC6/AC7): Phase-1 populate runs as ONE continuously-batched call and
# Phase-2 runs ROLLOUT_WORKERS concurrent SPINE dialogues, all on a single B200.
# All generation logic, staging, auto-resume and leak checks live in the inner
# script; this file only allocates the node and sets the backend + data root.
#
# Data root (AC7): $ALELAB_DRIVE/GREP-PRISM/data/n_100_vllm
#   skeletons  -> <root>/training/nav_n100_skeletons
#   outputs    -> <root>/gen/nav_n100_gemma_data (+ split/)
#
# Env: a conda env with vLLM >= 0.20 (the DEFAULT PyPI wheel is the right one
# on B200 — Blackwell nodes run CUDA-13 drivers, no cu12x juggling needed):
#   uv pip install --overrides <(printf "torch>=2.11,<2.12\ntorchvision>=0.26,<0.27\n") \
#       -r requirements.txt vllm ninja cmake -e . -e ./SPINE
#
# FIXED run config: MODEL google/gemma-4-31B-it, N_GRAPHS 33 (same as the HF run).
# Overridable (export before sbatch, or --export):
#   ROLLOUT_WORKERS   concurrent Phase-2 rollouts (default 8)
#   PRISM_VLLM_MAX_LEN / PRISM_VLLM_GPU_UTIL / PRISM_VLLM_QUANT (see vllm_llm.py)
#
# Submit:  sbatch scripts/nav_n100_vllm_generate.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

: "${ALELAB_DRIVE:?ALELAB_DRIVE must be set (shared drive root)}"

ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-vllm}
export HF_HOME=${HF_HOME:-/vast/projects/aribeiro/alelab/jporras/hf_cache}

module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export PRISM_LLM_BACKEND=vllm
# B200 compute nodes cannot JIT-build flashinfer's sampling kernels at engine
# warmup (write_ninja fails); vLLM's native torch sampler works fine.
export VLLM_USE_FLASHINFER_SAMPLER=0
export PRISM_VLLM_TP=1              # a single B200 holds the bf16 31B
export DATA_ROOT="${ALELAB_DRIVE}/GREP-PRISM/data/n_100_vllm"
export ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-8}"

mkdir -p "$DATA_ROOT"

# Inner script args: [GPU_ID] [MODEL] [N_GRAPHS]. GPU_ID=0 (allocation-relative).
srun bash scripts/nav_n100_gemma_generate.sh 0 "google/gemma-4-31B-it" 33
