#!/bin/bash
#SBATCH --job-name=e20-n60-oracle-gen
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#
# e20b — 3x oracle-path corpus: 156 FRESH n~60 graphs whose "teacher responses"
# are hardcoded NetworkX ground-truth routes (generate_data_spine.py
# --oracle-paths). Phase 1 (populate: rename + task generation) still runs on
# Gemma-4-31B via vLLM; Phase 2 needs NO rollout LLM — routes are shortest
# paths through each task's waypoints with avoided nodes removed, self-verified
# through the eval scorer before commit. Expected yield ~156 x 12 ~= 1.9k
# samples (~3x the v3 train corpus), token-wise much smaller (no reasoning).
#
#   * SEED_START=401: disjoint from v2 (101-133), v3 (201-252) topologies.
#   * VAL_IDS pins the LAST 6 fresh graphs as this corpus's own val/test split
#     (loss + a same-distribution eval set); the e19-comparable eval still runs
#     on the FROZEN n60_v3 test graphs at training time (e20_path_only_sft).
#
# Submit:  sbatch scripts/e20_n60_oracle_generate.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

export ALELAB_DRIVE="${ALELAB_DRIVE:-/vast/projects/aribeiro/alelab/jporras}"

ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-vllm}
export HF_HOME=${HF_HOME:-/vast/projects/aribeiro/alelab/jporras/hf_cache}

module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export PRISM_LLM_BACKEND=vllm
export VLLM_USE_FLASHINFER_SAMPLER=0
export PRISM_VLLM_MAX_LEN=32768
export PRISM_VLLM_TP=1
export PRISM_HF_MODEL="google/gemma-4-31B-it"
export DATA_ROOT="${DATA_ROOT:-${ALELAB_DRIVE}/GREP-PRISM/data/n_60_oracle_v1}"
export ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-8}"
export GRAPH_COMMUNITIES=6
export GRAPH_NODES_PER_COMMUNITY=8
export GRAPH_LABEL=60

export N_TASKS=12
export N_LONGHOP=2
export SEED_START=401
export VAL_IDS="${VAL_IDS:-150 151 152 153 154 155}"

# e20b: oracle rollouts (no rollout LLM; Phase-1 populate still vLLM-batched).
export EXTRA_GEN_ARGS="--oracle-paths"

N_GRAPHS=156

srun --ntasks=1 bash scripts/nav_n100_gemma_generate.sh 0 "$PRISM_HF_MODEL" "$N_GRAPHS"
