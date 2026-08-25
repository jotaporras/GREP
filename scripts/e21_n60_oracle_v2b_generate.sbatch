#!/bin/bash
#SBATCH --job-name=e21-n60-oracle-v2b-gen
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#SBATCH --array=0-3
#
# e21 — oracle corpus v2b: v2 minus the grounding directives.
#
# The v2 postmortem (memory + 08-25 audit): --grounding-directives collapsed
# "from the starting area" phrasing to 2% of train tasks while 63% of the
# frozen eval uses it, causing the 92.9 -> 79.8 regression (8/11 regressed
# questions are starting-area tasks; 0 fixed). v2b keeps the two levers that
# were the point of e21 and reverts only the phrasing change:
#
#   KEEP  --longhop-max-boost 3     (hop stratification, diameter bucket 3x)
#   KEEP  --longhop-allow-avoid     (validated off-shortest-path avoids)
#   DROP  --grounding-directives    (the regression cause)
#
# Sharded 4-way for wall clock: array task i generates 125 skeletons with
# SEED_START = 601 + 125*i — together exactly the v2 seed range 601-1100, so
# v2b reuses the SAME graph population as v2 and the corpus diff is phrasing
# only. Each shard writes its own DATA_ROOT (the resume logic in
# generate_data_spine is not safe for concurrent writers on one root); every
# shard holds out its last two local graphs (VAL_IDS="123 124", 8 val graphs
# total). After ALL tasks finish, merge the shard splits with:
#
#   python scripts/e21_merge_shards.py \
#     --root $ALELAB_DRIVE/GREP-PRISM/data/n_60_oracle_v2b
#
# Submit:  sbatch scripts/e21_n60_oracle_v2b_generate.sbatch

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
export ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-8}"
export GRAPH_COMMUNITIES=6
export GRAPH_NODES_PER_COMMUNITY=8
export GRAPH_LABEL=60

export N_TASKS=12
export N_LONGHOP=10

SHARD="${SLURM_ARRAY_TASK_ID:?run via sbatch --array}"
export DATA_ROOT="${ALELAB_DRIVE}/GREP-PRISM/data/n_60_oracle_v2b/shard${SHARD}"
export SEED_START=$((601 + 125 * SHARD))
export VAL_IDS="123 124"

export EXTRA_GEN_ARGS="--oracle-paths --longhop-max-boost 3 --longhop-allow-avoid"

N_GRAPHS=125

echo "=== v2b shard ${SHARD}: seeds ${SEED_START}-$((SEED_START + N_GRAPHS - 1)) -> ${DATA_ROOT} ==="
srun --ntasks=1 bash scripts/nav_n100_gemma_generate.sh 0 "$PRISM_HF_MODEL" "$N_GRAPHS"
