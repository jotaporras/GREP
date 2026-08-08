#!/bin/bash
#SBATCH --job-name=nav-n60-gemma31b-vllm-v3
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%j.out
#SBATCH --time=36:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#
# v3 n~60 corpus: tests whether widening the training data closes the GT arm's
# train/test transfer gap (v2: 0.429 on train graphs vs 0.114 on test).
#   * 52 NEW train skeletons (seeds 201-252 — disjoint from v2's 101-133, so no
#     topology overlap with the frozen test set)
#   * 12 tasks/graph: 10 per the v2 recipe + 2 long-hop Navigability tasks with
#     fixed endpoints sampled uniformly over hop lengths [3, diameter]
#   * the 7 v2 test graphs are FROZEN (same skeletons/names, original 10 tasks
#     intact), renumbered 052-058, and extended with the same 2 long-hop tasks
#     by scripts/append_longhop_tasks.py; VAL_IDS pins them as the test split
#
# Submit:  sbatch scripts/nav_n60_vllm_generate_v3.sbatch

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
export DATA_ROOT="${ALELAB_DRIVE}/GREP-PRISM/data/n_60_vllm_v3"
export ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-8}"
export GRAPH_COMMUNITIES=6
export GRAPH_NODES_PER_COMMUNITY=8
export GRAPH_LABEL=60

export N_TASKS=12
export N_LONGHOP=2
export SEED_START=201
export MAX_GRAPHS=59
export VAL_IDS="052 053 054 055 056 057 058"

V2_TEST="${ALELAB_DRIVE}/GREP-PRISM/data/n_60_vllm_v2/gen/nav_n60_gemma_data/split/test_graphs"
RUN_DIR="${DATA_ROOT}/gen/nav_n60_gemma_data"
mkdir -p "${RUN_DIR}/populated_graphs"

# Stage 0: extend the frozen v2 test graphs (own vLLM load; freed on exit so
# the main pipeline gets the whole GPU back).
srun --ntasks=1 python scripts/append_longhop_tasks.py \
  --src-dir "$V2_TEST" \
  --dst-dir "${RUN_DIR}/populated_graphs" \
  --start-index 52 \
  --n-longhop "$N_LONGHOP" \
  --seed 4242

srun --ntasks=1 bash scripts/nav_n100_gemma_generate.sh 0 "$PRISM_HF_MODEL" 52
