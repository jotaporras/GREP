#!/bin/bash
#SBATCH --job-name=e21-n60-oracle-v2-gen
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#
# e21 — oracle-path corpus v2 (docs/2026-08-24 e21_oracle_scale_design.md):
# ~3x oracle v1, with the question mix engineered at the measured e20 failure
# modes (5/6 residual errors were START-grounding, 1/6 avoid-constraint):
#
#   * 500 FRESH n~60 skeletons, SEED_START=601 (disjoint from v2 101-133,
#     v3 201-252, oracle_v1 401-556). Expected yield ~500 x 12 ~= 5.9k samples.
#   * Hop stratification: N_LONGHOP=10 of 12 tasks/graph get FIXED endpoints
#     sampled uniform over hop buckets [3, diameter] with the DIAMETER bucket
#     weight-boosted 3x (--longhop-max-boost 3) — replaces the e20b regime
#     where 10/12 endpoint pairs were the populate-LLM's free (short-biased)
#     choice. Delivered mix is auditable in populated_graphs/longhop_manifest.json.
#   * --longhop-allow-avoid: avoid-constraints on long-hop tasks, VALIDATED
#     against the graph to lie off every shortest path (hop length preserved).
#   * --grounding-directives: start-reference diversity (paraphrase / ordinal
#     sibling / object-hosted starts; <=1/3 of route tasks start at
#     robot_location) — targets the e20 start-grounding errors.
#
# Rollouts need NO LLM (--oracle-paths: NetworkX routes, self-verified through
# the eval scorer); Phase-1 populate runs Gemma-4-31B via vLLM.
#
# Submit:  sbatch scripts/e21_n60_oracle_v2_generate.sbatch

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
export DATA_ROOT="${DATA_ROOT:-${ALELAB_DRIVE}/GREP-PRISM/data/n_60_oracle_v2}"
export ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-8}"
export GRAPH_COMMUNITIES=6
export GRAPH_NODES_PER_COMMUNITY=8
export GRAPH_LABEL=60

export N_TASKS=12
export N_LONGHOP=10
export SEED_START=601
export VAL_IDS="${VAL_IDS:-494 495 496 497 498 499}"

export EXTRA_GEN_ARGS="--oracle-paths --longhop-max-boost 3 --longhop-allow-avoid --grounding-directives"

N_GRAPHS=500

srun --ntasks=1 bash scripts/nav_n100_gemma_generate.sh 0 "$PRISM_HF_MODEL" "$N_GRAPHS"
