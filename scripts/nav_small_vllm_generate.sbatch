#!/bin/bash
#SBATCH --job-name=nav-small-gemma31b-vllm
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%j.out
#SBATCH --time=06:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#
# e18 small-graph corpora for the node-identity dev loop (docs/2026-08-21
# e18_direction_discussion.md §5-6): same recipe as n30/n60_v3 (SBM skeletons ->
# vLLM Gemma-31B populate -> SPINE rollouts -> graph-level split), shrunk to
# n~10 / n~15 so an SFT run fits in ~30 min and the per-hop readout can be
# studied before scaling N.
#
#   SIZE=10  -> 2 communities x 4 regions =  8 regions (~10 nodes w/ objects), seeds 301+
#   SIZE=15  -> 3 communities x 4 regions = 12 regions (~15 nodes),            seeds 401+
#
# INTER_PROB is raised to 0.25 (n60 used 0.05): at 2x4 the n60 value expects
# 0.8 cross-community edges and the connectivity retry loop exhausts. Mean degree
# lands ~3 (n30: ~4.4, n60: ~6) — the candidate-set size the diagnosis says the
# readout degrades with.
#
# Submit:  SIZE=10 sbatch scripts/nav_small_vllm_generate.sbatch
#          SIZE=15 sbatch scripts/nav_small_vllm_generate.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

export ALELAB_DRIVE="${ALELAB_DRIVE:-/vast/projects/aribeiro/alelab/jporras}"

ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-vllm}
export HF_HOME=${HF_HOME:-/vast/projects/aribeiro/alelab/jporras/hf_cache}

SIZE="${SIZE:?set SIZE=10 or SIZE=15}"
case "$SIZE" in
  10) export GRAPH_COMMUNITIES=2; export SEED_START=301 ;;
  15) export GRAPH_COMMUNITIES=3; export SEED_START=401 ;;
  *) echo "unsupported SIZE=$SIZE (10|15)"; exit 2 ;;
esac
export GRAPH_NODES_PER_COMMUNITY=4
export GRAPH_LABEL="$SIZE"
export INTRA_PROB=0.6
export INTER_PROB=0.25

export PRISM_LLM_BACKEND=vllm
export VLLM_USE_FLASHINFER_SAMPLER=0
export PRISM_VLLM_MAX_LEN=32768
export PRISM_VLLM_TP=1
export PRISM_HF_MODEL="google/gemma-4-31B-it"
export DATA_ROOT="${ALELAB_DRIVE}/GREP-PRISM/data/n_${SIZE}_vllm"
export ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-8}"

# 45 graphs x 10 navigation tasks; the last 5 graph ids are the frozen test split.
# No long-hop reservation: diameters are 2-3 here, the uniform [3, diameter]
# sampler would degenerate to the diameter.
N_GRAPHS="${N_GRAPHS:-45}"
export N_TASKS=10
export N_LONGHOP=0
export MAX_GRAPHS="$N_GRAPHS"
export VAL_IDS="${VAL_IDS:-040 041 042 043 044}"

# PATH prepend, not `conda activate`: compute-node batch shells lost conda on
# 2026-08-16 (e17 jobs 7639051-53 died exit 127).
export PATH="${ENV_NAME}/bin:${PATH}"
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-28}"
echo "=== python: $(which python) ==="
echo "=== SIZE=$SIZE NC=$GRAPH_COMMUNITIES NPC=$GRAPH_NODES_PER_COMMUNITY INTER_PROB=$INTER_PROB seeds=$SEED_START.. N_GRAPHS=$N_GRAPHS ==="

srun --ntasks=1 bash scripts/nav_n100_gemma_generate.sh 0 "$PRISM_HF_MODEL" "$N_GRAPHS"
