#!/bin/bash
#SBATCH --job-name=e22-geo-v1-gen
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#SBATCH --array=0-4
#
# e22 — geo_v1 corpus: BSP floor-plan skeletons + robot-scale long-hop routes
# (docs/2026-11-02 e22_geometric_scale_design.md §2-§3, step 5).
#
#   NEW   LAYOUT=floorplan, size mix 40/70/100/130 regions (90/150/240/120 of
#         600 skeletons, seed->size table below), door_extra ~ U[0.05, 0.20],
#         corridor_bias 0.3, 8 zones, objects Poisson(0.4), random robot_location
#   NEW   --longhop-hop-weights  mode 6-7 hops, 15-hop tail (replaces max_boost)
#   NEW   --longhop-named-start-frac 0.5  (keep_robot_location; start named)
#   NEW   --longhop-allow-via 0.2         (waypoint off every shortest path)
#   NEW   --longhop-object-goal-frac 0.5  (goal phrased by a contained object)
#   KEEP  --longhop-allow-avoid, --longhop-start-area-frac 0.4 (now applied to
#         the un-named half only: ~20% of long-hop tasks say "starting area"),
#         --oracle-paths (NetworkX routes; via/avoid honoured by _oracle_route)
#   18 tasks per graph: 16 long-hop + 2 standard navigation.
#
# Array tasks 0-3: training shards, 150 skeletons each, seeds 3001-3600
# (SEED_START = 3001 + 150*i). Each shard writes its own DATA_ROOT (resume is
# not safe for concurrent writers on one root) and holds out its last two
# local graphs as val (VAL_IDS="148 149": val loss only).
#
# Array task 4: geo_test (eval set B), a separate root. 2 skeletons per size,
# seeds 3901-3908, 12 long-hop tasks each with UNIFORM hop weights over 1-15
# (the hop buckets a graph lacks are skipped), same role fractions. The split
# script refuses a split with no train graph, so this shard generates 10
# skeletons: graphs 000-007 are the test set (VAL_IDS), 008-009 are two filler
# 40-region graphs whose train split is never used.
#
# After tasks 0-3 finish, merge the training shards:
#
#   python scripts/e21_merge_shards.py \
#     --root $ALELAB_DRIVE/GREP-PRISM/data/n_geo_v1 \
#     --gen-subdir gen/nav_ngeo_gemma_data/split
#
# then audit BEFORE any training (§6 corpus gate):
#
#   python scripts/e22_corpus_audit.py --root $ALELAB_DRIVE/GREP-PRISM/data/n_geo_v1
#
# geo_test eval dir (EVAL set B):
#   $ALELAB_DRIVE/GREP-PRISM/data/n_geo_v1_test/shard4/gen/nav_ngeo_gemma_data/split/test_graphs
#
# The e21 v2/v2b/v2c corpora were generated on the vLLM batched path, which
# dropped --grounding-directives and --longhop-allow-avoid (only
# longhop_start_area_frac was forwarded after a8013ec). Both populate paths now
# read one DataGenerator._prompt_flags() dict; verify by corpus md5 and the
# audit's avoid/via rates, not by argparse echo.
#
# Submit:  sbatch scripts/e22_geo_v1_generate.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

export ALELAB_DRIVE="${ALELAB_DRIVE:-/vast/projects/aribeiro/alelab/jporras}"

ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-vllm}
# gemma-4-31B-it lives under $ALELAB_DRIVE/.cache/huggingface (hub symlinked from
# ~/.cache since 2026-09-01); the older hf_cache dir no longer holds it.
export HF_HOME=${HF_HOME:-/vast/projects/aribeiro/alelab/jporras/.cache/huggingface}

module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export PRISM_LLM_BACKEND=vllm
export VLLM_USE_FLASHINFER_SAMPLER=0
# 130-region skeletons with zones roughly double the v2c populate prompt, and
# Gemma's thinking shares the per-request budget with the JSON answer: at
# 20480 tokens / 32768 context the first run (job 9987719) lost 227 of 600
# training graphs, almost all 100-130 regions (answer missing or truncated).
export PRISM_VLLM_MAX_LEN=${PRISM_VLLM_MAX_LEN:-65536}
export PRISM_VLLM_MAX_NEW_TOKENS=${PRISM_VLLM_MAX_NEW_TOKENS:-40960}
export PRISM_VLLM_TP=1
export PRISM_HF_MODEL="google/gemma-4-31B-it"
export ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-8}"

export LAYOUT=floorplan
export GRAPH_LABEL=geo
export DOOR_EXTRA_MIN=0.05
export DOOR_EXTRA_MAX=0.20
export CORRIDOR_BIAS=0.3
export ZONE_DEPTH=3
export OBJECT_RATE=0.4
export ROBOT_LOCATION=random

# Train hop target (§3): mode 6-7, 35% in 6-7, ~15% >= 10, max 15.
TRAIN_HOP_WEIGHTS="1:0.04,2:0.04,3:0.06,4:0.08,5:0.12,6:0.17,7:0.17,8:0.08,9:0.07,10:0.05,11:0.04,12:0.03,13:0.02,14:0.02,15:0.01"
TEST_HOP_WEIGHTS="1:1,2:1,3:1,4:1,5:1,6:1,7:1,8:1,9:1,10:1,11:1,12:1,13:1,14:1,15:1"
ROLE_ARGS="--longhop-allow-avoid --longhop-allow-via 0.2 --longhop-named-start-frac 0.5 --longhop-start-area-frac 0.4 --longhop-object-goal-frac 0.5"

# Seed -> size: a 20-seed cycle with 3x40, 5x70, 8x100, 4x130 regions, so the
# 600 training seeds hold exactly 90/150/240/120 skeletons (§2 size mix).
SIZE_CYCLE=(100 70 130 40 100 70 100 130 100 40 70 100 130 100 70 40 100 130 70 100)

SHARD="${SLURM_ARRAY_TASK_ID:?run via sbatch --array}"
if [[ "$SHARD" -lt 4 ]]; then
  export DATA_ROOT="${ALELAB_DRIVE}/GREP-PRISM/data/n_geo_v1/shard${SHARD}"
  export SEED_START=$((3001 + 150 * SHARD))
  N_GRAPHS=150
  export N_TASKS=18
  export N_LONGHOP=16
  export VAL_IDS="148 149"
  mkdir -p "$DATA_ROOT"
  export SIZE_TABLE="${DATA_ROOT}/size_table.tsv"
  : > "$SIZE_TABLE"
  for ((s = SEED_START; s < SEED_START + N_GRAPHS; s++)); do
    echo "$s ${SIZE_CYCLE[$(((s - 3001) % 20))]}" >> "$SIZE_TABLE"
  done
  export EXTRA_GEN_ARGS="--oracle-paths --longhop-hop-weights ${TRAIN_HOP_WEIGHTS} ${ROLE_ARGS}"
else
  export DATA_ROOT="${ALELAB_DRIVE}/GREP-PRISM/data/n_geo_v1_test/shard4"
  export SEED_START=3901
  N_GRAPHS=10
  export N_TASKS=12
  export N_LONGHOP=12
  export VAL_IDS="000 001 002 003 004 005 006 007"
  mkdir -p "$DATA_ROOT"
  export SIZE_TABLE="${DATA_ROOT}/size_table.tsv"
  printf '%s\n' "3901 40" "3902 40" "3903 70" "3904 70" "3905 100" "3906 100" \
    "3907 130" "3908 130" "3909 40" "3910 40" > "$SIZE_TABLE"
  export EXTRA_GEN_ARGS="--oracle-paths --longhop-hop-weights ${TEST_HOP_WEIGHTS} ${ROLE_ARGS}"
fi

echo "=== geo_v1 shard ${SHARD}: seeds ${SEED_START}-$((SEED_START + N_GRAPHS - 1)) -> ${DATA_ROOT} ==="
echo "    size mix: $(awk '{print $2}' "$SIZE_TABLE" | sort -n | uniq -c | tr '\n' ' ')"
echo "    EXTRA_GEN_ARGS=${EXTRA_GEN_ARGS}"
srun --ntasks=1 bash scripts/nav_n100_gemma_generate.sh 0 "$PRISM_HF_MODEL" "$N_GRAPHS"
