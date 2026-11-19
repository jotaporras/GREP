#!/bin/bash
# e22 geo_v2 + geo_scale — LLM-free generation (templated populate + oracle routes).
# CPU only: runs on the Mac (or a CPU SLURM job). No GPU, no model, no API key.
#
# Design: docs/2026-11-02 e22_geometric_scale_design.md + coordinator brief 2026-11-18.
#   - templated naming/tasks: src/prism/data/template_populate.py, vocabulary harvested
#     from the Gemma-written geo_v1 corpus (scripts/e22_template_vocab.json)
#   - roles as geo_v1: named start 0.5, starting area 0.46 of the rest (the rate
#     geo_v1 realized; its nominal flag was 0.4), via 0.2,
#     object goal 0.5; oracle (NetworkX) routes, self-graded before commit
#
# Usage (repo root; PY defaults to python):
#   bash scripts/e22_geo_v2_local.sh v1      # rebuild geo_v1 shards 0-3 from local populated graphs
#   bash scripts/e22_geo_v2_local.sh train   # templated shards 4-6 (40-130, 200, 300, 400 regions)
#   bash scripts/e22_geo_v2_local.sh merge   # merge + audit -> $ROOT/split
#   bash scripts/e22_geo_v2_local.sh scale   # geo_scale evaluation set
#   bash scripts/e22_geo_v2_local.sh all
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
export PRISM_LLM_BACKEND=none        # graph_gen.TaskGraphGen builds no client
OUT=${OUT:-outputs/e22_geometric_scale/geo_v2_local}
ROOT=$OUT/n_geo_v2
SCALE=$OUT/geo_scale
V1_SRC=${V1_SRC:-outputs/e22_geometric_scale/notebook_data/train}
VOCAB=${VOCAB:-scripts/e22_template_vocab.json}
SEED=${SEED:-7}
LABEL=geo
SKEL_SUB=training/nav_n${LABEL}_skeletons
RUN_SUB=gen/nav_n${LABEL}_gemma_data
ROLE_ARGS="--named-start-frac 0.5 --start-area-frac 0.46 --via-frac 0.2 --object-goal-frac 0.5"
# geo_v1 train weights (e22_geo_v1_generate.sbatch): mode 6-7, max 15.
V1_HOP_WEIGHTS="1:0.04,2:0.04,3:0.06,4:0.08,5:0.12,6:0.17,7:0.17,8:0.08,9:0.07,10:0.05,11:0.04,12:0.03,13:0.02,14:0.02,15:0.01"
# 200-400 regions: the v1 weights scaled to 0.75, plus a 16-40 hop tail with mass
# 0.25, linearly decaying (16 hops 2.5x the weight of 40 hops). Buckets a graph
# lacks are skipped by the sampler.
LARGE_HOP_WEIGHTS=$($PY - <<'EOF'
v1 = "1:0.04,2:0.04,3:0.06,4:0.08,5:0.12,6:0.17,7:0.17,8:0.08,9:0.07,10:0.05,11:0.04,12:0.03,13:0.02,14:0.02,15:0.01"
w = {int(h): 0.75 * float(x) for h, x in (p.split(":") for p in v1.split(","))}
tail = {h: 1.0 - 0.6 * (h - 16) / 24 for h in range(16, 41)}
s = sum(tail.values())
w.update({h: 0.25 * x / s for h, x in tail.items()})
print(",".join(f"{h}:{x:.5f}" for h, x in sorted(w.items())))
EOF
)

skeletons() {  # skeletons <dir> <first seed> <n wanted> <regions>; skips seeds the generator rejects
  local dir=$1 seed=$2 want=$3 nr=$4 have=0 tries=0
  mkdir -p "$dir"
  while (( have < want )); do
    local f="$dir/eval_graph_unique_${LABEL}_seed${seed}.json"
    if [[ -f "$f" ]] || $PY scripts/generate_eval_graphs.py --layout floorplan --n-regions "$nr" \
        --door-extra-min 0.05 --door-extra-max 0.20 --corridor-bias 0.3 --zone-depth 3 \
        --region-area 25.0 --robot-location random --object-rate 0.4 --description-prob 0.05 \
        --n-tasks 16 --seed "$seed" --output "$f" > /dev/null 2>&1; then
      have=$((have + 1))
    else
      echo "  seed $seed ($nr regions): generator failed, skipped"
    fi
    seed=$((seed + 1)); tries=$((tries + 1))
    # 1000-region plans: ~65% of seeds are rejected, hence 8x
    (( tries < want * 8 )) || { echo "too many generator failures at $nr regions"; exit 1; }
  done
}

rollouts_and_split() {  # rollouts_and_split <shard dir> <val ids...>
  local shard=$1; shift
  local run=$shard/$RUN_SUB
  $PY scripts/training_data_generation/generate_data_spine.py \
    --data-dir "$run/populated_graphs" --name "$run" --skip-populate --oracle-paths \
    --n-tasks 16 --n-longhop-tasks 16 --seed 42 --task-proportions 0 0 0 1 > "$shard/rollouts.log" 2>&1
  tail -1 "$shard/rollouts.log"
  $PY scripts/training_data_generation/split_train_val.py --plans-dir "$run/generated_plans" \
    --graphs-dir "$run/populated_graphs" --output-dir "$run/split" --val-ids "$@" | tail -2
}

templated_shard() {  # templated_shard <shard#> <first seed> "<size:count ...>" <hop weights>
  local shard=$ROOT/shard$1 seed=$2 spec=$3 hw=$4
  echo "=== templated shard$1: $spec"
  for sc in $spec; do
    skeletons "$shard/$SKEL_SUB" "$seed" "${sc#*:}" "${sc%%:*}"
    seed=$((seed + 1000))
  done
  $PY scripts/e22_template_populate.py populate --vocab "$VOCAB" \
    --skeleton-dir "$shard/$SKEL_SUB" --out-dir "$shard/$RUN_SUB/populated_graphs" \
    --n-tasks 16 --hop-weights "$hw" $ROLE_ARGS --seed "$SEED"
  local n; n=$(ls "$shard/$SKEL_SUB" | wc -l | tr -d ' ')
  rollouts_and_split "$shard" $(printf '%03d ' $((n - 2)) $((n - 1)))
}

STEP=${1:-all}
if [[ $STEP == v1 || $STEP == all ]]; then
    # geo_v1 shards 0-3, rebuilt from the local copy of the Gemma-populated graphs.
    # Oracle routes are deterministic, so this reproduces betty's geo_v1 split.
    for i in 0 1 2 3; do
      sh=$ROOT/shard$i
      mkdir -p "$sh/$SKEL_SUB" "$sh/$RUN_SUB/populated_graphs"
      cp -n "$V1_SRC/shard$i"/skeletons/*.json "$sh/$SKEL_SUB/"
      cp -n "$V1_SRC/shard$i"/populated_graphs/*.json "$sh/$RUN_SUB/populated_graphs/"
      echo "=== geo_v1 shard$i"
      rollouts_and_split "$sh" 148 149
    done
fi
if [[ $STEP == train || $STEP == all ]]; then
    templated_shard 4 6001 "40:15 70:15 100:15 130:15" "$V1_HOP_WEIGHTS"
    templated_shard 5 6101 "200:${N_LARGE:-75}" "$LARGE_HOP_WEIGHTS"
    templated_shard 6 6201 "300:${N_LARGE:-75}" "$LARGE_HOP_WEIGHTS"
    templated_shard 7 6301 "400:${N_LARGE:-75}" "$LARGE_HOP_WEIGHTS"
fi
if [[ $STEP == merge || $STEP == all ]]; then
    $PY scripts/e22_geo_v2_merge.py --root "$ROOT" --gen-subdir "$RUN_SUB/split" \
      --v1-shards 0 1 2 3 --large-min-regions 200 --large-frac "${LARGE_FRAC:-0.40}" \
      --total "${TOTAL_CONV:-9000}" --seed 0
    $PY scripts/e22_corpus_audit.py --root "$ROOT" --min-coverage 0.75 || true
fi
if [[ $STEP == scale || $STEP == all ]]; then
    # geo_scale: 8 graphs x 6 tasks per size, route-length strata round robin,
    # object goals balanced, same start-role mix as training. New seeds 5001+.
    for nr in 40 130 200 400 550 700 1000; do
      d=$SCALE/n$(printf '%04d' "$nr")
      skeletons "$d/skeletons" $((5001 + nr * 10)) 8 "$nr"
      $PY scripts/e22_template_populate.py populate --vocab "$VOCAB" \
        --skeleton-dir "$d/skeletons" --out-dir "$d/test_graphs" --n-tasks 6 \
        --hop-strata "1-6,7-15,16-30,31-999" $ROLE_ARGS --seed "$SEED"
      rm -f "$d"/test_graphs/graph_gen_*.json
      mv "$d/test_graphs/longhop_manifest.json" "$d/longhop_manifest.json"  # keep only data_gen_*.json in the eval dir
    done
fi
