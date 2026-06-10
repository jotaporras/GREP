#!/usr/bin/env bash
# Gemma single-example probe: 1 graph (~30 nodes), 10 navigation-only tasks,
# populated + rolled out entirely by the LOCAL Gemma 4 backend.
#
# Run this BEFORE scripts/nav100_n30_gemma_generate.sh to spot-check task and
# rollout quality (and confirm the model loads) before committing compute to
# the full 10-graph run.
#
#   * Graph size : 3 communities x 8 nodes = 24 regions + ~30% objects ~= 30 nodes.
#   * Task mix   : --task-proportions 0 0 0 1  => 100% Navigability.
#   * Backend    : PRISM_LLM_BACKEND=hf  (Phase 1 + Phase 2 both local Gemma).
#
# No train/val split is produced here — split_train_val.py needs >= 2 graphs.
#
# Usage: bash scripts/nav100_n30_gemma_single_example.sh [GPU_ID] [MODEL]
#   GPU_ID  single CUDA device id to run on (default 0), OR -1 to use ALL GPUs
#           simultaneously (device_map="auto" shards the model across them).
#   MODEL   HF model id (default google/gemma-4-26B-A4B-it; e.g. pass
#           google/gemma-4-31B-it for the dense 31B). Overrides $PRISM_HF_MODEL.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "${REPO_ROOT}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

# --- GPU selection: single-GPU pin by id, or -1 for ALL GPUs ---
# Exported BEFORE any python/torch starts so CUDA enumerates exactly the chosen
# device(s). NOTE: CUDA_VISIBLE_DEVICES=-1 would hide ALL GPUs, so for the
# multi-GPU case we leave it unset (all visible) and drop the single-GPU assert.
GPU_ID="${1:-0}"
if [[ "$GPU_ID" == "-1" ]]; then
  # Multi-GPU: all visible GPUs; device_map="auto" shards the model across them.
  export PRISM_REQUIRE_SINGLE_GPU=0
elif [[ "$GPU_ID" == *,* || -z "$GPU_ID" ]]; then
  echo "ERROR: pass a single GPU id, or -1 for ALL GPUs (got '${GPU_ID}'). Usage: $0 [GPU_ID] [MODEL]" >&2
  exit 2
else
  # Single-GPU GUARANTEE: only this device is ever visible to CUDA.
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  export PRISM_REQUIRE_SINGLE_GPU=1   # backstop assertion inside local_llm.py
fi

# --- Local Gemma backend selection (consumed by prism.data.local_llm) ---
# Model precedence: positional arg $2 > $PRISM_HF_MODEL > default 26B-A4B (MoE,
# ~4B active). google/* checkpoints load via eager transformers; an NVFP4 repo
# does NOT (serve that with vLLM). PRISM_HF_QUANT=4bit -> NF4 (~13GB for 26B-A4B).
export PRISM_LLM_BACKEND=hf
export PRISM_HF_MODEL="${2:-${PRISM_HF_MODEL:-google/gemma-4-26B-A4B-it}}"
export PRISM_HF_QUANT="${PRISM_HF_QUANT:-4bit}"

# --- Quiet transformers / HuggingFace warnings, progress bars & banners ---
export TRANSFORMERS_VERBOSITY=error
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export BITSANDBYTES_NOWELCOME=1

SKEL_DIR="data/training/nav100_n30_gemma_single_skeleton"
RUN_DIR="data/gen/nav100_n30_gemma_single_example"

N_TASKS=10
TASK_PROPORTIONS="0 0 0 1"
SEED=42

# One ~30-node graph (matches the full-run config: 3 communities x 8 nodes).
NC=3
NPC=8

mkdir -p "$SKEL_DIR"

echo "=== Backend: local HF Transformers ==="
echo "    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>} (require_single_gpu=${PRISM_REQUIRE_SINGLE_GPU})"
echo "    PRISM_LLM_BACKEND=${PRISM_LLM_BACKEND}"
echo "    PRISM_HF_MODEL=${PRISM_HF_MODEL}"
echo "    PRISM_HF_QUANT=${PRISM_HF_QUANT}"

echo ""
echo "=== Preflight: verify data-gen code is current (fails fast vs. a 12-min load on a stale file) ==="
python - <<'PY'
import inspect, sys
try:
    from prism.data import graph_gen
except Exception as e:
    sys.stderr.write(f"PREFLIGHT FAIL: cannot import prism.data.graph_gen: {e}\n")
    sys.exit(1)
params = inspect.signature(graph_gen.TaskGraphGen.get_tasks).parameters
missing = [p for p in ("task_complexities", "reasoning_effort") if p not in params]
if missing:
    sys.stderr.write(
        "PREFLIGHT FAIL: graph_gen.TaskGraphGen.get_tasks() is missing "
        f"{missing}. This src/prism/data/graph_gen.py is STALE — copy the "
        "updated file to this machine before running.\n"
    )
    sys.exit(1)
print("preflight OK: get_tasks accepts task_complexities + reasoning_effort")
PY

echo ""
echo "=== Stage 1: Generate one skeleton (${NC} communities x ${NPC} nodes/community, ${N_TASKS} tasks) ==="

# Reuse-existing guard: if the skeleton already exists AND a graph has already
# been populated, keep what is on disk so populate resumes against it.
SKEL_FILE="${SKEL_DIR}/eval_graph_unique_30_seed${SEED}.json"
populated_count=$(find "${RUN_DIR}/populated_graphs" -maxdepth 1 -name 'data_gen_*.json' 2>/dev/null | wc -l | tr -d ' ')
if [ -f "$SKEL_FILE" ] && [ "$populated_count" -gt 0 ]; then
  echo "Reusing existing data: skeleton ${SKEL_FILE} present and "
  echo "${populated_count} populated graph(s) in ${RUN_DIR}/populated_graphs — skipping skeleton generation."
else
  python scripts/generate_eval_graphs.py \
    --n-communities "$NC" \
    --nodes-per-community "$NPC" \
    --intra-community-prob 0.6 \
    --inter-community-prob 0.05 \
    --object-rate 0.3 \
    --description-prob 0.05 \
    --n-tasks "$N_TASKS" \
    --seed "$SEED" \
    --output "$SKEL_FILE"
fi

echo ""
echo "=== Stage 2: Populate (local Gemma) + run SPINE rollouts (local Gemma) ==="
echo "    task mix: ${TASK_PROPORTIONS} (EXIST POS REACH NAV)"
# Auto-resume loop: a fatal GPU fault (CUDA launch failure, OOM, device assert)
# makes the python step exit nonzero, but both pipeline phases are resumable, so
# we just re-invoke it — each run skips completed graphs/rollouts and continues.
# Re-running reloads the model (~minutes); that is required for a fresh CUDA
# context. Tunable via PRISM_MAX_ATTEMPTS / PRISM_RETRY_SLEEP.
MAX_ATTEMPTS="${PRISM_MAX_ATTEMPTS:-5}"
RETRY_SLEEP="${PRISM_RETRY_SLEEP:-10}"
attempt=1
while true; do
  # shellcheck disable=SC2086
  if python scripts/training_data_generation/generate_data_spine.py \
      --data-dir "$SKEL_DIR" \
      --name "$RUN_DIR" \
      --task-proportions $TASK_PROPORTIONS \
      --n-tasks "$N_TASKS" \
      --max-graphs 1 \
      --seed "$SEED"; then
    echo "Stage 2 completed on attempt ${attempt}/${MAX_ATTEMPTS}."
    break
  else
    rc=$?
    if [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
      echo "ERROR: Stage 2 still failing after ${MAX_ATTEMPTS} attempts (last rc=${rc})." >&2
      echo "       Re-run the script to resume, or investigate the GPU fault." >&2
      exit "$rc"
    fi
    echo "Stage 2 exited nonzero (attempt ${attempt}/${MAX_ATTEMPTS}, rc=${rc}); resuming in ${RETRY_SLEEP}s..." >&2
    attempt=$((attempt + 1))
    sleep "$RETRY_SLEEP"
  fi
done

echo ""
echo "=== Stage 3: Leak check (must print CLEAN) ==="
if grep -RlE 'acceptance_criterion|"answer"[[:space:]]*:' "${RUN_DIR}/generated_plans/" >/dev/null 2>&1; then
  echo "LEAK in generated_plans/ — investigate before trusting this data"
  exit 2
else
  echo "CLEAN: generated_plans/"
fi

echo ""
echo "=== Stage 4: NEXT — spot-check quality before the full run ==="
echo "  /verify-tasks ${RUN_DIR}/populated_graphs --sample-size 1"
echo ""
echo "  Then inspect the ${N_TASKS} rollouts in ${RUN_DIR}/generated_plans/."
echo "  If quality looks good, run: bash scripts/nav100_n30_gemma_generate.sh"
