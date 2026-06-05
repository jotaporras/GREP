#!/usr/bin/env bash
# Generate navigation-only (~30-node) training data with a LOCAL Gemma 4
# model via HF Transformers — no OpenAI calls in either phase.
#
#   * Graph size : 3 communities x 8 nodes = 24 regions + ~30% objects ~= 30 nodes.
#   * Task mix   : --task-proportions 0 0 0 1  => 100% Navigability.
#   * LLM backend: PRISM_LLM_BACKEND=hf routes BOTH phases through the local model:
#       Phase 1 (populate) -> prism.data.local_llm.LocalHFQueryClient
#       Phase 2 (rollouts) -> prism.data.local_llm.GemmaSpineClient (SPINE client)
#     The model is loaded ONCE per process and shared across both phases.
#
# Weights: default google/gemma-4-26B-A4B-it (MoE, ~4B active). PRISM_HF_QUANT
# controls footprint for 26B-A4B: none=bf16 (~52GB), 4bit=NF4 (~13GB, single
# 24GB GPU), 8bit (~26GB). Pass google/gemma-4-31B-it as MODEL for the dense 31B.
# NOTE: an NVFP4 checkpoint will NOT load via eager transformers (FP4
# weight_scale tensors are not materialised); serve it with vLLM instead.
#
# Runtime deps on the GPU node:
#   pip install -U transformers torch accelerate
#   pip install -U bitsandbytes   # only for PRISM_HF_QUANT=4bit/8bit
#
# Run scripts/nav100_n30_gemma_single_example.sh spot-check first if you want to
# validate task/rollout quality before paying for the full run (compute, not $).
#
# Usage: bash scripts/nav100_n30_gemma_generate.sh [GPU_ID] [MODEL] [N_GRAPHS]
#   GPU_ID   single CUDA device id to run on (default 0), OR -1 to use ALL GPUs
#            simultaneously (device_map="auto" shards the model across them).
#   MODEL    HF model id (default google/gemma-4-26B-A4B-it). Overrides $PRISM_HF_MODEL.
#   N_GRAPHS number of graphs to generate (default 36). Training examples ~=
#            N_GRAPHS * 10 tasks * success_rate * 0.8 (train split); 36 targets
#            >=180 training examples with headroom.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# .env still sourced for any HF_TOKEN / HF_HOME / CUDA settings; OPENAI_API_KEY
# is NOT required with the local backend.
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
# Model precedence: positional arg $2 > $PRISM_HF_MODEL > default 26B-A4B (MoE).
export PRISM_LLM_BACKEND=hf
export PRISM_HF_MODEL="${2:-${PRISM_HF_MODEL:-google/gemma-4-26B-A4B-it}}"
# 4bit (NF4) fits a single 24GB GPU (~13GB for 26B-A4B); none=bf16 (~52GB).
export PRISM_HF_QUANT="${PRISM_HF_QUANT:-4bit}"

# --- Quiet transformers / HuggingFace warnings, progress bars & banners ---
export TRANSFORMERS_VERBOSITY=error
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export BITSANDBYTES_NOWELCOME=1

SKEL_DIR="data/training/nav100_n30_skeletons"
RUN_DIR="data/gen/nav100_n30_gemma_data"
SPLIT_DIR="${RUN_DIR}/split"

INTRA_PROB=0.6
INTER_PROB=0.05
OBJECT_RATE=0.3
DESC_PROB=0.05
N_TASKS=10

# 100% navigation (EXIST POS REACH NAV). GEN_SEED makes task-type sampling reproducible.
TASK_PROPORTIONS="0 0 0 1"
GEN_SEED=42

# Single ~30-node config (3 communities x 8 nodes = 24 regions; +~30% objects).
# Using one config keeps the AVERAGE graph size at ~30 nodes.
NC=3
NPC=8

# Number of graphs (skeletons). Each graph yields up to N_TASKS rollouts, and
# the train/val split keeps ~80% for training, so training examples are roughly
#   N_GRAPHS * N_TASKS * rollout_success_rate * 0.8.
# Default 36 (x10 tasks = 360 pairs) clears the >=180 training-example target
# with headroom even at a ~65% rollout success rate. Override as the 3rd arg.
N_GRAPHS="${3:-36}"
# Distinct per-graph seeds (101..), embedded in skeleton filenames for provenance.
seeds=( $(seq 101 $((100 + N_GRAPHS))) )

mkdir -p "$SKEL_DIR"

echo "=== Backend: local HF Transformers ==="
echo "    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>} (require_single_gpu=${PRISM_REQUIRE_SINGLE_GPU})"
echo "    PRISM_LLM_BACKEND=${PRISM_LLM_BACKEND}"
echo "    PRISM_HF_MODEL=${PRISM_HF_MODEL}"
echo "    PRISM_HF_QUANT=${PRISM_HF_QUANT}"
echo "    N_GRAPHS=${N_GRAPHS} x N_TASKS=${N_TASKS} = $((N_GRAPHS * N_TASKS)) (graph,task) pairs"

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
echo "=== Stage 1: Generate skeletons (${#seeds[@]} seeds x 1 config = ${#seeds[@]} skeletons, ~30 nodes each) ==="
for SEED in "${seeds[@]}"; do
  echo "--- N~30 (${NC} communities x ${NPC} nodes/community) seed=${SEED} ---"
  python scripts/generate_eval_graphs.py \
    --n-communities "$NC" \
    --nodes-per-community "$NPC" \
    --intra-community-prob "$INTRA_PROB" \
    --inter-community-prob "$INTER_PROB" \
    --object-rate "$OBJECT_RATE" \
    --description-prob "$DESC_PROB" \
    --n-tasks "$N_TASKS" \
    --seed "$SEED" \
    --output "${SKEL_DIR}/eval_graph_unique_30_seed${SEED}.json"
done

echo ""
echo "=== Stage 2: Populate (local Gemma) + run SPINE rollouts (local Gemma) ==="
echo "    task mix: ${TASK_PROPORTIONS} (EXIST POS REACH NAV)"
# shellcheck disable=SC2086
python scripts/training_data_generation/generate_data_spine.py \
  --data-dir "$SKEL_DIR" \
  --name "$RUN_DIR" \
  --task-proportions $TASK_PROPORTIONS \
  --n-tasks "$N_TASKS" \
  --seed "$GEN_SEED"

echo ""
echo "=== Stage 3: Train/val split ==="
python scripts/training_data_generation/split_train_val.py \
  --plans-dir "${RUN_DIR}/generated_plans" \
  --graphs-dir "${RUN_DIR}/populated_graphs" \
  --output-dir "$SPLIT_DIR"

echo ""
echo "=== Stage 4: Leak check (must print CLEAN twice) ==="
if grep -RlE 'acceptance_criterion|"answer"[[:space:]]*:' "${RUN_DIR}/generated_plans/" >/dev/null 2>&1; then
  echo "LEAK in generated_plans/ — investigate before training on this data"
  exit 2
else
  echo "CLEAN: generated_plans/"
fi

if [ -f "${RUN_DIR}/generated_plans/formatted.json" ]; then
  if grep -E 'acceptance_criterion|"answer"[[:space:]]*:' "${RUN_DIR}/generated_plans/formatted.json" >/dev/null 2>&1; then
    echo "LEAK in formatted.json — investigate before training on this data"
    exit 2
  else
    echo "CLEAN: formatted.json"
  fi
else
  echo "WARN: ${RUN_DIR}/generated_plans/formatted.json does not exist (aggregate may have skipped)"
fi

echo ""
echo "=== Stage 5: NEXT — open Claude Code and run: ==="
echo "  audit the run at ${RUN_DIR}"
