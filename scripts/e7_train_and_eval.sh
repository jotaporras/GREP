#!/usr/bin/env bash
#
# e7 augmented-graph: train, then run scalability eval twice.
#
#   1. Train with train_v2.py on the e7 config.
#   2. scalability_evaluation.py on the TEST SET from the training data
#      (the config's `eval_data`, overridable via --test-graphs).
#   3. scalability_evaluation.py once more on a graph path of YOUR choosing
#      (required: positional arg or --custom-graphs).
#
# Usage:
#   scripts/e7_train_and_eval.sh <custom_eval_graphs> [options]
#
# Options (also settable as env vars):
#   --config PATH        e7 yaml             (CONFIG,        default: experiments/e7_architecture_improvements/e7_composite_graph_gt_no_edges.yaml)
#   --device N           GPU index, -1=auto  (DEVICE,        default: 0)
#   --test-graphs PATH   training test set   (TEST_GRAPHS,   default: the config's eval_data)
#   --custom-graphs PATH your eval graphs     (CUSTOM_GRAPHS, required)
#   --checkpoint PATH    eval an existing ckpt and skip training (CHECKPOINT)
#   --no-train           skip the training stage             (SKIP_TRAIN=true)
#   --no-four-bit        load the model in full precision     (FOUR_BIT=false)
#   --use-icl true|false SPINE ICL examples   (USE_ICL,       default: true)
#   --conda-env NAME     conda env to activate (CONDA_ENV,    default: GREP-PRISM)
#
# Examples:
#   scripts/e7_train_and_eval.sh data/eval/e6_transferability/eval_graph_unique_100.json
#   scripts/e7_train_and_eval.sh data/eval/e6_transferability/ --device 1
#   CHECKPOINT=outputs/e7_architecture_improvements/e7_..._abcd1234 \
#       scripts/e7_train_and_eval.sh my_graphs/ --no-train

set -euo pipefail

# --- repo root (so relative config/data paths resolve) ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- defaults (env overridable) ---
CONFIG="${CONFIG:-experiments/e7_architecture_improvements/e7_composite_graph_gt_no_edges.yaml}"
DEVICE="${DEVICE:-0}"
FOUR_BIT="${FOUR_BIT:-true}"
USE_ICL="${USE_ICL:-true}"
CONDA_ENV="${CONDA_ENV:-GREP-PRISM}"
SKIP_TRAIN="${SKIP_TRAIN:-false}"
CHECKPOINT="${CHECKPOINT:-}"
TEST_GRAPHS="${TEST_GRAPHS:-}"
CUSTOM_GRAPHS="${CUSTOM_GRAPHS:-}"

# --- arg parsing (first non-flag positional => CUSTOM_GRAPHS) ---
while [ $# -gt 0 ]; do
  case "$1" in
    --config)        CONFIG="$2"; shift 2 ;;
    --device)        DEVICE="$2"; shift 2 ;;
    --test-graphs)   TEST_GRAPHS="$2"; shift 2 ;;
    --custom-graphs) CUSTOM_GRAPHS="$2"; shift 2 ;;
    --checkpoint)    CHECKPOINT="$2"; shift 2 ;;
    --no-train)      SKIP_TRAIN="true"; shift ;;
    --no-four-bit)   FOUR_BIT="false"; shift ;;
    --use-icl)       USE_ICL="$2"; shift 2 ;;
    --conda-env)     CONDA_ENV="$2"; shift 2 ;;
    -h|--help)       sed -n '2,40p' "$0"; exit 0 ;;
    --*)             echo "Unknown option: $1" >&2; exit 1 ;;
    *)               CUSTOM_GRAPHS="$1"; shift ;;
  esac
done

if [ -z "$CUSTOM_GRAPHS" ]; then
  echo "ERROR: a custom eval graph path is required (positional arg or --custom-graphs)." >&2
  echo "       e.g. scripts/e7_train_and_eval.sh data/eval/e6_transferability/" >&2
  exit 1
fi

# --- activate the project conda env if available (AGENTS.md: always activate) ---
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV" || echo "WARN: could not activate conda env '$CONDA_ENV'; using current python." >&2
fi
PYTHON="${PYTHON:-python}"
export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT/src"

# --- read checkpoint_dir + default test set from the yaml ---
CHECKPOINT_DIR="$("$PYTHON" -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['checkpoint_dir'])")"
if [ -z "$TEST_GRAPHS" ]; then
  TEST_GRAPHS="$("$PYTHON" -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['eval_data'])")"
fi

FOUR_BIT_ARG=""
[ "$FOUR_BIT" = "true" ] && FOUR_BIT_ARG="--four-bit"

echo "=================================================================="
echo " e7 train + scalability eval"
echo "   config        : $CONFIG"
echo "   checkpoint_dir : $CHECKPOINT_DIR"
echo "   device         : $DEVICE   (four_bit=$FOUR_BIT, use_icl=$USE_ICL)"
echo "   test graphs    : $TEST_GRAPHS"
echo "   custom graphs  : $CUSTOM_GRAPHS"
echo "   skip_train     : $SKIP_TRAIN"
echo "=================================================================="

# --- 1. train ---
if [ "$SKIP_TRAIN" != "true" ]; then
  echo; echo ">>> [1/3] Training (train_v2.py)"
  "$PYTHON" -m prism.training.train_v2 "$CONFIG" --device "$DEVICE"
else
  echo; echo ">>> [1/3] Skipping training (--no-train)"
fi

# --- resolve the checkpoint dir to evaluate ---
if [ -z "$CHECKPOINT" ]; then
  # newest run dir under checkpoint_dir (train_v2 appends a wandb run id we can't know ahead)
  CHECKPOINT="$(ls -dt "$CHECKPOINT_DIR"/*/ 2>/dev/null | head -1 || true)"
  CHECKPOINT="${CHECKPOINT%/}"
fi
if [ -z "$CHECKPOINT" ] || [ ! -d "$CHECKPOINT" ]; then
  echo "ERROR: could not resolve a checkpoint dir under '$CHECKPOINT_DIR'." >&2
  echo "       Pass one explicitly with --checkpoint." >&2
  exit 1
fi
echo; echo "Resolved checkpoint: $CHECKPOINT"

# --- 2. eval on the training-data test set ---
echo; echo ">>> [2/3] Scalability eval on TEST SET from training data: $TEST_GRAPHS"
"$PYTHON" -m prism.eval.scalability_evaluation \
  --checkpoint "$CHECKPOINT" \
  --graphs "$TEST_GRAPHS" \
  --device "$DEVICE" \
  --use-icl "$USE_ICL" \
  $FOUR_BIT_ARG
# (text-edge-list auto-resolves from the checkpoint's gnn_config.json -> "none")

# --- 3. eval on the user-chosen path (separate output dir) ---
CUSTOM_OUT="$CHECKPOINT/eval_logs/custom_$(basename "${CUSTOM_GRAPHS%/}")"
echo; echo ">>> [3/3] Scalability eval on custom graphs: $CUSTOM_GRAPHS"
echo "    output -> $CUSTOM_OUT"
"$PYTHON" -m prism.eval.scalability_evaluation \
  --checkpoint "$CHECKPOINT" \
  --graphs "$CUSTOM_GRAPHS" \
  --device "$DEVICE" \
  --use-icl "$USE_ICL" \
  --output "$CUSTOM_OUT" \
  $FOUR_BIT_ARG

echo; echo "=================================================================="
echo " DONE."
echo "   checkpoint     : $CHECKPOINT"
echo "   test-set eval  : $CHECKPOINT/eval_logs/cross_eval/"
echo "   custom eval    : $CUSTOM_OUT/"
echo "=================================================================="
