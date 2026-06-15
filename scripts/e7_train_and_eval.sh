#!/usr/bin/env bash
#
# e7 augmented-graph: train, then run the scalability eval on the test set.
#
#   1. Train with train_v2.py on the e7 config.
#   2. scalability_evaluation.py on the TEST SET from the nav100_n30_gemma_data
#      split (overridable via --test-graphs).
#   3. (OPTIONAL) scalability_evaluation.py once more on a transferability graph
#      path of YOUR choosing. Only runs when a custom path is given; transfer
#      eval is NOT enforced.
#
# Usage:
#   scripts/e7_train_and_eval.sh [custom_eval_graphs] [options]
#
# Options (also settable as env vars):
#   --config PATH        e7 yaml             (CONFIG,        default: experiments/e7_architecture_improvements/e7_composite_graph_gt_centered.yaml)
#   --device N           GPU index, -1=auto  (DEVICE,        default: 0)
#   --test-graphs PATH   testing-loop graphs (TEST_GRAPHS,   default: data/gen/nav100_n30_gemma_data/split/test_graphs/data_gen_023.json)
#   --custom-graphs PATH transferability eval graphs (CUSTOM_GRAPHS, optional — skipped if unset)
#   --checkpoint PATH    eval an existing ckpt and skip training (CHECKPOINT)
#   --no-train           skip the training stage             (SKIP_TRAIN=true)
#   --no-four-bit        load the model in full precision     (FOUR_BIT=false)
#   --use-icl true|false SPINE ICL examples   (USE_ICL,       default: true)
#   --conda-env NAME     conda env to activate (CONDA_ENV,    default: GREP-PRISM)
#
# Examples:
#   scripts/e7_train_and_eval.sh                       # train + test-set eval only
#   scripts/e7_train_and_eval.sh --device 1
#   scripts/e7_train_and_eval.sh data/eval/e6_transferability/   # + optional transfer eval
#   CHECKPOINT=outputs/e7_architecture_improvements/e7_..._abcd1234 \
#       scripts/e7_train_and_eval.sh --no-train

set -euo pipefail

# --- repo root (so relative config/data paths resolve) ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- defaults (env overridable) ---
CONFIG="${CONFIG:-experiments/e7_architecture_improvements/e7_composite_graph_gt_centered.yaml}"
DEVICE="${DEVICE:-0}"
FOUR_BIT="${FOUR_BIT:-true}"
USE_ICL="${USE_ICL:-true}"
CONDA_ENV="${CONDA_ENV:-GREP-PRISM}"
SKIP_TRAIN="${SKIP_TRAIN:-false}"
CHECKPOINT="${CHECKPOINT:-}"
# testing loop: the nav100_n30_gemma_data test split (overridable via --test-graphs)
TEST_GRAPHS="${TEST_GRAPHS:-data/gen/nav100_n30_gemma_data/split/test_graphs/data_gen_023.json}"
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

# --- activate the project conda env if available (AGENTS.md: always activate) ---
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV" || echo "WARN: could not activate conda env '$CONDA_ENV'; using current python." >&2
fi
PYTHON="${PYTHON:-python}"
export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT/src"

# --- pin the GPU (mirrors scripts/e5_train_and_eval.sh's CUDA_VISIBLE_DEVICES) ---
# train_v2 hardcodes device_map={"": 0} and never reads CUDA_VISIBLE_DEVICES, so
# --device alone leaves training on physical GPU 0. Mask the chosen GPU here so it
# becomes the only visible device (which then maps to index 0). -1 = auto: leave
# unmasked and let device_map="auto" decide.
if [ "$DEVICE" = "-1" ]; then
  :  # auto
elif [ "$DEVICE" -ge 0 ] 2>/dev/null; then
  export CUDA_VISIBLE_DEVICES="$DEVICE"
else
  echo "ERROR: --device must be an integer >= -1, got: $DEVICE" >&2; exit 1
fi

# --- read checkpoint_dir from the yaml ---
CHECKPOINT_DIR="$("$PYTHON" -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['checkpoint_dir'])")"

FOUR_BIT_ARG=""
[ "$FOUR_BIT" = "true" ] && FOUR_BIT_ARG="--four-bit"

echo "=================================================================="
echo " e7 train + scalability eval"
echo "   config        : $CONFIG"
echo "   checkpoint_dir : $CHECKPOINT_DIR"
echo "   device         : $DEVICE   (four_bit=$FOUR_BIT, use_icl=$USE_ICL)"
echo "   test graphs    : $TEST_GRAPHS"
echo "   custom graphs  : ${CUSTOM_GRAPHS:-<none — transfer eval skipped>}"
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
# GREP_GEMMA_REGRADE=1 attaches the per-sample Gemma-recovered-path second reading
# (matches scripts/e5_train_and_eval.sh so both experiments grade identically).
echo; echo ">>> [2/3] Scalability eval on TEST SET from training data: $TEST_GRAPHS"
GREP_GEMMA_REGRADE=1 "$PYTHON" -m prism.eval.scalability_evaluation \
  --checkpoint "$CHECKPOINT" \
  --graphs "$TEST_GRAPHS" \
  --device "$DEVICE" \
  --use-icl "$USE_ICL" \
  $FOUR_BIT_ARG
# (text-edge-list auto-resolves from the checkpoint's gnn_config.json -> "none")

# --- 3. (OPTIONAL) transferability eval on a user-chosen path ---
# Only runs when a custom graph path was supplied; transfer eval is not enforced.
CUSTOM_OUT=""
if [ -n "$CUSTOM_GRAPHS" ]; then
  CUSTOM_OUT="$CHECKPOINT/eval_logs/custom_$(basename "${CUSTOM_GRAPHS%/}")"
  echo; echo ">>> [3/3] Transferability eval on custom graphs: $CUSTOM_GRAPHS"
  echo "    output -> $CUSTOM_OUT"
  GREP_GEMMA_REGRADE=1 "$PYTHON" -m prism.eval.scalability_evaluation \
    --checkpoint "$CHECKPOINT" \
    --graphs "$CUSTOM_GRAPHS" \
    --device "$DEVICE" \
    --use-icl "$USE_ICL" \
    --output "$CUSTOM_OUT" \
    $FOUR_BIT_ARG
else
  echo; echo ">>> [3/3] No custom graphs given — skipping transferability eval."
fi

echo; echo "=================================================================="
echo " DONE."
echo "   checkpoint     : $CHECKPOINT"
echo "   test-set eval  : $CHECKPOINT/eval_logs/cross_eval/"
[ -n "$CUSTOM_OUT" ] && echo "   custom eval    : $CUSTOM_OUT/"
echo "=================================================================="
