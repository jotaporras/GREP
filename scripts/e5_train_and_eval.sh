#!/usr/bin/env bash
#
# e5 graph-oriented data: train, then run the no-SPINE scalability eval on the
# checkpoint that *this* training run just produced.
#
# The wandb run id that train_v2 appends to the checkpoint directory name is not
# known until training starts, and several runs of the same config share the same
# name prefix (e.g. both
#   e5_rpearl_gt_llm_llama-3.1-8b_r16_4bit_zj4i517s
#   e5_rpearl_gt_llm_llama-3.1-8b_r16_4bit_v5nriwvg
# live under outputs/e5_graph_oriented_data/). So "newest mtime" is ambiguous.
# Instead we:
#   1. derive the checkpoint name *prefix* from the config (same formula as
#      train_v2: "{name}_{architecture}_{model_slug}_r{r}[_4bit]"), and
#   2. scrape the wandb run id out of the training log via regex, matching the
#      line wandb prints, e.g.:
#        wandb: 🚀 View run <run_name> at: https://wandb.ai/alelab/GREP-PRISM/runs/902tgbc7
#        wandb: Find logs at: .../wandb/run-20260610_143238-902tgbc7/logs
# The exact checkpoint dir is then "<prefix>_<run_id>".
#
# Commands run (CONFIG_PATH and GPU come from the args; GPU is a physical index
# 0,1,... or -1 to use ALL GPUs via device_map="auto"). A single index masks
# CUDA_VISIBLE_DEVICES=$GPU; -1 leaves every GPU visible (no mask):
#   [CUDA_VISIBLE_DEVICES=$GPU] uv run -m prism.training.train_v2 $CONFIG_PATH --device $GPU
#   PRISM_DISABLE_SPINE_TOOLS=1 [CUDA_VISIBLE_DEVICES=$GPU] \
#     uv run -m prism.eval.scalability_evaluation \
#       --checkpoint outputs/e5_graph_oriented_data/$MODEL/ \
#       --graphs data/gen/nav100_n30_gemma_data/split/test_graphs/ \
#       --output results/${MODEL}_no_spine --four-bit --device $GPU
#
# Usage:
#   scripts/e5_train_and_eval.sh <config_path> <gpu>     # gpu: index 0,1,... or -1 for all GPUs
# Example:
#   scripts/e5_train_and_eval.sh experiments/e5_graph_oriented_data/e5_rpearl_llm_gt_no_edges.yaml 0
#
# Env overrides:
#   GRAPHS=<path>   eval graphs (default: data/gen/nav100_n30_gemma_data/split/test_graphs/ — all samples)
#   PY=<python>     python used to parse the yaml config (default: "uv run python")
#   DRY_RUN=1       print the train/eval commands instead of running them. Needs
#                   SMOKE_LOG=<file> to supply a captured wandb log to scrape the
#                   run id from. Used by the smoke test.

set -euo pipefail

# --- repo root, so the relative config/data/output paths resolve ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GRAPHS_DEFAULT="data/gen/nav100_n30_gemma_data/split/test_graphs/"
PY="${PY:-uv run python}"

usage() { sed -n '/^# Usage:/,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2; }

# --- derive the checkpoint-name prefix from the config (mirrors train_v2.py) ---
# Prints: "{name}_{architecture}_{model_slug}_r{r}[_4bit]"  (no wandb id)
_derive_prefix() {
  local config="$1"
  $PY - "$config" <<'PYEOF'
import re, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
def slug(bm):                       # mirrors train_v2._model_short_name
    n = bm.split("/")[-1]
    n = re.sub(r"-[Ii]nstruct$", "", n)
    n = n.lower()
    n = re.sub(r"-+", "-", n)
    return n
name = cfg["name"]
arch = cfg.get("architecture", "rpearl_llm")
bm   = cfg.get("base_model", "meta-llama/Llama-3.2-3B-Instruct")
r    = cfg.get("r", 16)
bit4 = cfg.get("bit4", False)
sn   = cfg.get("save_name")         # train_v2: "{save_name}_{run_id}" overrides the formula
prefix = sn if sn else f"{name}_{arch}_{slug(bm)}_r{r}" + ("_4bit" if bit4 else "")
print(prefix)
PYEOF
}

# --- read checkpoint_dir from the config ---
_derive_checkpoint_dir() {
  local config="$1"
  $PY - "$config" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
print(cfg.get("checkpoint_dir", "outputs/e5_graph_oriented_data"))
PYEOF
}

# --- scrape the wandb run id out of a training log ---
# Matches "https://wandb.ai/<entity>/<project>/runs/<id>" first, then the local
# "run-<ts>-<id>" dir name; takes the last occurrence (the run just finished).
_extract_run_id() {
  local log="$1" rid=""
  rid="$(grep -oE 'wandb\.ai/[^[:space:]]+/runs/[A-Za-z0-9]+' "$log" 2>/dev/null | tail -n1 | sed -E 's#.*/runs/##')"
  if [ -z "$rid" ]; then
    rid="$(grep -oE 'run-[0-9]{8}_[0-9]{6}-[A-Za-z0-9]+' "$log" 2>/dev/null | tail -n1 | sed -E 's#.*-##')"
  fi
  # last resort: the wandb/latest-run symlink (run-<ts>-<id>)
  if [ -z "$rid" ] && [ -L "$REPO_ROOT/wandb/latest-run" ]; then
    rid="$(basename "$(readlink "$REPO_ROOT/wandb/latest-run")" | sed -E 's#.*-##')"
  fi
  printf '%s' "$rid"
}

# --- combine prefix + run id, verify the dir exists, echo the model name ---
_resolve_model() {
  local ckpt_dir="$1" prefix="$2" rid="$3"
  local model="${prefix}_${rid}"
  if [ ! -d "$REPO_ROOT/$ckpt_dir/$model" ]; then
    echo "ERROR: resolved checkpoint dir does not exist: $ckpt_dir/$model" >&2
    echo "       (prefix='$prefix' run_id='$rid')" >&2
    echo "       candidates under $ckpt_dir matching the prefix:" >&2
    ls -d "$REPO_ROOT/$ckpt_dir/${prefix}_"*/ 2>/dev/null | sed "s#$REPO_ROOT/#         #" >&2 || true
    return 1
  fi
  printf '%s' "$model"
}

main() {
  if [ "$#" -ne 2 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage; exit 1
  fi
  local CONFIG="$1" GPU="$2"
  local GRAPHS="${GRAPHS:-$GRAPHS_DEFAULT}"

  cd "$REPO_ROOT"

  [ -f "$CONFIG" ] || { echo "ERROR: config not found: $CONFIG" >&2; exit 1; }
  case "$GPU" in
    ''|*[!0-9-]*) echo "ERROR: gpu must be a GPU index (0,1,...) or -1 for all GPUs, got: $GPU" >&2; exit 1 ;;
  esac

  # GPU selection: a single index (0,1,...) masks to that one physical GPU; -1 uses
  # ALL visible GPUs (no masking — train_v2 and the eval loaders fall back to
  # device_map="auto" when --device is -1). --device $GPU is passed through either way.
  local -a CUDA_ENV=()
  [ "$GPU" != "-1" ] && CUDA_ENV=("CUDA_VISIBLE_DEVICES=$GPU")

  local PREFIX CKPT_DIR
  PREFIX="$(_derive_prefix "$CONFIG")"
  CKPT_DIR="$(_derive_checkpoint_dir "$CONFIG")"

  echo "=================================================================="
  echo " e5 train + no-SPINE scalability eval"
  echo "   config         : $CONFIG"
  echo "   gpu            : $GPU"
  echo "   checkpoint_dir : $CKPT_DIR"
  echo "   ckpt prefix    : ${PREFIX}_<wandb_run_id>"
  echo "   eval graphs    : $GRAPHS"
  echo "=================================================================="

  # --- 1. train (tee the log so we can scrape the wandb run id) ---
  local LOG
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo ">>> [dry-run] env ${CUDA_ENV[*]+${CUDA_ENV[*]}} uv run -m prism.training.train_v2 $CONFIG --device $GPU"
    : "${SMOKE_LOG:?DRY_RUN needs SMOKE_LOG=<captured wandb log> to scrape a run id}"
    LOG="$SMOKE_LOG"
  else
    LOG="$(mktemp "${TMPDIR:-/tmp}/e5_train_log.XXXXXX")"
    echo ">>> [1/2] Training (train_v2) -> tee $LOG"
    env ${CUDA_ENV[@]+"${CUDA_ENV[@]}"} uv run -m prism.training.train_v2 "$CONFIG" --device "$GPU" 2>&1 | tee "$LOG"
  fi

  # --- resolve the checkpoint that this run produced ---
  local RUN_ID MODEL
  RUN_ID="$(_extract_run_id "$LOG")"
  [ -n "$RUN_ID" ] || { echo "ERROR: could not scrape a wandb run id from the training log ($LOG)." >&2; exit 1; }
  echo ">>> wandb run id: $RUN_ID"
  MODEL="$(_resolve_model "$CKPT_DIR" "$PREFIX" "$RUN_ID")"
  echo ">>> resolved checkpoint: $CKPT_DIR/$MODEL"

  # --- 2. no-SPINE scalability eval on the test graphs ---
  local -a EVAL_CMD=(
    env PRISM_DISABLE_SPINE_TOOLS=1 ${CUDA_ENV[@]+"${CUDA_ENV[@]}"}
    uv run -m prism.eval.scalability_evaluation
    --checkpoint "$CKPT_DIR/$MODEL/"
    --graphs "$GRAPHS"
    --output "results/${MODEL}_no_spine"
    --four-bit
    --device "$GPU"
  )
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo ">>> [dry-run] ${EVAL_CMD[*]}"
  else
    echo; echo ">>> [2/2] Scalability eval (no SPINE)"
    "${EVAL_CMD[@]}"
  fi

  echo; echo "=================================================================="
  echo " DONE."
  echo "   checkpoint : $CKPT_DIR/$MODEL"
  echo "   results    : results/${MODEL}_no_spine"
  echo "=================================================================="
}

# Only run main when executed directly; allows the smoke test to source and unit
# test the helper functions.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  main "$@"
fi
