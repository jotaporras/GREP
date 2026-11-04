#!/bin/bash
#SBATCH --job-name=e22-epoch-snap
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=genoa-std-mem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
# no --mem: the cli_filter derives memory from CPUs.
#
# e22: snapshot one per-epoch checkpoint of a RUNNING SFT job and cross_eval it
# on set B (geo_test). Needed because e22_geo_sft.sbatch keeps only the latest
# checkpoint (save_total_limit=1) while the design picks the checkpoint by the
# best epoch on B (docs/2026-11-02 e22_geometric_scale_design.md §5).
#
#   1. poll (every 5 min) until <RUN_DIR>/checkpoint-<STEP> is complete
#      (adapter + gnn_weights.pt + train_config.json + trainer_state.json);
#   2. rsync it to <RUN_DIR>_ep<EPOCH> without optimizer/scheduler/rng state;
#   3. sbatch scripts/e22_baseline_eval.sbatch on the snapshot
#      -> $PROJ/results/e22_geometric_scale/<run basename>_ep<EPOCH>/geo_test.
#
# Exits 1 if the training job ends before the checkpoint appears. Runs as a CPU
# job, never on the login node.
#
# Submit:  RUN_DIR=<run dir> STEP=4953 EPOCH=1 TRAIN_JOB=10020785 \
#            sbatch scripts/e22_epoch_snapshot_eval.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

ALELAB=/vast/projects/aribeiro/alelab/jporras
PROJ=$ALELAB/GREP-PRISM
RUN_DIR="${RUN_DIR:?set RUN_DIR=<SFT run dir>}"
STEP="${STEP:?set STEP=<global step of the epoch checkpoint>}"
EPOCH="${EPOCH:?set EPOCH=<epoch number>}"
TRAIN_JOB="${TRAIN_JOB:?set TRAIN_JOB=<SLURM id of the SFT job>}"
GRAPHS=${GRAPHS:-$PROJ/data/n_geo_v1_test/shard4/gen/nav_ngeo_gemma_data/split/test_graphs}
SET_NAME=${SET_NAME:-geo_test}
POLL=${POLL:-300}

CK="$RUN_DIR/checkpoint-$STEP"
SNAP="${RUN_DIR}_ep${EPOCH}"
OUT=$PROJ/results/e22_geometric_scale/$(basename "$SNAP")/$SET_NAME

echo "[$(date --iso-8601=seconds)] waiting for $CK (train job $TRAIN_JOB)"
while true; do
  if [[ -f "$CK/adapter_model.safetensors" && -f "$CK/gnn_weights.pt" \
        && -f "$CK/train_config.json" && -f "$CK/trainer_state.json" ]]; then
    break
  fi
  state=$(sacct -j "$TRAIN_JOB" -X --format=State -n 2>/dev/null | tr -d ' ' | head -1 || true)
  case "$state" in
    COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL)
      echo "train job $TRAIN_JOB is $state and $CK never appeared"; exit 1 ;;
  esac
  sleep "$POLL"
done
# trainer_state.json is written last, but give the adapter shard a safety margin.
sleep 120

echo "[$(date --iso-8601=seconds)] snapshot $CK -> $SNAP"
mkdir -p "$SNAP"
rsync -a --exclude optimizer.pt --exclude scheduler.pt --exclude rng_state.pth "$CK/" "$SNAP/"
ls -la "$SNAP"

# Nested sbatch inherits this job's SLURM_* environment; the child's srun then
# dies with "SLURM_MEM_PER_CPU, SLURM_MEM_PER_GPU, and SLURM_MEM_PER_NODE are
# mutually exclusive" (job 10037721). Drop every SLURM_* variable first.
UNSET_SLURM=$(env | grep -oE '^SLURM_[A-Za-z0-9_]+' | sed 's/^/-u /' | tr '\n' ' ')
JOB=$(env $UNSET_SLURM RUN_DIR="$SNAP" GRAPHS="$GRAPHS" SET_NAME="$SET_NAME" OUT="$OUT" \
      sbatch --parsable --job-name="e22-ep${EPOCH}-${SET_NAME}" scripts/e22_baseline_eval.sbatch)
echo "[$(date --iso-8601=seconds)] submitted cross_eval job $JOB -> $OUT"
