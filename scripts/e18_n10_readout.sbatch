#!/bin/bash
#SBATCH --job-name=e18-readout
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%j.out
#SBATCH --time=01:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
# betty's SLURM cli_filter (2026-08-03) enforces EXACTLY 28 CPUs / 224 GB per
# GPU on dgx-b200.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#
# e18 — the READOUT half of e18_n10_sft.sbatch on an EXISTING run dir: held-out
# navigation eval (same evaluate_model as the in-process post-train eval, same
# eval_logs/cross_eval/ output — overwrites it) + the neighbour-naming probe on
# test_graphs and a few train graphs. For a run whose training finished but
# whose readout died (7731158 / 7731160: decision_gain on CPU at reload), or to
# re-probe a checkpoint after a probe change, without retraining.
#
# Submit:  RUN_DIR=$PROJ/outputs/e18_identity/e18_n10_mask_a_x95yrcmf sbatch scripts/e18_n10_readout.sbatch
# RUN_NAME defaults to the run dir's basename minus the wandb id suffix
# (e18_n10_mask_a_x95yrcmf -> e18_n10_mask_a), so the probe files land where
# the training job would have put them. Override DATA_SPLIT for the n15 corpus.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

ALELAB=/vast/projects/aribeiro/alelab/jporras
PROJ=$ALELAB/GREP-PRISM
ENV_NAME=${ENV_NAME:-$ALELAB/envs/GREP-PRISM-v3}

RUN_DIR="${RUN_DIR:?set RUN_DIR=<run dir with train_config.json / gnn_weights.pt>}"
DATA_SPLIT=${DATA_SPLIT:-$PROJ/data/n_10_vllm/gen/nav_n10_gemma_data/split}
RUN_NAME=${RUN_NAME:-$(basename "$RUN_DIR" | sed -E 's/_[a-z0-9]{8}$//')}
PROBE_TRAIN_GRAPHS=${PROBE_TRAIN_GRAPHS:-3}
PROBE_OUT=${PROBE_OUT:-$PROJ/results/e18_probe}

export HF_HOME=${HF_HOME:-$ALELAB/hf_cache}
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-28}"
export PATH="$ENV_NAME/bin:$PATH"

test -f "$RUN_DIR/train_config.json" || { echo "missing $RUN_DIR/train_config.json"; exit 2; }
test -d "$DATA_SPLIT/test_graphs" || { echo "missing $DATA_SPLIT/test_graphs"; exit 2; }

echo "[$(date --iso-8601=seconds)] readout $RUN_NAME run_dir=$RUN_DIR data=$DATA_SPLIT"
echo "=== python: $(which python) ==="

srun python -m prism.eval.scalability_evaluation \
    --checkpoint "$RUN_DIR" \
    --graphs "$DATA_SPLIT/test_graphs" \
    --device 0

mkdir -p "$PROBE_OUT"
srun python scripts/neighbour_probe.py --checkpoint "$RUN_DIR" \
    --graphs "$DATA_SPLIT/test_graphs" \
    --output "$PROBE_OUT/${RUN_NAME}_test.json"
srun python scripts/neighbour_probe.py --checkpoint "$RUN_DIR" \
    --graphs "$DATA_SPLIT/train_graphs" --max-graphs "$PROBE_TRAIN_GRAPHS" \
    --output "$PROBE_OUT/${RUN_NAME}_train.json"
echo "[$(date --iso-8601=seconds)] $RUN_NAME READOUT DONE"
