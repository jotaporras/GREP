#!/bin/bash
#SBATCH --job-name=e9-gnn-navigation
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# Header matches the e13f/e14 scripts: same partition, and an ABSOLUTE --output under
# $PROJ. SLURM opens that file before the job body runs and will NOT create a missing
# directory, so a relative `logs/...` path fails the job at submission with no
# diagnostic — an in-script `mkdir` cannot rescue it. cpus/mem are lower than the LLM
# jobs on purpose: this trains GNNs, not gemma-4-31B.
#
# scripts/e9_gnn_navigation.py — the three-stage navigator pretraining that produces the
# suite weights every graph arch loads as its Ψ producer:
#
#   stage 1  edge detector      -> <SUITE>/edge_detector.pt,  edge_detector_<model_type>.pt
#   stage 2  shortest-path GNN  -> <SUITE>/spd_gnn.pt,        spd_gnn_<model_type>.pt
#   stage 3  path navigator     -> <SUITE>/path_navigator.pt, path_navigator_agt.pt,
#                                  path_navigator_<model_type>.pt
#
# `path_navigator_gt.pt` is what e14_stage1to3_binary.sbatch consumes via NAV_GT, and
# `path_navigator.pt` is what experiments/e9_navigator_gt.yaml points `navigator.weights`
# at — so this job is UPSTREAM of the whole e14 suite.
#
# The .py is an nbconvert'd notebook and runs top to bottom; there is no argparse. Every
# knob is therefore an ENVIRONMENT VARIABLE, read at the top of the file:
#
#   SUITE        output subdir under outputs/e9_multistage_training  (default suite9)
#   GNN_PATH     pretrained gnn_weights.pt the run initialises from
#   TRAIN_GRAPHS n=100 training graphs    EVAL_PATH   n=100 test graphs
#   EX_PATH      n=30 worked-example graph (ex_graph / render URL)
#   SAVE_PATH    pickled eval graphs      FIG_DIR     where figures are written
#   DEVICE       cuda | mps | cpu         (default: auto-detect, cuda here)
#   TRAIN_EDGES / TRAIN_DISTS / TRAIN_PATHS
#                1 = train that stage (default), 0 = LOAD its checkpoint instead.
#                Set the earlier stages to 0 to resume a partial run without redoing them.
#
# Paths are repo-relative and anchored under $PROJ, matching the e13f/e14 convention: an
# absolute override passes through untouched, a relative one resolves under the cluster
# checkout. The script prints every resolved path before doing any work and FAILS LOUD if
# GNN_PATH / EX_PATH / EVAL_PATH is missing, so a bad path costs seconds rather than
# surfacing after the first training stage.
#
# Submit:  sbatch scripts/e9_gnn_navigation.sbatch
#   env  :  SUITE=suite10 TRAIN_EDGES=0 TRAIN_DISTS=0 \
#             sbatch scripts/e9_gnn_navigation.sbatch
# Local :  bash   scripts/e9_gnn_navigation.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3}
ALELAB=/vast/projects/aribeiro/alelab/jporras
PROJ=$ALELAB/GREP-PRISM
export PROJ

# Home dir sits near its quota; keep the HF hub cache on shared vast.
export WANDB_ENTITY=${WANDB_ENTITY:-alelab}
# Every training stage calls wandb.init unconditionally, so an unauthenticated node (or
# one without egress) kills the job at the FIRST stage rather than at the end. Set
# WANDB_MODE=offline to log to disk and sync later, or disabled to drop logging entirely.
export WANDB_MODE=${WANDB_MODE:-online}
export HF_HOME=${HF_HOME:-$ALELAB/hf_cache}
export PYTHONPATH=${PYTHONPATH:-$PROJ/src}
# SLURM 22.05+ stopped propagating --cpus-per-task to srun; pass it explicitly to avoid
# "CPU binding outside of job step allocation" on dgx-b200 nodes (same as the e14 jobs).
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"

# Stage/suite knobs — every one is passed through to the .py untouched.
export SUITE=${SUITE:-suite9}
export GNN_PATH=${GNN_PATH:-outputs/e13f_alpha_khops/e13f_alpha00_binary_nyvi1tww/gnn_weights.pt}
# Train and test on n_100. EX_PATH stays n_30: it supplies the one worked-example graph
# (render URL + demo rollout), not training or scoring data.
export TRAIN_GRAPHS=${TRAIN_GRAPHS:-data/n_100/gen/nav_n100_gemma_data/train_graphs}
export EX_PATH=${EX_PATH:-data/n_30/gen/nav100_n30_gemma_data/split/test_graphs}
export EVAL_PATH=${EVAL_PATH:-data/n_100/gen/nav_n100_gemma_data/test_graphs}
export SAVE_PATH=${SAVE_PATH:-data/pickle/e6_eval_graphs.pkl}
export FIG_DIR=${FIG_DIR:-results/e9_gnn_navigation/$SUITE}
export DEVICE=${DEVICE:-cuda}
export TRAIN_EDGES=${TRAIN_EDGES:-1}
export TRAIN_DISTS=${TRAIN_DISTS:-1}
export TRAIN_PATHS=${TRAIN_PATHS:-1}

echo "[$(date --iso-8601=seconds)] e9_gnn_navigation  suite=$SUITE  device=$DEVICE"
echo "  stages: edges=$TRAIN_EDGES dists=$TRAIN_DISTS paths=$TRAIN_PATHS"
echo "  gnn   : $GNN_PATH"
echo "  train : $TRAIN_GRAPHS"
echo "  test  : $EVAL_PATH"
echo "  example: $EX_PATH"
echo "  figs  : $FIG_DIR"

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

srun python scripts/e9_gnn_navigation.py

echo "[$(date --iso-8601=seconds)] done — weights under outputs/e9_multistage_training/$SUITE"
