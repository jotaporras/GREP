#!/bin/bash
#SBATCH --job-name=e13f-alpha00-binary
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#
# e13f binary-adjacency A/B: the e13f_alpha00 recipe (pure learned sim on legal
# edges, mask_alpha=0.0, nav-GT Psi x decode_consistent, slr=0.012) trained on
# the PLAIN SBM adjacency instead of the historical Gaussian edge affinity.
# Single-flag change vs scripts/e13f_alpha_khops.sbatch task 0:
#
#   data.edge_weights=binary   (no edge_weight; plain adjacency into the PE GT)
#
# Identifiers carry a _binary suffix (save_name / run_name / tag) so this run is
# distinguishable from the Gaussian e13f_alpha00 in outputs/e13f_alpha_khops and
# in wandb. edge_weights is recorded in train_config.json and threaded to eval.
#
# The wandb names are overridable, in priority order:
#   positional arg  >  same-named env var  >  built-in default.
#   RUN_NAME (arg 1) -> wandb.run_name AND trainer.save_name (checkpoint subdir)
#   TAG      (arg 2) -> wandb.tag (run group)
#
# Submit:  sbatch scripts/e13f_alpha00_binary.sbatch [RUN_NAME] [TAG]
#   env  :  RUN_NAME=... TAG=... sbatch scripts/e13f_alpha00_binary.sbatch
# Local :  bash   scripts/e13f_alpha00_binary.sbatch [RUN_NAME] [TAG]

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3}
ALELAB=/vast/projects/aribeiro/alelab/jporras
PROJ=$ALELAB/GREP-PRISM
CHECKPOINT_DIR=$PROJ/outputs/e13f_alpha_khops
PE_DIR=$PROJ/pretrained_pe
STAGE1_LORA=$PROJ/outputs/e9_multistage_training/e9_ms_stage1/e9_ms_stage1_sqgk4o3j/checkpoint-100
TEST_GRAPHS=data/revised/gen/nav100_n30_gemma_data/split/test_graphs

# wandb names: positional arg > env var > default. RUN_NAME also drives the
# checkpoint subdir (trainer.save_name) so the on-disk run stays identifiable.
RUN_NAME=${1:-${RUN_NAME:-e13f_alpha00_binary}}
TAG=${2:-${TAG:-e13f_alpha_khops}}

# Home dir sits near its quota; keep the HF hub cache (base gemma-4-31B-it
# weights) on shared vast — downloaded once, reused across jobs.
export WANDB_ENTITY=${WANDB_ENTITY:-alelab}
export HF_HOME=${HF_HOME:-$ALELAB/hf_cache}
export HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf_datasets_${SLURM_JOB_ID:-manual}"
# SLURM 22.05+ stopped propagating --cpus-per-task to srun; pass it explicitly
# to avoid "CPU binding outside of job step allocation" on dgx-b200 nodes.
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-14}"

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "[$(date --iso-8601=seconds)] $RUN_NAME training (edge_weights=binary, tag=$TAG)"

srun python -m prism.training.train_v3 \
    --config-name=e9_base_config \
    model.path="google/gemma-4-31B-it" \
    gnn.arch=learnable_graph_mask \
    gnn.d_model=1024 gnn.dropout=0.1 gnn.pe_hidden_channels=256 gnn.pe_num_layers=5 \
    gnn.num_samples=320 gnn.k_pe=3 gnn.gt_num_layers=3 gnn.gt_heads=8 gnn.k_gt=2 \
    gnn.eps=1e-6 gnn.use_layer_norm=true gnn.pe_node_features=random \
    gnn.mask_alpha=0.0 gnn.mask_layer_scope=dense gnn.mask_psi_scale=cosine \
    gnn.mask_k_hops=1 gnn.mask_symmetrize=true gnn.mask_use_edges=true \
    gnn.pe_gt_from="$PE_DIR/path_navigator_gt.pt" \
    gnn.semantic_gt_from=null \
    gnn.structural_lr_mult=0.012 \
    data.text_edge_list=none \
    data.injection_scope=decode_consistent \
    data.edge_weights=binary \
    trainer.init_lora_from="$STAGE1_LORA" \
    trainer.freeze_lora=false trainer.freeze_pe=false \
    trainer.learning_rate=0.00025 trainer.epochs=3 \
    +trainer.sft.save_total_limit=1 \
    trainer.checkpoint_dir="$CHECKPOINT_DIR" \
    trainer.save_name="$RUN_NAME" \
    wandb.run_name="$RUN_NAME" wandb.tag="$TAG" \
    eval.data="$TEST_GRAPHS" eval.num_graphs=-1

echo "[$(date --iso-8601=seconds)] e13f_alpha00_binary DONE"
