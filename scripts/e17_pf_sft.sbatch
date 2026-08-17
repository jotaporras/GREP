#!/bin/bash
#SBATCH --job-name=e17-pf-sft
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
# betty's SLURM cli_filter (2026-08-03) enforces EXACTLY 28 CPUs / 224 GB per
# GPU on dgx-b200.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#
# e17 — SFT of the POST-FUSION mask architecture (candidate A: late-layer
# gated residual injection of projected Psi, gnn.post_fusion=true). Recipe is
# otherwise the e13f_alpha00_binary one verbatim (mask_alpha=0, binary
# adjacency, decode_consistent, slr=0.012), so any delta vs the e13f numbers
# is attributable to the post-fusion pathway. The pf modules (pf_proj /
# pf_norm / pf_gain) are structural parameters and train in the same LR group
# as the Psi tower.
#
# Dataset is parameterized: DATA_SPLIT points at a split/ dir containing
# formatted_all_new__{train,val}.json and test_graphs.
#
# Submit:  DATA_SPLIT=... RUN_NAME=... sbatch scripts/e17_pf_sft.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

ALELAB=/vast/projects/aribeiro/alelab/jporras
PROJ=$ALELAB/GREP-PRISM
ENV_NAME=${ENV_NAME:-$ALELAB/envs/GREP-PRISM-v3}
CHECKPOINT_DIR=$PROJ/outputs/e17_post_fusion
PE_DIR=$PROJ/pretrained_pe
STAGE1_LORA=$PROJ/outputs/e9_multistage_training/e9_ms_stage1/e9_ms_stage1_sqgk4o3j/checkpoint-100

# split/ dir with formatted_all_new__{train,val}.json + test_graphs.
# Default: the historical n30 corpus (the e13f 86% reference dataset).
DATA_SPLIT=${DATA_SPLIT:-data/revised/gen/nav100_n30_gemma_data/split}

RUN_NAME=${1:-${RUN_NAME:-e17_pf_sft}}
TAG=${2:-${TAG:-e17_post_fusion}}

export WANDB_ENTITY=${WANDB_ENTITY:-alelab}
export HF_HOME=${HF_HOME:-$ALELAB/hf_cache}
export HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf_datasets_${SLURM_JOB_ID:-manual}"
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-28}"

source "$ENV_NAME/bin/activate"

echo "[$(date --iso-8601=seconds)] $RUN_NAME e17 post-fusion SFT (data=$DATA_SPLIT, tag=$TAG)"

srun python -m prism.training.train_v3 \
    --config-name=e9_base_config \
    model.path="google/gemma-4-31B-it" \
    gnn.arch=learnable_graph_mask \
    gnn.d_model=1024 gnn.dropout=0.1 gnn.pe_hidden_channels=256 gnn.pe_num_layers=5 \
    gnn.num_samples=320 gnn.k_pe=3 gnn.gt_num_layers=3 gnn.gt_heads=8 gnn.k_gt=2 \
    gnn.eps=1e-6 gnn.use_layer_norm=true gnn.pe_node_features=random \
    gnn.mask_alpha=0.0 gnn.mask_layer_scope=dense gnn.mask_psi_scale=cosine \
    gnn.mask_k_hops=1 gnn.mask_symmetrize=true gnn.mask_use_edges=true \
    gnn.post_fusion=true \
    gnn.pe_gt_from="$PE_DIR/path_navigator_gt.pt" \
    gnn.semantic_gt_from=null \
    gnn.structural_lr_mult=0.012 \
    data.train_files="$DATA_SPLIT/formatted_all_new__train.json" \
    data.val_files="$DATA_SPLIT/formatted_all_new__val.json" \
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
    eval.data="$DATA_SPLIT/test_graphs" eval.num_graphs=-1

echo "[$(date --iso-8601=seconds)] $RUN_NAME DONE"
