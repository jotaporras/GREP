#!/bin/bash
#SBATCH --job-name=e13e-scope
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#SBATCH --array=0-1
#
# e13e: mask-layer-scope ablation on the BEST configuration (e13c_rpe_decode,
# 0.73 = learnable RPE mask + decode_consistent, edge-detector Ψ, slr=5.0).
# Single-flag A/Bs vs that run: ONLY gnn.mask_layer_scope changes.
#
#   idx  arm                  scope           layers carrying the mask (gemma-4-31B)
#   0    e13e_rpe_tophalf     dense_top_half  deeper 5 of the 10 global layers
#   1    e13e_rpe_firstglobal dense_first     first global layer only (1 of 60)
#
# Reference: e13c_rpe_decode (dense = all 10 globals) = 0.73 / halluc 0.06.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3}
ALELAB=/vast/projects/aribeiro/alelab/jporras
CHECKPOINT_DIR=$ALELAB/GREP-PRISM/outputs/e13e_layer_scope
STAGE1_LORA=$ALELAB/GREP-PRISM/outputs/e9_multistage_training/e9_ms_stage1/e9_ms_stage1_sqgk4o3j/checkpoint-100
PE_INIT_DIR=$ALELAB/GREP-PRISM/outputs/dev_e10/pe_init_edge_detector_gt
TEST_GRAPHS=data/revised/gen/nav100_n30_gemma_data/split/test_graphs

export WANDB_ENTITY=${WANDB_ENTITY:-alelab}
export HF_HOME=/vast/projects/aribeiro/alelab/jporras/hf_cache
export HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf_datasets_${SLURM_JOB_ID:-manual}"

module load anaconda3
source activate "$ENV_NAME"

TASK_ID=${SLURM_ARRAY_TASK_ID:-${1:-0}}
case "$TASK_ID" in
    0) NAME=e13e_rpe_tophalf;     SCOPE=dense_top_half ;;
    1) NAME=e13e_rpe_firstglobal; SCOPE=dense_first ;;
    *) echo "bad task id $TASK_ID" >&2; exit 1 ;;
esac
echo "[$(date --iso-8601=seconds)] e13e task ${TASK_ID}: $NAME (mask_layer_scope=$SCOPE)"

srun python -m prism.training.train_v3 \
    --config-name=e9_base_config \
    model.path="google/gemma-4-31B-it" \
    gnn.arch=learnable_graph_mask \
    gnn.d_model=1024 gnn.dropout=0.1 gnn.pe_hidden_channels=256 gnn.pe_num_layers=5 \
    gnn.num_samples=320 gnn.k_pe=3 gnn.gt_num_layers=3 gnn.gt_heads=8 gnn.k_gt=2 \
    gnn.eps=1e-6 gnn.use_layer_norm=true gnn.pe_node_features=random \
    gnn.mask_alpha=0.7 gnn.mask_layer_scope="$SCOPE" gnn.mask_psi_scale=cosine \
    gnn.mask_k_hops=1 gnn.mask_symmetrize=true gnn.mask_use_edges=true \
    gnn.structural_lr_mult=5.0 \
    data.text_edge_list=none \
    data.injection_scope=decode_consistent \
    trainer.init_lora_from="$STAGE1_LORA" \
    trainer.init_pe_from="$PE_INIT_DIR" \
    trainer.freeze_lora=false trainer.freeze_pe=false \
    trainer.learning_rate=0.00025 trainer.epochs=3 \
    +trainer.sft.save_total_limit=1 \
    trainer.checkpoint_dir="$CHECKPOINT_DIR" \
    trainer.save_name="$NAME" \
    wandb.run_name="$NAME" wandb.tag=e13e_layer_scope \
    eval.data="$TEST_GRAPHS" eval.num_graphs=-1

echo "[$(date --iso-8601=seconds)] e13e task ${TASK_ID} DONE: $NAME"
