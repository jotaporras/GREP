#!/bin/bash
#SBATCH --job-name=e13f-alpha
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#SBATCH --array=0-3
#
# e13f: mask-composition sweep on the BEST recipe (e13d_gt_decode, 0.79 =
# nav-GT Ψ × decode_consistent, slr=0.012). Single-flag A/Bs vs that run.
#
#   idx  arm               change vs 0.79 run          question
#   0    e13f_alpha00      mask_alpha=0.0              pure learned sim on legal edges
#   1    e13f_alpha03      mask_alpha=0.3              mostly-learned mix
#   2    e13f_alpha09      mask_alpha=0.9              nearly-uniform edge bonus
#   3    e13f_khops2       mask_k_hops=2 (alpha 0.7)   widen hard-allow to 2-hop pairs
#
# alpha reference: 0.7 (the 0.79 run). alpha = uniform bonus on legal edges;
# (1-alpha)*cosine(ΨΨᵀ) = learned per-edge differentiation; non-edges hard-blocked
# in arms 0-2 (k_hops=1). Arm 3 keeps alpha=0.7 but hard-allows 2-hop pairs, so
# the learned sim differentiates a wider legal set.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3}
ALELAB=/vast/projects/aribeiro/alelab/jporras
CHECKPOINT_DIR=$ALELAB/GREP-PRISM/outputs/e13f_alpha_khops
PE_DIR=$ALELAB/GREP-PRISM/pretrained_pe
STAGE1_LORA=$ALELAB/GREP-PRISM/outputs/e9_multistage_training/e9_ms_stage1/e9_ms_stage1_sqgk4o3j/checkpoint-100
TEST_GRAPHS=data/revised/gen/nav100_n30_gemma_data/split/test_graphs

export WANDB_ENTITY=${WANDB_ENTITY:-alelab}
export HF_HOME=/vast/projects/aribeiro/alelab/jporras/hf_cache
export HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf_datasets_${SLURM_JOB_ID:-manual}"

module load anaconda3
source activate "$ENV_NAME"

TASK_ID=${SLURM_ARRAY_TASK_ID:-${1:-0}}
case "$TASK_ID" in
    0) NAME=e13f_alpha00; ALPHA=0.0; KHOPS=1 ;;
    1) NAME=e13f_alpha03; ALPHA=0.3; KHOPS=1 ;;
    2) NAME=e13f_alpha09; ALPHA=0.9; KHOPS=1 ;;
    3) NAME=e13f_khops2;  ALPHA=0.7; KHOPS=2 ;;
    *) echo "bad task id $TASK_ID" >&2; exit 1 ;;
esac
echo "[$(date --iso-8601=seconds)] e13f task ${TASK_ID}: $NAME (alpha=$ALPHA k_hops=$KHOPS)"

srun python -m prism.training.train_v3 \
    --config-name=e9_base_config \
    model.path="google/gemma-4-31B-it" \
    gnn.arch=learnable_graph_mask \
    gnn.d_model=1024 gnn.dropout=0.1 gnn.pe_hidden_channels=256 gnn.pe_num_layers=5 \
    gnn.num_samples=320 gnn.k_pe=3 gnn.gt_num_layers=3 gnn.gt_heads=8 gnn.k_gt=2 \
    gnn.eps=1e-6 gnn.use_layer_norm=true gnn.pe_node_features=random \
    gnn.mask_alpha="$ALPHA" gnn.mask_layer_scope=dense gnn.mask_psi_scale=cosine \
    gnn.mask_k_hops="$KHOPS" gnn.mask_symmetrize=true gnn.mask_use_edges=true \
    gnn.pe_gt_from="$PE_DIR/path_navigator_gt.pt" \
    gnn.semantic_gt_from=null \
    gnn.structural_lr_mult=0.012 \
    data.text_edge_list=none \
    data.injection_scope=decode_consistent \
    trainer.init_lora_from="$STAGE1_LORA" \
    trainer.freeze_lora=false trainer.freeze_pe=false \
    trainer.learning_rate=0.00025 trainer.epochs=3 \
    +trainer.sft.save_total_limit=1 \
    trainer.checkpoint_dir="$CHECKPOINT_DIR" \
    trainer.save_name="$NAME" \
    wandb.run_name="$NAME" wandb.tag=e13f_alpha_khops \
    eval.data="$TEST_GRAPHS" eval.num_graphs=-1

echo "[$(date --iso-8601=seconds)] e13f task ${TASK_ID} DONE: $NAME"
