#!/bin/bash
#SBATCH --job-name=e13d-nav-dec
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#SBATCH --array=0-1
#
# e13d: the untested combination — navigator Ψ (e13's content win, +0.10 via the
# GT+AGT ensemble) × decode-consistent wiring (e13c's wiring win, +0.25).
# Single-flag A/Bs vs the e13 arms: ONLY data.injection_scope changes
# (prompt_only -> decode_consistent); GT LR stays the navigator-validated
# structural_lr_mult=0.012 (3e-6).
#
#   idx  arm              Ψ producer                paired e13 run   free-gen
#   0    e13d_agt_decode  NavigatorPE (GT + AGT)    e13_agt_joint3   0.58
#   1    e13d_gt_decode   nav GT only               e13_gt_joint3    0.44
#
# NOTE the cross-comparison to e13c_rpe_decode (0.73) is NOT single-flag: that
# arm used the edge-detector Ψ at e11's structural_lr_mult=5.0. The clean
# pairings are the e13 rows above.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3}
ALELAB=/vast/projects/aribeiro/alelab/jporras
CHECKPOINT_DIR=$ALELAB/GREP-PRISM/outputs/e13d_nav_decode
PE_DIR=$ALELAB/GREP-PRISM/pretrained_pe
STAGE1_LORA=$ALELAB/GREP-PRISM/outputs/e9_multistage_training/e9_ms_stage1/e9_ms_stage1_sqgk4o3j/checkpoint-100
TEST_GRAPHS=data/revised/gen/nav100_n30_gemma_data/split/test_graphs

PE_GT=$PE_DIR/path_navigator_gt.pt
PE_AGT=$PE_DIR/path_navigator_agt.pt
[ -f "$PE_GT" ]  || { echo "ERROR: pretrained GT not found: $PE_GT" >&2; exit 1; }
[ -f "$PE_AGT" ] || { echo "ERROR: pretrained AGT not found: $PE_AGT" >&2; exit 1; }

export WANDB_ENTITY=${WANDB_ENTITY:-alelab}
export HF_HOME=/vast/projects/aribeiro/alelab/jporras/hf_cache
export HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf_datasets_${SLURM_JOB_ID:-manual}"

module load anaconda3
source activate "$ENV_NAME"

TASK_ID=${SLURM_ARRAY_TASK_ID:-${1:-0}}
case "$TASK_ID" in
    0) NAME=e13d_agt_decode; SEMANTIC="$PE_AGT" ;;
    1) NAME=e13d_gt_decode;  SEMANTIC=null ;;
    *) echo "bad task id $TASK_ID" >&2; exit 1 ;;
esac
echo "[$(date --iso-8601=seconds)] e13d task ${TASK_ID}: $NAME"

srun python -m prism.training.train_v3 \
    --config-name=e9_base_config \
    model.path="google/gemma-4-31B-it" \
    gnn.arch=learnable_graph_mask \
    gnn.d_model=1024 gnn.dropout=0.1 gnn.pe_hidden_channels=256 gnn.pe_num_layers=5 \
    gnn.num_samples=320 gnn.k_pe=3 gnn.gt_num_layers=3 gnn.gt_heads=8 gnn.k_gt=2 \
    gnn.eps=1e-6 gnn.use_layer_norm=true gnn.pe_node_features=random \
    gnn.mask_alpha=0.7 gnn.mask_layer_scope=dense gnn.mask_psi_scale=cosine \
    gnn.mask_k_hops=1 gnn.mask_symmetrize=true gnn.mask_use_edges=true \
    gnn.pe_gt_from="$PE_GT" \
    gnn.semantic_gt_from="$SEMANTIC" \
    gnn.structural_lr_mult=0.012 \
    data.text_edge_list=none \
    data.injection_scope=decode_consistent \
    trainer.init_lora_from="$STAGE1_LORA" \
    trainer.freeze_lora=false trainer.freeze_pe=false \
    trainer.learning_rate=0.00025 trainer.epochs=3 \
    +trainer.sft.save_total_limit=1 \
    trainer.checkpoint_dir="$CHECKPOINT_DIR" \
    trainer.save_name="$NAME" \
    wandb.run_name="$NAME" wandb.tag=e13d_nav_decode \
    eval.data="$TEST_GRAPHS" eval.num_graphs=-1

echo "[$(date --iso-8601=seconds)] e13d task ${TASK_ID} DONE: $NAME"
