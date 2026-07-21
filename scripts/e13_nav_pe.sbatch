#!/bin/bash
#SBATCH --job-name=e13-nav-pe
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#SBATCH --array=0-3
#
# e13: navigation-pretrained PE (feature/gnn_nav_integration suite4) as the mask Ψ
# producer, replacing the e10/e12 edge-prediction pretraining. All arms:
# learnable_graph_mask on gemma-4-31B, NO edge text, prompt_only injection, LoRA
# warm-started from the SAME stage-1 adapter as e11/e12 (e9_ms_stage1_sqgk4o3j/
# checkpoint-100), lr 2.5e-4, in-config full 10-graph generation eval.
#
#   idx  arm             Ψ producer                     schedule
#   0    e13_gt_joint3   GT-only (path_navigator_gt)    joint 3 epochs
#   1    e13_agt_joint3  NavigatorPE (GT + AGT)         joint 3 epochs
#   2    e13_gt_pe1only  GT-only                        PE-only 1 epoch (LoRA frozen)
#   3    e13_agt_pe1only NavigatorPE (GT + AGT)         PE-only 1 epoch (LoRA frozen)
#
# Arms 2/3 are stage A of the freeze-then-joint schedule; scripts/e13_nav_pe_stageb
# .sbatch (submitted afterok) runs the 2-epoch joint stage B from their PE weights.
# GT LR: joint = 2.5e-4 x 0.012 = 3e-6; PE-only = 2.5e-4 x 0.12 = 3e-5 (the
# collaborator-validated standalone-GNN LR; scaled 10x down when the LLM co-trains).
# GT/PE hyperparameters shape-match the pretrained suite (72-key strict load).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3}
ALELAB=/vast/projects/aribeiro/alelab/jporras
CHECKPOINT_DIR=$ALELAB/GREP-PRISM/outputs/e13_nav_pe
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
    0) NAME=e13_gt_joint3;   SEMANTIC=null;      EPOCHS=3; SLR=0.012; FREEZE_LORA=false ;;
    1) NAME=e13_agt_joint3;  SEMANTIC="$PE_AGT"; EPOCHS=3; SLR=0.012; FREEZE_LORA=false ;;
    2) NAME=e13_gt_pe1only;  SEMANTIC=null;      EPOCHS=1; SLR=0.12;  FREEZE_LORA=true  ;;
    3) NAME=e13_agt_pe1only; SEMANTIC="$PE_AGT"; EPOCHS=1; SLR=0.12;  FREEZE_LORA=true  ;;
    *) echo "bad task id $TASK_ID" >&2; exit 1 ;;
esac
echo "[$(date --iso-8601=seconds)] e13 task ${TASK_ID}: $NAME"

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
    gnn.structural_lr_mult="$SLR" \
    data.text_edge_list=none data.injection_scope=prompt_only \
    trainer.init_lora_from="$STAGE1_LORA" \
    trainer.freeze_lora="$FREEZE_LORA" trainer.freeze_pe=false \
    trainer.learning_rate=0.00025 trainer.epochs="$EPOCHS" \
    +trainer.sft.save_total_limit=1 \
    trainer.checkpoint_dir="$CHECKPOINT_DIR" \
    trainer.save_name="$NAME" \
    eval.data="$TEST_GRAPHS" eval.num_graphs=-1 \
    wandb.run_name="$NAME" wandb.tag=e13_nav_pe

echo "[$(date --iso-8601=seconds)] e13 task ${TASK_ID} DONE: $NAME"
