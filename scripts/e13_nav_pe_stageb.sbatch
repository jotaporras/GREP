#!/bin/bash
#SBATCH --job-name=e13-nav-pe-b
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#SBATCH --array=0-1
#
# e13 stage B (freeze-then-joint schedule): joint PE+LoRA for 2 epochs, warm-starting
# the PE from the corresponding stage-A PE-only run (e13_{gt,agt}_pe1only, newest run
# dir under CHECKPOINT_DIR) and the LoRA from the SAME stage-1 adapter every other arm
# uses (stage A froze it, so it is bit-identical). Submit with
#   sbatch --dependency=afterok:<e13_nav_pe jobid> scripts/e13_nav_pe_stageb.sbatch
#
#   idx  arm            PE init from        Ψ producer
#   0    e13_gt_joint2  e13_gt_pe1only_*    GT-only
#   1    e13_agt_joint2 e13_agt_pe1only_*   NavigatorPE (GT + AGT)
#
# pe_gt_from/semantic_gt_from are STILL passed: they select the NavigatorPE topology
# and are recorded in train_config.json for eval reload; init_pe_from then overwrites
# the whole pe_model with the stage-A weights (train_v3 loads pe_gt_from first).

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

export WANDB_ENTITY=${WANDB_ENTITY:-alelab}
export HF_HOME=/vast/projects/aribeiro/alelab/jporras/hf_cache
export HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf_datasets_${SLURM_JOB_ID:-manual}"

module load anaconda3
source activate "$ENV_NAME"

TASK_ID=${SLURM_ARRAY_TASK_ID:-${1:-0}}
case "$TASK_ID" in
    0) NAME=e13_gt_joint2;  STAGE_A=e13_gt_pe1only;  SEMANTIC=null ;;
    1) NAME=e13_agt_joint2; STAGE_A=e13_agt_pe1only; SEMANTIC="$PE_AGT" ;;
    *) echo "bad task id $TASK_ID" >&2; exit 1 ;;
esac

PE_INIT=$(ls -dt "$CHECKPOINT_DIR/${STAGE_A}_"*/ | head -n1 | sed 's:/*$::')
[ -f "$PE_INIT/gnn_weights.pt" ] || { echo "ERROR: no gnn_weights.pt in stage-A run $PE_INIT" >&2; exit 1; }
echo "[$(date --iso-8601=seconds)] e13 stage B task ${TASK_ID}: $NAME  (PE init: $PE_INIT)"

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
    data.text_edge_list=none data.injection_scope=prompt_only \
    trainer.init_pe_from="$PE_INIT" \
    trainer.init_lora_from="$STAGE1_LORA" \
    trainer.freeze_lora=false trainer.freeze_pe=false \
    trainer.learning_rate=0.00025 trainer.epochs=2 \
    +trainer.sft.save_total_limit=1 \
    trainer.checkpoint_dir="$CHECKPOINT_DIR" \
    trainer.save_name="$NAME" \
    eval.data="$TEST_GRAPHS" eval.num_graphs=-1 \
    wandb.run_name="$NAME" wandb.tag=e13_nav_pe

echo "[$(date --iso-8601=seconds)] e13 stage B task ${TASK_ID} DONE: $NAME"
