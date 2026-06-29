#!/bin/bash
#SBATCH --job-name=e10-integrate-pretrained-gnn
#SBATCH --output=/vast/home/j/jporras/sourcecode/GREP/slurm-%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G

# Dev e10: stage-3-style joint (PE + LoRA) finetune initialised from a pretrained
# edge-detector GNN. The GNN was trained in notebooks/e9_gnn_navigation.ipynb to
# classify edge existence; here its GraphTransformer replaces the from-scratch /
# stage-2 PE.
#
# PREREQ: run scripts/dev_e10_build_pretrained_pe_planner.sh first to produce
# $PE_INIT_DIR/gnn_weights.pt. This job only trains; it does not build the PE.
#
# Mirrors scripts/e9_manual_run.sbatch stage 3 (joint PE+LoRA, lr 2.5e-4, 3
# epochs, both text_edge_list variants), but inits LoRA from the stage-1
# checkpoint and PE from the prebuilt edge-detector folder.

BASE_CONFIG=e9_base_config
MODEL_PATH="google/gemma-4-31B-it"

# Stage-1 SFT LoRA adapter (gemma-4-31B-it). Carried in via init_lora_from.
LLM_CHECKPOINT="$ALELAB_DRIVE/GREP-PRISM/outputs/e9_multistage_training/e9_ms_stage1/e9_ms_stage1_sqgk4o3j/checkpoint-100"
# PE-init folder built by dev_e10_build_pretrained_pe_planner.sh (holds gnn_weights.pt).
PE_INIT_DIR="$ALELAB_DRIVE/GREP-PRISM/outputs/dev_e10/pe_init_edge_detector_gt"
# Output root for this experiment's stage-3 runs.
CHECKPOINT_DIR="$ALELAB_DRIVE/GREP-PRISM/outputs/dev_e10/e10_integ_stage3"

# Loading env stuff
module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3

# rpearl_gt_llm architecture overrides MATCHING the edge detector's GraphTransformer
# (e9_gnn_navigation.ipynb). These intentionally differ from e9 stage-1/2 GNN_ARGS;
# d_model/pe_num_layers/num_samples/k_pe/gt_num_layers must reproduce the checkpoint's
# architecture or the PE load would silently drop weights. pe_node_features stays
# 'random' (base-config default) so node_feature_dim=None, as in the notebook.
GNN_ARGS=(
    model.path="$MODEL_PATH"
    gnn.arch=rpearl_gt_llm
    gnn.d_model=1024
    gnn.dropout=0.1
    gnn.pe_hidden_channels=256
    gnn.pe_num_layers=5
    gnn.num_samples=320
    gnn.k_pe=3
    gnn.gt_num_layers=3
    gnn.gt_heads=8
    gnn.k_gt=2
    gnn.eps=1e-6
    gnn.use_layer_norm=true
)

# Stage 3 - joint PE + LoRA. Both variants run sequentially, each initialised from
# the SAME stage-1 LoRA checkpoint and the SAME pretrained-PE folder.
#   no_edge_list   = connectivity must come from the graph PE
#   with_edge_list = edge list kept in the prompt (control)
for VARIANT_SPEC in "no_edge_list:none" "with_edge_list:present"; do
    VARIANT="${VARIANT_SPEC%%:*}"
    TEXT_EDGE_LIST="${VARIANT_SPEC##*:}"
    srun python -m prism.training.train_v3 \
        --config-name="$BASE_CONFIG" \
        "${GNN_ARGS[@]}" \
        trainer.save_name="e10_integ_stage3__${VARIANT}" \
        wandb.run_name="e10_integ_stage3__${VARIANT}" \
        trainer.init_lora_from="$LLM_CHECKPOINT" \
        trainer.init_pe_from="$PE_INIT_DIR" \
        trainer.checkpoint_dir="$CHECKPOINT_DIR" \
        trainer.freeze_lora=false \
        trainer.freeze_pe=false \
        data.text_edge_list="$TEXT_EDGE_LIST" \
        trainer.learning_rate=0.00025 \
        trainer.epochs=3
done
