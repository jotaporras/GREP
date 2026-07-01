#!/bin/bash
#SBATCH --job-name=e10-integrate-rpretrained-rpe
#SBATCH --output=/vast/home/j/jporras/sourcecode/GREP/slurm-%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G

# Dev e10: warm-start the LEARNABLE RELATIVE-PE mask (gnn.arch=learnable_graph_mask)
# from the pretrained edge-detector GraphTransformer. The mask is
#     M = alpha*A + (1-alpha)*scaled(Psi Psi^T),   Psi = GraphTransformer(graph)
# and here that GraphTransformer's weights are initialised from the edge detector
# (notebooks/e9_gnn_navigation.ipynb) instead of from scratch. Structurally this is
# the learnable_graph_mask sibling of scripts/dev_e10_integrate_pretrained_gnn.sbatch
# (which warm-starts the ADDITIVE rpearl_gt_llm planner from the same detector).
#
# PREREQ: run scripts/dev_e10_build_pretrained_pe_planner.sh first to produce
# $PE_INIT_DIR/gnn_weights.pt (the SAME folder the additive integrate job uses;
# learnable_graph_mask reads its "gt_model" state_dict and ignores "pe_proj").
#
# WHY THE GT HPARAMS BELOW ARE NON-NEGOTIABLE: init_pe_from now strict-loads the GT
# state_dict into the mask's pe_model (loaders.load_pe_weights_into). The GT
# hyperparameters MUST reproduce the detector's GT_HPARAMS (build script:
# gt_num_layers=3, pe_num_layers=5, num_samples=320, k_pe=3, k_gt=2, d_model=1024,
# pe_hidden_channels=256, gt_heads=8) or the load raises on missing/unexpected keys.
# These deliberately differ from the alpha-sweep sbatch's smaller L2 GT.
#
# Both arms: gemma-4-31B (confirmatory base), joint PE+LoRA, structural_lr_mult=5,
# full 10-graph eval. LoRA is warm-started from stage-1 (mirrors the additive
# integrate job, so the only from-scratch piece is the mask machinery, not the GT).

BASE_CONFIG=e9_base_config
MODEL_PATH="google/gemma-4-31B-it"

# Stage-1 SFT LoRA adapter (gemma-4-31B-it). Carried in via init_lora_from.
LLM_CHECKPOINT="$ALELAB_DRIVE/GREP-PRISM/outputs/e9_multistage_training/e9_ms_stage1/e9_ms_stage1_sqgk4o3j/checkpoint-100"
# PE-init folder built by dev_e10_build_pretrained_pe_planner.sh (holds gnn_weights.pt
# with the detector GT under "gt_model"). Same artifact as the additive integrate job.
PE_INIT_DIR="$ALELAB_DRIVE/GREP-PRISM/outputs/dev_e10/pe_init_edge_detector_gt"
# Output root for this experiment's runs.
CHECKPOINT_DIR="$ALELAB_DRIVE/GREP-PRISM/outputs/dev_e10/e10_integ_rpe"

# Loading env stuff
module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3

# GraphTransformer overrides reproducing the edge detector's GT (GT_HPARAMS in
# scripts/dev_e10_build_pretrained_pe_planner.py). Do NOT edit these without
# rebuilding the PE-init folder — the strict GT load asserts they match.
GT_ARGS=(
    model.path="$MODEL_PATH"
    gnn.arch=learnable_graph_mask
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
    gnn.pe_node_features=random
    # Eval on the WHOLE held-out set (all 10 graphs), not the num_graphs=5 default.
    eval.data=data/revised/gen/nav100_n30_gemma_data/split/test_graphs
    eval.num_graphs=-1
)

# Learnable-mask knobs. THESE ARE THE METHOD KNOBS TO TUNE (one-line overrides):
#   mask_psi_scale=cosine     -> bounded sim in [-1,1]. A PRETRAINED GT already has
#       structure-clustered Psi, so cosine carries real signal from step 0 (the
#       severe cosine cold-start noted for a RANDOM GT does not apply here). The
#       safe default. Alternative: mask_psi_scale=inv_sqrt_d (stronger O(?) signal
#       but scale depends on the detector's unknown output norm -> may saturate).
#   mask_alpha=0.7            -> adjacency-dominant. Lower it (e.g. 0.5) to give the
#       warm-started GT more voice in the logits; 1.0 is rejected (zeroes GT grad).
MASK_ARGS=(
    gnn.mask_alpha=0.7
    gnn.mask_layer_scope=dense
    gnn.mask_psi_scale=cosine
    gnn.mask_k_hops=1
    gnn.mask_symmetrize=true
    gnn.mask_use_edges=true
    gnn.structural_lr_mult=5.0
)

# Both edge-list variants run sequentially, each initialised from the SAME stage-1
# LoRA checkpoint and the SAME pretrained-GT folder.
#   no_edge_list   = connectivity must come from the graph (mask)
#   with_edge_list = edge list kept in the prompt (control)
for VARIANT_SPEC in "no_edge_list:none" "with_edge_list:present"; do
    VARIANT="${VARIANT_SPEC%%:*}"
    TEXT_EDGE_LIST="${VARIANT_SPEC##*:}"
    srun python -m prism.training.train_v3 \
        --config-name="$BASE_CONFIG" \
        "${GT_ARGS[@]}" \
        "${MASK_ARGS[@]}" \
        wandb.tag=e10_integ_rpe \
        trainer.save_name="e10_integ_rpe__${VARIANT}" \
        wandb.run_name="e10_integ_rpe__${VARIANT}" \
        trainer.init_lora_from="$LLM_CHECKPOINT" \
        trainer.init_pe_from="$PE_INIT_DIR" \
        trainer.checkpoint_dir="$CHECKPOINT_DIR" \
        trainer.freeze_lora=false \
        trainer.freeze_pe=false \
        data.text_edge_list="$TEXT_EDGE_LIST" \
        trainer.learning_rate=0.00025 \
        trainer.epochs=3
done
