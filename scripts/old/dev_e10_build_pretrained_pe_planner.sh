#!/bin/bash
# Build the stage-3 PE-init folder from the pretrained edge-detector GNN.
#
# Run this ONCE on the cluster (it needs the gemma-4-31B-it config to size
# pe_proj) before submitting scripts/dev_e10_integrate_pretrained_gnn.sbatch.
# It copies the source detector into the PE-init dir for provenance and invokes
# the aux Python script that writes gnn_weights.pt.
#
# Prereq: edge_detector_gt_final.pt has been rsync'd to the cluster artifact dir.
set -euo pipefail

# Loading env stuff
module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3

MODEL_PATH="google/gemma-4-31B-it"
GNN_CHECKPOINT="$ALELAB_DRIVE/GREP-PRISM/outputs/dev_e10/edge_detector_gt_final.pt"
PE_INIT_DIR="$ALELAB_DRIVE/GREP-PRISM/outputs/dev_e10/pe_init_edge_detector_gt"

mkdir -p "$PE_INIT_DIR"
cp "$GNN_CHECKPOINT" "$PE_INIT_DIR/"          # provenance copy of the source detector

python scripts/dev_e10_build_pretrained_pe_planner.py \
    --gnn-checkpoint "$GNN_CHECKPOINT" \
    --model-path "$MODEL_PATH" \
    --out-dir "$PE_INIT_DIR"
