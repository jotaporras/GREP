#!/bin/bash
#SBATCH --job-name=e10-postfusion
#SBATCH --output=/vast/home/j/jporras/sourcecode/GREP/slurm-%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#SBATCH --array=1-2

# DISPOSABLE: EXPERIMENTAL post-fusion (output) graph attention. The graph is fused AFTER
# the LLM, just before the vocab head:
#     Y_tilde = Y + tanh(gate) * CrossAttn(Q=Y, K,V=Psi),   Psi = GraphTransformer(graph)
# so every output token (incl. each generated one) cross-attends the N node embeddings.
# gate starts at 0 => cold start (== base LLM at init), fusion ramps in. Standalone arch
# (models/postfusion_graph_llm.py); may be deleted wholesale if it doesn't pan out.
#
# Two arms on the edge-list axis (does the fusion help WITH vs WITHOUT edges in the prompt):
#   1  text_edge_list=present   edges IN the prompt (fusion is auxiliary structure)
#   2  text_edge_list=none      edges REMOVED (connectivity must come from the graph fusion)
#
# Both: gemma-4-31B (confirmatory base), structural_lr_mult=5, full 10-graph eval.
#
#   sbatch scripts/dev_e10_postfusion.sbatch          # both arms
#   sbatch --array=2 scripts/dev_e10_postfusion.sbatch   # just the no-edges arm

BASE_CONFIG=e9_base_config
MODEL_PATH="google/gemma-4-31B-it"   # confirmatory base (matches the alpha-sweep / refactor-verify reruns)
CHECKPOINT_DIR="$ALELAB_DRIVE/GREP-PRISM/outputs/dev_e10/postfusion"

# Loading env stuff
module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3

# Shared overrides. Full held-out eval (all 10 graphs), joint PE+LoRA finetune.
COMMON=(
    model.path="$MODEL_PATH"
    wandb.tag=e10_postfusion
    trainer.checkpoint_dir="$CHECKPOINT_DIR"
    trainer.freeze_llm=false
    trainer.freeze_pe=false
    trainer.learning_rate=0.00025
    trainer.epochs=3
    eval.data=data/revised/gen/nav100_n30_gemma_data/split/test_graphs
    eval.num_graphs=-1
)

# Post-fusion arch args: standalone GraphTransformer (Psi producer) + output cross-attention.
ARCH_ARGS=(
    gnn.arch=postfusion_graph_llm
    gnn.d_model=1024
    gnn.dropout=0.1
    gnn.pe_hidden_channels=256
    gnn.pe_num_layers=2
    gnn.num_samples=40
    gnn.k_pe=2
    gnn.gt_num_layers=2
    gnn.gt_heads=8
    gnn.k_gt=2
    gnn.eps=1e-6
    gnn.use_layer_norm=true
    gnn.pe_node_features=random
    gnn.postfusion_num_heads=8
    gnn.postfusion_gate_init=0.0
    gnn.structural_lr_mult=5.0
)

case "$SLURM_ARRAY_TASK_ID" in
  1) NAME=edges;   TEXT_EDGE_LIST=present ;;
  2) NAME=noedges; TEXT_EDGE_LIST=none ;;
  *) echo "Unknown SLURM_ARRAY_TASK_ID='$SLURM_ARRAY_TASK_ID' (expected 1-2)"; exit 1 ;;
esac

srun python -m prism.training.train_v3 \
    --config-name="$BASE_CONFIG" \
    "${COMMON[@]}" \
    "${ARCH_ARGS[@]}" \
    data.text_edge_list="$TEXT_EDGE_LIST" \
    trainer.save_name="e10_postfusion__${NAME}" \
    wandb.run_name="e10_postfusion__${NAME}"
