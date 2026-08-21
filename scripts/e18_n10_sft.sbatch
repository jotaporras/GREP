#!/bin/bash
#SBATCH --job-name=e18-n10
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%j.out
#SBATCH --time=02:30:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
# betty's SLURM cli_filter (2026-08-03) enforces EXACTLY 28 CPUs / 224 GB per
# GPU on dgx-b200.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#
# e18 — node-identity dev loop on the n~10 corpus (docs/2026-08-21
# e18_n10_identity_plan.md). One job = one ARM of the e17 mask recipe
# (e17_pf_sft.sbatch verbatim: 31B-it, stage-1 LoRA warm start, navigator-GT
# tower, mask_alpha 0, binary edges, decode_consistent, slr 0.012) with the
# e18 flags toggled, capped at MAX_STEPS optimizer steps so a run fits ~30 min
# of training, then: (1) held-out navigation eval on test_graphs, (2) the
# neighbour-naming probe on test_graphs and a few train graphs.
#
#   ARM=mask         control: the e17 mask, nothing new
#   ARM=mask_a       + decision gating (A; diagnostic — e13b decode_trail prior
#                      says the zero-shot hard version did nothing)
#   ARM=mask_b       + structural key channel (B; the candidate)
#   ARM=mask_ab      + both
#   ARM=mask_bind    + binding auxiliary loss only
#   ARM=mask_b_bind  + B + binding
#   ARM=mask_d       + soft edge tokens (D; graph-side upper bound — one learned
#                      embedding per directed edge spliced in after BOS)
#   ARM=text_edges   plain-LLM upper bound: edge list in text, fresh LoRA
#
# Submit:  ARM=mask_b sbatch scripts/e18_n10_sft.sbatch
#          ARM=mask MAX_STEPS=40 RUN_NAME=e18_calib sbatch scripts/e18_n10_sft.sbatch   # s/step calibration
# Override DATA_SPLIT for the n15 corpus.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

ALELAB=/vast/projects/aribeiro/alelab/jporras
PROJ=$ALELAB/GREP-PRISM
ENV_NAME=${ENV_NAME:-$ALELAB/envs/GREP-PRISM-v3}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-$PROJ/outputs/e18_identity}
PE_DIR=$PROJ/pretrained_pe
STAGE1_LORA=$PROJ/outputs/e9_multistage_training/e9_ms_stage1/e9_ms_stage1_sqgk4o3j/checkpoint-100

ARM="${ARM:?set ARM=mask|mask_a|mask_b|mask_ab|mask_bind|mask_b_bind|mask_d|text_edges}"
DATA_SPLIT=${DATA_SPLIT:-$PROJ/data/n_10_vllm/gen/nav_n10_gemma_data/split}
MAX_STEPS=${MAX_STEPS:-300}
RUN_NAME=${RUN_NAME:-e18_n10_${ARM}}
TAG=${TAG:-e18_identity}
PROBE_TRAIN_GRAPHS=${PROBE_TRAIN_GRAPHS:-3}
PROBE_OUT=${PROBE_OUT:-$PROJ/results/e18_probe}

export WANDB_ENTITY=${WANDB_ENTITY:-alelab}
export HF_HOME=${HF_HOME:-$ALELAB/hf_cache}
export HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf_datasets_${SLURM_JOB_ID:-manual}"
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-28}"
# PATH prepend, not `conda activate`: compute-node batch shells have no conda
# (e17 jobs 7639051-53 died exit 127).
export PATH="$ENV_NAME/bin:$PATH"

test -f "$DATA_SPLIT/formatted_all_new__train.json" || { echo "missing $DATA_SPLIT/formatted_all_new__train.json"; exit 2; }

# --- the e17 mask recipe (shared by every mask arm) ---------------------------
MASK_ARGS=(
    gnn.arch=learnable_graph_mask
    gnn.d_model=1024 gnn.dropout=0.1 gnn.pe_hidden_channels=256 gnn.pe_num_layers=5
    gnn.num_samples=320 gnn.k_pe=3 gnn.gt_num_layers=3 gnn.gt_heads=8 gnn.k_gt=2
    gnn.eps=1e-6 gnn.use_layer_norm=true gnn.pe_node_features=random
    gnn.mask_alpha=0.0 gnn.mask_layer_scope=dense gnn.mask_psi_scale=cosine
    gnn.mask_k_hops=1 gnn.mask_symmetrize=true gnn.mask_use_edges=true
    gnn.pe_gt_from="$PE_DIR/path_navigator_gt.pt"
    gnn.semantic_gt_from=null
    gnn.structural_lr_mult=0.012
    data.text_edge_list=none
    data.injection_scope=decode_consistent
    trainer.init_lora_from="$STAGE1_LORA"
)
A_ARGS=(gnn.decision_gating=true gnn.decision_gain_init=3.0)
B_ARGS=(gnn.struct_keys=true gnn.struct_keys_dim=64 gnn.struct_keys_layer_scope=dense
        gnn.struct_keys_gain_init=1.0)
BIND_ARGS=(gnn.binding_head=true gnn.binding_temperature=0.1 gnn.binding_loss_weight=0.1)
D_ARGS=(gnn.soft_edges=true)

case "$ARM" in
  mask)        ARM_ARGS=("${MASK_ARGS[@]}") ;;
  mask_a)      ARM_ARGS=("${MASK_ARGS[@]}" "${A_ARGS[@]}") ;;
  mask_b)      ARM_ARGS=("${MASK_ARGS[@]}" "${B_ARGS[@]}") ;;
  mask_ab)     ARM_ARGS=("${MASK_ARGS[@]}" "${A_ARGS[@]}" "${B_ARGS[@]}") ;;
  mask_bind)   ARM_ARGS=("${MASK_ARGS[@]}" "${BIND_ARGS[@]}") ;;
  mask_b_bind) ARM_ARGS=("${MASK_ARGS[@]}" "${B_ARGS[@]}" "${BIND_ARGS[@]}") ;;
  mask_d)      ARM_ARGS=("${MASK_ARGS[@]}" "${D_ARGS[@]}") ;;
  text_edges)  ARM_ARGS=(gnn.arch=llm data.text_edge_list=present) ;;
  *) echo "unknown ARM=$ARM"; exit 2 ;;
esac

echo "[$(date --iso-8601=seconds)] $RUN_NAME arm=$ARM data=$DATA_SPLIT max_steps=$MAX_STEPS tag=$TAG"
echo "=== python: $(which python) ==="

# eval.epoch_interval=999: no in-training generation eval (the step-based HF
# eval at max_steps//2 is the cheap val loss only). The held-out navigation
# eval runs once post-training from the saved checkpoint (post_train_graphs).
srun python -m prism.training.train_v3 \
    --config-name=e9_base_config \
    model.path="google/gemma-4-31B-it" \
    "${ARM_ARGS[@]}" \
    data.train_files="$DATA_SPLIT/formatted_all_new__train.json" \
    data.val_files="$DATA_SPLIT/formatted_all_new__val.json" \
    data.edge_weights=binary \
    trainer.freeze_lora=false trainer.freeze_pe=false \
    trainer.learning_rate=0.00025 trainer.epochs=3 trainer.max_steps="$MAX_STEPS" \
    +trainer.sft.save_total_limit=1 \
    trainer.checkpoint_dir="$CHECKPOINT_DIR" \
    trainer.save_name="$RUN_NAME" \
    wandb.run_name="$RUN_NAME" wandb.tag="$TAG" \
    eval.data="$DATA_SPLIT/test_graphs" eval.num_graphs=-1 eval.epoch_interval=999 \
    eval.post_train_graphs="$DATA_SPLIT/test_graphs"

# The run dir is <CHECKPOINT_DIR>/<RUN_NAME>_<wandb_id>/ (train_v3
# _construct_output_dir) — take the newest one matching this run name.
RUN_DIR=$(ls -dt "$CHECKPOINT_DIR/${RUN_NAME}"_* | head -1)
echo "[$(date --iso-8601=seconds)] run dir: $RUN_DIR"

# --- neighbour-naming probe: held-out graphs, then a few TRAIN graphs ---------
mkdir -p "$PROBE_OUT"
srun python scripts/neighbour_probe.py --checkpoint "$RUN_DIR" \
    --graphs "$DATA_SPLIT/test_graphs" \
    --output "$PROBE_OUT/${RUN_NAME}_test.json"
srun python scripts/neighbour_probe.py --checkpoint "$RUN_DIR" \
    --graphs "$DATA_SPLIT/train_graphs" --max-graphs "$PROBE_TRAIN_GRAPHS" \
    --output "$PROBE_OUT/${RUN_NAME}_train.json"

echo "[$(date --iso-8601=seconds)] $RUN_NAME DONE"
