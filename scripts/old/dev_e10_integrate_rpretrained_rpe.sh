#!/usr/bin/env bash
#
# LOCAL (non-SLURM) runner for the learnable relative-PE warm-start experiment.
# Mirrors scripts/dev_e10_integrate_rpretrained_rpe.sbatch but targets THIS box:
#   * 2x RTX A6000 (48 GB each)      * conda env: GREP-PRISM (miniconda3)
#   * prism imported via PYTHONPATH=src (no pip install / no uv here)
#
# Warm-starts the learnable_graph_mask GraphTransformer from the pretrained
# edge-detector GT (see the sbatch header for the full method description). The
# GT hyperparameters below are NON-NEGOTIABLE — they must reproduce the detector's
# GT_HPARAMS or init_pe_from's strict GT load raises on missing/unexpected keys.
#
# LOCAL PREREQ (the cluster $ALELAB_DRIVE artifacts do NOT exist on this box):
#   1. $PE_INIT_DIR/gnn_weights.pt  — the pretrained GT. Build it locally with
#        python scripts/dev_e10_build_pretrained_pe_planner.py \
#          --gnn-checkpoint <edge_detector_gt_final.pt> \
#          --model-path "$MODEL_PATH" --out-dir "$PE_INIT_DIR"
#      (edge_detector_gt_final.pt comes from notebooks/e9_gnn_navigation.ipynb,
#      or rsync the cluster's pe_init_edge_detector_gt/ folder down.)
#   2. LoRA warm-start is OPTIONAL locally: the stage-1 adapter is cluster-only,
#      so by default this trains LoRA FROM SCRATCH (the GT is still warm-started).
#      That diverges from the sbatch (which warm-starts LoRA from stage-1) — it is
#      arguably the cleaner ablation (no pretrained-LLM confound), but it IS a
#      different run. Set LLM_CHECKPOINT=<dir> to restore the sbatch behaviour.
#
# MEMORY: gemma-4-31B bf16 is ~62 GB of weights + the multimodal towers; across
# 2x48 GB it is TIGHT and may OOM. Escape hatches (env overrides), in order of
# preference for a local dev run:  BIT4=true (4-bit, ~18 GB, fits one A6000) or
# MODEL_PATH=google/gemma-4-12B-it (dev-only base). Both change numerics vs the
# confirmatory 31B/bf16 sbatch — chosen explicitly, not silently.
#
# Usage:
#   scripts/dev_e10_integrate_rpretrained_rpe.sh                 # 31B bf16, both GPUs, LoRA from scratch
#   BIT4=true scripts/dev_e10_integrate_rpretrained_rpe.sh       # 4-bit (fits comfortably)
#   GPU=1 BIT4=true scripts/dev_e10_integrate_rpretrained_rpe.sh # pin to the idle A6000
#   LLM_CHECKPOINT=/path/to/stage1/checkpoint-100 scripts/...    # restore LoRA warm-start
#   VARIANTS="no_edge_list:none" scripts/...                     # just the no-edges arm
#
# Env overrides (with defaults):
#   GPU=-1            physical index (0,1,...) masks to one GPU; -1 = all GPUs (device_map=auto)
#   MODEL_PATH        google/gemma-4-31B-it
#   BIT4=false        true => 4-bit quantized base
#   PE_INIT_DIR       outputs/dev_e10/pe_init_edge_detector_gt   (must hold gnn_weights.pt)
#   LLM_CHECKPOINT    ""  (empty => LoRA from scratch; set a dir to warm-start LoRA)
#   CHECKPOINT_DIR    outputs/dev_e10/e10_integ_rpe
#   EPOCHS=3   LR=0.00025
#   VARIANTS="no_edge_list:none with_edge_list:present"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---- env overrides ----
GPU="${GPU:--1}"
MODEL_PATH="${MODEL_PATH:-google/gemma-4-31B-it}"
BIT4="${BIT4:-false}"
BASE_CONFIG="${BASE_CONFIG:-e9_base_config}"
PE_INIT_DIR="${PE_INIT_DIR:-outputs/dev_e10/pe_init_edge_detector_gt}"
LLM_CHECKPOINT="${LLM_CHECKPOINT:-}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-outputs/dev_e10/e10_integ_rpe}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-0.00025}"
VARIANTS="${VARIANTS:-no_edge_list:none with_edge_list:present}"

# ---- conda env + import path (prism is not pip-installed on this box) ----
source /home/jporras/miniconda3/etc/profile.d/conda.sh
conda activate GREP-PRISM
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# GPU selection: a single index masks CUDA_VISIBLE_DEVICES to that physical GPU;
# -1 leaves every GPU visible so device_map="auto" (trainer.device=-1) spreads the
# 31B weights across both A6000s. expandable_segments reduces reserve fragmentation.
if [ "$GPU" != "-1" ]; then export CUDA_VISIBLE_DEVICES="$GPU"; fi
export PYTORCH_ALLOC_CONF=expandable_segments:True

# ---- fail-fast prereq: the pretrained GT must be staged locally ----
if [ ! -f "$PE_INIT_DIR/gnn_weights.pt" ]; then
    echo "ERROR: pretrained-GT init not found: $PE_INIT_DIR/gnn_weights.pt" >&2
    echo "  Build it locally, e.g.:" >&2
    echo "    python scripts/dev_e10_build_pretrained_pe_planner.py \\" >&2
    echo "      --gnn-checkpoint <edge_detector_gt_final.pt> \\" >&2
    echo "      --model-path \"$MODEL_PATH\" --out-dir \"$PE_INIT_DIR\"" >&2
    echo "  (edge_detector_gt_final.pt: notebooks/e9_gnn_navigation.ipynb, or rsync" >&2
    echo "   the cluster's pe_init_edge_detector_gt/ folder into $PE_INIT_DIR)" >&2
    exit 1
fi

# GraphTransformer overrides reproducing the edge detector's GT (GT_HPARAMS in
# scripts/dev_e10_build_pretrained_pe_planner.py). Do NOT edit without rebuilding
# the PE-init folder — the strict GT load asserts they match.
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
    eval.data=data/revised/gen/nav100_n30_gemma_data/split/test_graphs
    eval.num_graphs=-1
)

# Learnable-mask knobs (method knobs — see the sbatch header for the rationale):
#   mask_psi_scale=cosine : bounded [-1,1]; a PRETRAINED GT already carries
#       structure-clustered Psi so cosine has real signal from step 0.
#   mask_alpha=0.7        : adjacency-dominant; lower it to give the GT more voice.
MASK_ARGS=(
    gnn.mask_alpha=0.7
    gnn.mask_layer_scope=dense
    gnn.mask_psi_scale=cosine
    gnn.mask_k_hops=1
    gnn.mask_symmetrize=true
    gnn.mask_use_edges=true
    gnn.structural_lr_mult=5.0
)

echo "=================================================================="
echo " e10 integrate rpretrained rPE (LOCAL)"
echo "   base model     : $MODEL_PATH   (bit4=$BIT4)"
echo "   gpu            : $GPU"
echo "   PE init (GT)   : $PE_INIT_DIR/gnn_weights.pt"
echo "   LoRA init      : ${LLM_CHECKPOINT:-<from scratch>}"
echo "   checkpoint_dir : $CHECKPOINT_DIR"
echo "   variants       : $VARIANTS"
echo "=================================================================="

for VARIANT_SPEC in $VARIANTS; do
    VARIANT="${VARIANT_SPEC%%:*}"
    TEXT_EDGE_LIST="${VARIANT_SPEC##*:}"

    CMD=(
        python -m prism.training.train_v3
        --config-name="$BASE_CONFIG"
        "${GT_ARGS[@]}"
        "${MASK_ARGS[@]}"
        trainer.bit4="$BIT4"
        wandb.tag=e10_integ_rpe
        trainer.save_name="e10_integ_rpe__${VARIANT}"
        wandb.run_name="e10_integ_rpe__${VARIANT}"
        trainer.init_pe_from="$PE_INIT_DIR"
        trainer.checkpoint_dir="$CHECKPOINT_DIR"
        trainer.freeze_lora=false
        trainer.freeze_pe=false
        data.text_edge_list="$TEXT_EDGE_LIST"
        trainer.learning_rate="$LR"
        trainer.epochs="$EPOCHS"
    )

    # Optional LoRA warm-start (empty by default on this box; see header).
    if [ -n "$LLM_CHECKPOINT" ]; then
        [ -d "$LLM_CHECKPOINT" ] || { echo "ERROR: LLM_CHECKPOINT set but dir not found: $LLM_CHECKPOINT" >&2; exit 1; }
        CMD+=( trainer.init_lora_from="$LLM_CHECKPOINT" )
    fi

    echo ""
    echo ">>> [$VARIANT] ${CMD[*]}"
    "${CMD[@]}"
    echo ">>> [$VARIANT] done."
done

echo ""
echo "All variants finished. Compare on W&B under tag: e10_integ_rpe"
