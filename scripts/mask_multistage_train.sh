#!/usr/bin/env bash
# Multistage training of LearnableGraphMaskLLM whose mask Ψ producer is the notebook's
# NavigatorPE — pretrained PE GT + Semantic GT (AGT), composed as Ψ = SemanticGT(PE_GT(graph)),
# loaded from a navigator suite dir ($1) holding path_navigator_gt.pt + path_navigator_agt.pt.
# The per-stage regime (freeze flags, loss_target, LR, epochs, text_edge_list) is drawn from
# experiments/:
#   stage1  experiments/e9_ms_stage1.yaml  — SFT LoRA, PE frozen, edges in text (run only if SKIP_STAGE1=false; else reused)
#   stage3  experiments/e9_ms_stage3.yaml  — joint PE+LoRA, no edges in text
# Stage 2 (PE-only edge reconstruction) is intentionally SKIPPED: it retrained the
# NavigatorPE at a destructive LR (5e-3, ~167x its 3e-5 notebook pretraining) and was
# redundant with that pretraining. Stage 3 loads the pretrained NavigatorPE tensors
# directly and throttles their LR via gnn.structural_lr_mult (see below).
# Stage 1 (SFT warmup) is NOT re-run by default (SKIP_STAGE1=true): Stage 3 resolves the newest
# existing gt_pe_stage1_* run and warm-starts from its adapter. Set SKIP_STAGE1=false to
# (re)train Stage 1 first. Either way Stage 3 initializes its LoRA from Stage-1 outputs.
# Then a transferability eval of the stage-3 checkpoint.
# GT/PE hyperparameters below are the navigator's from notebooks/e9_gnn_navigation.ipynb
# (model_hparams, cells 29 & 73) — they must match the .pt state dicts (shape-matched load).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

GT_DIR="${1:?usage: $0 <navigator_suite_dir>}"
PE_GT="$GT_DIR/path_navigator_gt.pt"
SEM_GT="$GT_DIR/path_navigator_agt.pt"
for f in "$PE_GT" "$SEM_GT"; do [ -f "$f" ] || { echo "ERROR: $f not found" >&2; exit 1; }; done
OUT="${CHECKPOINT_DIR:-outputs/gt_pe_multistage}"

# {checkpoint_dir}/{save_name}_{random wandb id}: resolve the newest matching dir.
resolve_run_dir() { ls -dt "$1/${2}_"*/ 2>/dev/null | head -n1 | sed 's:/*$::' \
    || { echo "ERROR: no run dir $1/${2}_*" >&2; exit 1; }; }

# learnable_graph_mask with NavigatorPE as the mask Ψ producer. pe_gt_from/semantic_gt_from
# stay set every stage so load_navigator_pe_into rebuilds the NavigatorPE from the notebook's
# pretrained tensors. Stage 3 passes NO trainer.init_pe_from, so those pretrained weights are
# used DIRECTLY (nothing overwrites them). GT/PE hyperparameters are the navigator's from
# notebooks/e9_gnn_navigation.ipynb (cells 29 & 73) — they must match the .pt state dicts. Mask
# knobs (mask_alpha/psi_scale/…) and the freeze regime, loss_target, LR, epochs, text_edge_list
# come from base_config + stage configs.
MODEL_ARGS=(
    gnn.arch=learnable_graph_mask
    ++gnn.pe_gt_from="$PE_GT" ++gnn.semantic_gt_from="$SEM_GT"
    gnn.d_model=1024 gnn.dropout=0.1 gnn.eps=1e-6 gnn.use_layer_norm=true
    gnn.pe_hidden_channels=256 gnn.pe_num_layers=5 gnn.num_samples=320 gnn.k_pe=3
    gnn.gt_num_layers=3 gnn.gt_heads=8 gnn.k_gt=2
)

# $1=stage config, $2=save_name, rest=extra overrides (carry dirs).
stage() { uv run -m prism.training.train_v3 --config-name="$1" "${MODEL_ARGS[@]}" \
    trainer.save_name="$2" trainer.checkpoint_dir="$OUT" \
    wandb.run_name="$2" wandb.tag=gt_pe_multistage "${@:3}"; }

# SKIP_STAGE1 (default true): do NOT re-run the Stage-1 SFT warmup; instead resolve the newest
# existing $OUT/gt_pe_stage1_* run (errors loudly if none) and warm-start Stage 3 from its
# adapter. Set SKIP_STAGE1=false to (re)train Stage 1 first. Stage 3 is ALWAYS initialized from
# Stage-1 outputs via trainer.init_lora_from=$S1 — it never trains a fresh adapter.
SKIP_STAGE1="${SKIP_STAGE1:-true}"

if [ "$SKIP_STAGE1" = "true" ]; then
    S1=$(resolve_run_dir "$OUT" gt_pe_stage1)
    echo "[chain] SKIP_STAGE1=true — reusing existing Stage-1 outputs: $S1"
else
    stage e9_ms_stage1 gt_pe_stage1
    S1=$(resolve_run_dir "$OUT" gt_pe_stage1)
    echo "[chain] Stage 1 trained: $S1"
fi

# Stage 3 (joint PE+LoRA, no edges). The NavigatorPE comes from the pretrained notebook tensors
# via load_navigator_pe_into (NO trainer.init_pe_from, so nothing overwrites them).
# gnn.structural_lr_mult decouples the NavigatorPE LR from the LoRA LR: PE_lr =
# trainer.learning_rate (0.003, stage3.yaml) x 0.001 = 3e-6 — an order below the 3e-5 notebook
# pretraining, so the pretrained navigator is fine-tuned, not blown out.
stage e9_ms_stage3 gt_pe_stage3 trainer.init_lora_from="$S1" gnn.structural_lr_mult=0.001
S3=$(resolve_run_dir "$OUT" gt_pe_stage3)

# Transferability eval of the stage-3 checkpoint → $S3/eval_logs/cross_eval/.
EVAL_GRAPHS="data_store/old/eval/e6_transferability"
[ -e "$EVAL_GRAPHS" ] || { echo "ERROR: eval graphs not found: $EVAL_GRAPHS" >&2; exit 1; }
uv run -m prism.eval.scalability_evaluation \
    --checkpoint "$S3" --graphs "$EVAL_GRAPHS" \
    --four-bit --text-edge-list none --device "${DEVICE:--1}"
