#!/usr/bin/env bash
# e11 injection-scope run — LearnableGraphMaskLLM on google/gemma-4-12B-it, GT-only mask Ψ.
#
# One train_v3 run: warm-starts the newest Stage-1 LoRA adapter (resolved by glob search), loads
# the suite navigator's pretrained GT as the STANDALONE mask Ψ producer via gnn.pe_gt_from (no
# AGT/Semantic GT), trains PE+LoRA jointly with edges removed (text_edge_list=none) under
# prompt-only Ψ injection, then the in-config eval (eval.data inherited from base_config).
#
# No hard-coded run/checkpoint paths — every path is an ARGUMENT, an env var, or a glob SEARCH:
#   * suite dir      — positional $1 (default: $OUTPUTS_ROOT/e9_multistage_training/suite4)
#   * Stage-1 LoRA   — INIT_LORA env, else newest e9_ms_stage1_* run (+ its newest checkpoint-N)
#                      under $STAGE1_BASE, via resolve_run_dir/resolve_checkpoint (as before)
#   * output dir     — CHECKPOINT_DIR env (default $OUTPUTS_ROOT/e11_injection_scope)
#   * roots          — OUTPUTS_ROOT env (default repo-relative ./outputs; set to /vast/… on cluster)
#
# Two deliberate reconciliations vs. the pasted e11 param set:
#   * GT init — <suite>/path_navigator_gt.pt via gnn.pe_gt_from (a RAW state_dict → the GT-only
#     path). The pasted trainer.init_pe_from=<dev_e10 edge detector> is DROPPED: with pe_gt_from
#     set it would overwrite the suite GT (train_v3 loads pe_gt_from first, then init_pe_from).
#   * GT LR — forced to 3e-6 = trainer.learning_rate(2.5e-4) × gnn.structural_lr_mult(0.012); the
#     pasted structural_lr_mult=5.0 (=1.25e-3, a boost) is OVERRIDDEN to honor the 3e-6 GT LR.
#
# GT/PE hyperparameters below shape-match <suite>/path_navigator_gt.pt (verified: 72-key exact —
# d_model=1024, pe_hidden_channels=256, pe_num_layers=5, k_pe=3, gt_num_layers=3). The GT-only
# load in load_navigator_pe_into is fail-loud: any missing/unexpected key or size mismatch raises.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# --- roots / paths (env-overridable; no hard-coded absolute paths) ---
OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs}"
SUITE_DIR="${1:-$OUTPUTS_ROOT/e9_multistage_training/suite4}"
OUT="${CHECKPOINT_DIR:-$OUTPUTS_ROOT/e11_injection_scope}"
STAGE1_BASE="${STAGE1_BASE:-$OUTPUTS_ROOT/e9_multistage_training/e9_ms_stage1}"

PE_GT="$SUITE_DIR/path_navigator_gt.pt"
[ -f "$PE_GT" ] || { echo "ERROR: suite GT not found: $PE_GT" >&2; exit 1; }

# Newest matching run dir by mtime (trailing slash stripped); empty if none.
resolve_run_dir()   { ls -dt "$1/${2}_"*/    2>/dev/null | head -n1 | sed 's:/*$::' || true; }
# Newest checkpoint-N under a run dir; fall back to the run dir itself (adapter at its root).
resolve_checkpoint() { local c; c=$(ls -dt "$1/checkpoint-"*/ 2>/dev/null | head -n1 | sed 's:/*$::' || true); echo "${c:-$1}"; }

# Stage-1 adapter: explicit INIT_LORA env wins; otherwise search (like the original resolve_run_dir).
if [ -z "${INIT_LORA:-}" ]; then
    S1_RUN=$(resolve_run_dir "$STAGE1_BASE" e9_ms_stage1)
    [ -n "$S1_RUN" ] || { echo "ERROR: no e9_ms_stage1_* run under $STAGE1_BASE (set INIT_LORA or STAGE1_BASE)" >&2; exit 1; }
    INIT_LORA=$(resolve_checkpoint "$S1_RUN")
fi
[ -d "$INIT_LORA" ] || { echo "ERROR: Stage-1 LoRA checkpoint not found: $INIT_LORA" >&2; exit 1; }

echo "[resolve] suite GT     : $PE_GT"
echo "[resolve] Stage-1 LoRA : $INIT_LORA"
echo "[resolve] output dir   : $OUT"

uv run -m prism.training.train_v3 --config-name=e9_base_config \
    model.path=google/gemma-4-12B-it \
    gnn.arch=learnable_graph_mask \
    ++gnn.pe_gt_from="$PE_GT" \
    gnn.d_model=1024 gnn.dropout=0.1 gnn.pe_hidden_channels=256 gnn.pe_num_layers=5 \
    gnn.num_samples=320 gnn.k_pe=3 gnn.gt_num_layers=3 gnn.gt_heads=8 gnn.k_gt=2 \
    gnn.eps=1e-6 gnn.use_layer_norm=true gnn.pe_node_features=random \
    gnn.mask_alpha=0.7 gnn.mask_layer_scope=dense gnn.mask_psi_scale=cosine \
    gnn.mask_k_hops=1 gnn.mask_symmetrize=true gnn.mask_use_edges=true \
    gnn.structural_lr_mult=0.012 \
    data.text_edge_list=none data.injection_scope=prompt_only \
    trainer.init_lora_from="$INIT_LORA" \
    trainer.freeze_lora=false trainer.freeze_pe=false \
    trainer.learning_rate=0.00025 trainer.epochs=3 \
    trainer.save_name=e11_integ_rpe_noedges_promptonly \
    trainer.checkpoint_dir="$OUT" \
    wandb.run_name=e11_integ_rpe_noedges_promptonly wandb.tag=e11_injection_scope \
    eval.num_graphs=-1
