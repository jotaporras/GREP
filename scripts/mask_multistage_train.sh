#!/usr/bin/env bash
# e11 injection-scope run — LearnableGraphMaskLLM on google/gemma-4-12B-it, GT-only mask Ψ.
#
# One train_v3 run: warm-starts the fixed Stage-1 LoRA adapter, loads suite4's pretrained GT as
# the STANDALONE mask Ψ producer via gnn.pe_gt_from (no AGT/Semantic GT), and trains PE+LoRA
# jointly with edges removed (text_edge_list=none) under prompt-only Ψ injection. The in-config
# eval (eval.data / eval.num_graphs) runs at the end — no separate transferability pass.
#
# Two deliberate reconciliations vs. the pasted e11 param set:
#   * GT init  — suite4/path_navigator_gt.pt via gnn.pe_gt_from (a RAW state_dict → the GT-only
#     path; suite4 has no gnn_weights.pt so trainer.init_pe_from cannot consume it). The pasted
#     trainer.init_pe_from=<dev_e10 edge detector> is DROPPED: with pe_gt_from set it would
#     overwrite the suite4 GT (train_v3.py loads pe_gt_from first, then init_pe_from clobbers it).
#   * GT LR   — forced to 3e-6 = trainer.learning_rate(2.5e-4) × gnn.structural_lr_mult(0.012).
#     The pasted structural_lr_mult=5.0 (=1.25e-3, a 5× BOOST) is OVERRIDDEN to honor the
#     requested 3e-6 GT LR: the GT is gently fine-tuned, not boosted. LoRA/LLM stay at 2.5e-4.
#
# GT/PE hyperparameters below must shape-match suite4/path_navigator_gt.pt (verified: 72-key
# exact match — d_model=1024, pe_hidden_channels=256, pe_num_layers=5, k_pe=3, gt_num_layers=3).
# load_navigator_pe_into now loads the GT-only PE fail-loud: any missing/unexpected key or size
# mismatch raises, so a hyperparameter drift can never silently load a partially-random PE.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# --- environment paths ---
# suite4 GT is repo-relative (resolved from the repo root this script cd's into); the Stage-1
# adapter and output dir are the cluster-absolute paths from the e11 param set.
PE_GT="outputs/e9_multistage_training/suite4/path_navigator_gt.pt"
INIT_LORA="/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/outputs/e9_multistage_training/e9_ms_stage1/e9_ms_stage1_sqgk4o3j/checkpoint-100"
OUT="/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/outputs/e11_injection_scope"
EVAL_DATA="data/revised/gen/nav100_n30_gemma_data/split/test_graphs"
[ -f "$PE_GT" ]    || { echo "ERROR: suite4 GT not found: $PE_GT" >&2; exit 1; }
[ -d "$INIT_LORA" ] || { echo "ERROR: Stage-1 LoRA checkpoint not found: $INIT_LORA" >&2; exit 1; }

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
    eval.data="$EVAL_DATA" eval.num_graphs=-1
