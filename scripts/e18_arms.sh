# e18 arm definitions — sourced by scripts/e18_n10_sft.sbatch (betty) and
# scripts/e18_n10_plaza.sh (plaza). Single source of truth for the arm table in
# docs/2026-08-21 e18_n10_identity_plan.md §1.4.
#
# Caller must set: PE_GT (path_navigator_gt.pt), STAGE1_LORA (checkpoint dir).
# Sets: ARM_ARGS (array of Hydra overrides) for the given ARM; exits 2 on an
# unknown arm.
#
#   mask         control: the e17 mask, nothing new
#   mask_a       + decision gating (A; diagnostic)
#   mask_b       + structural key channel (B; the candidate)
#   mask_ab      + both
#   mask_bind    + binding auxiliary loss only
#   mask_b_bind  + B + binding
#   mask_a_bind  + A + binding (fleet 2026-08-21: A and bind were the two arms above the control)
#   mask_d       + soft edge tokens (D; graph-side upper bound)
#   text_edges   plain-LLM upper bound: edge list in text, fresh LoRA

E18_ARMS="mask|mask_a|mask_b|mask_ab|mask_bind|mask_b_bind|mask_a_bind|mask_d|text_edges"

e18_arm_args() {
    local arm="$1"
    : "${PE_GT:?e18_arm_args: set PE_GT}" "${STAGE1_LORA:?e18_arm_args: set STAGE1_LORA}"
    # --- the e17 mask recipe (shared by every mask arm) -----------------------
    local MASK_ARGS=(
        gnn.arch=learnable_graph_mask
        gnn.d_model=1024 gnn.dropout=0.1 gnn.pe_hidden_channels=256 gnn.pe_num_layers=5
        gnn.num_samples=320 gnn.k_pe=3 gnn.gt_num_layers=3 gnn.gt_heads=8 gnn.k_gt=2
        gnn.eps=1e-6 gnn.use_layer_norm=true gnn.pe_node_features=random
        gnn.mask_alpha=0.0 gnn.mask_layer_scope=dense gnn.mask_psi_scale=cosine
        gnn.mask_k_hops=1 gnn.mask_symmetrize=true gnn.mask_use_edges=true
        gnn.pe_gt_from="$PE_GT"
        gnn.semantic_gt_from=null
        gnn.structural_lr_mult=0.012
        data.text_edge_list=none
        data.injection_scope=decode_consistent
        trainer.init_lora_from="$STAGE1_LORA"
    )
    local A_ARGS=(gnn.decision_gating=true gnn.decision_gain_init=3.0)
    local B_ARGS=(gnn.struct_keys=true gnn.struct_keys_dim=64 gnn.struct_keys_layer_scope=dense
                  gnn.struct_keys_gain_init=1.0)
    local BIND_ARGS=(gnn.binding_head=true gnn.binding_temperature=0.1 gnn.binding_loss_weight=0.1)
    local D_ARGS=(gnn.soft_edges=true)

    case "$arm" in
      mask)        ARM_ARGS=("${MASK_ARGS[@]}") ;;
      mask_a)      ARM_ARGS=("${MASK_ARGS[@]}" "${A_ARGS[@]}") ;;
      mask_b)      ARM_ARGS=("${MASK_ARGS[@]}" "${B_ARGS[@]}") ;;
      mask_ab)     ARM_ARGS=("${MASK_ARGS[@]}" "${A_ARGS[@]}" "${B_ARGS[@]}") ;;
      mask_bind)   ARM_ARGS=("${MASK_ARGS[@]}" "${BIND_ARGS[@]}") ;;
      mask_b_bind) ARM_ARGS=("${MASK_ARGS[@]}" "${B_ARGS[@]}" "${BIND_ARGS[@]}") ;;
      mask_a_bind) ARM_ARGS=("${MASK_ARGS[@]}" "${A_ARGS[@]}" "${BIND_ARGS[@]}") ;;
      mask_d)      ARM_ARGS=("${MASK_ARGS[@]}" "${D_ARGS[@]}") ;;
      text_edges)  ARM_ARGS=(gnn.arch=llm data.text_edge_list=present) ;;
      *) echo "unknown ARM=$arm (one of $E18_ARMS)" >&2; exit 2 ;;
    esac
}
