# e22 arm definitions — sourced by scripts/e22_geo_sft.sbatch after e18_arms.sh.
# Single source of truth for the GEGRv2 port (docs/2026-11-05 e22_greg_port.md).
#
#   greg     = the GEGRv2 e028 / e033 `fusion_rpearl` settings on the e22 mask
#              recipe (navigator-GT Ψ at 0.012×, stage-1 LoRA warm start):
#                - mask: graded hop table, k = 3, one learnable additive level per
#                  hop distance (mask_form=hop_table), on every Gemma dense layer
#                - write: post-fusion depth mode from the R-PEARL tap ALONE
#                  (post_fusion_hop_taps=first), on every Gemma dense layer,
#                  gates open (gain 1.0), RMSNorm scales zero (e19 v2 init)
#                - no decision gating (GEGRv2 has none)
#   greg_a   = greg + the e18-A decision rows (gain 3.0; the e22 hop_depth base)
#
# Hydra rejects duplicate overrides, so the recipe is spelled out here instead of
# appending to e18_arm_args' MASK_ARGS (which pins mask_k_hops=1, mask_alpha=0).
# Caller sets PE_GT and STAGE1_LORA. Sets ARM_ARGS; exits 2 on an unknown arm.

E22_ARMS="greg|greg_a"

e22_arm_args() {
    local arm="$1"
    : "${PE_GT:?e22_arm_args: set PE_GT}" "${STAGE1_LORA:?e22_arm_args: set STAGE1_LORA}"
    local GREG_ARGS=(
        gnn.arch=learnable_graph_mask
        # Ψ tower = the e18/e19 recipe (gt.build_psi_producer; e13 navigator GT weights).
        gnn.d_model=1024 gnn.dropout=0.1 gnn.pe_hidden_channels=256 gnn.pe_num_layers=5
        gnn.num_samples=320 gnn.k_pe=3 gnn.gt_num_layers=3 gnn.gt_heads=8 gnn.k_gt=2
        gnn.eps=1e-6 gnn.use_layer_norm=true gnn.pe_node_features=random
        gnn.pe_gt_from="$PE_GT"
        gnn.semantic_gt_from=null
        gnn.structural_lr_mult=0.012
        # Mask: GEGRv2 graded hop table (e028 TABLE = learnable=false graded=true k_hops=3).
        gnn.mask_form=hop_table gnn.mask_k_hops=3 gnn.mask_layer_scope=dense
        gnn.mask_symmetrize=true gnn.mask_use_edges=true
        # Write: GEGRv2 e033 fusion_rpearl (R-PEARL tap alone) on all dense layers.
        gnn.post_fusion=true gnn.post_fusion_layer_scope=dense
        gnn.post_fusion_hop_mode=depth gnn.post_fusion_hop_taps=first
        gnn.post_fusion_gain_init=1.0
        data.text_edge_list=none
        data.injection_scope=decode_consistent
        trainer.init_lora_from="$STAGE1_LORA"
    )
    local A_ARGS=(gnn.decision_gating=true gnn.decision_gain_init=3.0)
    case "$arm" in
      greg)   ARM_ARGS=("${GREG_ARGS[@]}") ;;
      greg_a) ARM_ARGS=("${GREG_ARGS[@]}" "${A_ARGS[@]}") ;;
      *) echo "unknown ARM=$arm (one of $E22_ARMS)" >&2; exit 2 ;;
    esac
}
