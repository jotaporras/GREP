# e19 arm definitions — sourced by scripts/e19_n60_sft.sbatch. Single source of
# truth for the fleet table in docs/2026-08-23 e19_hop_separated_fusion_design.md.
#
# Every arm = the e18 mask_a base (decision gating open at 3.0 — the winning
# e18 arm) + hop-separated post-fusion channels, gates OPEN at init
# (post_fusion_gain_init, the e17/e18 zero-init-gates-never-open lesson).
# *_bind arms use the e18 mask_a_bind base (bind was the other above-control arm).
#
#   hop_shift        V1: no-R-PEARL codebook GT, shift channels K=hop_k+1=4
#   hop_shift_k5     V1 with hop_k=5 (K=6 channels — deeper reasoning range)
#   hop_shift_bind   V1 on the mask_a_bind base
#   hop_depth        V2: shared navigator-GT per-block taps (K=gt_num_layers+1=4)
#   hop_depth_bind   V2 on the mask_a_bind base
#   hop_depth_gain3  V2 with gates opened WIDE (gain_init=3.0, tanh≈0.995)
#
# Requires e18_arms.sh sourced first (uses e18_arm_args; caller sets PE_GT,
# STAGE1_LORA). Sets ARM_ARGS; exits 2 on an unknown arm.

E19_ARMS="hop_shift|hop_shift_k5|hop_shift_bind|hop_depth|hop_depth_bind|hop_depth_gain3"

e19_arm_args() {
    local arm="$1"
    # Shared hop-fusion switches (gain baked per arm below — Hydra rejects
    # duplicate overrides, so an arm must not re-set a key ARM_ARGS carries).
    local PF_COMMON=(
        gnn.post_fusion=true
        gnn.post_fusion_layer_scope=dense_top_half
        gnn.post_fusion_codebook_size=256
        gnn.post_fusion_hop_gt_layers=3
        gnn.post_fusion_hop_gt_heads=8
        gnn.post_fusion_hop_gt_k=1
    )
    case "$arm" in
      hop_shift)
        e18_arm_args mask_a
        ARM_ARGS+=("${PF_COMMON[@]}" gnn.post_fusion_hop_mode=shift
                   gnn.post_fusion_hop_k=3 gnn.post_fusion_gain_init=1.0) ;;
      hop_shift_k5)
        e18_arm_args mask_a
        ARM_ARGS+=("${PF_COMMON[@]}" gnn.post_fusion_hop_mode=shift
                   gnn.post_fusion_hop_k=5 gnn.post_fusion_gain_init=1.0) ;;
      hop_shift_bind)
        e18_arm_args mask_a_bind
        ARM_ARGS+=("${PF_COMMON[@]}" gnn.post_fusion_hop_mode=shift
                   gnn.post_fusion_hop_k=3 gnn.post_fusion_gain_init=1.0) ;;
      hop_depth)
        e18_arm_args mask_a
        ARM_ARGS+=("${PF_COMMON[@]}" gnn.post_fusion_hop_mode=depth
                   gnn.post_fusion_hop_k=3 gnn.post_fusion_gain_init=1.0) ;;
      hop_depth_bind)
        e18_arm_args mask_a_bind
        ARM_ARGS+=("${PF_COMMON[@]}" gnn.post_fusion_hop_mode=depth
                   gnn.post_fusion_hop_k=3 gnn.post_fusion_gain_init=1.0) ;;
      hop_depth_gain3)
        e18_arm_args mask_a
        ARM_ARGS+=("${PF_COMMON[@]}" gnn.post_fusion_hop_mode=depth
                   gnn.post_fusion_hop_k=3 gnn.post_fusion_gain_init=3.0) ;;
      *) echo "unknown ARM=$arm (one of $E19_ARMS)" >&2; exit 2 ;;
    esac
}
