"""
Module with different LLM-based planner architectures (Gemma-4 bases only).

Public surface:
- `build_planner_model`  — construct (model, collator) for a config.
- `peft_tower_exclude`   — PEFT exclude_modules regex for multimodal bases (Gemma-4 31B etc.).
"""

from prism.data import data
from prism.models import gnn_llm
from prism.models import r_pearl as r_pearl_module
from prism.models import gt as gt_module


def build_planner_model(gnn, llm, tokenizer, *, disable_graph_token_rope=False,
                        freeze_llm=False):
    """Instantiate the model and data collator for the given architecture.

    Args:
        gnn: the ``gnn`` config section (OmegaConf; read-only) — the architecture
            switch plus its hyperparameters.
        llm: base AutoModelForCausalLM already loaded on the target device.
        tokenizer: corresponding tokenizer with pad token set.
        disable_graph_token_rope: identity-RoPE (position_id 0) on node-name tokens.
        freeze_llm: freeze the base LLM (PE-only training; no LoRA adapter added).

    Returns:
        (model, collator) — model is the architecture wrapper (or ``llm`` itself
        for the plain-LLM baseline); collator is the matching data collator.
    """
    _text_hidden = llm.config.get_text_config().hidden_size
    _node_feature_dim = _text_hidden if gnn.pe_node_features == "word_embeddings" else None

    if gnn.arch == "rpearl_llm":
        # R-PEARL only: GCN positional encodings, no GT attention blocks.
        pe_model = r_pearl_module.RandomGNNPositionalEncodings(
            pe_hidden_channels=gnn.pe_hidden_channels,
            pe_num_layers=gnn.pe_num_layers,
            d_model=gnn.d_model,
            num_samples=gnn.num_samples,
            dropout=gnn.dropout,
            k=gnn.k_pe,
            eps=gnn.eps,
            use_layer_norm=gnn.use_layer_norm,
            node_feature_dim=_node_feature_dim,
        )
        model = gnn_llm.GraphAugmentedLLM(llm, pe_model, d_model=gnn.d_model,
                                          eps=gnn.eps, pe_gain_init=gnn.pe_gain_init,
                                          disable_graph_token_rope=disable_graph_token_rope,
                                          use_pe_norm=gnn.use_pe_norm,
                                          pe_node_features=gnn.pe_node_features)
        collator = data.SpineDataCollator(tokenizer, mlm=False)

        if freeze_llm:
            model.llm.requires_grad_(False)
    elif gnn.arch == "rpearl_gt_llm":
        # Full Graph Transformer: R-PEARL inside Sparse Attention blocks.
        pe_model = gt_module.GraphTransformer(
            num_layers=gnn.gt_num_layers,
            pe_hidden_channels=gnn.pe_hidden_channels,
            pe_num_layers=gnn.pe_num_layers,
            d_model=gnn.d_model,
            heads=gnn.gt_heads,
            num_samples=gnn.num_samples,
            dropout=gnn.dropout,
            k_pe=gnn.k_pe,
            k_gt=gnn.k_gt,
            eps=gnn.eps,
            use_layer_norm=gnn.use_layer_norm,
            node_feature_dim=_node_feature_dim,
        )
        model = gnn_llm.GraphAugmentedLLM(llm, pe_model, d_model=gnn.d_model,
                                          eps=gnn.eps, pe_gain_init=gnn.pe_gain_init,
                                          disable_graph_token_rope=disable_graph_token_rope,
                                          use_pe_norm=gnn.use_pe_norm,
                                          pe_node_features=gnn.pe_node_features)
        collator = data.SpineDataCollator(tokenizer, mlm=False)

        if freeze_llm:
            model.llm.requires_grad_(False)
    elif gnn.arch == "gt_llm":
        # Pure Graph Transformer over semantic node features — no R-PEARL, no probes.
        if gnn.pe_node_features != "word_embeddings":
            raise ValueError(
                "architecture 'gt_llm' requires pe_node_features='word_embeddings' "
                f"(got {gnn.pe_node_features!r}); the GT has no random-probe input."
            )
        pe_model = gt_module.SemanticGraphTransformer(
            node_feature_dim=_text_hidden,
            d_model=gnn.d_model,
            num_layers=gnn.gt_num_layers,
            heads=gnn.gt_heads,
            dropout=gnn.dropout,
            k_gt=gnn.k_gt,
        )
        model = gnn_llm.GraphAugmentedLLM(llm, pe_model, d_model=gnn.d_model,
                                          eps=gnn.eps, pe_gain_init=gnn.pe_gain_init,
                                          disable_graph_token_rope=disable_graph_token_rope,
                                          use_pe_norm=gnn.use_pe_norm,
                                          pe_node_features=gnn.pe_node_features)
        collator = data.SpineDataCollator(tokenizer, mlm=False)

        if freeze_llm:
            model.llm.requires_grad_(False)
    elif gnn.arch == "graph_mask_llm":
        # Parameter-free structural attention mask: node tokens attend only along graph
        # edges. Reuses SpineDataCollator so graphs + injection_maps reach the model.
        model = gnn_llm.GraphMaskLLM(
            llm, k_hops=gnn.mask_k_hops, symmetrize=gnn.mask_symmetrize,
            use_edges=gnn.mask_use_edges,
            buggy_causal_fold=gnn.mask_buggy_causal_fold,
            layer_scope=gnn.mask_layer_scope)
        collator = data.SpineDataCollator(tokenizer, mlm=False)

        if freeze_llm:
            model.llm.requires_grad_(False)
    elif gnn.arch == "learnable_graph_mask":
        # Learnable relative-PE attention mask: M = alpha*A + (1-alpha)*sim(Psi Psi^T),
        # with Psi from a STANDALONE Graph Transformer (independent of the additive
        # pe_model; reuses the gt_*/pe_* config block). Mask-only — no q/k/v injection.
        if gnn.pe_node_features != "random":
            raise ValueError(
                "architecture 'learnable_graph_mask' currently supports only "
                "pe_node_features='random' (the mask GT samples probes; word-embedding "
                f"feature prep is not wired). Got {gnn.pe_node_features!r}.")
        # Ψ producer: a standalone GT (the navigator's PE stage — gt.NavigatorPE's half),
        # or the LEGACY two-stage gt.TwoStagePE when semantic_gt_from is recorded. Built by
        # gt.build_psi_producer, the single site eval rebuilds from too. Weights are loaded
        # afterwards in train_v3 (load_navigator_pe_into).
        pe_model = gt_module.build_psi_producer(gnn, node_feature_dim=_node_feature_dim)
        model = gnn_llm.LearnableGraphMaskLLM(
            llm, pe_model, alpha=gnn.mask_alpha,
            layer_scope=gnn.mask_layer_scope,
            k_hops=gnn.mask_k_hops, symmetrize=gnn.mask_symmetrize,
            use_edges=gnn.mask_use_edges, psi_scale=gnn.mask_psi_scale,
            buggy_causal_fold=gnn.mask_buggy_causal_fold,
            disable_graph_token_rope=disable_graph_token_rope,
            post_fusion=gnn.get("post_fusion", False),
            post_fusion_layer_scope=gnn.get("post_fusion_layer_scope",
                                            "dense_top_half"),
            post_fusion_d_gt=gnn.d_model,
            graph_lora=gnn.get("graph_lora", False),
            graph_lora_rank=gnn.get("graph_lora_rank", 8),
            graph_lora_targets=gnn.get("graph_lora_targets", "o_proj"),
            graph_lora_layer_scope=gnn.get("graph_lora_layer_scope",
                                           "dense_top_half"),
            pointer_fusion=gnn.get("pointer_fusion", False),
            cross_fusion=gnn.get("cross_fusion", False),
            cross_fusion_heads=gnn.get("cross_fusion_heads", 8),
            cross_fusion_dim=gnn.get("cross_fusion_dim"),
            fusion_d_gt=gnn.d_model,
            # e18 node-identity pathways (docs/2026-08-21 e18_direction_discussion.md).
            decision_gating=gnn.get("decision_gating", False),
            decision_gain_init=gnn.get("decision_gain_init", 0.0),
            struct_keys=gnn.get("struct_keys", False),
            struct_keys_dim=gnn.get("struct_keys_dim", 64),
            struct_keys_layer_scope=gnn.get("struct_keys_layer_scope", "dense"),
            struct_keys_gain_init=gnn.get("struct_keys_gain_init", 0.0),
            binding_head=gnn.get("binding_head", False),
            binding_temperature=gnn.get("binding_temperature", 0.1),
            binding_loss_weight=gnn.get("binding_loss_weight", 0.1))
        collator = data.SpineDataCollator(tokenizer, mlm=False)

        if freeze_llm:
            model.llm.requires_grad_(False)
    elif gnn.arch == "wire_llm":
        # WIRE: Ψ enters as a ROTATION of q/k (not an added signal). Reuses the same
        # standalone GraphTransformer Ψ producer as learnable_graph_mask, but consumes
        # Ψ as rotation angles, so there is no pe_proj/pe_norm to the LLM hidden size.
        if gnn.pe_node_features != "random":
            raise ValueError(
                "architecture 'wire_llm' currently supports only pe_node_features="
                f"'random' (word-embedding feature prep is not wired). Got {gnn.pe_node_features!r}.")
        # Same Ψ-producer factory as learnable_graph_mask: a standalone GT (navigator PE
        # weights poured in next by train_v3), or the legacy two-stage producer.
        # WIRE consumes Ψ as rotation angles; the producer's topology is identical.
        pe_model = gt_module.build_psi_producer(gnn)
        model = gnn_llm.WireGraphLLM(
            llm, pe_model, d_model=gnn.d_model,
            layer_scope=gnn.wire_layer_scope,
            sigma_init=gnn.wire_sigma_init,
            freeze_sigma=gnn.wire_freeze_sigma,
            omega_seed=gnn.wire_omega_seed,
            rotate_nope_planes=gnn.wire_rotate_nope_planes,
            max_angle=gnn.wire_max_angle,
            pe_gain_init=gnn.pe_gain_init,
            decode=gnn.wire_decode,
            pe_node_features=gnn.pe_node_features,
            # Default TRUE (the paper's algorithm) and it must agree with the class
            # signature and base_config.yaml — an absent key must not silently select
            # the expectation arm.
            vanilla=gnn.get("wire_vanilla", True),
            vanilla_omega_init=gnn.get("wire_vanilla_omega_init", "zero"))
        collator = data.SpineDataCollator(tokenizer, mlm=False)

        if freeze_llm:
            model.llm.requires_grad_(False)
    elif gnn.arch == "llm":
        # Pure LLM baseline; TokenIndexCollator carries graph-token index columns so
        # graph_acc/* metrics are comparable to graph architectures.
        model = llm
        collator = data.TokenIndexCollator(tokenizer, mlm=False)
    else:
        raise ValueError(
            f"Unknown architecture: {gnn.arch!r}. "
            "Choose 'rpearl_llm', 'rpearl_gt_llm', 'gt_llm', 'graph_mask_llm', "
            "'learnable_graph_mask', 'wire_llm', or 'llm'.")

    return model, collator


# Multimodal Gemma-4 bases (e.g. gemma-4-31B) include vision/audio towers whose
# projections use Gemma4ClippableLinear (not nn.Linear) — PEFT cannot adapt them,
# and their leaf names collide with the text decoder. Exclude them from LoRA targets.
# Text-only bases (gemma-4-12B) have no towers; the regex then hits nothing.
_MM_TOWER_KEYS = ("vision_tower", "audio_tower")


def peft_tower_exclude(model) -> "str | None":
    """Return a PEFT ``exclude_modules`` regex for a multimodal base, else None.

    Detected from the actual module tree (not the model-id) so it is robust to
    the LoRA wrapper prefix and to future multimodal variants.
    """
    has_tower = any(
        any(k in name for k in _MM_TOWER_KEYS) for name, _ in model.named_modules()
    )
    return r".*(?:" + "|".join(_MM_TOWER_KEYS) + r").*" if has_tower else None
