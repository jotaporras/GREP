"""
Module with different LLM-based planner architectures.

Public surface:
- `build_planner_architecture` — construct (model, collator) for a given TrainConfig.
- `peft_tower_exclude`         — PEFT exclude_modules regex for multimodal bases (Gemma-4 31B etc.).
"""

from prism.data import data
from prism.models import gnn_llm
from prism.models import r_pearl as r_pearl_module
from prism.models import gt as gt_module


def build_planner_model(config, llm, tokenizer):
    """Instantiate the model and data collator for the given architecture.

    Args:
        config: TrainConfig (duck-typed; read-only).
        llm: base AutoModelForCausalLM already loaded on the target device.
        tokenizer: corresponding tokenizer with pad token set.

    Returns:
        (model, collator) — model is the architecture wrapper (or ``llm`` itself
        for the plain-LLM baseline); collator is the matching data collator.
    """
    _text_hidden = llm.config.get_text_config().hidden_size
    _node_feature_dim = _text_hidden if config.pe_node_features == "word_embeddings" else None

    if config.architecture == "rpearl_llm":
        # R-PEARL only: GCN positional encodings, no GT attention blocks.
        pe_model = r_pearl_module.RandomGNNPositionalEncodings(
            pe_hidden_channels=config.pe_hidden_channels,
            pe_num_layers=config.pe_num_layers,
            d_model=config.d_model,
            num_samples=config.num_samples,
            dropout=config.dropout,
            k=config.k_pe,
            eps=config.eps,
            use_layer_norm=config.use_layer_norm,
            node_feature_dim=_node_feature_dim,
        )
        model = gnn_llm.GraphAugmentedLLM(llm, pe_model, d_model=config.d_model,
                                          eps=config.eps, pe_gain_init=config.pe_gain_init,
                                          disable_graph_token_rope=config.disable_graph_token_rope,
                                          use_pe_norm=config.use_pe_norm,
                                          pe_node_features=config.pe_node_features)
        collator = data.SpineDataCollator(tokenizer, mlm=False)

        if config.freeze_llm:
            model.llm.requires_grad_(False)
    elif config.architecture == "rpearl_gt_llm":
        # Full Graph Transformer: R-PEARL inside Sparse Attention blocks.
        pe_model = gt_module.GraphTransformer(
            num_layers=config.gt_num_layers,
            pe_hidden_channels=config.pe_hidden_channels,
            pe_num_layers=config.pe_num_layers,
            d_model=config.d_model,
            heads=config.gt_heads,
            num_samples=config.num_samples,
            dropout=config.dropout,
            k_pe=config.k_pe,
            k_gt=config.k_gt,
            eps=config.eps,
            use_layer_norm=config.use_layer_norm,
            node_feature_dim=_node_feature_dim,
        )
        model = gnn_llm.GraphAugmentedLLM(llm, pe_model, d_model=config.d_model,
                                          eps=config.eps, pe_gain_init=config.pe_gain_init,
                                          disable_graph_token_rope=config.disable_graph_token_rope,
                                          use_pe_norm=config.use_pe_norm,
                                          pe_node_features=config.pe_node_features)
        collator = data.SpineDataCollator(tokenizer, mlm=False)

        if config.freeze_llm:
            model.llm.requires_grad_(False)
    elif config.architecture == "gt_llm":
        # Pure Graph Transformer over semantic node features — no R-PEARL, no probes.
        if config.pe_node_features != "word_embeddings":
            raise ValueError(
                "architecture 'gt_llm' requires pe_node_features='word_embeddings' "
                f"(got {config.pe_node_features!r}); the GT has no random-probe input."
            )
        pe_model = gt_module.SemanticGraphTransformer(
            node_feature_dim=_text_hidden,
            d_model=config.d_model,
            num_layers=config.gt_num_layers,
            heads=config.gt_heads,
            dropout=config.dropout,
            k_gt=config.k_gt,
        )
        model = gnn_llm.GraphAugmentedLLM(llm, pe_model, d_model=config.d_model,
                                          eps=config.eps, pe_gain_init=config.pe_gain_init,
                                          disable_graph_token_rope=config.disable_graph_token_rope,
                                          use_pe_norm=config.use_pe_norm,
                                          pe_node_features=config.pe_node_features)
        collator = data.SpineDataCollator(tokenizer, mlm=False)

        if config.freeze_llm:
            model.llm.requires_grad_(False)
    elif config.architecture == "graph_mask_llm":
        # Parameter-free structural attention mask: node tokens attend only along graph
        # edges. Reuses SpineDataCollator so graphs + injection_maps reach the model.
        model = gnn_llm.GraphMaskLLM(
            llm, k_hops=config.mask_k_hops, symmetrize=config.mask_symmetrize,
            use_edges=config.mask_use_edges)
        collator = data.SpineDataCollator(tokenizer, mlm=False)

        if config.freeze_llm:
            model.llm.requires_grad_(False)
    elif config.architecture == "composite_graph_gt":
        # Composite-graph pipeline: token-cycle + scene graph + cross-links; R-PEARL +
        # GT refine it; cold-start gate injects into the RoPE-disabled LLM.
        gt_model = gt_module.GraphTransformer(
            num_layers=config.gt_num_layers,
            pe_hidden_channels=config.pe_hidden_channels,
            pe_num_layers=config.pe_num_layers,
            d_model=config.d_model,
            heads=config.gt_heads,
            num_samples=config.num_samples,
            dropout=config.dropout,
            k_pe=config.k_pe,
            k_gt=config.k_gt,
            eps=config.eps,
            use_layer_norm=config.use_layer_norm,
            probe_distribution=config.probe_distribution,
            m_test=config.m_test,
            max_gather_rows=config.max_gather_rows,
            fixed_seed_mode=config.fixed_seed_mode,
            fixed_seed_value=config.fixed_seed_value,
            pe_readout=config.pe_readout,
            center_second_moment=config.pe_center_moment,
        )
        composite_kwargs = dict(
            gate_init=config.gate_init,
            gate_per_dim=config.gate_per_dim,
            injection_mode=config.injection_mode,
            disable_llm_rope=config.disable_rope,
            cycle_weight=config.cycle_weight,
            cycle_directed=config.cycle_directed,
            crosslink_weight=config.crosslink_weight,
            crosslink_mention_to_node=config.crosslink_mention_to_node,
            crosslink_mention_clique=config.crosslink_mention_clique,
        )
        if config.pe_qk_injection or config.c_per_layer or config.c_bias:
            # Selects InjectedCompositeGraphLLM: pe_qk_injection adds GT code to q/k/v;
            # c_per_layer replaces q/k with C_tok; c_bias uses C_tok as an additive bias.
            composite_kwargs["inject_v"] = config.pe_inject_v
            composite_kwargs["c_per_layer"] = config.c_per_layer
            composite_kwargs["c_bias"] = config.c_bias
            composite_kwargs["use_scene_bias"] = config.use_scene_bias
            composite_kwargs["c_kernel"] = config.c_kernel
            model = gnn_llm.InjectedCompositeGraphLLM(
                llm, gt_model, d_model=config.d_model, **composite_kwargs)
        else:
            model = gnn_llm.CompositeGraphLLM(
                llm, gt_model, d_model=config.d_model, **composite_kwargs)
        collator = data.SpineDataCollator(tokenizer, mlm=False)

        if config.freeze_llm:
            model.llm.requires_grad_(False)
    elif config.architecture == "llm":
        # Pure LLM baseline; TokenIndexCollator carries graph-token index columns so
        # graph_acc/* metrics are comparable to graph architectures.
        model = llm
        collator = data.TokenIndexCollator(tokenizer, mlm=False)
    else:
        raise ValueError(
            f"Unknown architecture: {config.architecture!r}. "
            "Choose 'rpearl_llm', 'rpearl_gt_llm', 'gt_llm', 'graph_mask_llm', "
            "'composite_graph_gt', or 'llm'.")

    return model, collator


# Multimodal Gemma-4 bases (e.g. gemma-4-31B) include vision/audio towers whose
# projections use Gemma4ClippableLinear (not nn.Linear) — PEFT cannot adapt them,
# and their leaf names collide with the text decoder. Exclude them from LoRA targets.
# Text-only / "unified" bases (Llama, gemma-4-12B) have no towers; regex hits nothing.
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
