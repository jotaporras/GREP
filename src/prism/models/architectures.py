"""
Module with different LLM-based planner architectures.

Public surface:
- `build_planner_architecture` — construct (model, collator) for a given TrainConfig.
- `peft_tower_exclude`         — PEFT exclude_modules regex for multimodal bases (Gemma-4 31B etc.).
"""

from prism.data import data
from prism.models import gnn_llm
from prism.models import composite_graph_llm
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
    _node_feature_dim = _text_hidden if config.gnn.pe_node_features == "word_embeddings" else None

    if config.gnn.arch == "rpearl_llm":
        # R-PEARL only: GCN positional encodings, no GT attention blocks.
        pe_model = r_pearl_module.RandomGNNPositionalEncodings(
            pe_hidden_channels=config.gnn.pe_hidden_channels,
            pe_num_layers=config.gnn.pe_num_layers,
            d_model=config.gnn.d_model,
            num_samples=config.gnn.num_samples,
            dropout=config.gnn.dropout,
            k=config.gnn.k_pe,
            eps=config.gnn.eps,
            use_layer_norm=config.gnn.use_layer_norm,
            node_feature_dim=_node_feature_dim,
        )
        model = gnn_llm.GraphAugmentedLLM(llm, pe_model, d_model=config.gnn.d_model,
                                          eps=config.gnn.eps, pe_gain_init=config.gnn.pe_gain_init,
                                          disable_graph_token_rope=config.model.disable_graph_token_rope,
                                          use_pe_norm=config.gnn.use_pe_norm,
                                          pe_node_features=config.gnn.pe_node_features)
        collator = data.SpineDataCollator(tokenizer, mlm=False)

        if config.trainer.freeze_llm:
            model.llm.requires_grad_(False)
    elif config.gnn.arch == "rpearl_gt_llm":
        # Full Graph Transformer: R-PEARL inside Sparse Attention blocks.
        pe_model = gt_module.GraphTransformer(
            num_layers=config.gnn.gt_num_layers,
            pe_hidden_channels=config.gnn.pe_hidden_channels,
            pe_num_layers=config.gnn.pe_num_layers,
            d_model=config.gnn.d_model,
            heads=config.gnn.gt_heads,
            num_samples=config.gnn.num_samples,
            dropout=config.gnn.dropout,
            k_pe=config.gnn.k_pe,
            k_gt=config.gnn.k_gt,
            eps=config.gnn.eps,
            use_layer_norm=config.gnn.use_layer_norm,
            node_feature_dim=_node_feature_dim,
        )
        model = gnn_llm.GraphAugmentedLLM(llm, pe_model, d_model=config.gnn.d_model,
                                          eps=config.gnn.eps, pe_gain_init=config.gnn.pe_gain_init,
                                          disable_graph_token_rope=config.model.disable_graph_token_rope,
                                          use_pe_norm=config.gnn.use_pe_norm,
                                          pe_node_features=config.gnn.pe_node_features)
        collator = data.SpineDataCollator(tokenizer, mlm=False)

        if config.trainer.freeze_llm:
            model.llm.requires_grad_(False)
    elif config.gnn.arch == "gt_llm":
        # Pure Graph Transformer over semantic node features — no R-PEARL, no probes.
        if config.gnn.pe_node_features != "word_embeddings":
            raise ValueError(
                "architecture 'gt_llm' requires pe_node_features='word_embeddings' "
                f"(got {config.gnn.pe_node_features!r}); the GT has no random-probe input."
            )
        pe_model = gt_module.SemanticGraphTransformer(
            node_feature_dim=_text_hidden,
            d_model=config.gnn.d_model,
            num_layers=config.gnn.gt_num_layers,
            heads=config.gnn.gt_heads,
            dropout=config.gnn.dropout,
            k_gt=config.gnn.k_gt,
        )
        model = gnn_llm.GraphAugmentedLLM(llm, pe_model, d_model=config.gnn.d_model,
                                          eps=config.gnn.eps, pe_gain_init=config.gnn.pe_gain_init,
                                          disable_graph_token_rope=config.model.disable_graph_token_rope,
                                          use_pe_norm=config.gnn.use_pe_norm,
                                          pe_node_features=config.gnn.pe_node_features)
        collator = data.SpineDataCollator(tokenizer, mlm=False)

        if config.trainer.freeze_llm:
            model.llm.requires_grad_(False)
    elif config.gnn.arch == "graph_mask_llm":
        # Parameter-free structural attention mask: node tokens attend only along graph
        # edges. Reuses SpineDataCollator so graphs + injection_maps reach the model.
        model = gnn_llm.GraphMaskLLM(
            llm, k_hops=config.gnn.mask_k_hops, symmetrize=config.gnn.mask_symmetrize,
            use_edges=config.gnn.mask_use_edges)
        collator = data.SpineDataCollator(tokenizer, mlm=False)

        if config.trainer.freeze_llm:
            model.llm.requires_grad_(False)
    elif config.gnn.arch == "composite_graph_gt":
        # Composite-graph pipeline: token-cycle + scene graph + cross-links; R-PEARL +
        # GT refine it; cold-start gate injects into the RoPE-disabled LLM.
        gt_model = gt_module.GraphTransformer(
            num_layers=config.gnn.gt_num_layers,
            pe_hidden_channels=config.gnn.pe_hidden_channels,
            pe_num_layers=config.gnn.pe_num_layers,
            d_model=config.gnn.d_model,
            heads=config.gnn.gt_heads,
            num_samples=config.gnn.num_samples,
            dropout=config.gnn.dropout,
            k_pe=config.gnn.k_pe,
            k_gt=config.gnn.k_gt,
            eps=config.gnn.eps,
            use_layer_norm=config.gnn.use_layer_norm,
            probe_distribution=config.gnn.probe_distribution,
            max_gather_rows=config.gnn.max_gather_rows,
            fixed_seed_mode=config.gnn.fixed_seed_mode,
            fixed_seed_value=config.gnn.fixed_seed_value,
            pe_readout=config.gnn.pe_readout,
            center_second_moment=config.gnn.pe_center_moment,
        )
        composite_kwargs = dict(
            gate_init=config.gnn.gate_init,
            gate_per_dim=config.gnn.gate_per_dim,
            injection_mode=config.gnn.injection_mode,
            disable_llm_rope=config.model.disable_rope,
            cycle_weight=config.gnn.cycle_weight,
            cycle_directed=config.gnn.cycle_directed,
            crosslink_weight=config.gnn.crosslink_weight,
            crosslink_mention_to_node=config.gnn.crosslink_mention_to_node,
            crosslink_mention_clique=config.gnn.crosslink_mention_clique,
        )
        if config.gnn.pe_qk_injection or config.gnn.c_per_layer or config.gnn.c_bias:
            # Selects InjectedCompositeGraphLLM: pe_qk_injection adds GT code to q/k/v;
            # c_per_layer replaces q/k with C_tok; c_bias uses C_tok as an additive bias.
            composite_kwargs["inject_v"] = config.gnn.pe_inject_v
            composite_kwargs["c_per_layer"] = config.gnn.c_per_layer
            composite_kwargs["c_bias"] = config.gnn.c_bias
            composite_kwargs["use_scene_bias"] = config.gnn.use_scene_bias
            composite_kwargs["c_kernel"] = config.gnn.c_kernel
            model = composite_graph_llm.InjectedCompositeGraphLLM(
                llm, gt_model, d_model=config.gnn.d_model, **composite_kwargs)
        else:
            model = composite_graph_llm.CompositeGraphLLM(
                llm, gt_model, d_model=config.gnn.d_model, **composite_kwargs)
        collator = data.SpineDataCollator(tokenizer, mlm=False)

        if config.trainer.freeze_llm:
            model.llm.requires_grad_(False)
    elif config.gnn.arch == "llm":
        # Pure LLM baseline; TokenIndexCollator carries graph-token index columns so
        # graph_acc/* metrics are comparable to graph architectures.
        model = llm
        collator = data.TokenIndexCollator(tokenizer, mlm=False)
    else:
        raise ValueError(
            f"Unknown architecture: {config.gnn.arch!r}. "
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
