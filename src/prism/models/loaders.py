import json
import os
import re
from typing import Tuple

import torch
from peft import PeftConfig, get_peft_model
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizer

from prism.models import gnn_llm
from prism.models import composite_graph_llm
from prism.models import r_pearl
from prism.models import gt as gt_module


def _bnb_config(load_in_4bit: bool):
    if not load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def from_pretrained(
    path: str,
    load_in_4bit: bool = False,
    base_model: str = None,
    device: int = -1,
    **kwargs,
) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    """Load a plain LLM checkpoint (LoRA or full fine-tune) from a local path or HuggingFace Hub.

    For LoRA checkpoints (detected by the presence of adapter_config.json),
    the base model is loaded first and the adapter is applied on top.  The
    base model path is read from adapter_config.json unless ``base_model``
    is provided explicitly.

    Args:
        device: Physical GPU index (e.g. 0, 1). -1 uses device_map="auto".
    """
    device_map = {"": 0} if device >= 0 else "auto"
    adapter_cfg_path = os.path.join(path, "adapter_config.json")
    if os.path.exists(adapter_cfg_path):
        if base_model is None:
            with open(adapter_cfg_path) as f:
                base_model = json.load(f)["base_model_name_or_path"]
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype="auto",
            device_map=device_map,
            quantization_config=_bnb_config(load_in_4bit),
        )
        peft_model = get_peft_model(model, PeftConfig.from_pretrained(path))
        state_dict = load_file(os.path.join(path, "adapter_model.safetensors"))
        remapped = {re.sub(r'base_model\.model\.llm\.model\.', 'base_model.model.model.', k): v
                    for k, v in state_dict.items()}
        peft_model.load_state_dict(remapped, strict=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype="auto",
            device_map=device_map,
            quantization_config=_bnb_config(load_in_4bit),
        )
    tokenizer = AutoTokenizer.from_pretrained(path)
    model.eval()
    return model, tokenizer


def graph_augmented_llm_from_pretrained(
    path: str,
    load_in_4bit: bool = False,
    device: int = -1,
) -> Tuple[gnn_llm.GraphAugmentedLLM, PreTrainedTokenizer]:
    """Load a GraphAugmentedLLM checkpoint saved by GraphSFTTrainer.

    Expects the checkpoint directory to contain:
      - gnn_config.json      GNN hyperparameters + base_model path
      - gnn_weights.pt       pe_model and pe_proj state dicts
      - adapter_config.json  (optional — present when freeze_llm=False)
      - tokenizer files

    Args:
        device: Physical GPU index (e.g. 0, 1). -1 uses device_map="auto".
    """
    device_map = {"": 0} if device >= 0 else "auto"

    with open(os.path.join(path, "gnn_config.json")) as f:
        gnn_cfg = json.load(f)

    base_model_path = gnn_cfg["base_model"]

    llm = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype="auto",
        device_map=device_map,
        quantization_config=_bnb_config(load_in_4bit),
    )

    if os.path.exists(os.path.join(path, "adapter_config.json")):
        peft_model = get_peft_model(llm, PeftConfig.from_pretrained(path))
        state_dict = load_file(os.path.join(path, "adapter_model.safetensors"))
        remapped = {}
        for k, v in state_dict.items():
            k = re.sub(r'base_model\.model\.llm\.model\.', 'base_model.model.model.', k)
            k = re.sub(r'\.lora_([AB])\.weight', r'.lora_\1.default.weight', k)
            remapped[k] = v
        peft_model.load_state_dict(remapped, strict=False)
        llm = peft_model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(path)

    architecture = gnn_cfg.get("architecture", "rpearl_llm")

    # Semantic-feature mode: GNN input width = LLM text hidden size.
    pe_node_features = gnn_cfg.get("pe_node_features", "random")
    node_feature_dim = (llm.config.get_text_config().hidden_size
                        if pe_node_features == "word_embeddings" else None)

    if architecture == "rpearl_gt_llm":
        pe_model = gt_module.GraphTransformer(
            num_layers=gnn_cfg["gt_num_layers"],
            pe_hidden_channels=gnn_cfg["pe_hidden_channels"],
            pe_num_layers=gnn_cfg["pe_num_layers"],
            d_model=gnn_cfg["d_model"],
            heads=gnn_cfg["gt_heads"],
            num_samples=gnn_cfg["num_samples"],
            dropout=gnn_cfg["dropout"],
            k_pe=gnn_cfg["k_pe"],
            k_gt=gnn_cfg["k_gt"],
            eps=gnn_cfg["eps"],
            use_layer_norm=gnn_cfg["use_layer_norm"],
            node_feature_dim=node_feature_dim,
        )
        model = gnn_llm.GraphAugmentedLLM(llm, pe_model, d_model=gnn_cfg["d_model"],
                                          eps=gnn_cfg["eps"],
                                          pe_gain_init=gnn_cfg.get("pe_gain_init", 1.0),
                                          disable_graph_token_rope=gnn_cfg.get("disable_graph_token_rope", False),
                                          use_pe_norm=gnn_cfg.get("use_pe_norm", False),
                                          pe_node_features=pe_node_features)
        gnn_weights = torch.load(os.path.join(path, "gnn_weights.pt"), map_location="cpu")
        model.pe_model.load_state_dict(gnn_weights["gt_model"], strict=False)
        model.pe_proj.load_state_dict(gnn_weights["pe_proj"])
        if "pe_gain" in gnn_weights:
            model.pe_gain.data.copy_(gnn_weights["pe_gain"])
        if model.pe_norm is not None and "pe_norm" in gnn_weights:
            model.pe_norm.load_state_dict(gnn_weights["pe_norm"])
    elif architecture == "gt_llm":
        # Pure GT over semantic node features; key 'pe_model' holds the SemanticGraphTransformer.
        pe_model = gt_module.SemanticGraphTransformer(
            node_feature_dim=llm.config.get_text_config().hidden_size,
            d_model=gnn_cfg["d_model"],
            num_layers=gnn_cfg["gt_num_layers"],
            heads=gnn_cfg["gt_heads"],
            dropout=gnn_cfg["dropout"],
            k_gt=gnn_cfg["k_gt"],
        )
        model = gnn_llm.GraphAugmentedLLM(llm, pe_model, d_model=gnn_cfg["d_model"],
                                          eps=gnn_cfg["eps"],
                                          pe_gain_init=gnn_cfg.get("pe_gain_init", 1.0),
                                          disable_graph_token_rope=gnn_cfg.get("disable_graph_token_rope", False),
                                          use_pe_norm=gnn_cfg.get("use_pe_norm", False),
                                          pe_node_features=gnn_cfg.get("pe_node_features", "word_embeddings"))
        gnn_weights = torch.load(os.path.join(path, "gnn_weights.pt"), map_location="cpu")
        model.pe_model.load_state_dict(gnn_weights["pe_model"], strict=False)
        model.pe_proj.load_state_dict(gnn_weights["pe_proj"])
        if "pe_gain" in gnn_weights:
            model.pe_gain.data.copy_(gnn_weights["pe_gain"])
        if model.pe_norm is not None and "pe_norm" in gnn_weights:
            model.pe_norm.load_state_dict(gnn_weights["pe_norm"])
    elif architecture == "graph_mask_llm":
        # Parameter-free structural attention mask: rebuild from gnn_config, no weights
        # to load (the LoRA adapter was already merged into `llm` above).
        model = gnn_llm.GraphMaskLLM(
            llm,
            k_hops=gnn_cfg.get("mask_k_hops", 1),
            symmetrize=gnn_cfg.get("mask_symmetrize", True),
            use_edges=gnn_cfg.get("mask_use_edges", True),
            buggy_causal_fold=gnn_cfg.get("mask_buggy_causal_fold", False),
            layer_scope=gnn_cfg.get("mask_layer_scope", "all"),
        )
    elif architecture == "learnable_graph_mask":
        # Learnable relative-PE mask: rebuild the standalone GraphTransformer (Psi
        # producer) and load it; adjacency + mask rebuild from gnn_config. The LoRA
        # adapter was already merged into `llm` above.
        pe_model = gt_module.GraphTransformer(
            num_layers=gnn_cfg["gt_num_layers"],
            pe_hidden_channels=gnn_cfg["pe_hidden_channels"],
            pe_num_layers=gnn_cfg["pe_num_layers"],
            d_model=gnn_cfg["d_model"],
            heads=gnn_cfg["gt_heads"],
            num_samples=gnn_cfg["num_samples"],
            dropout=gnn_cfg["dropout"],
            k_pe=gnn_cfg["k_pe"],
            k_gt=gnn_cfg["k_gt"],
            eps=gnn_cfg["eps"],
            use_layer_norm=gnn_cfg["use_layer_norm"],
            node_feature_dim=node_feature_dim,
        )
        model = gnn_llm.LearnableGraphMaskLLM(
            llm, pe_model,
            alpha=gnn_cfg.get("mask_alpha", 0.7),
            layer_scope=gnn_cfg.get("mask_layer_scope", "dense"),
            k_hops=gnn_cfg.get("mask_k_hops", 1),
            symmetrize=gnn_cfg.get("mask_symmetrize", True),
            use_edges=gnn_cfg.get("mask_use_edges", True),
            psi_scale=gnn_cfg.get("mask_psi_scale", "cosine"),
            buggy_causal_fold=gnn_cfg.get("mask_buggy_causal_fold", False),
        )
        gnn_weights = torch.load(os.path.join(path, "gnn_weights.pt"), map_location="cpu")
        model.pe_model.load_state_dict(gnn_weights["pe_model"], strict=False)
    elif architecture == "composite_graph_gt":
        # Composite-graph model: Graph Transformer (R-PEARL inside) + cold-start gate
        # over a RoPE-disabled LLM. Rebuild from gnn_config and load the saved weights.
        gt_model = gt_module.GraphTransformer(
            num_layers=gnn_cfg["gt_num_layers"],
            pe_hidden_channels=gnn_cfg["pe_hidden_channels"],
            pe_num_layers=gnn_cfg["pe_num_layers"],
            d_model=gnn_cfg["d_model"],
            heads=gnn_cfg["gt_heads"],
            num_samples=gnn_cfg["num_samples"],
            dropout=gnn_cfg["dropout"],
            k_pe=gnn_cfg["k_pe"],
            k_gt=gnn_cfg["k_gt"],
            eps=gnn_cfg["eps"],
            use_layer_norm=gnn_cfg["use_layer_norm"],
            probe_distribution=gnn_cfg.get("probe_distribution", "gaussian"),
            max_gather_rows=gnn_cfg.get("max_gather_rows", 2_000_000),
            fixed_seed_mode=gnn_cfg.get("fixed_seed_mode", False),
            fixed_seed_value=gnn_cfg.get("fixed_seed_value", 0),
            pe_readout=gnn_cfg.get("pe_readout", "mean"),
            center_second_moment=gnn_cfg.get("pe_center_moment", True),
        )
        composite_kwargs = dict(
            gate_init=gnn_cfg.get("gate_init", 0.0),
            gate_per_dim=gnn_cfg.get("gate_per_dim", False),
            injection_mode=gnn_cfg.get("injection_mode", "interpolate"),
            disable_llm_rope=gnn_cfg.get("disable_rope", True),
            cycle_weight=gnn_cfg.get("cycle_weight", 1.0),
            cycle_directed=gnn_cfg.get("cycle_directed", True),
            crosslink_weight=gnn_cfg.get("crosslink_weight", 1.0),
            crosslink_mention_to_node=gnn_cfg.get("crosslink_mention_to_node", True),
            crosslink_mention_clique=gnn_cfg.get("crosslink_mention_clique", True),
        )
        if (gnn_cfg.get("pe_qk_injection", False) or gnn_cfg.get("c_per_layer", False)
                or gnn_cfg.get("c_bias", False)):
            # In-attention injection variants: pe_qk_injection adds GT code to q/k/v;
            # c_per_layer replaces q/k; c_bias (Design D) is an additive logit bias.
            model = composite_graph_llm.InjectedCompositeGraphLLM(
                llm, gt_model, d_model=gnn_cfg["d_model"],
                inject_v=gnn_cfg.get("pe_inject_v", True),
                c_per_layer=gnn_cfg.get("c_per_layer", False),
                c_bias=gnn_cfg.get("c_bias", False),
                use_scene_bias=gnn_cfg.get("use_scene_bias", True),
                c_kernel=gnn_cfg.get("c_kernel", "sampled"),
                **composite_kwargs,
            )
        else:
            model = composite_graph_llm.CompositeGraphLLM(llm, gt_model, d_model=gnn_cfg["d_model"], **composite_kwargs)
        gnn_weights = torch.load(os.path.join(path, "gnn_weights.pt"), map_location="cpu")
        model.gt_model.load_state_dict(gnn_weights["gt_model"], strict=False)
        model.injection.load_state_dict(gnn_weights["injection"])
        if hasattr(model, "pe_q_proj") and "pe_q_proj" in gnn_weights:
            model.pe_q_proj.load_state_dict(gnn_weights["pe_q_proj"])
            model.pe_k_proj.load_state_dict(gnn_weights["pe_k_proj"])
            if getattr(model, "pe_v_proj", None) is not None and "pe_v_proj" in gnn_weights:
                model.pe_v_proj.load_state_dict(gnn_weights["pe_v_proj"])
        if getattr(model, "c_bias", False) and "c_bias_gains" in gnn_weights:
            for k, v in gnn_weights["c_bias_gains"].items():
                if hasattr(model, k):                  # skip retired gains (e.g. lam_s)
                    getattr(model, k).data.copy_(v)
    else:
        pe_model = r_pearl.RandomGNNPositionalEncodings(
            pe_hidden_channels=gnn_cfg["pe_hidden_channels"],
            pe_num_layers=gnn_cfg["pe_num_layers"],
            d_model=gnn_cfg["d_model"],
            num_samples=gnn_cfg["num_samples"],
            dropout=gnn_cfg["dropout"],
            k=gnn_cfg["k_pe"],
            eps=gnn_cfg["eps"],
            use_layer_norm=gnn_cfg["use_layer_norm"],
            node_feature_dim=node_feature_dim,
        )
        model = gnn_llm.GraphAugmentedLLM(llm, pe_model, d_model=gnn_cfg["d_model"],
                                          eps=gnn_cfg["eps"],
                                          pe_gain_init=gnn_cfg.get("pe_gain_init", 1.0),
                                          disable_graph_token_rope=gnn_cfg.get("disable_graph_token_rope", False),
                                          use_pe_norm=gnn_cfg.get("use_pe_norm", False),
                                          pe_node_features=pe_node_features)
        gnn_weights = torch.load(os.path.join(path, "gnn_weights.pt"), map_location="cpu")
        model.pe_model.load_state_dict(gnn_weights["pe_model"], strict=False)
        model.pe_proj.load_state_dict(gnn_weights["pe_proj"])
        if "pe_gain" in gnn_weights:
            model.pe_gain.data.copy_(gnn_weights["pe_gain"])
        if model.pe_norm is not None and "pe_norm" in gnn_weights:
            model.pe_norm.load_state_dict(gnn_weights["pe_norm"])

    model.eval()
    return model, tokenizer


def load_pe_weights_into(model, init_pe_from: str, architecture: str) -> None:
    """Load a saved PE module (``gnn_weights.pt``) from a prior stage into ``model``.

    Operates on a training model (no merge, no eval). Only rpearl_llm /
    rpearl_gt_llm PE layouts are supported.

    Args:
        model: GraphAugmentedLLM training model with pe_model / pe_proj / pe_gain attributes.
        init_pe_from: path to a checkpoint directory containing ``gnn_weights.pt``.
        architecture: one of ``"rpearl_llm"`` or ``"rpearl_gt_llm"``.
    """
    weights_path = os.path.join(init_pe_from, "gnn_weights.pt")
    gnn_weights = torch.load(weights_path, map_location="cpu")
    if architecture == "rpearl_gt_llm":
        model.pe_model.load_state_dict(gnn_weights["gt_model"], strict=False)
    elif architecture == "rpearl_llm":
        model.pe_model.load_state_dict(gnn_weights["pe_model"], strict=False)
    else:
        raise NotImplementedError(
            f"load_pe_weights_into is only wired for rpearl_llm / rpearl_gt_llm, got {architecture!r}")
    model.pe_proj.load_state_dict(gnn_weights["pe_proj"])
    if "pe_gain" in gnn_weights:
        model.pe_gain.data.copy_(gnn_weights["pe_gain"])
    if getattr(model, "pe_norm", None) is not None and "pe_norm" in gnn_weights:
        model.pe_norm.load_state_dict(gnn_weights["pe_norm"])
    print(f"[multistage] loaded PE weights from {weights_path}")
