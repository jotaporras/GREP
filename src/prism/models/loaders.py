import json
import os
import re
from typing import Tuple

import torch
from peft import PeftConfig, get_peft_model
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizer

from prism.models import gnn_llm
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


def load_gnn_config(path: str) -> dict:
    """Flat GNN rebuild config for a graph-augmented checkpoint.

    Current checkpoints store ONE ``train_config.json`` with shared run metadata
    (architecture / base_model / text_edge_list / injection_scope) at the top
    level and the architecture hyperparameters nested under ``"gnn"``. Legacy
    checkpoints store a single flat ``gnn_config.json`` instead. Returns the FLAT
    merged dict either way — the shape the rebuild branches below consume.
    """
    tc_path = os.path.join(path, "train_config.json")
    if os.path.exists(tc_path):
        with open(tc_path) as f:
            tc = json.load(f)
        if "gnn" in tc:
            return {**tc["gnn"], **{k: v for k, v in tc.items() if k != "gnn"}}
    with open(os.path.join(path, "gnn_config.json")) as f:
        return json.load(f)


def graph_augmented_llm_from_pretrained(
    path: str,
    load_in_4bit: bool = False,
    device: int = -1,
) -> Tuple[gnn_llm.GraphAugmentedLLM, PreTrainedTokenizer]:
    """Load a GraphAugmentedLLM checkpoint saved by GraphSFTTrainer.

    Expects the checkpoint directory to contain:
      - train_config.json    run metadata + "gnn" hyperparameters (legacy
                             checkpoints: a flat gnn_config.json instead)
      - gnn_weights.pt       pe_model and pe_proj state dicts
      - adapter_config.json  (optional — present when freeze_llm=False)
      - tokenizer files

    Args:
        device: Physical GPU index (e.g. 0, 1). -1 uses device_map="auto".
    """
    device_map = {"": 0} if device >= 0 else "auto"

    gnn_cfg = load_gnn_config(path)

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
        # Learnable relative-PE mask: rebuild the Psi producer and load it; adjacency +
        # mask rebuild from gnn_config. The LoRA adapter was already merged into `llm`.
        # Navigator mode (pe_gt_from/semantic_gt_from) uses NavigatorPE as the Psi producer;
        # otherwise a standalone GraphTransformer.
        if gnn_cfg.get("pe_gt_from") or gnn_cfg.get("semantic_gt_from"):
            pe_gt = gt_module.GraphTransformer(
                num_layers=gnn_cfg["gt_num_layers"], pe_hidden_channels=gnn_cfg["pe_hidden_channels"],
                pe_num_layers=gnn_cfg["pe_num_layers"], d_model=gnn_cfg["d_model"], heads=gnn_cfg["gt_heads"],
                num_samples=gnn_cfg["num_samples"], dropout=gnn_cfg["dropout"], k_pe=gnn_cfg["k_pe"],
                k_gt=gnn_cfg["k_gt"], eps=gnn_cfg["eps"], use_layer_norm=gnn_cfg["use_layer_norm"],
                node_feature_dim=None,
            )
            semantic_gt = gt_module.SemanticGraphTransformer(
                node_feature_dim=gnn_cfg["d_model"], d_model=gnn_cfg["d_model"],
                num_layers=gnn_cfg["gt_num_layers"], heads=gnn_cfg["gt_heads"],
                dropout=gnn_cfg["dropout"], k_gt=gnn_cfg["k_gt"],
            )
            pe_model = gt_module.NavigatorPE(pe_gt, semantic_gt)
        else:
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
    elif architecture in ("postfusion_graph_llm", "composite_graph_gt"):
        raise ValueError(
            f"architecture {architecture!r} was removed from the codebase (legacy "
            "e7/e10 experiments); check out an older commit to reload this checkpoint.")
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


def load_navigator_pe_into(model, pe_gt_from: str, semantic_gt_from: str) -> None:
    """Load the notebook's pretrained PE GT + Semantic GT (AGT) into a NavigatorPE pe_model.

    Both args are plain state_dict ``.pt`` files (path_navigator_gt.pt / path_navigator_agt.pt).
    """
    pe = getattr(model, "pe_model", None)
    if pe is None or not hasattr(pe, "pe_gt") or not hasattr(pe, "semantic_gt"):
        raise RuntimeError(
            "model.pe_model is not a NavigatorPE (gnn.pe_gt_from/semantic_gt_from set?).")
    pe.pe_gt.load_state_dict(torch.load(pe_gt_from, map_location="cpu"), strict=False)
    pe.semantic_gt.load_state_dict(torch.load(semantic_gt_from, map_location="cpu"), strict=False)
    print(f"[navigator] loaded PE GT {pe_gt_from} + Semantic GT {semantic_gt_from}")


def load_pe_weights_into(model, init_pe_from: str, architecture: str) -> None:
    """Load a saved PE module (``gnn_weights.pt``) from a prior stage into ``model``.

    Operates on a training model (no merge, no eval). Supports the
    GraphAugmentedLLM PE layouts (rpearl_llm / rpearl_gt_llm; pe_model + pe_proj
    [+ optional pe_gain / pe_norm]) and the mask-only ``learnable_graph_mask``
    (a standalone GraphTransformer as ``pe_model``, no pe_proj/pe_gain/pe_norm).

    Args:
        model: training model exposing ``pe_model`` (and, for the GraphAugmented
            layouts, ``pe_proj`` / ``pe_gain`` / ``pe_norm``).
        init_pe_from: path to a checkpoint directory containing ``gnn_weights.pt``.
        architecture: ``"rpearl_llm"``, ``"rpearl_gt_llm"``, or
            ``"learnable_graph_mask"``.
    """
    weights_path = os.path.join(init_pe_from, "gnn_weights.pt")
    gnn_weights = torch.load(weights_path, map_location="cpu")

    if architecture == "learnable_graph_mask":
        # Mask-only relative PE: the standalone GraphTransformer *is* the whole PE
        # (the mask uses Psi Psi^T directly — there is no pe_proj/pe_gain/pe_norm).
        # Accept an rpearl_gt_llm / edge-detector checkpoint ("gt_model") or another
        # learnable_graph_mask checkpoint ("pe_model"). Strict on keys:
        # a missing/unexpected key means the gnn.* GT hyperparameters did NOT
        # reproduce the pretrained GT (silent strict=False drop is exactly the
        # config-drift failure we want to make loud).
        gt_state = gnn_weights.get("gt_model", gnn_weights.get("pe_model"))
        if gt_state is None:
            raise KeyError(
                f"{weights_path} has neither 'gt_model' nor 'pe_model'; cannot "
                "initialise a learnable_graph_mask GT from it.")
        missing, unexpected = model.pe_model.load_state_dict(gt_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "learnable_graph_mask GT init did not match the checkpoint "
                f"(missing={list(missing)}, unexpected={list(unexpected)}). The "
                "gnn.* GT hyperparameters (gt_num_layers/pe_num_layers/num_samples/"
                "k_pe/k_gt/d_model/pe_hidden_channels/gt_heads) must reproduce the "
                "pretrained GT exactly.")
        print(f"[multistage] loaded GT (relative-PE) weights from {weights_path}")
        return

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
