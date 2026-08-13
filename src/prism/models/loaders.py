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
        # is_bf16_supported() itself raises without a CUDA device; guard it the same way
        # train_v3._bf16_supported does, so --four-bit on a non-CUDA host fails in
        # bitsandbytes (with the real reason) instead of here.
        bnb_4bit_compute_dtype=(torch.bfloat16
                                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                                else torch.float16),
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


def _remap_psi_producer_state(pe_model, state: dict, path: str) -> dict:
    """Reconcile a saved Ψ state dict with the rebuilt producer's key layout.

    Three layouts have existed under ``gnn_weights.pt["pe_model"]``:

    * bare ``GraphTransformer``  — ``blocks.*`` / ``pe_model.*``  (the current Ψ)
    * ``NavigatorPE``            — ``pe_gt.*``                     (PE stage, wrapped)
    * ``TwoStagePE``             — ``pe_gt.*`` + ``semantic_gt.*``  (LEGACY two-stage Ψ)

    The first two hold the SAME function (Ψ = PE GT), differing only by a ``pe_gt.``
    prefix, so they are remapped into each other — a lossless key rename. The third is a
    DIFFERENT function (Ψ = SemanticGT(PE_GT(·))): dropping ``semantic_gt.*`` to fit a
    bare GT would reload the checkpoint as a Ψ the run never trained with, silently. That
    case raises here instead, naming the knob that reproduces it.
    """
    has_pe_gt = any(k.startswith("pe_gt.") for k in state)
    has_semantic = any(k.startswith("semantic_gt.") for k in state)
    wants_semantic = hasattr(pe_model, "semantic_gt")
    wants_pe_gt = hasattr(pe_model, "pe_gt")

    if has_semantic and not wants_semantic:
        raise RuntimeError(
            f"{path} was trained with the LEGACY two-stage Ψ producer "
            "(gnn_weights.pt['pe_model'] carries semantic_gt.*, i.e. "
            "Ψ = SemanticGT(PE_GT(graph))), but the rebuild from train_config.json is "
            f"{type(pe_model).__name__} whose Ψ = PE_GT(graph) alone — a DIFFERENT "
            "function under the same weights. Reloading it that way would report numbers "
            "for a model that was never trained. Restore gnn.semantic_gt_from in the "
            "recorded train_config.json (it selects gt.TwoStagePE) to reproduce the "
            "trained topology, or re-train with semantic_gt_from null.")
    if wants_semantic and not has_semantic:
        raise RuntimeError(
            f"{path} carries no semantic_gt.* but the rebuild is {type(pe_model).__name__} "
            "(the legacy two-stage Ψ, selected by gnn.semantic_gt_from in "
            "train_config.json). Clear gnn.semantic_gt_from so Ψ rebuilds as the PE GT "
            "alone, exactly as the run was trained.")
    if has_pe_gt and not wants_pe_gt:      # NavigatorPE checkpoint -> bare GraphTransformer
        return {k[len("pe_gt."):]: v for k, v in state.items()}
    if wants_pe_gt and not has_pe_gt:      # bare GraphTransformer checkpoint -> NavigatorPE
        return {"pe_gt." + k: v for k, v in state.items()}
    return state


def _load_psi_producer_state(pe_model, state: dict, path: str, gnn_cfg: dict) -> None:
    """Load a checkpointed Ψ producer (``gnn_weights.pt["pe_model"]``) FAIL-LOUD.

    Used by the ``learnable_graph_mask`` / ``wire_llm`` eval rebuilds, whose ``pe_model``
    is a standalone ``GraphTransformer`` (Ψ = PE GT) or, for pre-split checkpoints, a
    ``TwoStagePE`` (``pe_gt.*`` + ``semantic_gt.*``). Key sets that denote the same
    function are reconciled by :func:`_remap_psi_producer_state`; anything left over is a
    genuine topology/shape disagreement, and a ``strict=False`` load would drop every
    tensor and evaluate a randomly-initialised Ψ at full silence — the exact provenance
    failure this check exists to prevent. Mirrors the strictness ``load_pe_weights_into``
    already applies to the same tensors on the multistage path.
    """
    state = _remap_psi_producer_state(pe_model, state, path)
    missing, unexpected = pe_model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Psi producer rebuild for {path} does not match its gnn_weights.pt "
            f"(missing={list(missing)}, unexpected={list(unexpected)}). Rebuilt as "
            f"{type(pe_model).__name__} from train_config.json "
            f"(pe_gt_from={gnn_cfg.get('pe_gt_from')!r}, "
            f"semantic_gt_from={gnn_cfg.get('semantic_gt_from')!r}); the recorded "
            "gnn.semantic_gt_from must select the same topology (gt.TwoStagePE vs "
            "standalone GraphTransformer) and the gnn.* GT hyperparameters "
            "(gt_num_layers/gt_heads/k_gt/d_model/pe_hidden_channels/"
            "pe_num_layers/num_samples/k_pe) the same shapes as the trained run.")


def additive_model_from_config(llm, gnn_cfg: dict, path: str) -> gnn_llm.GraphAugmentedLLM:
    """Rebuild an additive-family ``GraphAugmentedLLM`` (Ψ tower + weights) around ``llm``.

    Extracted from :func:`graph_augmented_llm_from_pretrained` so the vLLM
    Ψ producer (``vllm_graph.psi_producer``) rebuilds the SAME tower from the
    SAME recorded config — ``llm`` there is an embeddings-only shim, here the
    full base model. Covers ``rpearl_gt_llm`` / ``gt_llm`` and the default
    R-PEARL branch; mask/WIRE archs are not additive and stay in the caller.
    """
    architecture = gnn_cfg.get("architecture", "rpearl_llm")
    pe_node_features = gnn_cfg.get("pe_node_features", "random")
    node_feature_dim = (llm.config.get_text_config().hidden_size
                        if pe_node_features == "word_embeddings" else None)

    if architecture == "rpearl_gt_llm":
        fuse_x = gnn_cfg.get("fuse_node_features", False)
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
            node_feature_dim=(llm.config.get_text_config().hidden_size if fuse_x
                              else node_feature_dim),
            fuse_node_features=fuse_x,
            directed=gnn_cfg.get("directed", False),
        )
        weights_key = "gt_model"
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
        weights_key = "pe_model"
        pe_node_features = gnn_cfg.get("pe_node_features", "word_embeddings")
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
            directed=gnn_cfg.get("directed", False),
        )
        weights_key = "pe_model"

    model = gnn_llm.GraphAugmentedLLM(llm, pe_model, d_model=gnn_cfg["d_model"],
                                      eps=gnn_cfg["eps"],
                                      pe_gain_init=gnn_cfg.get("pe_gain_init", 1.0),
                                      disable_graph_token_rope=gnn_cfg.get("disable_graph_token_rope", False),
                                      use_pe_norm=gnn_cfg.get("use_pe_norm", False),
                                      pe_node_features=pe_node_features)
    gnn_weights = torch.load(os.path.join(path, "gnn_weights.pt"), map_location="cpu")
    model.pe_model.load_state_dict(gnn_weights[weights_key], strict=False)
    model.pe_proj.load_state_dict(gnn_weights["pe_proj"])
    if "pe_gain" in gnn_weights:
        model.pe_gain.data.copy_(gnn_weights["pe_gain"])
    if model.pe_norm is not None and "pe_norm" in gnn_weights:
        model.pe_norm.load_state_dict(gnn_weights["pe_norm"])
    return model


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
        # 4-bit: merging requantizes base+delta to the nf4 grid, rounding the LoRA
        # away (erased the e14 fine-tunes: 0.700 -> 0.057). Keep the adapter
        # attached — llm's module tree already carries the loaded LoRA layers —
        # matching training, which never merges.
        if not load_in_4bit:
            llm = peft_model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(path)

    architecture = gnn_cfg.get("architecture", "rpearl_llm")

    # Semantic-feature mode: GNN input width = LLM text hidden size.
    pe_node_features = gnn_cfg.get("pe_node_features", "random")
    node_feature_dim = (llm.config.get_text_config().hidden_size
                        if pe_node_features == "word_embeddings" else None)

    if architecture in ("rpearl_gt_llm", "gt_llm"):
        model = additive_model_from_config(llm, gnn_cfg, path)
    elif architecture == "wire_llm":
        # WIRE rotary injection: rebuild the Psi producer + the per-layer omega tables.
        # omega is loaded STRICTLY (not regenerated from the seed) so a checkpoint can
        # never silently diverge from the frequencies the run was trained with.
        # The Psi producer is a standalone GraphTransformer (Psi = PE GT), or the legacy
        # TwoStagePE when the run recorded semantic_gt_from (same factory as training).
        pe_model = gt_module.build_psi_producer(gnn_cfg)
        # Absent key = a run predating wire_composite, i.e. the scene-graph WIRE. The
        # two are different functions of the same state_dict (composite Ψ, RoPE off on
        # the block), so this must be recorded, never defaulted true.
        _composite = bool(gnn_cfg.get("wire_composite", False))
        _wire_cls = gnn_llm.CompositeWireGraphLLM if _composite else gnn_llm.WireGraphLLM
        model = _wire_cls(
            llm, pe_model, d_model=gnn_cfg["d_model"],
            **({"tokenizer": tokenizer,
                "magnet_r": gnn_cfg.get("wire_magnet_r", 0.1250305176),
                "context_window": gnn_cfg.get("wire_context_window", 1024),
                "cycle_weight": gnn_cfg.get("wire_cycle_weight", 1.0),
                "cycle_causal": gnn_cfg.get("wire_cycle_causal", False),
                "crosslink_weight": gnn_cfg.get("wire_crosslink_weight", 0.1),
                "crosslink_bidirectional": gnn_cfg.get("wire_crosslink_bidirectional", True),
                "anchor_weight": gnn_cfg.get("wire_anchor_weight", 10.0)} if _composite else {}),
            layer_scope=gnn_cfg.get("wire_layer_scope", "dense"),
            sigma_init=gnn_cfg.get("wire_sigma_init", 0.01),
            freeze_sigma=gnn_cfg.get("wire_freeze_sigma", False),
            omega_seed=gnn_cfg.get("wire_omega_seed", 0),
            rotate_nope_planes=gnn_cfg.get("wire_rotate_nope_planes", False),
            max_angle=gnn_cfg.get("wire_max_angle", 1.0),
            pe_gain_init=gnn_cfg.get("pe_gain_init", 1.0),
            # Missing key = the run predates wire_decode entirely; "rotate" is the only
            # mode that evaluates, and "skip" was never the default. A recorded legacy
            # "error" is normalised (with a warning) inside WireGraphLLM.
            decode=gnn_cfg.get("wire_decode", "rotate"),
            # Absent key = a run written before wire_vanilla existed, which could ONLY
            # have been the expectation arm. Defaulting to True there would rebuild the
            # wrong architecture, so the recorded-config default differs from the
            # config-file default on purpose.
            vanilla=gnn_cfg.get("wire_vanilla", False),
            vanilla_omega_init=gnn_cfg.get("wire_vanilla_omega_init", "zero"),
        )
        gnn_weights = torch.load(os.path.join(path, "gnn_weights.pt"), map_location="cpu")
        _load_psi_producer_state(model.pe_model, gnn_weights["pe_model"], path, gnn_cfg)
        if "pe_gain" in gnn_weights:
            model.pe_gain.data.copy_(gnn_weights["pe_gain"])
        # The frequency store the ACTIVE mode uses must be present; the other is empty
        # in both the model and the checkpoint, so loading it is a no-op either way.
        needed = ("wire_omega",) if model._wire_vanilla else ("wire_eps", "wire_sigma")
        missing = [k for k in needed if k not in gnn_weights]
        if missing:
            raise KeyError(
                f"{os.path.join(path, 'gnn_weights.pt')} is missing {missing}; the WIRE "
                "frequencies were not checkpointed and cannot be recovered "
                "(regenerating them from wire_omega_seed would not be guaranteed "
                "identical across torch versions/devices). Expected keys for "
                f"wire_vanilla={model._wire_vanilla}: {list(needed)}.")
        for key, module in (("wire_eps", model._wire_eps),
                            ("wire_sigma", model._wire_sigma),
                            ("wire_omega", model._wire_omega)):
            if key in gnn_weights:
                module.load_state_dict(gnn_weights[key])
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
        # Psi = PE GT (a standalone GraphTransformer); a recorded semantic_gt_from selects
        # the legacy two-stage TwoStagePE so pre-split checkpoints reload as trained.
        pe_model = gt_module.build_psi_producer(gnn_cfg, node_feature_dim=node_feature_dim)
        # Absent key = a run predating mask_composite, i.e. the Psi Psi^T mask. The two
        # are different functions of the same state_dict (beta * C_tok over the composite
        # token block), so this must be recorded, never defaulted true.
        _composite = bool(gnn_cfg.get("mask_composite", False))
        _mask_cls = gnn_llm.MagCompGraphLLM if _composite else gnn_llm.LearnableGraphMaskLLM
        model = _mask_cls(
            llm, pe_model,
            alpha=gnn_cfg.get("mask_alpha", 0.7),
            **({"tokenizer": tokenizer,
                "cycle_size": gnn_cfg.get("mask_cycle_size", 8192),
                "beta_init": gnn_cfg.get("mask_beta_init", 0.0),
                "magnet_r": gnn_cfg.get("mask_magnet_r", 0.126),
                "cycle_weight": gnn_cfg.get("mask_cycle_weight", 1.0),
                "cycle_causal": gnn_cfg.get("mask_cycle_causal", False),
                "crosslink_weight": gnn_cfg.get("mask_crosslink_weight", 0.1),
                "anchor_enabled": gnn_cfg.get("mask_anchor_enabled", True),
                "anchor_weight": gnn_cfg.get("mask_anchor_weight", 10.0),
                "cache_pe": gnn_cfg.get("cache_pe", True)} if _composite else {}),
            layer_scope=gnn_cfg.get("mask_layer_scope", "dense"),
            k_hops=gnn_cfg.get("mask_k_hops", 1),
            symmetrize=gnn_cfg.get("mask_symmetrize", True),
            use_edges=gnn_cfg.get("mask_use_edges", True),
            psi_scale=gnn_cfg.get("mask_psi_scale", "cosine"),
            buggy_causal_fold=gnn_cfg.get("mask_buggy_causal_fold", False),
            disable_graph_token_rope=gnn_cfg.get("disable_graph_token_rope", False),
        )
        gnn_weights = torch.load(os.path.join(path, "gnn_weights.pt"), map_location="cpu")
        _load_psi_producer_state(model.pe_model, gnn_weights["pe_model"], path, gnn_cfg)
        if _composite:
            # beta is the whole graph channel's gain; a checkpoint without it would
            # evaluate at beta_init (0 = the base LLM) and report a clean load.
            if "mask_beta" not in gnn_weights:
                raise KeyError(
                    f"{os.path.join(path, 'gnn_weights.pt')} is missing 'mask_beta': the "
                    "learned bias scale of a mask_composite run. Without it the rebuild "
                    "would silently evaluate at mask_beta_init, i.e. the base LLM at 0.")
            model.beta.data.copy_(gnn_weights["mask_beta"].to(model.beta.device))
    elif architecture in ("postfusion_graph_llm", "composite_graph_gt"):
        raise ValueError(
            f"architecture {architecture!r} was removed from the codebase (legacy "
            "e7/e10 experiments); check out an older commit to reload this checkpoint.")
    else:
        model = additive_model_from_config(llm, gnn_cfg, path)

    model.eval()
    return model, tokenizer


def load_navigator_pe_into(model, pe_gt_from: str, semantic_gt_from: str) -> None:
    """Load the notebook's pretrained PE GT (+ the LEGACY Semantic GT) into the pe_model.

    Both args are plain state_dict ``.pt`` files (path_navigator_gt.pt / path_navigator_agt.pt).
    The Ψ producer is the PE GT alone (``gt.build_psi_producer`` returns a standalone
    ``GraphTransformer``), so ``semantic_gt_from`` must be falsy for every current run; it
    is honoured only for the legacy ``gt.TwoStagePE`` producer, which exists so pre-split
    checkpoints reload as the function they were trained as.

    Architecture-agnostic: it only needs ``model.pe_model``, so it serves every Ψ-consuming
    architecture — ``learnable_graph_mask`` (Ψ Ψᵀ attention bias, or ``β·C_tok`` over the
    composite graph under ``mask_composite``) and ``wire_llm`` (Ψ as q/k rotation angles).
    This is also the channel a PRETRAINED MagNet GT arrives through: the notebook's
    resistance-regression stage (``2026-08-10 e17_magnet_composite_graphs.ipynb`` §3)
    trains one with the charge PINNED (``learn_r=False``, so the target is a fixed
    function of the topology) and saves ``gnn.state_dict()``. Its key set is then the
    consumer's minus ``*.r_logit``, and exactly that difference is tolerated below — the
    charge cold-starts at the consumer's own ``r`` init, which is the only sane value for
    a parameter the source never had. Any OTHER missing/unexpected key is a genuine
    topology mismatch and still raises.
    """
    pe = model.pe_model
    # The GT to fill is the wrapped `pe_gt` (NavigatorPE / TwoStagePE) or `pe_model` itself
    # (the standalone GraphTransformer that build_psi_producer returns).
    target = pe.pe_gt if hasattr(pe, "pe_gt") else pe
    # Strict on keys (missing/unexpected => the gnn.* GT hyperparameters do NOT reproduce the
    # pretrained GT; a size mismatch raises from load_state_dict regardless) so a dimension
    # mismatch fails loudly instead of silently loading a partially-random PE.
    missing, unexpected = target.load_state_dict(
        torch.load(pe_gt_from, map_location="cpu"), strict=False)
    charge_only = [k for k in missing if k.endswith(".r_logit")]
    missing = [k for k in missing if not k.endswith(".r_logit")]
    if charge_only:
        backbone = getattr(getattr(target, "pe_model", None), "pe_gcn", None)
        print(f"[navigator] {pe_gt_from} pins the MagNet charge ({len(charge_only)} "
              f"r_logit keys absent); r cold-starts at this run's init "
              f"r={float(torch.as_tensor(backbone.r).detach()):.6f}")
    if missing or unexpected:
        raise RuntimeError(
            f"PE GT load from {pe_gt_from} did not match the {type(target).__name__} "
            f"(missing={list(missing)}, unexpected={list(unexpected)}); the gnn.* GT "
            "hyperparameters (d_model/pe_hidden_channels/pe_num_layers/k_pe/gt_num_layers/"
            "gt_heads) must reproduce the pretrained GT exactly.")
    if hasattr(pe, "semantic_gt"):
        if not semantic_gt_from:
            raise RuntimeError(
                f"pe_model is a {type(pe).__name__} (the legacy two-stage Ψ) but "
                "gnn.semantic_gt_from is unset — its Semantic GT would stay randomly "
                "initialised and silently corrupt Ψ.")
        pe.semantic_gt.load_state_dict(torch.load(semantic_gt_from, map_location="cpu"), strict=True)
        print(f"[navigator] loaded PE GT {pe_gt_from} + LEGACY Semantic GT {semantic_gt_from} "
              f"(Ψ = SemanticGT(PE_GT(graph)))")
        return
    if semantic_gt_from:
        raise RuntimeError(
            "gnn.semantic_gt_from is set but the Ψ producer is a standalone GraphTransformer "
            "(Ψ = PE GT alone). The Semantic GT is the AGT head and belongs to "
            "gt.NavigatorGT, not to Ψ; clear gnn.semantic_gt_from, or set gnn.pe_gt_from "
            "too if you are deliberately reloading a legacy two-stage checkpoint.")
    print(f"[navigator] loaded PE GT {pe_gt_from} (Ψ = PE GT alone)")


def load_pe_weights_into(model, init_pe_from: str, architecture: str) -> None:
    """Load a saved PE module (``gnn_weights.pt``) from a prior stage into ``model``.

    Operates on a training model (no merge, no eval). Supports the
    GraphAugmentedLLM PE layouts (rpearl_llm / rpearl_gt_llm; pe_model + pe_proj
    [+ optional pe_gain / pe_norm]), the mask-only ``learnable_graph_mask``
    (a standalone GraphTransformer as ``pe_model``, no pe_proj/pe_gain/pe_norm)
    and ``wire_llm`` (``pe_model`` + the angle gate ``pe_gain`` + the frozen
    ``_wire_eps`` directions + the learned ``_wire_sigma``; no pe_proj/pe_norm).

    ``wire_llm`` accepts either a prior WIRE checkpoint (Ψ + gate + ε + σ all
    carried, so ω = σ·ε is the one that was trained) or a prior GT / mask
    checkpoint (Ψ only; the gate, ε and σ cold-start and the loader says so).

    Args:
        model: training model exposing ``pe_model`` (and, for the GraphAugmented
            layouts, ``pe_proj`` / ``pe_gain`` / ``pe_norm``; for ``wire_llm``,
            ``pe_gain`` / ``_wire_eps`` / ``_wire_sigma``).
        init_pe_from: path to a checkpoint directory containing ``gnn_weights.pt``.
        architecture: ``"rpearl_llm"``, ``"rpearl_gt_llm"``,
            ``"learnable_graph_mask"``, or ``"wire_llm"``.
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
        gt_state = _remap_psi_producer_state(model.pe_model, gt_state, weights_path)
        missing, unexpected = model.pe_model.load_state_dict(gt_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "learnable_graph_mask GT init did not match the checkpoint "
                f"(missing={list(missing)}, unexpected={list(unexpected)}). The "
                "gnn.* GT hyperparameters (gt_num_layers/pe_num_layers/num_samples/"
                "k_pe/k_gt/d_model/pe_hidden_channels/gt_heads) must reproduce the "
                "pretrained GT exactly, and gnn.semantic_gt_from must select the SAME "
                "Ψ-producer topology as the source run (a legacy TwoStagePE checkpoint "
                "carries semantic_gt.* keys the PE-only Ψ lacks).")
        print(f"[multistage] loaded GT (relative-PE) weights from {weights_path}")
        return

    if architecture == "wire_llm":
        # WIRE: the Psi producer + the angle gate + the frozen eps directions + the
        # learned per-layer sigma (exactly what trainers.save_model writes). Returns
        # BEFORE the pe_proj/pe_norm block below: WireGraphLLM deliberately has
        # neither (rotation angles do not live in the LLM's hidden space).
        # Two source layouts are accepted:
        #   (a) a prior GT / mask checkpoint ("gt_model" or "pe_model", no wire_*) —
        #       carry Psi only, and SAY what was not carried so a partial carry is
        #       never mistaken for a full WIRE resume. pe_gain is NOT carried from
        #       such a checkpoint: there it gates an additive hidden-space PE, here
        #       it gates a rotation angle.
        #   (b) a prior wire_llm checkpoint — carry Psi + pe_gain + eps + sigma, so
        #       the reloaded model rotates by the omega = sigma*eps it trained with.
        gt_state = gnn_weights.get("pe_model", gnn_weights.get("gt_model"))
        if gt_state is None:
            raise KeyError(
                f"{weights_path} has neither 'pe_model' nor 'gt_model'; cannot "
                "initialise a wire_llm Psi producer from it.")
        # Strict on keys for the same reason as learnable_graph_mask above.
        gt_state = _remap_psi_producer_state(model.pe_model, gt_state, weights_path)
        missing, unexpected = model.pe_model.load_state_dict(gt_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "wire_llm GT (Psi producer) init did not match the checkpoint "
                f"(missing={list(missing)}, unexpected={list(unexpected)}). The "
                "gnn.* GT hyperparameters (gt_num_layers/pe_num_layers/num_samples/"
                "k_pe/k_gt/d_model/pe_hidden_channels/gt_heads) must reproduce the "
                "pretrained GT exactly, and gnn.semantic_gt_from must select the SAME "
                "Ψ-producer topology as the source run (a legacy TwoStagePE checkpoint "
                "carries semantic_gt.* keys the PE-only Ψ lacks).")
        # Which store defines omega depends on the mode the TARGET model was built in.
        # vanilla: the learnable omega table. expectation: eps and sigma jointly — one
        # without the other is a malformed checkpoint, not a carry we can complete with
        # a fresh draw.
        # getattr default: a carrier that predates / omits the flag (e.g. a PE-only
        # holder in a cold-start carry) is the historical reparameterised form.
        is_vanilla = bool(getattr(model, "_wire_vanilla", False))
        if is_vanilla:
            wire_keys = ("wire_omega",)
            has_wire = "wire_omega" in gnn_weights and bool(gnn_weights["wire_omega"])
        else:
            wire_keys = ("wire_eps", "wire_sigma")
            # NON-EMPTY, not merely present: save_model always writes all three stores so
            # the key set does not depend on the mode, which means a VANILLA checkpoint
            # still carries 'wire_eps'/'wire_sigma' as empty dicts. Testing presence
            # alone would read those as a carry, skip the cross-mode guard below, and
            # strict-load {} — surfacing as a missing-key RuntimeError instead of the
            # loud "set gnn.wire_vanilla to match the source run". Mirrors the vanilla
            # branch above.
            has_eps = "wire_eps" in gnn_weights and bool(gnn_weights["wire_eps"])
            has_sigma = "wire_sigma" in gnn_weights and bool(gnn_weights["wire_sigma"])
            if has_eps != has_sigma:
                raise KeyError(
                    f"{weights_path} carries only one of 'wire_eps'/'wire_sigma' "
                    f"(eps={has_eps}, sigma={has_sigma}); omega = sigma*eps is defined "
                    "by both, so this checkpoint cannot initialise WIRE.")
            has_wire = has_eps
        # A WIRE checkpoint written in the OTHER mode carries the other mode's keys, so
        # its frequencies simply are not present under the names this mode reads. Fail
        # loud rather than cold-starting omega while reporting a full WIRE resume.
        if not has_wire:
            other = ("wire_eps", "wire_sigma") if is_vanilla else ("wire_omega",)
            if any(k in gnn_weights and bool(gnn_weights[k]) for k in other):
                raise KeyError(
                    f"{weights_path} is a WIRE checkpoint from the OTHER mode: it "
                    f"carries {[k for k in other if k in gnn_weights]} but this model "
                    f"was built with wire_vanilla={is_vanilla}, which reads "
                    f"{list(wire_keys)}. omega is not transferable between the "
                    "reparameterised (sigma*eps) and vanilla (learnable table) forms — "
                    "set gnn.wire_vanilla to match the source run.")
        carried, cold = ["pe_model"], []
        if has_wire:
            if "pe_gain" not in gnn_weights:
                raise KeyError(
                    f"{weights_path} is a wire_llm checkpoint ({'/'.join(wire_keys)} "
                    "present) but has no 'pe_gain'; the angle gate cannot be silently "
                    "reset to gnn.pe_gain_init on a WIRE resume.")
            model.pe_gain.data.copy_(gnn_weights["pe_gain"])
            carried.append("pe_gain")
            # strict=True: a key mismatch means a different set of WIRE-active layers,
            # a shape mismatch a different plane count / Psi width. Either way omega
            # would not be the one that was trained — fail loud, naming the knob.
            planes_knobs = ("gnn.wire_layer_scope / gnn.wire_rotate_nope_planes / "
                            "gnn.d_model (and the base model's head_dim)")
            stores = {
                "wire_eps": (model._wire_eps, planes_knobs),
                "wire_sigma": (model._wire_sigma, "gnn.wire_layer_scope"),
                "wire_omega": (model._wire_omega, planes_knobs),
            }
            for name in wire_keys:
                module, knobs = stores[name]
                try:
                    module.load_state_dict(gnn_weights[name], strict=True)
                except RuntimeError as e:
                    raise RuntimeError(
                        f"wire_llm {name} init did not match the checkpoint ({e}). "
                        f"{knobs} must reproduce the pretrained WIRE run exactly.") from e
                carried.append(name)
        else:
            cold = ["pe_gain", *wire_keys]
        print(f"[multistage] loaded WIRE weights from {weights_path}: "
              f"carried {'/'.join(carried)}")
        if cold:
            print(f"[multistage] NOT carried: {'/'.join(cold)} — cold-start at "
                  "gnn.pe_gain_init / the gnn.wire_omega_seed draw / gnn.wire_sigma_init "
                  "(GT-only checkpoint; this is NOT a full WIRE resume)")
        return

    if architecture == "rpearl_gt_llm":
        model.pe_model.load_state_dict(gnn_weights["gt_model"], strict=False)
    elif architecture == "rpearl_llm":
        model.pe_model.load_state_dict(gnn_weights["pe_model"], strict=False)
    else:
        raise NotImplementedError(
            "load_pe_weights_into is only wired for rpearl_llm / rpearl_gt_llm / "
            f"learnable_graph_mask / wire_llm, got {architecture!r}")
    model.pe_proj.load_state_dict(gnn_weights["pe_proj"])
    if "pe_gain" in gnn_weights:
        model.pe_gain.data.copy_(gnn_weights["pe_gain"])
    if getattr(model, "pe_norm", None) is not None and "pe_norm" in gnn_weights:
        model.pe_norm.load_state_dict(gnn_weights["pe_norm"])
    print(f"[multistage] loaded PE weights from {weights_path}")
