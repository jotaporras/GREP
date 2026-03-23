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
    **kwargs,
) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    """Load a plain LLM checkpoint (LoRA or full fine-tune) from a local path or HuggingFace Hub.

    For LoRA checkpoints (detected by the presence of adapter_config.json),
    the base model is loaded first and the adapter is applied on top.  The
    base model path is read from adapter_config.json unless ``base_model``
    is provided explicitly.
    """
    adapter_cfg_path = os.path.join(path, "adapter_config.json")
    if os.path.exists(adapter_cfg_path):
        if base_model is None:
            with open(adapter_cfg_path) as f:
                base_model = json.load(f)["base_model_name_or_path"]
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype="auto",
            device_map="auto",
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
            device_map="auto",
            quantization_config=_bnb_config(load_in_4bit),
        )
    tokenizer = AutoTokenizer.from_pretrained(path)
    model.eval()
    return model, tokenizer


def graph_augmented_llm_from_pretrained(
    path: str,
    load_in_4bit: bool = False,
) -> Tuple[gnn_llm.GraphAugmentedLLM, PreTrainedTokenizer]:
    """Load a GraphAugmentedLLM checkpoint saved by GraphSFTTrainer.

    Expects the checkpoint directory to contain:
      - gnn_config.json      GNN hyperparameters + base_model path
      - gnn_weights.pt       pe_model and pe_proj state dicts
      - adapter_config.json  (optional — present when freeze_llm=False)
      - tokenizer files
    """
    with open(os.path.join(path, "gnn_config.json")) as f:
        gnn_cfg = json.load(f)

    base_model_path = gnn_cfg["base_model"]

    llm = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype="auto",
        device_map="auto",
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
        )
        model = gnn_llm.GraphAugmentedLLM(llm, pe_model, d_model=gnn_cfg["d_model"], eps=gnn_cfg["eps"])
        gnn_weights = torch.load(os.path.join(path, "gnn_weights.pt"), map_location="cpu")
        model.pe_model.load_state_dict(gnn_weights["gt_model"])
        model.pe_proj.load_state_dict(gnn_weights["pe_proj"])
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
        )
        model = gnn_llm.GraphAugmentedLLM(llm, pe_model, d_model=gnn_cfg["d_model"], eps=gnn_cfg["eps"])
        gnn_weights = torch.load(os.path.join(path, "gnn_weights.pt"), map_location="cpu")
        model.pe_model.load_state_dict(gnn_weights["pe_model"])
        model.pe_proj.load_state_dict(gnn_weights["pe_proj"])

    model.eval()
    return model, tokenizer
