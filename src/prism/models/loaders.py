import json
import os
from typing import Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizer

from prism.models import gnn_llm
from prism.models import r_pearl


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
        model = PeftModel.from_pretrained(model, path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype="auto",
            device_map="auto",
            quantization_config=_bnb_config(load_in_4bit),
        )
    tokenizer = AutoTokenizer.from_pretrained(path)
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
        llm = PeftModel.from_pretrained(llm, path)

    tokenizer = AutoTokenizer.from_pretrained(path)

    r_pearl_model = r_pearl.RandomGNNPositionalEncodings(
        pe_hidden_channels=gnn_cfg["pe_hidden_channels"],
        pe_num_layers=gnn_cfg["pe_num_layers"],
        d_model=gnn_cfg["d_model"],
        num_samples=gnn_cfg["num_samples"],
        dropout=gnn_cfg["dropout"],
        k=gnn_cfg["k"],
        use_layer_norm=gnn_cfg["use_layer_norm"],
    )

    model = gnn_llm.GraphAugmentedLLM(llm, r_pearl_model, tokenizer, pe_dim=gnn_cfg["d_model"])

    gnn_weights = torch.load(os.path.join(path, "gnn_weights.pt"), map_location="cpu")
    model.pe_model.load_state_dict(gnn_weights["pe_model"])
    model.pe_proj.load_state_dict(gnn_weights["pe_proj"])

    return model, tokenizer
