"""Diagnostic: check if PEFT/SFTTrainer silently freezes GNN PE weights.

Reproduces the train_v2 model setup with freeze_llm=False and checks
requires_grad on pe_model / pe_proj parameters after PEFT wrapping.
"""
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from prism.models import r_pearl, gnn_llm


def check():
    print("=" * 60)
    print("Loading small LLM for diagnostic...")
    print("=" * 60)

    # Use a tiny model to keep it fast
    llm = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu",
    )

    r_pearl_model = r_pearl.RandomGNNPositionalEncodings(
        pe_hidden_channels=256,
        pe_num_layers=3,
        d_model=896,  # Qwen2.5-0.5B hidden size
        num_samples=40,
        dropout=0.1,
        k=3,
        use_layer_norm=True,
    )

    model = gnn_llm.GraphAugmentedLLM(llm, r_pearl_model, pe_dim=896)

    # --- BEFORE PEFT ---
    print("\n--- BEFORE PEFT wrapping ---")
    _report(model)

    # Apply PEFT exactly as train_v2 does (via get_peft_model, which is
    # what SFTTrainer calls internally when peft_config is provided)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.2,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(model, lora_config)

    # --- AFTER PEFT ---
    print("\n--- AFTER PEFT wrapping (get_peft_model) ---")
    _report(peft_model)

    # Check if we can get gradients through a dummy forward
    print("\n--- Gradient check with dummy forward/backward ---")
    _gradient_check(peft_model)


def _report(model):
    """Print requires_grad status for pe_model, pe_proj, and a summary of LLM."""
    # pe_model
    pe_params = list(model.pe_model.named_parameters()) if hasattr(model, 'pe_model') else []
    # For PeftModel, the base model is wrapped
    if not pe_params and hasattr(model, 'base_model'):
        base = model.base_model
        if hasattr(base, 'model') and hasattr(base.model, 'pe_model'):
            pe_params = list(base.model.pe_model.named_parameters())

    pe_proj_params = list(model.pe_proj.named_parameters()) if hasattr(model, 'pe_proj') else []
    if not pe_proj_params and hasattr(model, 'base_model'):
        base = model.base_model
        if hasattr(base, 'model') and hasattr(base.model, 'pe_proj'):
            pe_proj_params = list(base.model.pe_proj.named_parameters())

    print(f"\n  pe_model parameters:")
    for name, p in pe_params:
        print(f"    {name}: requires_grad={p.requires_grad}, shape={list(p.shape)}")

    print(f"\n  pe_proj parameters:")
    for name, p in pe_proj_params:
        print(f"    {name}: requires_grad={p.requires_grad}, shape={list(p.shape)}")

    # Summary of trainable vs frozen
    all_params = list(model.parameters())
    trainable = [p for p in all_params if p.requires_grad]
    frozen = [p for p in all_params if not p.requires_grad]
    total_trainable = sum(p.numel() for p in trainable)
    total_frozen = sum(p.numel() for p in frozen)
    print(f"\n  Total: {len(all_params)} params, {len(trainable)} trainable, {len(frozen)} frozen")
    print(f"  Trainable param count: {total_trainable:,}")
    print(f"  Frozen param count: {total_frozen:,}")


def _gradient_check(model):
    """Run a tiny forward/backward and check if GNN params got gradients."""
    # Access the underlying model
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
        inner = model.base_model.model
    else:
        inner = model

    pe_model = inner.pe_model
    pe_proj = inner.pe_proj

    # Zero grads
    model.zero_grad()

    # Create a tiny dummy graph
    from torch_geometric.data import Data
    num_nodes = 5
    edge_index = torch.tensor([[0,1,1,2,2,3,3,4],[1,0,2,1,3,2,4,3]], dtype=torch.long)
    x = torch.randn(num_nodes, 1)
    graph = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)

    # Forward through PE
    pe_out = pe_proj(pe_model(graph))
    loss = pe_out.sum()
    loss.backward()

    print(f"\n  pe_model gradients:")
    for name, p in pe_model.named_parameters():
        grad_status = "HAS GRAD" if p.grad is not None else "NO GRAD (None)"
        grad_norm = p.grad.norm().item() if p.grad is not None else 0.0
        print(f"    {name}: {grad_status}, norm={grad_norm:.6f}, requires_grad={p.requires_grad}")

    print(f"\n  pe_proj gradients:")
    for name, p in pe_proj.named_parameters():
        grad_status = "HAS GRAD" if p.grad is not None else "NO GRAD (None)"
        grad_norm = p.grad.norm().item() if p.grad is not None else 0.0
        print(f"    {name}: {grad_status}, norm={grad_norm:.6f}, requires_grad={p.requires_grad}")


if __name__ == "__main__":
    check()
