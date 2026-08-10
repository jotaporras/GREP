"""Shared setup for the vLLM graph-engine tests (noop + parity).

Env pinning MUST happen before the first ``import vllm`` anywhere in the
process: torch CPU backend (the Metal/MLX plugin cannot take torch patches) and
an in-process engine (runtime model registration).
"""
import os

os.environ["VLLM_PLUGINS"] = ""
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch
from torch_geometric.data import Data

from gemma4_fixture import BASE_MODEL, gemma4_31b_shaped

NODE_NAMES = ["kitchen", "hallway", "garage", "bedroom", "office"]
EDGES = [(0, 1), (1, 2), (1, 3), (3, 4)]

PROMPT = ("Scene graph nodes: kitchen, hallway, garage, bedroom, office. "
          "You are in the kitchen. Give the shortest route to the office.")


def make_graph() -> Data:
    edge_index = torch.tensor(EDGES + [(b, a) for a, b in EDGES]).T
    g = Data(x=torch.zeros(len(NODE_NAMES), 1), edge_index=edge_index,
             num_nodes=len(NODE_NAMES))
    g.node_names = list(NODE_NAMES)
    return g


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)


def save_fixture_dir(path, seed: int = 0, num_kv_shared_layers: int = 0):
    """Materialize the tiny fixture as an HF-format dir vLLM can serve.

    Head dims are raised from the structural mirror's 16/32 to 32/64 — vLLM's
    CPU attention kernels reject head_dim=16 ("Unsupported CPU attention
    configuration"); the 2x sliding-vs-global ratio is preserved.
    """
    llm = gemma4_31b_shaped(
        seed=seed, num_kv_shared_layers=num_kv_shared_layers,
        head_dim=32, global_head_dim=64,
    )
    llm.save_pretrained(path, safe_serialization=True)
    load_tokenizer().save_pretrained(path)
    return llm


def build_hf_graph_model(llm, disable_graph_token_rope=False, seed: int = 0):
    """GraphAugmentedLLM over the given HF fixture — the Ψ producer for both sides."""
    from prism.models.gnn_llm import GraphAugmentedLLM
    from prism.models.r_pearl import RandomGNNPositionalEncodings
    torch.manual_seed(seed)
    # fixed_seed_mode: deterministic probes, so Ψ is reproducible across calls.
    pe_model = RandomGNNPositionalEncodings(
        pe_hidden_channels=32, pe_num_layers=2, d_model=16, num_samples=8,
        fixed_seed_mode=True,
    )
    return GraphAugmentedLLM(
        llm, pe_model, d_model=16, pe_gain_init=1.0,
        disable_graph_token_rope=disable_graph_token_rope,
    ).eval()  # eval semantics: dropout off, as in the eval/rollout path


def spin_engine(model_dir, *, identity_rope=False, pe_inject_value=True):
    from prism.models.vllm_graph.engine import build_graph_llm
    return build_graph_llm(
        str(model_dir),
        identity_rope=identity_rope,
        pe_inject_value=pe_inject_value,
        dtype="float32",
        max_model_len=512,
        gpu_memory_utilization=0.15,  # CPU backend: fraction of RAM to reserve
        max_num_seqs=8,              # keep the profiling pass tiny
    )
