"""Shared tiny random-init Gemma-4 fixture mirroring the 31B text config.

Extracted from ``tests/test_wire_mps_smoke.py`` so the vLLM graph-engine tests
(`test_vllm_graph_*.py`) and the MPS smoke test build the SAME ~29M-param
structural mirror instead of drifting copies. The real ``google/gemma-4-31B-it``
weights are never needed — config + tokenizer only.
"""
import torch

BASE_MODEL = "google/gemma-4-31B-it"  # config + tokenizer only, no weights


def gemma4_31b_shaped(seed: int = 0, dtype=torch.float32, num_kv_shared_layers: int = 0,
                      **config_overrides):
    """Tiny random-init ``Gemma4ForCausalLM`` mirroring the 31B text config.

    Same idiom as ``tests/test_wire_smoke.py::_gemma4`` and
    ``tests/test_pe_injection_parity.py`` — a ``Gemma4TextConfig`` built by hand — only
    scaled to keep the 31B structure instead of the minimum that constructs.

    ``num_kv_shared_layers`` defaults to 0 (the 31B value); the vLLM parity test
    raises it to exercise the KV-shared injection skip. ``config_overrides``
    patch individual config fields (the vLLM tests need head dims the CPU
    attention kernels support).
    """
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig
    torch.manual_seed(seed)
    cfg = Gemma4TextConfig(
        vocab_size=262_144,                # real tokenizer
        hidden_size=96,
        intermediate_size=384,             # 4x hidden, as 21504 = 4 x 5376
        num_hidden_layers=24,              # 4 globals under the default 1:6 pattern
        num_attention_heads=8,
        num_key_value_heads=4,             # GQA 2x, as 32/16
        num_global_key_value_heads=1,      # GQA 8x, as 32/4
        head_dim=16,                       # sliding
        global_head_dim=32,                # global: 2x sliding, as 512 vs 256
        attention_k_eq_v=True,             # => v_proj is None on global layers
        num_kv_shared_layers=num_kv_shared_layers,
        max_position_embeddings=4096,
        sliding_window=64,
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=262_144,
        final_logit_softcapping=30.0,
        tie_word_embeddings=True,
        rms_norm_eps=1e-6,
        attention_bias=False,
        attn_implementation="eager",
        # rope_parameters left to the class default, which IS the 31B value:
        #   full_attention  -> proportional, partial_rotary_factor 0.25, theta 1e6
        #   sliding_attention -> default, theta 1e4
    )
    for k, v in config_overrides.items():
        setattr(cfg, k, v)
    return Gemma4ForCausalLM(cfg).to(dtype)
