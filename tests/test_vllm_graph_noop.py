"""Ψ=0 bitwise no-op invariant for the vLLM graph engine.

Regression-locks the property verified in the demo notebook: a request carrying
an all-zero Ψ transport must generate token-for-token identically to a plain
request with no graph data at all — the injection machinery contributes exactly
nothing when Ψ is zero. Also proves the channel is LIVE (Ψ scaled up changes
the generation, dbg counters show attention consumed it).

Runs on vLLM's torch CPU backend (Mac-safe); the tiny Gemma-4 fixture keeps the
engine small.
"""
import os

# Before the FIRST vllm import anywhere in the process: platform selection
# happens at import time, and the Metal/MLX plugin cannot take torch patches.
os.environ["VLLM_PLUGINS"] = ""
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import pytest

pytest.importorskip("vllm")

import torch

from vllm_graph_helpers import (
    PROMPT, load_tokenizer, make_graph, save_fixture_dir, spin_engine,
)


@pytest.fixture(scope="module")
def engine_setup(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("gemma4_tiny")
    hf_llm = save_fixture_dir(model_dir)
    llm, wrapper = spin_engine(model_dir)
    tokenizer = load_tokenizer()
    prompt_ids = tokenizer(PROMPT, add_special_tokens=False)["input_ids"]
    yield llm, wrapper, hf_llm, tokenizer, prompt_ids
    # Each CPU engine reserves GiBs of host RAM at startup; a leftover engine
    # starves the next module's ("Available memory ... less than desired").
    import gc
    del llm, wrapper
    gc.collect()


def _gen(llm, prompt_ids, transport=None, max_tokens=32):
    from vllm import SamplingParams
    req = {"prompt_token_ids": list(prompt_ids)}
    if transport is not None:
        req["multi_modal_data"] = {"image": {"graph_embeds": transport.unsqueeze(0)}}
    sp = SamplingParams(temperature=0, max_tokens=max_tokens)
    return llm.generate([req], sp)[0].outputs[0]


def test_psi_zero_is_bitwise_noop_and_channel_is_live(engine_setup):
    llm, wrapper, hf_llm, tokenizer, prompt_ids = engine_setup

    from prism.models.vllm_graph.psi import build_psi_transport
    from vllm_graph_helpers import build_hf_graph_model

    graph_model = build_hf_graph_model(hf_llm)
    transport, _ = build_psi_transport(graph_model, tokenizer, prompt_ids, make_graph())

    out_base = _gen(llm, prompt_ids)
    dbg_base = dict(wrapper.dbg)
    out_zero = _gen(llm, prompt_ids, torch.zeros_like(transport))
    out_real = _gen(llm, prompt_ids, transport)
    out_amp = _gen(llm, prompt_ids, transport * 20)

    assert out_zero.token_ids == out_base.token_ids, "psi=0 must be a bitwise no-op"
    assert wrapper.dbg["psi_armed"] > dbg_base["psi_armed"], \
        "psi never armed — mm data not reaching the model"
    assert wrapper.dbg["attn_hit"] > dbg_base["attn_hit"], \
        "psi armed but the attention patch never consumed it"
    # Untrained Ψ at trained scale may or may not move a random-init model, but
    # amplified Ψ must — otherwise the channel is dead despite the counters.
    assert out_amp.token_ids != out_base.token_ids, \
        "psi x20 did not change the generation — channel dead"
