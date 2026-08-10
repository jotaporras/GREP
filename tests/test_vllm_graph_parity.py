"""HF-vs-vLLM generation parity for the graph engine (tiny Gemma-4 fixture).

Both sides serve the SAME fixture weights and the SAME Ψ (built once by the HF
GraphAugmentedLLM chain, shipped to vLLM as the transport tensor), so any
divergence is the injection port itself: qk-norm placement, KV-shared skip,
``_pe_inject_value``, identity-RoPE. Greedy, float32, CPU on both sides.

THE REFEREE IS THE TRAINING-CONSISTENT FORWARD, NOT THE HF EVAL PATH. The two
differ, and the difference is a property of the HF stack, not of this port:
``Gemma4TextAttention.forward`` writes k/v into the KV cache BEFORE
``_prism_pe_attention_forward`` adds Ψ, so HF cached decode attends over Ψ-FREE
keys — while the training forward (no cache) attends over Ψ-carrying prompt
keys. vLLM's paged cache is written inside the attention op and therefore
persists Ψ, reproducing the TRAINING semantics. Verified empirically: vLLM
matches a no-cache HF greedy loop token-for-token and diverges from the cached
HF eval path from the first decode step. The cached-path relationship is pinned
separately: prefill (token 0) must agree with it exactly.
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
    PROMPT, build_hf_graph_model, load_tokenizer, make_graph, save_fixture_dir,
    spin_engine,
)

from prism.models.vllm_graph.psi import build_psi_transport

MAX_TOKENS = 24


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer()


@pytest.fixture(scope="module")
def base_setup(tmp_path_factory, tokenizer):
    model_dir = tmp_path_factory.mktemp("gemma4_tiny_parity")
    hf_llm = save_fixture_dir(model_dir)
    llm, wrapper = spin_engine(model_dir)
    prompt_ids = tokenizer(PROMPT, add_special_tokens=False)["input_ids"]
    yield hf_llm, llm, wrapper, prompt_ids
    # Each CPU engine reserves GiBs of host RAM at startup; a leftover engine
    # starves the next module's ("Available memory ... less than desired").
    import gc
    del llm, wrapper
    gc.collect()


def hf_generate_train_consistent(graph_model, prompt_ids, injection_map, psi_2d,
                                 max_tokens=MAX_TOKENS):
    """Greedy decode under TRAINING semantics: full no-cache forward each step,
    Ψ zero-extended to the grown sequence (prompt spans keep their signal, so
    every query attends Ψ-carrying prompt keys — exactly the trained forward)."""
    hidden = psi_2d.shape[-1]
    cur = list(prompt_ids)
    out = []
    with torch.no_grad():
        for _ in range(max_tokens):
            seq = torch.tensor([cur])
            psi_t = torch.zeros(1, len(cur), hidden)
            psi_t[0, :len(prompt_ids)] = psi_2d
            kwargs = {}
            if graph_model._disable_graph_token_rope:
                kwargs["position_ids"] = graph_model.graph_token_position_ids(
                    [injection_map], len(cur), seq.device)
            graph_model._pe_signal = psi_t
            try:
                logits = graph_model.llm(input_ids=seq, **kwargs).logits[0, -1]
            finally:
                graph_model._pe_signal = None
            nxt = int(logits.argmax())
            out.append(nxt)
            cur.append(nxt)
    return out


def hf_first_token_eval_path(graph_model, tokenizer, prompt_ids, injection_map,
                             psi_2d):
    """Token 0 of the stock HF EVAL decode (cached) — the prefill contract."""
    ids = torch.tensor([prompt_ids])
    emb = graph_model.llm.get_input_embeddings()(ids).clone()
    graph_model._pe_signal = psi_2d.unsqueeze(0).to(emb.dtype)
    gen_kwargs = {}
    if graph_model._disable_graph_token_rope:
        gen_kwargs["position_ids"] = graph_model.graph_token_position_ids(
            [injection_map], emb.shape[1], emb.device)
    try:
        with torch.no_grad():
            out = graph_model.llm.generate(
                inputs_embeds=emb, attention_mask=torch.ones_like(ids),
                max_new_tokens=1, do_sample=False, use_cache=True,
                pad_token_id=tokenizer.eos_token_id, **gen_kwargs,
            )
    finally:
        graph_model._pe_signal = None
    return int(out[0, 0])


def vllm_generate(llm, prompt_ids, transport, max_tokens=MAX_TOKENS):
    from vllm import SamplingParams
    req = {
        "prompt_token_ids": list(prompt_ids),
        "multi_modal_data": {"image": {"graph_embeds": transport.unsqueeze(0)}},
    }
    sp = SamplingParams(temperature=0, max_tokens=max_tokens)
    return list(llm.generate([req], sp)[0].outputs[0].token_ids)


def _run_case(hf_llm, llm, wrapper, tokenizer, prompt_ids, *,
              disable_graph_token_rope=False, pe_inject_value=True, psi_scale=1.0):
    graph = make_graph()
    graph_model = build_hf_graph_model(
        hf_llm, disable_graph_token_rope=disable_graph_token_rope)
    graph_model._pe_inject_value = pe_inject_value
    transport, imap = build_psi_transport(graph_model, tokenizer, prompt_ids, graph)
    hidden = hf_llm.config.hidden_size
    psi_2d = transport[:, :hidden] * psi_scale
    transport = torch.cat([psi_2d, transport[:, hidden:]], dim=-1)

    wrapper._identity_rope = disable_graph_token_rope
    wrapper._pe_inject_value = pe_inject_value
    try:
        got = vllm_generate(llm, prompt_ids, transport)
    finally:
        wrapper._identity_rope = False
        wrapper._pe_inject_value = True

    want = hf_generate_train_consistent(graph_model, prompt_ids, imap, psi_2d)
    n = min(len(want), len(got))
    assert got[:n] == want[:n], (
        f"HF/vLLM diverge at token {next(i for i in range(n) if got[i] != want[i])}: "
        f"hf={want[:n]} vllm={got[:n]}")

    # Prefill contract: first generated token must also match the stock HF EVAL
    # path (cached decode) — prefill injection is identical in both stacks.
    first_eval = hf_first_token_eval_path(
        graph_model, tokenizer, prompt_ids, imap, psi_2d)
    assert got[0] == first_eval, (
        f"prefill diverges from the HF eval path: vllm={got[0]} hf_eval={first_eval}")


def test_parity_psi_amplified(base_setup, tokenizer):
    """Ψ large enough to steer a random-init model — the strongest parity probe:
    both stacks must be steered IDENTICALLY."""
    hf_llm, llm, wrapper, prompt_ids = base_setup
    _run_case(hf_llm, llm, wrapper, tokenizer, prompt_ids, psi_scale=20.0)


def test_parity_psi_trained_scale(base_setup, tokenizer):
    hf_llm, llm, wrapper, prompt_ids = base_setup
    _run_case(hf_llm, llm, wrapper, tokenizer, prompt_ids, psi_scale=1.0)


def test_parity_identity_rope(base_setup, tokenizer):
    hf_llm, llm, wrapper, prompt_ids = base_setup
    _run_case(hf_llm, llm, wrapper, tokenizer, prompt_ids,
              disable_graph_token_rope=True, psi_scale=20.0)


def test_parity_inject_value_off(base_setup, tokenizer):
    hf_llm, llm, wrapper, prompt_ids = base_setup
    _run_case(hf_llm, llm, wrapper, tokenizer, prompt_ids,
              pe_inject_value=False, psi_scale=20.0)


def test_kv_shared_layers_are_refused(tmp_path_factory, tokenizer):
    """KV-shared configs must fail loud at engine build: HF captures
    shared_kv_states pre-Ψ (shared layers attend Ψ-free keys) while vLLM's
    paged cache persists the source layer's Ψ-carrying k/v — no port can honor
    both, and the 31B target has num_kv_shared_layers=0. Verified empirically:
    a ported kv-shared engine diverges from the HF reference at the first
    global-layer interaction."""
    model_dir = tmp_path_factory.mktemp("gemma4_tiny_kvshared")
    save_fixture_dir(model_dir, num_kv_shared_layers=6)
    with pytest.raises(Exception, match="KV-shared"):
        spin_engine(model_dir)
