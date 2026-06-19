"""Parity tests for the architecture-agnostic Ψ-injection (``prism_pe``).

``GraphAugmentedLLM`` injects the graph signal Ψ by registering a custom attention
function and pointing the wrapped LLM's ``config._attn_implementation`` at it,
rather than reimplementing each model family's attention ``forward``. These tests
assert the two invariants that make that safe:

  1. **Ψ = 0 (or absent) ⇒ identical logits to the stock model.** This catches the
     causal-mask regression: a custom impl name is excluded from HF's mask builder
     unless its mask is registered (``create_causal_mask`` returns ``None``
     otherwise), which would silently disable causal masking.
  2. **Ψ ≠ 0 ⇒ the logits change.** The injection actually does something.

Llama runs everywhere. gemma-4 (``gemma4_unified``) runs only where transformers
ships it (≥5.5) and additionally exercises q/k-norm, sliding-window and KV-shared
layers — the differences that broke the old Llama-specific patch.

Run directly:  ``python tests/test_pe_injection_parity.py``
Or via pytest: ``pytest tests/test_pe_injection_parity.py``
"""
import sys
sys.path.insert(0, "src")

import torch
from torch import nn

from prism.models.gnn_llm import GraphAugmentedLLM

_TOL = 1e-5


def _check_parity(llm, hidden, seq=7, batch=2, vocab=64):
    """Build a wrapper around ``llm`` and assert the two injection invariants."""
    llm = llm.eval()
    ids = torch.randint(0, vocab, (batch, seq))
    with torch.no_grad():
        ref = llm(input_ids=ids).logits

    # Trivial pe_model: never invoked here (we drive _pe_signal directly), it only
    # needs to be an nn.Module so the wrapper can place/own it.
    wrap = GraphAugmentedLLM(llm, nn.Linear(8, 8), d_model=8, eps=1e-8).eval()

    def logits():
        with torch.no_grad():
            return wrap.llm(input_ids=ids).logits

    impl = next(iter(wrap._decoder_layers())).self_attn.config._attn_implementation
    assert impl == "prism_pe", f"expected prism_pe dispatch, got {impl!r}"

    wrap._pe_signal = None
    d_none = (logits() - ref).abs().max().item()
    wrap._pe_signal = torch.zeros(batch, seq, hidden)
    d_zero = (logits() - ref).abs().max().item()
    wrap._pe_signal = torch.randn(batch, seq, hidden) * 0.5
    d_nz = (logits() - ref).abs().max().item()

    assert d_none < _TOL, f"Ψ=None changed logits (max|Δ|={d_none:.2e}) — masking regressed?"
    assert d_zero < _TOL, f"Ψ=0 changed logits (max|Δ|={d_zero:.2e})"
    assert d_nz > 1e-3, f"Ψ≠0 did not change logits (max|Δ|={d_nz:.2e}) — injection inert?"
    return d_none, d_zero, d_nz


def test_llama_injection_parity():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=64,
                      attn_implementation="eager")
    _check_parity(LlamaForCausalLM(cfg), hidden=32)


def test_injection_rejects_biased_attention():
    """The 'only graph tokens modified' invariant needs bias-free projections
    (Ψ=0 ⇒ W·Ψ=0). Constructing the wrapper on a biased model must fail loud."""
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=1, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=64,
                      attention_bias=True, attn_implementation="eager")
    raised = False
    try:
        GraphAugmentedLLM(LlamaForCausalLM(cfg), nn.Linear(8, 8), d_model=8)
    except ValueError as e:
        raised = "bias" in str(e)
    assert raised, "expected a bias-free-projection ValueError, none raised"


def _skip(msg):
    """Skip under pytest; print and bail when run as a plain script."""
    if __name__ != "__main__" and "pytest" in sys.modules:
        import pytest
        pytest.skip(msg)
    print(f"[SKIP] {msg}")


def test_gemma4_unified_injection_parity():
    """Same invariants on gemma-4 (q/k-norm, single-tensor RoPE, sliding, KV-share).

    Skips where transformers doesn't ship ``gemma4_unified`` (e.g. 5.0.x)."""
    try:
        from transformers import Gemma4UnifiedForCausalLM, Gemma4UnifiedTextConfig
    except Exception as e:  # noqa: BLE001 — any import failure ⇒ unsupported here
        return _skip(f"gemma4_unified unavailable: {e}")
    torch.manual_seed(0)
    cfg = Gemma4UnifiedTextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=64, attn_implementation="eager",
    )
    _check_parity(Gemma4UnifiedForCausalLM(cfg), hidden=cfg.hidden_size)


if __name__ == "__main__":
    print("Llama  :", "PASS", test_llama_injection_parity() or "")
    test_injection_rejects_biased_attention(); print("bias-guard : PASS")
    test_gemma4_unified_injection_parity()
    print("done")
