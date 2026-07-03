"""Standalone test: validates which attention implementation is used by the inner
LLM before and after GraphAugmentedLLM construction, then traces actual dispatch
during a forward pass.

Run from repo root:
    python tests/test_attention_impl.py
"""
import os
import sys, types
sys.path.insert(0, "src")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prism.models.gnn_llm import GraphAugmentedLLM
from prism.models import r_pearl

BASE_MODEL = os.environ.get("GREP_TEST_MODEL", "google/gemma-4-12B-it")


# ---------------------------------------------------------------------------
# Part 1: config-mutation check (no GPU needed, instant)
# ---------------------------------------------------------------------------
def test_config_mutation():
    print("\n" + "="*60)
    print("PART 1: config object identity / mutation check")
    print("="*60)

    # Build a minimal mock LLM so we don't load weights.
    from transformers import LlamaConfig
    cfg = LlamaConfig(
        hidden_size=64, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, intermediate_size=128,
    )
    cfg._attn_implementation = "sdpa"          # simulate normal load

    mock_llm = torch.nn.Module()
    mock_llm.config = cfg
    mock_llm.get_input_embeddings = lambda: torch.nn.Embedding(32000, 64)

    impl_before = mock_llm.config._attn_implementation
    print(f"  llm.config._attn_implementation BEFORE construction : {impl_before!r}")
    print(f"  id(llm.config)                                       : {id(mock_llm.config)}")

    # Replicate what GraphAugmentedLLM.__init__ does with the *current* code
    # (post copy.copy fix).  We just call the relevant two lines manually so
    # we don't need the full model graph.
    import copy
    wrapper_cfg = copy.copy(mock_llm.config)
    wrapper_cfg._attn_implementation = "eager"

    impl_after_inner   = mock_llm.config._attn_implementation
    impl_after_wrapper = wrapper_cfg._attn_implementation

    print(f"  llm.config._attn_implementation AFTER  construction : {impl_after_inner!r}")
    print(f"  wrapper config._attn_implementation                 : {impl_after_wrapper!r}")
    print(f"  same object? {mock_llm.config is wrapper_cfg}")

    if impl_after_inner != impl_before:
        print("\n  *** MUTATION BUG STILL PRESENT – inner config was changed! ***")
    else:
        print("\n  [OK] copy.copy isolated the wrapper config; inner LLM untouched.")

    # Also show what happens WITHOUT the fix (reference assignment)
    cfg2 = LlamaConfig(hidden_size=64, num_hidden_layers=1, num_attention_heads=2,
                       num_key_value_heads=2, intermediate_size=128)
    cfg2._attn_implementation = "sdpa"
    bad_wrapper = cfg2                   # ← reference, not copy
    bad_wrapper._attn_implementation = "eager"
    print(f"\n  WITHOUT fix: llm.config._attn_implementation = {cfg2._attn_implementation!r}  ← mutated!")


# ---------------------------------------------------------------------------
# Part 2: live attention-dispatch trace (requires the model on GPU)
# ---------------------------------------------------------------------------
def test_live_dispatch():
    # Loads the full base model — opt-in only (heavy: download + VRAM).
    if os.environ.get("GREP_HEAVY_TESTS") != "1":
        print("  [skip] test_live_dispatch: set GREP_HEAVY_TESTS=1 to run (loads the full base model)")
        return
    print("\n" + "="*60)
    print("PART 2: live attention dispatch trace  (loads real model)")
    print("="*60)

    import transformers.models.llama.modeling_llama as llama_mod
    from transformers.integrations.sdpa_attention import sdpa_attention_forward as _sdpa_fn
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except ImportError:
        ALL_ATTENTION_FUNCTIONS = getattr(llama_mod, "ALL_ATTENTION_FUNCTIONS", {})

    calls = []

    orig_eager = ALL_ATTENTION_FUNCTIONS.get("eager", llama_mod.eager_attention_forward)
    orig_sdpa  = ALL_ATTENTION_FUNCTIONS.get("sdpa",  _sdpa_fn)

    def patched_eager(module, query, key, value, attention_mask, *args, **kwargs):
        calls.append("eager")
        return orig_eager(module, query, key, value, attention_mask, *args, **kwargs)

    def patched_sdpa(module, query, key, value, attention_mask, *args, **kwargs):
        calls.append("sdpa")
        return orig_sdpa(module, query, key, value, attention_mask, *args, **kwargs)

    ALL_ATTENTION_FUNCTIONS["eager"] = patched_eager
    ALL_ATTENTION_FUNCTIONS["sdpa"]  = patched_sdpa

    try:
        print(f"  Loading {BASE_MODEL} …")
        llm = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

        print(f"  llm.config._attn_implementation (fresh load): {llm.config._attn_implementation!r}")

        pe_model = r_pearl.RandomGNNPositionalEncodings(
            pe_hidden_channels=64, pe_num_layers=2,
            d_model=llm.config.hidden_size,
            num_samples=4, dropout=0.0, k=2, use_layer_norm=True,
        )

        graph_llm = GraphAugmentedLLM(llm, pe_model, d_model=llm.config.hidden_size)

        print(f"  graph_llm.llm.config._attn_implementation    : {graph_llm.llm.config._attn_implementation!r}")
        print(f"  graph_llm.config._attn_implementation        : {graph_llm.config._attn_implementation!r}")

        # Short forward through the inner LLM directly
        input_ids = tokenizer.encode("Hello world", return_tensors="pt").to("cuda")
        print(f"\n  Running graph_llm.llm forward (seq_len={input_ids.shape[1]}) …")
        calls.clear()
        graph_llm.eval()
        with torch.no_grad():
            graph_llm.llm(input_ids=input_ids)

        inner_attn = set(calls)
        print(f"  Attention functions called on inner LLM: {inner_attn}")

        # Now run through the wrapper's generate path (the eval path)
        print(f"\n  Running graph_llm.generate forward (inputs_embeds) …")
        calls.clear()
        embeds = graph_llm.llm.get_input_embeddings()(input_ids)
        with torch.no_grad():
            # Simulate what GraphAugmentedInMemoryLLM does
            graph_llm.llm(inputs_embeds=embeds)

        wrapper_attn = set(calls)
        print(f"  Attention functions called via wrapper  : {wrapper_attn}")

        if inner_attn == {"eager"} or wrapper_attn == {"eager"}:
            print("\n  *** eager attention confirmed — this is the OOM source at long seq lengths ***")
        else:
            print("\n  [OK] no eager attention observed (SDPA active — fix is working).")

        # Simulate the PRE-FIX bug: mutate inner LLM config to eager.
        # The eager path uses the module GLOBAL `eager_attention_forward` directly,
        # NOT via ALL_ATTENTION_FUNCTIONS, so we must patch the module global.
        print("\n  --- Simulating pre-fix mutation (llm.config._attn_implementation = 'eager') ---")

        orig_eager_global = llama_mod.eager_attention_forward
        def patched_eager_global(module, query, key, value, attention_mask, *args, **kwargs):
            calls.append("eager")
            return orig_eager_global(module, query, key, value, attention_mask, *args, **kwargs)
        llama_mod.eager_attention_forward = patched_eager_global

        try:
            graph_llm.llm.config._attn_implementation = "eager"
            inner = getattr(getattr(graph_llm.llm, 'base_model', None), 'model', graph_llm.llm)
            if hasattr(inner, 'config'):
                inner.config._attn_implementation = "eager"

            calls.clear()
            with torch.no_grad():
                graph_llm.llm(input_ids=input_ids)
            post_mutation_attn = set(calls)
            print(f"  Attention functions called after mutation: {post_mutation_attn}")

            if post_mutation_attn == {"eager"}:
                print("\n  *** PRE-FIX BUG CONFIRMED: config mutation switches dispatch to eager. ***")
                print("  *** At 2048 tokens (training): 32 × 2048² × 4B ≈  0.5 GB/layer → fits.  ***")
                print("  *** At 7200 tokens (eval):     32 × 7200² × 4B ≈  6.6 GB/layer → OOM.   ***")
                print("  *** Training never OOMs because max_seq_length=2048.                      ***")
            else:
                print("\n  Unexpected: mutation didn't change dispatch.")
        finally:
            llama_mod.eager_attention_forward = orig_eager_global
            graph_llm.llm.config._attn_implementation = "sdpa"

    finally:
        ALL_ATTENTION_FUNCTIONS["eager"] = orig_eager
        ALL_ATTENTION_FUNCTIONS["sdpa"]  = orig_sdpa


if __name__ == "__main__":
    test_config_mutation()   # always runs, no GPU needed
    test_live_dispatch()     # needs GPU + model weights
