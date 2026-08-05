"""DL-mode verification of the eval inference path the eval module actually drives.

`evaluate.eval_model_single_graph` selects an inference client by architecture and
runs the planning loop through it. The load-bearing forward driver is
`prism.models.inference.GraphAugmentedInMemoryLLM._generate_tokens`, which dispatches
per architecture (R-PEARL / GraphMask / Composite / InjectedComposite) and runs the
base LLM's `generate` with the graph signal injected. The existing suite covers the
architecture forwards individually and the client's *output handling* with a STUBBED
model (`tests/test_inference_graph_parser.py` uses a MagicMock generate). This file
closes the remaining gap: it drives a **real** (tiny, random-init, CPU) model through
the client so the e9-configured `rpearl_llm` path and the plain `llm` path are proven
to run `generate` end-to-end without errors and with correct state hygiene.

Scope: `GraphAugmentedInMemoryLLM.{query_llm,_generate_tokens}` (R-PEARL branch +
no-graph fallback), `InMemoryLLM._generate_tokens`, `_core_graph_model` dispatch.
Deps exercised for real: `GraphAugmentedLLM` + `RandomGNNPositionalEncodings`
(build_pe_signal), `node_token_variants`, `find_last_graph_scope`,
`build_injection_map`, `scene_graph_dict_to_pyg`, the Gemma-4 tokenizer, and the
tiny Gemma-4 base `generate`. Boundary: the real 12B weights (never loaded).

DL-mode discipline: random-init weights produce garbage tokens — we assert ONLY that
the forward runs, returns the documented shape/dtype/structure, and clears its armed
graph state. We never assert the generated *content* is correct.

Run directly:  python tests/test_eval_inference_path.py
Or via pytest: pytest tests/test_eval_inference_path.py -v
"""
import sys
import os

sys.path.insert(0, "src")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import types

import torch

from prism.models.inference import InMemoryLLM, GraphAugmentedInMemoryLLM, _core_graph_model
from prism.models.gnn_llm import GraphAugmentedLLM, GraphMaskLLM
from prism.models import r_pearl as r_pearl_module
from prism.models import gt as gt_module
from prism.data import utils as data_utils


def _skip(msg):
    """Skip under pytest; print and bail when run as a plain script."""
    if __name__ != "__main__" and "pytest" in sys.modules:
        import pytest
        pytest.skip(msg)
    print(f"[SKIP] {msg}")
    return None


# Tiny scene graph used to drive a real PE forward (mirrors test_inference_graph_parser).
_SCENE = {
    "objects": [{"name": "house_1", "coords": [0, 0]}],
    "regions": [{"name": "field_1", "coords": [1, 1]}],
    "object_connections": [["house_1", "field_1"]],
    "region_connections": [["field_1", "field_1"]],
    "robot_location": "field_1",
}
_HID = 32  # tiny LLM hidden size == d_model for Ψ injection


def _tokenizer():
    """Real Gemma-4 tokenizer (offline). Needed so node-name tokens really match the
    prompt ids that node_token_variants / build_injection_map look for."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("google/gemma-4-12B-it")


def _tiny_gemma(vocab):
    """Random-init Gemma-4 (the e8/e9 base architecture) shrunk to CPU scale. `vocab`
    matches the real tokenizer so its ids are in range for a real generate."""
    from transformers import Gemma4UnifiedForCausalLM, Gemma4UnifiedTextConfig
    torch.manual_seed(0)
    cfg = Gemma4UnifiedTextConfig(
        vocab_size=vocab, hidden_size=_HID, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=128, attn_implementation="eager")
    return Gemma4UnifiedForCausalLM(cfg)


def _rpearl_model(llm):
    """The exact `rpearl_llm` wrapper architectures.build_planner_model constructs:
    RandomGNNPositionalEncodings PE feeding GraphAugmentedLLM (d_model == hidden)."""
    pe = r_pearl_module.RandomGNNPositionalEncodings(
        pe_hidden_channels=16, pe_num_layers=2, d_model=_HID, num_samples=4,
        dropout=0.0, k=3, eps=1e-8, use_layer_norm=True, node_feature_dim=None)
    return GraphAugmentedLLM(llm, pe, d_model=_HID, eps=1e-8)


def _rpearl_gt_model(llm):
    """`rpearl_gt_llm`: a GraphTransformer PE (R-PEARL inside GT attention blocks)
    feeding the same GraphAugmentedLLM wrapper — so it shares the build_pe_signal
    dispatch branch but exercises the GT forward instead of the plain GCN."""
    pe = gt_module.GraphTransformer(
        num_layers=2, pe_hidden_channels=16, pe_num_layers=2, d_model=_HID,
        heads=4, num_samples=4, dropout=0.0, k_pe=3, k_gt=3, eps=1e-8,
        use_layer_norm=True, node_feature_dim=None)
    return GraphAugmentedLLM(llm, pe, d_model=_HID, eps=1e-8)


def _graph_mask_model(llm):
    """`graph_mask_llm`: parameter-free structural attention mask — node tokens attend
    only along graph edges. Distinct dispatch branch (_struct_bias) in _generate_tokens."""
    return GraphMaskLLM(llm, k_hops=1, symmetrize=True, use_edges=True)


def _make_client(cls, model, tok, **kw):
    """Build a client via real __init__ (reads device from model params)."""
    return cls(model, tok, include_edges=False, include_tools=False,
               icl_examples=0, **kw)


def _prompt_ids(tok):
    """Tokenize a string carrying the scene node names so injection has real targets."""
    enc = tok("Scene graph: field_1 house_1 robot at field_1", return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


# ---------------------------------------------------------------------------
# Headline gate: e9-configured rpearl_llm runs a REAL PE-injected generate
# ---------------------------------------------------------------------------

def test_rpearl_generate_tokens_real_forward():
    """R-PEARL branch: build_pe_signal (real GNN) + Gemma generate with Ψ armed must
    run without error, return newly-generated token ids (prompt prefix stripped) as a
    LongTensor, and CLEAR `_pe_signal` afterward (the finally-block contract)."""
    try:
        tok = _tokenizer()
        model = _rpearl_model(_tiny_gemma(tok.vocab_size))
    except Exception as e:  # noqa: BLE001 — optional post-cutoff dep / offline weights
        return _skip(f"gemma4_unified/tokenizer unavailable: {e}")
    model.eval()
    client = _make_client(GraphAugmentedInMemoryLLM, model, tok)
    pyg = data_utils.scene_graph_dict_to_pyg(_SCENE)
    input_ids, attn = _prompt_ids(tok)

    out = client._generate_tokens(input_ids, attn, [pyg], max_new_tokens=4)

    assert isinstance(out, torch.Tensor) and out.dtype == torch.long
    assert out.dim() == 2 and out.shape[0] == 1
    assert 0 <= out.shape[1] <= 4               # prefix stripped to new tokens only
    assert model._pe_signal is None, "_pe_signal must be disarmed after generate"


def test_rpearl_no_graph_fallback_real_forward():
    """No parseable graph -> the branch falls back to plain base-LLM generate. Must
    run, strip the prefix, and never arm `_pe_signal`."""
    try:
        tok = _tokenizer()
        model = _rpearl_model(_tiny_gemma(tok.vocab_size))
    except Exception as e:  # noqa: BLE001
        return _skip(f"gemma4_unified/tokenizer unavailable: {e}")
    model.eval()
    client = _make_client(GraphAugmentedInMemoryLLM, model, tok)
    input_ids, attn = _prompt_ids(tok)

    out = client._generate_tokens(input_ids, attn, [], max_new_tokens=4)

    assert isinstance(out, torch.Tensor) and out.dtype == torch.long
    assert out.dim() == 2 and 0 <= out.shape[1] <= 4
    assert model._pe_signal is None


def test_rpearl_gt_generate_tokens_real_forward():
    """rpearl_gt_llm (GraphTransformer PE on Gemma-4): shares the build_pe_signal
    branch but runs the GT forward. Must generate without error, strip the prefix,
    and disarm `_pe_signal`."""
    try:
        tok = _tokenizer()
        model = _rpearl_gt_model(_tiny_gemma(tok.vocab_size))
    except Exception as e:  # noqa: BLE001
        return _skip(f"gemma4_unified/tokenizer unavailable: {e}")
    model.eval()
    client = _make_client(GraphAugmentedInMemoryLLM, model, tok)
    pyg = data_utils.scene_graph_dict_to_pyg(_SCENE)
    input_ids, attn = _prompt_ids(tok)

    out = client._generate_tokens(input_ids, attn, [pyg], max_new_tokens=4)

    assert isinstance(out, torch.Tensor) and out.dtype == torch.long
    assert out.dim() == 2 and out.shape[0] == 1 and 0 <= out.shape[1] <= 4
    assert model._pe_signal is None, "_pe_signal must be disarmed after generate"


def test_graph_mask_generate_tokens_real_forward():
    """graph_mask_llm (GraphMaskLLM on Gemma-4): the _struct_bias branch must build the
    structural mask, run a real masked generate, strip the prefix, and clear
    `_struct_bias` in the finally block."""
    try:
        tok = _tokenizer()
        model = _graph_mask_model(_tiny_gemma(tok.vocab_size))
    except Exception as e:  # noqa: BLE001
        return _skip(f"gemma4_unified/tokenizer unavailable: {e}")
    model.eval()
    client = _make_client(GraphAugmentedInMemoryLLM, model, tok)
    pyg = data_utils.scene_graph_dict_to_pyg(_SCENE)
    input_ids, attn = _prompt_ids(tok)

    out = client._generate_tokens(input_ids, attn, [pyg], max_new_tokens=4)

    assert isinstance(out, torch.Tensor) and out.dtype == torch.long
    assert out.dim() == 2 and out.shape[0] == 1 and 0 <= out.shape[1] <= 4
    assert model._struct_bias is None, "_struct_bias must be cleared after generate"


def test_rpearl_query_llm_end_to_end_returns_spine_tuple():
    """Full client entry point: parse graph -> compact prompt -> chat template ->
    PE-injected generate -> decode -> inverse-translate. Must return the documented
    (planner_response, True) 2-tuple and leave `_pe_signal` cleared. Content unchecked
    (random weights)."""
    try:
        tok = _tokenizer()
        model = _rpearl_model(_tiny_gemma(tok.vocab_size))
    except Exception as e:  # noqa: BLE001
        return _skip(f"gemma4_unified/tokenizer unavailable: {e}")
    model.eval()
    client = _make_client(GraphAugmentedInMemoryLLM, model, tok)
    msg = [
        {"role": "system", "content": "You are a planner."},
        {"role": "user", "content": f"Task: go to the house.\nScene graph:{_SCENE}"},
    ]

    result = client.query_llm(msg, max_new_tokens=4)

    assert isinstance(result, tuple) and len(result) == 2
    planner_response, ok = result
    assert ok is True
    assert planner_response is None or isinstance(planner_response, (dict, str))
    assert model._pe_signal is None


# ---------------------------------------------------------------------------
# Plain-LLM gate (architecture: llm)
# ---------------------------------------------------------------------------

def test_plain_inmemory_generate_tokens_real_forward():
    """InMemoryLLM drives an unwrapped base LLM: generate runs, prefix stripped."""
    try:
        tok = _tokenizer()
        model = _tiny_gemma(tok.vocab_size).eval()
    except Exception as e:  # noqa: BLE001
        return _skip(f"gemma4_unified/tokenizer unavailable: {e}")
    client = _make_client(InMemoryLLM, model, tok)
    input_ids, attn = _prompt_ids(tok)

    out = client._generate_tokens(input_ids, attn, msg=None, max_new_tokens=4)

    assert isinstance(out, torch.Tensor) and out.dtype == torch.long
    assert out.dim() == 2 and 0 <= out.shape[1] <= 4


# ---------------------------------------------------------------------------
# Dispatch: PEFT-shell unwrap reaches the graph core (deterministic, all archs)
# ---------------------------------------------------------------------------

def test_core_graph_model_unwraps_peft_shell():
    """`_core_graph_model` must peel a PEFT-style wrapper (`.base_model.model`) to the
    real graph core so the per-architecture branch in _generate_tokens fires, and
    return an already-bare graph core unchanged. (It is only ever called on
    graph-augmented models — its descent on a plain CausalLM is a don't-care path,
    so not asserted.)"""
    try:
        core = _rpearl_model(_tiny_gemma(64))   # 64-vocab ok: no forward here
    except Exception as e:  # noqa: BLE001
        return _skip(f"gemma4_unified unavailable: {e}")
    peft_shell = types.SimpleNamespace(base_model=types.SimpleNamespace(model=core))

    assert _core_graph_model(peft_shell) is core
    assert _core_graph_model(core) is core


# ---------------------------------------------------------------------------
# wire_llm: the ROTATION branch of _generate_tokens (cached-key re-rotation)
# ---------------------------------------------------------------------------

def _wire_model(llm):
    """`wire_llm`: a GraphTransformer Ψ producer feeding WireGraphLLM. Ψ enters as a
    ROTATION of q/k inside attention, so this is its own dispatch branch in
    _generate_tokens — the one that used to raise NotImplementedError."""
    from prism.models.gnn_llm import WireGraphLLM
    pe = gt_module.GraphTransformer(
        num_layers=2, pe_hidden_channels=16, pe_num_layers=2, d_model=_HID,
        heads=4, num_samples=4, dropout=0.0, k_pe=3, k_gt=3, eps=1e-8,
        use_layer_norm=True, node_feature_dim=None)
    return WireGraphLLM(llm, pe, d_model=_HID, layer_scope="dense",
                        sigma_init=0.05, max_angle=8.0)


def test_wire_generate_tokens_real_forward():
    """wire_llm through the REAL eval client: prefill arms the signal, every cached
    decode step re-rotates the prompt keys, tokens come back, and both the signal and
    the decode-angle cache are cleared afterwards.

    This is the eval path `scalability_evaluation` drives; it previously raised
    NotImplementedError here, turning every sample into an error record.
    """
    try:
        tok = _tokenizer()
        model = _wire_model(_tiny_gemma(tok.vocab_size))
    except Exception as e:  # noqa: BLE001
        return _skip(f"gemma4_unified/tokenizer unavailable: {e}")
    model.eval()
    client = _make_client(GraphAugmentedInMemoryLLM, model, tok)
    pyg = data_utils.scene_graph_dict_to_pyg(_SCENE)
    input_ids, attn = _prompt_ids(tok)

    out = client._generate_tokens(input_ids, attn, [pyg], max_new_tokens=4)

    assert isinstance(out, torch.Tensor) and out.dtype == torch.long
    assert out.dim() == 2 and out.shape[0] == 1 and 0 < out.shape[1] <= 4
    assert model._wire_signal is None, "_wire_signal must be disarmed after generate"
    assert not model._wire_decode_cos_sin, "decode angle cache must be cleared"


def _learnable_mask_model(llm):
    """`learnable_graph_mask`: a GraphTransformer Ψ producer feeding the LEARNED mask.
    Shares the `_struct_bias` dispatch branch with GraphMaskLLM, but the bias depends on
    Ψ — which is why `--permutation-seed` has to reach it."""
    from prism.models.gnn_llm import LearnableGraphMaskLLM
    pe = gt_module.GraphTransformer(
        num_layers=2, pe_hidden_channels=16, pe_num_layers=2, d_model=_HID,
        heads=4, num_samples=4, dropout=0.0, k_pe=3, k_gt=3, eps=1e-8,
        use_layer_norm=True, node_feature_dim=None)
    return LearnableGraphMaskLLM(llm, pe, alpha=0.7, layer_scope="dense")


def test_learnable_mask_client_threads_the_permutation():
    """`--permutation-seed` must reach `build_structural_mask`.

    REGRESSION GUARD: the client used to call build_structural_mask with no permutation
    argument at all, so every permuted-transferability number for this architecture was
    produced on the UNPERMUTED graph while the report said otherwise. The spy asserts the
    exact object the client was constructed with arrives at the mask builder.
    """
    try:
        tok = _tokenizer()
        model = _learnable_mask_model(_tiny_gemma(tok.vocab_size))
    except Exception as e:  # noqa: BLE001
        return _skip(f"gemma4_unified/tokenizer unavailable: {e}")
    model.eval()
    from prism.models.utils import Permutation
    perm = Permutation(seed=5)
    client = _make_client(GraphAugmentedInMemoryLLM, model, tok, permutation=perm)
    pyg = data_utils.scene_graph_dict_to_pyg(_SCENE)
    input_ids, attn = _prompt_ids(tok)

    seen = {}
    real = model.build_structural_mask

    def _spy(*a, **kw):
        seen["permutation"] = kw.get("permutation", "ABSENT")
        return real(*a, **kw)

    model.build_structural_mask = _spy
    try:
        out = client._generate_tokens(input_ids, attn, [pyg], max_new_tokens=4)
    finally:
        del model.build_structural_mask
    assert seen.get("permutation") is perm, \
        f"client did not thread --permutation-seed into the mask: {seen!r}"
    assert isinstance(out, torch.Tensor) and out.dtype == torch.long
    assert model._struct_bias is None, "_struct_bias must be cleared after generate"


def test_learnable_mask_identity_rope_reaches_generate_and_positions_advance():
    """`disable_graph_token_rope` must survive the mask branch's generate call.

    Two things only a REAL generate can settle, both transformers-version sensitive:
    the prefill must receive the zeroed position_ids (not silently drop them, which
    would evaluate an identity-RoPE checkpoint with normal RoPE), and the decode steps
    must keep counting from the last prompt position rather than restarting at 0.
    """
    try:
        tok = _tokenizer()
        model = _learnable_mask_model(_tiny_gemma(tok.vocab_size))
    except Exception as e:  # noqa: BLE001
        return _skip(f"gemma4_unified/tokenizer unavailable: {e}")
    model._disable_graph_token_rope = True
    model.eval()
    client = _make_client(GraphAugmentedInMemoryLLM, model, tok)
    pyg = data_utils.scene_graph_dict_to_pyg(_SCENE)
    # Ends OUTSIDE a mention, as a chat-templated prompt does (it closes with the model
    # turn header); the shared `_prompt_ids` ends on a node name, which the guard rejects
    # — see test_identity_rope_rejects_prompt_ending_in_a_mention.
    enc = tok("Scene graph: field_1 house_1 robot at field_1. Plan:", return_tensors="pt")
    input_ids, attn = enc["input_ids"], enc["attention_mask"]

    seen = []
    real_forward = model.llm.forward

    def _spy(*a, **kw):
        seen.append(kw.get("position_ids"))
        return real_forward(*a, **kw)

    model.llm.forward = _spy
    try:
        out = client._generate_tokens(input_ids, attn, [pyg], max_new_tokens=4)
    finally:
        del model.llm.forward

    prefill = seen[0]
    assert prefill is not None, "identity-RoPE position_ids never reached generate"
    n = input_ids.shape[1]
    assert prefill.shape == (1, n)
    zeroed = (prefill[0] == 0).nonzero().flatten().tolist()
    assert zeroed != [0], "no node span was zeroed — the prompt mentions no graph node"
    # Decode steps continue from the prompt, so RoPE does not restart mid-rollout.
    steps = [p for p in seen[1:] if p is not None]
    assert steps, "decode steps received no position_ids"
    assert int(steps[0][0, -1]) == n, (int(steps[0][0, -1]), n)
    assert isinstance(out, torch.Tensor) and out.dtype == torch.long
    assert model._struct_bias is None


def test_identity_rope_rejects_prompt_ending_in_a_mention():
    """The precondition behind the test above, asserted rather than assumed.

    transformers numbers generated tokens from `position_ids[..., -1:]`, so a prompt
    whose LAST token is inside a zeroed node span would restart the rollout at position
    1 — every generated token silently mis-positioned. That must fail loudly, not
    produce a plausible-looking rollout. (`_prompt_ids` ends on a node name, which is
    exactly the offending shape.)
    """
    try:
        tok = _tokenizer()
        model = _learnable_mask_model(_tiny_gemma(tok.vocab_size))
    except Exception as e:  # noqa: BLE001
        return _skip(f"gemma4_unified/tokenizer unavailable: {e}")
    model._disable_graph_token_rope = True
    model.eval()
    client = _make_client(GraphAugmentedInMemoryLLM, model, tok)
    pyg = data_utils.scene_graph_dict_to_pyg(_SCENE)
    input_ids, attn = _prompt_ids(tok)

    try:
        client._generate_tokens(input_ids, attn, [pyg], max_new_tokens=4)
    except RuntimeError as e:
        assert "FINAL prompt position" in str(e), e
    else:
        raise AssertionError("a prompt ending inside a node mention generated silently")


def test_wire_decode_consistent_scope_is_rejected():
    """`injection_scope='decode_consistent'` has no rotation analogue (it needs the mask
    family's q/kv split). It must fail LOUDLY at the client rather than decode
    prompt-only and silently disagree with how the checkpoint was trained."""
    try:
        tok = _tokenizer()
        model = _wire_model(_tiny_gemma(tok.vocab_size))
    except Exception as e:  # noqa: BLE001
        return _skip(f"gemma4_unified/tokenizer unavailable: {e}")
    model.eval()
    client = _make_client(GraphAugmentedInMemoryLLM, model, tok,
                          injection_scope="decode_consistent")
    pyg = data_utils.scene_graph_dict_to_pyg(_SCENE)
    input_ids, attn = _prompt_ids(tok)
    raised = ""
    try:
        client._generate_tokens(input_ids, attn, [pyg], max_new_tokens=2)
    except NotImplementedError as e:
        raised = str(e)
    assert "decode_consistent" in raised, \
        f"expected a loud rejection naming the scope, got {raised!r}"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"{name}: PASS")
    print("done")
