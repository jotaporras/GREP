"""Contract tests for prism.models.inference.InMemoryLLM (plain, non-graph client).

CS-only verification: InMemoryLLM does no model architecture itself — it translates a
SPINE message list to the compact prompt, calls ``model.generate`` as an opaque function,
strips the prompt prefix, decodes, and inverse-translates the compact text back to SPINE
JSON. These tests pin the deterministic plumbing and the harness's handling of the model's
(stubbed, deterministic) output:

  * __init__         — device is read from the model's own parameters, not SPINE's "cuda".
  * _decode          — first sequence only, stripped; decode flags plumbed correctly.
  * _generate_tokens — returned tensor is exactly the newly generated suffix; decode kwargs
                       (do_sample=False, use_cache=True, pad_token_id=eos) are plumbed.
  * query_llm        — returns (spine_json_str, True); forward translation drops the SPINE
                       system prompt and hoists the scene graph; include_edges threads
                       through; inverse translation yields the 4 SPINE keys with the plan
                       re-wrapped as [answer(...)].

The oracle is the documented compact-prompt contract, computed by hand here — never by
calling the function under test. The model + tokenizer are the boundary: model.generate is
exercised only as a black box returning a fixed tensor (handling is what's asserted, never
output accuracy).
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

import json

import torch

from prism.models.inference import InMemoryLLM, DECODE_KWARGS
from prism.data import compact_prompt


# ---------------------------------------------------------------------------
# Fixtures / stubs (boundaries: tokenizer + model)
# ---------------------------------------------------------------------------

# A real SPINE-style scene graph and message list. region_connections is non-trivial so the
# include_edges branch has something to emit. The verbose system turn must be DROPPED by the
# forward translator; the graph-bearing user turn must be hoisted to a compact system block.
_GRAPH = {
    "regions": [{"name": "a", "coords": [0, 0]}, {"name": "b", "coords": [1, 1]}],
    "objects": [{"name": "obj_1", "coords": [2, 2]}],
    "region_connections": [["a", "b"]],
    "object_connections": [["obj_1", "a"]],
    "robot_location": "a",
}
_SPINE_MSG = [
    {"role": "system", "content": "VERBOSE SPINE SYSTEM PROMPT — must be dropped"},
    {"role": "user", "content": f"task: navigate to b\nScene graph: {_GRAPH}"},
]


class _StubTokenizer:
    """Minimal tokenizer boundary: records what the harness sends it, returns fixed tensors.

    ``apply_chat_template`` mirrors transformers>=5 (return_dict defaults True), so it returns
    a mapping with ``input_ids``/``attention_mask`` — exactly what query_llm subscripts.
    ``batch_decode`` returns a caller-supplied string so the inverse-translation path runs on
    known text.
    """

    eos_token_id = 7

    def __init__(self, decoded: str = ""):
        self._decoded = decoded
        self.chat_messages = None
        self.chat_kwargs = None
        self.decode_kwargs = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_tensors):
        self.chat_messages = messages
        self.chat_kwargs = dict(
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            return_tensors=return_tensors,
        )
        ids = torch.tensor([[1, 2, 3, 4]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def batch_decode(self, outputs, skip_special_tokens, clean_up_tokenization_spaces):
        self.decode_kwargs = dict(
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
        return [self._decoded]


class _StubModel:
    """Opaque generate() boundary: records kwargs, returns a fixed [1, T] token tensor."""

    def __init__(self, out: torch.Tensor):
        self._out = out
        self.gen_kwargs = None

    def generate(self, **kwargs):
        self.gen_kwargs = kwargs
        return self._out


def _client(decoded: str = "", model_out: torch.Tensor | None = None,
            include_edges: bool = False) -> InMemoryLLM:
    """InMemoryLLM with __init__ bypassed (like tests/test_inference_graph_parser.py),
    wired to the stub tokenizer/model so only the plumbing is exercised."""
    llm = InMemoryLLM.__new__(InMemoryLLM)
    llm.tokenizer = _StubTokenizer(decoded)
    llm.model = _StubModel(model_out if model_out is not None else torch.zeros(1, 1, dtype=torch.long))
    llm.device = "cpu"
    llm.include_edges = include_edges
    return llm


# ---------------------------------------------------------------------------
# __init__ — device extraction (deterministic plumbing)
# ---------------------------------------------------------------------------

def test_init_reads_device_from_model_params():
    """Device must come from the model's own parameters, overriding SPINE's "cuda" default;
    model/tokenizer/include_edges are stored on the instance."""
    model = torch.nn.Linear(2, 2)  # CPU param holder, not under test — just a device source
    tok = _StubTokenizer()
    llm = InMemoryLLM(model, tok, include_edges=True)

    assert llm.device == next(model.parameters()).device
    assert llm.device.type == "cpu"
    assert llm.model is model
    assert llm.tokenizer is tok
    assert llm.include_edges is True


# ---------------------------------------------------------------------------
# _decode — first-sequence-only, stripped, with the right decode flags
# ---------------------------------------------------------------------------

def test_decode_returns_first_sequence_stripped():
    """_decode returns batch_decode(...)[0].strip() — first row only, surrounding ws removed."""
    llm = _client()
    llm.tokenizer = _StubTokenizer()

    class _Tok(_StubTokenizer):
        def batch_decode(self, outputs, skip_special_tokens, clean_up_tokenization_spaces):
            self.decode_kwargs = dict(
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )
            return ["  hello world \n", "SHOULD-BE-IGNORED"]

    llm.tokenizer = _Tok()
    out = llm._decode(torch.zeros(2, 3, dtype=torch.long))

    assert out == "hello world"
    # clean_up_tokenization_spaces=False is load-bearing (BPE plan text); skip_special_tokens=True.
    assert llm.tokenizer.decode_kwargs == {
        "skip_special_tokens": True,
        "clean_up_tokenization_spaces": False,
    }


# ---------------------------------------------------------------------------
# _generate_tokens — black-box generate(): prefix stripping + kwarg plumbing
# ---------------------------------------------------------------------------

def test_generate_tokens_strips_prompt_prefix_and_plumbs_kwargs():
    """Returns only the newly generated suffix (columns past the prompt length) and forwards
    the deterministic decode config + eos pad id to generate()."""
    prompt_len = 4
    new = torch.tensor([[101, 102, 103]])
    full = torch.cat([torch.arange(prompt_len).view(1, -1), new], dim=1)  # [1, 7]

    llm = _client(model_out=full)
    input_ids = torch.arange(prompt_len).view(1, -1)
    attn = torch.ones_like(input_ids)

    suffix = llm._generate_tokens(input_ids, attn, msg=None, max_new_tokens=3)

    # Handling: exactly the 3 generated columns, nothing from the prompt.
    assert suffix.shape == (1, 3)
    assert torch.equal(suffix, new)

    gk = llm.model.gen_kwargs
    assert gk["max_new_tokens"] == 3
    assert gk["do_sample"] is False and gk["use_cache"] is True   # DECODE_KWARGS
    assert gk["do_sample"] == DECODE_KWARGS["do_sample"]
    assert gk["pad_token_id"] == llm.tokenizer.eos_token_id
    assert torch.equal(gk["input_ids"], input_ids)
    assert torch.equal(gk["attention_mask"], attn)


# ---------------------------------------------------------------------------
# query_llm — end-to-end orchestration (real forward + inverse translation)
# ---------------------------------------------------------------------------

def test_query_llm_returns_spine_json_and_true():
    """Full pipeline: SPINE msg -> compact prompt -> (stub) generate -> decode -> SPINE JSON.

    The decoded compact output is fixed; the returned JSON must carry the four SPINE keys,
    with the plan re-wrapped as [answer(...)] and reasoning/relevant_graph extracted from the
    <think> block. Hand-computed expectation — not via compact_output_to_spine_json."""
    decoded = "<think>Relevant graph: a, b\n\nReasoning: hop a to b</think>a -> b"
    llm = _client(decoded=decoded, model_out=torch.zeros(1, 6, dtype=torch.long))

    result = llm.query_llm(_SPINE_MSG, max_new_tokens=16)

    assert isinstance(result, tuple) and len(result) == 2
    planner_json, ok = result
    assert ok is True
    assert isinstance(planner_json, str)

    parsed = json.loads(planner_json)
    assert set(parsed) == {"primary_goal", "relevant_graph", "reasoning", "plan"}
    assert parsed["primary_goal"] == ""
    assert parsed["relevant_graph"] == "a, b"
    assert parsed["reasoning"] == "hop a to b"
    assert parsed["plan"] == "[answer(a -> b)]"


def test_query_llm_no_think_block_routes_all_text_to_plan():
    """When the model emits no <think> block, the whole output becomes the plan (reasoning
    empty) — the inverse translator's documented fallback, still re-wrapped as [answer(...)]"""
    llm = _client(decoded="x -> y -> z", model_out=torch.zeros(1, 6, dtype=torch.long))

    planner_json, ok = llm.query_llm(_SPINE_MSG)
    parsed = json.loads(planner_json)

    assert ok is True
    assert parsed["reasoning"] == ""
    assert parsed["relevant_graph"] == ""
    assert parsed["plan"] == "[answer(x -> y -> z)]"


def test_query_llm_forward_translation_drops_system_and_hoists_graph():
    """The messages handed to apply_chat_template are the COMPACT form: the verbose SPINE
    system turn is gone, replaced by a compact system block carrying the scene graph, and the
    graph-bearing user turn is reduced to the task text."""
    llm = _client(decoded="a -> b", model_out=torch.zeros(1, 6, dtype=torch.long),
                  include_edges=False)
    llm.query_llm(_SPINE_MSG)

    sent = llm.tokenizer.chat_messages
    assert sent[0]["role"] == "system"
    assert "Scene graph:" in sent[0]["content"]
    assert "VERBOSE SPINE SYSTEM PROMPT" not in sent[0]["content"]
    # The query user turn is reduced to the bare task (no "task:" prefix, no graph text).
    user_turns = [m for m in sent if m["role"] == "user"]
    assert user_turns[-1]["content"] == "navigate to b"
    assert llm.tokenizer.chat_kwargs == {
        "tokenize": True, "add_generation_prompt": True, "return_tensors": "pt",
    }


def test_query_llm_include_edges_threads_to_compact_block():
    """include_edges=True writes the Region/Object Edges bullets into the hoisted system block;
    include_edges=False omits them (the GNN-supplies-connectivity ablation)."""
    with_edges = _client(decoded="a -> b", model_out=torch.zeros(1, 6, dtype=torch.long),
                         include_edges=True)
    with_edges.query_llm(_SPINE_MSG)
    sys_with = with_edges.tokenizer.chat_messages[0]["content"]

    without = _client(decoded="a -> b", model_out=torch.zeros(1, 6, dtype=torch.long),
                      include_edges=False)
    without.query_llm(_SPINE_MSG)
    sys_without = without.tokenizer.chat_messages[0]["content"]

    assert "Region Edges:" in sys_with and "a <=> b" in sys_with
    assert "Region Edges:" not in sys_without
    assert "Object Edges:" not in sys_without


# ---------------------------------------------------------------------------
# Runner footer — standalone + pytest
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: PASS")
    print("done")
