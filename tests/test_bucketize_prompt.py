"""
Tests for ``bucketize_prompt`` and ``has_match`` – the helpers that map
scene-graph node names to their sub-word token indices inside a tokenized
conversation.

Three properties are validated on a single, carefully constructed
conversation drawn from data/eval/gpt_gen_formatted.json:

  1. **Multi-token nodes** – ``grocery_store_1`` spans several BPE
     pieces; the token sequence must be longer than 1.
  2. **Repeated mentions** – ``field_11`` and ``shed_1`` appear in both
     the user turn (scene graph) *and* the assistant turn (plan /
     reasoning); the bucket must capture *every* mention.
  3. **Partial-match rejection** – the bare word ``shed`` and the
     plural ``sheds`` appear in running text but must *not* produce
     bucket entries (only ``shed_1`` should).

Run with::

    PYTHONPATH=src pytest test_bucketize_prompt.py -v
"""

import re

import pytest
from transformers import AutoTokenizer

from prism.models.gnn_llm import bucketize_prompt, has_match


# ── test fixtures ───────────────────────────────────────────────────────

# Conversation based on data/eval/gpt_gen_formatted.json, edited so that
# a single prompt exercises all three edge-case patterns:
#
#   • multi-token node:     grocery_store_1 (many BPE pieces)
#   • repeated mentions:    field_11, shed_1 across user & assistant turns
#   • partial-match traps:  bare "shed" and "sheds" in running text

CONVERSATION = [
    {
        "role": "user",
        "content": (
            "task: I need a shovel. Is there one near the shed in the scene?\n"
            "Scene graph:{"
            "'objects': ["
            "{'name': 'house_1', 'coords': [-1, -1]}, "
            "{'name': 'grocery_store_1', 'coords': [-5, -1]}, "
            "{'name': 'shed_1', 'coords': [1, 3]}], "
            "'regions': [{'name': 'field_11', 'coords': [0, 1]}], "
            "'object_connections': [['shed_1', 'field_11']], "
            "'region_connections': [], "
            "'robot_location': 'field_11'}"
        ),
    },
    {
        "role": "assistant",
        "content": (
            '{"primary_goal": "find a shovel near the shed", '
            '"relevant_graph": "field_11, shed_1, unobserved_node(shovel)", '
            '"reasoning": "The graph has one shed, shed_1, connected to '
            "field_11. There are two sheds total but only shed_1 is "
            "observed. I will explore field_11 to look for a shovel near "
            'shed_1.", '
            '"plan": "[goto(field_11), map_region(field_11)]"}'
        ),
    },
]

NODE_LIST = ["house_1", "grocery_store_1", "shed_1", "field_11"]
# Indices:      0              1              2          3

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


# ── shared setup (downloaded once per session) ──────────────────────────

@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="module")
def input_ids(tokenizer):
    prompt = tokenizer.apply_chat_template(
        CONVERSATION, tokenize=False, add_generation_prompt=False,
    )
    return tokenizer(prompt, return_tensors="pt")["input_ids"]


@pytest.fixture(scope="module")
def node_token_seqs(tokenizer):
    return tokenizer.encode(NODE_LIST)


@pytest.fixture(scope="module")
def bucket(input_ids, node_token_seqs):
    return bucketize_prompt(input_ids[0, :].tolist(), node_token_seqs)


# ── 1. Multi-token nodes ───────────────────────────────────────────────

def test_multitoken_node_present(bucket):
    """grocery_store_1 (node index 1) must appear in the bucket."""
    assert 1 in bucket


def test_multitoken_node_has_multiple_tokens(node_token_seqs):
    """grocery_store_1 is long enough that any BPE tokenizer will split
    it into more than one piece."""
    assert len(node_token_seqs[1]) > 1, (
        f"Expected >1 token for grocery_store_1, got {node_token_seqs[1]}"
    )


def test_multitoken_node_indices_decode_to_name(bucket, input_ids, tokenizer, node_token_seqs):
    """Decoding the token-ids at each match start must reproduce
    grocery_store_1."""
    starts = bucket[1]
    seq_len = len(node_token_seqs[1])
    ids_list = input_ids[0].tolist()
    for s in starts:
        span_ids = ids_list[s:s + seq_len]
        reconstructed = tokenizer.decode(span_ids).replace(" ", "")
        assert "grocery_store_1" in reconstructed, (
            f"Decoded tokens at pos {s} gave '{reconstructed}', "
            f"expected 'grocery_store_1'"
        )


# ── 2. Repeated mentions across roles ──────────────────────────────────

def test_repeated_field_11_has_multiple_matches(bucket):
    """field_11 appears multiple times across user and assistant turns;
    bucketize_prompt must find more than one token-level match.
    (Not every text mention will match because BPE tokenization is
    context-dependent — a leading space changes the first token.)"""
    assert len(bucket[3]) > 1, (
        f"field_11 should have multiple token-level matches, got {len(bucket[3])}"
    )


def test_repeated_shed_1_has_multiple_matches(bucket):
    """shed_1 appears in both user and assistant turns; bucketize_prompt
    must find more than one token-level match."""
    assert len(bucket[2]) > 1, (
        f"shed_1 should have multiple token-level matches, got {len(bucket[2])}"
    )


# ── 3. Partial-match rejection ──────────────────────────────────────────

def test_bare_shed_not_bucketed(bucket, input_ids, tokenizer, node_token_seqs):
    """Positions matching bare 'shed' (no _1 suffix) must not appear as
    bucket entries for shed_1. Every match must decode to 'shed_1'."""
    decoded = tokenizer.decode(input_ids[0])
    assert re.search(r"\bshed\b", decoded), (
        "Test conversation should contain the bare word 'shed'"
    )
    ids_list = input_ids[0].tolist()
    seq_len = len(node_token_seqs[2])
    for s in bucket[2]:
        span = tokenizer.decode(ids_list[s:s + seq_len]).replace(" ", "")
        assert "shed_1" in span, (
            f"Bucket entry at pos {s} decoded to '{span}', not 'shed_1'"
        )


def test_sheds_plural_not_bucketed(bucket, input_ids, tokenizer, node_token_seqs):
    """The plural 'sheds' must not produce any bucket entries."""
    decoded = tokenizer.decode(input_ids[0])
    assert "sheds" in decoded, (
        "Test conversation should contain the word 'sheds'"
    )
    ids_list = input_ids[0].tolist()
    seq_len = len(node_token_seqs[2])
    for s in bucket[2]:
        span = tokenizer.decode(ids_list[s:s + seq_len]).replace(" ", "")
        assert "sheds" not in span


def test_shed_1_still_present_despite_partial_neighbours(bucket):
    """shed_1 must be correctly bucketed even when 'shed' and 'sheds'
    appear nearby in the text."""
    assert 2 in bucket
    assert len(bucket[2]) >= 1


# ── 4. Sanity: every declared node is present ───────────────────────────

@pytest.mark.parametrize("node_idx,node_name", list(enumerate(NODE_LIST)))
def test_all_nodes_present(bucket, node_idx, node_name):
    """Every node in NODE_LIST should have at least one bucket entry."""
    assert node_idx in bucket, f"{node_name} (index {node_idx}) missing from bucket"


# ── 5. has_match unit tests ─────────────────────────────────────────────

class TestHasMatch:
    """Unit tests for the has_match helper function."""

    def test_match_at_start(self):
        assert has_match([1, 2, 3], [1, 2], start_pos=0) is True

    def test_no_match_at_wrong_position(self):
        assert has_match([1, 2, 3], [1, 2], start_pos=1) is False

    def test_no_match_different_subsequence(self):
        assert has_match([1, 2, 3], [2, 3], start_pos=0) is False

    def test_match_at_middle(self):
        assert has_match([1, 2, 3], [2, 3], start_pos=1) is True

    def test_start_pos_at_end_of_list(self):
        """start_pos at the boundary should return False for non-empty to_match."""
        assert has_match([1, 2, 3], [2, 3], start_pos=3) is False

    def test_start_pos_beyond_end(self):
        """start_pos past the end should not raise; returns False."""
        assert has_match([1, 2, 3], [4, 5], start_pos=5) is False

    def test_empty_to_match(self):
        """Empty subsequence matches everywhere (vacuous truth)."""
        assert has_match([1, 2, 3], [], start_pos=0) is True
        assert has_match([1, 2, 3], [], start_pos=2) is True

    def test_single_element_match(self):
        assert has_match([10, 20, 30], [20], start_pos=1) is True

    def test_single_element_no_match(self):
        assert has_match([10, 20, 30], [20], start_pos=0) is False

    def test_full_sequence_match(self):
        assert has_match([1, 2, 3], [1, 2, 3], start_pos=0) is True

    def test_to_match_longer_than_remaining(self):
        """When to_match extends past the end of input_ids_b, it cannot match."""
        assert has_match([1, 2, 3], [3, 4], start_pos=2) is False
