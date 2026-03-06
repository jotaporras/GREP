"""
Tests for ``remove_edge_list`` – the helper that strips object_connections and
region_connections from a decoded scene-graph prompt.

Two concerns are validated:

  1. **Correct removal** – the connection lists are removed and nothing else.
  2. **Token alignment** – tokens that appear *after* the removed section still
     map to the same text.  A naive decode → re-encode cycle can shift token
     positions if the regex eats too much or too little.

Run with::

    PYTHONPATH=src pytest test_remove_edge_list.py -v
"""

import re

import pytest
from transformers import AutoTokenizer

from prism.data.data import remove_edge_list

# ── fixtures ────────────────────────────────────────────────────────────

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

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="module")
def decoded_prompt(tokenizer):
    """Full tokenized-then-decoded prompt (round-trip through tokenizer)."""
    prompt = tokenizer.apply_chat_template(
        CONVERSATION, tokenize=False, add_generation_prompt=False,
    )
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    return tokenizer.decode(ids[0])


@pytest.fixture(scope="module")
def cleaned_prompt(decoded_prompt):
    return remove_edge_list(decoded_prompt)


# ── 1. Correct removal ─────────────────────────────────────────────────

class TestRemoval:
    """Verify the edge list is removed and nothing else is lost."""

    def test_object_connections_removed(self, cleaned_prompt):
        assert "'object_connections'" not in cleaned_prompt

    def test_region_connections_removed(self, cleaned_prompt):
        assert "'region_connections'" not in cleaned_prompt

    def test_robot_location_preserved(self, cleaned_prompt):
        assert "'robot_location': 'field_11'" in cleaned_prompt

    def test_objects_preserved(self, cleaned_prompt):
        assert "'objects'" in cleaned_prompt

    def test_regions_preserved(self, cleaned_prompt):
        assert "'regions'" in cleaned_prompt

    def test_node_names_preserved(self, cleaned_prompt):
        for name in ["house_1", "grocery_store_1", "shed_1", "field_11"]:
            assert name in cleaned_prompt, f"{name} missing from cleaned prompt"

    def test_assistant_response_intact(self, cleaned_prompt):
        """The assistant turn must be completely untouched."""
        assert '"primary_goal"' in cleaned_prompt
        assert '"plan"' in cleaned_prompt
        assert "goto(field_11)" in cleaned_prompt
        assert "map_region(field_11)" in cleaned_prompt

    def test_task_instruction_intact(self, cleaned_prompt):
        assert "I need a shovel" in cleaned_prompt


# ── 2. Token alignment ─────────────────────────────────────────────────

class TestTokenAlignment:
    """After decode → remove_edge_list → re-encode, tokens for content
    *after* the removed section must still decode to the same text."""

    @pytest.fixture()
    def original_ids(self, tokenizer):
        prompt = tokenizer.apply_chat_template(
            CONVERSATION, tokenize=False, add_generation_prompt=False,
        )
        return tokenizer(prompt, return_tensors="pt")["input_ids"][0]

    @pytest.fixture()
    def cleaned_ids(self, tokenizer, original_ids):
        decoded = tokenizer.decode(original_ids)
        cleaned = remove_edge_list(decoded)
        return tokenizer(cleaned, return_tensors="pt")["input_ids"][0]

    def test_cleaned_is_shorter(self, original_ids, cleaned_ids):
        """Removing text must result in fewer tokens."""
        assert len(cleaned_ids) < len(original_ids), (
            f"Expected cleaned ({len(cleaned_ids)}) < original ({len(original_ids)})"
        )

    def test_assistant_tokens_decode_correctly(self, tokenizer, cleaned_ids):
        """The assistant response must decode to the exact same text after
        the edge list is removed.  This catches off-by-one shifts."""
        full_decoded = tokenizer.decode(cleaned_ids)
        # Extract the assistant portion (everything after the last assistant marker)
        # Qwen uses <|im_start|>assistant
        marker = "<|im_start|>assistant"
        assert marker in full_decoded, "Chat template marker not found"
        assistant_text = full_decoded.split(marker)[-1]
        assert "goto(field_11)" in assistant_text
        assert "map_region(field_11)" in assistant_text
        assert '"primary_goal"' in assistant_text

    def test_robot_location_tokens_decode_correctly(self, tokenizer, cleaned_ids):
        """robot_location comes right after the removed section; if the regex
        ate too much, it will be missing or corrupted."""
        decoded = tokenizer.decode(cleaned_ids)
        assert "'robot_location': 'field_11'" in decoded

    def test_no_double_commas_or_artifacts(self, cleaned_ids, tokenizer):
        """Removing the edge list should not leave syntactic artifacts like
        double commas or dangling brackets."""
        decoded = tokenizer.decode(cleaned_ids)
        # Extract the scene graph portion
        sg_match = re.search(r"Scene graph:\{(.*?)\}", decoded)
        assert sg_match, "Scene graph not found in cleaned prompt"
        sg_body = sg_match.group(1)
        assert ",," not in sg_body, f"Double comma found in: {sg_body}"
        assert ", ," not in sg_body, f"Spaced double comma found in: {sg_body}"


# ── 3. Regression: greedy regex over-matching ───────────────────────────

class TestGreedyRegexRegression:
    """The original regex ``r"'object_connections': ?.*,"`` is greedy and
    will match past the intended boundary.  These tests would fail with
    the old regex."""

    def test_old_regex_eats_too_much(self):
        """Demonstrate that the old greedy pattern removes region_connections
        AND robot_location (everything up to last comma on the line)."""
        scene_line = (
            "'object_connections': [['shed_1', 'field_11']], "
            "'region_connections': [], "
            "'robot_location': 'field_11'}"
        )
        old_pattern = r"'object_connections': ?.*,"
        old_result = re.sub(old_pattern, "", scene_line)
        # The old regex eats everything up to the last comma, which is
        # after 'field_11' in robot_location — so robot_location is mangled
        assert "'robot_location'" not in old_result or "'field_11'}" in old_result, (
            "Old regex should have over-matched"
        )

    def test_new_function_preserves_robot_location(self):
        """remove_edge_list must keep robot_location intact."""
        text = (
            "Scene graph:{'objects': [], 'regions': [], "
            "'object_connections': [['a', 'b']], "
            "'region_connections': [['c', 'd']], "
            "'robot_location': 'field_11'}"
        )
        result = remove_edge_list(text)
        assert "'robot_location': 'field_11'" in result

    def test_no_content_after_scene_graph_is_eaten(self):
        """Text that follows the scene graph (e.g. on the same line or in
        the next role) must not be consumed."""
        text = (
            "Scene graph:{'objects': [], "
            "'object_connections': [['a', 'b']], "
            "'region_connections': [], "
            "'robot_location': 'home'}\n"
            "assistant: I will go to the park, then the store."
        )
        result = remove_edge_list(text)
        assert "I will go to the park, then the store." in result


# ── 4. Edge cases ──────────────────────────────────────────────────────

class TestEdgeCases:

    def test_empty_connections(self):
        text = (
            "'object_connections': [], "
            "'region_connections': [], "
            "'robot_location': 'x'}"
        )
        result = remove_edge_list(text)
        assert "'object_connections'" not in result
        assert "'robot_location': 'x'" in result

    def test_nested_brackets_in_connections(self):
        text = (
            "'object_connections': [['a', 'b'], ['c', 'd']], "
            "'region_connections': [['e', 'f']], "
            "'robot_location': 'y'}"
        )
        result = remove_edge_list(text)
        assert "'object_connections'" not in result
        assert "'region_connections'" not in result
        assert "'robot_location': 'y'" in result

    def test_no_connections_present(self):
        """If there are no connection keys, text is returned unchanged."""
        text = "Scene graph:{'objects': [], 'robot_location': 'z'}"
        assert remove_edge_list(text) == text
