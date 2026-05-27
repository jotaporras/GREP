"""Tests for SPINE.clean_llm_output — markdown fence stripping."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "SPINE" / "src"))

from spine.spine import SPINE


@pytest.fixture
def spine():
    """Minimal SPINE instance — no model/graph needed for clean_llm_output."""
    s = SPINE.__new__(SPINE)
    return s


PLAIN_JSON = '{"primary_goal": "test", "plan": "goto(x)"}'


class TestCleanLlmOutput:
    # --- previous behavior: plain JSON passes through unchanged ---------------

    def test_plain_json_unchanged(self, spine):
        assert spine.clean_llm_output(PLAIN_JSON) == PLAIN_JSON

    def test_leading_trailing_whitespace_stripped(self, spine):
        assert spine.clean_llm_output(f"  {PLAIN_JSON}  ") == PLAIN_JSON

    # --- fence variants the fix must handle -----------------------------------

    def test_fenced_json_tag(self, spine):
        s = f"```json\n{PLAIN_JSON}\n```"
        assert spine.clean_llm_output(s) == PLAIN_JSON

    def test_fenced_no_tag(self, spine):
        s = f"```\n{PLAIN_JSON}\n```"
        assert spine.clean_llm_output(s) == PLAIN_JSON

    def test_fenced_with_leading_newline(self, spine):
        """Regression: old strip('```') was a no-op when output started with \\n."""
        s = f"\n```json\n{PLAIN_JSON}\n```\n"
        assert spine.clean_llm_output(s) == PLAIN_JSON

    def test_fenced_with_leading_spaces(self, spine):
        s = f"  ```json\n{PLAIN_JSON}\n```  "
        assert spine.clean_llm_output(s) == PLAIN_JSON

    # --- the exact error case from the eval run -------------------------------

    def test_reported_error_case(self, spine):
        """Model output that started with \\n before the fence previously produced
        an empty string after the old character-strip, causing
        'Expecting value: line 1 column 1 (char 0)'."""
        model_output = (
            "\n```json\n"
            '{\n  "primary_goal": "Find path",\n'
            '  "relevant_graph": "...",\n'
            '  "reasoning": "...",\n'
            '  "plan": [{"action": "goto", "node": "hallway_2"}]\n'
            "}\n```\n"
        )
        result = spine.clean_llm_output(model_output)
        # Must be parseable JSON — this was the failure before the fix
        import json
        parsed = json.loads(result)
        assert parsed["plan"][0]["action"] == "goto"

    def test_fenced_with_trailing_explanation(self, spine):
        """Model appends text after the closing fence, e.g. 'I removed the extra newline...'
        The closing-fence regex must consume everything after ``` not just the fence itself."""
        s = (
            "```json\n"
            f"{PLAIN_JSON}\n"
            "```\n\n"
            "I removed the extra newline at the end of the JSON object, "
            "which was causing the parsing error."
        )
        assert spine.clean_llm_output(s) == PLAIN_JSON

    # --- old behaviour that must NOT be broken --------------------------------

    def test_old_strip_would_have_failed_this(self, spine):
        """Document that the old implementation would silently corrupt this input.
        The new implementation handles it correctly."""
        s = f"\n```json\n{PLAIN_JSON}\n```\n"
        # Old: s.strip('```').strip('json') on leading-\\n input → still fenced
        old_result = s.strip("```").strip("json")
        assert old_result != PLAIN_JSON, "sanity: old approach does NOT handle this"
        # New: correct
        assert spine.clean_llm_output(s) == PLAIN_JSON
