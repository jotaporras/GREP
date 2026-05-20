"""Tests for prism.data.utils.strip_icl — ICL-prefix removal from SPINE rollouts."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prism.data.utils import strip_icl


def _sys(c="system prompt"):
    return {"role": "system", "content": c}


def _user(c):
    return {"role": "user", "content": c}


def _asst(c="plan"):
    return {"role": "assistant", "content": c}


def test_strips_icl_prefix_keeps_system_and_real_rollout():
    msgs = [
        _sys(),
        _user("task: ICL example one"),
        _asst(),
        _user("task: ICL example two"),
        _asst(),
        _user("task: THE REAL TASK"),
        _asst("real plan"),
        _user("updates: ..."),
        _asst("final answer"),
    ]
    assert strip_icl(msgs) == [
        _sys(),
        _user("task: THE REAL TASK"),
        _asst("real plan"),
        _user("updates: ..."),
        _asst("final answer"),
    ]


def test_real_task_is_the_last_task_user_message():
    msgs = [_sys(), _user("task: a"), _asst(), _user("task: b"), _asst()]
    assert strip_icl(msgs)[1]["content"] == "task: b"


def test_no_system_turn_is_kept_as_is():
    msgs = [_user("task: only task"), _asst()]
    assert strip_icl(msgs) == msgs


def test_task_prefix_match_is_case_insensitive_and_ignores_leading_ws():
    msgs = [_sys(), _user("  Task: padded"), _asst()]
    assert strip_icl(msgs)[1]["content"] == "  Task: padded"


def test_raises_when_no_task_user_message():
    with pytest.raises(ValueError, match="no `task:`"):
        strip_icl([_sys(), _user("updates: nothing"), _asst()])


def test_raises_when_no_assistant_after_task():
    with pytest.raises(ValueError, match="no assistant"):
        strip_icl([_sys(), _user("task: unanswered")])


def test_raises_on_empty_input():
    with pytest.raises(ValueError, match="non-empty"):
        strip_icl([])
