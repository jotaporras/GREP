"""Tests for GPTQueryClient.batch_query_gpt_5 — OpenAI Batch API integration."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prism.data.utils import GPTQueryClient


def _make_batch(status: str, completed: int = 0, total: int = 0, output_file_id: str = "out-file"):
    return SimpleNamespace(
        id="batch-1",
        status=status,
        output_file_id=output_file_id,
        request_counts=SimpleNamespace(completed=completed, total=total, failed=0),
    )


def _make_client(*, statuses, output_lines):
    """Build a fake OpenAI client.

    `statuses` is the sequence of batch.status values returned by successive
    `batches.retrieve` calls. The first `batches.create` returns the same as
    statuses[0]. `output_lines` is the JSONL body of the result file.
    """
    client = MagicMock()
    client.files.create.return_value = SimpleNamespace(id="file-1")
    client.batches.create.return_value = _make_batch(statuses[0], total=2)
    client.batches.retrieve.side_effect = [
        _make_batch(s, completed=i + 1, total=2) for i, s in enumerate(statuses[1:])
    ]
    client.files.content.return_value = SimpleNamespace(text="\n".join(output_lines))
    return client


def _response_line(custom_id: str, text: str) -> str:
    return json.dumps({
        "custom_id": custom_id,
        "response": {"body": {"output_text": text}},
    })


@pytest.fixture
def gpt_client():
    c = GPTQueryClient.__new__(GPTQueryClient)
    c.client = MagicMock()
    return c


class TestBatchQueryGpt5:
    def test_returns_responses_in_input_order(self, gpt_client, monkeypatch):
        monkeypatch.setattr("prism.data.utils.time.sleep", lambda _: None)
        # Responses come back out of order; method must reorder by custom_id.
        gpt_client.client = _make_client(
            statuses=["validating", "completed"],
            output_lines=[
                _response_line("2", "third"),
                _response_line("0", "first"),
                _response_line("1", "second"),
            ],
        )

        out = gpt_client.batch_query_gpt_5(["q0", "q1", "q2"], poll_interval=0)

        assert out == ["first", "second", "third"]

    def test_submits_request_payload_with_default_params(self, gpt_client, monkeypatch):
        monkeypatch.setattr("prism.data.utils.time.sleep", lambda _: None)
        gpt_client.client = _make_client(
            statuses=["completed"],
            output_lines=[_response_line("0", "ok")],
        )

        gpt_client.batch_query_gpt_5(["hello"], poll_interval=0)

        upload_call = gpt_client.client.files.create.call_args
        _, file_bytes, _ = upload_call.kwargs["file"]
        record = json.loads(file_bytes.getvalue().decode())
        assert record["custom_id"] == "0"
        assert record["method"] == "POST"
        assert record["url"] == "/v1/responses"
        assert record["body"]["model"] == "gpt-5.1"
        assert record["body"]["reasoning"] == {"effort": "low", "summary": "auto"}
        assert record["body"]["input"][0]["content"][0]["text"] == "hello"

        gpt_client.client.batches.create.assert_called_once_with(
            input_file_id="file-1",
            endpoint="/v1/responses",
            completion_window="24h",
        )

    def test_model_and_reasoning_effort_override_defaults(self, gpt_client, monkeypatch):
        monkeypatch.setattr("prism.data.utils.time.sleep", lambda _: None)
        gpt_client.client = _make_client(
            statuses=["completed"],
            output_lines=[_response_line("0", "ok")],
        )

        gpt_client.batch_query_gpt_5(
            ["hello"],
            model="gpt-5-nano",
            reasoning_effort="minimal",
            poll_interval=0,
        )

        _, file_bytes, _ = gpt_client.client.files.create.call_args.kwargs["file"]
        record = json.loads(file_bytes.getvalue().decode())
        assert record["body"]["model"] == "gpt-5-nano"
        assert record["body"]["reasoning"]["effort"] == "minimal"

    def test_polls_until_terminal_status(self, gpt_client, monkeypatch):
        sleeps = []
        monkeypatch.setattr("prism.data.utils.time.sleep", lambda s: sleeps.append(s))
        gpt_client.client = _make_client(
            statuses=["validating", "in_progress", "finalizing", "completed"],
            output_lines=[_response_line("0", "done")],
        )

        gpt_client.batch_query_gpt_5(["q"], poll_interval=7)

        assert gpt_client.client.batches.retrieve.call_count == 3
        assert sleeps == [7, 7, 7]

    @pytest.mark.parametrize("bad_status", ["failed", "expired", "cancelled"])
    def test_raises_on_non_completed_terminal_status(self, gpt_client, monkeypatch, bad_status):
        monkeypatch.setattr("prism.data.utils.time.sleep", lambda _: None)
        gpt_client.client = _make_client(
            statuses=["validating", bad_status],
            output_lines=[],
        )

        with pytest.raises(RuntimeError, match=bad_status):
            gpt_client.batch_query_gpt_5(["q"], poll_interval=0)

    def test_raises_on_missing_response(self, gpt_client, monkeypatch):
        monkeypatch.setattr("prism.data.utils.time.sleep", lambda _: None)
        gpt_client.client = _make_client(
            statuses=["completed"],
            output_lines=[_response_line("0", "only-first")],
        )

        with pytest.raises(KeyError):
            gpt_client.batch_query_gpt_5(["q0", "q1"], poll_interval=0)

    def test_ignores_blank_lines_in_output_file(self, gpt_client, monkeypatch):
        monkeypatch.setattr("prism.data.utils.time.sleep", lambda _: None)
        gpt_client.client = _make_client(
            statuses=["completed"],
            output_lines=["", _response_line("0", "a"), "", _response_line("1", "b"), ""],
        )

        assert gpt_client.batch_query_gpt_5(["q0", "q1"], poll_interval=0) == ["a", "b"]
