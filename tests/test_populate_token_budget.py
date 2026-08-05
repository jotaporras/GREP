"""Tests for the Phase-1 populate generation budget.

Guards the defect these tests were written for: the budget was hard-coded at
10240 and unreachable from config, because ``query_gpt*``'s ``max_tokens``
parameter defaulted to a number rather than None — which made the
``max_tokens or self.max_new_tokens`` fallback (and therefore the constructor
argument set from config) dead code. Truncated generations then surfaced as
bogus ``json.loads`` parse errors.

No model weights are loaded: every test drives ``TaskGraphGen`` with an
explicit fake client, and the ``LocalHFQueryClient`` checks are on the
signature only.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prism.data import local_llm
from prism.data.graph_gen import TaskGraphGen


class _RecordingClient:
    """Captures the kwargs TaskGraphGen sends, then aborts before parsing."""

    def __init__(self):
        self.calls = []

    def query_gpt(self, **kwargs):
        self.calls.append(kwargs)
        raise RuntimeError("stop-before-parse")


@pytest.fixture
def client():
    return _RecordingClient()


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("PRISM_HF_POPULATE_MAX_NEW_TOKENS", raising=False)


class TestPopulateMaxTokens:
    def test_default_is_twice_the_measured_output_for_the_110_node_regime(self):
        # 184 tok/node at p95 (google/gemma-4-31B-it) x 110 nodes = 20240 tokens
        # of required output; the budget must leave at least as much again for
        # the <think> block, which shares it.
        assert local_llm.DEFAULT_POPULATE_MAX_TOKENS >= 2 * 184 * 110

    def test_env_var_overrides_default(self, monkeypatch):
        monkeypatch.setenv("PRISM_HF_POPULATE_MAX_NEW_TOKENS", "12345")
        assert local_llm.populate_max_tokens() == 12345

    def test_falls_back_to_default_when_env_unset(self):
        assert local_llm.populate_max_tokens() == local_llm.DEFAULT_POPULATE_MAX_TOKENS


class TestTaskGraphGenPlumbing:
    def test_budget_reaches_the_client_on_every_query(self, client):
        gen = TaskGraphGen(client=client, max_tokens=4321)

        with pytest.raises(RuntimeError, match="stop-before-parse"):
            gen.get_tasks(base_graph="{}", n_tasks=1)

        assert client.calls[0]["max_tokens"] == 4321

    def test_unset_budget_resolves_to_config_not_none(self, client):
        gen = TaskGraphGen(client=client)

        with pytest.raises(RuntimeError, match="stop-before-parse"):
            gen.get_tasks(base_graph="{}", n_tasks=1)

        # An unbounded generate() must never be what reaches the backend.
        assert client.calls[0]["max_tokens"] == local_llm.populate_max_tokens()

    def test_env_var_reaches_the_client(self, client, monkeypatch):
        monkeypatch.setenv("PRISM_HF_POPULATE_MAX_NEW_TOKENS", "777")
        gen = TaskGraphGen(client=client)

        with pytest.raises(RuntimeError, match="stop-before-parse"):
            gen.get_tasks(base_graph="{}", n_tasks=1)

        assert client.calls[0]["max_tokens"] == 777


class TestClientFallbackIsLive:
    @pytest.mark.parametrize("method", ["query_gpt", "query_gpt_5"])
    def test_local_client_max_tokens_defaults_to_none(self, method):
        # If this default is a number again, `max_tokens or self.max_new_tokens`
        # can never reach self.max_new_tokens and the config becomes dead code.
        sig = inspect.signature(getattr(local_llm.LocalHFQueryClient, method))
        assert sig.parameters["max_tokens"].default is None

    def test_local_client_constructor_default_is_the_measured_budget(self):
        sig = inspect.signature(local_llm.LocalHFQueryClient.__init__)
        assert (
            sig.parameters["max_new_tokens"].default
            == local_llm.DEFAULT_POPULATE_MAX_TOKENS
        )
