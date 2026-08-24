"""e20 path-only prediction: the ``response_format`` axis and the generator modes.

Three layers, mirroring ``tests/test_spine_tools_icl.py``:

* ``route_only`` prompt/target contract in ``compact_prompt`` (torch-free, pure text):
  the system prompt asks for a bare arrow route and NOTHING else; the training target
  is the route alone (no ``<think>`` scaffold); the default ``think_route`` path is a
  byte-identical no-op; illegal pairings (tools / ICL) fail loud.
* Wiring: ``response_format`` recorded at train time and threaded to every eval
  client build (checked by source parse, like the sibling policy tests).
* Generator modes (``data_gen``): the oracle NetworkX route agrees with the eval
  scorer by construction; committed samples have the exact SPINE-rollout shape
  (``strip_icl``-compatible, leak-free); ``rollout_stats.json`` counts pass/fail.
"""
import ast
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

from prism.data import compact_prompt  # noqa: E402  (torch-free; safe to import)

COMPACT_PY = SRC / "prism/data/compact_prompt.py"
DATA_PY = SRC / "prism/data/data.py"
DATA_GEN_PY = SRC / "prism/data/data_gen.py"
INFERENCE_PY = SRC / "prism/models/inference.py"
EVALUATE_PY = SRC / "prism/eval/evaluate.py"
TRAIN_V3_PY = SRC / "prism/training/train_v3.py"
CALLBACKS_PY = SRC / "prism/eval/callbacks.py"
CHECKPOINT_PY = SRC / "prism/eval/checkpoint.py"


def _graph_dict(robot="kitchen_1"):
    return {
        "regions": [
            {"name": n, "coords": [i, 0, 0]}
            for i, n in enumerate(
                ["kitchen_1", "hall_2", "lab_3", "store_4", "yard_5"])
        ],
        "objects": [{"name": "mug_1", "coords": [0, 0, 0]}],
        "object_connections": [["mug_1", "lab_3"]],
        "region_connections": [
            ["kitchen_1", "hall_2"], ["hall_2", "lab_3"],
            ["kitchen_1", "store_4"], ["store_4", "yard_5"],
            ["yard_5", "lab_3"],
        ],
        "robot_location": robot,
    }


def _rollout(plan="[answer(kitchen_1 -> hall_2 -> lab_3)]"):
    sg = _graph_dict()
    return [
        {"role": "system", "content": "path-only rollout (path_only)"},
        {"role": "user",
         "content": f"task: go to lab_3Scene graph:{json.dumps(sg)}"},
        {"role": "assistant", "content": json.dumps({
            "primary_goal": "go to lab_3", "relevant_graph": "",
            "reasoning": "", "plan": plan})},
    ]


# ---------------------------------------------------------------------------
# Contract text
# ---------------------------------------------------------------------------

class TestAnswerContract:
    def test_route_only_contract_wording(self):
        c = compact_prompt._answer_contract(include_tools=False, route_only=True)
        assert "ONLY" in c
        assert "->" in c
        assert "no <think>" in c
        # No reasoning scaffold is ever requested.
        assert "Reasoning:" not in c

    def test_route_only_with_tools_raises(self):
        with pytest.raises(ValueError):
            compact_prompt._answer_contract(include_tools=True, route_only=True)

    def test_default_contract_unchanged_by_flag_default(self):
        assert compact_prompt._answer_contract(
            include_tools=False
        ) == compact_prompt._answer_contract(include_tools=False, route_only=False)


class TestExtractRoute:
    def test_plain_chain(self):
        assert compact_prompt.extract_route("a_1 -> b_2 -> c_3") == "a_1 -> b_2 -> c_3"

    def test_longest_chain_wins_and_normalizes(self):
        text = "maybe x_1->y_2 but the route is a_1 ->  b_2->c_3 -> d_4."
        assert compact_prompt.extract_route(text) == "a_1 -> b_2 -> c_3 -> d_4"

    def test_no_chain(self):
        assert compact_prompt.extract_route("the goal is reachable") is None
        assert compact_prompt.extract_route("") is None
        assert compact_prompt.extract_route(None) is None


# ---------------------------------------------------------------------------
# Targets and conversation builders
# ---------------------------------------------------------------------------

class TestRouteOnlyTargets:
    def test_format_assistant_route_only_is_bare_route(self):
        out = compact_prompt._format_assistant(
            _rollout()[-1]["content"], include_tools=False, route_only=True)
        assert out == "kitchen_1 -> hall_2 -> lab_3"

    def test_format_assistant_no_route_fails_loud(self):
        bad = json.dumps({"primary_goal": "g", "relevant_graph": "",
                          "reasoning": "", "plan": "[answer(yes it is reachable)]"})
        with pytest.raises(RuntimeError):
            compact_prompt._format_assistant(bad, include_tools=False, route_only=True)

    def test_strict_route_false_keeps_malformed_answer_verbatim(self):
        # Live-inference seam: the model's own degenerate answer (e.g. a bare
        # single node — no "a -> b" route) must re-render as-is and grade
        # wrong, NOT abort the eval sample (e20 epoch-1 crash bug).
        bad = json.dumps({"primary_goal": "g", "relevant_graph": "",
                          "reasoning": "", "plan": "[answer(ship_berth_1)]"})
        out = compact_prompt._format_assistant(
            bad, include_tools=False, route_only=True, strict_route=False)
        assert out == "ship_berth_1"

    def test_strict_route_default_is_loud(self):
        # Training/preprocess callers that do not pass strict_route keep the
        # fail-loud contract: a corrupt dataset must never be silently kept.
        bad = json.dumps({"primary_goal": "g", "relevant_graph": "",
                          "reasoning": "", "plan": "[answer(ship_berth_1)]"})
        with pytest.raises(RuntimeError):
            compact_prompt._format_assistant(bad, include_tools=False, route_only=True)

    def test_strict_route_false_still_extracts_good_routes(self):
        out = compact_prompt._format_assistant(
            _rollout()[-1]["content"], include_tools=False, route_only=True,
            strict_route=False)
        assert out == "kitchen_1 -> hall_2 -> lab_3"

    def test_training_messages_route_only(self):
        msgs = compact_prompt.format_training_messages(
            _rollout(), include_edges=True, include_tools=False, route_only=True)
        assert msgs[0]["role"] == "system"
        assert "no <think>" in msgs[0]["content"]
        assert msgs[-1]["role"] == "assistant"
        assert msgs[-1]["content"] == "kitchen_1 -> hall_2 -> lab_3"
        assert "<think>" not in msgs[-1]["content"]

    def test_think_route_default_is_noop(self):
        base = compact_prompt.format_training_messages(
            _rollout(), include_edges=True, include_tools=False)
        explicit = compact_prompt.format_training_messages(
            _rollout(), include_edges=True, include_tools=False, route_only=False)
        assert base == explicit
        assert base[-1]["content"].startswith("<think>")

    def test_live_translator_route_only(self):
        msgs = compact_prompt.spine_to_compact_messages(
            _rollout()[:2], include_edges=True, include_tools=False,
            icl_examples=0, route_only=True)
        assert [m["role"] for m in msgs] == ["system", "user"]
        assert "ONLY" in msgs[0]["content"]
        assert "Robot location: kitchen_1" in msgs[0]["content"]
        assert msgs[1]["content"] == "go to lab_3"

    @pytest.mark.parametrize("kw", [dict(icl_examples=2), dict(include_tools=True)])
    def test_live_translator_illegal_pairings(self, kw):
        with pytest.raises(ValueError):
            compact_prompt.spine_to_compact_messages(
                _rollout()[:2], include_edges=True,
                include_tools=kw.get("include_tools", False),
                icl_examples=kw.get("icl_examples", 0), route_only=True)

    def test_inverse_wraps_bare_route(self):
        parsed = json.loads(
            compact_prompt.compact_output_to_spine_json("kitchen_1 -> lab_3"))
        assert parsed["plan"] == "[answer(kitchen_1 -> lab_3)]"


# ---------------------------------------------------------------------------
# Wiring (source parse — no torch import needed)
# ---------------------------------------------------------------------------

class TestWiring:
    def test_preprocess_dataset_accepts_response_format(self):
        src = DATA_PY.read_text()
        assert 'response_format: str = "think_route"' in src
        assert "route_only=route_only" in src

    def test_train_v3_records_and_threads(self):
        src = TRAIN_V3_PY.read_text()
        # recorded in both run-meta dicts
        assert src.count('"response_format": config.data.response_format') == 2
        # threaded to the callback and both post-hoc evals
        assert src.count("response_format=config.data.response_format") == 3
        # validated
        assert "'think_route' or 'route_only'" in src or \
            '"think_route", "route_only"' in src

    def test_eval_callback_param(self):
        src = CALLBACKS_PY.read_text()
        assert 'response_format: str = "think_route"' in src
        assert "response_format=self.response_format" in src

    def test_evaluate_threads_to_clients(self):
        src = EVALUATE_PY.read_text()
        assert src.count('response_format: str = "think_route"') == 3
        assert "response_format=response_format" in src

    def test_inference_clients_translate_flag(self):
        src = INFERENCE_PY.read_text()
        assert src.count(
            'route_only=(self.response_format == "route_only")') == 2

    def test_checkpoint_resolver_defaults_historical(self):
        src = CHECKPOINT_PY.read_text()
        assert "def resolve_response_format" in src
        assert '"think_route"' in src


# ---------------------------------------------------------------------------
# Generator modes (need numpy + the spine package; skipped where absent)
# ---------------------------------------------------------------------------

def _data_generator():
    pytest.importorskip("numpy")
    pytest.importorskip("spine")
    pytest.importorskip("networkx")
    from prism.data.data_gen import DataGenerator
    # __init__ eagerly builds Phase-1 LLM clients (TaskGraphGen -> OpenAI(),
    # which demands OPENAI_API_KEY) that the e20 runners under test never
    # touch — every helper they use is a staticmethod or self-free. Construct
    # without __init__ so these tests run keyless on the login node.
    return DataGenerator.__new__(DataGenerator)


class TestOracleMode:
    def test_oracle_route_matches_scorer(self):
        gen = _data_generator()
        from prism.eval import path_validator
        graph = _graph_dict()
        entry = {
            "task": "Navigate to lab_3 without passing through hall_2.",
            "init_node": "kitchen_1",
            "answer": "kitchen_1.*lab_3",
            "acceptance_criterion": "The robot reaches lab_3 while avoiding hall_2.",
        }
        route, reason = gen._oracle_route(graph, entry)
        assert route == "kitchen_1 -> store_4 -> yard_5 -> lab_3", reason
        ok, verdict = gen._grade_route(route, graph, entry)
        assert ok and verdict["structured_correct"]

    def test_oracle_route_honors_waypoint(self):
        gen = _data_generator()
        graph = _graph_dict()
        entry = {
            "task": "Go to lab_3 by way of yard_5.",
            "init_node": "kitchen_1",
            "answer": "kitchen_1.*lab_3",
            "acceptance_criterion": "Reach lab_3 by way of yard_5.",
        }
        route, reason = gen._oracle_route(graph, entry)
        assert route is not None, reason
        assert "yard_5" in route.split(" -> ")
        ok, _ = gen._grade_route(route, graph, entry)
        assert ok

    def test_oracle_run_commits_and_stats(self, tmp_path):
        gen = _data_generator()
        graph = _graph_dict()
        good = {
            "task": "Navigate to lab_3.",
            "init_node": "kitchen_1",
            "answer": "kitchen_1.*lab_3",
            "acceptance_criterion": "The robot reaches lab_3.",
        }
        bad = dict(good, task="How many mugs are there?", answer="1",
                   acceptance_criterion="The count is 1.")
        assert gen._run_one_oracle(idx=0, task_idx=0, task_entry=good,
                                   graph=graph, log_dir=str(tmp_path))
        assert not gen._run_one_oracle(idx=0, task_idx=1, task_entry=bad,
                                       graph=graph, log_dir=str(tmp_path))
        sample = tmp_path / "sample_000_000.json"
        failed = tmp_path / "sample_000_001_failed.json"
        assert sample.exists() and failed.exists()

        # committed sample: exact rollout shape, leak-free, route_only-buildable
        msgs = json.loads(sample.read_text())
        assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
        assert msgs[1]["content"].startswith("task: ")
        assert "Scene graph:{" in msgs[1]["content"]
        blob = sample.read_text() + failed.read_text()
        assert "acceptance_criterion" not in blob
        assert '"answer":' not in blob
        train = compact_prompt.format_training_messages(
            msgs, include_edges=True, include_tools=False, route_only=True)
        assert " -> " in train[-1]["content"]

        # resume: a second run skips the committed sample
        assert gen._run_one_oracle(idx=0, task_idx=0, task_entry=good,
                                   graph=graph, log_dir=str(tmp_path))

        gen._write_rollout_stats(str(tmp_path), "oracle")
        stats = json.loads((tmp_path / "rollout_stats.json").read_text())
        assert stats["n_pass"] == 1 and stats["n_fail"] == 1
        assert stats["per_graph"]["000"] == {"pass": 1, "fail": 1}
        assert "not_a_path_task" in stats["fail_reasons"] or \
            "no_goal_resolved" in stats["fail_reasons"]


class _FakeTeacher:
    """Minimal query_llm client: returns a canned response per call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def query_llm(self, msg, max_new_tokens=None):
        self.prompts.append(msg)
        return self.responses.pop(0), True


class TestPathOnlyMode:
    def test_path_only_grades_and_commits(self, tmp_path):
        gen = _data_generator()
        graph = _graph_dict()
        entry = {
            "task": "Navigate to lab_3.",
            "init_node": "kitchen_1",
            "answer": "kitchen_1.*lab_3",
            "acceptance_criterion": "The robot reaches lab_3.",
        }
        teacher = _FakeTeacher([
            "kitchen_1 -> hall_2 -> lab_3",          # correct
            "kitchen_1 -> yard_5 -> lab_3",           # hallucinated edge
            "The lab is reachable from the kitchen.",  # no route at all
        ])
        assert gen._run_one_path_only(idx=1, task_idx=0, task_entry=entry,
                                      graph=graph, log_dir=str(tmp_path),
                                      spine_client=teacher)
        assert not gen._run_one_path_only(idx=1, task_idx=1, task_entry=entry,
                                          graph=graph, log_dir=str(tmp_path),
                                          spine_client=teacher)
        assert not gen._run_one_path_only(idx=1, task_idx=2, task_entry=entry,
                                          graph=graph, log_dir=str(tmp_path),
                                          spine_client=teacher)
        assert (tmp_path / "sample_001_000.json").exists()
        wrong = json.loads((tmp_path / "sample_001_001_failed.json").read_text())
        assert wrong["reason"] == "wrong_route"
        noroute = json.loads((tmp_path / "sample_001_002_failed.json").read_text())
        assert noroute["reason"] == "no_route"

        # the teacher prompt carries the instruction + task + graph, and NO GT
        prompt_text = json.dumps(teacher.prompts)
        assert "route ONLY" in prompt_text
        assert "acceptance_criterion" not in prompt_text
        assert "kitchen_1.*lab_3" not in prompt_text

        gen._write_rollout_stats(str(tmp_path), "path_only")
        stats = json.loads((tmp_path / "rollout_stats.json").read_text())
        assert stats["n_pass"] == 1 and stats["n_fail"] == 2
        assert stats["fail_reasons"] == {"wrong_route": 1, "no_route": 1}


class TestGeneratorDispatch:
    def test_source_flags(self):
        src = DATA_GEN_PY.read_text()
        assert '"spine", "path_only", "oracle"' in src
        gds = (REPO_ROOT / "scripts/training_data_generation/"
               "generate_data_spine.py").read_text()
        assert "--path-only" in gds and "--oracle-paths" in gds

    def test_oracle_mode_never_builds_a_client(self):
        # oracle branch must set spine_client = None BEFORE _make_spine_client
        # could load a model.
        src = DATA_GEN_PY.read_text()
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "generate_example_plans")
        body_src = ast.get_source_segment(src, fn)
        assert 'if rollout_mode == "oracle":' in body_src
        oracle_block = body_src.split('if rollout_mode == "oracle":')[1]
        oracle_branch = oracle_block.split("elif")[0]
        assert "spine_client = None" in oracle_branch
        assert "_make_spine_client" not in oracle_branch

    def test_spine_client_thinking_param(self):
        for path in (SRC / "prism/data/vllm_llm.py",
                     SRC / "prism/data/local_llm.py"):
            src = path.read_text()
            assert "enable_thinking: bool = True" in src
            assert "enable_thinking=self.enable_thinking" in src
