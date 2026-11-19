"""CPU-only regression test: every eval task is its own planning episode.

Bug (e22 set B/C "starting area" failures): `evaluate.eval_model_single_graph`
built ONE SPINE GraphHandler + planner per graph file and reused them for every
task. SPINE's `GraphHandler.reset` only sets `current_location` while it is empty
(SPINE src/spine/mapping/graph_util.py, reset: "priority is current location ->
incoming argument -> value in data"), and `SPINE.request` keeps the first request
as `base_request` and appends later ones to `msg_history`. So every task after the
first saw task 0's robot location and task 0's conversation.

This test drives the REAL `eval_model_single_graph` loop with boundary fakes that
reproduce exactly those two SPINE contracts, and asserts what each task's planner
sees. The fakes record state at `run_planning` time; scoring is not under test.

Run: python -m pytest tests/test_eval_episode_reset.py -v
"""
import copy
import importlib
import sys
import types

import pytest

sys.path.insert(0, "src")


class _AnyModule(types.ModuleType):
    """Stub module: unknown attributes resolve to a placeholder (import-time only)."""

    def __getattr__(self, item):
        if item.startswith("__"):
            raise AttributeError(item)
        return type(item, (), {})


def _mod(name, **attrs):
    m = _AnyModule(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, m)
    return m


def _import_evaluate():
    """Import prism.eval.evaluate; stub only the boundaries that are missing here."""
    try:
        from prism.eval import evaluate
        return evaluate
    except Exception:
        pass
    for k in [k for k in sys.modules if k.startswith("prism.eval.evaluate")]:
        del sys.modules[k]
    for pkg in ("prism", "prism.models", "prism.data", "prism.eval"):
        importlib.import_module(pkg)
    if "spine" not in sys.modules or not hasattr(sys.modules["spine"], "spine"):
        _mod("spine")
        _mod("spine.spine")
        sys.modules["spine"].spine = sys.modules["spine.spine"]
        _mod("spine.mapping")
        _mod("spine.mapping.graph_util", GraphHandler=object)
        _mod("spine.spine_util")
        _mod("spine.prompts")
        _mod("spine.prompts.examples", EXAMPLE_1=[], EXAMPLE_2=[])
        _mod("spine.prompts.prompts", get_base_prompt_update_graph=(lambda *a, **k: None),
             SYS_PROMPT={"role": "system", "content": "x"})
    try:
        from prism.models import gnn_llm, inference  # noqa: F401
    except Exception:
        class _G:
            pass
        _mod("prism.models.gnn_llm", GraphAugmentedLLM=_G, GraphMaskLLM=_G,
             CompositeGraphLLM=_G)
        _mod("prism.models.inference", GraphAugmentedInMemoryLLM=_G, InMemoryLLM=_G)
    try:
        from prism.data import graph_sim, planning_sim  # noqa: F401
    except Exception:
        class GraphSim:
            def take_action(self, action, argument):
                return False
        _mod("prism.data.graph_sim", GraphSim=GraphSim)
        _mod("prism.data.planning_sim", PlanningSim=object)
    from prism.eval import evaluate
    return evaluate


evaluate = _import_evaluate()


# --- boundary fakes reproducing the two SPINE contracts ------------------------

class StickyGraphHandler:
    """SPINE GraphHandler.reset: current_location is set only while empty."""

    def __init__(self, graph=""):
        self.current_location = ""
        self.graph_dict = None

    def reset(self, graph_as_dict=None, current_location="", **_):
        if self.current_location in (None, ""):
            self.current_location = current_location
        self.graph_dict = graph_as_dict
        return True


class FakeGraphSim:
    """Mirrors prism.data.graph_sim.GraphSim.__init__/reset (lines 24-59)."""

    def __init__(self, graph):
        self.graph = graph
        self.partial_graph = copy.deepcopy(graph)

    def reset(self, **args):
        self.graph.reset(**args)
        self.partial_graph = copy.deepcopy(self.graph)


class FakeSPINE:
    """SPINE.request: first request is base_request, later ones become history."""

    def __init__(self, graph=None, client=None, use_icl=False, **_):
        self.graph = graph
        self.base_request = ""
        self.msg_history = []

    def request(self, text):
        if self.base_request == "":
            self.base_request = text
        else:
            self.msg_history.append({"role": "user", "content": text})


SEEN = []


class FakePlanningSim:
    def __init__(self, debug=False):
        pass

    def run_planning(self, *, llm_planner, task, graph_data_gen, max_iterations=10):
        llm_planner.request(f"task: {task}")
        SEEN.append({
            "location": llm_planner.graph.current_location,
            "prompt_robot_location": llm_planner.graph.graph_dict.get("robot_location"),
            "base_request": llm_planner.base_request,
            "n_history": len(llm_planner.msg_history),
        })
        return types.SimpleNamespace(
            response={"plan": [["answer", "a"]]}, trace=[], terminated_by="answer")


class _Client:
    response_format = "think_route"


def _graph_and_tasks():
    """Set B data_gen_001 pattern: task 0 named start (keep), tasks 1-2 starting area."""
    graph = {
        "regions": [{"name": n, "coords": [i, 0]} for i, n in
                    enumerate(["office_1", "storage_3", "hall_2", "corridor_2"])],
        "objects": [],
        "region_connections": [["office_1", "storage_3"], ["storage_3", "hall_2"],
                               ["hall_2", "corridor_2"]],
        "object_connections": [],
        "robot_location": "office_1",
    }
    tasks = [
        {"task": "From the third storage, go to the second hall.", "init_node": "storage_3",
         "answer": "(?i)storage_3.*hall_2", "keep_robot_location": True},
        {"task": "From the starting area, go to the second hall.", "init_node": "corridor_2",
         "answer": "(?i)corridor_2.*hall_2"},
        {"task": "From the starting area, go to the third storage.", "init_node": "hall_2",
         "answer": "(?i)hall_2.*storage_3"},
    ]
    return graph, tasks


@pytest.fixture
def patched(monkeypatch):
    SEEN.clear()
    monkeypatch.setattr(evaluate.graph_util, "GraphHandler", StickyGraphHandler, raising=False)
    monkeypatch.setattr(evaluate.graph_sim, "GraphSim", FakeGraphSim, raising=False)
    monkeypatch.setattr(evaluate, "_NoToolsGraphSim", FakeGraphSim, raising=False)
    monkeypatch.setattr(evaluate.spine, "SPINE", FakeSPINE, raising=False)
    monkeypatch.setattr(evaluate.planning_sim, "PlanningSim", FakePlanningSim, raising=False)
    monkeypatch.setattr(evaluate, "_is_navigator", lambda m: False, raising=False)
    monkeypatch.setattr(evaluate, "_sample_path_metrics", lambda *a, **k: None, raising=False)
    monkeypatch.setenv("PRISM_DISABLE_SPINE_TOOLS", "1")
    return monkeypatch


def _run():
    graph, tasks = _graph_and_tasks()
    samples = evaluate.construct_eval_samples_from_dict(graph, tasks, "g")
    evaluate.eval_model_single_graph(
        None, None, samples, include_edge_list=True, use_icl=False,
        permutation=None, client=_Client())
    return tasks


def test_each_task_sees_its_own_robot_location_and_no_history(patched):
    tasks = _run()
    assert len(SEEN) == 3
    want = ["office_1", "corridor_2", "hall_2"]  # keep -> graph's; else init_node
    assert [s["location"] for s in SEEN] == want
    assert [s["prompt_robot_location"] for s in SEEN] == want
    for s, t in zip(SEEN, tasks):
        assert s["base_request"] == f"task: {t['task']}"
        assert s["n_history"] == 0


def test_shared_episode_flag_reproduces_the_old_bug(patched):
    patched.setenv("PRISM_EVAL_SHARED_EPISODE", "1")
    _run()
    # Old loop: task 0's location and request stick to every later task.
    assert [s["location"] for s in SEEN] == ["office_1"] * 3
    assert [s["n_history"] for s in SEEN] == [0, 1, 2]
