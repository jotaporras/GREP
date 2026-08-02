"""Tests for the ``spine_tools`` / ``icl_examples`` prompt policies.

Two switches, both required everywhere (no library-level defaults), mirroring the
``text_edge_list`` -> ``include_edges`` policy tested in ``tests/test_text_edge_list.py``:

* ``include_tools`` — SPINE tool calling. True inserts the SPINE API tutorial into the
  compact system prompt (including the ratify-before-answer rule that exists to prevent
  hallucinated edges) and keeps each assistant plan as a SPINE action list; False adds
  no tool text at all — that arm is the pre-SPINE prompt (intro + answer contract) with
  a bare arrow route. It must agree with the eval simulator:
  ``evaluate._spine_tools_disabled`` selects ``_NoToolsGraphSim`` AND is the flag
  threaded into the inference clients.
* ``icl_examples`` — how many leading SPINE few-shot examples survive into the compact
  prompt (0 = none, -1 = all). When any survive, the query graph must NOT be hoisted
  into the system message: it stays inline at the head of the query user turn so it
  remains the LAST ``Scene graph: •`` block, which is what ``find_last_graph_scope``
  scopes PE injection to.

Pure-text tests import ``compact_prompt`` (torch-free); wiring is checked by parsing
source with ``ast``, like the sibling policy test.
"""
import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

from prism.data import compact_prompt  # noqa: E402  (torch-free; safe to import)

COMPACT_PY = SRC / "prism/data/compact_prompt.py"
DATA_PY = SRC / "prism/data/data.py"
INFERENCE_PY = SRC / "prism/models/inference.py"
EVALUATE_PY = SRC / "prism/eval/evaluate.py"


# A minimal SPINE conversation: system + two ICL examples (each with its own graph and
# a tool-calling plan) + the query graph turn + its answer. Same shape as a logged
# rollout, small enough to assert on exactly.
def _graph(prefix: str) -> str:
    return (
        "{'objects': [{'name': '%s_obj_1', 'coords': [0, 0]}], "
        "'regions': [{'name': '%s_a', 'coords': [0, 0]}, {'name': '%s_b', 'coords': [1, 0]}], "
        "'object_connections': [['%s_obj_1', '%s_b']], "
        "'region_connections': [['%s_a', '%s_b']], "
        "'robot_location': '%s_a'}" % ((prefix,) * 8)
    )


def _answer(relevant: str, plan: str) -> str:
    return (
        '{"primary_goal": "g", "relevant_graph": "%s", "reasoning": "r", "plan": "%s"}'
        % (relevant, plan)
    )


SPINE_CONVO = [
    {"role": "system", "content": "VERBOSE SPINE SYSTEM PROMPT"},
    {"role": "user", "content": f"task: icl one\nScene graph:{_graph('one')}"},
    {"role": "assistant", "content": _answer("one_a", "[map_region(one_a), goto(one_b)]")},
    {"role": "user", "content": "updates:[no_updates()]"},
    {"role": "assistant", "content": _answer("one_b", "[answer(one_a -> one_b)]")},
    {"role": "user", "content": f"task: icl two\nScene graph:{_graph('two')}"},
    {"role": "assistant", "content": _answer("two_a", "[answer(two_a -> two_b)]")},
    {"role": "user", "content": f"reach the object\nAdvice: \n- x\n\nScene graph:{_graph('qry')}"},
    {"role": "assistant", "content": _answer("qry_a", "[map_region(qry_a), answer(qry_a -> qry_b)]")},
]


def _compact(include_tools: bool, icl_examples: int, include_edges: bool = False):
    return compact_prompt.spine_to_compact_messages(
        SPINE_CONVO, include_edges=include_edges,
        include_tools=include_tools, icl_examples=icl_examples)


def _system(messages) -> str:
    systems = [m["content"] for m in messages if m["role"] == "system"]
    assert len(systems) == 1, f"expected exactly one system message, got {len(systems)}"
    return systems[0]


def _plans(messages):
    return [m["content"].split("</think>", 1)[-1].strip()
            for m in messages if m["role"] == "assistant"]


# ---------------------------------------------------------------------------
# include_tools — system prompt
# ---------------------------------------------------------------------------

def test_tools_on_documents_api_and_demands_ratification():
    """With tools on the prompt names every SPINE action, says when they must be used,
    and requires ratifying hops with map_region before answer() (no hallucinated edges)."""
    sys_on = _system(_compact(include_tools=True, icl_examples=0))
    for action in ("goto(", "map_region(", "explore_region(", "extend_map(", "inspect(", "answer("):
        assert action in sys_on, f"SPINE API section missing {action}"
    assert "Ratify the path before committing to it" in sys_on
    assert "hallucinated edge" in sys_on
    assert "TOOL CALLING IS DISABLED" not in sys_on
    # The plan format asked for is the action list, not a bare route.
    assert "SPINE action list" in sys_on


def test_tools_off_is_the_pre_spine_prompt_plus_the_latent_note():
    """With tools off the prompt says NOTHING about tools — that arm is the pre-SPINE
    original (intro + answer contract). The one deliberate difference is the expanded
    latent-space note, so the contract must still be the historical text verbatim."""
    sys_off = _system(_compact(include_tools=False, icl_examples=0))
    for tool_text in ("SPINE tool API", "TOOL CALLING IS DISABLED", "map_region",
                      "Ratify the path", "NEW CAPABILITY", "tool call"):
        assert tool_text not in sys_off, f"tool-free prompt mentions {tool_text!r}"
    # Pre-SPINE contract, verbatim (bare arrow route, not an action list).
    assert "give the final plan only: the route the robot follows" in sys_off
    assert "SPINE action list" not in sys_off
    # Bolstered latent note.
    assert "available to you in latent space" in sys_off
    assert "The edges are NOT missing" in sys_off
    assert "never ask for an edge list" in sys_off


def test_tools_off_plain_llm_points_at_listed_edges_and_never_claims_latent():
    """The plain-LLM arm (include_edges=True) has no latent pathway: its tool-free
    prompt is the edge-aware intro plus the historical contract, with no latent claim
    and no tool text."""
    sys_off = _system(_compact(include_tools=False, icl_examples=0, include_edges=True))
    assert "listed under Region Edges" in sys_off
    assert "paths using these listed edges" in sys_off
    assert "latent" not in sys_off
    assert "SPINE tool API" not in sys_off


def test_tools_off_matches_the_pre_spine_prompt_byte_for_byte():
    """Regression guard for the tool-free baseline: the plain-LLM tool-free prompt must
    equal the pre-SPINE constant exactly (that arm carries no latent note to change), and
    the graph-aug one must differ ONLY by the expanded latent paragraph — its contract
    tail must be identical."""
    import subprocess
    import types

    src = subprocess.run(["git", "show", "HEAD:src/prism/data/compact_prompt.py"],
                         capture_output=True, text=True, cwd=str(REPO_ROOT)).stdout
    if not src:
        return  # no git object available (shallow/exported tree) — skip silently
    head = types.ModuleType("head")
    exec(compile(src, "head", "exec"), head.__dict__)  # noqa: S102

    assert compact_prompt.compact_system_prompt(include_edges=True, include_tools=False) \
        == head.COMPACT_SYSTEM_PROMPT_WITH_EDGES, "plain-LLM tool-free prompt drifted"

    tail = lambda s: s[s.index("Answer in two parts"):]  # noqa: E731
    assert tail(compact_prompt.compact_system_prompt(include_edges=False, include_tools=False)) \
        == tail(head.COMPACT_SYSTEM_PROMPT), "tool-free answer contract drifted"


# ---------------------------------------------------------------------------
# include_tools — plan format, both directions across the seam
# ---------------------------------------------------------------------------

def test_tools_on_keeps_action_list_off_unwraps_to_route():
    """Targets match the contract: action list with tools on, bare route with tools off.

    Tools off relies on ``_unwrap_plan``, which only unwraps a plan that is exactly
    ``[answer(...)]``; a MIXED plan keeps its actions (minus the brackets), which is the
    documented historical behavior and the one the tool-free baseline was trained on.
    Such rollouts therefore still show action text in a no-tool-calls prompt — a known
    inconsistency of the corpus, not of the flag.
    """
    on = _plans(_compact(include_tools=True, icl_examples=0))[-1]
    off = _plans(_compact(include_tools=False, icl_examples=0))[-1]
    assert on == "[map_region(qry_a), answer(qry_a -> qry_b)]"
    assert off == "map_region(qry_a), answer(qry_a -> qry_b)"

    # Pure-answer plans (the nav100 norm) do unwrap to the bare route.
    icl = _compact(include_tools=False, icl_examples=1)
    assert "one_a -> one_b" in _plans(icl)
    assert "[answer(one_a -> one_b)]" not in _plans(icl)


def test_unterminated_think_block_yields_an_empty_plan():
    """A generation cut off before ``</think>`` committed to no plan, so the plan must be
    EMPTY — never the reasoning wrapped in answer(), which would let the answer-key regex
    score a truncated sample correct off text it merely mentioned."""
    import json

    truncated = ("<think>Relevant graph: hub_1, observatory_1\n\nReasoning: hub_1 connects to "
                 "hallway_1, and hallway_1 connects to observatory_1, so the route is "
                 "hub_1 -> hallway_1 -> observatory_1 and I should")
    out = json.loads(compact_prompt.compact_output_to_spine_json(truncated))
    assert out["plan"] == "", f"truncated generation produced a plan: {out['plan']!r}"
    # The reasoning survives for diagnosis, and the route text inside it is NOT promoted.
    assert "hallway_1" in out["reasoning"]
    assert "answer(" not in out["plan"]

    # A closed block with nothing after it is equally empty — no plan was written.
    empty = json.loads(compact_prompt.compact_output_to_spine_json(
        "<think>Relevant graph: a\n\nReasoning: r</think>   "))
    assert empty["plan"] == ""

    # A model that never opens a think block still has its whole output taken as the plan.
    plain = json.loads(compact_prompt.compact_output_to_spine_json("a -> b"))
    assert plain["plan"] == "[answer(a -> b)]"


def test_inverse_passes_action_lists_through_and_wraps_bare_routes():
    """``compact_output_to_spine_json`` must not bury an action list inside answer() —
    that would stop the planning loop ever executing a tool — but must still wrap a bare
    route so the SPINE grader sees a valid command."""
    import json

    action = json.loads(compact_output_to_spine_json_text(
        "<think>Relevant graph: a\n\nReasoning: r</think>[map_region(a), answer(a -> b)]"))
    assert action["plan"] == "[map_region(a), answer(a -> b)]"

    route = json.loads(compact_output_to_spine_json_text(
        "<think>Relevant graph: a\n\nReasoning: r</think>a -> b"))
    assert route["plan"] == "[answer(a -> b)]"

    # Bracketed prose is NOT an action list: it must still be wrapped.
    prose = json.loads(compact_output_to_spine_json_text("<think>r</think>[a -> b]"))
    assert prose["plan"] == "[answer([a -> b])]"


def compact_output_to_spine_json_text(text: str) -> str:
    return compact_prompt.compact_output_to_spine_json(text)


# ---------------------------------------------------------------------------
# icl_examples — how many survive, and where the graphs live
# ---------------------------------------------------------------------------

def test_icl_zero_drops_examples_and_hoists_query_graph():
    """icl_examples=0 is the historical layout: ICL gone, one system message carrying
    the compact prompt AND the query scene graph."""
    out = _compact(include_tools=True, icl_examples=0)
    sys_content = _system(out)
    assert "Scene graph:" in sys_content
    assert sum(m["content"].count("Scene graph:") for m in out) == 1, "an ICL graph leaked"
    assert "one_a" not in str(out) and "two_a" not in str(out), "ICL example content leaked"


def test_icl_keeps_leading_examples_in_order():
    """icl_examples=N keeps the FIRST N examples (SPINE header order EXAMPLE_1, EXAMPLE_2…)
    with their planning turns, then the query; -1 keeps every one."""
    one = _compact(include_tools=True, icl_examples=1)
    assert "one_a" in str(one) and "two_a" not in str(one), "kept the wrong example"
    # EXAMPLE_1's receding-horizon turns travel with it (task, answer, updates, replan).
    assert any(m["content"].startswith("updates:") for m in one), "ICL planning turns dropped"

    both = _compact(include_tools=True, icl_examples=-1)
    assert "one_a" in str(both) and "two_a" in str(both)
    # Asking for more examples than exist is capped, not an error.
    assert _compact(include_tools=True, icl_examples=99) == both


def test_icl_layout_keeps_query_graph_as_the_last_block():
    """With ICL the system message is prompt-only and every graph stays inline in its own
    user turn — the query's LAST, since find_last_graph_scope anchors PE injection there."""
    out = _compact(include_tools=True, icl_examples=-1)
    assert "Scene graph:" not in _system(out), "graph hoisted despite ICL — injection would misscope"
    blocks = [i for i, m in enumerate(out) if "Scene graph:" in m["content"]]
    assert len(blocks) == 3, f"expected 2 demo graphs + query graph, got {len(blocks)}"
    assert "qry_a" in out[blocks[-1]]["content"], "query graph is not the last block"
    # The block precedes the task text inside the turn, so task/answer node mentions stay
    # inside the injection scope.
    assert out[blocks[-1]]["content"].startswith("Scene graph:")
    assert out[blocks[-1]]["content"].rstrip().endswith("reach the object")


def test_verbose_spine_system_prompt_always_dropped():
    """Whatever the policy, SPINE's own long system prompt never reaches the model."""
    for tools in (True, False):
        for icl in (0, 1, -1):
            assert "VERBOSE SPINE SYSTEM PROMPT" not in str(_compact(tools, icl))


# ---------------------------------------------------------------------------
# Constructed conversations: build_conversation / icl_demos_from_rollouts
# ---------------------------------------------------------------------------

def _demo_rollout(prefix: str):
    """A logged-rollout shape (ICL prefix + real task) for icl_demos_from_rollouts."""
    return [
        {"role": "system", "content": "VERBOSE SPINE SYSTEM PROMPT"},
        {"role": "user", "content": "task: canned icl\nScene graph:" + _graph("canned")},
        {"role": "assistant", "content": _answer("canned_a", "[answer(canned_a -> canned_b)]")},
        {"role": "user", "content": f"task: demo task\nScene graph:{_graph(prefix)}"},
        {"role": "assistant", "content": _answer(f"{prefix}_a", f"[goto({prefix}_b), answer({prefix}_a -> {prefix}_b)]")},
    ]


def test_icl_demos_from_rollouts_merge_by_graph_and_render_before_the_query():
    """Demos drawn from real rollouts: rollouts on the same graph merge into one demo, and
    the rendered conversation puts demo graphs first with the query graph last."""
    demos = compact_prompt.icl_demos_from_rollouts(
        [_demo_rollout("demo"), _demo_rollout("demo"), _demo_rollout("other")],
        include_tools=True)
    assert len(demos) == 2, "rollouts sharing a graph must merge into one demo"
    assert len(demos[0]["turns"]) == 2, "both tasks of the shared graph must stack"
    assert demos[0]["turns"][0]["assistant"].endswith("[goto(demo_b), answer(demo_a -> demo_b)]")

    query_graph = compact_prompt._extract_scene_graph_dict("Scene graph:" + _graph("qry"))
    msgs = compact_prompt.format_eval_messages(
        query_graph, "reach it", include_edges=False, include_tools=True, icl_demos=demos)
    assert "Scene graph:" not in msgs[0]["content"], "system message must be prompt-only with ICL"
    blocks = [i for i, m in enumerate(msgs) if "Scene graph:" in m["content"]]
    assert blocks[-1] == len(msgs) - 1, "query graph must open the final turn"
    assert "qry_a" in msgs[-1]["content"]


def test_build_conversation_without_demos_is_unchanged():
    """No demos = the historical layout: graph hoisted into the system message, tasks bare."""
    graph = compact_prompt._extract_scene_graph_dict("Scene graph:" + _graph("qry"))
    msgs = compact_prompt.build_conversation(
        graph, [{"task": "t1"}, {"task": "t2"}], include_edges=False, include_tools=False)
    assert [m["role"] for m in msgs] == ["system", "user", "user"]
    assert "Scene graph:" in msgs[0]["content"]
    assert msgs[1]["content"] == "t1" and msgs[2]["content"] == "t2"


# ---------------------------------------------------------------------------
# Wiring (torch-free ast checks)
# ---------------------------------------------------------------------------

def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _find_func(tree, name: str, cls: str = None):
    if cls is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == cls:
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef) and sub.name == name:
                        return sub
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _param_default_status(func, argname: str) -> str:
    a = func.args
    first_default = len(a.args) - len(a.defaults)
    for i, arg in enumerate(a.args):
        if arg.arg == argname:
            return "has_default" if i >= first_default else "no_default"
    for arg, default in zip(a.kwonlyargs, a.kw_defaults):
        if arg.arg == argname:
            return "has_default" if default is not None else "no_default"
    return "absent"


def _calls_to(scope, name: str):
    def callee(call):
        f = call.func
        return f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
    return [n for n in ast.walk(scope) if isinstance(n, ast.Call) and callee(n) == name]


def _kw(call, key: str):
    for k in call.keywords:
        if k.arg == key:
            return k.value
    return None


def test_no_defaults_on_the_policy_flags_in_compact_prompt():
    """No function in ``compact_prompt`` may default ``include_tools`` or ``icl_examples``
    — the caller must state the policy, exactly as for ``include_edges``."""
    tree = _parse(COMPACT_PY)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for flag in ("include_tools", "icl_examples"):
                assert _param_default_status(node, flag) != "has_default", (
                    f"compact_prompt.{node.name} defaults {flag} (no library defaults allowed)")
    for fn in ("build_conversation", "spine_to_compact_messages", "_system_content"):
        func = _find_func(tree, fn)
        assert _param_default_status(func, "include_tools") == "no_default", \
            f"compact_prompt.{fn} must take include_tools with no default"
    assert _param_default_status(
        _find_func(tree, "spine_to_compact_messages"), "icl_examples") == "no_default"


def test_spine_mode_quadruples_the_generation_budget():
    """SPINE plans cost far more output (tutorial + ratification + action list), and a
    generation cut short now yields an empty plan — so the client must resolve
    ``max_new_tokens`` to 4x the tool-free budget when tools are on."""
    tree = _parse(INFERENCE_PY)
    src = INFERENCE_PY.read_text()
    assert "MAX_NEW_TOKENS" in src and "SPINE_TOKEN_MULTIPLIER" in src, \
        "the budget and its SPINE multiplier must be named constants"
    ns: dict = {}
    for node in tree.body:  # module-level constants only; no torch import needed
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    ns[t.id] = node.value.value
    assert ns.get("SPINE_TOKEN_MULTIPLIER") == 4, \
        f"SPINE budget multiplier must be 4, got {ns.get('SPINE_TOKEN_MULTIPLIER')}"
    init = _find_func(tree, "__init__", "InMemoryLLM")
    assigns = [n for n in ast.walk(init) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Attribute) and t.attr == "max_new_tokens"
                       for t in n.targets)]
    assert assigns, "InMemoryLLM.__init__ must resolve self.max_new_tokens"
    dumped = ast.dump(assigns[0].value)
    assert "SPINE_TOKEN_MULTIPLIER" in dumped and "include_tools" in dumped, \
        "the budget must scale by the multiplier under include_tools"
    for cls in ("InMemoryLLM", "GraphAugmentedInMemoryLLM"):
        q = _find_func(tree, "query_llm", cls)
        assert _param_default_status(q, "max_new_tokens") == "has_default", \
            f"{cls}.query_llm must keep an overridable max_new_tokens"
        assert any(isinstance(n, ast.Attribute) and n.attr == "max_new_tokens"
                   for n in ast.walk(q)), \
            f"{cls}.query_llm must fall back to self.max_new_tokens"


def test_inference_clients_thread_both_flags():
    """Both clients take the flags with no default, store them, and pass them into the
    formatter as ``self.*`` (never hardcoded)."""
    tree = _parse(INFERENCE_PY)
    for cls in ("InMemoryLLM", "GraphAugmentedInMemoryLLM"):
        init = _find_func(tree, "__init__", cls)
        for flag in ("include_tools", "icl_examples"):
            assert _param_default_status(init, flag) == "no_default", \
                f"{cls}.__init__ must take {flag} with no default"
        q = _find_func(tree, "query_llm", cls)
        calls = _calls_to(q, "spine_to_compact_messages")
        assert calls, f"{cls}.query_llm does not call spine_to_compact_messages"
        for call in calls:
            for flag in ("include_tools", "icl_examples"):
                v = _kw(call, flag)
                assert isinstance(v, ast.Attribute) and v.attr == flag \
                    and isinstance(v.value, ast.Name) and v.value.id == "self", \
                    f"{cls}.query_llm must pass {flag}=self.{flag}"


def test_evaluate_derives_tool_policy_from_the_simulator_switch():
    """The prompt's tool policy and the simulator must come from ONE flag: eval builds the
    clients with ``include_tools = not _spine_tools_disabled()`` and picks
    ``_NoToolsGraphSim`` from the same call."""
    tree = _parse(EVALUATE_PY)
    func = _find_func(tree, "eval_model_single_graph")
    assigns = [n for n in ast.walk(func) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "include_tools" for t in n.targets)]
    assert assigns, "eval_model_single_graph must resolve include_tools"
    v = assigns[0].value
    assert isinstance(v, ast.UnaryOp) and isinstance(v.op, ast.Not) \
        and _calls_to(v, "_spine_tools_disabled"), \
        "include_tools must be `not _spine_tools_disabled()`"
    src = EVALUATE_PY.read_text()
    assert "_NoToolsGraphSim if _spine_tools_disabled()" in src, \
        "the simulator must read the same switch"
    for client in ("InMemoryLLM", "GraphAugmentedInMemoryLLM"):
        for call in _calls_to(func, client):
            assert isinstance(_kw(call, "include_tools"), ast.Name), \
                f"{client} must be built with the resolved include_tools"
            assert isinstance(_kw(call, "icl_examples"), ast.Name), \
                f"{client} must be built with the resolved icl_examples"


def test_data_preprocess_resolves_both_policies_from_config():
    """``preprocess_dataset`` takes ``spine_tools`` / ``icl_examples``, resolves
    ``include_tools = (spine_tools == 'present')`` and passes both to the formatter."""
    tree = _parse(DATA_PY)
    func = _find_func(tree, "preprocess_dataset")
    params = {a.arg for a in func.args.args}
    assert {"spine_tools", "icl_examples"} <= params, \
        "preprocess_dataset must take spine_tools and icl_examples"
    found = False
    for node in ast.walk(func):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "include_tools" for t in node.targets):
            v = node.value
            assert isinstance(v, ast.Compare) and isinstance(v.left, ast.Name) \
                and v.left.id == "spine_tools" and v.comparators[0].value == "present", \
                "include_tools must be (spine_tools == 'present')"
            found = True
    assert found, "no include_tools resolution in preprocess_dataset"
    for call in _calls_to(func, "spine_to_compact_messages"):
        assert isinstance(_kw(call, "include_tools"), ast.Name)
        assert isinstance(_kw(call, "icl_examples"), ast.Name)


def test_assistant_loss_positions_are_clamped_to_the_query_scope():
    """``assistant_idx`` (loss_target='responses') must be filtered to positions at/after
    the query graph block. Without the clamp, a few-shot prompt would supervise the ICL
    demos' assistant turns — training the model to reproduce SPINE's canned answers."""
    tree = _parse(DATA_PY)
    func = _find_func(tree, "preprocess_dataset")
    assign = [n for n in ast.walk(func) if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Subscript)
                      and isinstance(t.slice, ast.Constant)
                      and t.slice.value == "assistant_idx" for t in n.targets)]
    assert assign, "preprocess_dataset must set example['assistant_idx']"
    v = assign[0].value
    assert isinstance(v, ast.ListComp), \
        "assistant_idx must be a comprehension filtering on the graph scope"
    src = ast.dump(v)
    assert "scope_start" in src and _calls_to(v, "assistant_token_positions"), \
        "assistant_idx must filter assistant_token_positions by scope_start"


if __name__ == "__main__":
    failures = []
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                failures.append((name, e))
                print(f"FAIL {name}: {e}")
    raise SystemExit(1 if failures else 0)
