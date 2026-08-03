"""Demo: the compact plain-text prompt format fed to the GREP-PRISM models.

Shows what training and eval prompts become once the verbose SPINE JSON (long system
prompt + few-shot ICL examples + full scene-graph JSON) is translated to the compact
format the LLM consumes, and what the model's answer becomes on the way back out.

THE THREE POLICY SWITCHES, which every section below is one setting of:

  * ``include_edges`` — the scene graph is always a compact ``Scene graph:`` block
    (bulleted node-name lists + robot location). Graph-augmented archs OMIT the edges
    (the GNN supplies connectivity) and get the latent-connectivity intro, written at
    length on purpose: the graph channel is the premise of the architecture, so the
    prompt tells the model in several ways that the edges really are present. The
    plain-LLM baseline (``include_edges=True``) instead lists ``• Region Edges:`` /
    ``• Object Edges:`` and gets an edge-aware intro pointing at them.
  * ``include_tools`` — SPINE tool calling. ``False`` adds NOTHING: that arm is the
    pre-SPINE prompt exactly (intro + answer contract, no tool text), so the tool-free
    baseline never drifts; its plan is a bare ``a -> b`` route. ``True`` inserts the
    SPINE tutorial (8-action planning API, 7-function inbound update API, argument
    rules, plus a plannability gate / ``replan()`` trigger / substitution ban added
    after zero-shot probing showed ``clarify`` and ``replan`` never fired) and switches
    the contract's part 2 to an action list.
  * ``icl_examples`` — how many leading SPINE few-shot examples survive, compacted.
    With ICL the query graph is NOT hoisted into the system message: it opens the query
    ``user`` turn so it stays the LAST graph block and PE injection
    (``find_last_graph_scope``) still scopes to the query graph rather than a demo's.

Tasks stack as ``user``/``assistant`` pairs in one conversation per graph; the assistant
target wraps reasoning in ``<think>Relevant graph: …\\n\\nReasoning: …</think>`` followed
by the plan alone.

Coming back out, ``compact_output_to_spine_json`` maps model text to the SPINE JSON the
grader expects: a bare route is wrapped as ``[answer(route)]``, an action list passes
through untouched (wrapping it would bury the actions inside one ``answer()`` and no tool
would ever execute), and an unterminated ``<think>`` yields an EMPTY plan — never the
reasoning wrapped in ``answer(...)``, which is how a truncated generation used to score
"correct" off node names its prose happened to contain.

TRAINING and DEPLOYMENT run different policies and both are shown. The shipped config
TRAINS tool-free and zero-shot and DEPLOYS with the SPINE API live, still zero-shot; the
seam absorbs the gap, since the model emits the bare route it was trained on and the
inverse translator wraps it. SPINE mode also gets ``SPINE_TOKEN_MULTIPLIER``x the
tool-free generation budget.

PRESENTATION. One canonical prompt is printed in full, byte-for-byte, as the centrepiece;
every other section shows only its delta or a framed excerpt with a counted
``… N lines elided …`` marker. ``--full`` disables all elision, so the byte-exact artifact
is always reachable. Nothing printed is reflowed or hand-edited — text either comes from
the live translator verbatim or is a counted elision of it. Correctness checks all remain
in the code but report as one ``invariants: N/N passed`` line, with detail only on failure.

DATA. Everything is drawn live from the corpus named in the config
(``experiments/demo_compact_prompt.yaml``), selected deterministically by a live census —
no RNG, no dates, reproducible run to run. The one exception is labeled:
``RECORDED_SPINE_ROLLOUT``, a real two-turn tool-calling exchange captured from an eval
run, kept because the logged corpus contains no ``updates:`` round-trip at all (the demo
measures this rather than assuming it).

The ``User:``/``Assistant:`` delimiters come from the tokenizer's chat template, NOT
literal text — with ``--tokenizer`` the demo renders the byte-exact
``apply_chat_template`` output; without it, an illustrative ``Role:``-labelled view.

Pure pre-processing; does NOT modify anything under data/, and imports nothing heavier
than ``compact_prompt`` (torch-free) unless ``--tokenizer`` is given.

Run:  uv run python scripts/demo_compact_prompt.py
      uv run python scripts/demo_compact_prompt.py --full
      uv run python scripts/demo_compact_prompt.py --tokenizer meta-llama/Llama-3.2-3B-Instruct
      uv run python scripts/demo_compact_prompt.py --spread 8 --canonical 12
      uv run python scripts/demo_compact_prompt.py --train-spine-tools --icl-examples 2
"""

import argparse
import ast
import json
import sys
import textwrap
from pathlib import Path

from omegaconf import OmegaConf

from prism.data.compact_prompt import (
    _SPINE_ACTIONS,
    _SPINE_TOOLS_SECTION,
    append_followup_task,
    assemble_training_conversation,
    compact_output_to_spine_json,
    compact_system_prompt,
    format_eval_messages,
    format_training_messages,
    icl_demos_from_rollouts,
    render,
    spine_to_compact_messages,
    try_load_json,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "demo_compact_prompt.yaml"

# A REAL two-turn SPINE rollout, captured verbatim from a live eval run so the tool
# section can show tool calls actually working. The logged training corpus answers
# directly ([answer(...)]), so replaying it would show a SPINE section containing no
# SPINE calls; this is the same graph (data_gen_000), just a task where the model chose
# to ratify before committing.
#
# `reasoning` / `plan` are the model's own text, unedited. `updates` is what GraphSim
# returned after executing the turn's first action. The demo re-wraps these into SPINE's
# logged 4-key JSON shape and runs them through the LIVE translator, so the printed
# conversation is produced by the pipeline rather than written out here.
#
# The section RENDERS this at ``include_edges=False`` — the GT+LLM arch, where the scene
# graph block carries node names only and connectivity arrives through the graph channel.
# The capture itself came from the edge-list arm, because a trained GT checkpoint could
# not be run on the capture machine (see the provenance string). What is being shown is
# the tool-call mechanism and the prompt shape, which are identical either way; the one
# thing NOT demonstrated is where the model's adjacency knowledge came from.
RECORDED_SPINE_ROLLOUT = {
    "model": "google/gemma-4-E2B-it",
    "provenance": ("live eval run, MPS bf16, zero-shot, tools ON, graph data_gen_000 task 4; "
                   "captured on the plain-LLM arm (edges in text, no GNN) and rendered here "
                   "as the GT+LLM arm reads it (no edge list)"),
    "task": "Route the robot from the starting area to the battery bank and list the path.",
    "turns": [
        {
            "relevant_graph": "hub_1, battery_bank_1",
            "reasoning": (
                "The robot starts at hub_1 and needs to reach battery_bank_1. I need to find a "
                "path through the connected regions.\n"
                "1. hub_1 is connected to mess_hall_1, corridor_1, lobby_1, lounge_1, hallway_1, "
                "archive_1.\n"
                "2. battery_bank_1 is connected to generator_room_1 and maintenance_bay_1.\n"
                "3. I need to find a path from the hub_1 cluster to the battery_bank_1 cluster.\n"
                "4. lobby_1 is connected to office_1 and clean_room_1.\n"
                "5. office_1 is connected to power_plant_1.\n"
                "6. power_plant_1 is connected to generator_room_1.\n"
                "7. generator_room_1 is connected to battery_bank_1.\n"
                "8. A potential path is hub_1 -> lobby_1 -> office_1 -> power_plant_1 -> "
                "generator_room_1 -> battery_bank_1. I must confirm these edges via map_region "
                "calls if they are not explicitly known. I will start by mapping the starting "
                "region to confirm immediate neighbors."),
            "plan": "[map_region(hub_1)]",
            "note": "ratifies the start region before committing to any hop",
            "updates": "update_node_attributes({name: hub_1, description: })",
        },
        {
            "relevant_graph": "hub_1, battery_bank_1",
            "reasoning": (
                "I have mapped hub_1. I need to re-evaluate the path to battery_bank_1. The "
                "previous plan was to go through lobby_1 -> office_1 -> power_plant_1 -> "
                "generator_room_1 -> battery_bank_1. I will now attempt to navigate directly to "
                "the goal region, battery_bank_1, relying on the graph search to find the path "
                "through the confirmed or existing edges."),
            "plan": ("[goto(battery_bank_1), answer(hub_1 -> lobby_1 -> office_1 -> "
                     "power_plant_1 -> generator_room_1 -> battery_bank_1)]"),
            "note": "hop confirmed, so it commits: goto the goal, then the terminal answer()",
            "updates": None,
        },
    ],
    # From path_validator on that run's final route.
    "metrics": {
        "structured_correct": True, "edge_validity_rate": 1.0, "hallucination_rate": 0.0,
        "full_path_valid": True, "hop_optimality": 2.5,
        "parsed_nodes": ["hub_1", "lobby_1", "office_1", "power_plant_1",
                         "generator_room_1", "battery_bank_1"],
    },
}


# --------------------------------------------------------------------------- presentation
class Style:
    """Subtle ANSI (dim/bold only), auto-off when stdout is not a TTY."""

    def __init__(self, enabled: bool):
        self.on = enabled

    def dim(self, s: str) -> str:
        return f"\033[2m{s}\033[0m" if self.on else s

    def bold(self, s: str) -> str:
        return f"\033[1m{s}\033[0m" if self.on else s


class Invariants:
    """Correctness checks stay in the code; they report as one line, detail on failure.

    Every check that used to narrate itself to stdout runs here instead. Failure is still
    loud — `report` raises — so silencing the prose costs no guarantee.
    """

    def __init__(self):
        self.passed = 0
        self.failures = []

    def __call__(self, ok, msg: str) -> bool:
        if ok:
            self.passed += 1
        else:
            self.failures.append(msg)
        return bool(ok)

    def report(self, style: Style) -> None:
        total = self.passed + len(self.failures)
        if self.failures:
            print(style.bold(f"\ninvariants: {self.passed}/{total} passed — FAILED:"))
            for f in self.failures:
                print(f"  x {f}")
            raise AssertionError(f"{len(self.failures)} invariant(s) failed")
        print(style.dim(f"\ninvariants: {self.passed}/{total} passed"))


class Out:
    """Rules, tables and counted elision at one consistent width."""

    def __init__(self, style: Style, width: int, head: int, tail: int, full: bool):
        self.s, self.w, self.head, self.tail, self.full = style, width, head, tail, full

    def rule(self, title: str) -> None:
        print(f"\n{self.s.dim('-' * self.w)}\n{self.s.bold(title)}")

    def note(self, text: str = "", indent: str = "  ") -> None:
        """Commentary, wrapped to the rule width.

        Notes are this script's own prose, never prompt or model bytes, so wrapping them
        is free — the no-reflow rule applies to `block`, which prints the real artifacts.
        """
        if not text:
            print()
            return
        for ln in textwrap.wrap(text, width=self.w - len(indent)) or [""]:
            print(self.s.dim(f"{indent}{ln}"))

    def table(self, headers, rows, align: str = "", indent: str = "  ", maxw=None) -> None:
        rows = [[str(c) for c in r] for r in rows]
        if not rows:
            return
        align = (align or "l" * len(headers)).ljust(len(headers), "l")
        if maxw:
            rows = [[c if len(c) <= maxw[i] else c[:maxw[i] - 1] + "…"
                     for i, c in enumerate(r)] for r in rows]
        w = [max(len(str(headers[i])), max(len(r[i]) for r in rows))
             for i in range(len(headers))]

        def line(cells):
            return indent + "  ".join(
                c.rjust(w[i]) if align[i] == "r" else c.ljust(w[i])
                for i, c in enumerate(cells)).rstrip()

        print(self.s.dim(line(headers)))
        print(self.s.dim(indent + "  ".join("-" * x for x in w)))
        for r in rows:
            print(line(r))

    def block(self, text: str, indent: str = "    ", elide: bool = True,
              clip: bool = True) -> None:
        """Print text verbatim, with counted markers wherever it is shortened.

        Two independent budgets, because prompt text is long in both directions: `elide`
        drops whole LINES from the middle, `clip` trims each line to the rule width. Both
        only ever remove, and both say exactly how much they removed, so what remains is
        byte-accurate. --full disables the pair.
        """
        lines = text.split("\n")
        if clip and not self.full:
            limit = max(24, self.w - len(indent))
            lines = [ln if len(ln) <= limit
                     else ln[:limit] + self.s.dim(f"… +{len(ln) - limit:,} chars")
                     for ln in lines]
        if elide and not self.full and len(lines) > self.head + self.tail + 1:
            n = len(lines) - self.head - self.tail
            lines = (lines[:self.head]
                     + [self.s.dim(f"… {n} lines elided (--full prints all) …")]
                     + lines[-self.tail:])
        for ln in lines:
            print(f"{indent}{ln}")


# --------------------------------------------------------------------------- corpus census
def _minmedmax(values):
    v = sorted(values)
    return (v[0], v[len(v) // 2], v[-1]) if v else (0, 0, 0)


def _graph_stats(payload) -> dict:
    g = payload.get("graph", {}) or {}
    return {"regions": len(g.get("regions") or {}), "objects": len(g.get("objects") or {}),
            "edges": len(g.get("region_connections") or []),
            "tasks": len(payload.get("tasks") or [])}


def _census(cfg, inv: Invariants):
    """Measure the corpus live: which graphs are populated, and which targets call tools.

    Everything the selection later claims comes from these numbers, never from a constant —
    populated_graphs/ also holds 50 empty skeletons that must not be selected, and the
    tool-call rate is the kind of thing that silently changes when the corpus is regenerated.
    """
    root = REPO_ROOT / cfg.corpus.root
    gdir, pdir = root / cfg.corpus.graphs_subdir, root / cfg.corpus.plans_subdir
    inv(gdir.is_dir() and pdir.is_dir(), f"corpus dirs missing under {root}")

    # Glob EVERY graph file, not just the configured populated pattern, so that excluding
    # the empty skeletons is a measured decision (0 regions / 0 tasks / no rollouts) rather
    # than a naming assumption this demo would silently inherit.
    graphs = []
    for p in sorted(gdir.glob("*.json")):
        st = _graph_stats(try_load_json(p))
        idx = p.stem.rsplit("_", 1)[-1]
        # Rollouts are keyed by index alone, and BOTH families run 000-049 — so a rollout
        # may only be attributed to a graph that actually has the tasks it answers,
        # otherwise every sample file would be counted once per family.
        st.update(name=p.stem, idx=idx, path=p, family=p.stem.rsplit("_", 1)[0],
                  plans=sorted(pdir.glob(cfg.corpus.plan_glob.format(idx=idx)))
                  if st["tasks"] else [])
        graphs.append(st)
    populated = [g for g in graphs if g["regions"] and g["tasks"] and g["plans"]]
    inv(bool(populated), "no populated graph with rollouts found")
    inv(all(g["path"].match(cfg.corpus.graph_glob) for g in populated),
        f"populated graphs no longer match corpus.graph_glob ({cfg.corpus.graph_glob})")

    totals = {"tool_targets": 0, "total": 0, "updates": 0, "multi_turn": 0}
    for g in populated:
        hits = 0
        for pp in g["plans"]:
            c = try_load_json(pp)
            if not isinstance(c, list):
                continue
            msgs = spine_to_compact_messages(c, include_edges=False, include_tools=True,
                                             icl_examples=0)
            targets = [m["content"].split("</think>", 1)[-1].strip()
                       for m in msgs if m["role"] == "assistant"]
            if not targets:
                continue
            totals["total"] += 1
            totals["multi_turn"] += len(targets) > 1
            totals["updates"] += any(m["role"] == "user"
                                     and m["content"].lstrip().startswith("updates")
                                     for m in msgs)
            if not targets[-1].startswith("[answer("):
                hits += 1
                totals["tool_targets"] += 1
        g["tool_targets"] = hits
    totals.update(all=graphs, populated=populated)
    return populated, totals


def _spread(populated, n: int):
    """Evenly spaced indices across the populated range — deterministic, no RNG, no dates."""
    n = max(1, min(int(n), len(populated)))
    if n == 1:
        return [populated[0]]
    last = len(populated) - 1
    return [populated[round(i * last / (n - 1))] for i in range(n)]


def _pick(populated, want: str, key, describe, default_reason: str):
    """Resolve a selection knob: a named measured policy, 'first', or an explicit index."""
    if want == "first":
        return populated[0], default_reason
    if key is not None and want != "first" and not want.isdigit():
        best = max(populated, key=key)
        return best, describe(best)
    hit = next((g for g in populated if g["idx"] == want.zfill(3)), populated[0])
    return hit, f"requested index {want}"


def _select(cfg, populated, inv: Invariants) -> dict:
    spread = _spread(populated, cfg.selection.spread)
    canonical, why = _pick(
        populated, str(cfg.selection.canonical),
        key=lambda g: (g["regions"] + g["objects"], -int(g["idx"])),
        describe=lambda g: (f"widest: {g['regions']} regions + {g['objects']} objects = "
                            f"{g['regions'] + g['objects']} nodes, {g['edges']} edges"),
        default_reason="first populated graph")
    tool, why_tool = _pick(
        populated, str(cfg.selection.tool_demo),
        key=lambda g: (g["tool_targets"], -int(g["idx"])),
        describe=lambda g: f"most tool-calling targets: {g['tool_targets']}/{len(g['plans'])}",
        default_reason="first populated graph")
    off = int(cfg.selection.icl_offset) % len(spread)
    icl = spread[off] if spread[off]["idx"] != canonical["idx"] else spread[(off + 1) % len(spread)]
    inv(icl["idx"] != canonical["idx"], "ICL demos must come from a different graph")
    return {"spread": spread, "canonical": canonical, "why": why,
            "tool": tool, "why_tool": why_tool, "icl": icl}


def _generation_budget(cfg):
    """``(base, multiplier)`` read out of models/inference.py by AST.

    Read from source rather than imported: importing that module would drag in torch +
    spine and cost this demo its torch-free property. Reading the real constants still
    beats restating them here, which would rot the moment they change.
    """
    tree = ast.parse((REPO_ROOT / cfg.budget_source).read_text())
    consts = {t.id: n.value.value
              for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
              for t in n.targets if isinstance(t, ast.Name)}
    return consts.get("MAX_NEW_TOKENS"), consts.get("SPINE_TOKEN_MULTIPLIER")


# --------------------------------------------------------------------------- helpers
def _roles(messages) -> str:
    return "".join(m["role"][0].upper() for m in messages)


def _counter(tokenizer):
    if tokenizer is not None:
        return "tokens", lambda s: len(tokenizer.encode(s, add_special_tokens=False))
    return "chars", len


def _plan_lines(messages):
    return [m["content"].split("</think>", 1)[-1].strip()
            for m in messages if m["role"] == "assistant"]


def _graph_blocks(messages):
    return [i for i, m in enumerate(messages) if "Scene graph:" in m["content"]]


def main() -> None:
    ap = argparse.ArgumentParser(description="Compact-prompt format demo.")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--tokenizer", type=str, default="",
                    help="render the byte-exact apply_chat_template output")
    ap.add_argument("--full", action="store_true", help="disable all elision; print verbatim")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--spread", type=int, default=None, help="how many graphs to sample")
    ap.add_argument("--canonical", type=str, default=None, help="widest | first | <index>")
    ap.add_argument("--tool-demo", type=str, default=None,
                    help="most_tool_calls | first | <index>")
    ap.add_argument("--task-index", type=int, default=0)
    ap.add_argument("--train-spine-tools", dest="train_tools", action="store_true", default=None)
    ap.add_argument("--train-icl-examples", type=int, default=None)
    ap.add_argument("--no-spine-tools", dest="deploy_tools", action="store_false", default=None)
    ap.add_argument("--icl-examples", type=int, default=None, help="deployed few-shot count")
    args = ap.parse_args()

    # The config is required, not optional: it carries the corpus paths and the selection
    # policy, and silently falling back to built-in defaults would reintroduce exactly the
    # hard-coded indices this script was refactored to remove.
    if not args.config.is_file():
        sys.exit(f"config not found: {args.config}\n"
                 f"expected the sidecar at {DEFAULT_CONFIG.relative_to(REPO_ROOT)}, "
                 f"or pass --config <path>.")
    cfg = OmegaConf.load(args.config)
    for key, val in (("selection.spread", args.spread), ("selection.canonical", args.canonical),
                     ("selection.tool_demo", args.tool_demo),
                     ("policy.train_tools", args.train_tools),
                     ("policy.train_icl_examples", args.train_icl_examples),
                     ("policy.deploy_tools", args.deploy_tools),
                     ("policy.deploy_icl_examples", args.icl_examples)):
        if val is not None:
            OmegaConf.update(cfg, key, val)

    style = Style((not args.no_color) and cfg.display.color != "off" and sys.stdout.isatty())
    out = Out(style, int(cfg.display.width), int(cfg.display.excerpt_head),
              int(cfg.display.excerpt_tail), args.full)
    inv = Invariants()

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    label, count = _counter(tokenizer)

    train_tools, train_icl = bool(cfg.policy.train_tools), int(cfg.policy.train_icl_examples)
    tools, icl = bool(cfg.policy.deploy_tools), int(cfg.policy.deploy_icl_examples)

    # ---------------------------------------------------------------- 1. corpus & selection
    populated, corpus = _census(cfg, inv)
    sel = _select(cfg, populated, inv)
    canonical, tool_g, icl_g = sel["canonical"], sel["tool"], sel["icl"]

    out.rule("1  CORPUS & SELECTION")
    out.note(f"{cfg.corpus.root} — census run live; every choice below is measured, not fixed")
    fams = {}
    for g in corpus["all"]:
        f = fams.setdefault(g["family"], {"files": 0, "pop": 0, "rolls": 0})
        f["files"] += 1
        f["pop"] += bool(g["regions"] and g["tasks"] and g["plans"])
        f["rolls"] += len(g["plans"])
    out.table(["family", "files", "populated", "rollouts", "nodes (min/med/max)"],
              [[f"{name}_*", v["files"], v["pop"], v["rolls"],
                "/".join(str(s) for s in _minmedmax(
                    [g["regions"] + g["objects"] for g in corpus["all"] if g["family"] == name]))]
               for name, v in sorted(fams.items())], align="lrrrl")
    out.note("Both families are globbed; the empty ones are dropped for measuring 0 "
             "regions / 0 tasks / 0 rollouts, not for being named differently.")
    out.table(["role", "graph", "why (measured)"],
              [["canonical", canonical["name"], sel["why"]],
               ["tool demo", tool_g["name"], sel["why_tool"]],
               ["ICL source", icl_g["name"], "spread offset "
                f"{cfg.selection.icl_offset}, distinct from the query graph"],
               ["spread", " ".join(g["idx"] for g in sel["spread"]),
                f"{len(sel['spread'])} evenly spaced over {len(populated)} graphs"]])
    out.note(f"policy — train: tools={train_tools} icl={train_icl} | "
             f"deploy: tools={tools} icl={icl}")

    # ---------------------------------------------------------------- 2. prompt policy
    sizes = {(e, t): count(compact_system_prompt(include_edges=e, include_tools=t))
             for e in (False, True) for t in (False, True)}
    off_p = compact_system_prompt(include_edges=False, include_tools=False)
    on_p = compact_system_prompt(include_edges=False, include_tools=True)

    out.rule(f"2  PROMPT POLICY   system-prompt size, include_edges x include_tools ({label})")
    out.table(["arch", "tools OFF", "tools ON", "delta"],
              [["graph-aug (no edge list)", f"{sizes[(False, False)]:,}",
                f"{sizes[(False, True)]:,}", f"+{sizes[(False, True)] - sizes[(False, False)]:,}"],
               ["plain-LLM (edge list)", f"{sizes[(True, False)]:,}",
                f"{sizes[(True, True)]:,}", f"+{sizes[(True, True)] - sizes[(True, False)]:,}"]],
              align="lrrr")
    tutorial = _SPINE_TOOLS_SECTION.split("\n\n")
    contract_delta = sizes[(False, True)] - sizes[(False, False)] - count(_SPINE_TOOLS_SECTION)
    out.table([label, "what tools ON adds"],
              [[f"{count(b):,}", b.splitlines()[0][:56]] for b in tutorial]
              + [[f"{count(_SPINE_TOOLS_SECTION):,}", "= the whole tutorial"],
                 [f"+{contract_delta:,}", "answer contract part 2 (route -> action list) + sep"]],
              align="rl")
    base, mult = _generation_budget(cfg)
    out.note(f"documented: {len(_SPINE_ACTIONS)} planning actions + 7 inbound update fns "
             f"(received, never called)")
    out.note(f"  {', '.join(_SPINE_ACTIONS)}")
    out.note(f"generation budget: tool-free {base:,} tokens, SPINE {base * mult:,} "
             f"({mult}x) — ratifying answers run long, and a generation cut before the "
             f"plan line is a lost sample, not a wrong one.")

    # Tools OFF must stay byte-free of tool text: this is what keeps the tool-free arm equal
    # to the pre-SPINE prompt rather than quietly drifting toward the SPINE one.
    for leak in ("PLANNING API", "GRAPH UPDATE API", "map_region", "clarify(", "replan()"):
        inv(leak not in off_p, f"tool-free prompt leaked {leak!r}")
    inv("give the final plan only: the route the robot follows" in off_p,
        "tool-free arm must keep the pre-SPINE bare-route contract")
    inv("SPINE action list" in on_p, "tools-on must request an action list")
    for a in _SPINE_ACTIONS:
        inv(f"{a}(" in on_p, f"planning API missing {a}")
    for fn in ("add_nodes", "remove_nodes", "add_connections", "remove_connections",
               "update_robot_location", "update_node_attributes", "no_updates"):
        inv(fn in on_p, f"graph-update API missing {fn}")
    for rule_text in ("decide whether the instruction is plannable",
                      "close that plan with replan()",
                      "NEVER substitute a node that happens to be in the graph",
                      # routes are checked with goto, not map_region (map_region needs the
                      # reachability it would be testing, so it cannot ratify a hop)
                      "Put goto(goal_region) in front of the answer()",
                      "Use it to discover, and goto to confirm",
                      # navigation_update is advisory text, not a graph mutation
                      "Treat it as a correction to your behaviour"):
        inv(rule_text in on_p, f"missing behavioral rule: {rule_text!r}")
    # No "new capability" framing anywhere: goto is in the training targets and the rest of
    # the actions are in the ICL examples, so the model is not being handed something new.
    for framing in ("NEW CAPABILITY", "new to you", "You can now act", "This is new"):
        inv(framing not in on_p, f"tool prompt regained new-capability framing: {framing!r}")
    # Undeclared-but-emitted update calls must be named, or the model meets them blind.
    for undeclared in ("navigation_update", "add_node(", "add_connection("):
        inv(undeclared in on_p, f"update API omits emitted call {undeclared!r}")

    # ---------------------------------------------------------------- 3. canonical prompt
    payload = try_load_json(canonical["path"])
    graph_dict, tasks = payload["graph"], payload["tasks"]
    task = tasks[args.task_index % len(tasks)]["task"]
    eval_msgs = format_eval_messages(graph_dict, task, include_edges=False, include_tools=tools)

    view = "apply_chat_template, byte-exact" if tokenizer else "illustrative Role: view"
    out.rule(f"3  THE PROMPT — printed in full, byte-for-byte   "
             f"{canonical['name']} task {args.task_index}")
    out.note(f"graph-aug arch (no edge list) | tools {'ON' if tools else 'OFF'} | icl {icl} | "
             f"{view} | +gen prompt")
    out.block(render(eval_msgs, tokenizer=tokenizer, add_generation_prompt=True),
              indent="  ", elide=False, clip=False)
    inv(_graph_blocks(eval_msgs) == [0], "zero-shot eval must carry exactly one graph, in system")

    # ---------------------------------------------------------------- 4. deltas
    # Every other prompt in the pipeline is section 3 with one switch moved, so these are
    # shown as deltas rather than as four more ~95-line dumps of near-identical text.
    out.rule("4  DELTAS FROM THAT PROMPT   one switch moved at a time")
    llm_p = compact_system_prompt(include_edges=True, include_tools=tools)
    gnn_p = compact_system_prompt(include_edges=False, include_tools=tools)
    out.note("include_edges=True — the plain-LLM baseline. Its intro cites written "
             "edges instead of latent structure:")
    out.block(llm_p.split("\n\n")[0])
    llm_eval = format_eval_messages(graph_dict, task, include_edges=True, include_tools=tools)
    # Only the scene-graph bullets: the tutorial's API list is bulleted too, so filter from
    # the block itself rather than by "• " over the whole message.
    llm_block = llm_eval[0]["content"].rsplit("Scene graph:", 1)[-1]
    edge_lines = [ln for ln in llm_block.split("\n") if ln.startswith("• ")]
    out.note(f"and the scene-graph block gains edge bullets ({len(edge_lines)} bullet lines):")
    out.block("\n".join(edge_lines))
    # The bullets only exist once a graph is rendered — compact_system_prompt() alone just
    # describes them in the intro, so these must be checked on the rendered message.
    inv("• Region Edges:" in llm_block and "• Object Edges:" in llm_block, "edge bullets missing")
    inv("latent" not in llm_p, "plain-LLM prompt must not claim latent connectivity")
    inv("latent space" in gnn_p, "graph-aug prompt lost its latent-connectivity intro")
    # The latent note's job is verification-and-recovery, not an assertion that the model
    # already knows everything — check the instruction it actually has to carry.
    inv("Verify the route before you give it" in gnn_p, "latent note lost its verify step")
    inv("Correct that step, or look for an alternative route" in gnn_p,
        "latent note lost its recovery step")
    for banned in ("The edges are NOT missing", "because you have", "never refuse for"):
        inv(banned not in gnn_p, f"latent note regained over-strict claim {banned!r}")
    inv("• Region Edges:" not in eval_msgs[0]["content"], "graph-aug prompt leaked an edge list")

    out.note()
    out.note(f"include_tools=False — the tutorial disappears entirely "
             f"({sizes[(False, True)] - sizes[(False, False)]:,} {label}); the only other "
             f"prose that differs is the contract's part 2:")
    diff = [b for b in on_p.split("\n\n") if b not in off_p.split("\n\n") and b not in tutorial]
    out.block("\n".join(diff) if diff else "(contract identical)")

    rollouts = [try_load_json(p) for p in canonical["plans"]]
    pick = args.task_index % len(rollouts)
    train_msgs = (format_training_messages(rollouts[pick], include_edges=False,
                                           include_tools=train_tools)
                  if train_icl == 0 else
                  spine_to_compact_messages(rollouts[pick], include_edges=False,
                                            include_tools=train_tools, icl_examples=train_icl))
    out.note()
    out.note(f"TRAINING vs EVAL — same prompt, plus the assistant target and no "
             f"generation prompt (tools {'ON' if train_tools else 'OFF'}, "
             f"icl {train_icl}). Its target line:")
    out.block(_plan_lines(train_msgs)[-1])
    if train_tools != tools:
        out.note("train/deploy tool policies differ by design: the model learns bare "
                 "routes, and compact_output_to_spine_json wraps them as [answer(route)] "
                 "for the planner.")

    # ---------------------------------------------------------------- 5. SPINE tool calling
    rec = RECORDED_SPINE_ROLLOUT
    spine_convo = [
        {"role": "system", "content": "<SPINE system prompt — dropped by the translator>"},
        {"role": "user", "content": f"{rec['task']}\nAdvice: \n- Recall the scene may be "
                                    f"incomplete.\n\nScene graph:{graph_dict}"},
    ]
    for turn in rec["turns"]:
        spine_convo.append({"role": "assistant", "content": json.dumps({
            "primary_goal": rec["task"], "relevant_graph": turn["relevant_graph"],
            "reasoning": turn["reasoning"], "plan": turn["plan"]})})
        if turn.get("updates"):
            spine_convo.append({"role": "user", "content": turn["updates"]})
    compact_rollout = spine_to_compact_messages(spine_convo, include_edges=False,
                                                include_tools=True, icl_examples=0)

    out.rule("5  SPINE TOOL CALLING   receding horizon: only the FIRST action executes")
    out.note(f"recorded rollout — {rec['model']}; {rec['provenance']}")
    out.table(["turn", "plan emitted", "what happened"],
              [[str(i), (t["plan"][:50] + "…") if len(t["plan"]) > 50 else t["plan"], t["note"]]
               for i, t in enumerate(rec["turns"], 1)]
              + [["->", f"updates: {rec['turns'][0]['updates']}",
                  "GraphSim's reply, between turns 1 and 2"]], align="rll",
              maxw=[4, 40, 30])
    pm = rec["metrics"]
    out.note(f"graded route: structured_correct={pm['structured_correct']} "
             f"edge_validity={pm['edge_validity_rate']} "
             f"hallucination={pm['hallucination_rate']} hops={len(pm['parsed_nodes'])}")

    out.note()
    out.table([f"corpus evidence ({corpus['total']} rollouts, post-ICL-strip)", "n", "of"],
              [["targets calling a non-answer SPINE action", corpus["tool_targets"],
                corpus["total"]],
               ["graphs with >=1 such target",
                sum(1 for g in populated if g["tool_targets"]), len(populated)],
               ["rollouts with >1 assistant turn", corpus["multi_turn"], corpus["total"]],
               ["rollouts containing an `updates:` round-trip", corpus["updates"],
                corpus["total"]]], align="lrr")
    out.note("The corpus DOES call tools, but contains no update round-trip at all — "
             "that is what the recorded rollout uniquely shows, and why it is kept.")
    for pp in tool_g["plans"]:
        pl = _plan_lines(spine_to_compact_messages(try_load_json(pp), include_edges=False,
                                                   include_tools=True, icl_examples=0))
        if pl and not pl[-1].startswith("[answer("):
            out.note(f"a real corpus target that calls a tool ({pp.name}):")
            out.block(pl[-1])
            break
    inv("PLANNING API" in compact_rollout[0]["content"], "tool prompt missing from rollout")
    inv("• Region Edges:" not in compact_rollout[0]["content"],
        "recorded rollout must render edge-free")
    inv(sum(1 for m in compact_rollout if m["role"] == "assistant") == len(rec["turns"]),
        "recorded rollout turn count changed")

    # ---------------------------------------------------------------- 6. conversation layout
    # One table replaces four sections that each re-dumped a near-identical prompt. The only
    # thing they showed that section 3 does not is message SHAPE, which is what this measures.
    all_tasks = [t["task"] for t in tasks]
    icl_demos = icl_demos_from_rollouts(
        [try_load_json(p) for p in icl_g["plans"][:int(cfg.selection.icl_demos)]],
        include_tools=tools)
    icl_key = f"SPINE + ICL (n={max(icl, 2)})"
    variants = {
        "eval, zero-shot": eval_msgs,
        "eval, multi-task": format_eval_messages(graph_dict, all_tasks, include_edges=False,
                                                 include_tools=tools),
        "eval + data-drawn ICL": format_eval_messages(graph_dict, task, include_edges=False,
                                                      include_tools=tools, icl_demos=icl_demos),
        "training, 1 task": train_msgs,
        "training, multi-task": assemble_training_conversation(rollouts, include_edges=False,
                                                               include_tools=train_tools),
        icl_key: spine_to_compact_messages(rollouts[pick], include_edges=False,
                                           include_tools=tools, icl_examples=max(icl, 2)),
    }
    variants["follow-up appended"] = append_followup_task(
        format_eval_messages(graph_dict, all_tasks[0], include_edges=False, include_tools=tools)
        + [{"role": "assistant", "content": _plan_lines(train_msgs)[-1]}], all_tasks[1])

    out.rule("6  CONVERSATION LAYOUT   the graph is stated ONCE; tasks stack after it")
    out.table(["variant", "roles", "turns", "graphs", "graph@", f"size ({label})"],
              [[name, (_roles(m)[:22] + "…") if len(m) > 23 else _roles(m), len(m),
                len(_graph_blocks(m)),
                ",".join(map(str, _graph_blocks(m)[:3]))
                + ("…" if len(_graph_blocks(m)) > 3 else ""),
                f"{count(render(m, tokenizer=tokenizer)):,}"]
               for name, m in variants.items()], align="llrrlr")
    out.note("S=system U=user A=assistant. 'graphs' counts Scene graph: blocks — one "
             "per conversation, except with ICL, where each demo brings its own and the "
             "QUERY graph must come LAST, since find_last_graph_scope anchors PE "
             "injection to it.")

    # Test the layout each variant actually has, not the one its name suggests: with
    # few-shot TRAINING the "1 task" variant legitimately carries demo graphs too, so the
    # rule is keyed off whether the system message hoisted the graph.
    for name, msgs in variants.items():
        inv(sum(1 for m in msgs if m["role"] == "system") == 1, f"{name}: expected 1 system msg")
        blocks = _graph_blocks(msgs)
        inv(bool(blocks), f"{name}: no scene graph block")
        if "Scene graph:" in msgs[0]["content"]:          # hoisted (zero-shot) form
            inv(blocks == [0], f"{name}: hoisted graph must be the only one")
        else:                                             # ICL form: demos bring their own
            inv(blocks[-1] == max(i for i, m in enumerate(msgs)
                                  if m["role"] == "user" and "Scene graph:" in m["content"]),
                f"{name}: query graph must be the LAST block (PE injection anchor)")
            inv(len(blocks) > 1, f"{name}: ICL layout expects a demo graph plus the query")
    inv(len(_graph_blocks(variants["follow-up appended"])) == 1,
        "a follow-up must not restate the graph")
    inv(variants["follow-up appended"][-1]["content"] == all_tasks[1].strip(),
        "follow-up task text mismatch")
    inv("Agent Role: You are an excellent graph planner" not in render(variants[icl_key]),
        "verbose SPINE system prompt leaked")

    verbose = [{"role": "user", "content": f"task: {t}\nScene graph:{graph_dict}"}
               for t in all_tasks]
    b = count(render(verbose, tokenizer=tokenizer))
    a = count(render(variants["eval, multi-task"], tokenizer=tokenizer))
    out.note(f"compaction: restating the graph dict per task = {b:,} {label}; compact "
             f"multi-task = {a:,} ({100 * (b - a) / b:.0f}% saved).")
    out.note("stacked shape — one system(graph), then (user task, assistant answer) pairs:")
    out.block(render(assemble_training_conversation(rollouts[:3], include_edges=False,
                                                    include_tools=train_tools),
                     tokenizer=tokenizer))

    # ---------------------------------------------------------------- 7. inverse translator
    tool_free_target = _plan_lines(spine_to_compact_messages(
        rollouts[pick], include_edges=False, include_tools=False, icl_examples=0))[-1]
    real_assistant = [m["content"] for m in compact_rollout if m["role"] == "assistant"][0]
    cases = [
        ("bare route (real target)",
         f"<think>Relevant graph: x\n\nReasoning: r</think>{tool_free_target}", "wrap"),
        ("action list, tool call (real)", real_assistant, "pass"),
        ("action list + answer() (real)",
         [m["content"] for m in compact_rollout if m["role"] == "assistant"][-1], "pass"),
        ("TRUNCATED <think> (real, cut)", real_assistant[:len(real_assistant) // 2], "empty"),
        ("closed <think>, empty plan",
         "<think>Relevant graph: a\n\nReasoning: r</think>   ", "empty"),
        ("no <think> at all", "hub_1 -> mess_hall_1", "wrap"),
        ("bracketed prose, not actions", "<think>r</think>[hub_1 -> b]", "wrap"),
    ]
    out.rule("7  INVERSE TRANSLATOR   model text -> the SPINE JSON the grader consumes")
    rows = []
    for name, text, kind in cases:
        o = json.loads(compact_output_to_spine_json(text))
        plan = o["plan"]
        rows.append([name, kind, (plan[:50] + "…") if len(plan) > 50 else (plan or "(empty)")])
        if kind == "wrap":
            inv(plan.startswith("[answer("), f"{name}: should be wrapped as answer()")
        elif kind == "pass":
            inv(plan == text.split("</think>", 1)[1].strip(), f"{name}: must pass through intact")
        else:
            inv(plan == "", f"{name}: must yield an EMPTY plan")
            if name.startswith("TRUNCATED"):
                inv(bool(o["reasoning"]), "truncated: partial reasoning must survive")
    out.table(["model output shape", "rule", "resulting plan"], rows)
    out.note("pass = action list kept intact; wrapping it would bury the actions inside "
             "one answer() and no tool would ever run. empty = nothing was committed to, "
             "so nothing is graded — a truncated generation must never score off node "
             "names in its prose.")

    inv.report(style)


if __name__ == "__main__":
    main()
