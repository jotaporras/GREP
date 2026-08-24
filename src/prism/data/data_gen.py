import json
import os
import re
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from spine.mapping.graph_util import GraphHandler
from spine.spine import SPINE

from prism.data import graph_gen, graph_sim, local_llm, planning_sim, utils

TASK_TAXONOMY = {
    0: "Existence",
    1: "Positionality",
    2: "Reachability",
    3: "Navigability",
}
N_TASK_TYPES = len(TASK_TAXONOMY)

TASK_COMPLEXITY = {
    0: "Simple",
    1: "Complex",
}
N_TASK_COMPLEXITIES = len(TASK_COMPLEXITY)

DEFAULT_COMPLEXITY_PROPORTIONS = [1.0, 1.0]


def sample_task_types(
    n_tasks: int,
    proportions: List[float],
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    """Sample a list of task-type labels from a multinomial distribution.

    Parameters
    ----------
    n_tasks : int
        How many tasks to generate.
    proportions : List[float]
        Weights for each task type (0-Existence, 1-Positionality,
        2-Reachability, 3-Navigability). Automatically normalised.
    rng : np.random.Generator, optional
        Random generator for reproducibility.

    Returns
    -------
    List[int]
        e.g. [0, 1, 3, 2, 3, 3, 0, ...]
    """
    if rng is None:
        rng = np.random.default_rng()
    p = np.asarray(proportions, dtype=float)
    p = p / p.sum()
    counts = rng.multinomial(n_tasks, p)
    labels: List[int] = []
    for type_id, count in enumerate(counts):
        labels.extend([type_id] * count)
    rng.shuffle(labels)
    return labels


def sample_task_complexities(
    n_tasks: int,
    proportions: List[float],
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    """Sample simple (0) vs complex (1) labels from a multinomial distribution."""
    if rng is None:
        rng = np.random.default_rng()
    p = np.asarray(proportions, dtype=float)
    p = p / p.sum()
    counts = rng.multinomial(n_tasks, p)
    labels: List[int] = []
    for complexity_id, count in enumerate(counts):
        labels.extend([complexity_id] * count)
    rng.shuffle(labels)
    return labels


class DataGenerator:
    n_graph_gen_attempts = 10

    def __init__(
        self,
        graph_unknown: Union[int, List[int]],
        task_proportions: Optional[List[float]] = None,
        complexity_proportions: Optional[List[float]] = None,
        seed: Optional[int] = None,
        n_longhop_tasks: int = 0,
        longhop_max_boost: float = 1.0,
        longhop_allow_avoid: bool = False,
        grounding_directives: bool = False,
        fake_edge_frac: float = 0.0,
        fake_edges_n: int = 2,
        nav_walk_directive: bool = False,
    ):
        self.unknown_pcts = graph_unknown
        self.task_proportions = task_proportions
        self.complexity_proportions = complexity_proportions
        self.n_longhop_tasks = n_longhop_tasks
        # e21: weight multiplier for the DIAMETER hop bucket in longhop
        # endpoint sampling (1.0 = the e19/e20 uniform sampler, bit-identical
        # rng stream), plus prompt directives for avoid-constrained longhop
        # tasks and start-reference diversity. All default to the historical
        # no-op so pre-e21 corpora regenerate unchanged.
        self.longhop_max_boost = longhop_max_boost
        self.longhop_allow_avoid = longhop_allow_avoid
        self.grounding_directives = grounding_directives
        # e19 SPINE closed loop: fraction of rollouts whose PROMPT graph gets
        # fake 2-hop shortcut edges (GraphSim.corrupt_with_fake_edges); goto
        # ratification against the true graph then produces rejection-recovery
        # turns. Seeded per (seed, graph, task) so resume reruns are stable.
        self.fake_edge_frac = fake_edge_frac
        self.fake_edges_n = fake_edges_n
        # Route tasks are otherwise answered straight from the graph text
        # without any tool call (smoke2 7827252: every corrupted rollout
        # answered with the fake shortcut, zero goto executed). The directive
        # makes the teacher walk the route first, so tool trajectories — and
        # rejection-recovery turns on corrupted maps — actually happen.
        self.nav_walk_directive = nav_walk_directive
        self._seed_base = 0 if seed is None else seed
        self.rng = np.random.default_rng(seed)
        self.context_gen = graph_gen.TaskGraphGen()
        self.planning_sim = planning_sim.PlanningSim()

    def _sample_task_labels(self, base_graph: str, n_tasks: int):
        """Per-graph (task_types, task_complexities, longhop_constraints).

        The last ``n_longhop_tasks`` slots are forced to Navigability and bound
        to endpoint pairs sampled on the skeleton's region graph, so the hop
        length of those tasks is fixed before the LLM ever sees the prompt.
        """
        n_recipe = n_tasks - self.n_longhop_tasks
        task_types = None
        if self.task_proportions is not None:
            task_types = sample_task_types(
                n_tasks=n_recipe, proportions=self.task_proportions, rng=self.rng
            ) + [3] * self.n_longhop_tasks
        complexity_props = (
            self.complexity_proportions
            if self.complexity_proportions is not None
            else DEFAULT_COMPLEXITY_PROPORTIONS
        )
        task_complexities = sample_task_complexities(
            n_tasks=n_tasks, proportions=complexity_props, rng=self.rng
        )
        longhop = None
        if self.n_longhop_tasks:
            longhop = graph_gen.sample_longhop_constraints(
                json.loads(base_graph)["graph"],
                self.n_longhop_tasks,
                self.rng,
                max_boost=self.longhop_max_boost,
            )
        return task_types, task_complexities, longhop

    @staticmethod
    def _record_longhop(log_dir, idx: int, longhop: Optional[list]) -> None:
        """Sidecar manifest of sampled endpoints, so the delivered hop-length
        mix can be verified without re-deriving goals from acceptance criteria."""
        if not longhop:
            return
        path = Path(log_dir) / "longhop_manifest.json"
        manifest = json.loads(path.read_text()) if path.exists() else {}
        manifest[f"data_gen_{idx:03d}"] = longhop
        path.write_text(json.dumps(manifest, indent=2))

    def populate_graphs_and_tasks(
        self,
        base_graphs: List[str],
        log_dir: str,
        n_tasks: int = 10,
        reasoning_effort: str = "low",
    ) -> None:
        """Populate graphs with semantics and tasks.

        This will use an LLM to populate generic object and region names
        (e.g., object_1, ...) with meaningful semantics.

        Parameters
        ----------
        base_graphs : List[str]
            List to base scene graphs containing structure to populate
        log_dir : str
            Graphs and tasks will be saved under here
        """
        previous_tasks = ""

        for idx, base_graph in enumerate(base_graphs):
            out_path = f"{log_dir}/data_gen_{idx:03d}.json"

            # Resume: skip graphs already populated by a prior (possibly crashed)
            # run, but still fold their tasks into `previous_tasks` so the
            # de-duplication hint stays intact across restarts.
            existing = self._load_valid_populated(out_path)
            if existing is not None:
                print(f"Skipping populate for graph {idx}: {out_path} already valid")
                previous_tasks += ",".join(
                    e["task"] for e in existing.get("tasks", []) if "task" in e
                )
                continue

            rnd_data = None
            for _ in range(self.n_graph_gen_attempts):
                try:
                    # error handling in case data generation fails
                    task_types, task_complexities, longhop = (
                        self._sample_task_labels(base_graph, n_tasks)
                    )

                    rnd_data = self.context_gen.get_tasks(
                        base_graph=base_graph,
                        n_tasks=n_tasks,
                        previous_tasks=previous_tasks,
                        task_types=task_types,
                        task_complexities=task_complexities,
                        longhop_constraints=longhop,
                        reasoning_effort=reasoning_effort,
                        grounding_directives=self.grounding_directives,
                        longhop_allow_avoid=self.longhop_allow_avoid,
                    )

                    break

                except Exception as ex:
                    # A GPU fault corrupts the CUDA context for the rest of this
                    # process, so retrying here (and on every later graph) is
                    # hopeless. Abort fast; nothing was written for this idx, so
                    # re-running the script resumes from exactly this graph.
                    if self._is_fatal_gpu_error(ex):
                        print(
                            f"graph {idx}: fatal GPU error — aborting so a re-run "
                            f"can resume from here: {ex}"
                        )
                        raise
                    print(f"graph generator invalid: {ex}")

            # A graph that fails every attempt for a NON-fatal reason (e.g. the
            # LLM kept emitting malformed JSON) must not kill the whole run or
            # write the previous graph's data. Skip it (leaving no output file)
            # and continue; a later re-run will re-attempt this idx.
            if rnd_data is None:
                print(
                    f"WARNING: graph {idx} failed all {self.n_graph_gen_attempts} "
                    f"populate attempts — skipping (will be retried on re-run)"
                )
                continue

            tasks = [entry["task"] for entry in rnd_data["tasks"]]

            previous_tasks += ",".join(tasks)

            print(f"logging to: {log_dir}")
            # Atomic write: a crash mid-write must not leave a half-written file
            # that resume would mistake for complete. Write to .tmp then replace.
            tmp_path = f"{out_path}.tmp"
            with open(tmp_path, "w") as f:
                f.write(json.dumps(rnd_data, indent=2))
            os.replace(tmp_path, out_path)
            self._record_longhop(log_dir, idx, longhop)

            # save graphs separately for Graph handler
            graph_path = f"{log_dir}/graph_gen_{idx:03d}.json"
            tmp_graph = f"{graph_path}.tmp"
            with open(tmp_graph, "w") as f:
                f.write(json.dumps(rnd_data["graph"], indent=2))
            os.replace(tmp_graph, graph_path)

    def populate_graphs_and_tasks_batch(
        self,
        base_graphs: List[str],
        log_dir: str,
        n_tasks: int = 10,
        model: str = "gpt-5.1",
        reasoning_effort: str = "low",
        poll_interval: int = 60,
    ) -> None:
        """Like populate_graphs_and_tasks but batches all prompts in one call.

        With the OpenAI backend this is the Batch API (~50% cheaper); with the
        vLLM backend it is one continuously-batched generate over the list.
        Resumes like the sequential path: graphs whose ``data_gen_XXX.json`` is
        already valid are skipped, so the auto-resume retry loop in the launch
        scripts only pays for the missing graphs.
        """
        pending = []  # (original idx, base_graph)
        for idx, g in enumerate(base_graphs):
            if self._load_valid_populated(f"{log_dir}/data_gen_{idx:03d}.json"):
                print(f"Skipping populate for graph {idx}: already valid")
            else:
                pending.append((idx, g))
        if not pending:
            print("Batch populate: nothing to do (all graphs already populated)")
            return

        prompts = []
        longhops = []
        for _, g in pending:
            task_types, task_complexities, longhop = (
                self._sample_task_labels(g, n_tasks)
            )
            longhops.append(longhop)
            prompts.append(
                self.context_gen.build_prompt(
                    base_graph=g,
                    n_tasks=n_tasks,
                    task_types=task_types,
                    task_complexities=task_complexities,
                    longhop_constraints=longhop,
                )
            )
        responses = self.context_gen.client.batch_query_gpt_5(
            prompts,
            model=model,
            reasoning_effort=reasoning_effort,
            poll_interval=poll_interval,
        )

        for (idx, base_graph), response, longhop in zip(pending, responses, longhops):
            # One malformed response must not discard the whole batch: skip the
            # graph (no file written) so a re-run's resume regenerates only it.
            try:
                rnd_data = self.context_gen.parse_response(
                    response, base_graph, n_tasks=n_tasks,
                    longhop_constraints=longhop,
                )
            except Exception as ex:
                print(f"graph {idx}: batch populate response unparseable — {ex}")
                continue
            # Atomic writes, matching the sequential path.
            out_path = f"{log_dir}/data_gen_{idx:03d}.json"
            with open(f"{out_path}.tmp", "w") as f:
                f.write(json.dumps(rnd_data, indent=2))
            os.replace(f"{out_path}.tmp", out_path)
            self._record_longhop(log_dir, idx, longhop)
            graph_path = f"{log_dir}/graph_gen_{idx:03d}.json"
            with open(f"{graph_path}.tmp", "w") as f:
                f.write(json.dumps(rnd_data["graph"], indent=2))
            os.replace(f"{graph_path}.tmp", graph_path)

    @staticmethod
    def _has_valid_rollout(path: str) -> bool:
        """True if ``path`` is a rollout ``split_train_val`` would accept: a
        parseable JSON list of role/content dicts with a strippable ICL prefix.

        Lets a recovery rerun of :meth:`generate_example_plans` skip tasks whose
        rollout is already good and regenerate only failed/corrupt/missing ones.
        """
        try:
            with open(path) as f:
                obj = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False
        if not (isinstance(obj, list) and obj):
            return False
        if not all(
            isinstance(m, dict) and "role" in m and "content" in m for m in obj
        ):
            return False
        try:
            utils.strip_icl(obj)
        except Exception:
            return False
        return True

    @staticmethod
    def _load_valid_populated(path: str):
        """Return the populated-graph dict at ``path`` if it is complete, else None.

        Lets :meth:`populate_graphs_and_tasks` resume after a crash by skipping
        graphs already written (a dict with a ``graph`` dict and a non-empty
        ``tasks`` list). A truncated/half-written file fails json parsing and is
        treated as missing, so it gets regenerated.
        """
        try:
            with open(path) as f:
                obj = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(obj, dict):
            return None
        if not isinstance(obj.get("graph"), dict):
            return None
        if not isinstance(obj.get("tasks"), list) or not obj["tasks"]:
            return None
        return obj

    @staticmethod
    def _is_fatal_gpu_error(ex: Exception) -> bool:
        """True for GPU faults that corrupt the process CUDA context.

        After a "CUDA error: unspecified launch failure", a device-side assert,
        cuBLAS failure, or OOM, every later GPU call in the SAME process keeps
        failing — so retrying in-process is futile. The caller re-raises these to
        exit fast; a fresh re-run then resumes from the unfinished work.
        """
        s = str(ex).lower()
        return any(
            tok in s
            for tok in ("cuda error", "cuda kernel", "device-side assert",
                        "cublas", "out of memory", "cudnn")
        )

    def _run_one_rollout(
        self,
        *,
        idx: int,
        task_idx: int,
        task_entry: dict,
        graph: dict,
        log_dir: str,
        spine_client,
    ) -> bool:
        """Run one SPINE rollout for (graph idx, task_idx) with the atomic
        .partial-commit / *_failed.json-quarantine protocol. Returns True when a
        sample file was produced (committed or previously valid). Raises on
        fatal GPU errors so the caller can abort the run.

        Thread-safe: every mutable object (GraphHandler, GraphSim, SPINE,
        the .partial log) is task-local; `spine_client` must be thread-safe
        when called with rollout_workers > 1 (the vLLM client is).
        """
        log_name = f"{log_dir}/sample_{idx:03d}_{task_idx:03d}.json"
        # Recovery rerun: skip tasks whose rollout is already valid
        # so only failed/corrupt/missing ones are regenerated.
        if self._has_valid_rollout(log_name):
            print(
                f"Skipping sample_{idx:03d}_{task_idx:03d}: "
                "valid rollout already exists"
            )
            return True
        # Atomic-commit a rollout. SPINE rewrites its log file after
        # EVERY planner turn, so a ^C between turns would otherwise leave
        # a valid-looking but answer-less `sample_GGG_TTT.json` that
        # `_has_valid_rollout` mistakes for complete and skips forever.
        # Instead SPINE writes to a `.partial`, which we rename to the
        # real name only once planning RETURNS. An interrupted task leaves
        # only the `.partial` (ignored by resume, split, and aggregate)
        # and is cleanly regenerated on the next run.
        failed_name = log_name.replace(".json", "_failed.json")
        tmp_log = f"{log_name}.partial"
        if os.path.exists(tmp_log):
            os.remove(tmp_log)  # drop any stale partial from a prior ^C
        try:
            task = task_entry["task"]
            init_location = task_entry["init_node"]
            # SPINE's GraphHandler loads from a path or dict; build an
            # empty one (graph="") and populate it from the in-memory
            # graph dict via reset() (same idiom as parse_response / the
            # eval path) so GraphSim sees a populated graph at
            # construction.
            graph_handle = GraphHandler(graph="")
            graph_handle.reset(graph, current_location=init_location)
            graph_data_gen = graph_sim.GraphSim(graph_handle)
            unknown_pct = self.unknown_pcts[task_idx % len(self.unknown_pcts)]
            graph_data_gen.randomly_remove_nodes(pct=unknown_pct)
            route_str = None
            if self.fake_edge_frac > 0 or self.nav_walk_directive:
                route_str, _reason = self._oracle_route(graph, task_entry)
            if self.nav_walk_directive and route_str:
                # Nav task: without this the teacher answers the route straight
                # from the graph text with zero tool calls (see __init__ note).
                task = (
                    f"{task} First physically navigate to the destination "
                    "using goto (the robot walks your route and reports what "
                    "actually happens), then answer with the route you "
                    "traversed.")
            if self.fake_edge_frac > 0:
                fe_rng = np.random.default_rng(
                    [self._seed_base, idx, task_idx])
                if fe_rng.random() < self.fake_edge_frac:
                    # Prefer shortcuts that skip an intermediate node of the
                    # task's ground-truth route: a random 2-hop shortcut is
                    # rarely on the planned route, so the planner almost never
                    # takes the bait and no recovery turn happens (smoke
                    # 7826590: 1 rejection / 11 corrupted rollouts).
                    preferred = []
                    if route_str:
                        route = route_str.split(" -> ")
                        preferred = [(route[i], route[i + 2])
                                     for i in range(len(route) - 2)]
                        fe_rng.shuffle(preferred)
                    fake = graph_data_gen.corrupt_with_fake_edges(
                        self.fake_edges_n, fe_rng, preferred_pairs=preferred)
                    if fake:
                        print(
                            f"sample_{idx:03d}_{task_idx:03d}: prompt graph "
                            f"corrupted with fake edges {fake}"
                        )
            planner = SPINE(
                graph=graph_data_gen.partial_graph,
                log_name=tmp_log,
                client=spine_client,
            )
            out = self.planning_sim.run_planning(
                llm_planner=planner, task=task, graph_data_gen=graph_data_gen
            )
            # Commit: a rollout that reached an answer becomes the real
            # sample (picked up by split_train_val); anything else is
            # quarantined as *_failed.json (ignored by split, retried on a
            # later run since `sample_GGG_TTT.json` stays absent).
            os.replace(
                tmp_log,
                log_name if out.terminated_by == "answer" else failed_name,
            )
            return True
        except Exception as ex:
            # Quarantine whatever partial trace exists as *_failed.json.
            if os.path.exists(tmp_log):
                os.replace(tmp_log, failed_name)
            # A GPU fault corrupts the CUDA context for every later task
            # in this process; abort fast rather than quarantining all of
            # them. Completed rollouts are already committed, so a re-run
            # resumes (skips them via _has_valid_rollout) and retries the
            # missing/failed ones.
            if self._is_fatal_gpu_error(ex):
                print(
                    f"sample_{idx:03d}_{task_idx:03d}: fatal GPU error — "
                    f"aborting so a re-run can resume: {ex}"
                )
                raise
            # Any other single-task failure must not abort the run.
            print(
                f"Skipping sample_{idx:03d}_{task_idx:03d}: "
                f"rollout failed — {type(ex).__name__}: {ex}"
            )
            return False

    @staticmethod
    def _make_spine_client(enable_thinking: bool = True):
        """Pick the SPINE planner client from PRISM_LLM_BACKEND.

        "vllm" -> thread-safe batched vLLM client; "hf"/"gemma"/"local" ->
        eager HF Gemma client; anything else -> None so SPINE uses its own
        default (OpenAI). Built once so the model is reused across all tasks.
        ``enable_thinking=False`` (e20 path-only) renders the no-think chat
        template so the teacher answers without a reasoning block.
        """
        from prism.data import vllm_llm

        if vllm_llm.vllm_backend_enabled():
            return vllm_llm.VLLMSpineClient(enable_thinking=enable_thinking)
        if local_llm.hf_backend_enabled():
            return local_llm.GemmaSpineClient(enable_thinking=enable_thinking)
        return None

    # ------------------------------------------------------------------
    # e20 path-only rollout modes. Both produce sample files with the SAME
    # message-list shape as a SPINE rollout ([system?, user "task: ... scene
    # graph {...}", assistant SPINE-JSON answer]) so strip_icl / aggregate /
    # split_train_val / compact_prompt all work unchanged; the assistant
    # answer carries an empty think scaffold and `plan="[answer(a -> b)]"`.
    # NOTE (leak check): sample and *_failed.json files must never contain the
    # strings `acceptance_criterion` or a JSON `"answer":` key — the corpus
    # driver greps generated_plans/ for exactly those (GT stays out of
    # training text; grading uses it in-process only).
    # ------------------------------------------------------------------

    # Teacher instruction for path-only distillation. Mirrors the eval-side
    # route_only answer contract (compact_prompt._answer_contract) so the
    # teacher is graded under the same wording the student will be trained on.
    _PATH_ONLY_INSTRUCTION = (
        "You are a robot operating in the scene graph below. Complete the "
        "navigation task by answering with the final route ONLY: the nodes the "
        "robot visits in order, starting from its current location, joined by "
        "arrows (for example, start_region -> middle_region -> goal_region). "
        "Output nothing else — no reasoning, no explanations, no labels."
    )

    @staticmethod
    def _grade_route(route_text: str, graph: dict, task_entry: dict):
        """Deterministic verdict for a candidate route. Returns (ok, verdict).

        Uses the SAME scorer as eval (path_validator.validate_structured with
        ``full_response=None``: pure RegEx/NetworkX, no judge model, no rescue),
        so "discard the ones that have errors" means exactly "would not have
        been graded correct at eval time". The GT fields are consumed here
        in-process and never written to disk.
        """
        from prism.eval import path_validator

        verdict = path_validator.validate_structured(
            route_text,
            graph,
            init_node=task_entry.get("init_node"),
            answer=task_entry.get("answer"),
            criterion=task_entry.get("acceptance_criterion"),
            task=task_entry.get("task"),
            full_response=None,
        )
        return bool(verdict and verdict.get("structured_correct")), verdict

    @staticmethod
    def _commit_plan_sample(log_name: str, *, task: str, graph: dict,
                            init_node: str, route: str, mode: str) -> None:
        """Atomically write a committed path-only sample file (message list)."""
        assistant = json.dumps({
            "primary_goal": task,
            "relevant_graph": "",
            "reasoning": "",
            "plan": f"[answer({route})]",
        })
        sg = dict(graph)
        sg["robot_location"] = init_node
        messages = [
            {"role": "system", "content": f"path-only rollout ({mode})"},
            # Marker format matches a real SPINE rollout's query turn
            # ("task: <task>Scene graph:<dict>") so strip_icl and
            # _extract_scene_graph_dict/_extract_task parse it identically.
            {"role": "user", "content": f"task: {task}Scene graph:{json.dumps(sg)}"},
            {"role": "assistant", "content": assistant},
        ]
        tmp = f"{log_name}.partial"
        with open(tmp, "w") as f:
            json.dump(messages, f)
        os.replace(tmp, log_name)

    @staticmethod
    def _quarantine(failed_name: str, payload: dict) -> None:
        """Write a *_failed.json diagnostic (never picked up by the split)."""
        tmp = f"{failed_name}.partial"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=1)
        os.replace(tmp, failed_name)

    def _run_one_path_only(
        self, *, idx: int, task_idx: int, task_entry: dict, graph: dict,
        log_dir: str, spine_client,
    ) -> bool:
        """One single-turn path-only teacher query for (graph idx, task_idx).

        The teacher sees ONLY the instruction + task text + full scene graph
        (no GT, no tools, no exploration — unknown_pct does not apply since a
        single-turn prediction has no way to discover hidden nodes). The
        response is graded deterministically; wrong/unparseable routes are
        quarantined as *_failed.json with a machine-readable ``reason`` so
        ``rollout_stats.json`` can report the teacher's accuracy — the monitor
        for whether no-think route-only prompting degrades the base model.
        """
        from prism.data import compact_prompt

        log_name = f"{log_dir}/sample_{idx:03d}_{task_idx:03d}.json"
        if self._has_valid_rollout(log_name):
            print(f"Skipping sample_{idx:03d}_{task_idx:03d}: valid rollout already exists")
            return True
        failed_name = log_name.replace(".json", "_failed.json")
        task = task_entry["task"]
        init_node = task_entry["init_node"]
        sg = dict(graph)
        sg["robot_location"] = init_node
        msg = [{
            "role": "user",
            "content": (f"{self._PATH_ONLY_INSTRUCTION}\n\n"
                        f"task: {task}\nScene graph:{json.dumps(sg)}"),
        }]
        response, ok = spine_client.query_llm(msg)
        if not ok:
            self._quarantine(failed_name, {
                "mode": "path_only", "reason": "generation_error",
                "task": task, "response": response})
            return False
        route = compact_prompt.extract_route(response)
        if route is None:
            self._quarantine(failed_name, {
                "mode": "path_only", "reason": "no_route",
                "task": task, "response": response})
            return False
        graded_ok, verdict = self._grade_route(route, graph, task_entry)
        if not graded_ok:
            reason = "wrong_route" if verdict else "no_goal_resolved"
            self._quarantine(failed_name, {
                "mode": "path_only", "reason": reason, "task": task,
                "route": route, "response": response,
                "goal": (verdict or {}).get("goal"),
                "full_path_valid": (verdict or {}).get("full_path_valid"),
                "start_goal_ok": (verdict or {}).get("start_goal_ok"),
                "waypoints_ok": (verdict or {}).get("waypoints_ok"),
                "avoid_ok": (verdict or {}).get("avoid_ok")})
            return False
        self._commit_plan_sample(
            log_name, task=task, graph=graph, init_node=init_node,
            route=route, mode="path_only")
        return True

    @staticmethod
    def _oracle_route(graph: dict, task_entry: dict):
        """NetworkX ground-truth route for a nav task. Returns (route|None, reason).

        Endpoints/waypoints/avoid come from the same resolver the scorer uses
        (path_validator.derive_targets), so the produced route is by
        construction the route the grader wants: shortest path through each
        waypoint in order, with avoided nodes removed from the graph first.
        """
        import networkx as nx

        from prism.eval import path_validator

        init_node = task_entry.get("init_node")
        goal, waypoints, avoid, _required, kind = path_validator.derive_targets(
            graph,
            init_node=init_node,
            answer=task_entry.get("answer"),
            criterion=task_entry.get("acceptance_criterion"),
            task=task_entry.get("task"),
        )
        if goal is None:
            return None, "no_goal_resolved"
        if kind != "path":
            return None, "not_a_path_task"
        g = path_validator.build_graph(graph, directed=False)
        protected = {init_node, goal, *waypoints}
        g.remove_nodes_from([n for n in avoid if n not in protected])
        hops = [init_node] + list(waypoints) + [goal]
        route = [init_node]
        for a, b in zip(hops, hops[1:]):
            try:
                seg = nx.shortest_path(g, a, b)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return None, "no_path"
            route.extend(seg[1:])
        return " -> ".join(route), ""

    def _run_one_oracle(
        self, *, idx: int, task_idx: int, task_entry: dict, graph: dict,
        log_dir: str, spine_client=None,
    ) -> bool:
        """One LLM-free oracle rollout: the 'teacher response' is the NetworkX
        ground-truth route (e20b — hardcoded generations, no rollout LLM).

        The route is still self-verified through the eval scorer before commit
        (catches any resolver/constructor disagreement instead of training on
        it); failures are quarantined with a ``reason`` like the other modes.
        """
        log_name = f"{log_dir}/sample_{idx:03d}_{task_idx:03d}.json"
        if self._has_valid_rollout(log_name):
            print(f"Skipping sample_{idx:03d}_{task_idx:03d}: valid rollout already exists")
            return True
        failed_name = log_name.replace(".json", "_failed.json")
        task = task_entry["task"]
        init_node = task_entry["init_node"]
        route, reason = self._oracle_route(graph, task_entry)
        if route is None:
            self._quarantine(failed_name, {
                "mode": "oracle", "reason": reason, "task": task})
            return False
        graded_ok, verdict = self._grade_route(route, graph, task_entry)
        if not graded_ok:
            self._quarantine(failed_name, {
                "mode": "oracle", "reason": "oracle_verify_failed", "task": task,
                "route": route, "goal": (verdict or {}).get("goal"),
                "waypoints_ok": (verdict or {}).get("waypoints_ok"),
                "avoid_ok": (verdict or {}).get("avoid_ok")})
            return False
        self._commit_plan_sample(
            log_name, task=task, graph=graph, init_node=init_node,
            route=route, mode="oracle")
        return True

    @staticmethod
    def _write_rollout_stats(log_dir: str, rollout_mode: str) -> None:
        """Scan ``log_dir`` and write ``rollout_stats.json``: per-graph and total
        pass/fail counts with failure reasons — the e20 teacher-accuracy monitor.

        Counts are over (graph, task) ids present on disk: a task counts as
        passed when its ``sample_GGG_TTT.json`` exists, failed when only the
        ``*_failed.json`` does (a retried-then-passed task counts as passed).
        """
        passed, failed, reasons = {}, {}, {}
        for f in Path(log_dir).glob("sample_*_failed.json"):
            gid = f.name.split("_")[1]
            key = f.name.replace("_failed.json", ".json")
            if (Path(log_dir) / key).exists():
                continue  # later retry committed a valid sample
            failed[gid] = failed.get(gid, 0) + 1
            try:
                with open(f) as fh:
                    reason = json.load(fh).get("reason", "spine_no_answer")
            except Exception:
                reason = "unreadable"
            reasons[reason] = reasons.get(reason, 0) + 1
        for f in Path(log_dir).glob("sample_*.json"):
            if f.name.endswith("_failed.json"):
                continue
            gid = f.name.split("_")[1]
            passed[gid] = passed.get(gid, 0) + 1
        n_pass = sum(passed.values())
        n_fail = sum(failed.values())
        total = n_pass + n_fail
        stats = {
            "rollout_mode": rollout_mode,
            "n_pass": n_pass,
            "n_fail": n_fail,
            "n_total": total,
            "pass_rate": (n_pass / total) if total else None,
            "fail_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
            "per_graph": {
                gid: {"pass": passed.get(gid, 0), "fail": failed.get(gid, 0)}
                for gid in sorted(set(passed) | set(failed))
            },
        }
        out = Path(log_dir) / "rollout_stats.json"
        with open(out, "w") as f:
            json.dump(stats, f, indent=1)
        rate = f"{stats['pass_rate']:.3f}" if total else "n/a"
        print(f"[{rollout_mode}] rollout stats: {n_pass}/{total} passed "
              f"(rate {rate}); reasons {stats['fail_reasons']} -> {out}")

    def generate_example_plans(
        self,
        generated_data: List[str],
        log_dir: str,
        rollout_workers: int = 1,
        rollout_mode: str = "spine",
        path_only_thinking: bool = False,
    ) -> None:
        """Run SPINE rollouts for every (graph, task) pair.

        Parameters
        ----------
        generated_data : List[str]
            List of paths to json of generated data. Each JSON should have the following fields
                graph: scene graph
                robot_location: robot's start location
                tasks: List of corresponding tasks
        log_dir : str
            Path to log generated plans. Plans will be saved in the following structure
                {log_dir}_sample_{graph_idx}_{task_idx}.json
        rollout_workers : int
            Number of rollouts to run concurrently. 1 (default) preserves the
            historical sequential behavior. >1 requires a thread-safe planner
            client (PRISM_LLM_BACKEND=vllm) — concurrent dialogues are then
            micro-batched into shared vLLM generate calls.
        rollout_mode : str
            "spine" (default): the historical multi-turn SPINE planner rollout.
            "path_only" (e20a): ONE route-only teacher query per task — no
            tools, no reasoning scaffold in the target; responses are graded
            with the eval scorer and wrong routes discarded.
            "oracle" (e20b): NO rollout LLM — the target route is the NetworkX
            ground-truth path (still self-verified through the eval scorer).
            Both non-spine modes write ``rollout_stats.json`` (pass/fail per
            graph + failure reasons — the teacher-accuracy monitor).
        path_only_thinking : bool
            path_only mode only: keep the teacher's thinking mode ON (the
            think block is stripped before route extraction). Default False =
            the no-think chat template, per the e20 hypothesis.
        """

        Path(log_dir).mkdir(parents=True, exist_ok=True)

        valid_modes = ("spine", "path_only", "oracle")
        if rollout_mode not in valid_modes:
            raise ValueError(
                f"rollout_mode must be one of {valid_modes}, got {rollout_mode!r}")
        if path_only_thinking and rollout_mode != "path_only":
            raise ValueError(
                "path_only_thinking only applies to rollout_mode='path_only'")

        from prism.data import vllm_llm

        if rollout_mode == "oracle":
            # LLM-free: routes come from NetworkX; never load a model.
            spine_client = None
        elif rollout_mode == "path_only":
            spine_client = self._make_spine_client(
                enable_thinking=path_only_thinking)
            if spine_client is None:
                raise ValueError(
                    "rollout_mode='path_only' needs a local teacher client with "
                    "query_llm; set PRISM_LLM_BACKEND=vllm (or hf) — the OpenAI "
                    "default has no route-only single-turn path.")
            print(f"[path_only] teacher thinking "
                  f"{'ON (stripped)' if path_only_thinking else 'OFF (no-think template)'}")
        else:
            spine_client = self._make_spine_client()

        if (rollout_workers > 1 and rollout_mode != "oracle"
                and not isinstance(spine_client, vllm_llm.VLLMSpineClient)):
            raise ValueError(
                f"rollout_workers={rollout_workers} needs the thread-safe vLLM "
                "planner client; set PRISM_LLM_BACKEND=vllm (got backend "
                f"{os.environ.get('PRISM_LLM_BACKEND', 'openai')!r})."
            )

        # Flatten to one job per (graph, task); unreadable graphs are skipped
        # here exactly as before.
        jobs = []
        for pos, data_path in enumerate(generated_data):
            # Sample ids must carry the data_gen file's OWN index, not the list
            # position: with any gap in the file sequence (a populate-rejected
            # graph) positional ids shift, mis-grouping every later graph's
            # rollouts in the split and breaking resume across re-runs.
            id_match = re.search(r"data_gen_(\d+)", Path(data_path).name)
            idx = int(id_match.group(1)) if id_match else pos
            try:
                with open(data_path) as f:
                    data = json.load(f)

                # INVARIANT: the planner only ever sees the natural-language task
                # text. `init_node` sets the robot's start region in the simulator
                # (matching run_eval.py) and is NOT passed to the planner as task
                # content. `answer` and `acceptance_criterion` must NEVER reach
                # SPINE or run_planning — that would leak ground truth into the
                # rollout traces and contaminate training data.
                task_entries = data["tasks"]
                tasks = [entry["task"] for entry in task_entries]
                assert all(
                    isinstance(t, str) for t in tasks
                ), "task field must be a plain string"

                graph = data["graph"]
                assert isinstance(graph, dict)
            except Exception as ex:
                # A malformed data_gen_*.json (unparseable JSON, missing keys)
                # must not abort the whole run — skip this graph so the
                # remaining graphs still produce rollouts.
                print(f"Skipping graph {idx} ({data_path}): unreadable — {ex}")
                continue

            print(f"Generating example data for tasks: {tasks}")

            for task_idx, task_entry in enumerate(task_entries):
                jobs.append(
                    dict(
                        idx=idx,
                        task_idx=task_idx,
                        task_entry=task_entry,
                        graph=graph,
                        log_dir=str(log_dir),
                        spine_client=spine_client,
                    )
                )

        run_one = {
            "spine": self._run_one_rollout,
            "path_only": self._run_one_path_only,
            "oracle": self._run_one_oracle,
        }[rollout_mode]

        if rollout_workers <= 1:
            for job in jobs:
                run_one(**job)
        else:
            # N dialogues in flight; their planner turns are micro-batched by
            # the vLLM client's generate gate. A fatal GPU error in any worker
            # aborts the whole run (same contract as sequential mode): pending
            # jobs are cancelled and the error is re-raised, so a re-run
            # resumes from the committed rollouts.
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=rollout_workers) as pool:
                futures = [pool.submit(run_one, **job) for job in jobs]
                fatal = None
                for fut in futures:
                    try:
                        fut.result()
                    except Exception as ex:  # noqa: BLE001 — only fatal errors escape
                        fatal = fatal or ex
                        for pending in futures:
                            pending.cancel()
                if fatal is not None:
                    raise fatal

        utils.aggregate(
            root_dir=log_dir,
            glob_str="sample*json",
            out_file=f"{log_dir}/formatted.json",
        )

        if rollout_mode != "spine":
            self._write_rollout_stats(str(log_dir), rollout_mode)
