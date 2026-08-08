"""Generate SPINE-style training data from skeleton scene graphs.

This script runs a two-phase data generation pipeline:

Phase 1 — Populate graphs and tasks (LLM-driven)
    Reads every ``*graph*json`` file from ``--data-dir``. Each file is expected
    to be a *skeleton* scene graph produced by ``scripts/generate_eval_graphs.py``:
    nodes have generic names (e.g. ``region_1``, ``object_3``), empty or
    ``"__FILL__"`` descriptions, an empty ``tasks: []`` list, and a ``_metadata``
    block. The LLM (gpt-5.1) renames nodes with realistic semantics, fills
    descriptions, and generates tasks. Outputs land in:

        <name>/populated_graphs/data_gen_XXX.json   # graph + tasks + description
        <name>/populated_graphs/graph_gen_XXX.json  # graph dict only

Phase 2 — Generate example plans (SPINE planner)
    For every populated ``data_gen_*.json``, iterates over its tasks and runs
    the SPINE planner to produce a step-by-step plan trace. Partial-observability
    is simulated by randomly hiding a fraction of nodes per task. Outputs:

        <name>/generated_plans/sample_GGG_TTT.json         # successful rollout
        <name>/generated_plans/sample_GGG_TTT_failed.json  # planner did not answer
        <name>/generated_plans/formatted.json              # aggregated rollouts

Side effects:
    - Creates ``<name>/`` and writes ``<name>/data_gen_params.json`` recording
      the CLI args used for this run.
    - Requires ``OPENAI_API_KEY`` in the environment (GPT-5.1 calls).

Note: ``--data-dir`` must contain *skeleton* graphs, not already-populated ones.
If you point it at populated graphs, Phase 1 will re-process them through the
LLM, which is not the intended flow.
"""

import argparse
import json
from pathlib import Path

from prism.data.data_gen import DataGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate SPINE training data by populating skeleton scene graphs (generated from a random graph model) "
            "with LLM-generated semantics and tasks, then running the SPINE "
            "planner on each task to produce plan-trace rollouts."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--name",
        type=str,
        default="non-iterative-data",
        help=(
            "Output run directory. Populated graphs are written to "
            "<name>/populated_graphs/ and generated plans to "
            "<name>/generated_plans/. The directory is created if missing."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help=(
            "Directory containing skeleton scene graph JSON files. All files "
            "matching the glob '*graph*json' are loaded. Each file must be a "
            "skeleton produced by scripts/generate_eval_graphs.py (empty tasks, "
            "'__FILL__' or empty descriptions, with a _metadata block). "
            "When --skip-populate is set, this directory is instead expected "
            "to contain already-populated 'data_gen_*.json' files."
        ),
    )
    parser.add_argument(
        "--rollout-workers",
        type=int,
        default=1,
        help=(
            "Number of Phase-2 SPINE rollouts to run concurrently. 1 keeps the "
            "historical sequential behavior; >1 requires PRISM_LLM_BACKEND=vllm "
            "(concurrent dialogues are micro-batched into shared vLLM generate "
            "calls)."
        ),
    )
    parser.add_argument(
        "--skip-populate",
        action="store_true",
        help=(
            "Skip Phase 1 (LLM populate). Use --data-dir as a source of "
            "already-populated 'data_gen_*.json' files and run only Phase 2 "
            "(SPINE planner rollouts) on them."
        ),
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Use the OpenAI Batch API for Phase 1 (~50%% cheaper, up to 24h turnaround).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.1",
        help="OpenAI model for Phase 1 populate calls (batch or sync).",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="low",
        help="Reasoning effort for Phase 1 calls (e.g. minimal, low, medium, high, xhigh).",
    )
    parser.add_argument(
        "--task-proportions",
        type=float,
        nargs=4,
        default=None,
        metavar=("EXIST", "POS", "REACH", "NAV"),
        help=(
            "Multinomial weights for the task types: "
            "Existence (DISALLOWED — remapped to Positionality), Positionality, "
            "Reachability, Navigability. Values are normalised to sum to 1. "
            "Existence/yes-no tasks are no longer generated; put 0 in the first "
            "slot. E.g. --task-proportions 0 1 1 1."
        ),
    )
    parser.add_argument(
        "--complexity-proportions",
        type=float,
        nargs=2,
        default=None,
        metavar=("SIMPLE", "COMPLEX"),
        help=(
            "Multinomial weights for task difficulty: simple vs complex. "
            "Values are normalised to sum to 1. "
            "Default when omitted: 1 1 (50%% simple, 50%% complex)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for task-type sampling reproducibility.",
    )
    parser.add_argument(
        "--n-tasks",
        type=int,
        default=10,
        help="Number of tasks to generate per graph in Phase 1.",
    )
    parser.add_argument(
        "--n-longhop-tasks",
        type=int,
        default=0,
        help=(
            "Reserve the LAST N of --n-tasks per graph as long-hop Navigability "
            "tasks with fixed (init, goal) region pairs sampled uniformly over "
            "shortest-path lengths [3, graph diameter]. Endpoints are chosen in "
            "Python on the skeleton (topology survives the rename untouched) and "
            "enforced against the LLM's output, and recorded in "
            "populated_graphs/longhop_manifest.json."
        ),
    )
    parser.add_argument(
        "--max-graphs",
        type=int,
        default=None,
        help=(
            "Cap the run to the first N graphs (sorted): Phase 1 populates only "
            "the first N skeletons and Phase 2 rolls out only the first N "
            "populated graphs. Use this to FINISH an interrupted run at its "
            "current size — set N to the number of graphs already populated and "
            "re-run: resume skips the done work, no new graphs are populated, "
            "and the train/val split is formed from exactly those N graphs. "
            "Default: no cap (process every skeleton present in --data-dir)."
        ),
    )

    args = parser.parse_args()

    Path(args.name).mkdir(parents=True, exist_ok=True)
    with open(f"{args.name}/data_gen_params.json", "w") as f:
        json.dump(vars(args), f)

    output_dir = args.name

    ## `unknown_pcts` is the percentage of nodes (integer percentage) to remove for each task.
    # Currently it's hardcoded to 0 for all tasks.
    unknown_pcts = [0] * 10
    # unknown_pcts = [0, 5, 10, 15] * 10

    # unknown_pcts = [10, 15] * 10
    # n_regions_list = [10, 15, 20] * 10  # np.arange(20, 30, 2)
    # n_objects_list = [3, 6, 9] * 10  # np.arange(10, 30, 1)

    data_generator = DataGenerator(
        graph_unknown=unknown_pcts,
        task_proportions=args.task_proportions,
        complexity_proportions=args.complexity_proportions,
        seed=args.seed,
        n_longhop_tasks=args.n_longhop_tasks,
    )

    if args.skip_populate:
        # Phase 1 skipped: use --data-dir directly as the source of populated
        # data_gen_*.json files for Phase 2.
        populated_graphs_dir = Path(args.data_dir)
    else:
        graph_paths = sorted(Path(args.data_dir).glob("*graph*json"))
        if args.max_graphs is not None:
            graph_paths = graph_paths[: args.max_graphs]
        print(f"Populating graphs and tasks")
        graphs = []
        for graph_path in graph_paths:
            print(graph_path)
            with open(graph_path) as f:
                # json.dumps, not str(): parse_response reads the skeleton back
                # to rebuild the graph from the LLM's rename map.
                graphs.append(json.dumps(json.load(f)))

        populated_graphs_dir = Path(output_dir) / "populated_graphs"
        populated_graphs_dir.mkdir(exist_ok=True)

        if args.batch:
            data_generator.populate_graphs_and_tasks_batch(
                graphs,
                log_dir=populated_graphs_dir,
                n_tasks=args.n_tasks,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
            )
        else:
            data_generator.populate_graphs_and_tasks(
                graphs,
                log_dir=populated_graphs_dir,
                n_tasks=args.n_tasks,
                reasoning_effort=args.reasoning_effort,
            )

    output_generated_plans_dir = Path(output_dir) / "generated_plans"
    generated_graphs_dirs = sorted(populated_graphs_dir.glob("*data_gen*json"))
    if args.max_graphs is not None:
        generated_graphs_dirs = generated_graphs_dirs[: args.max_graphs]
    print(f"List of generated graph directories: {generated_graphs_dirs}")


    print("Generating plans for each graph and task.")
    data_generator.generate_example_plans(
        generated_data=generated_graphs_dirs,
        log_dir=output_generated_plans_dir,
        rollout_workers=args.rollout_workers,
    )

    # for graph in graphs:
    #     with open(graph) as f:
    #         base_graph = json.load(f)

    #     data_generator.generate(
    #         log_dir=log_dir,
    #         base_graph=base_graph,
    #         n_samples=args.n_samples,
    #         n_tasks=args.n_tasks,
    #         description=args.description,
    #     )
