import json
import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from spine.mapping.graph_util import GraphHandler
from spine.spine import SPINE

from prism.data import graph_gen, graph_sim, planning_sim, utils

TASK_TAXONOMY = {
    0: "Existence",
    1: "Positionality",
    2: "Reachability",
    3: "Navigability",
}
N_TASK_TYPES = len(TASK_TAXONOMY)


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


class DataGenerator:
    n_graph_gen_attempts = 10

    def __init__(
        self,
        graph_unknown: Union[int, List[int]],
        task_proportions: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ):
        self.unknown_pcts = graph_unknown
        self.task_proportions = task_proportions
        self.rng = np.random.default_rng(seed)
        self.context_gen = graph_gen.TaskGraphGen()
        self.planning_sim = planning_sim.PlanningSim()

    def populate_graphs_and_tasks(
        self,
        base_graphs: List[str],
        log_dir: str,
        n_tasks: int = 10,
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
            for _ in range(self.n_graph_gen_attempts):
                try:
                    # error handling in case data generation fails
                    task_types = None
                    if self.task_proportions is not None:
                        task_types = sample_task_types(
                            n_tasks=n_tasks,
                            proportions=self.task_proportions,
                            rng=self.rng,
                        )

                    rnd_data = self.context_gen.get_tasks(
                        base_graph=base_graph,
                        n_tasks=n_tasks,
                        previous_tasks=previous_tasks,
                        task_types=task_types,
                    )

                    break

                except Exception as ex:
                    print(f"graph generator invalid: {ex}")

            tasks = [entry["task"] for entry in rnd_data["tasks"]]

            previous_tasks += ",".join(tasks)

            print(f"logging to: {log_dir}")
            with open(f"{log_dir}/data_gen_{idx:03d}.json", "w") as f:
                json_str = json.dumps(rnd_data, indent=2)
                f.write(json_str)

            # save graphs separately for Graph handler
            graph_path = f"{log_dir}/graph_gen_{idx:03d}.json"
            with open(graph_path, "w") as f:
                json_str = json.dumps(rnd_data["graph"], indent=2)
                f.write(json_str)

    def populate_graphs_and_tasks_batch(
        self,
        base_graphs: List[str],
        log_dir: str,
        n_tasks: int = 10,
        model: str = "gpt-5.5",
        reasoning_effort: str = "xhigh",
        poll_interval: int = 60,
    ) -> None:
        """Like populate_graphs_and_tasks but uses the OpenAI Batch API (~50% cheaper)."""
        prompts = []
        for g in base_graphs:
            task_types = None
            if self.task_proportions is not None:
                task_types = sample_task_types(
                    n_tasks=n_tasks,
                    proportions=self.task_proportions,
                    rng=self.rng,
                )
            prompts.append(
                self.context_gen.build_prompt(
                    base_graph=g, n_tasks=n_tasks, task_types=task_types
                )
            )
        responses = self.context_gen.client.batch_query_gpt_5(
            prompts,
            model=model,
            reasoning_effort=reasoning_effort,
            poll_interval=poll_interval,
        )

        for idx, response in enumerate(responses):
            rnd_data = self.context_gen.parse_response(response)
            with open(f"{log_dir}/data_gen_{idx:03d}.json", "w") as f:
                f.write(json.dumps(rnd_data, indent=2))
            with open(f"{log_dir}/graph_gen_{idx:03d}.json", "w") as f:
                f.write(json.dumps(rnd_data["graph"], indent=2))

    def generate_example_plans(
        self,
        generated_data: List[str],
        log_dir: str,
    ) -> None:
        """_summary_

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
        """

        Path(log_dir).mkdir(parents=True, exist_ok=True)

        data_counter = 0

        for idx, data_path in enumerate(generated_data):
            try:

                with open(data_path) as f:
                    data = json.load(f)

                # INVARIANT: the planner MUST only see the natural-language task text.
                # Never pass `answer`, `acceptance_criterion`, `init_node`, or any
                # other task-dict field to SPINE or to run_planning — that would
                # leak ground truth into the rollout traces and contaminate
                # training data.
                tasks = [entry["task"] for entry in data["tasks"]]
                assert all(
                    isinstance(t, str) for t in tasks
                ), "task field must be a plain string"

                print(f"Generating example data for tasks: {tasks}")

                graph = data["graph"]
                init_location = graph["robot_location"]
                assert isinstance(graph, dict)

                for task_idx, task in enumerate(tasks):
                    graph_handle = GraphHandler(graph=graph, init_node=init_location)
                    graph_data_gen = graph_sim.GraphSim(graph_handle)
                    unknown_pct = self.unknown_pcts[task_idx % len(self.unknown_pcts)]
                    graph_data_gen.randomly_remove_nodes(pct=unknown_pct)
                    log_name = f"{log_dir}/sample_{idx:03d}_{task_idx:03d}.json"
                    planner = SPINE(
                        graph=graph_data_gen.partial_graph, log_name=log_name
                    )
                    out = self.planning_sim.run_planning(
                        llm_planner=planner, task=task, graph_data_gen=graph_data_gen
                    )
                    # some simple verification. Mark plans that don't come up with an answer
                    try:
                        if not out.response["plan"][-1][0].startswith("answer"):
                            os.rename(
                                log_name, log_name.replace(".json", "_failed") + ".json"
                            )
                    except:
                        os.rename(
                            log_name, log_name.replace(".json", "failed") + ".json"
                        )

                    data_counter += 1
            except Exception as ex:
                print(f"data generation produced exception:")
                raise ex

        utils.aggregate(
            root_dir=log_dir,
            glob_str="sample*json",
            out_file=f"{log_dir}/formatted.json",
        )
