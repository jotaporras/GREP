import json
import os
from pathlib import Path
from typing import List, Union

from spine.mapping.graph_util import GraphHandler
from spine.spine import SPINE

from prism.data import graph_gen, graph_sim, planning_sim, utils


class DataGenerator:
    n_graph_gen_attempts = 10

    def __init__(
        self,
        graph_unknown: Union[int, List[int]],
    ):
        self.unknown_pcts = graph_unknown
        self.context_gen = graph_gen.TaskGraphGen()
        self.planning_sim = planning_sim.PlanningSim()

    def populate_graphs_and_tasks(
        self,
        base_graphs: List[str],
        log_dir: str,
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
                    rnd_data = self.context_gen.get_tasks(
                        base_graph=base_graph, previous_tasks=previous_tasks
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

            if idx <= 7:
                continue

            try:

                with open(data_path) as f:
                    data = json.load(f)

                tasks = [entry["task"] for entry in data["tasks"]]

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
                print(f"data generation produced exception: {ex}")

        utils.aggregate(
            root_dir=log_dir,
            glob_str="sample*json",
            out_file=f"{log_dir}/formatted.json",
        )
