import argparse
import json
from pathlib import Path

from prism.data.data_gen import DataGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--n-tasks", type=int, default=25)
    parser.add_argument("--name", type=str, default="non-iterative-data")
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.31)
    parser.add_argument("--data-dir", type=str, help="path to base graphs")

    args = parser.parse_args()

    Path(args.name).mkdir(parents=True, exist_ok=True)
    with open(f"{args.name}/data_gen_params.json", "w") as f:
        json.dump(vars(args), f)

    graphs = Path(args.data_dir).glob("*json")

    log_dir = args.name

    unknown_pcts = [0] * 10
    # unknown_pcts = [0, 5, 10, 15] * 10

    # unknown_pcts = [10, 15] * 10
    # n_regions_list = [10, 15, 20] * 10  # np.arange(20, 30, 2)
    # n_objects_list = [3, 6, 9] * 10  # np.arange(10, 30, 1)

    data_generator = DataGenerator(
        graph_unknown=unknown_pcts,
    )

    graph_dir = Path(log_dir) / "populated_graphs"
    graph_dir.mkdir(exist_ok=True)

    # data_generator.populate_graphs_and_tasks(graphs, log_dir=graph_dir)

    plan_dir = Path(log_dir) / "generated_plans"
    generated_data = sorted(graph_dir.glob("*data_gen*json"))
    print(f"generated_data path: {generated_data}")
    data_generator.generate_example_plans(
        generated_data=generated_data, log_dir=plan_dir
    )

    for graph in graphs:
        with open(graph) as f:
            base_graph = json.load(f)

        data_generator.generate(
            log_dir=log_dir,
            base_graph=base_graph,
            n_samples=args.n_samples,
            n_tasks=args.n_tasks,
         description=args.description,
        )
