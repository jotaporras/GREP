# Transferability Dataset — Data Generation

The training corpus used in `e3_new_training_data` (`data/training_data_graphs_20260422/training_data_20260422.json`) is built in two stages: a **structural skeleton** sampled from random-graph distributions, then a **semantic fill** by an LLM via the `/fill-graph` Claude skill.

## Pipeline overview

1. `scripts/training_data_generation/generate_training_graphs.sh` loops over five target sizes and calls `scripts/generate_eval_graphs.py` to produce skeleton JSONs.
2. For each skeleton, `/fill-graph <path>` rewrites placeholders (region/object names, `__FILL__` descriptions, `n_tasks` task entries) with realistic, semantically coherent content.
3. Filled graphs are aggregated (one `data_gen_*.json` per graph, 50 total) into the single training file consumed by training (`scripts/training_data_generation/aggregate_data.py` style).

## Stage 1 — Skeleton generation

Implemented in [scripts/generate_eval_graphs.py](../../scripts/generate_eval_graphs.py). All randomness derives from one `numpy.random.default_rng(seed)`.

### Step 1: Region graph — Stochastic Block Model

- Block sizes: `[nodes_per_community] * n_communities` (uniform).
- Edge-probability matrix: `intra_community_prob` on the diagonal, `inter_community_prob` off-diagonal.
- Drawn via `networkx.generators.community.stochastic_block_model`. Up to 100 reseeded retries until `nx.is_connected` holds; raises if not.

### Step 2: Region coordinates — circular layout + Gaussian jitter

- Let `scale = 5 · |V|`, `radius = scale/2`, `jitter_std = scale / (4 · n_communities)`.
- Community centroids equally spaced on a circle: `(radius cos θ_c, radius sin θ_c)` with `θ_c = 2π c / n_communities`.
- Each region's coords = its community centroid + `N(0, jitter_std²)` per axis, rounded to 1 decimal.

### Step 3: Objects — Poisson per region

- For each region, sample `n_objects ~ Poisson(object_rate)`.
- Object coords = region coords + `N(0, 2.0²)` per axis (fixed std = 2.0), rounded to 1 decimal.
- Each object is connected to exactly its parent region (`object_connections`).

### Step 4: Object descriptions — Bernoulli mask

- Each object's `description` is set to `"__FILL__"` independently with probability `description_prob`; otherwise left as `""`. This is the placeholder the fill stage rewrites.

### Step 5: Skeleton assembly

- Regions, objects, region/region edges (from SBM), object/region edges, and `robot_location = "region_1"` are written to the `graph` block.
- `tasks` is left empty; `_metadata` records every parameter, the community assignment per region, plus the realized `n_regions` / `n_objects`.

### Parameters (defaults from `generate_training_graphs.sh`)

| Parameter | Value | Role |
|---|---|---|
| `intra_community_prob` | 0.6 | SBM diagonal — within-community edge prob |
| `inter_community_prob` | 0.05 | SBM off-diagonal — cross-community edge prob |
| `object_rate` | 0.3 | Poisson rate for objects per region |
| `description_prob` | 0.05 | Bernoulli prob an object gets `__FILL__` |
| `n_tasks` | 10 | Tasks the fill stage must author |
| `seed` | 42 | Single RNG seed for the whole skeleton |

Sweeps generate one skeleton per size config `(N, n_communities, nodes_per_community)` chosen so `n_regions ≈ N / 1.3` (Poisson rate 0.3 adds ~30% objects):

```
(30, 3, 8)  (35, 3, 10)  (40, 4, 9)  (45, 4, 10)  (50, 4, 11)
```

## Stage 2 — Semantic fill (`/fill-graph`)

For each skeleton the `/fill-graph` skill replaces:

- generic `region_i` / `object_j` names with thematic, unique names,
- empty / `__FILL__` descriptions with realistic attributes,
- the empty `tasks` list with `n_tasks` multi-step tasks grounded in the populated graph.

The graph topology, coordinates, and connectivity from Stage 1 are preserved verbatim — only the textual content changes.

## Output

Final filled files live under `data/training_data_graphs_20260422/grep_training_data/graphs/data_gen_*.json` (50 graphs across the five size buckets) and are aggregated into `training_data_20260422.json` consumed by the e3 training config [e3_llm.yaml](e3_llm.yaml).

## Graph ↔ plan relationship

Populated graphs and generated plans have a **one-to-many** relationship: each graph defines `n_tasks` tasks, and a separate plan trajectory is generated per (graph, task) pair.

- `populated_graphs/data_gen_GGG.json` — a single scene graph with `graph`, `description`, and a `tasks` list (10 entries by default, each `{task, answer, init_node}`).
- `generated_plans/sample_GGG_TTT.json` — a chat-format trajectory (list of role/content messages) of the planner solving task index `TTT` on graph `GGG`, starting from that task's `init_node`.

Filename encoding is `sample_<graph_id>_<task_id>.json`, so a complete batch contains `n_graphs × n_tasks` plan files (e.g. 100 graphs × 10 tasks ≈ 1000 plans in `aggregate_20260428/20260428_data/`; a partial count indicates generation is still in progress). The per-sample conversations are concatenated into `formatted.json` / `formatted_all.json` for training consumption.
