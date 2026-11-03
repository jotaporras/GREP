---

## experiment: e22 geometric scale-up — design
date: 2026-11-02
status: design; implementation handed to an Opus agent (nothing generated or submitted yet)
sources: robot graph analysis in [2026-11-01 hospital_graphs_analysis.ipynb](../notebooks/2026-11-01%20hospital_graphs_analysis.ipynb); hospital export `data/hospital_graphs/README.md`; e21 result in [2026-08-26 e21_oracle_scale_v2c_results.md](2026-08-26%20e21_oracle_scale_v2c_results.md)
wandb: alelab/GREP-PRISM, tag `e22_geometric_scale`

# e22 — floor-plan graphs and robot-scale routes

**TL;DR.** The deployed model (e21 v2c, `4ghta3jm`, 94.0% on the frozen n60 set) scores
2/30 on the robot's 7–12-hop routes while untuned Gemma-4-31B with the edge list in the
prompt scores 26/30 (hospital_v4 sweep, Arjun 2026-10-16). The robot graph is out of
distribution on three axes, and the training generator cannot reach it by adding nodes.
e22 replaces the SBM skeleton generator with a floor-plan generator, retargets the
long-hop sampler at 6–7 hops with a 15-hop tail, adds named starts that differ from
`robot_location`, and trains the v2c recipe on ~11k samples. Acceptance is the hospital
sweep itself, run through our own harness.

## 1. What the robot graphs look like (from the notebook)

| | hospital_v4 | n60_v3 (59 graphs) |
|---|---|---|
| regions / objects | 37 / 14 | 48 / ~15 |
| region edges | 40 | 150 |
| mean degree | 2.16 | 6.25 |
| dead ends (degree 1) | 24% | 0.1% |
| bridge edges | 72% | 0% |
| cycles (E − N + 1) | 4 | ~103 |
| diameter | 20 | 4.5 |
| clustering | 0.05 | 0.32 |

Three shifts, in causal order:

1. **Density.** Routes are long because the graph is a tree with a few loops. In a dense
   random graph the diameter grows with log N: the SBM at 12×8 communities (96 regions)
   still has diameter 4.1, because every community pair gets ~3 bridge edges and every
   region sees ~8 neighbours. N alone cannot produce long routes.
2. **Route length.** 64% of hospital_v4 region pairs and 58% of the sweep tasks are ≥6
   hops. The v2c training routes are 1.1% ≥6 hops and 0% ≥8. The 7–12 bucket has zero
   training support.
3. **Start grounding.** 58/60 sweep tasks start at a *named* region that is not
   `robot_location` (which stays `plaza_1`), and 3% say "starting area". In every GREP
   corpus so far, the pipeline sets `robot_location := init_node` per sample
   (`src/prism/data/data_gen.py:610,656`), so "start = robot_location" was always a
   valid shortcut. The sweep removes it.

Smaller: names are `<zone>_<kind>_<k>` with up to 6 siblings per type (training ≤3);
sweep tasks are all object-goal with no via/avoid; v3/v4 main tasks have 12–20% via and
20–25% avoid.

## 2. Generator: BSP floor plan replaces the SBM

Spatial models compared on degree, dead ends, bridges, cycles and diameter
(`docs/hospital_graphs_generator_candidates.csv`, notebook §8b). Random geometric graphs,
kNN, Delaunay and Gabriel graphs are too dense and too clustered (triangles; no dead
ends). Two models match: Euclidean MST + a few extra edges, and the **BSP floor plan**.
The floor plan is chosen because it is the hospital's own generative process and gives
zones, corridors and physical coordinates for free.

**BSP floor plan** (new `--layout floorplan` in `scripts/generate_eval_graphs.py`):

1. Recursively split a rectangle (aspect ~1.4) into `n_regions` rooms. Split the largest
   room each time; split position uniform in [0.35, 0.65]; prefer the long axis.
   Optional `--corridor-bias b`: with probability `b` the first split along an axis
   carves a thin corridor room (aspect ≥ 4) that the next rooms hang off. Default 0.3.
2. Rooms that share a wall segment of length ≥ 2% of the plan are *candidate* doors.
3. Doors = a random spanning tree over the candidate-door graph + `door_extra` of the
   remaining candidates. `door_extra` is drawn per graph from `U[0.05, 0.20]`.
4. Zones = the rooms under each top-level split node at depth `--zone-depth` (default 3,
   i.e. 8 zones). Written to `_metadata.zones` and used by the populate prompt as a
   naming hint (`<zone>_<kind>_<k>`), replacing `_metadata.community_assignments`.
5. Coordinates = room centroids in metres (scale the plan to ~2.5 m² per region... keep
   the existing coordinate schema `[x, y]`). Objects: Poisson per region as today,
   `--object-rate 0.4`.

Targets the generator must hit (mean over ≥20 seeds at 100 regions; measured in the
corpus audit before any training):

| statistic | target | SBM today |
|---|---|---|
| mean degree | 2.1–2.6 | 6.3 |
| dead-end fraction | 0.15–0.30 | 0.00 |
| bridge fraction | 0.3–0.7 | 0.00 |
| cycles per region | 0.05–0.25 | 2.2 |
| diameter | 18–28 | 4.4 |
| clustering | ≤ 0.15 | 0.32 |

Simulated BSP at 100 regions: `door_extra` 0.05 → degree 2.10, dead 0.30, bridge 0.64,
diameter 25.6; `door_extra` 0.15 → 2.34 / 0.22 / 0.33 / 21.3. The `U[0.05, 0.20]` draw
spans the hospital (bridge 0.72) and somewhat loopier buildings.

**Fallback:** `--layout emst --extra-frac 0.05..0.15` (Delaunay MST + extra edges) if the
BSP code proves fragile; same statistics, no zones.

**Size mix** (skeleton count by region count; 600 skeletons):

| regions | 40 | 70 | 100 | 130 |
|---|---|---|---|---|
| skeletons | 90 | 150 | 240 | 120 |

40 = hospital_v4 size; 130 gives headroom above the "~100 nodes" the robot team plans.

## 3. Tasks

Per graph **18 tasks: 16 long-hop + 2 standard** (`N_TASKS=18 N_LONGHOP=16`), oracle
routes (`--oracle-paths`, NetworkX; via/avoid already honoured by `_oracle_route`).

**Hop targeting.** Replace the single `--longhop-max-boost` with
`--longhop-hop-weights "1:0.04,2:0.04,3:0.06,4:0.08,5:0.12,6:0.17,7:0.17,8:0.08,9:0.07,10:0.05,11:0.04,12:0.03,13:0.02,14:0.02,15:0.01"`:
hop `h` is drawn with weight `w_h` restricted to buckets that exist in the graph
(`h ≤ diameter`), then an (init, goal) pair at exactly that distance. `max_boost` stays
as the legacy path (`1.0` = byte-identical old stream). Target train histogram: mode at
6–7, 35% in 6–7, 15% ≥10, max 15.

**Start grounding (new).** `--longhop-named-start-frac 0.5`: half of the long-hop tasks
mark `keep_robot_location: true`. For those, (a) the task text must name the start
region explicitly (plain name, paraphrase, or ordinal sibling — reuse the directive text
from `--grounding-directives`, start side only), (b) `init_node` = the named region,
(c) the pipeline does **not** overwrite `robot_location` for that sample
(`data_gen.py:610,656` and the batched populate path — audit by corpus md5, see
`batched-populate-drops-flags`). `robot_location` of the skeleton is a random region, not
`region_1`. The other half keep today's behaviour with `--longhop-start-area-frac 0.4`
applied to them (so ≈20% of all long-hop tasks say "starting area"; the hospital main
tasks are 80%, the sweep 3%; the n60 eval 63%).

**Constraints.** Keep `--longhop-allow-avoid`. Allow waypoints on ~20% of long-hop tasks
(`--longhop-allow-via`): the waypoint must lie on *no* shortest init→goal path and the
reference route is init→via→goal by oracle; the recorded `hops` becomes the constrained
length. Today waypoints are forbidden on long-hop tasks.

**Goals.** ≥50% of long-hop tasks phrase the goal as an object ("the area with the drinks
machine"), as the sweep does (100%). `init_node`/regex still name the region.

Expected corpus: 600 × 18 ≈ 10,800 samples before populate loss (v2c lost 4.6%).

## 4. Evaluation sets (all no-ICL, `cross_eval` family for comparisons)

| set | what | why |
|---|---|---|
| A | frozen n60_v3, 84 questions | regression vs 94.0% |
| B | new frozen `geo_test`: 8 held-out skeletons (2 per size) × 12 tasks, stratified over hops 1–15, 50% named starts | in-distribution measure for the new regime |
| C | `data/hospital_graphs/hospital_v4_length_sweep.json` (60), `hospital_v4.json` (10), `hospital_v3.json` (8), `_neutral` (60) | the robot's own test; acceptance |
| D | **baseline**: `4ghta3jm` on B and C before any training | confirms the diagnosis; expect ≈ Arjun's 2/30 on 7–12 hops |

Harness check for C: `src/prism/data/rl_dataset.py:34` sets
`robot_location = task["init_node"]` at eval time. The hospital files carry
`robot_location: plaza_1`; JP (2026-11-02): that value is a placeholder, so set C
runs in the default mode (robot at each task's start). `eval.keep_robot_location=true`
stays available as an optional second measurement.

Grader: simple-path check and avoid/via checks already exist in `path_validator`; make
sure `avoid_ok=false` fails the sample (Arjun's "fourth hall" case passed the text grader).

## 5. Training cells (betty `dgx-b200`, one GPU each)

| cell | arm | base | epochs | est. wall clock |
|---|---|---|---|---|
| `e22_geo_hop_depth` | hop_depth, v2c recipe (lr 2.5e-4, route_only, binary edge weights) | gemma-4-31B-it | 4 | ~14 h |
| `e22_geo_text_edges` | text baseline, edge list in prompt | gemma-4-31B-it | 4 | ~20 h |
| ~~`e22_geo_hop_depth_8b`~~ | deferred (2026-11-02): the smaller base came from the email to Arjun, not from a decision; no sub-31B Gemma has a stage-1 LoRA in the repo, so it is a separate protocol. Stays with the robot deploy plan. | | | |

Per-epoch built-in eval on A; B and C via `cross_eval` on the saved checkpoints (the
trainer takes one eval dir). Checkpoint pick = best epoch on B. Launcher: copy of
`scripts/e20_path_only_sft.sbatch` with `DATASET=geo_v1` and `EVAL_GRAPHS` override.

## 6. Success criteria

- Set C, hop 7–12: ≥ 20/30 (text-edges Gemma: 26/30). Set C, hop 1–6: ≥ 25/30.
- Set A: ≥ 88% (within the ~±5 pt seed band of 94.0).
- Set B: hop_depth within 5 pts of `e22_geo_text_edges`.
- Corpus audit before training: degree/dead-end/bridge/diameter inside the §2 target
  band; hop histogram mode at 6–7 with ≥15% ≥10 hops; named-start rate 50±5%;
  `robot_location ≠ init_node` on exactly the named-start samples; md5 differs from v2c.

## 7. Implementation steps (for the implementing agent)

1. `scripts/generate_eval_graphs.py`: `--layout {sbm,floorplan,emst}` (default `sbm`,
   byte-identical output for old seeds), `--n-regions`, `--door-extra-min/max`,
   `--corridor-bias`, `--zone-depth`, `--robot-location random|region_1`. Write
   `_metadata.layout`, `.door_extra`, `.zones`, `.n_regions`. Keep `community_assignments`
   populated from zones so no downstream reader breaks; grep for its consumers first.
2. `src/prism/data/graph_gen.py`: `sample_longhop_constraints(..., hop_weights=None)`
   (dict `{hops: weight}`; `None` = legacy); `validate_longhop_tasks` learns the via rule.
   `get_tasks`/`build_prompt`: named-start marking (`keep_robot_location`), via
   directive, object-goal directive, zone-naming hint from `_metadata.zones`.
3. `src/prism/data/data_gen.py`: honour `keep_robot_location` in both
   `_commit_plan_sample` and the path-only/oracle sample writers; thread every new flag
   through `DataGenerator`, `generate_data_spine.py` argparse, **and**
   `populate_graphs_and_tasks_batch` (the vLLM path that dropped flags in e21).
4. `scripts/nav_n100_gemma_generate.sh`: pass `LAYOUT`, `N_REGIONS`, door/corridor knobs
   to stage 1; `EXTRA_GEN_ARGS` carries the task flags.
5. `scripts/e22_geo_v1_generate.sbatch`: 4-shard array, 150 skeletons each, size mix per
   §2 via a seed→size table, `DATA_ROOT=$ALELAB_DRIVE/GREP-PRISM/data/n_geo_v1/shard{i}`;
   reuse `scripts/e21_merge_shards.py`. Hold out 2 skeletons per size as `geo_test`.
6. `scripts/e22_corpus_audit.py`: degree/dead-end/bridge/diameter per graph, hop
   histogram, named-start and starting-area rates, `robot_location≠init_node` count, md5.
7. Eval: `eval.keep_robot_location` flag in `rl_dataset.py`; a tiny
   `scripts/hospital_eval_set.py` that drops the four hospital JSONs into a
   `test_graphs/`-shaped dir for `cross_eval`.
8. Tests: extend `tests/test_e21_longhop_sampler.py` (hop_weights sums, legacy
   equivalence at `None`, via validator) and add `tests/test_floorplan_generator.py`
   (connected, planar adjacency, statistics inside the §2 band over 20 seeds,
   `--layout sbm` byte-identical to the current output for seeds 201–203).
9. `scripts/e22_geo_sft.sbatch`: `DATASET=geo_v1` cells from §5 with `EVAL_GRAPHS`
   override and tag `e22_geometric_scale`.

Local checks possible on the Mac (`conda run -n GREP-PRISM-v3`): generator + statistics,
sampler tests with stubbed `prism.data` imports as in e21. Pytest of the full suite and
any generation or training need betty (auth currently down: `ssh betty true` first).

## 8. Risks

- Populate LLM must name 100–130 regions with zone prefixes without collisions; the rename
  map validator already rejects duplicates, so expect a higher skeleton rejection rate
  than v2c's 4.6%. Budget 700 skeletons if the first shard shows >10% loss.
- Prompt length at 130 regions + 50 objects is ~2× v2c; check `max_length` in the trainer
  config before launch.
- Named starts make the task-author's `init_node` the only ground truth for the start;
  `validate_longhop_tasks` must keep rejecting tasks whose text does not single out the
  named region (ordinal/sibling ambiguity).
- hop_depth taps see 3 GT layers; Ψ is numerically the binary adjacency mask, so routing
  depth is decode-side and should transfer. If set C fails at ≥10 hops while B passes,
  the gap is prompt/grounding, not depth — do not change the architecture in this round.

## 9. Implementation status (2026-11-02)

An implementing subagent made the code changes below on the Mac. Nothing is committed and no jobs were submitted.

Files changed or created:

| file | change |
|---|---|
| `scripts/generate_eval_graphs.py` | `--layout {sbm,floorplan,emst}`, `--n-regions`, door/corridor/zone/area knobs, `--robot-location`; BSP port of `bsp_floorplan`, EMST port of `mst_plus`, `layout_statistics`, `TARGET_BAND_100` |
| `src/prism/data/graph_gen.py` | `hop_weights` sampler, `object_goal_frac`, `assign_longhop_roles` (named start, via), via validator, prompt blocks (via, named start, object goal, zone naming) |
| `src/prism/data/data_gen.py` | role flags on `DataGenerator`; one `_prompt_flags()` for both populate paths; `keep_robot_location` in the oracle and path-only writers |
| `scripts/training_data_generation/generate_data_spine.py` | `--longhop-hop-weights`, `--longhop-named-start-frac`, `--longhop-allow-via`, `--longhop-object-goal-frac` |
| `scripts/nav_n100_gemma_generate.sh` | `LAYOUT`, `N_REGIONS`, `SIZE_TABLE`, door/corridor/zone/area knobs, `ROBOT_LOCATION`, overridable `OBJECT_RATE` |
| `scripts/e22_geo_v1_generate.sbatch` | new: 4 training shards + geo_test shard |
| `scripts/e22_corpus_audit.py` | new: §6 corpus gate, exit 1 on any FAIL |
| `scripts/hospital_eval_set.py` | new: set C dir; "never entering" → "without using"; `hops`; optional `keep_robot_location` |
| `scripts/e22_geo_sft.sbatch` | new: §5 cells, 4 epochs, `MAX_SEQ_LENGTH` default 16384 |
| `src/prism/data/rl_dataset.py` | `keep_robot_location` argument and per-task mark |
| `tests/test_e21_longhop_sampler.py`, `tests/test_floorplan_generator.py` | e22 tests (below) |

Tests run on the Mac, with `spine` stubbed:

| suite | result |
|---|---|
| `tests/test_e21_longhop_sampler.py` | 43 passed |
| `tests/test_floorplan_generator.py` | 16 passed (6 are the sbm md5 checks for seeds 201–203) |
| e20 route-only, edge notation, rename contract | 56 passed |
| `tests/test_populate_token_budget.py` | 9 failed before these edits (`DEFAULT_POPULATE_MAX_TOKENS` is not in `src/`) |
| eval_scoring, path_metrics_plumbing, goto_ratify, scalability, eval_render | not collected (need real `spine`): run on betty |

Generator statistics (20 seeds, `door_extra` ~ U[0.05, 0.20], corridor bias 0.3):

| regions | degree | dead end | bridge | cycles/N | diameter | clustering |
|---|---|---|---|---|---|---|
| §2 band (100) | 2.1–2.6 | 0.15–0.30 | 0.3–0.7 | 0.05–0.25 | 18–28 | ≤ 0.15 |
| 100 | 2.31 | 0.235 | 0.40 | 0.166 | 20.3 | 0.065 |
| 40 | 2.25 | 0.285 | 0.49 | 0.148 | 12.9 | 0.071 |
| EMST 100 | 2.48 | 0.108 | 0.19 | 0.248 | 25.7 | 0.099 |

Decisions on the open items (JP, 2026-11-02; patches by Claude, Mac, uncommitted):

- Done: `src/prism/eval/evaluate.py` reads the mark. `EvalSample` gained an optional `robot_location`; `construct_eval_samples_from_dict(..., keep_robot_location)` sets it from the graph file for flagged tasks or for every task; the eval loop resets the simulator to it and overrides `robot_location` in the prompt graph, while grading still starts the route at `init_node`. Unflagged tasks are byte-identical to before. Wired through `data.load_samples_by_graph` / `load_eval_samples_by_graph`, `train_v3.py`, `train_rl.py` (`rl.keep_robot_location`), `eval.keep_robot_location: false` in `experiments/base_config.yaml`, and `KEEP_ROBOT_LOCATION` in `scripts/e22_geo_sft.sbatch`.
- Done: `path_validator._AVOID` parses "never entering|using|passing through|going through". `scripts/hospital_eval_set.py` now copies the criteria verbatim. Test: `test_structured_avoid_never_entering_phrasing`.
- Accepted: the via role makes `hops` the constrained length (≥10-hop share 0.18 → 0.24). The robot's own route lengths are measured the same way.
- Deferred to the corpus audit: sibling-ambiguity check for named starts.
- Dropped for v1: the 8B cell (see §5) and EMST in the size mix (under the band on dead ends 0.11 and bridges 0.19; `--layout emst` stays as a code path).
- Accepted: 40-region floor plans at diameter 12.9 (hospital_v4: 20); the 70–130-region graphs carry the long routes. Accepted: `MAX_SEQ_LENGTH` 16384; check memory in the first epoch log.
