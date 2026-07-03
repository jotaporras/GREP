# GREP-PRISM

Research code for **graph-augmented LLM planners** on scene-graph navigation
tasks. A Gemma-4 language model plans routes over a scene graph described in its
prompt; the graph's structure is additionally delivered through a *graph
channel* — either node-level positional encodings added into the token stream,
or a structural bias on the attention logits — and we study when that channel
lets the model plan without a textual edge list.

Architectures (select with `gnn.arch=...`):

| arch | graph channel |
|---|---|
| `llm` | none — text-only baseline |
| `rpearl_llm`, `rpearl_gt_llm`, `gt_llm` | additive: GNN/Graph-Transformer node encodings Ψ injected at node-name token positions |
| `graph_mask_llm` | parameter-free adjacency mask on attention logits |
| `learnable_graph_mask` | learnable relative-PE attention bias `α·A + (1−α)·sim(ΨΨᵀ)` |

Tasks and scene graphs follow the SPINE navigation setting; answers are graded
by a deterministic path-validity walk over the graph.

## Installation

```bash
git clone <this-repo> && cd GREP-PRISM
conda create -n GREP-PRISM python=3.10 uv pip -c conda-forge
conda activate GREP-PRISM
uv pip install -r requirements.txt
uv pip install -e .
```

Models are pulled from the HuggingFace Hub (Gemma-4 requires accepting the
license). Training data ships with the repo under
`data/revised/gen/nav100_n30_gemma_data/`.

## Running

Runs are Hydra-driven; every parameter (with docs) lives in
`experiments/base_config.yaml`, experiment files override deltas, and any field
can be overridden on the CLI.

```bash
# Smoke test the stack (Gemma-4-12B, 4 optimizer steps, 1 eval graph, no wandb)
python -m prism.training.train_v3 --config-name=smoke/smoke_gemma12b

# Main result: adjacency-mask planner WITHOUT a textual edge list, trained with
# generation-consistent injection (prompt_only) — vs. the text-only baseline.
python -m prism.training.train_v3 --config-name=refactor_verify/rv_gmask_noedges \
    data.injection_scope=prompt_only eval.num_graphs=-1
python -m prism.training.train_v3 --config-name=e9_baseline_llm_no_edges eval.num_graphs=-1
```

Each run writes a checkpoint dir (`train_config.json`, adapter/PE weights) and
its generation-eval results under `<run_dir>/eval_logs/`. `eval/accuracy` in the
logs is the headline metric: fraction of held-out navigation tasks answered with
a valid route.

To re-evaluate a saved checkpoint on other graphs:

```bash
python -m prism.eval.scalability_evaluation --checkpoint <run_dir> \
    --graphs data/revised/gen/nav100_n30_gemma_data/split/test_graphs
```

## Layout

```
src/prism/        the package: models/ (architectures), data/ (prompting +
                  collators + generation), training/ (train_v3 entrypoint),
                  eval/ (grading, metrics, checkpoint reload)
experiments/      Hydra configs (base_config.yaml = the documented schema;
                  old/ = frozen legacy series)
scripts/          current launchers + diagnostics (old/ = frozen legacy)
data/             navigation corpora and held-out eval graphs
docs/             internal documentation (metrics catalog, ...)
notebooks/        lab notes for the experiment series
```

Built on PRISM-style planner training and the SPINE task setting.
