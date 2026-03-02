# GREP-PRISM

Work-in-progress research code for a new paper extending [PRISM](PRISM_README.md) with graph-augmented LLM planning.

The core idea is to augment a small language model (SLM) planner with structural information from the scene graph. Rather than relying solely on the text serialization of the graph, we compute node-level positional encodings via a GNN (R-PEARL) and inject them directly into the LLM's token embeddings at the positions of matching node names. This lets the model reason over graph topology without changing the prompt format.

Two architectures are supported:

- `rpearl_llm` — graph-augmented LLM: R-PEARL GNN encodings are injected into token embeddings at training and inference time.
- `llm` — text-only baseline: standard LoRA SFT on the same data, no graph injection.

The planner is evaluated on multi-turn SPINE planning tasks using scene graphs from the SPINE simulator.

---

## Installation

```bash
git clone <this-repo> && cd GREP-PRISM
conda create -n GREP-PRISM python=3.10 uv pip -c conda-forge
conda activate GREP-PRISM
uv pip install -r requirements.txt
uv pip install -e .
```

### SPINE dependency

SPINE is required for data generation and evaluation. Clone the PRISM branch:

```bash
git clone git@github.com:KumarRobotics/SPINE.git
cd SPINE
git checkout feature/prism
git pull
uv pip install -e .
```

Ensure SPINE is importable before running any data generation or evaluation scripts.

---

## Generating training data

Data generation synthesizes scene graphs and tasks using GPT, runs a GPT-enabled SPINE planner, and saves the resulting conversations to a `formatted.json` file.

```bash
python scripts/generate_data_spine.py --n-samples 500 --n-tasks 3 --name my_run
```

The aggregated dataset is saved under `data/gen/<name>/formatted.json`.

---

## Training

Training is driven by YAML config files and the unified entry point `prism.training.train_v2`.

### Running a single experiment

```bash
python -m prism.training.train_v2 experiments/e2_rpearl_llm.yaml
```

### Running multiple experiments sequentially

```bash
./scripts/run_experiment.sh experiments/e2_llm.yaml experiments/e2_rpearl_llm.yaml
```

### Smoke test (fast sanity check)

```bash
python -m prism.training.train_v2 experiments/e2_qwen05b_smoke.yaml
```

This runs on 1% of the data with a tiny Qwen 0.5B model to verify the pipeline end-to-end.

### Config reference

All fields map directly to the `TrainConfig` dataclass in `src/prism/training/train_v2.py`. Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `name` | — | Run name prefix (used in checkpoint path and W&B) |
| `checkpoint_dir` | — | Directory to save checkpoints |
| `data` | — | Path to training `formatted.json` |
| `architecture` | `rpearl_llm` | `rpearl_llm` (graph-augmented) or `llm` (text-only baseline) |
| `base_model` | `meta-llama/Llama-3.2-3B-Instruct` | HuggingFace model ID |
| `r` | `16` | LoRA rank |
| `epochs` | `2` | Number of training epochs |
| `learning_rate` | `2e-4` | Learning rate |
| `text_edge_list` | `present` | Whether to include edge lists in the text prompt (`present` or `none`) |
| `freeze_llm` | `false` | Freeze LLM weights and train only the GNN components |
| `bit4` | `false` | Enable 4-bit quantization (QLoRA) |
| `debug` | `false` | Use only `dataset_proportion` of the data |
| `dataset_proportion` | `0.1` | Fraction of data to use when `debug=true` |
| `max_seq_length` | `2048` | Maximum sequence length |
| `wandb_project` | `SLM-distill` | W&B project name |
| `wandb_run_name` | `spine_lora` | W&B run name |
| `wandb_tag` | `spine` | W&B run tag / group |

**R-PEARL GNN parameters** (only used when `architecture: rpearl_llm`):

| Parameter | Default | Description |
|---|---|---|
| `d_model` | `3072` | Output dimension of the GNN (must match LLM hidden size) |
| `pe_hidden_channels` | `256` | Hidden dimension of the GCN backbone |
| `pe_num_layers` | `3` | Number of GCN layers |
| `num_samples` | `40` | Number of random feature samples (M) for R-PEARL |
| `k` | `3` | TAGConv polynomial order |
| `dropout` | `0.1` | GCN dropout rate |
| `use_layer_norm` | `true` | Use layer norm on GNN output |


---

## Evaluation

Evaluation is done via SPINE simulations. It is run automatically during training via `EvalCallback` (once per epoch by default). Results are saved as JSON under `<checkpoint_dir>/eval_logs/` and logged to W&B under `eval/accuracy`.

To run evaluation standalone against a saved checkpoint:

```bash
python scripts/eval.py
```

---

## Key changes from PRISM

### New training pipeline

`src/prism/training/train_v2.py` replaces `scripts/train_llama3_llora.py` as the primary trainer. It uses HuggingFace `SFTTrainer` + PEFT LoRA directly (no Unsloth dependency for training), accepts YAML configs via `HfArgumentParser`, and supports both the graph-augmented and text-only architectures from a single entry point.

### Graph-augmented architecture (R-PEARL + LLM)

Three new modules implement the graph injection mechanism:

- `src/prism/models/gcn.py` — TAGConv-based GCN backbone with skip connections and layer norm.
- `src/prism/models/r_pearl.py` — `RandomGNNPositionalEncodings`: runs M random feature vectors through the GCN and averages the results to produce stable, structure-aware node embeddings.
- `src/prism/models/gnn_llm.py` — `GraphAugmentedLLM`: wraps any HuggingFace causal LM, tokenizes node names, locates their positions in the input token sequence (`bucketize_prompt`), and adds the projected R-PEARL embeddings to the corresponding token embeddings before the LLM forward pass.

### Custom data collator

`src/prism/data/data_col.py` provides `DataCollatorForGraphAugmentedLLM`, which parses the scene graph from the conversation text into a PyTorch Geometric `Data` object at collation time. It also supports `text_edge_list: none` to strip the textual edge list from the prompt, so the model must rely on the GNN for connectivity information.

### YAML-driven experiment configs

All hyperparameters (model, data, LoRA, GNN, W&B) are specified in YAML files under `experiments/`. `scripts/run_experiment.sh` iterates over a list of configs and runs them sequentially.

### In-training evaluation

`EvalCallback` (`src/prism/eval/callbacks.py`) runs multi-turn SPINE planning evaluation at configurable epoch intervals during training. Results are logged to W&B (`eval/accuracy`) and saved as JSON files alongside the checkpoints.



