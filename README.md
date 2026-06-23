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

---

## Usage

Runs are driven by a YAML config in `experiments/`; override any field with `--key value`.

```bash
# Train (smoke config: tiny Qwen-0.5B, runs locally)
python -m prism.training.train_v2 experiments/smoke/e2_qwen05b_smoke.yaml

# Evaluate only — zero-shot accuracy, no training
python -m prism.training.train_v2 experiments/smoke/e2_qwen05b_smoke.yaml --no_train true
```

