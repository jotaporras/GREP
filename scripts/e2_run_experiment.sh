#!/bin/bash
# Usage: run each command in a separate terminal/tmux pane with desired CUDA_VISIBLE_DEVICES

python -m prism.training.train_v2 experiments/e2_rpearl_improvements/e2_rpearl_llm.yaml
python -m prism.training.train_v2 experiments/e2_rpearl_improvements/e2_llm.yaml
python -m prism.training.train_v2 experiments/e2_rpearl_improvements/e2_rpearl_llm_no_edges.yaml

