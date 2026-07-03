#!/bin/bash
# Usage: run each command in a separate terminal/tmux pane with desired CUDA_VISIBLE_DEVICES.
# Mirrors e3_run_experiment.sh but points at the 80/20 split of the
# 20260428 multi-turn rollout dataset (see split_train_val.py).

python -m prism.training.train_v2 experiments/e4_new_training_data_2/e4_rpearl_llm_no_edges.yaml
python -m prism.training.train_v2 experiments/e4_new_training_data_2/e4_rpearl_llm_gt_no_edges.yaml
python -m prism.training.train_v2 experiments/e4_new_training_data_2/e4_llm.yaml
