#!/bin/bash

set -e

python prism/training/train_v2.py \
  --name "Overfitting" \
  --checkpoint_dir /home/arushar/snap/snapd-desktop-integration/current/Documents/GitHub/GREP-PRISM/output/training/ \
  --data /home/arushar/snap/snapd-desktop-integration/current/Documents/GitHub/GREP-PRISM/data/eval/gpt_gen_formatted.json \
  --wandb_project GREP-PRISM \
  --wandb_run_name Overfitting \
  --wandb_tag grep-prism \
  --learning_rate 5e-4 \
  --debug True \
  --dataset_proportion 0.8 \
  --pe_hidden_channels 64 \
  --pe_num_layers 3 \
  --num_samples 50 \
  --dropout 0.0 \
  --k 5 \
  --use_layer_norm True \
  --freeze_llm True
