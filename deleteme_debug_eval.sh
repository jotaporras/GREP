#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=1 conda run -n GREP-PRISM python -m prism.training.train_v2 experiments/deleteme_debug_eval.yaml
