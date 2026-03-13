"""Debug entry point for train_v2 — equivalent to:
    python -m prism.training.train_v2 experiments/e2_llm.yaml
Run this file directly in the debugger to set breakpoints inside train_v2.
"""
import sys

from transformers import HfArgumentParser

from prism.training.train_v2 import TrainConfig, train_model

CONFIG = "../experiments/e2_llm.yaml"

if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else CONFIG
    parser = HfArgumentParser(TrainConfig)
    (cfg,) = parser.parse_yaml_file(config_file)
    print(cfg)
    train_model(cfg, config_file=config_file)
