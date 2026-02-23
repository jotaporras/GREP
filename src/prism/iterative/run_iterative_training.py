#!/usr/bin/env python
"""Run the iterative training pipeline."""
from transformers import HfArgumentParser

from prism.iterative import iterative_training
from prism.training import train

if __name__ == "__main__":
    parser = HfArgumentParser((iterative_training.IterativePipelineConfig, train.TrainConfig))
    pipeline_args, train_args = parser.parse_args_into_dataclasses()
    iterative_training.run_iterative_training(pipeline_args, train_args)
