import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from transformers import HfArgumentParser

from prism.training.train_v2 import TrainConfig, train_model

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEBUG_YAML = (
    _PROJECT_ROOT
    / "experiments"
    / "e2_rpearl_improvements"
    / "e2_rpearl_llm_no_edges.yaml"
)

load_dotenv(_PROJECT_ROOT / ".env")

if __name__ == "__main__":
    os.chdir(_PROJECT_ROOT)

    parser = HfArgumentParser(TrainConfig)
    if len(sys.argv) == 1:
        config_path = str(_DEBUG_YAML)
        (cfg,) = parser.parse_yaml_file(config_path)
        config_file = config_path
    elif len(sys.argv) == 2 and sys.argv[1].endswith((".yaml", ".yml")):
        (cfg,) = parser.parse_yaml_file(sys.argv[1])
        config_file = sys.argv[1]
    else:
        (cfg,) = parser.parse_args_into_dataclasses()
        config_file = None

    print(cfg)
    train_model(cfg, config_file=config_file)
