"""Static smoke test for scripts/e16_rl_gt.sbatch — no ML, no torch, no cluster.

Same doctrine as test_e14_sbatch_smoke.py: catch pre-flight failure modes
(shell typos, Hydra overrides naming nonexistent keys, a drifted campaign tag)
in under a second instead of after a B200 allocation has loaded a 31B model.
"""

import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "e16_rl_gt.sbatch"
BASE_CONFIG = REPO / "experiments" / "base_config.yaml"
E16_CONFIG = REPO / "experiments" / "e16_rl_config.yaml"


def _text(p):
    return p.read_text()


def test_script_exists():
    assert SCRIPT.exists()


def test_shell_syntax_is_valid():
    r = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr.strip()


def test_fails_fast():
    assert re.search(r"^set -euo pipefail$", _text(SCRIPT), re.M)


def test_campaign_tag_is_locked():
    """The wandb tag is LOCKED as e16_rl_training — a literal, not a knob."""
    body = _text(SCRIPT)
    assert re.search(r"^TAG=e16_rl_training$", body, re.M), \
        "tag must be the hardcoded literal e16_rl_training"
    assert 'wandb.tag="$TAG"' in body


def _config_keys():
    """Dotted key paths across base_config.yaml + e16_rl_config.yaml (the rl
    surface lives only in the latter)."""
    out = set()

    def walk(node, prefix=""):
        if not isinstance(node, dict):
            return
        for k, v in node.items():
            dotted = f"{prefix}{k}"
            out.add(dotted)
            walk(v, f"{dotted}.")

    for cfg_path in (BASE_CONFIG, E16_CONFIG):
        walk(yaml.safe_load(_text(cfg_path)))
    return out


def test_hydra_overrides_name_real_config_keys():
    """Bare ``key=value`` overrides must exist in the merged config; ``+key``
    appends are exempt (they create keys — the free-form grpo/engine dicts)."""
    body = _text(SCRIPT)
    keys = _config_keys()
    overrides = re.findall(r"^\s+([a-z_]+(?:\.[a-z_]+)+)=", body, re.M)
    assert overrides, "no Hydra overrides found — parser drifted?"
    unknown = sorted({k for k in overrides if k not in keys})
    assert not unknown, f"overrides absent from configs: {unknown}"


def test_two_gpu_split_is_wired():
    """The engine owns cuda:0; the policy MUST be pushed to cuda:1 through both
    channels (hydra device + accelerate env) or the trainer lands on the
    engine's card and OOMs mid-run."""
    body = _text(SCRIPT)
    assert "trainer.rl.policy_device=1" in body
    assert re.search(r"^export ACCELERATE_TORCH_DEVICE=cuda:1$", body, re.M)
    assert re.search(r"^#SBATCH --gpus=2$", body, re.M)


def test_referenced_repo_files_exist():
    refs = set(re.findall(r"(?<![\w/])(experiments/[\w./-]+|scripts/[\w./-]+)",
                          _text(SCRIPT)))
    missing = sorted(r for r in refs if not (REPO / r).exists())
    assert not missing, f"referenced repo files missing: {missing}"
