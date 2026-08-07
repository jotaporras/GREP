"""Static smoke test for the e14 sbatch suite — no ML, no torch, no cluster.

Every check here is text/AST/YAML only, so the whole file runs in well under a second.
The point is to catch the failure modes that otherwise only surface after a job has been
queued, allocated a GPU, and loaded a 31B checkpoint:

  * a shell syntax error, or a missing ``set -euo pipefail`` so a mid-script failure is
    swallowed and the run reports success on garbage;
  * a ``key=value`` Hydra override naming a config key that does not exist (Hydra rejects
    it at startup, after the allocation);
  * a CLI flag the entrypoint's argparse does not accept;
  * a run-artifact path pinned to one prior run id with no way to repoint it, which makes
    the script unusable on a rerun rather than reconfigurable;
  * the ICL policy silently drifting away from "eval few-shot, training zero-shot".

Entrypoint flags are read by AST rather than by importing the module: importing
``prism.training.train_v3`` or ``prism.eval.scalability_evaluation`` drags in torch and
spine and would cost this file its speed (and its no-ML guarantee).
"""

import ast
import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
# The e14 suite plus e9_gnn_navigation, which is upstream of it: it produces the
# path_navigator_gt.pt that e14_stage1to3_binary consumes as NAV_GT; plus the e15 WIRE
# suite, which shares the same entrypoints and the same failure modes.
SCRIPTS = sorted((REPO / "scripts").glob("e14_*.sbatch")) + \
    sorted((REPO / "scripts").glob("e15_*.sbatch")) + \
    sorted((REPO / "scripts").glob("e9_gnn_navigation.sbatch"))
BASE_CONFIG = REPO / "experiments" / "base_config.yaml"

# The two entrypoints take their configuration by DIFFERENT mechanisms, so the checks
# below are not interchangeable: train_v3 is a Hydra app configured by `key=value`
# overrides against base_config.yaml, while scalability_evaluation is a plain argparse
# CLI. Asserting Hydra overrides on an argparse script (or vice versa) tests nothing.
HYDRA_ENTRY = "prism.training.train_v3"
ARGPARSE_ENTRY = {"module": "prism.eval.scalability_evaluation",
                  "path": "src/prism/eval/scalability_evaluation.py"}

# Variables naming a PRIOR RUN's artifact. A rerun (or another cluster) produces a
# different run id, so each must be env-overridable — a bare assignment pins the script
# to one run and makes it silently unusable rather than reconfigurable.
RUN_ARTIFACT_VARS = ("STAGE1_LORA", "NAV_GT")


def _text(p):
    return p.read_text()


def test_scripts_exist():
    assert SCRIPTS, "no e14_*.sbatch found — the suite is the thing under test"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_shell_syntax_is_valid(script):
    """``bash -n`` parses without executing — catches the typo that would otherwise
    fail 20 minutes into an allocation."""
    r = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert r.returncode == 0, f"{script.name}: {r.stderr.strip()}"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_fails_fast(script):
    """Without ``set -euo pipefail`` a failed step is swallowed and the job exits 0,
    which reads as a successful run that produced nothing."""
    assert re.search(r"^set -euo pipefail$", _text(script), re.M), \
        f"{script.name}: missing 'set -euo pipefail'"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_run_artifact_paths_are_overridable(script):
    """Prior-run artifacts must use ``${VAR:-default}``, not a bare assignment."""
    body = _text(script)
    for var in RUN_ARTIFACT_VARS:
        for m in re.finditer(rf"^{var}=(.*)$", body, re.M):
            assert m.group(1).startswith("${" + var + ":-"), (
                f"{script.name}: {var} is pinned to one run id and cannot be repointed; "
                f"use {var}=${{{var}:-<default>}}")


def _config_keys():
    """Dotted key paths present in base_config.yaml (e.g. 'gnn.arch')."""
    cfg = yaml.safe_load(_text(BASE_CONFIG))
    out = set()

    def walk(node, prefix=""):
        if not isinstance(node, dict):
            return
        for k, v in node.items():
            dotted = f"{prefix}{k}"
            out.add(dotted)
            walk(v, f"{dotted}.")

    walk(cfg)
    return out


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_hydra_overrides_name_real_config_keys(script):
    """A ``key=value`` override for a key base_config does not define is rejected by
    Hydra at startup — after the job has been allocated."""
    body = _text(script)
    if HYDRA_ENTRY not in body:
        pytest.skip("argparse entrypoint; it takes no Hydra overrides")
    keys = _config_keys()
    # Only lines that are clearly Hydra overrides: dotted key, '=', continued arg list.
    overrides = re.findall(r"^\s+([a-z_]+(?:\.[a-z_]+)+)=", body, re.M)
    assert overrides, f"{script.name}: no Hydra overrides found — parser drifted?"
    unknown = sorted({k for k in overrides if k not in keys})
    assert not unknown, f"{script.name}: overrides absent from base_config.yaml: {unknown}"


def _argparse_flags(module_path):
    """Long flags an entrypoint accepts, read by AST (no import, so no torch)."""
    tree = ast.parse(_text(REPO / module_path))
    flags = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and str(arg.value).startswith("--"):
                    flags.add(arg.value)
    return flags


def _invocation_flags(body, module):
    """Long flags on the ``python -m <module>`` command only.

    Scoped to the invocation and its backslash-continued lines, because the file also
    contains ``#SBATCH --gpus/--mem/--time`` directives and unrelated shell calls
    (``date --iso-8601``); scraping the whole file would flag those as unknown args.
    """
    lines = body.splitlines()
    start = next((i for i, ln in enumerate(lines)
                  if f"-m {module}" in ln and not ln.lstrip().startswith("#")), None)
    if start is None:
        return None
    block = [lines[start]]
    while block[-1].rstrip().endswith("\\") and start + len(block) < len(lines):
        block.append(lines[start + len(block)])
    return set(re.findall(r"(?<![-\w])(--[a-z][a-z0-9-]+)", "\n".join(block)))


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_cli_flags_are_accepted_by_the_entrypoint(script):
    """Every ``--flag`` handed to the entrypoint must exist in its argparse."""
    body = _text(script)
    if ARGPARSE_ENTRY["module"] not in body:
        pytest.skip("Hydra entrypoint; it has no argparse to check flags against")
    accepted = _argparse_flags(ARGPARSE_ENTRY["path"])
    assert accepted, f"{ARGPARSE_ENTRY['path']}: no argparse flags found — AST reader drifted?"
    used = _invocation_flags(body, ARGPARSE_ENTRY["module"])
    assert used, f"{script.name}: found no flags on the entrypoint invocation"
    unknown = sorted(used - accepted)
    assert not unknown, (
        f"{script.name}: flags not accepted by {ARGPARSE_ENTRY['path']}: {unknown}")


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_icl_policy_is_explicit(script):
    """The ICL policy is written out, and it is the one that suite runs under.

    The VALUE is per-suite — e14 is few-shot on both sides, e15 mirrors the zero-shot
    e9_ms_stage1 — so what is asserted globally is that the training-side value is
    stated and reaches the entrypoint. Both sides are written out even where they match
    a config default, because leaving one implicit means the policy changes silently if
    that default ever moves — and the two must agree (base_config.yaml:207).
    """
    body = _text(script)
    is_eval = ARGPARSE_ENTRY["module"] in body
    is_train = HYDRA_ENTRY in body
    if not (is_eval or is_train):
        pytest.skip("neither entrypoint; the ICL switches have no consumer here")

    # e14 is the few-shot suite and stays pinned to 2; other suites must still declare
    # their value rather than inherit it.
    expected = "2" if script.name.startswith("e14_") else r"\d+"
    assert re.search(rf"^DATA_ICL_EXAMPLES=\$\{{DATA_ICL_EXAMPLES:-{expected}\}}$", body, re.M), \
        f"{script.name}: training-side ICL must be set explicitly (expected {expected})"

    if is_train:
        # The training job is the ONLY place data.icl_examples has a consumer, so here it
        # must actually reach Hydra rather than just being declared.
        assert 'data.icl_examples="$DATA_ICL_EXAMPLES"' in body, \
            f"{script.name}: DATA_ICL_EXAMPLES is set but never passed to train_v3"
    if is_eval:
        assert re.search(r"^USE_ICL=\$\{USE_ICL:-true\}$", body, re.M), \
            f"{script.name}: eval-side ICL must be ON and overridable"
        assert "--use-icl" in body, f"{script.name}: USE_ICL is set but never passed"
        # scalability_evaluation has no training stage, so the training value has no
        # consumer here. It is recorded and ECHOED rather than passed: carrying an unused
        # knob silently is how a train/eval policy mismatch goes unnoticed.
        assert "training policy (recorded, not consumed here)" in body, \
            f"{script.name}: unconsumed DATA_ICL_EXAMPLES must be echoed, not hidden"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_referenced_repo_files_exist(script):
    """Repo-relative configs/scripts the job reads must be present.

    Deliberately excludes ``outputs/`` (run artifacts, produced on the cluster) and
    ``data/`` (large corpora, not all of which live on every machine).
    """
    refs = set(re.findall(r"(?<![\w/])(experiments/[\w./-]+|scripts/[\w./-]+)", _text(script)))
    missing = sorted(r for r in refs if not (REPO / r).exists())
    assert not missing, f"{script.name}: referenced repo files missing: {missing}"
