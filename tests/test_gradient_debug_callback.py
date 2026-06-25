"""Contract tests for prism.eval.callbacks.GradientDebugCallback (CS-only mode).

Verifies the deterministic *orchestration* surface of the gradient-debug callback
and its Hydra ``gradient_debug`` tag — never a model forward/backward or any
learned quantity. Everything is driven by stub ``nn.Module`` parameter-bags whose
``.grad`` tensors are set by hand, so no architecture is exercised.

Contracts under test (derived from the docstring / call sites, not the body):
  - _unwrap_peft        : PeftModel→LoraModel→inner attribute navigation
  - _is_augmented       : True iff (gt_model AND injection)
  - _supported          : True iff (pe_model OR gt_model)
  - _grad_norm          : global L2 norm over the params' grads, skipping grad=None
  - _capture_grad_norms : per-component dict keyed correctly per architecture branch
  - _install_hooks      : idempotent; legacy wrap forwards args & counts injections
  - on_log              : assembles the right W&B key set per branch; lr from log_history
  - Hydra tag           : `gradient_debug` present+bool in e7/e8/e9.yaml and gating
                          the callback in train_v3.py
"""
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, "src")

# callbacks.py does `from prism.eval import evaluate`, which drags in a heavy
# (and here unimportable) datasets/spine chain that GradientDebugCallback never
# touches. Register a stub so the import binds. Mirrors scripts/smoke_test_callbacks.py.
_ev = types.ModuleType("prism.eval.evaluate")
_ev.__getattr__ = lambda _a: MagicMock()
sys.modules.setdefault("prism.eval.evaluate", _ev)

import torch
from torch import nn

import prism.eval.callbacks as cbmod
from prism.eval.callbacks import GradientDebugCallback as GDC


# --------------------------------------------------------------------------- #
# Fixtures — stub parameter bags (no model architecture)
# --------------------------------------------------------------------------- #
class Bag(nn.Module):
    """nn.Module that exposes whatever sub-modules/params/attrs it is given.

    Submodules register so ``.parameters()`` aggregates them, exactly as the
    real GraphAugmentedLLM / CompositeGraphLLM containers do."""

    def __init__(self, **kw):
        super().__init__()
        for k, v in kw.items():
            setattr(self, k, v)


class _LLM(nn.Module):
    """Minimal stand-in for the wrapped LLM: has get_input_embeddings()."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(16, 4)

    def get_input_embeddings(self):
        return self.emb


class FakeWandb:
    """Truthy .run + capturing .log() (from smoke_test_callbacks.py)."""

    def __init__(self):
        self.run = object()
        self.logged = []

    def log(self, metrics, step=None):
        self.logged.append((step, dict(metrics)))


def _global_norm(params):
    """Independent oracle for _grad_norm: concatenate all grads, take one L2.

    Computed via cat + linalg.norm — a different route than the impl's
    sum-of-squared-per-tensor-norms loop, so it is a genuine cross-check."""
    vecs = [p.grad.reshape(-1) for p in params if p.grad is not None]
    return torch.linalg.norm(torch.cat(vecs)).item() if vecs else 0.0


def _set_grads(*modules, seed=0):
    torch.manual_seed(seed)
    for m in modules:
        for p in m.parameters():
            p.grad = torch.randn_like(p)


# --------------------------------------------------------------------------- #
# _grad_norm — arithmetic contract
# --------------------------------------------------------------------------- #
def test_grad_norm_matches_global_l2():
    """_grad_norm == ||concat(grads)||_2 for params that all have grads."""
    torch.manual_seed(1)
    ps = [nn.Parameter(torch.randn(3, 2)), nn.Parameter(torch.randn(4))]
    for p in ps:
        p.grad = torch.randn_like(p)
    assert abs(GDC._grad_norm(ps) - _global_norm(ps)) < 1e-5


def test_grad_norm_skips_none_grads():
    """Params with grad is None contribute nothing; all-None ⇒ 0.0."""
    a = nn.Parameter(torch.randn(5)); a.grad = torch.randn(5)
    b = nn.Parameter(torch.randn(5))  # grad stays None
    assert abs(GDC._grad_norm([a, b]) - torch.linalg.norm(a.grad).item()) < 1e-6
    assert GDC._grad_norm([b]) == 0.0
    assert GDC._grad_norm([]) == 0.0


# --------------------------------------------------------------------------- #
# Predicates
# --------------------------------------------------------------------------- #
def test_supported_and_augmented_predicates():
    """_supported: pe_model OR gt_model. _is_augmented: gt_model AND injection."""
    legacy = Bag(pe_model=nn.Linear(2, 2))
    aug = Bag(gt_model=nn.Linear(2, 2),
              injection=Bag(gate=nn.Parameter(torch.zeros(2))))
    gt_only = Bag(gt_model=nn.Linear(2, 2))            # gt but no injection
    plain = SimpleNamespace(foo=1)

    assert GDC._supported(legacy) and not GDC._is_augmented(legacy)
    assert GDC._supported(aug) and GDC._is_augmented(aug)
    assert GDC._supported(gt_only) and not GDC._is_augmented(gt_only)
    assert not GDC._supported(plain) and not GDC._is_augmented(plain)


def test_unwrap_peft_navigates_nesting():
    """PeftModel(.base_model.model) and LoraModel(.base_model) both reach inner."""
    target = Bag(pe_proj=nn.Linear(2, 2))
    peft = SimpleNamespace(base_model=SimpleNamespace(model=target))
    assert GDC._unwrap_peft(peft) is target

    # gt_model also triggers the `.model` descent.
    gt_target = SimpleNamespace(gt_model=object())
    peft2 = SimpleNamespace(base_model=SimpleNamespace(model=gt_target))
    assert GDC._unwrap_peft(peft2) is gt_target

    # LoraModel: base_model IS the inner (no further `.model`).
    lora = SimpleNamespace(base_model=target)
    assert GDC._unwrap_peft(lora) is target

    # Bare inner returned unchanged.
    bare = SimpleNamespace(x=1)
    assert GDC._unwrap_peft(bare) is bare


# --------------------------------------------------------------------------- #
# _capture_grad_norms — per-component bookkeeping
# --------------------------------------------------------------------------- #
def test_capture_grad_norms_legacy_rpearl():
    """R-PEARL-only legacy: keys {lora,gnn,pe_proj,pe_gain}; no GT sub-norms."""
    inner = Bag(llm=nn.Linear(3, 3), pe_model=nn.Linear(3, 3),
                pe_proj=nn.Linear(3, 3), pe_gain=nn.Parameter(torch.tensor(0.5)))
    _set_grads(inner.llm, inner.pe_model, inner.pe_proj, seed=2)
    inner.pe_gain.grad = torch.tensor(0.3)

    cb = GDC()
    cb._capture_grad_norms(inner)
    g = cb._captured_grad_norms
    assert set(g) == {"lora", "gnn", "pe_proj", "pe_gain"}
    assert abs(g["lora"] - _global_norm(list(inner.llm.parameters()))) < 1e-5
    assert abs(g["gnn"] - _global_norm(list(inner.pe_model.parameters()))) < 1e-5
    assert abs(g["pe_proj"] - _global_norm(list(inner.pe_proj.parameters()))) < 1e-5
    assert abs(g["pe_gain"] - _global_norm([inner.pe_gain])) < 1e-6


def test_capture_grad_norms_legacy_gt():
    """rpearl_gt_llm legacy: GT blocks + inner R-PEARL split out under pe_model."""
    pe = Bag(blocks=nn.Linear(2, 2), pe_model=nn.Linear(2, 2), trunk=nn.Linear(2, 2))
    inner = Bag(llm=nn.Linear(2, 2), pe_model=pe, pe_proj=nn.Linear(2, 2),
                pe_gain=nn.Parameter(torch.tensor(0.1)))
    _set_grads(inner.llm, pe, inner.pe_proj, seed=3)
    inner.pe_gain.grad = torch.tensor(0.2)

    cb = GDC()
    cb._capture_grad_norms(inner)
    g = cb._captured_grad_norms
    assert {"gt_blocks", "rpearl"} <= set(g)
    assert abs(g["gt_blocks"] - _global_norm(list(pe.blocks.parameters()))) < 1e-5
    assert abs(g["rpearl"] - _global_norm(list(pe.pe_model.parameters()))) < 1e-5
    # gnn is the whole pe_model (blocks + inner pe_model + trunk).
    assert abs(g["gnn"] - _global_norm(list(pe.parameters()))) < 1e-5


def test_capture_grad_norms_augmented():
    """composite_graph_gt: keys {lora,gt,rpearl,gt_blocks,gate}, none legacy."""
    gt = Bag(pe_model=nn.Linear(2, 2), blocks=nn.Linear(2, 2), trunk=nn.Linear(2, 2))
    inj = Bag(gate=nn.Parameter(torch.tensor([0.1, 0.2, 0.3])))
    inner = Bag(llm=nn.Linear(2, 2), gt_model=gt, injection=inj)
    _set_grads(inner.llm, gt, seed=4)
    inj.gate.grad = torch.randn(3)

    cb = GDC()
    cb._capture_grad_norms(inner)
    g = cb._captured_grad_norms
    assert set(g) == {"lora", "gt", "rpearl", "gt_blocks", "gate"}
    assert abs(g["gt"] - _global_norm(list(gt.parameters()))) < 1e-5
    assert abs(g["rpearl"] - _global_norm(list(gt.pe_model.parameters()))) < 1e-5
    assert abs(g["gt_blocks"] - _global_norm(list(gt.blocks.parameters()))) < 1e-5
    assert abs(g["gate"] - _global_norm([inj.gate])) < 1e-6


def test_capture_grad_norms_gt_llm_no_inner_rpearl():
    """gt_llm (SemanticGraphTransformer): pe_model has `blocks` but NO inner
    `pe_model` ⇒ gt_blocks captured, `rpearl` key absent (the docstring's crash guard)."""
    pe = Bag(blocks=nn.Linear(2, 2), trunk=nn.Linear(2, 2))  # no inner pe_model
    inner = Bag(llm=nn.Linear(2, 2), pe_model=pe, pe_proj=nn.Linear(2, 2),
                pe_gain=nn.Parameter(torch.tensor(0.1)))
    _set_grads(inner.llm, pe, inner.pe_proj, seed=5)
    inner.pe_gain.grad = torch.tensor(0.2)

    cb = GDC()
    cb._capture_grad_norms(inner)
    g = cb._captured_grad_norms
    assert "gt_blocks" in g and "rpearl" not in g
    assert abs(g["gt_blocks"] - _global_norm(list(pe.blocks.parameters()))) < 1e-5


def test_capture_grad_norms_noop_when_unsupported():
    """Unsupported inner ⇒ nothing captured (early return)."""
    cb = GDC()
    cb._capture_grad_norms(SimpleNamespace(foo=1))
    assert cb._captured_grad_norms == {}


# --------------------------------------------------------------------------- #
# Hook installation / injection counting
# --------------------------------------------------------------------------- #
def _legacy_hookable(record=None):
    inner = Bag(llm=_LLM(), pe_model=nn.Linear(2, 2), pe_proj=nn.Linear(2, 2),
                pe_gain=nn.Parameter(torch.tensor(0.0)))

    def _aug(input_ids, graphs, injection_maps):
        if record is not None:
            record.append((input_ids, graphs, injection_maps))
        return "ORIG_RESULT"

    inner._augment_embeddings = _aug
    return inner


def test_install_hooks_idempotent():
    """on_train_begin installs once; a second call does not re-wrap."""
    inner = _legacy_hookable()
    cb = GDC()
    cb.on_train_begin(None, None, None, model=inner)
    assert cb._hooked
    wrapped = inner._augment_embeddings
    assert wrapped.__name__ == "_wrapped_augment"   # wrapper installed
    cb.on_train_begin(None, None, None, model=inner)
    assert inner._augment_embeddings is wrapped       # not double-wrapped


def test_injection_count_plumbing():
    """_wrapped_augment counts total spans and forwards args to the original."""
    rec = []
    inner = _legacy_hookable(record=rec)
    cb = GDC()
    cb.on_train_begin(None, None, None, model=inner)

    ids = torch.zeros(1, 6, dtype=torch.long)
    imaps = [{0: [(2, 3)], 1: [(4, 5), (0, 1)]}]      # 1 + 2 = 3 spans
    out = inner._augment_embeddings(ids, ["g"], imaps)

    assert out == "ORIG_RESULT"                        # original called through
    assert cb._num_injections == 3                     # CS counting contract
    assert rec and rec[0][2] is imaps                  # injection_maps forwarded verbatim


# --------------------------------------------------------------------------- #
# on_log — W&B payload assembly (routing only; values are stub inputs)
# --------------------------------------------------------------------------- #
def _state(lr=1e-4, step=5):
    return SimpleNamespace(
        global_step=step,
        log_history=([{"learning_rate": lr}] if lr is not None else []),
    )


def test_on_log_legacy_keys_and_lr():
    inner = Bag(llm=nn.Linear(2, 2), pe_model=nn.Linear(2, 2),
                pe_proj=nn.Linear(2, 2), pe_gain=nn.Parameter(torch.tensor(0.42)))
    cb = GDC()
    cb._captured_grad_norms = {"gnn": 1.0, "pe_proj": 2.0, "lora": 3.0, "pe_gain": 4.0}
    cb._pe_norm, cb._emb_norm, cb._num_injections, cb._pe_has_nan = 0.5, 0.7, 9, True

    fake = FakeWandb(); cbmod.wandb = fake
    cb.on_log(None, _state(lr=1e-4), None, model=inner)
    m = fake.logged[-1][1]

    expected = {"debug/grad_norm_gnn", "debug/grad_norm_pe_proj", "debug/grad_norm_lora",
                "debug/pe_output_norm", "debug/pe_has_nan", "debug/embedding_norm",
                "debug/num_injections", "debug/lr", "debug/pe_gain",
                "debug/grad_norm_pe_gain"}
    assert expected <= set(m)
    assert "debug/grad_norm_gt_blocks" not in m        # pe_model has no `blocks`
    assert m["debug/grad_norm_gnn"] == 1.0
    assert m["debug/num_injections"] == 9
    assert m["debug/pe_has_nan"] == 1                   # int(True)
    assert m["debug/lr"] == 1e-4
    assert abs(m["debug/pe_gain"] - 0.42) < 1e-6


def test_on_log_legacy_gt_blocks_present():
    """rpearl_gt_llm on_log: pe_model with `blocks`+inner `pe_model` adds the
    GT-split keys (gt_blocks, rpearl) on top of the base legacy payload."""
    pe = Bag(blocks=nn.Linear(2, 2), pe_model=nn.Linear(2, 2))
    inner = Bag(llm=nn.Linear(2, 2), pe_model=pe, pe_proj=nn.Linear(2, 2),
                pe_gain=nn.Parameter(torch.tensor(0.0)))
    cb = GDC()
    cb._captured_grad_norms = {"gnn": 1.0, "pe_proj": 2.0, "lora": 3.0,
                               "pe_gain": 4.0, "gt_blocks": 5.0, "rpearl": 6.0}
    fake = FakeWandb(); cbmod.wandb = fake
    cb.on_log(None, _state(lr=1e-4), None, model=inner)
    m = fake.logged[-1][1]
    assert m["debug/grad_norm_gt_blocks"] == 5.0
    assert m["debug/grad_norm_rpearl"] == 6.0


def test_on_log_lr_nan_when_no_history():
    inner = Bag(llm=nn.Linear(2, 2), pe_model=nn.Linear(2, 2),
                pe_proj=nn.Linear(2, 2), pe_gain=nn.Parameter(torch.tensor(0.0)))
    cb = GDC()
    fake = FakeWandb(); cbmod.wandb = fake
    cb.on_log(None, _state(lr=None), None, model=inner)
    assert math.isnan(fake.logged[-1][1]["debug/lr"])


def test_on_log_augmented_keys_and_gains():
    gt = Bag(pe_model=Bag(output_gain=nn.Parameter(torch.tensor(0.2))),
             output_gain=nn.Parameter(torch.tensor(0.3)), blocks=nn.Linear(2, 2))
    inj = Bag(gate=nn.Parameter(torch.tensor([0.1, 0.2])))
    inner = Bag(llm=nn.Linear(2, 2), gt_model=gt, injection=inj)
    cb = GDC()
    cb._captured_grad_norms = {"lora": 1.0, "gt": 2.0, "rpearl": 3.0,
                               "gt_blocks": 4.0, "gate": 5.0}
    cb._pe_norm, cb._emb_norm, cb._num_injections, cb._pe_has_nan = 0.9, 0.4, 7, False

    fake = FakeWandb(); cbmod.wandb = fake
    cb.on_log(None, _state(lr=2e-4), None, model=inner)
    m = fake.logged[-1][1]

    expected = {"debug/grad_norm_lora", "debug/grad_norm_gt", "debug/grad_norm_rpearl",
                "debug/grad_norm_gt_blocks", "debug/grad_norm_gate", "debug/gt_output_norm",
                "debug/gt_has_nan", "debug/embedding_norm", "debug/num_injections",
                "debug/gate_value", "debug/gt_output_gain", "debug/rpearl_output_gain",
                "debug/lr"}
    assert expected <= set(m)
    assert m["debug/grad_norm_gt"] == 2.0
    assert m["debug/num_injections"] == 7
    assert m["debug/gt_has_nan"] == 0
    assert abs(m["debug/gt_output_gain"] - math.tanh(0.3)) < 1e-5
    assert abs(m["debug/rpearl_output_gain"] - math.tanh(0.2)) < 1e-5
    assert abs(m["debug/gate_value"] - 0.15) < 1e-5     # mean([0.1, 0.2])
    assert not any(k.startswith("grep/c_bias") for k in m)  # no c_bias attr


# --------------------------------------------------------------------------- #
# Hydra tag — config plumbing
# --------------------------------------------------------------------------- #
def test_gradient_debug_tag_present_and_bool():
    """`gradient_debug` exists and is a bool in every overview config."""
    import yaml
    base = Path("experiments/e9_hydra_training/overview")
    for f in ("e7.yaml", "e8.yaml", "e9.yaml"):
        cfg = yaml.safe_load((base / f).read_text())
        assert "gradient_debug" in cfg, f"{f}: missing gradient_debug tag"
        assert isinstance(cfg["gradient_debug"], bool), \
            f"{f}: gradient_debug is {type(cfg['gradient_debug'])}, not bool"


def test_gradient_debug_gates_callback_registration():
    """train_v3 forks callback registration on the tag (the switch is wired)."""
    src = Path("src/prism/training/train_v3.py").read_text()
    assert "if config.gradient_debug:" in src
    assert "callbacks.GradientDebugCallback()" in src


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"{name}: PASS")
    print("done")
