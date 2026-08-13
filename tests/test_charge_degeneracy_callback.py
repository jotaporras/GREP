"""Contract tests for prism.eval.callbacks.ChargeDegeneracyCallback (CS-only mode).

Verifies the deterministic arithmetic and orchestration of the delta monitor — the
sawtooth delta = dist(2rc, Z), the integer-crossing counter, the learn_r=False
stand-down, and the no-write guarantee. Driven by a stub module with a settable
``r``; no MagNet forward and no learned quantity is exercised.

Contracts under test:
  - __init__          : c > 0 required (raise, never guess)
  - _find_magnet      : pe_gcn under R-PEARL / GT-nested R-PEARL; None for TAGConv
  - _read_charge      : one scalar; RuntimeError when the layers untie
  - _delta            : delta correct at 0, 1/2, and generic points across periods
  - on_step_end       : crossings += 1 per integer of s; first step has no diffs;
                        never writes r
  - learn_r=False     : logged once at init, on_step_end is a no-op
"""
import sys
import types
import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, "src")

# callbacks.py does `from prism.eval import evaluate`, which drags in a heavy (and here
# unimportable) datasets/spine chain this callback never touches. Mirrors
# tests/test_gradient_debug_callback.py.
_ev = types.ModuleType("prism.eval.evaluate")
_ev.__getattr__ = lambda _a: MagicMock()
sys.modules.setdefault("prism.eval.evaluate", _ev)

import pytest
import torch
from torch import nn

import prism.eval.callbacks as cbmod
from prism.eval.callbacks import ChargeDegeneracyCallback as CDC


# --------------------------------------------------------------------------- #
# Fixtures — stub MagNet (settable charge, no graph op)
# --------------------------------------------------------------------------- #
class StubConv(nn.Module):
    """MagChebConv stand-in: `r` derived from r_logit exactly as magnet.py does."""

    def __init__(self, r=0.126, learn_r=True):
        super().__init__()
        self.r_const = r
        self.r_logit = (nn.Parameter(torch.tensor(min(max(r, 1e-3), 0.249) / 0.25).logit())
                        if learn_r else None)

    @property
    def r(self):
        return self.r_const if self.r_logit is None else 0.25 * self.r_logit.sigmoid()


class StubMagNet(nn.Module):
    """MagNet stand-in: `num_layers` convs sharing ONE r_logit, plus `set_r`."""

    def __init__(self, r=0.126, learn_r=True, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList([StubConv(r, learn_r) for _ in range(num_layers)])
        for conv in self.convs[1:]:
            conv.r_logit = self.convs[0].r_logit

    @property
    def r(self):
        return self.convs[0].r

    def set_r(self, value):
        """Drive r directly by inverting the 0.25*sigmoid reparameterization."""
        with torch.no_grad():
            self.convs[0].r_logit.copy_(torch.tensor(value / 0.25).logit())


class FakeWandb:
    """Truthy .run + capturing .log() (from test_gradient_debug_callback.py)."""

    def __init__(self):
        self.run = object()
        self.logged = []

    def log(self, metrics, step=None):
        self.logged.append((step, dict(metrics)))


def _state(step=1):
    return SimpleNamespace(global_step=step, log_history=[])


def _delta_oracle(r, c):
    """Independent oracle: distance from 2rc to the nearest integer, via round()."""
    s = 2.0 * r * c
    return abs(s - round(s))


def _cb(c=8, magnet=None):
    """Callback with its module already resolved (skips on_train_begin plumbing)."""
    cb = CDC(cycle_length=c)
    cb._magnet = magnet
    return cb


# --------------------------------------------------------------------------- #
# Construction — c must be supplied
# --------------------------------------------------------------------------- #
def test_cycle_length_required():
    """c is not on the module; absent/zero/negative must raise, never default."""
    for bad in (None, 0, -8):
        with pytest.raises(ValueError):
            CDC(cycle_length=bad)
    assert CDC(cycle_length=8192).c == 8192


# --------------------------------------------------------------------------- #
# Module resolution
# --------------------------------------------------------------------------- #
def test_find_magnet_across_architectures():
    """pe_gcn found under R-PEARL and under the GT's inner R-PEARL; None for TAGConv."""
    magnet = StubMagNet()
    rpearl = SimpleNamespace(pe_model=SimpleNamespace(pe_gcn=magnet))
    assert CDC._find_magnet(rpearl) is magnet

    gt = SimpleNamespace(pe_model=SimpleNamespace(pe_model=SimpleNamespace(pe_gcn=magnet)))
    assert CDC._find_magnet(gt) is magnet

    # Undirected TAGConv backbone: convs but no charge ⇒ nothing to watch.
    tag = SimpleNamespace(pe_model=SimpleNamespace(pe_gcn=SimpleNamespace(convs=[1, 2])))
    assert CDC._find_magnet(tag) is None
    assert CDC._find_magnet(SimpleNamespace(foo=1)) is None


def test_read_charge_fails_loudly_when_layers_untie():
    """Untied charges invalidate delta ⇒ RuntimeError, not a silent convs[0]."""
    magnet = StubMagNet(r=0.126, num_layers=3)
    cb = _cb(magnet=magnet)
    assert abs(cb._read_charge(magnet) - 0.126) < 1e-6

    magnet.convs[1].r_logit = nn.Parameter(torch.tensor(0.3 / 0.25).logit())
    with pytest.raises(RuntimeError, match="diverged"):
        cb._read_charge(magnet)


# --------------------------------------------------------------------------- #
# delta — the sawtooth
# --------------------------------------------------------------------------- #
def test_delta_at_degenerate_and_safe_points():
    """delta = 0 on integer s (collision) and 1/2 on half-integer s (best case)."""
    c = 64
    cb = _cb(c=c)
    for k in (1, 2, 7, 16):
        s, delta = cb._delta(k / (2 * c))            # s = k exactly
        assert abs(s - k) < 1e-9
        assert delta < 1e-9
        s, delta = cb._delta((k + 0.5) / (2 * c))    # s = k + 1/2
        assert abs(delta - 0.5) < 1e-9


def test_delta_sweep_matches_oracle_across_periods():
    """Sweep r over several sawtooth periods; delta tracks dist(2rc, Z) throughout."""
    c = 8
    cb = _cb(c=c)
    period = 1 / (2 * c)
    for i in range(200):
        r = 0.02 + i * period / 17          # incommensurate step ⇒ generic points
        s, delta = cb._delta(r)
        assert abs(s - 2 * r * c) < 1e-9
        assert abs(delta - _delta_oracle(r, c)) < 1e-9
        assert 0.0 <= delta <= 0.5


def test_cond_proxy_and_warning_threshold():
    """cond_proxy = 0.637/max(delta,1e-6); the near-degenerate warning fires ONCE."""
    c = 64
    magnet = StubMagNet(r=(1.0 + 0.5) / (2 * c))     # s = 1.5 ⇒ delta = 1/2, safe
    cb = _cb(c=c, magnet=magnet)
    fake = FakeWandb(); cbmod.wandb = fake

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        cb.on_step_end(None, _state(1), None)
        m = fake.logged[-1][1]
        assert abs(m["charge/delta"] - 0.5) < 1e-5
        assert abs(m["charge/cond_proxy"] - 0.637 / 0.5) < 1e-5
        assert not rec

        # Walk onto a collision: delta ≈ 0 ⇒ warn, but only the first time.
        magnet.set_r(2.0 / (2 * c))
        cb.on_step_end(None, _state(2), None)
        cb.on_step_end(None, _state(3), None)
        assert len(rec) == 1 and "Nearest safe charge" in str(rec[0].message)

    m = fake.logged[-1][1]
    assert m["charge/delta"] < 1e-4
    assert m["charge/cond_proxy"] > 6000                 # 0.637 / max(delta, 1e-6)
    assert m["charge/delta_min_since_start"] < 1e-4


# --------------------------------------------------------------------------- #
# on_step_end — crossings, step diffs, no-write
# --------------------------------------------------------------------------- #
def test_crossings_increment_once_per_integer_of_s():
    """Each integer of s crossed bumps `charge/crossings` by exactly one."""
    c = 64
    magnet = StubMagNet(r=0.25 / (2 * c))            # s = 0.25
    cb = _cb(c=c, magnet=magnet)
    fake = FakeWandb(); cbmod.wandb = fake

    cb.on_step_end(None, _state(0), None)
    assert fake.logged[-1][1]["charge/crossings"] == 0
    assert "charge/periods_per_step" not in fake.logged[-1][1]   # no predecessor yet

    # s: 0.25 → 1.25 → 2.25 → 3.25, one integer per step.
    for step, s_target in enumerate([1.25, 2.25, 3.25], start=1):
        magnet.set_r(s_target / (2 * c))
        cb.on_step_end(None, _state(step), None)
        m = fake.logged[-1][1]
        assert m["charge/crossings"] == step
        assert abs(m["charge/periods_per_step"] - 1.0) < 1e-5
        assert abs(m["charge/dr_per_step"] - 1.0 / (2 * c)) < 1e-6

    # A multi-period jump counts every integer it spans, not just one.
    magnet.set_r(7.25 / (2 * c))
    cb.on_step_end(None, _state(4), None)
    assert fake.logged[-1][1]["charge/crossings"] == 7
    assert abs(fake.logged[-1][1]["charge/periods_per_step"] - 4.0) < 1e-5


def test_callback_never_writes_the_charge():
    """Diagnostic only: r_logit is byte-identical after monitoring."""
    c = 128
    magnet = StubMagNet(r=0.126)
    cb = _cb(c=c, magnet=magnet)
    cbmod.wandb = FakeWandb()
    before = magnet.convs[0].r_logit.detach().clone()

    for step in range(5):
        cb.on_step_end(None, _state(step), None)

    assert torch.equal(before, magnet.convs[0].r_logit.detach())
    assert magnet.convs[0].r_logit.grad is None


def test_noop_when_learn_r_false():
    """learn_r=False: delta is constant ⇒ logged once at init, then nothing."""
    c = 64
    magnet = StubMagNet(r=0.126, learn_r=False)
    model = SimpleNamespace(pe_model=SimpleNamespace(pe_gcn=magnet))
    cb = CDC(cycle_length=c)
    fake = FakeWandb(); cbmod.wandb = fake

    cb.on_train_begin(None, _state(0), None, model=model)
    assert cb._static
    assert len(fake.logged) == 1
    m = fake.logged[0][1]
    assert abs(m["charge/r"] - 0.126) < 1e-7             # float32 r_const roundtrip
    assert abs(m["charge/delta"] - _delta_oracle(0.126, c)) < 1e-5

    for step in range(3):
        cb.on_step_end(None, _state(step), None)
    assert len(fake.logged) == 1                     # on_step_end stood down


def test_noop_when_backbone_is_not_magnet():
    """directed=False (TAGConv): no charge ⇒ no resolution, no logs, no crash."""
    cb = CDC(cycle_length=64)
    fake = FakeWandb(); cbmod.wandb = fake
    cb.on_train_begin(None, _state(0), None,
                      model=SimpleNamespace(pe_model=SimpleNamespace(pe_gcn=nn.Linear(2, 2))))
    assert cb._magnet is None
    cb.on_step_end(None, _state(1), None)
    assert fake.logged == []


def test_realistic_step_scale_flags_unlearnable_delta():
    """The headline claim: at c=8192 one ~3e-5 optimizer step spans ~0.5 periods."""
    c, dr = 8192, 3e-5
    magnet = StubMagNet(r=0.126)
    cb = _cb(c=c, magnet=magnet)
    cbmod.wandb = FakeWandb()

    cb.on_step_end(None, _state(0), None)
    magnet.set_r(0.126 + dr)
    cb.on_step_end(None, _state(1), None)
    periods = cbmod.wandb.logged[-1][1]["charge/periods_per_step"]
    assert abs(periods - 2 * dr * c) < 1e-3
    assert periods >= 0.4                            # delta is not learnable at this lr


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"{name}: PASS")
    print("done")
