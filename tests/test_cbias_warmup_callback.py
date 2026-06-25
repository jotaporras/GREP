"""Contract tests for prism.eval.callbacks.CBiasWarmupCallback (CS-only).

Invariant under test: the callback linearly ramps the c_bias covariance gain
``λ_C·_lam_c_warmup`` from 0→1 over the first ``warmup_steps`` optimizer steps by
mutating the model's ``_lam_c_warmup`` buffer IN PLACE, navigating PEFT wrappers via
``LoraWarmupCallback._unwrap_peft``; and is a strict no-op when ``warmup_steps<=0``,
when ``model is None``, or when the buffer is absent.

These are deterministic orchestration/plumbing assertions (ramp arithmetic + where the
value is written). No model forward/architecture is exercised — the model is a stub
mirror of PeftModel→LoraModel→InjectedCompositeGraphLLM with a real tensor buffer.
The oracle (value(step)=min(1, step/warmup)) is derived from the docstring/spec, not
copied from the implementation.
"""

import sys
import types

sys.path.insert(0, "src")

import torch

from prism.eval.callbacks import CBiasWarmupCallback


# ---------------------------------------------------------------------------
# Fixtures — faithful, minimal stand-ins for the structures _unwrap_peft walks.
# ---------------------------------------------------------------------------
class _State:
    """Stub TrainerState: only ``global_step`` is read by the callback."""

    def __init__(self, step: int):
        self.global_step = step


def _bare_model(value: float = 1.0):
    """Non-PEFT composite model: owns ``_lam_c_warmup`` (+ ``injection``) directly.

    No ``base_model``/``model`` attrs, so ``_unwrap_peft`` returns it unchanged.
    """
    return types.SimpleNamespace(injection=object(), _lam_c_warmup=torch.tensor(value))


def _peft_model(value: float = 1.0):
    """PeftModel → LoraModel → InjectedCompositeGraphLLM(injection, _lam_c_warmup).

    Returns (wrapped_model, inner) so tests can read the deeply-nested buffer.
    """
    inner = types.SimpleNamespace(injection=object(), _lam_c_warmup=torch.tensor(value))
    lora = types.SimpleNamespace(model=inner)
    peft = types.SimpleNamespace(base_model=lora)
    return peft, inner


def _val(model) -> float:
    return float(model._lam_c_warmup.item())


# ---------------------------------------------------------------------------
# __init__ contract
# ---------------------------------------------------------------------------
def test_init_coerces_warmup_steps_to_int():
    """``warmup_steps`` is stored coerced to int (guards division / comparisons)."""
    cb = CBiasWarmupCallback(5.0)
    assert cb.warmup_steps == 5 and isinstance(cb.warmup_steps, int)


# ---------------------------------------------------------------------------
# on_train_begin: pin the ramp to 0 at start
# ---------------------------------------------------------------------------
def test_on_train_begin_sets_buffer_zero():
    """With warmup>0, on_train_begin forces the buffer to 0.0 (ramp starts cold)."""
    m = _bare_model(value=1.0)
    CBiasWarmupCallback(4).on_train_begin(None, _State(0), None, model=m)
    assert _val(m) == 0.0


# ---------------------------------------------------------------------------
# on_step_begin: linear ramp arithmetic, clamped at 1.0
# ---------------------------------------------------------------------------
def test_on_step_begin_linear_ramp_exact():
    """value(step) == min(1, step/warmup): exact for warmup=4 over steps 0..4."""
    cb = CBiasWarmupCallback(4)
    expected = {0: 0.0, 1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0}
    for step, want in expected.items():
        m = _bare_model(value=-1.0)  # sentinel ≠ any expected value
        cb.on_step_begin(None, _State(step), None, model=m)
        assert _val(m) == want, f"step {step}: got {_val(m)}, want {want}"


def test_on_step_begin_true_division_not_floor():
    """step/warmup is true division: step=1,warmup=2 → 0.5 (catches a // regression)."""
    m = _bare_model(value=-1.0)
    CBiasWarmupCallback(2).on_step_begin(None, _State(1), None, model=m)
    assert _val(m) == 0.5


def test_on_step_begin_clamps_at_one():
    """Past warmup_steps the ramp saturates at 1.0, never overshoots."""
    m = _bare_model(value=-1.0)
    CBiasWarmupCallback(4).on_step_begin(None, _State(10), None, model=m)
    assert _val(m) == 1.0


# ---------------------------------------------------------------------------
# No-op guards
# ---------------------------------------------------------------------------
def test_warmup_zero_is_noop_both_hooks():
    """warmup_steps==0 ⇒ neither hook touches the buffer (sentinel survives)."""
    cb = CBiasWarmupCallback(0)
    m = _bare_model(value=0.75)  # fp32-exact sentinel
    cb.on_train_begin(None, _State(0), None, model=m)
    cb.on_step_begin(None, _State(3), None, model=m)
    assert _val(m) == 0.75


def test_negative_warmup_is_noop():
    """Negative warmup_steps is treated as disabled (no division-by-sign surprise)."""
    m = _bare_model(value=0.5)  # fp32-exact sentinel
    cb = CBiasWarmupCallback(-3)
    cb.on_train_begin(None, _State(0), None, model=m)
    cb.on_step_begin(None, _State(1), None, model=m)
    assert _val(m) == 0.5


def test_none_model_is_noop():
    """model=None must not raise in either hook (callback list runs before model attach)."""
    cb = CBiasWarmupCallback(4)
    cb.on_train_begin(None, _State(0), None, model=None)
    cb.on_step_begin(None, _State(2), None, model=None)  # no exception == pass


def test_missing_buffer_is_graceful():
    """A model without ``_lam_c_warmup`` (non-c_bias) ⇒ _set no-ops, no AttributeError."""
    m = types.SimpleNamespace(injection=object())  # no _lam_c_warmup
    CBiasWarmupCallback(4).on_step_begin(None, _State(2), None, model=m)
    assert not hasattr(m, "_lam_c_warmup")


# ---------------------------------------------------------------------------
# PEFT navigation + in-place mutation (the load-bearing plumbing)
# ---------------------------------------------------------------------------
def test_unwrap_reaches_buffer_through_peft():
    """_unwrap_peft descends PeftModel→LoraModel→model and writes the nested buffer."""
    peft, inner = _peft_model(value=1.0)
    cb = CBiasWarmupCallback(4)
    cb.on_train_begin(None, _State(0), None, model=peft)
    assert float(inner._lam_c_warmup.item()) == 0.0
    cb.on_step_begin(None, _State(2), None, model=peft)
    assert float(inner._lam_c_warmup.item()) == 0.5


def test_mutates_buffer_in_place_preserving_identity():
    """The consumer (model.lam_c * model._lam_c_warmup) holds a live ref to the buffer,
    so updates must use ``fill_`` (same tensor object), never rebind the attribute."""
    m = _bare_model(value=1.0)
    captured = m._lam_c_warmup  # the reference patched-attention would have closed over
    cb = CBiasWarmupCallback(4)
    cb.on_train_begin(None, _State(0), None, model=m)
    cb.on_step_begin(None, _State(2), None, model=m)
    assert m._lam_c_warmup is captured, "buffer was rebound, not filled in place"
    assert float(captured.item()) == 0.5, "live reference did not observe the ramp"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: PASS")
    print("done")
