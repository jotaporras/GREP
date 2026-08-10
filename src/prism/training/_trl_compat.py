"""trl 0.27 ↔ transformers ≥5.12 availability-check compatibility.

transformers 5.12 changed ``_is_package_available`` to ALWAYS return a
``(bool, version)`` tuple. trl 0.27's ``is_*_available()`` helpers pass that
straight through, so a MISSING optional package yields the truthy
``(False, None)`` — and ``trl.trainer.*`` then imports every absent optional
(weave, vllm_ascend, ...) unconditionally and crashes at import.

Importing this module normalizes every ``is_*_available`` helper on
``trl.import_utils`` to a real boolean. Import it BEFORE any ``trl.trainer``
module. Idempotent; helpers that already return bools pass through unchanged.
"""
import trl.import_utils as _iu


def _normalized(fn):
    def wrapped(*args, **kwargs):
        out = fn(*args, **kwargs)
        return out[0] if isinstance(out, tuple) else out
    wrapped._trl_compat_normalized = True
    return wrapped


for _name in dir(_iu):
    if _name.startswith("is_") and _name.endswith("_available"):
        _fn = getattr(_iu, _name)
        if not getattr(_fn, "_trl_compat_normalized", False):
            setattr(_iu, _name, _normalized(_fn))
