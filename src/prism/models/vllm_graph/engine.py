"""Engine construction for graph-conditioned vLLM rollouts and eval.

``build_graph_llm`` spins up a ``vllm.LLM`` serving the wrapper model
(:mod:`prism.models.vllm_graph.model`) with the engine constraints the Ψ
transport requires:

- ``enforce_eager=True`` — the attention patch is Python state; CUDA graphs /
  torch.compile would capture stale Ψ.
- ``enable_prefix_caching=False`` — Ψ varies per request but is invisible to
  the prefix-cache key, so cached prefixes would silently reuse another
  request's injected state.
- ``enable_mm_embeds=True`` — the transport is a precomputed tensor, not media.
- ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` — runtime model registration must be
  visible to the model runner (single-process engine).

Engine sizing knobs follow the ``PRISM_VLLM_*`` env convention from
``prism.data.vllm_llm._engine_config``.
"""
from __future__ import annotations

import os

import torch

# Archs whose graph channel is the additive Ψ path (GraphAugmentedLLM family).
# Mask archs (graph_mask_llm / learnable_graph_mask) inject decode-time mask
# rows (MaskDecodeInjector) that have no vLLM analog — HF backend only.
ADDITIVE_ARCHS = {"gt_llm", "rpearl_gt_llm", "rpearl_llm"}


def _engine_env(name: str, default, cast):
    raw = os.environ.get(name)
    return default if raw is None else cast(raw)


def build_graph_llm(
    model_path: str,
    *,
    identity_rope: bool = False,
    pe_inject_value: bool = True,
    dtype: str = "auto",
    max_model_len: int | None = None,
    gpu_memory_utilization: float | None = None,
    seed: int = 0,
    **extra_engine_kwargs,
):
    """Construct the graph-conditioned engine over HF-format weights at ``model_path``.

    ``model_path`` must hold full model weights (base model, or an SFT
    checkpoint whose LoRA has been merged at bf16 — never merge into nf4, see
    ``loaders.graph_augmented_llm_from_pretrained``). Returns ``(llm, wrapper)``
    where ``wrapper`` is the in-process model instance carrying the ``dbg``
    counters.
    """
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    # vLLM 0.26 selects the attention backend via the ``attention_backend``
    # engine arg and no longer consults VLLM_ATTENTION_BACKEND. Honour the env
    # var here so one export pins the backend across every engine this process
    # spins (tests, RL rollouts, eval) — on sm100 the auto-pick is FLASHINFER,
    # whose kernels JIT-compile with nvcc that cluster nodes may not have.
    # fp32 engines (parity tests) must stay on auto-selection: explicit
    # FLASH_ATTN is refused for fp32 ("dtype not supported"), while fp32
    # auto-pick already excludes FLASHINFER — only half/bf16 engines need the
    # pin (their auto-pick prefers FLASHINFER on sm100).
    env_backend = os.environ.get("VLLM_ATTENTION_BACKEND")
    if (env_backend and "attention_backend" not in extra_engine_kwargs
            and str(dtype).replace("torch.", "") not in ("float32", "float")):
        extra_engine_kwargs["attention_backend"] = env_backend

    from vllm import LLM

    from prism.models.vllm_graph.model import GRAPH_ARCH_NAME, register_graph_gemma4

    register_graph_gemma4()
    llm = LLM(
        model=model_path,
        hf_overrides={"architectures": [GRAPH_ARCH_NAME]},
        dtype=dtype,
        enforce_eager=True,
        enable_mm_embeds=True,
        enable_prefix_caching=False,
        disable_log_stats=True,
        seed=seed,
        max_model_len=_engine_env("PRISM_VLLM_MAX_LEN", max_model_len or 16384, int),
        gpu_memory_utilization=_engine_env(
            "PRISM_VLLM_GPU_UTIL", gpu_memory_utilization or 0.90, float),
        **extra_engine_kwargs,
    )
    wrapper = llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
    wrapper._identity_rope = identity_rope
    wrapper._pe_inject_value = pe_inject_value
    return llm, wrapper


def checkpoint_engine_policy(checkpoint_dir: str) -> dict:
    """Engine policy read from the checkpoint's recorded config — fails loud.

    Returns ``{"identity_rope": bool, "pe_inject_value": bool, "base_model": str,
    "architecture": str}`` after asserting the arch is in the additive family.
    Reads through ``loaders.load_gnn_config`` so legacy flat ``gnn_config.json``
    checkpoints resolve identically to the HF eval path.
    """
    from prism.models import loaders

    cfg = loaders.load_gnn_config(checkpoint_dir)
    arch = cfg.get("architecture", "rpearl_llm")
    if arch not in ADDITIVE_ARCHS:
        raise ValueError(
            f"architecture {arch!r} is not vLLM-servable: only the additive Ψ family "
            f"{sorted(ADDITIVE_ARCHS)} is supported (mask archs need the HF backend — "
            "their decode-time MaskDecodeInjector has no vLLM analog)."
        )
    return {
        "identity_rope": bool(cfg.get("disable_graph_token_rope", False)),
        "pe_inject_value": bool(cfg.get("pe_inject_value", True)),
        "base_model": cfg.get("base_model"),
        "architecture": arch,
    }


def generate_with_psi(llm, prompt_ids: list[int], psi_transport: torch.Tensor,
                      sampling_params):
    """One graph-conditioned generation. ``psi_transport`` is [seq, hidden+1]."""
    req = {
        "prompt_token_ids": prompt_ids,
        "multi_modal_data": {"image": {"graph_embeds": psi_transport.unsqueeze(0)}},
    }
    return llm.generate([req], sampling_params)[0]


def build_plain_llm(model_path: str, *, dtype: str = "auto",
                    max_model_len: int | None = None,
                    gpu_memory_utilization: float | None = None, seed: int = 0,
                    **extra_engine_kwargs):
    """Stock vLLM engine for plain-LLM checkpoints (no graph channel, no
    registration, prefix caching allowed)."""
    from vllm import LLM

    return LLM(
        model=model_path,
        dtype=dtype,
        disable_log_stats=True,
        seed=seed,
        max_model_len=_engine_env("PRISM_VLLM_MAX_LEN", max_model_len or 16384, int),
        gpu_memory_utilization=_engine_env(
            "PRISM_VLLM_GPU_UTIL", gpu_memory_utilization or 0.90, float),
        **extra_engine_kwargs,
    )


def materialize_serving_dir(checkpoint_dir: str, is_gnn: bool,
                            out_dir: str | None = None) -> str:
    """HF-format weights vLLM can serve for this checkpoint.

    No adapter → the base model itself (nothing to materialize). With an
    adapter → merge it into the bf16 base ONCE and cache under
    ``<checkpoint>/vllm_bf16/``. Merging at bf16 is exact (unlike merge-into-nf4
    — see ``loaders.graph_augmented_llm_from_pretrained`` — which rounds the
    adapter away); it needs the full model in memory, so this runs on the
    cluster, not a laptop.
    """
    from prism.models import loaders

    if not os.path.exists(os.path.join(checkpoint_dir, "adapter_config.json")):
        if is_gnn:
            return loaders.load_gnn_config(checkpoint_dir)["base_model"]
        return checkpoint_dir

    out = out_dir or os.path.join(checkpoint_dir, "vllm_bf16")
    if os.path.exists(os.path.join(out, "config.json")):
        return out

    if is_gnn:
        # bf16 path merges the adapter (merge_and_unload) before returning.
        model, tokenizer = loaders.graph_augmented_llm_from_pretrained(
            checkpoint_dir, load_in_4bit=False, device=-1)
        llm = model.llm
    else:
        model, tokenizer = loaders.from_pretrained(
            checkpoint_dir, load_in_4bit=False, device=-1)
        # Plain loader keeps the adapter attached; serving needs merged weights.
        llm = model.merge_and_unload() if hasattr(model, "merge_and_unload") else model
    llm.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    return out
