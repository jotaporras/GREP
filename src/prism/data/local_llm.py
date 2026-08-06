"""Local HuggingFace-Transformers backend for the data generator.

This module lets the two LLM-driven phases of the data generator run fully
offline on a local Gemma 4 model (default 26B-A4B) instead of calling OpenAI:

    Phase 1 (populate graphs + tasks)  -> :class:`LocalHFQueryClient`
        Drop-in for ``prism.data.utils.GPTQueryClient``. Mirrors its
        ``query_gpt`` / ``query_gpt_5`` / ``batch_query_gpt_5`` surface so
        ``TaskGraphGen`` can swap backends with no other code changes.

    Phase 2 (SPINE planner rollouts)   -> :class:`GemmaSpineClient`
        Implements the SPINE ``client`` contract (``query_llm(msg) ->
        (str, bool)`` and ``format_prompt``) so it can be passed as
        ``SPINE(..., client=GemmaSpineClient())``.

Both clients share a SINGLE model instance via :func:`load_gemma` (module-level
cache), so the ~31B weights are loaded into VRAM exactly once per process even
though populate and rollouts each grab a client.

Model selection
---------------
``PRISM_HF_MODEL`` picks the checkpoint (default ``google/gemma-4-26B-A4B-it``,
a Mixture-of-Experts model: 26B total params but only ~4B active per token, so
it is much faster and lighter than the dense 31B while staying a Gemma 4 model).
Set it to ``google/gemma-4-31B-it`` for the dense 31B. ``PRISM_HF_QUANT`` picks
how it is loaded (footprints below are for the 26B-A4B default; the dense 31B is
~2x larger):

    none  (default) : bf16/auto. 26B-A4B ~52GB; dense 31B ~62GB. Needs a big
                      GPU or CPU offload via device_map="auto".
    4bit / nf4      : bitsandbytes NF4. 26B-A4B ~13GB; dense 31B ~18-20GB. Fits
                      a single 24GB GPU on Ampere/Ada/Hopper (no Blackwell).
    8bit            : bitsandbytes int8. 26B-A4B ~26GB; dense 31B ~33GB.

NOTE on NVFP4: ``nvidia/Gemma-4-31B-IT-NVFP4`` does NOT load through eager
``AutoModelForCausalLM`` — its FP4 weight_scale/input_scale tensors are not
materialised by a plain ``from_pretrained`` (you get UNEXPECTED-key + half-width
shape-MISMATCH errors), and native FP4 compute needs a Blackwell GPU. To use it,
serve it with vLLM/TensorRT-LLM instead. For eager HF Transformers, prefer the
bf16 model above with ``PRISM_HF_QUANT=4bit`` to hit a comparable footprint.

Gemma 4 ships a step-by-step "thinking" mode. It is ENABLED here (both phases)
via ``apply_chat_template(..., enable_thinking=True)`` so the model reasons
before answering; the reasoning is emitted in a ``<think>`` block that
``processor.parse_response`` strips, so the saved data stays clean JSON (Phase 1)
and clean SPINE plans (Phase 2). NOTE: thinking consumes generation tokens, so
the budgets account for the ``<think>`` block — Phase 1 uses 20480 (10240 was
enough for ~30-node graphs but truncated every ~100-node populate JSON
mid-document once thinking took its share), Phase 2 is raised from 2048 to
4096. If a harder model still truncates before emitting ``answer(...)``, raise
``max_new_tokens`` further.

Runtime deps (install on the GPU node):
    pip install -U transformers torch accelerate
    # for PRISM_HF_QUANT=4bit / 8bit:
    pip install -U bitsandbytes
"""

import json
import os
import re
import warnings
from typing import List, Optional

import torch


def _silence_hf_warnings() -> None:
    """Quiet transformers / HF Hub logging, progress bars and Python warnings.

    Keeps the data-gen logs readable instead of drowned in HF chatter
    (deprecation warnings, the weight-loading LOAD REPORT, download progress
    bars, the bitsandbytes banner, the tokenizer-fork warning). ``setdefault``
    is used so an explicit env override from the caller still wins. Runs at
    import — before the lazy ``transformers`` import in :func:`load_gemma` — so
    the env vars are in place when transformers/huggingface_hub first load.
    """
    for key, value in {
        "TRANSFORMERS_VERBOSITY": "error",
        "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "BITSANDBYTES_NOWELCOME": "1",
    }.items():
        os.environ.setdefault(key, value)
    warnings.filterwarnings("ignore")


_silence_hf_warnings()

DEFAULT_GEMMA_MODEL = "google/gemma-4-26B-A4B-it"

# model_id -> (model, processor). Module-level so populate (Phase 1) and SPINE
# rollouts (Phase 2) share one loaded model within a process.
_GEMMA_CACHE: dict = {}


def load_gemma(model_id: Optional[str] = None):
    """Load (and cache) the Gemma 4 model + processor.

    Parameters
    ----------
    model_id : str, optional
        HF repo id. Defaults to ``$PRISM_HF_MODEL`` or
        :data:`DEFAULT_GEMMA_MODEL`.

    Returns
    -------
    (model, processor)
        ``model`` is in eval mode on ``device_map="auto"``; ``processor`` is the
        matching ``AutoProcessor`` (wraps the tokenizer and exposes
        ``apply_chat_template`` / ``parse_response``).
    """
    model_id = model_id or os.environ.get("PRISM_HF_MODEL", DEFAULT_GEMMA_MODEL)
    if "nvfp4" in model_id.lower():
        # Fail fast: NVFP4 checkpoints cannot be loaded by eager HF Transformers
        # (their FP4 weight_scale/input_scale tensors are never materialised, so
        # from_pretrained falls back to a dense model and dies on a half-width
        # shape MISMATCH). This almost always means PRISM_HF_MODEL is a stale env
        # var rather than a deliberate choice.
        raise RuntimeError(
            f"PRISM_HF_MODEL={model_id!r} is an NVFP4 checkpoint, which this "
            "eager-Transformers backend cannot load. It is most likely a stale "
            "environment variable. Fix with ONE of:\n"
            "    unset PRISM_HF_MODEL            # fall back to google/gemma-4-26B-A4B-it\n"
            "    export PRISM_HF_MODEL=google/gemma-4-26B-A4B-it\n"
            "and grep your repo .env for a PRISM_HF_MODEL line "
            "(the scripts `source .env`). To actually use NVFP4 weights, serve "
            "them with vLLM/TensorRT-LLM (needs a Blackwell GPU), not this backend."
        )
    if model_id not in _GEMMA_CACHE:
        # Imported lazily so importing this module (e.g. for the OpenAI path)
        # never forces a transformers import or a model download.
        from transformers import AutoModelForCausalLM, AutoProcessor
        from transformers.utils import logging as hf_logging

        # Silence transformers' own logger + progress bars (env vars alone don't
        # cover the in-process logger / the weight-loading LOAD REPORT table).
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()

        # Single-GPU guarantee. The real enforcement is CUDA_VISIBLE_DEVICES
        # (set by the launch script before this process starts), which makes
        # only one device visible so device_map="auto" cannot span GPUs. This
        # assertion is a backstop: if PRISM_REQUIRE_SINGLE_GPU=1 and more than
        # one GPU is still visible, fail loudly rather than silently sharding.
        n_visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if os.environ.get("PRISM_REQUIRE_SINGLE_GPU") == "1" and n_visible > 1:
            raise RuntimeError(
                f"PRISM_REQUIRE_SINGLE_GPU=1 but {n_visible} GPUs are visible "
                f"(CUDA_VISIBLE_DEVICES="
                f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r}). Restrict it to a "
                "single device id, e.g. `export CUDA_VISIBLE_DEVICES=0`."
            )
        if n_visible:
            where = (
                torch.cuda.get_device_name(0)
                if n_visible == 1
                else f"{n_visible} GPUs (device_map='auto' sharding)"
            )
            print(
                f"[local-llm] CUDA_VISIBLE_DEVICES="
                f"{os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} -> "
                f"{n_visible} visible GPU(s); using {where}"
            )

        # A checkpoint that ships its own quantization (e.g. a compressed-tensors
        # QAT w4a16 build like google/gemma-4-E4B-it-qat-w4a16-ct) must be loaded
        # WITHOUT a bitsandbytes config: passing a BitsAndBytesConfig on top of a
        # baked-in CompressedTensorsConfig raises "The model is quantized with
        # CompressedTensorsConfig but you are passing a BitsAndBytesConfig". We
        # honour the native quant via dtype="auto" instead (same path the Gemma
        # judge in eval.path_validator already uses).
        from transformers import AutoConfig

        try:
            prequantized = (
                getattr(AutoConfig.from_pretrained(model_id), "quantization_config", None)
                is not None
            )
        except Exception:
            prequantized = False

        # device_map="auto" shards / offloads across whatever is available.
        # With one visible device above, that means this single GPU (+ CPU
        # offload only if it does not fit) — never a second GPU.
        load_kwargs = {"device_map": "auto"}
        quant = os.environ.get("PRISM_HF_QUANT", "none").lower()
        if prequantized:
            # Native compressed-tensors quant drives the footprint; dtype="auto"
            # lets it engage. Any PRISM_HF_QUANT bitsandbytes request is ignored.
            if quant not in ("none", ""):
                print(
                    f"[local-llm] {model_id} already ships quantized weights "
                    f"(compressed-tensors); ignoring PRISM_HF_QUANT={quant!r} "
                    "and loading the checkpoint's native quantization."
                )
            load_kwargs["dtype"] = "auto"
        elif quant in ("4bit", "nf4", "bnb4"):
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif quant in ("8bit", "bnb8"):
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            # No bitsandbytes config: let transformers pick the checkpoint dtype.
            load_kwargs["dtype"] = "auto"

        print(
            f"[local-llm] loading {model_id} quant={quant} "
            "(this happens once per process)"
        )
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        model.eval()

        # Report the REALISED quantization so it's unambiguous whether 4-bit/8-bit
        # actually took effect (vs. silently loading full-precision because the
        # env var was unset). These flags are set by transformers' bnb integration.
        try:
            param_dtype = next(model.parameters()).dtype
        except StopIteration:
            param_dtype = "unknown"
        print(
            f"[local-llm] loaded {model_id} | "
            f"requested quant={'native(compressed-tensors)' if prequantized else quant} | "
            f"is_loaded_in_4bit={getattr(model, 'is_loaded_in_4bit', False)} "
            f"is_loaded_in_8bit={getattr(model, 'is_loaded_in_8bit', False)} | "
            f"param_dtype={param_dtype}"
        )

        _GEMMA_CACHE[model_id] = (model, processor)
    return _GEMMA_CACHE[model_id]


def _to_text(parsed) -> str:
    """Coerce ``processor.parse_response`` output to a plain string.

    ``parse_response`` may return the assistant text directly, or a structured
    dict (e.g. ``{"role": ..., "content": ...}``). Normalise to a string.
    """
    if isinstance(parsed, str):
        return parsed
    if isinstance(parsed, dict):
        return parsed.get("content") or parsed.get("text") or json.dumps(parsed)
    return str(parsed)


def _extract_json(text: str) -> str:
    """Best-effort extraction of a single JSON object from model output.

    Strips Markdown code fences and returns the substring spanning the first
    ``{`` to the last ``}`` so the downstream ``json.loads`` in
    ``TaskGraphGen.parse_response`` succeeds even if the model adds prose.
    """
    s = _to_text(text).strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1]
    return s


class LocalHFQueryClient:
    """Phase-1 populate client mirroring ``utils.GPTQueryClient``.

    This is a pure backend swap: it accepts and honours the SAME call
    parameters the OpenAI client does, so every other knob in the data
    generator is unchanged. ``temperature`` and ``max_tokens`` map directly to
    ``model.generate`` (temperature matches the ``GPTQueryClient`` default of
    0.31; ``max_tokens`` defaults to 20480 here — with thinking sharing the
    cap, the OpenAI-side 10240 truncates ~100-node populate JSON).
    ``reasoning_effort`` has no
    HF equivalent; thinking is enabled, and its ``<think>`` block is stripped by
    ``parse_response`` so the returned output is still a single well-formed JSON
    object (matching gpt-5.1 ``output_text``, whose reasoning never appears
    inline).
    """

    def __init__(self, model_id: Optional[str] = None, max_new_tokens: int = 20480):
        self.model, self.processor = load_gemma(model_id)
        self.max_new_tokens = max_new_tokens

    def query_gpt(
        self,
        query: str,
        temperature: Optional[float] = 0.31,
        max_tokens: Optional[int] = 20480,
        reasoning_effort: str = "low",
    ) -> str:
        return self.query_gpt_5(query, temperature, max_tokens, reasoning_effort)

    def query_gpt_5(
        self,
        query: str,
        temperature: Optional[float] = 0.31,
        max_tokens: Optional[int] = 20480,
        reasoning_effort: str = "low",
    ) -> str:
        messages = [{"role": "user", "content": query}]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]
        # Honour the temperature the caller passes (GPTQueryClient default 0.31).
        # temperature in (None, <=0) means greedy decoding.
        sample = temperature is not None and temperature > 0
        gen_kwargs = {"max_new_tokens": max_tokens or self.max_new_tokens}
        if sample:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs.update(do_sample=False)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        response = self.processor.decode(
            outputs[0][input_len:], skip_special_tokens=False
        )
        return _extract_json(self.processor.parse_response(response))

    def batch_query_gpt_5(
        self,
        queries: List[str],
        model: str = DEFAULT_GEMMA_MODEL,
        reasoning_effort: str = "low",
        poll_interval: int = 60,
    ) -> List[str]:
        """Sequential local fallback for the OpenAI Batch API.

        There is no offline batch endpoint, so queries are run one at a time
        (still ~free vs. the API). ``model`` / ``poll_interval`` are accepted
        only for signature compatibility with ``GPTQueryClient``.
        """
        return [
            self.query_gpt_5(q, reasoning_effort=reasoning_effort) for q in queries
        ]


class GemmaSpineClient:
    """Phase-2 SPINE client backed by the shared local Gemma model.

    Satisfies the SPINE ``client`` contract: ``query_llm(msg) -> (str, bool)``
    and ``format_prompt(base_request, graph_as_json)``. Shares the same loaded
    model as :class:`LocalHFQueryClient` via :func:`load_gemma`.

    Generation parameters mirror the repo's canonical HF planner client
    (``prism.models.inference.InMemoryLLM``): ``temperature=0.01``,
    ``min_p=0.1``, ``use_cache=True`` and the same ``format_prompt`` wording — so
    swapping in this backend changes only WHICH model answers, not how the
    planner is driven. The one deliberate deviation is ``max_new_tokens=4096``
    (vs. InMemoryLLM's 2048): with thinking enabled the ``<think>`` block shares
    the budget with the plan, so the cap is doubled to keep short SPINE plans
    from being truncated before ``answer(...)`` is emitted.
    """

    def __init__(self, model_id: Optional[str] = None, max_new_tokens: int = 4096):
        self.model, self.processor = load_gemma(model_id)
        self.max_new_tokens = max_new_tokens
        # pad token for generation; matches inference.InMemoryLLM's
        # pad_token_id=tokenizer.eos_token_id. Resolved defensively since
        # AutoProcessor wraps (but does not always expose) the tokenizer.
        tok = getattr(self.processor, "tokenizer", self.processor)
        self.pad_token_id = getattr(tok, "eos_token_id", None)

    def format_prompt(self, base_request: str, graph_as_json: str) -> List[dict]:
        return [
            {
                "role": "user",
                "content": f"task: {base_request}. scene graph {graph_as_json}",
            }
        ]

    def query_llm(self, msg: List[dict], max_new_tokens: Optional[int] = None):
        try:
            text = self.processor.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            inputs = self.processor(text=text, return_tensors="pt").to(
                self.model.device
            )
            input_len = inputs["input_ids"].shape[-1]
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens or self.max_new_tokens,
                    use_cache=True,
                    temperature=0.01,
                    min_p=0.1,
                    pad_token_id=self.pad_token_id,
                )
            response = self.processor.decode(
                outputs[0][input_len:], skip_special_tokens=False
            )
            return _to_text(self.processor.parse_response(response)), True
        except Exception as ex:  # noqa: BLE001 — surface as a planner failure
            print(f"[gemma-spine] generation failed: {ex}")
            return "Error: local generation failed", False


def hf_backend_enabled() -> bool:
    """True when ``PRISM_LLM_BACKEND`` selects the local HF/Gemma backend."""
    return os.environ.get("PRISM_LLM_BACKEND", "openai").lower() in (
        "hf",
        "gemma",
        "local",
    )
