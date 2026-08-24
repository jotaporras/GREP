"""vLLM backend for Phase-1 populate (plain-LLM data generation).

Why this exists
---------------
:class:`prism.data.local_llm.LocalHFQueryClient` implements
``batch_query_gpt_5`` by looping over the queries one at a time — eager HF
``generate`` has no offline batch endpoint, so a 36-graph populate run is 36
strictly sequential decodes at batch size 1. The prompts are mutually
independent, so that leaves nearly all of the GPU's throughput unused.

vLLM turns the same list into one continuously-batched ``llm.generate`` call:
paged KV cache, CUDA graphs, and many sequences in flight at once. The call
site in :mod:`prism.data.data_gen` already hands ``batch_query_gpt_5`` a LIST
of per-graph prompts, so this is a pure backend swap — no call-site change.

Scope: PLAIN-LLM only. The graph-augmented architectures (``rpearl_llm``,
``gt_llm``, ``graph_mask_llm``) inject positional encodings into the embedding
stream and patch ``config._attn_implementation``; vLLM's model runner cannot
execute those without a custom model plugin.

Phase-2 SPINE rollouts are supported via :class:`VLLMSpineClient`: N rollout
worker threads (``generate_example_plans(rollout_workers=N)``) each run the
unchanged sequential planning loop, and their per-turn prompts are micro-batched
into shared ``llm.generate`` calls by :func:`_gated_generate`.

Prompt/parse parity with the HF backend
---------------------------------------
Generation is the ONLY thing vLLM does here. The chat template is applied and
the response parsed with the same ``AutoProcessor`` the HF backend uses
(tokenizer-only; no weights are loaded), so the prompt string is byte-identical
and ``<think>`` stripping is identical. That keeps this a throughput change
rather than a silent change to what the model is asked or how it is read.

NOTE ON NUMERICS: vLLM's kernels are not bitwise-identical to eager HF even at
the same dtype, and any ``PRISM_VLLM_QUANT`` setting changes outputs outright.
Generated data is a research artifact — benchmark and diff before adopting a
config for a confirmatory corpus.

Model support: Gemma 4 12B Unified (``Gemma4UnifiedForConditionalGeneration``)
requires a vLLM NIGHTLY build; it is not in a stable release as of 0.26.0.

Environment
-----------
``PRISM_LLM_BACKEND=vllm``  selects this backend (see :func:`vllm_backend_enabled`).
``PRISM_VLLM_MODEL``        HF repo id; falls back to ``PRISM_HF_MODEL``, then
                            :data:`prism.data.local_llm.DEFAULT_GEMMA_MODEL`.
                            Point it at an AWQ/GPTQ repo for pre-quantized int4.
``PRISM_VLLM_TP``           tensor-parallel size (default 1). Only raise this
                            with a fast interconnect — over PCIe/SMP (`nvidia-smi
                            topo -m` shows ``SYS``) the per-layer all-reduce
                            usually costs more than the extra bandwidth buys.
``PRISM_VLLM_QUANT``        vLLM ``quantization`` value, e.g. ``bitsandbytes``
                            for in-flight NF4 from a bf16 checkpoint. Unset =
                            the checkpoint's native precision.
``PRISM_VLLM_GPU_UTIL``     ``gpu_memory_utilization`` (default 0.90).
``PRISM_VLLM_MAX_LEN``      ``max_model_len`` (default 16384).
``PRISM_VLLM_SEED``         engine seed (default 0) — set for reproducible runs.
"""

import os
import threading
from typing import List, Optional

from prism.data import local_llm

# Engine + processor are cached per (model_id, engine-config) so a process that
# builds several clients loads the weights exactly once.
_VLLM_CACHE = {}

# ---------------------------------------------------------------------------
# Thread-safe gated generation.
#
# The sync ``vllm.LLM`` engine does not support concurrent ``generate`` calls
# from multiple threads. Phase-2 rollouts run N SPINE dialogues on N worker
# threads, each issuing one prompt per planner turn. The gate below serialises
# engine access while MICRO-BATCHING: whichever thread wins the engine lock
# drains every request that queued up while the previous batch was decoding and
# submits them as ONE ``llm.generate`` call, so concurrent dialogues are batched
# together without an extra dispatcher thread or an async engine.
# ---------------------------------------------------------------------------

_ENGINE_LOCK = threading.Lock()  # one llm.generate in flight at a time
_PENDING_LOCK = threading.Lock()  # protects _PENDING
_PENDING: list = []


class _Request:
    __slots__ = ("prompt", "params", "event", "output")

    def __init__(self, prompt, params):
        self.prompt = prompt
        self.params = params
        self.event = threading.Event()
        self.output = None


def _gated_generate(llm, prompt: str, params):
    """Generate one completion, batching with other threads' pending requests.

    Enqueues the request, then loops: if another thread's batch already served
    it, return; otherwise try to become the driver, drain the queue, and run one
    ``llm.generate`` over the whole batch. Output order follows prompt order, so
    results map back to requests positionally.
    """
    req = _Request(prompt, params)
    with _PENDING_LOCK:
        _PENDING.append(req)
    while True:
        if req.event.wait(timeout=0.05):
            return req.output
        if _ENGINE_LOCK.acquire(blocking=False):
            try:
                with _PENDING_LOCK:
                    batch = list(_PENDING)
                    _PENDING.clear()
                if batch:
                    outputs = llm.generate(
                        [r.prompt for r in batch], [r.params for r in batch]
                    )
                    for r, out in zip(batch, outputs):
                        r.output = out
                        r.event.set()
            finally:
                _ENGINE_LOCK.release()
            if req.event.is_set():
                return req.output


def _engine_config() -> dict:
    """Read the vLLM engine knobs from the environment.

    Returns a dict of ``LLM(...)`` kwargs. Kept in one place so the benchmark
    harness and the populate path cannot drift apart.
    """
    quant = os.environ.get("PRISM_VLLM_QUANT") or None
    return {
        "tensor_parallel_size": int(os.environ.get("PRISM_VLLM_TP", "1")),
        "quantization": quant,
        "gpu_memory_utilization": float(os.environ.get("PRISM_VLLM_GPU_UTIL", "0.90")),
        "max_model_len": int(os.environ.get("PRISM_VLLM_MAX_LEN", "16384")),
        "seed": int(os.environ.get("PRISM_VLLM_SEED", "0")),
    }


def load_vllm(model_id: Optional[str] = None):
    """Load (and cache) the vLLM engine + the matching HF processor.

    ``model_id`` defaults to ``$PRISM_VLLM_MODEL``, then ``$PRISM_HF_MODEL``,
    then :data:`prism.data.local_llm.DEFAULT_GEMMA_MODEL`.

    Returns ``(llm, processor)``. The processor is tokenizer-only (no weights)
    and supplies ``apply_chat_template`` / ``parse_response`` so prompt
    construction matches the HF backend exactly.
    """
    model_id = (
        model_id
        or os.environ.get("PRISM_VLLM_MODEL")
        or os.environ.get("PRISM_HF_MODEL")
        or local_llm.DEFAULT_GEMMA_MODEL
    )
    cfg = _engine_config()
    key = (model_id, tuple(sorted(cfg.items(), key=lambda kv: kv[0])))
    if key not in _VLLM_CACHE:
        # Imported lazily so importing this module never forces a vLLM import
        # (the OpenAI and HF paths must keep working in envs without vLLM).
        # The datagen pipeline touches CUDA (torch import chains) before the
        # engine starts; vLLM's default fork-ed TP workers then die with
        # "CUDA error: initialization error". Spawned workers are immune.
        # setdefault so an explicit caller override still wins.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        from transformers import AutoProcessor, AutoTokenizer
        from vllm import LLM

        print(
            f"[vllm] loading {model_id} tp={cfg['tensor_parallel_size']} "
            f"quant={cfg['quantization'] or 'native'} "
            f"max_model_len={cfg['max_model_len']} seed={cfg['seed']}"
        )
        try:
            processor = AutoProcessor.from_pretrained(model_id)
        except ValueError:
            # Text-only repos (e.g. google/gemma-4-E4B-it) ship a tokenizer but
            # no processor class. The tokenizer also provides
            # apply_chat_template; <think>-stripping then falls back to the
            # regex in _parse_response since tokenizers lack parse_response.
            processor = AutoTokenizer.from_pretrained(model_id)
        llm = LLM(model=model_id, **cfg)
        _VLLM_CACHE[key] = (llm, processor)
    return _VLLM_CACHE[key]


def _parse_response(processor, text: str, prefix: str = "") -> str:
    """Strip the ``<think>`` block from raw model text.

    Uses ``processor.parse_response`` when available (AutoProcessor for the
    full Gemma repos, HF-backend parity); otherwise (tokenizer-only repos)
    removes any ``<think>...</think>`` span — or everything through a lone
    closing tag — by regex. ``prefix`` is the rendered prompt that preceded
    generation: transformers >= 5.14 requires it because chat templates may
    pre-write part of the assistant turn (e.g. an opening ``<think>`` tag)
    that the parser must see. Older parse_response signatures ignore it.
    """
    parse = getattr(processor, "parse_response", None)
    if parse is not None:
        try:
            return local_llm._to_text(parse(text, prefix=prefix))
        except TypeError:  # transformers < 5.14: no prefix kwarg
            return local_llm._to_text(parse(text))
    import re

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


class VLLMQueryClient:
    """Phase-1 populate client mirroring :class:`local_llm.LocalHFQueryClient`.

    Same call surface as ``utils.GPTQueryClient`` (``query_gpt`` /
    ``query_gpt_5`` / ``batch_query_gpt_5``), so ``TaskGraphGen`` swaps backends
    with no other change. The difference is ``batch_query_gpt_5``, which issues
    ONE continuously-batched ``llm.generate`` over the whole prompt list instead
    of a Python loop.

    ``reasoning_effort`` is accepted for signature compatibility and ignored —
    Gemma's thinking mode has no effort dial; the ``<think>`` block is stripped
    by ``processor.parse_response`` exactly as on the HF backend.
    """

    def __init__(self, model_id: Optional[str] = None, max_new_tokens: int = 20480):
        self.llm, self.processor = load_vllm(model_id)
        self.max_new_tokens = max_new_tokens

    def _render(self, query: str) -> str:
        """Apply the Gemma chat template with thinking enabled (HF-backend parity)."""
        messages = [{"role": "user", "content": query}]
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

    def _sampling_params(self, temperature: Optional[float], max_tokens: Optional[int]):
        """Build SamplingParams matching the HF backend's decode policy.

        ``temperature`` of ``None`` or ``<= 0`` means greedy (vLLM spells that
        ``temperature=0.0``). ``skip_special_tokens=False`` is REQUIRED: the
        ``<think>`` delimiters must survive detokenization for
        ``parse_response`` to find and strip the reasoning block.

        The per-request seed is set ONLY when PRISM_VLLM_SEED is explicitly
        exported. A fixed default seed would make retries pointless: a graph
        whose populate response was malformed JSON would re-sample the exact
        same malformed output on every resume.
        """
        from vllm import SamplingParams

        sample = temperature is not None and temperature > 0
        seed_env = os.environ.get("PRISM_VLLM_SEED")
        return SamplingParams(
            temperature=temperature if sample else 0.0,
            max_tokens=max_tokens or self.max_new_tokens,
            skip_special_tokens=False,
            seed=int(seed_env) if seed_env is not None else None,
        )

    def _finish(self, output) -> str:
        """Parse one vLLM RequestOutput into the JSON string the caller expects."""
        text = output.outputs[0].text
        return local_llm._extract_json(
            _parse_response(self.processor, text, prefix=output.prompt or "")
        )

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
        output = _gated_generate(
            self.llm,
            self._render(query),
            self._sampling_params(temperature, max_tokens),
        )
        return self._finish(output)

    def batch_query_gpt_5(
        self,
        queries: List[str],
        model: str = local_llm.DEFAULT_GEMMA_MODEL,
        reasoning_effort: str = "low",
        poll_interval: int = 60,
        temperature: Optional[float] = 0.31,
        max_tokens: Optional[int] = 20480,
    ) -> List[str]:
        """Generate all ``queries`` in ONE continuously-batched vLLM call.

        This is the whole point of the backend: the HF client runs the same list
        sequentially at batch size 1. ``model`` / ``poll_interval`` are accepted
        only for signature compatibility with ``GPTQueryClient``'s OpenAI Batch
        API path. Returns one JSON string per query, in input order (vLLM
        preserves the ordering of the prompt list).
        """
        with _ENGINE_LOCK:
            outputs = self.llm.generate(
                [self._render(q) for q in queries],
                self._sampling_params(temperature, max_tokens),
            )
        return [self._finish(o) for o in outputs]


class VLLMSpineClient:
    """Phase-2 SPINE client backed by the shared vLLM engine.

    Satisfies the SPINE ``client`` contract (``query_llm(msg) -> (str, bool)``
    and ``format_prompt``) exactly like :class:`local_llm.GemmaSpineClient`,
    with the same decode policy (``temperature=0.01``, ``min_p=0.1``,
    ``max_new_tokens=4096``, thinking enabled and stripped by
    ``parse_response``). Unlike the HF client it is THREAD-SAFE: concurrent
    rollout workers each call ``query_llm`` and the gate micro-batches their
    planner turns into shared ``llm.generate`` calls.
    """

    def __init__(self, model_id: Optional[str] = None, max_new_tokens: int = 4096,
                 enable_thinking: bool = True):
        self.llm, self.processor = load_vllm(model_id)
        self.max_new_tokens = max_new_tokens
        # e20 path-only distillation: enable_thinking=False renders the Gemma
        # chat template's no-think variant so the teacher answers directly.
        # parse_response still runs (it is a no-op when no <think> block exists).
        self.enable_thinking = enable_thinking

    def format_prompt(self, base_request: str, graph_as_json: str) -> List[dict]:
        return [
            {
                "role": "user",
                "content": f"task: {base_request}. scene graph {graph_as_json}",
            }
        ]

    def query_llm(self, msg: List[dict], max_new_tokens: Optional[int] = None):
        from vllm import SamplingParams

        try:
            text = self.processor.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            params = SamplingParams(
                temperature=0.01,
                min_p=0.1,
                max_tokens=max_new_tokens or self.max_new_tokens,
                skip_special_tokens=False,
            )
            output = _gated_generate(self.llm, text, params)
            response = output.outputs[0].text
            return _parse_response(self.processor, response, prefix=text), True
        except Exception as ex:  # noqa: BLE001 — surface as a planner failure
            print(f"[vllm-spine] generation failed: {ex}")
            return "Error: local generation failed", False


def vllm_backend_enabled() -> bool:
    """True when ``PRISM_LLM_BACKEND`` selects the vLLM backend."""
    return os.environ.get("PRISM_LLM_BACKEND", "openai").lower() == "vllm"
