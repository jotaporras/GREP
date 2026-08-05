import copy
import importlib
import re
import warnings
from collections import defaultdict

import torch
from torch import nn
from torch_geometric.data import Batch
import transformers.masking_utils as masking_utils
from transformers import AttentionInterface, PreTrainedModel


# Attention implementation name: GraphAugmentedLLM routes every decoder layer through
# _prism_pe_attention_forward to inject Ψ post-RoPE (see _install_pe_injection).
_PRISM_PE_IMPL = "prism_pe"


def _prism_pe_attention_forward(module, query, key, value, attention_mask,
                                scaling=None, dropout=0.0, **kwargs):
    """Attention fn that injects graph signal Ψ post-RoPE into q/k/v.

    Registered as ``"prism_pe"``. Receives q/k already rotary-embedded as [B, H, S, head_dim].
    Adds unrotated W_q·Ψ / W_k·Ψ (and W_v·Ψ) then delegates to the LLM's original attention impl.
    With Ψ absent the output is identical to stock attention.
    """
    model = getattr(module, "_prism_pe_model", None)
    psi = None if model is None else model._pe_signal
    # Inject only on the prompt forward whose Ψ length matches the query length;
    # cached single-token decode steps (S mismatch) fall through to stock attn.
    if psi is not None and psi.shape[0] == query.shape[0] and psi.shape[1] == query.shape[-2]:
        # Device too, not just dtype: under multi-GPU sharding (device_map="auto")
        # Ψ lives on the embeddings' device while this layer may be elsewhere
        # (mirrors _graph_mask_attention_forward's bias.to below).
        psi = psi.to(device=query.device, dtype=query.dtype)
        b, s, hd = psi.shape[0], psi.shape[1], module.head_dim
        query = query + module.q_proj(psi).view(b, s, -1, hd).transpose(1, 2)
        # KV-shared layers (e.g. gemma-4) reuse k/v from an earlier layer that
        # already carries Ψ; re-injecting here would double-count, so skip k/v.
        if not getattr(module, "is_kv_shared_layer", False):
            key = key + module.k_proj(psi).view(b, s, -1, hd).transpose(1, 2)
            if getattr(model, "_pe_inject_value", True) and getattr(module, "v_proj", None) is not None:
                value = value + module.v_proj(psi).view(b, s, -1, hd).transpose(1, 2)
    return module._prism_orig_attn_fn(
        module, query, key, value, attention_mask,
        scaling=scaling, dropout=dropout, **kwargs)


AttentionInterface.register(_PRISM_PE_IMPL, _prism_pe_attention_forward)


# Attention implementation name: GraphMaskLLM routes every decoder layer through
# _graph_mask_attention_forward. Unlike prism_pe, touches nothing in q/k/v —
# only adds a graph-adjacency bias to the attention mask.
_GRAPH_MASK_IMPL = "prism_graph_mask"


def _graph_mask_attention_forward(module, query, key, value, attention_mask,
                                  scaling=None, dropout=0.0, **kwargs):
    """Attention fn that folds a graph-adjacency mask into ``attention_mask``.

    Registered as ``"prism_graph_mask"``. Adds ``self._struct_bias`` ([B, 1, seq, seq],
    0 = allowed / finfo.min = blocked) to the model's causal/sliding mask; q/k/v untouched.
    Cached decode steps (bias length mismatch) fall through to stock attention.
    """
    model = getattr(module, "_graph_mask_model", None)
    # Layer-scope gate: a layer routed through this impl but flagged inactive
    # (e.g. sliding-window layers under LearnableGraphMaskLLM's dense-only scope)
    # delegates to stock attention untouched. Absent flag ⇒ active (GraphMaskLLM).
    active = getattr(module, "_graph_mask_active", True)
    bias = None if (model is None or not active) else model._struct_bias
    q_len = query.shape[-2]
    k_len = key.shape[-2]
    # Decode-time extension (design note §2.2): a per-step [1, 1, 1, K] bias row armed
    # by MaskDecodeInjector replaces the historical silent fall-through on cached
    # decode steps. Layers whose cached key length differs (e.g. cropped sliding
    # windows) still fall through — run decode_consistent arms with
    # mask_layer_scope=dense so train and decode wire the same layers.
    row = None if (model is None or not active) else model._decode_bias_row
    if (row is not None and q_len == 1 and k_len == row.shape[-1]
            and not (bias is not None and q_len == bias.shape[-2]
                     and k_len == bias.shape[-1])):
        # The prefill _struct_bias stays armed throughout generate (its shape only
        # matches the prefill forward), so the decode row applies whenever the
        # prefill branch below does not.
        row = row.to(device=query.device, dtype=query.dtype)
        if attention_mask is None:
            attention_mask = row
        else:
            am = attention_mask[..., :k_len]
            if am.dtype == torch.bool:
                am = torch.zeros_like(am, dtype=query.dtype).masked_fill(
                    ~am, torch.finfo(query.dtype).min)
            else:
                am = am.to(device=query.device, dtype=query.dtype)
            attention_mask = am + row
    if bias is not None and q_len == bias.shape[-2] and k_len == bias.shape[-1]:
        bias = bias.to(device=query.device, dtype=query.dtype)
        if attention_mask is None:
            # SDPA's is_causal fast path passes no mask (full-attention layer, no
            # padding); supplying an explicit mask disables it, so fold the causal
            # triangle into the structural bias. Sliding-window layers always pass an
            # explicit mask (they can't use is_causal), so they hit the else branch
            # and keep their band — only the structural −inf is added on top.
            neg = torch.finfo(query.dtype).min
            causal = torch.triu(
                torch.full((q_len, k_len), neg, device=query.device, dtype=query.dtype),
                diagonal=1)
            attention_mask = bias + causal[None, None]
        else:
            am = attention_mask[..., :k_len]
            # buggy_fold reproduces the ORIGINAL pre-fix behavior for an A/B ablation: a
            # BOOLEAN mask (SDPA's True=attend) is cast bool→float so False→0.0, making
            # blocked positions additive 0.0 (attendable). At the sliding-window layers HF
            # supplies an explicit boolean mask even at batch=1, so this leaks future and
            # out-of-window tokens (near-bidirectional attention). Default (False) converts
            # bool→0/−inf, the correct additive convention.
            buggy_fold = getattr(module, "_graph_mask_buggy_fold", False)
            if am.dtype == torch.bool and not buggy_fold:
                am = torch.zeros_like(am, dtype=query.dtype).masked_fill(
                    ~am, torch.finfo(query.dtype).min)
            else:
                am = am.to(device=query.device, dtype=query.dtype)
            attention_mask = am + bias
    return module._graph_mask_orig_attn_fn(
        module, query, key, value, attention_mask,
        scaling=scaling, dropout=dropout, **kwargs)


AttentionInterface.register(_GRAPH_MASK_IMPL, _graph_mask_attention_forward)


# Attention implementation name: WireGraphLLM routes the in-scope decoder layers
# through _wire_attention_forward, which composes a SECOND rotation onto the
# already-RoPE'd q/k. The rotation is the ONLY path — there is no score-level
# variant — which is what keeps the arch linear-attention compatible. See
# WireGraphLLM.
_WIRE_IMPL = "prism_wire"

# "rotate" = the real decode path (prompt-position keys keep their graph phase, see
# _wire_attention_forward); "skip" = the labeled diagnostic in which WIRE is OFF for
# every cached step. "error" is a LEGACY value: it existed only while decode-time key
# rotation was unimplemented, and is normalised to "rotate" in WireGraphLLM.__init__ so
# checkpoints trained before this landed still evaluate.
WIRE_DECODE_MODES = ("rotate", "skip")
WIRE_DECODE_LEGACY = {"error": "rotate"}

# ω initialisation strategies for wire_vanilla=True. These are exactly the reference
# implementation's `cfg.gt.graphrope.init_omega` values, with its default ("zero"):
# cederikhoefs/Graph-RoPE, graphgps/config/gt_config.py:75 and the init_omega_matrix
# helper in graphgps/layer/graphrope.py. Only used in vanilla mode — the expectation
# arm draws ε ~ N(0, I) and scales it by the learnable σ instead.
#   "zero"        nn.init.zeros_ ⇒ θ = 0 at step 0 ⇒ the rotation is EXACTLY the
#                 identity and ω learns away from it. Trainable at that point only
#                 because the sin term is retained: ∂sin(θ)/∂ω = cos(θ)·r = r ≠ 0
#                 while ∂cos(θ)/∂ω = −sin(θ)·r = 0. This is the reference DEFAULT.
#   "uniform"     the reference's `"uniform"` branch is a bare `pass`, leaving torch's
#                 default nn.Linear init, i.e. U(−1/√m, 1/√m). Reproduced literally.
#   "exponential" the RoPE-style decay of the reference: rand(P, m) / 10000^(2i/P).
#                 NOTE two quirks reproduced verbatim from the source: the exponent
#                 denominator is P (= d/2), twice the decay rate of standard RoPE's
#                 10000^(−2i/d); and the decay is CONSTANT along the spectral axis, so
#                 every one of the m coordinates shares one profile.
WIRE_VANILLA_OMEGA_INITS = ("zero", "uniform", "exponential")


def wire_rope_planes(attn, rotate_nope: bool) -> int:
    """Number of 2-D planes WIRE rotates on ``attn`` (out of ``head_dim // 2``).

    Gemma's global layers use ``rope_type="proportional"`` with a
    ``partial_rotary_factor``: only ``int(factor * head_dim // 2)`` leading planes
    carry text RoPE and the remainder are NoPE (zero inv_freq ⇒ cos=1, sin=0), i.e.
    channels the model treats as position-free content. This mirrors the upstream
    formula in ``modeling_rope_utils._compute_proportional_rope_parameters`` (the
    ``rope_angles`` line) rather than hard-coding a count, so it tracks the config.

    ``rotate_nope=True`` returns every plane (graph phase also enters the NoPE
    channels — a deliberate, auditable choice, not the default).

    The return value IS WIRE's Monte-Carlo sample count for the layer (see
    :class:`WireGraphLLM`), and it is not uniform across layer types. gemma-4-31B::

        layer type   head_dim   rotate_nope=False   rotate_nope=True
        global        512        64  (factor 0.25)   256
        sliding       256       128  (no factor)     128
    """
    head_dim = int(attn.head_dim)
    total = head_dim // 2
    if rotate_nope:
        return total
    params = getattr(attn.config, "rope_parameters", None) or {}
    layer_type = getattr(attn, "layer_type", None)
    if isinstance(params, dict) and layer_type in params:
        params = params[layer_type] or {}
    factor = (params or {}).get("partial_rotary_factor", 1.0)
    return int(factor * head_dim // 2)


def wire_cos_sin(r, omega, head_dim: int):
    """cos/sin of the WIRE angles ``θ_n = ω_n · r``, laid out for Gemma's rotary.

    Args:
        r: ``[B, S, m]`` graph feature per position (0 at non-graph positions).
        omega: ``[P, m]`` frequency table, ``P <= head_dim // 2`` planes. ONE table per
            layer, shared by every head — so this is computed once per layer, not once
            per head.
        head_dim: attention head width.

    Returns ``(cos, sin)`` shaped ``[B, S, head_dim]``, broadcast over heads by the
    caller. Planes beyond ``P`` are padded with **zero** angle, so ``cos=1, sin=0``
    there and those channels are left exactly untouched. Both halves carry the same
    angle because ``rotate_half`` pairs channel ``n`` with ``n + head_dim/2``
    (matching upstream's ``cat((freqs, freqs))``).

    Computed in fp32: ``θ`` is a length-``m`` dot product (m is 1024 by default) and
    cos/sin are periodic, so a bf16 reduction produces an angle error that does NOT
    shrink with magnitude. Mirrors upstream's own forced-fp32 rotary
    (``Gemma4TextRotaryEmbedding.forward``).
    """
    theta = r.float() @ omega.float().t()                    # [B, S, P]
    pad = head_dim // 2 - theta.shape[-1]
    if pad < 0:
        raise ValueError(
            f"omega has {theta.shape[-1]} planes but head_dim//2 = {head_dim // 2}")
    if pad > 0:
        theta = torch.cat(
            (theta, theta.new_zeros(*theta.shape[:-1], pad)), dim=-1)
    emb = torch.cat((theta, theta), dim=-1)                  # [..., head_dim]
    return emb.cos(), emb.sin()


def _wire_rotate(x, cos, sin):
    """WIRE's Eq. 11 efficient instantiation: ``cos ⊙ z + sin ⊙ Pz``.

    ``x`` is ``[B, H, S, hd]``, cos/sin are ``[B, S, hd]``. ``O(d)`` per token, and it
    allocates nothing whose shape depends on the graph size — the paper's
    "parameters/compute independent of N" property (§3.1) holds literally here.

    ``P`` is the pairing permutation. The paper writes it as the **alternate-entry**
    swap (channels 2n, 2n+1), but Gemma pairs channel ``n`` with ``n + hd/2`` — its
    ``rotate_half``. Using the paper's literal ``P`` here would rotate different
    channel pairs than the text RoPE does, so the two rotations would no longer act in
    the same 2-D planes and would NOT commute; the composition with RoPE that makes
    this hook point valid would be lost. We therefore use Gemma's own pairing, which is
    the same operator up to a fixed channel relabeling.

    Out-of-place throughout (autograd-safe). ``cos``/``sin`` are ``[B, S, hd]`` and
    BROADCAST over the head axis — the angles are shared by every head in the layer, so
    they are computed once per layer and no per-head copy is ever materialized. This
    also makes the q-side and k-side rotations use identical frequencies under GQA,
    which is what keeps Eq. 3 exact when a key head is shared by several query heads.
    """
    cos = cos.unsqueeze(1).to(x.dtype)
    sin = sin.unsqueeze(1).to(x.dtype)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


def _wire_rotate_prefix(x, cos, sin, n: int):
    """Rotate only the FIRST ``n`` positions of ``x`` ``[B, H, S, hd]``.

    The tail carries ``r = 0`` ⇒ ``θ = 0`` ⇒ ``cos=1, sin=0`` ⇒ the identity, so it is
    returned untouched and no cos/sin is built for it at all. ``cos``/``sin`` are
    ``[B, S', hd]`` with ``S' >= n`` and are sliced, never padded. Out-of-place
    throughout (autograd-safe), same as :func:`_wire_rotate`.
    """
    if n <= 0:
        return x
    seq = x.shape[-2]
    if n >= seq:
        return _wire_rotate(x, cos[:, :seq], sin[:, :seq])
    head = _wire_rotate(x[..., :n, :], cos[:, :n], sin[:, :n])
    return torch.cat((head, x[..., n:, :]), dim=-2)


def wire_decode_key_slots_aligned(module) -> bool:
    """True iff ``key slot j == token index j`` holds for ``module``'s KV cache.

    That identity is what lets a cached step re-apply ``θ_j`` to key slot ``j``. It holds
    for every full-attention layer — ``DynamicLayer.update`` only ever appends — and fails
    for a sliding-window layer, which DROPS leading entries once the sequence passes its
    window (``DynamicSlidingWindowLayer``, transformers 5.14.1). ``is_kv_shared_layer``
    cannot reach here: :meth:`WireGraphLLM._install_wire` deactivates those outright.

    Deliberately NOT checked against ``position_ids``: those are ROPE positions, not cache
    slots, and ``generate`` re-derives them from the attention mask — so any padded prompt
    (or a prompt token that happens to equal ``pad_token_id``) shifts them below the slot
    index while the cache is still perfectly aligned. Comparing the two conflates padding
    with cropping and aborts healthy runs.
    """
    return not getattr(module, "is_sliding", False)


def _wire_attention_forward(module, query, key, value, attention_mask,
                            scaling=None, dropout=0.0, **kwargs):
    """Attention fn applying WIRE to the already-RoPE'd q/k.

    Registered as ``"prism_wire"``. Receives q/k as ``[B, H, S, head_dim]`` AFTER
    ``apply_rotary_pos_emb`` (``modeling_gemma4.py:1251`` for q, ``:1267`` for k) and
    AFTER the KV-cache write (``:1274``); Gemma dispatches here at ``:1282``. (Line
    numbers verified against transformers 5.14.1.) Two consequences: the text RoPE is
    already applied, so this COMPOSES rather than replaces — and the cache stores
    UNROTATED keys, which is what the decode branch below exists to repair.

    Rotates q and k by ``R(θ)``, ``θ_n = ω_n·r``. Same-plane rotations commute, so the
    score picks up ``r_j − r_i`` on top of the text phase ``p_j − p_i``. ``v`` and the
    attention mask are untouched and NO ``S×S`` (or ``N×N``) object is ever built —
    that is what keeps this linear-attention compatible rather than a bias-style RPE.

    **Decode semantics (``wire_decode='rotate'``, the default).** ``r`` is a function of
    POSITION only: :meth:`WireGraphLLM.build_wire_signal` reads the prompt injection map
    once, so ``r_j`` is fixed for the whole rollout the moment node spans are known.
    ``Cache.update`` returns the concatenation of the stored (unrotated) prefill keys and
    the new key, so re-applying ``R(θ_j)`` to key slot ``j`` here reconstructs exactly the
    key the prefill forward scored against — the score is again a function of
    ``r_j − r_i`` only, and Theorem 3's relative-only hypothesis holds unchanged at every
    step. Generated positions are absent from the injection map ⇒ ``r = 0`` ⇒ their keys
    and the decode query are left bit-identical, which is the same prompt-only decode
    wiring the additive (``GraphAugmentedLLM``) and mask families already use at eval.
    The identity the re-rotation rests on — ``key slot j == token index j`` — is checked
    per cached step by :func:`wire_decode_key_slots_aligned`; note it is a statement about
    CACHE SLOTS, not about ``position_ids`` (see that function for why).

    With the signal absent, or on an inactive layer, this delegates untouched (``r=0``
    would also give the identity, but the early return avoids the work entirely).
    """
    model = getattr(module, "_wire_model", None)
    active = getattr(module, "_wire_active", False)
    r = None if (model is None or not active) else model._wire_signal
    if r is None:
        return module._wire_orig_attn_fn(
            module, query, key, value, attention_mask,
            scaling=scaling, dropout=dropout, **kwargs)

    q_len, k_len, sig_len = query.shape[-2], key.shape[-2], r.shape[1]
    if r.shape[0] != query.shape[0]:
        raise RuntimeError(
            f"WireGraphLLM: graph signal batch {r.shape[0]} != attention batch "
            f"{query.shape[0]}. The signal is armed per forward from the SAME batch "
            "(WireGraphLLM.forward / generate_with_graph); a mismatch means a stale "
            "signal leaked across batches.")

    # ONE table for the whole layer (ω = σ_ℓ·ε), so the angles are computed once here
    # and broadcast over every head — H× less modulation work than a per-head table,
    # and the same frequencies necessarily rotate q and k, keeping Eq. 3 exact under GQA.
    omega = model.layer_omega(module.layer_idx, query.device)      # [P, m]

    if q_len == k_len == sig_len:
        # Uncached forward (training, teacher-forced scoring, prefill): q and k are the
        # same block and both carry the graph phase.
        cos, sin = wire_cos_sin(r.to(device=query.device), omega, module.head_dim)
        query = _wire_rotate(query, cos, sin)
        key = _wire_rotate(key, cos, sin)
    elif model._wire_decode == "skip":
        pass                       # labeled diagnostic: WIRE OFF for every cached step
    else:
        if not wire_decode_key_slots_aligned(module):
            raise NotImplementedError(
                f"WireGraphLLM: layer {module.layer_idx} is a sliding-window layer "
                f"reached at a cached step (q_len={q_len}, k_len={k_len}, "
                f"signal_len={sig_len}). Its KV cache drops leading entries past the "
                "window, so key slot j is no longer token index j and the graph phase "
                "cannot be aligned. Set gnn.wire_layer_scope to a dense scope ('dense', "
                "'dense_top_half', 'dense_first') — those carry WIRE on full-attention "
                "layers only, whose cache is never cropped — or gnn.wire_decode='skip' "
                "to run WIRE off at decode as a labeled diagnostic. "
                "(WireGraphLLM.generate_with_graph refuses this up front; reaching here "
                "means generate was driven around it.)")
        # key slot j == token index j on a full-attention cache, and Cache.update returns
        # keys ending where the queries do — so query slot 0 is token index k_len - q_len.
        q_start = k_len - q_len
        cos, sin = model.decode_cos_sin(module.layer_idx, omega, module.head_dim,
                                        query.device)
        key = _wire_rotate_prefix(key, cos, sin, min(sig_len, k_len))
        # Query slots are absolute q_start..q_start+q_len-1; only those inside the
        # prompt carry a phase (generated positions are absent from the injection map).
        if q_start < sig_len:
            query = _wire_rotate_prefix(query, cos[:, q_start:], sin[:, q_start:],
                                        min(sig_len - q_start, q_len))
    return module._wire_orig_attn_fn(
        module, query, key, value, attention_mask,
        scaling=scaling, dropout=dropout, **kwargs)


AttentionInterface.register(_WIRE_IMPL, _wire_attention_forward)


def _wire_resolve_orig_attn_fn(wrapper, first_attn):
    """Resolve the layers' pre-patch attention fn and mirror its mask registration.

    The impl name is captured ONCE (configs are shared across layers, so a
    post-mutation read would already see ``prism_wire``) and persisted for idempotent
    re-install. ``prism_wire`` is registered in the mask registry mirroring the original
    impl so HF builds the right causal/sliding mask for the delegated fn
    (``create_causal_mask`` returns ``None`` for an unregistered impl, which would
    silently disable causal masking).

    Mirrors the identical dance inside the other wrappers' ``_install_*`` methods and
    is deliberately NOT shared with them — the WIRE arch is kept self-contained so it
    cannot perturb the existing architectures. Used only by :class:`WireGraphLLM`.
    """
    mod = importlib.import_module(type(first_attn).__module__)
    if not hasattr(wrapper, "_wire_orig_attn_impl"):
        impl = first_attn.config._attn_implementation
        wrapper._wire_orig_attn_impl = "eager" if impl == _WIRE_IMPL else impl
    orig_impl = wrapper._wire_orig_attn_impl
    # Resolve original attention fn: ≥5.12 uses get_interface(); older subscripts the registry.
    attn_fns = mod.ALL_ATTENTION_FUNCTIONS
    if hasattr(attn_fns, "get_interface"):
        orig_attn_fn = attn_fns.get_interface(orig_impl, mod.eager_attention_forward)
    else:
        orig_attn_fn = (
            mod.eager_attention_forward if orig_impl == "eager" else attn_fns[orig_impl]
        )
    mask_fns = getattr(masking_utils, "ALL_MASK_ATTENTION_FUNCTIONS", None)
    if mask_fns is not None and orig_impl in mask_fns._global_mapping:
        masking_utils.AttentionMaskInterface.register(
            _WIRE_IMPL, mask_fns._global_mapping[orig_impl])
    return mod, orig_attn_fn


MASK_LAYER_SCOPES = ("all", "dense", "dense_top_half", "dense_first")


def resolve_mask_active_flags(layers, layer_scope) -> list[bool]:
    """Per-layer activation of the structural mask under ``layer_scope``.

    - ``"all"``: every self-attn layer (historical).
    - ``"dense"``: every non-sliding (Gemma full_attention / global) layer.
    - ``"dense_top_half"``: the LATER half of the global layers (10 globals → the
      last 5, i.e. the deeper ones).
    - ``"dense_first"``: only the first (shallowest) global layer.

    All dense_* scopes are subsets of "dense", so they stay compatible with
    decode-time injection (sliding layers never carry the channel).
    """
    if layer_scope not in MASK_LAYER_SCOPES:
        raise ValueError(f"layer_scope must be one of {MASK_LAYER_SCOPES}, got {layer_scope!r}")
    sliding = [bool(getattr(layer.self_attn, "is_sliding", False)) for layer in layers]
    if layer_scope == "all":
        return [True] * len(layers)
    globals_idx = [i for i, sl in enumerate(sliding) if not sl]
    if layer_scope == "dense":
        keep = set(globals_idx)
    elif layer_scope == "dense_top_half":
        keep = set(globals_idx[len(globals_idx) // 2:])
    else:  # dense_first
        keep = {globals_idx[0]}
    return [i in keep for i in range(len(layers))]


def tok2node_vector(injection_map, seq_len, device) -> torch.Tensor:
    """``[seq_len]`` long tensor: token position → node id, −1 for non-node tokens.

    Spans are disjoint (``build_injection_map`` dedups longest-first), so each token
    maps to at most one node; spans extending past ``seq_len`` are truncated.
    """
    tok2node = torch.full((seq_len,), -1, dtype=torch.long, device=device)
    for node_idx, spans in injection_map.items():
        for start, end in spans:
            end = min(end, seq_len)
            if start < end:
                tok2node[start:end] = node_idx
    return tok2node


def graph_token_position_ids(injection_maps, seq_len: int, device) -> torch.Tensor:
    """``[B, seq_len]`` position_ids with the injected spans set to 0 (identity RoPE).

    Non-graph tokens keep their natural ``arange`` index. Causality is unaffected (HF
    derives the causal mask from cache_position, not position_ids). Shared by every
    architecture that honours ``disable_graph_token_rope`` — the additive family
    (:class:`GraphAugmentedLLM`) and the learned mask (:class:`LearnableGraphMaskLLM`) —
    so there is ONE definition of which positions get identity RoPE.

    The caller decides WHICH map to pass, and that choice is the arch's contract: both
    callers pass the QUERY-role map, i.e. exactly the positions carrying the graph
    channel under ``data.injection_scope``. A ``prompt_only`` run therefore matches
    generation exactly; see :meth:`LearnableGraphMaskLLM.forward` for the
    ``decode_consistent`` case.
    """
    B = len(injection_maps)
    pos = torch.arange(seq_len, device=device).unsqueeze(0).repeat(B, 1)
    for b in range(B):
        for spans in injection_maps[b].values():
            for start, end in spans:
                end = min(end, seq_len)
                if start < end:
                    pos[b, start:end] = 0
    return pos


def wire_place_at_node_spans(dest, b: int, rows, injection_map, seq_len: int) -> None:
    """Accumulate per-node rows into ``dest[b]`` at each node's token spans, in place.

    ``dest`` is ``[B, seq, D]`` (pre-zeroed), ``rows`` is ``[N, D]``. Spans are disjoint
    (``build_injection_map`` dedups longest-first) so the accumulate is really an
    assignment; written as ``x = x + row`` to stay out-of-place w.r.t. autograd.

    Mirrors the placement loop inside ``GraphAugmentedLLM.build_pe_signal`` and is
    deliberately NOT shared with it: the WIRE arch is kept self-contained so it cannot
    perturb the additive family. Used only by :class:`WireGraphLLM`.
    """
    for node_idx, spans in injection_map.items():
        for start, end in spans:
            end = min(end, seq_len)
            if start < end:
                dest[b, start:end] = dest[b, start:end] + rows[node_idx].to(dest.dtype)


def node_adjacency(g, device, k_hops: int = 1, symmetrize: bool = True,
                   use_edges: bool = True, permutation=None) -> torch.Tensor:
    """Boolean ``[N, N]`` node adjacency — True where two nodes may attend.

    Built from ``edge_index`` with self-loops (a node always sees itself and its own
    repeated mentions), optional symmetrization, and ``(A+I)^k`` reachability for
    ``k_hops > 1``. ``use_edges=False`` ⇒ edgeless ablation: only self-loops remain.
    Shared by ``GraphMaskLLM`` and ``LearnableGraphMaskLLM`` (mirrors the undirected
    adjacency the GNN/GT consume).

    ``permutation``: eval-time node relabelling (``models.utils.Permutation``), applied to
    ``edge_index`` with EXACTLY the same call ``GraphTransformer.forward`` makes on the
    Ψ side. It must be threaded here as well as into ``pe_model``: A and Ψ index the same
    node axis, so permuting only one of them would mask a relabelled Ψ against an
    unrelabelled topology — an inconsistency that is not a permutation of anything.
    """
    N = g.num_nodes
    adj = torch.zeros(N, N, dtype=torch.bool, device=device)
    ei = getattr(g, "edge_index", None)
    if use_edges and ei is not None and ei.numel() > 0:
        ei = ei.to(device)
        if permutation is not None:
            ei = permutation.apply(ei, N, device=device)
        adj[ei[0], ei[1]] = True
        if symmetrize:
            adj = adj | adj.t()
    adj.fill_diagonal_(True)  # self-loops: same node (and its repeats) always visible
    if use_edges and k_hops > 1:
        reach = adj.clone()
        f_adj = adj.float()
        power = adj.clone()
        for _ in range(k_hops - 1):
            power = (power.float() @ f_adj) > 0
            reach = reach | power
        adj = reach
    return adj


class GraphMaskLLM(PreTrainedModel):  # ty:ignore[unsupported-base]
    """LLM whose attention mask mirrors the scene-graph adjacency.

    No positional encoding, no GNN, no learnable graph params. The only change vs
    the plain LLM baseline is the attention mask: two node-token positions may attend
    iff their graph nodes are adjacent within ``k_hops`` (or identical). Non-node
    tokens keep normal causal attention.

    Per-forward additive bias ``self._struct_bias`` [B, 1, seq, seq] (0 = allowed,
    finfo.min = blocked) is added to the model's causal/sliding mask inside each
    attention layer. Cached decode steps skip the fold (generated tokens are non-graph).

    Args:
        llm: base causal LLM.
        k_hops: hops within which node tokens may attend (1 = direct edges only).
        symmetrize: OR adjacency with transpose (scene graph is already undirected).
        use_edges: False = edgeless ablation; only self-loops remain, all cross-node
            attention is blocked.
    """

    def __init__(self, llm: nn.Module, k_hops: int = 1, symmetrize: bool = True,
                 use_edges: bool = True, buggy_causal_fold: bool = False,
                 layer_scope: str = "all"):
        # Wrapper is not a registered HF architecture or MoE class; force "eager" so
        # PreTrainedModel doesn't reject SDPA/flash or expert-impl validation.
        config = copy.copy(llm.config)
        config._attn_implementation = "eager"  # ty: ignore[invalid-assignment]
        config._experts_implementation = "eager"  # ty: ignore[invalid-assignment]
        super().__init__(config)
        self.llm = llm
        self._mask_k_hops = int(k_hops)
        if self._mask_k_hops < 1:
            raise ValueError(f"k_hops must be >= 1, got {k_hops}")
        self._mask_symmetrize = bool(symmetrize)
        self._mask_use_edges = bool(use_edges)
        # "all" = every self-attn layer (default, historical behavior); "dense" = only Gemma
        # full_attention (global) layers — the non-learnable control matching the learnable mask.
        if layer_scope not in MASK_LAYER_SCOPES:
            raise ValueError(f"layer_scope must be one of {MASK_LAYER_SCOPES}, got {layer_scope!r}")
        self._mask_layer_scope = layer_scope
        # A/B ablation knob: reproduce the pre-fix causal/sliding-mask leak (see
        # _graph_mask_attention_forward). Default False = correct masking.
        self._mask_buggy_fold = bool(buggy_causal_fold)
        # Per-forward additive attention bias [B, 1, seq, seq]; read by the patched
        # attention layers, set in forward / inference, disarmed afterwards.
        self._struct_bias: torch.Tensor | None = None
        self._decode_bias_row: torch.Tensor | None = None
        self._install_graph_mask()

    def structural_parameters(self) -> list[nn.Parameter]:
        """Parameter-free architecture: no graph params (only the LLM/LoRA train)."""
        return []

    def _decoder_layers(self):
        """Return the decoder layer list (Gemma text-only: ``model.layers``; multimodal: ``get_decoder().layers``)."""
        base = getattr(self.llm, "model", None)
        if base is not None and hasattr(base, "layers"):
            return base.layers
        return self.llm.get_decoder().layers

    def _install_graph_mask(self) -> None:
        """Route every self-attention layer through ``prism_graph_mask``.

        Captures each layer's original attn impl/fn, registers the mask function to
        mirror it (so HF builds the right causal/sliding mask for the delegated fn),
        and sets a back-reference. Instance-level to survive PEFT.
        """
        layers = self._decoder_layers()
        if len(layers) == 0:
            return
        first_attn = layers[0].self_attn
        mod = importlib.import_module(type(first_attn).__module__)
        if not hasattr(self, "_graph_mask_orig_attn_impl"):
            impl = first_attn.config._attn_implementation
            self._graph_mask_orig_attn_impl = "eager" if impl == _GRAPH_MASK_IMPL else impl
        attn_fns = mod.ALL_ATTENTION_FUNCTIONS
        if hasattr(attn_fns, "get_interface"):
            orig_attn_fn = attn_fns.get_interface(
                self._graph_mask_orig_attn_impl, mod.eager_attention_forward)
        else:
            orig_attn_fn = (
                mod.eager_attention_forward
                if self._graph_mask_orig_attn_impl == "eager"
                else attn_fns[self._graph_mask_orig_attn_impl]
            )
        # Register our mask impl to mirror the original's so HF builds the right
        # causal/sliding mask for the delegated fn; we ADD the structural bias on top.
        mask_fns = getattr(masking_utils, "ALL_MASK_ATTENTION_FUNCTIONS", None)
        if mask_fns is not None and self._graph_mask_orig_attn_impl in mask_fns._global_mapping:
            masking_utils.AttentionMaskInterface.register(
                _GRAPH_MASK_IMPL, mask_fns._global_mapping[self._graph_mask_orig_attn_impl])

        active_flags = resolve_mask_active_flags(layers, self._mask_layer_scope)
        for layer, active in zip(layers, active_flags):
            attn = layer.self_attn
            # Bypass nn.Module.__setattr__ to avoid registering a submodule cycle (attn→wrapper→llm→attn).
            object.__setattr__(attn, "_graph_mask_model", self)
            attn._graph_mask_orig_attn_fn = orig_attn_fn
            attn._graph_mask_buggy_fold = self._mask_buggy_fold
            attn._graph_mask_active = active
            attn.config._attn_implementation = _GRAPH_MASK_IMPL

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)  # defer to nn.Module first
        except AttributeError:
            return getattr(self.llm, name)

    def _node_adjacency(self, g, device, permutation=None) -> torch.Tensor:
        """Boolean ``[N, N]`` node adjacency (see module-level :func:`node_adjacency`)."""
        return node_adjacency(g, device, k_hops=self._mask_k_hops,
                              symmetrize=self._mask_symmetrize, use_edges=self._mask_use_edges,
                              permutation=permutation)

    def build_structural_mask(self, seq_len, graphs, injection_maps, device, dtype=None,
                              key_injection_maps=None, permutation=None):
        """Additive attention bias ``[B, 1, seq, seq]`` — 0 allowed, ``finfo.min`` blocked.

        ``bias[b,0,i,j] = finfo.min`` iff tokens i and j BOTH belong to graph nodes
        AND those nodes are non-adjacent (within ``k_hops``). Every other entry
        (node↔non-node, non-node↔non-node, same node, adjacent) stays 0. Because it
        is ADDED to the model's causal/sliding mask, blocking only ever removes
        already-causal pairs. Each node-token row keeps BOS (a non-node) and its own
        diagonal, so no row is fully masked (no softmax NaN).

        ``key_injection_maps``: optional separate map for the KEY role. When given,
        bias rows (queries) are wired from ``injection_maps`` and bias columns (keys)
        from ``key_injection_maps`` — the decode-consistency rule of the decode-time
        design note §3 (answer mentions act as keys everywhere, as queries only where
        the assignment is decode-knowable). Default None = same map for both roles.

        ``permutation``: eval-time node relabelling, applied to the adjacency (this class
        has no Ψ, so A is the ONLY thing the permutation can touch). Without it
        ``--permutation-seed`` is a no-op for this architecture.
        """
        if dtype is None:
            dtype = self.llm.get_input_embeddings().weight.dtype
        B = len(injection_maps)
        neg = torch.finfo(dtype).min
        bias = torch.zeros(B, 1, seq_len, seq_len, device=device, dtype=dtype)
        for b in range(B):
            g = graphs[b]
            tok2node_q = tok2node_vector(injection_maps[b], seq_len, device)
            tok2node_k = (tok2node_q if key_injection_maps is None
                          else tok2node_vector(key_injection_maps[b], seq_len, device))
            q_pos = (tok2node_q >= 0).nonzero(as_tuple=True)[0]
            k_pos = (tok2node_k >= 0).nonzero(as_tuple=True)[0]
            if q_pos.numel() == 0 or k_pos.numel() == 0:
                continue
            adj = self._node_adjacency(g, device, permutation=permutation)   # [N, N] bool
            q_nid = tok2node_q[q_pos]                     # node id per query node-token
            k_nid = tok2node_k[k_pos]                     # node id per key node-token
            allowed = adj[q_nid][:, k_nid]                # [Pq, Pk] bool over pairs
            blocked = ~allowed
            if blocked.any():
                bi, bj = blocked.nonzero(as_tuple=True)
                bias[b, 0, q_pos[bi], k_pos[bj]] = neg
        return bias

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        graphs: Batch | None = None,
        injection_maps: list[dict[int, list[tuple[int, int]]]] | None = None,
        key_injection_maps: list[dict[int, list[tuple[int, int]]]] | None = None,
        **kwargs,
    ):
        kwargs.pop("inputs_embeds", None)
        kwargs.pop("input_ids", None)
        # Arm the structural bias for the patched attention layers. No graph (e.g. a
        # non-graph batch) ⇒ plain causal LLM.
        if graphs is not None and injection_maps is not None and input_ids is not None:
            self._struct_bias = self.build_structural_mask(
                input_ids.shape[1], graphs, injection_maps, input_ids.device,
                key_injection_maps=key_injection_maps)
        else:
            self._struct_bias = None
        try:
            return self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )
        finally:
            # Disarm so a later forward can't reuse a stale mask — except under
            # gradient checkpointing, where backward recomputes the attention forwards
            # and must see the same bias (every forward rebuilds it anyway).
            if not getattr(self.llm, "is_gradient_checkpointing", False):
                self._struct_bias = None


class LearnableGraphMaskLLM(PreTrainedModel):  # ty:ignore[unsupported-base]
    """LLM with a LEARNABLE relative-positional attention mask (Gemma-4 only).

    Like ``GraphMaskLLM``, the only change vs the plain LLM is an additive bias on the
    attention logits at node-token positions. Unlike ``GraphMaskLLM`` (parameter-free,
    0 = allowed / finfo.min = blocked), the per-node-pair bias is **learned**::

        M[i,j] = log(α·1 + (1−α)·sim(Ψ_i, Ψ_j))   i,j adjacent (or self-loop)
        M[i,j] = finfo.min                        i,j non-adjacent (hard block)

    where ``Ψ = pe_model(graph)`` ∈ ``[N, D]`` is produced by a standalone Graph
    Transformer (a relative PE), and the ``N×N`` form is induced by the outer product
    ``Ψ Ψᵀ`` — so the parameter count is independent of graph size. Node↔non-node and
    non-node↔non-node token pairs keep normal causal attention (bias 0). Because the
    bias is ADDED to the model's causal/sliding mask, blocking only removes
    already-causal pairs, and each node row keeps its diagonal + BOS (softmax safe).

    Gradient flows loss → bias → sim → Ψ → ``pe_model``, so the GT trains (reported via
    :meth:`structural_parameters`); the LLM trains via LoRA. The mask never touches
    q/k/v, so Gemma-4's KV-sharing is irrelevant.

    Per-forward additive bias ``self._struct_bias`` [B,1,seq,seq] is read by the patched
    attention layers; cached decode steps (length mismatch) fall through to stock attn.

    Args:
        llm: base causal LLM (Gemma-4 12B / 31B).
        pe_model: standalone Graph Transformer; ``forward(graph) → [N, D]``.
        alpha: mix of the constant edge bias (α·1) vs the learned term ((1−α)·sim).
            Must be in ``[0, 1)``: α=1 zeroes the learnable term ⇒ the GT gets no
            gradient (degenerate), so it is rejected.
        layer_scope: ``"dense"`` injects only into Gemma ``full_attention`` (global)
            layers; ``"all"`` injects into every self-attention layer.
        k_hops, symmetrize, use_edges: adjacency ``A`` construction (see
            :func:`node_adjacency`).
        psi_scale: ``"cosine"`` row-normalizes Ψ so sim ∈ [−1,1] (bounded mask); or
            ``"inv_sqrt_d"`` uses raw ``Ψ Ψᵀ / √D`` (attention-style scaling).
        eps: epsilon for the cosine normalization.
        disable_graph_token_rope: identity-RoPE (position_id 0) on the injected spans —
            the mask says *which* nodes may attend, this says the node tokens carry no
            sequence position. Orthogonal to the bias: it changes q/k RoPE, not logits.
            See :meth:`forward` for the train/decode contract.
    """

    def __init__(self, llm: nn.Module, pe_model: nn.Module, alpha: float = 0.7,
                 layer_scope: str = "dense", k_hops: int = 1, symmetrize: bool = True,
                 use_edges: bool = True, psi_scale: str = "cosine", eps: float = 1e-8,
                 buggy_causal_fold: bool = False, disable_graph_token_rope: bool = False):
        # Wrapper is not a registered HF architecture or MoE class; force "eager" so
        # PreTrainedModel doesn't reject SDPA/flash or expert-impl validation.
        config = copy.copy(llm.config)
        config._attn_implementation = "eager"  # ty: ignore[invalid-assignment]
        config._experts_implementation = "eager"  # ty: ignore[invalid-assignment]
        super().__init__(config)
        self.llm = llm

        # Place the GT on the LLM's device so PEFT doesn't leave it on CPU.
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = llm.device
        self.pe_model = pe_model.to(device)

        self._mask_alpha = float(alpha)
        if not 0.0 <= self._mask_alpha < 1.0:
            raise ValueError(
                f"alpha must be in [0, 1); alpha=1 zeroes the learnable term so the GT "
                f"gets no gradient. Got {alpha}.")
        if layer_scope not in MASK_LAYER_SCOPES:
            raise ValueError(f"layer_scope must be one of {MASK_LAYER_SCOPES}, got {layer_scope!r}")
        self._mask_layer_scope = layer_scope
        if psi_scale not in ("cosine", "inv_sqrt_d"):
            raise ValueError(f"psi_scale must be 'cosine' or 'inv_sqrt_d', got {psi_scale!r}")
        self._mask_psi_scale = psi_scale
        self._mask_eps = float(eps)

        self._mask_k_hops = int(k_hops)
        if self._mask_k_hops < 1:
            raise ValueError(f"k_hops must be >= 1, got {k_hops}")
        self._mask_symmetrize = bool(symmetrize)
        self._mask_use_edges = bool(use_edges)
        # A/B ablation knob (shared with GraphMaskLLM): reproduce the pre-fix mask leak.
        self._mask_buggy_fold = bool(buggy_causal_fold)
        # Same attribute name GraphAugmentedLLM uses: inference.py duck-types on it.
        self._disable_graph_token_rope: bool = bool(disable_graph_token_rope)
        if self._mask_psi_scale == "cosine" and not self._mask_use_edges:
            # Edgeless ⇒ adjacency is self-loops only; cosine self-similarity is the constant
            # 1, so every allowed entry is constant ⇒ the whole bias is constant ⇒ the GT gets
            # ZERO gradient (silent dead training). Fail loud like the alpha=1 guard.
            raise ValueError(
                "psi_scale='cosine' with use_edges=False is degenerate: only self-loops remain "
                "and cosine self-similarity is constant 1, so the GT receives zero gradient. "
                "Use psi_scale='inv_sqrt_d' or keep use_edges=True.")

        # Per-forward additive attention bias [B, 1, seq, seq]; carries grad to the GT.
        self._struct_bias: torch.Tensor | None = None
        self._decode_bias_row: torch.Tensor | None = None
        self._install_graph_mask()

    def structural_parameters(self) -> list[nn.Parameter]:
        """Graph-side parameters for the boosted-LR group: the standalone GT."""
        return list(self.pe_model.parameters())

    def graph_token_position_ids(
        self,
        injection_maps: list[dict[int, list[tuple[int, int]]]],
        seq_len: int,
        device,
    ) -> torch.Tensor:
        """position_ids ([B, seq]) with graph-token spans set to 0 (identity RoPE).

        Same delegate as :meth:`GraphAugmentedLLM.graph_token_position_ids`; kept as a
        method so ``inference.py`` can call it on either architecture.
        """
        return graph_token_position_ids(injection_maps, seq_len, device)

    def _decoder_layers(self):
        """Return the LLM's decoder layer list (Gemma-4: ``llm.get_decoder().layers``)."""
        base = getattr(self.llm, "model", None)
        if base is not None and hasattr(base, "layers"):
            return base.layers
        return self.llm.get_decoder().layers

    def _install_graph_mask(self) -> None:
        """Route self-attention layers through ``prism_graph_mask``, gated by ``layer_scope``.

        The decoder layers share one ``config`` object, so the attn-impl name can't be
        flipped per layer. We therefore route EVERY layer through the mask impl but flag
        only the in-scope layers ``_graph_mask_active``; out-of-scope (sliding-window)
        layers delegate to stock attention. Reuses the shared ``prism_graph_mask`` fn and
        mask registration. Instance-level to survive PEFT.
        """
        layers = self._decoder_layers()
        if len(layers) == 0:
            return
        first_attn = layers[0].self_attn
        mod = importlib.import_module(type(first_attn).__module__)
        if not hasattr(self, "_graph_mask_orig_attn_impl"):
            impl = first_attn.config._attn_implementation
            self._graph_mask_orig_attn_impl = "eager" if impl == _GRAPH_MASK_IMPL else impl
        attn_fns = mod.ALL_ATTENTION_FUNCTIONS
        if hasattr(attn_fns, "get_interface"):
            orig_attn_fn = attn_fns.get_interface(
                self._graph_mask_orig_attn_impl, mod.eager_attention_forward)
        else:
            orig_attn_fn = (
                mod.eager_attention_forward
                if self._graph_mask_orig_attn_impl == "eager"
                else attn_fns[self._graph_mask_orig_attn_impl]
            )
        # Register our mask impl to mirror the original's so HF builds the right
        # causal/sliding mask for the delegated fn; we ADD the structural bias on top.
        mask_fns = getattr(masking_utils, "ALL_MASK_ATTENTION_FUNCTIONS", None)
        if mask_fns is not None and self._graph_mask_orig_attn_impl in mask_fns._global_mapping:
            masking_utils.AttentionMaskInterface.register(
                _GRAPH_MASK_IMPL, mask_fns._global_mapping[self._graph_mask_orig_attn_impl])

        active_flags = resolve_mask_active_flags(layers, self._mask_layer_scope)
        for layer, _active in zip(layers, active_flags):
            attn = layer.self_attn
            # Bypass nn.Module.__setattr__ to avoid registering a submodule cycle.
            object.__setattr__(attn, "_graph_mask_model", self)
            attn._graph_mask_orig_attn_fn = orig_attn_fn
            # Dense-only scope deactivates sliding-window layers (they delegate untouched).
            attn._graph_mask_active = _active
            attn._graph_mask_buggy_fold = self._mask_buggy_fold
            attn.config._attn_implementation = _GRAPH_MASK_IMPL

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)  # defer to nn.Module first
        except AttributeError:
            return getattr(self.llm, name)

    def _node_adjacency(self, g, device, permutation=None) -> torch.Tensor:
        """Boolean ``[N, N]`` node adjacency (see module-level :func:`node_adjacency`)."""
        return node_adjacency(g, device, k_hops=self._mask_k_hops,
                              symmetrize=self._mask_symmetrize, use_edges=self._mask_use_edges,
                              permutation=permutation)

    def build_structural_mask(self, seq_len, graphs, injection_maps, device, dtype=None,
                              key_injection_maps=None, permutation=None):
        """Additive attention bias ``[B, 1, seq, seq]`` with the learned relative-PE mask.

        For token pairs (i, j) where BOTH tokens map to graph nodes::

            adjacent / self-loop:  bias = log(α + (1−α)·sim(Ψ_i, Ψ_j))  (multiplicative gate)
            non-adjacent:          bias = finfo.min   (hard block)

        Every other pair (node↔non-node, non-node↔non-node) stays 0. The allowed
        entries carry gradient through ``sim → Ψ → pe_model``; blocked entries and
        non-node entries are constants. Each node row keeps its diagonal (self-loop)
        and BOS, so no row is fully masked (softmax safe).

        ``key_injection_maps``: optional separate map for the KEY role (queries wired
        from ``injection_maps``, keys from ``key_injection_maps`` — decode-consistency
        rule, decode-time design note §3). Default None = same map for both roles.

        ``permutation``: eval-time node relabelling (``--permutation-seed``), threaded to
        BOTH halves of the mask — ``pe_model`` (Ψ) and ``node_adjacency`` (A) — exactly as
        ``GraphAugmentedLLM.build_pe_signal`` / ``WireGraphLLM.build_wire_signal`` thread
        it to their single Ψ call. Both are required: the mask is
        ``A ⊙ log-gate(ΨΨᵀ)``, so permuting one factor and not the other is not a
        permutation of the mask at all, and permuting neither (the pre-fix behaviour)
        makes the flag a silent no-op for this architecture.
        """
        if dtype is None:
            dtype = self.llm.get_input_embeddings().weight.dtype
        B = len(injection_maps)
        neg = torch.finfo(dtype).min
        bias = torch.zeros(B, 1, seq_len, seq_len, device=device, dtype=dtype)
        alpha = self._mask_alpha
        for b in range(B):
            g = graphs[b]
            tok2node_q = tok2node_vector(injection_maps[b], seq_len, device)
            tok2node_k = (tok2node_q if key_injection_maps is None
                          else tok2node_vector(key_injection_maps[b], seq_len, device))
            q_pos = (tok2node_q >= 0).nonzero(as_tuple=True)[0]
            k_pos = (tok2node_k >= 0).nonzero(as_tuple=True)[0]
            if q_pos.numel() == 0 or k_pos.numel() == 0:
                continue
            adj = self._node_adjacency(g, device, permutation=permutation)   # [N, N] bool
            psi = self.pe_model(g, permutation=permutation).float()   # [N, D] (GT runs fp32)
            if self._mask_psi_scale == "cosine":
                psi = psi / psi.norm(dim=-1, keepdim=True).clamp_min(self._mask_eps)
                sim = psi @ psi.t()                            # cosine ∈ [−1, 1]
            else:  # inv_sqrt_d
                sim = (psi @ psi.t()) / (psi.shape[-1] ** 0.5)
            gate = (alpha + (1.0 - alpha) * sim).clamp_min(self._mask_eps)
            node_M = gate.log()                                # [N, N] log-gate (fp32)
            q_nid = tok2node_q[q_pos]                          # node id per query node-token
            k_nid = tok2node_k[k_pos]                          # node id per key node-token
            allowed = adj[q_nid][:, k_nid]                     # [Pq, Pk] bool
            block = node_M[q_nid][:, k_nid].to(dtype)          # [Pq, Pk] log-gate values
            neg_t = torch.tensor(neg, dtype=dtype, device=device)
            block = torch.where(allowed, block, neg_t)         # hard-block non-edges
            bias[b, 0, q_pos.unsqueeze(1), k_pos.unsqueeze(0)] = block
        return bias

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        graphs: Batch | None = None,
        injection_maps: list[dict[int, list[tuple[int, int]]]] | None = None,
        key_injection_maps: list[dict[int, list[tuple[int, int]]]] | None = None,
        **kwargs,
    ):
        """Arm the structural bias, then run the LLM.

        ``_disable_graph_token_rope`` additionally zeroes the RoPE positions of the
        QUERY-role spans (``injection_maps``) — the same rule
        :class:`GraphAugmentedLLM` follows, so identity-RoPE always covers exactly the
        positions that carry the graph channel under ``data.injection_scope``, and no
        second definition of "graph token" exists. Consequences, worth knowing before
        reading a result:

        - ``prompt_only``: every zeroed position is a prompt position, and
          ``inference.py`` passes the identical prompt map to ``generate`` ⇒ exact
          train/decode parity.
        - ``decode_consistent``: prompt spans as above; answer mentions are zeroed at
          their single knowability position, which ``MaskDecodeInjector`` reproduces
          step-by-step at decode ⇒ parity there too.
        - ``full_sequence``: answer-side name spans are zeroed in training but get
          natural positions at decode (HF advances position_ids per step and generated
          tokens are not known in advance) — a train/decode mismatch that is inherent
          to that scope, not to this flag.
        """
        kwargs.pop("inputs_embeds", None)
        kwargs.pop("input_ids", None)
        # Arm the learned structural bias for the patched attention layers. No graph
        # (e.g. a non-graph batch) ⇒ plain causal LLM.
        if graphs is not None and injection_maps is not None and input_ids is not None:
            self._struct_bias = self.build_structural_mask(
                input_ids.shape[1], graphs, injection_maps, input_ids.device,
                key_injection_maps=key_injection_maps)
        else:
            self._struct_bias = None
        # Identity-RoPE the injected spans when requested, unless the caller already
        # supplied position_ids (mirrors GraphAugmentedLLM.forward).
        if (self._disable_graph_token_rope and injection_maps is not None
                and input_ids is not None and kwargs.get("position_ids") is None):
            kwargs["position_ids"] = graph_token_position_ids(
                injection_maps, input_ids.shape[1], input_ids.device)
        try:
            return self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )
        finally:
            # Disarm unless under gradient checkpointing (backward recomputes the
            # attention forwards and must see the same bias; every forward rebuilds it).
            if not getattr(self.llm, "is_gradient_checkpointing", False):
                self._struct_bias = None


class GraphAugmentedLLM(PreTrainedModel):  # ty:ignore[unsupported-base]
    """Graph-Augmented LLM: injects graph PE Ψ post-RoPE into q/k/v at every layer.

    Receives pre-computed injection maps; pe_model: ``forward(data) → Tensor[n, d_model]``.

    Injection scheme — ``RoPE(X) + Ψ`` at every layer via the ``"prism_pe"`` custom
    attention impl; Ψ added to the already-rotated q/k/v so it is unrotated in the score::

        q = RoPE(W_q · h) + W_q · Ψ
        k = RoPE(W_k · h) + W_k · Ψ
        v =      W_v · h  + W_v · Ψ

    Ψ projected through each layer's own (LoRA-adapted) q/k/v_proj. Architecture-agnostic
    (Gemma-4 text-only and multimodal). ``self._pe_signal`` [B, seq, hidden]; injection skipped on
    cached decode steps (seq mismatch).

    Args:
        llm: base causal LLM
        pe_model: R-PEARL or GraphTransformer; ``forward(data) → [n, d_model]``
        d_model: PE width
        eps: normalization epsilon
    """

    def __init__(self, llm: nn.Module, pe_model: nn.Module,
                 d_model: int, eps: float = 1e-8, pe_gain_init: float = 1.0,
                 disable_graph_token_rope: bool = False, use_pe_norm: bool = True,
                 pe_node_features: str = "random"):
        # Wrapper is not a registered HF architecture or MoE class; force "eager"
        # on both so PreTrainedModel doesn't reject SDPA/flash or expert-impl validation.
        config = copy.copy(llm.config)
        config._attn_implementation = "eager" # ty: ignore[invalid-assignment]
        config._experts_implementation = "eager" # ty: ignore[invalid-assignment]
        super().__init__(config)
        self.llm = llm

        # Place pe_model and pe_proj on the LLM's device so PEFT doesn't leave them on CPU.
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = llm.device
        self.pe_model = pe_model.to(device)
        # Linear projection from d_model to LLM hidden size (no bias, no norm — gate below).
        self.pe_proj = nn.Linear(
            d_model, llm.config.get_text_config().hidden_size, device=device)
        # Learnable gate g = tanh(pe_gain) ∈ (-1,1) controlling injection strength.
        # pe_gain_init=0.0 → cold-start (Ψ off at init, no grad to structural path until gate moves).
        self.pe_gain = nn.Parameter(torch.tensor(float(pe_gain_init), device=device))
        # RMSNorm on projected Ψ, weight initialized to the LLM's mean token-embedding RMS
        # so Ψ enters at text scale. The norm sets the scale; pe_gain sets the ramp.
        if use_pe_norm:
            H = llm.config.get_text_config().hidden_size
            self.pe_norm = nn.RMSNorm(H, device=device)
            with torch.no_grad():
                emb = llm.get_input_embeddings().weight
                r_text = (emb.norm(dim=-1).float().mean() / (H ** 0.5)).item()
                if not (r_text > 0):  # guard meta/empty/degenerate embeddings
                    r_text = 1.0
                self.pe_norm.weight.fill_(r_text)
        else:
            self.pe_norm = None

        # When True, graph-token spans get position_id 0 (identity RoPE); Ψ is the sole position.
        self._disable_graph_token_rope: bool = bool(disable_graph_token_rope)

        # "random": GNN samples its own probes (data.x ignored).
        # "word_embeddings": mean word-embedding of each node's name tokens fed as data.x.
        if pe_node_features not in ("random", "word_embeddings"):
            raise ValueError(
                f"pe_node_features must be 'random' or 'word_embeddings', got {pe_node_features!r}"
            )
        self._pe_node_features: str = pe_node_features

        # Per-forward graph signal Ψ ([B, seq, hidden]); read by the patched
        # attention forwards and set by _augment_embeddings / inference.
        self._pe_signal: torch.Tensor | None = None
        # Whether Ψ also enters the value/content path (v += W_v·Ψ).
        self._pe_inject_value: bool = True
        self._install_pe_injection()

    def structural_parameters(self) -> list[nn.Parameter]:
        """Graph-side parameters eligible for the boosted LR group: the graph encoder,
        the Ψ→hidden projection, and the injection gate. ``pe_norm`` is intentionally
        excluded here so it stays at base LR (it is grad-enabled by the trainer separately).
        """
        return (
            list(self.pe_model.parameters())
            + list(self.pe_proj.parameters())
            + [self.pe_gain]
        )

    def _decoder_layers(self):
        """Return the decoder layer list (Gemma text-only: ``model.layers``; multimodal: ``get_decoder().layers``)."""
        base = getattr(self.llm, "model", None)
        if base is not None and hasattr(base, "layers"):
            return base.layers
        return self.llm.get_decoder().layers

    def _install_pe_injection(self) -> None:
        """Route every self-attention layer through ``prism_pe`` by swapping the attention fn.

        Captures each layer's original attn impl/fn, sets a wrapper back-reference
        (so ``prism_pe`` can read ``self._pe_signal``), and points
        ``config._attn_implementation`` at ``prism_pe``. Instance-level to survive PEFT.
        """
        layers = self._decoder_layers()
        if len(layers) == 0:
            return
        first_attn = layers[0].self_attn
        mod = importlib.import_module(type(first_attn).__module__)
        # Ψ=0 at non-graph tokens, so W·Ψ=0 only if projections are bias-free.
        # A bias would perturb all positions. Fail loud if a future base model adds one.
        for _name in ("q_proj", "k_proj", "v_proj"):
            _proj = getattr(first_attn, _name, None)
            if _proj is not None and getattr(_proj, "bias", None) is not None:
                raise ValueError(
                    f"{type(first_attn).__name__}.{_name} has a bias; prism_pe injection "
                    "assumes bias-free attention projections so non-graph tokens stay "
                    "untouched (Ψ=0 ⇒ W·Ψ=0). This base model needs a bias-aware injection."
                )
        # Capture the impl ONCE before mutating: configs are shared across layers,
        # so a post-mutation read would already see "prism_pe". Persisted for idempotent re-install.
        if not hasattr(self, "_prism_orig_attn_impl"):
            impl = first_attn.config._attn_implementation
            self._prism_orig_attn_impl = "eager" if impl == _PRISM_PE_IMPL else impl
        # Resolve original attention fn: ≥5.12 uses get_interface(); older subscripts the registry.
        attn_fns = mod.ALL_ATTENTION_FUNCTIONS
        if hasattr(attn_fns, "get_interface"):
            orig_attn_fn = attn_fns.get_interface(
                self._prism_orig_attn_impl, mod.eager_attention_forward)
        else:
            orig_attn_fn = (
                mod.eager_attention_forward
                if self._prism_orig_attn_impl == "eager"
                else attn_fns[self._prism_orig_attn_impl]
            )
        # Register prism_pe in the mask registry mirroring the original impl's mask, so
        # HF builds the right causal/sliding mask for the delegated fn (not None).
        mask_fns = getattr(masking_utils, "ALL_MASK_ATTENTION_FUNCTIONS", None)
        if mask_fns is not None and self._prism_orig_attn_impl in mask_fns._global_mapping:
            masking_utils.AttentionMaskInterface.register(
                _PRISM_PE_IMPL, mask_fns._global_mapping[self._prism_orig_attn_impl])

        for layer in layers:
            attn = layer.self_attn
            # Bypass nn.Module.__setattr__ to avoid registering a submodule cycle (attn→wrapper→llm→attn).
            object.__setattr__(attn, "_prism_pe_model", self)
            attn._prism_orig_attn_fn = orig_attn_fn
            attn.config._attn_implementation = _PRISM_PE_IMPL

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)  # defer to nn.Module first
        except AttributeError:
            return getattr(self.llm, name)

    def build_pe_signal(
        self,
        embeddings: torch.Tensor,
        graphs: Batch,
        injection_maps: list[dict[int, list[tuple[int, int]]]],
        permutation=None,
    ) -> torch.Tensor:
        """Assemble the graph signal Ψ ([B, seq, hidden]) placed at node-token spans.

        Ψ is consumed *inside* the patched attention layers (added post-RoPE), so
        here we only build the placed, scaled, gated signal — no rotation.

        Args:
            embeddings: [B, seq, hidden] token embeddings (for magnitude scaling).
            graphs: PyG Batch (or list) of graphs, one per batch element.
            injection_maps: Per-batch-element dict mapping node index to a list of
                (start, end) token spans where that node's PE should be placed.
            permutation: Optional node permutation passed to ``pe_model`` (used by
                eval-time permutation-equivariance checks; ``None`` in training).
        """
        B, seq_len, hidden = embeddings.shape
        psi = torch.zeros(B, seq_len, hidden, device=embeddings.device, dtype=embeddings.dtype)
        for b in range(B):
            g = graphs[b]
            if self._pe_node_features == "word_embeddings":
                # Per-node feature = mean word-embedding over mention spans. Fail loud if any node has no span.
                N = g.num_nodes
                feats = torch.zeros(N, hidden, device=embeddings.device, dtype=torch.float32)
                covered = [False] * N
                for node_idx, spans in injection_maps[b].items():
                    rows = [embeddings[b, start:min(end, seq_len)]
                            for start, end in spans if start < min(end, seq_len)]
                    if rows:
                        feats[node_idx] = torch.cat(rows, dim=0).float().mean(dim=0)
                        covered[node_idx] = True
                missing = [i for i in range(N) if not covered[i]]
                if missing:
                    names = getattr(g, "node_names", None)
                    raise ValueError(
                        "pe_node_features='word_embeddings' requires every graph node to be "
                        f"mentioned in the prompt, but these have no span: "
                        f"{[(i, names[i] if names else '?') for i in missing]}"
                    )
                # Detach from embedding table; grad still flows through GNN/pe_proj/pe_gain.
                g.x = feats.detach()
            pe = self.pe_proj(self.pe_model(g, permutation=permutation))  # [n, hidden_size]
            if self.pe_norm is not None:
                # Norm weight is fp32; under bf16 autocast pe is bf16. Compute the norm in
                # fp32 (matches the codebase's fp32-norm convention; tiny [n, hidden] tensor)
                # and cast back, so the fused rms_norm kernel dispatches instead of falling
                # back on an input/weight dtype mismatch.
                pe = self.pe_norm(pe.float()).to(pe.dtype)
            # Gate: g = tanh(pe_gain) ∈ (-1, 1); norm sets scale, gate sets ramp.
            pe = pe * torch.tanh(self.pe_gain)
            for node_idx, spans in injection_maps[b].items():
                for start, end in spans:
                    end = min(end, seq_len)
                    if start < end:
                        psi[b, start:end] = psi[b, start:end] + pe[node_idx].to(psi.dtype)
        return psi

    def graph_token_position_ids(
        self,
        injection_maps: list[dict[int, list[tuple[int, int]]]],
        seq_len: int,
        device,
    ) -> torch.Tensor:
        """position_ids ([B, seq]) with graph-token spans set to 0 (identity RoPE).

        Thin delegate to the module-level :func:`graph_token_position_ids`; kept as a
        method because ``inference.py`` calls it on the model (duck-typed across the
        architectures that honour the flag). Used only when
        ``_disable_graph_token_rope`` is set.
        """
        return graph_token_position_ids(injection_maps, seq_len, device)

    def _augment_embeddings(
        self,
        input_ids: torch.Tensor,
        graphs: Batch,
        injection_maps: list[dict[int, list[tuple[int, int]]]],
    ) -> torch.Tensor:
        """Return plain token embeddings X and arm ``self._pe_signal`` = Ψ for attention layers.

        Ψ is added post-RoPE inside each attention layer (not to the residual stream here).
        """
        embeddings = (
            self.llm.get_input_embeddings()(input_ids)
                .clone()
                .to(input_ids.device)
        )  # [B, seq_len, d]
        self._pe_signal = self.build_pe_signal(embeddings, graphs, injection_maps)
        return embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        graphs: Batch | None = None,
        injection_maps: list[dict[int, list[tuple[int, int]]]] | None = None,
        **kwargs,
    ):
        embeddings = self._augment_embeddings(input_ids, graphs, injection_maps)

        kwargs.pop("inputs_embeds", None)
        kwargs.pop("input_ids", None)

        # Identity-RoPE the graph-token spans (position_id 0) when requested, unless the
        # caller already supplied position_ids.
        if (self._disable_graph_token_rope and injection_maps is not None
                and kwargs.get("position_ids") is None):
            kwargs["position_ids"] = self.graph_token_position_ids(
                injection_maps, embeddings.shape[1], embeddings.device)

        try:
            return self.llm(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )
        finally:
            # Disarm Ψ after forward; keep armed under gradient checkpointing
            # (backward recomputes attention forwards and must see the same signal).
            if not getattr(self.llm, "is_gradient_checkpointing", False):
                self._pe_signal = None


class WireGraphLLM(PreTrainedModel):  # ty:ignore[unsupported-base]
    """WIRE: graph PE injected as a **rotation** of q/k, not as an added signal.

    Implements Wave-Induced Rotary Encodings (Reid et al., *Rotary Position Encodings
    for Graphs*, ICML 2026) on Gemma-4. Gemma applies its own RoPE to q/k *before*
    dispatching to the attention interface, so this composes a SECOND rotation there::

        q_i → R(θ_i)·RoPE(p_i)·q_i        θ_i[n] = ω_n · r_i
        k_j → R(θ_j)·RoPE(p_j)·k_j        r_i    = Ψ_i  (the GT's node feature)

    The hook (:func:`_wire_attention_forward`) runs after ``apply_rotary_pos_emb``
    (``modeling_gemma4.py:1251`` for q, ``:1267`` for k) and after the KV-cache write
    (``:1274``), reached by the attention dispatch at ``:1282``.

    Same-plane 2-D rotations commute, so the score becomes
    ``qᵀ R(p_j − p_i + r_j − r_i) k``: the text phase and the graph phase simply add.
    ``v`` is untouched (WIRE is a q/k mechanism). Because ``Ψ = 0`` at non-graph
    tokens, ``θ = 0`` there ⇒ ``cos = 1, sin = 0`` ⇒ the rotation is **exactly** the
    identity, not an approximation — a stronger invariant than the additive family's,
    which needs bias-free projections to get it.

    **TWO MODES. Read this first — it decides which half of this docstring applies.**
    ``vanilla`` (config ``gnn.wire_vanilla``) selects between the paper's algorithm and
    an expectation-motivated variant of it. ``vanilla=True`` is the **DEFAULT**::

        vanilla=True  (DEFAULT)   ω_ℓ is ONE LEARNABLE [P, m] table per layer, and that
                                  is the whole mechanism. This is Alg. 1 / §3.3: "one
                                  does not sample and average over an ensemble of random
                                  WIRE transformations, but instead takes one learnable
                                  instantiation." No ε, no σ, no Monte-Carlo reading —
                                  the expectation those exist to serve is NEVER TAKEN.
        vanilla=False             the expectation arm: ω_ℓ = σ_ℓ·ε_ℓ with ε ~ N(0, I)
                                  FROZEN and σ_ℓ a learnable per-layer scalar, so ω is a
                                  genuine Gaussian sample (Theorem 3's hypothesis) and σ
                                  is exactly the ω in qᵀk(1 − ω²R(i,j)/2). Unchanged
                                  from before the switch existed.

    What is IDENTICAL in both modes, and must stay so:

    - The rotation itself (:func:`_wire_rotate`, Eq. 11): ``cos ⊙ z + sin ⊙ Pz``. The
      **sin term is RETAINED in both modes.** Dropping it is an expectation-motivated
      approximation (``E[sin] = 0`` under a symmetric ω distribution) that appears
      NOWHERE in the paper or its reference implementation, and it would destroy both
      orthogonality and Eq. 3. It must never appear on either path.
    - ω is ``[P, m]`` — one table per LAYER, shared by every head — in both modes. See
      the frequencies section below for why that is the paper's own sanctioned choice
      and the only one compatible with Gemma's GQA.
    - The angle clamp stays ACTIVE in both modes. Its *justification* changes; see the
      clamp section.
    - ``r`` is Ψ from ``pe_model`` in both modes. Neither mode makes ``r`` the paper's
      Alg. 1 feature — see the next section.

    **What the theory does and does not give you** (this section describes the
    ``vanilla=False`` arm; in vanilla mode no expectation is taken at all, so none of
    it is claimed). Theorem 3 states, for ``ω_i ~ N(0, ω²I)``::

        E[(RoPE(r_i)q_i)ᵀ RoPE(r_j)k_j] = q_iᵀk_j (1 − ω²R(i,j)/2) + O(ω⁴)

    Two independent steps hide in that, and only the first one transfers.

    Step 1 — the Johnson–Lindenstrauss averaging that kills the odd ``sin`` term and
    leaves ``qᵀk(1 − ω²‖r_i−r_j‖²/2) + O(ω⁴)`` — holds for ANY feature ``r``. That
    step is what this class delivers.

    Step 2 — the algebraic identity ``‖r_i−r_j‖² = R(i,j)`` (effective resistance) —
    holds ONLY for ``r_i = [u_k[i]/√λ_k]_k`` over all nontrivial Laplacian modes. The
    ``r`` fed in here is not that feature: ``pe_model`` is a ``GraphTransformer``
    whose PE branch is a mean-pooled R-PEARL readout (a degree-``k_pe``=2 TAGConv
    polynomial with LeakyReLU, LayerNorm and biases), followed by sparse-attention
    blocks. Even on the most generous linear-filter reading of that stack you get
    ``‖Ψ_i−Ψ_j‖² = Σ_k g(λ_k)²(u_k[i] − u_k[j])²`` for a LEARNED spectral response
    ``g`` — and ``g = 1/√λ`` is neither a polynomial nor bounded as ``λ → 0``, so a
    degree-2 filter cannot reach it even in principle; the nonlinearities and
    attention blocks put the actual Ψ further away still.

    Note that a single layer's score is ONE draw of ω, not the expectation: the
    identity above is an expectation over ω, and a single draw carries the full
    oscillatory ``cos``/``sin`` terms of the paper's Eq. 8.

    **Alg. 1's ``r`` and Theorem 3's ``r`` are DIFFERENT features — do not conflate
    them.** Alg. 1 (the algorithm ``vanilla=True`` implements) uses the plain Laplacian
    eigenvector coordinates ``r_i = [u_k[i]]_{k=0}^{m-1}``. Theorem 3's
    resistance-normalized ``[u_k[i]/√λ_k]_k`` appears ONLY in that theorem, as the
    feature under which ``‖r_i−r_j‖² = R(i,j)``. So running vanilla mode is NOT a claim
    about effective resistance even if ``r`` were exact eigenvectors — and here it is
    neither: ``r`` is the learned Ψ described above. Both modes inherit that gap
    equally; ``vanilla`` changes how ω is parameterized, never what ``r`` is.

    **Linear-attention compatibility is RETAINED here.** WIRE's headline architectural
    claim (§3.3) is that the structural downweighting arises without instantiating the
    attention matrix. Because the shared modulation is applied as a ROTATION of q and k
    — not as a score-level ``[B, S, S]`` (or ``[N, N]``) multiplicative gate — no
    quadratic object is built at any point, and the property survives. Sharing ω across
    heads changes only how many times the angles are computed, never where they are
    applied. If this class is ever changed to apply the modulation to logits instead,
    that property is forfeited.

    **Frequencies.** Every in-scope layer holds ONE ``[P, m]`` table, shared by every
    head. What that table IS differs by mode::

        vanilla=True    _wire_omega[ℓ] : nn.Parameter [P, m]   ω_ℓ = s_ℓ · Ω_ℓ
                        The paper's ω, learned directly. Initialised per
                        ``vanilla_omega_init`` (default "zero" — the reference
                        implementation's own default, giving θ = 0 and hence the EXACT
                        identity at step 0). ``_wire_eps`` / ``_wire_sigma`` stay EMPTY.

        vanilla=False   _wire_eps[ℓ]   : buffer    [P, m]   frozen ε ~ N(0, I)
                        _wire_sigma[ℓ] : Parameter scalar   learnable σ_ℓ
                        ω_ℓ = σ_ℓ · s_ℓ · ε_ℓ. ``_wire_omega`` stays EMPTY.

    (``s_ℓ`` is the angle clamp below, exactly 1.0 in the normal regime.)
    ``freeze_sigma=True`` pins the learnable term in BOTH modes — σ in the expectation
    arm, Ω in vanilla — and drops it from :meth:`structural_parameters`.

    Zero init is trainable *because the sin term is retained*: at ``θ = 0``,
    ``∂cos(θ)/∂ω = −sin(θ)·r = 0`` but ``∂sin(θ)/∂ω = cos(θ)·r = r ≠ 0``. An
    implementation that dropped sin as an ``E[sin] = 0`` approximation would make the
    paper's own default init a dead starting point.

    **One table per LAYER, not per head — in both modes, and this is load-bearing.**
    §3.1 budgets "dm/2 parameters per transformer layer" and explicitly sanctions
    sharing: "For additional savings, one can share WIRE weights between layers or
    heads." The reference implementation applies its single per-layer table to the full
    ``[b, n, d]`` q and k *before* the head split, which under ordinary MHA hands each
    head a distinct contiguous slice — but q and k have the SAME head count there, so
    head ``h``'s query and head ``h``'s key still share a slice and Eq. 3 holds.

    That construction does not survive Gemma's **GQA**, and this is the reason head
    sharing is not merely a saving here. Gemma-4-31B has 32 query heads against 16 kv
    heads on sliding layers and 4 on global. Query head ``h`` is dotted against kv head
    ``h // n_rep``. Per-query-head frequencies would rotate them by different ω, so
    ``qᵀR(θ_j)ᵀR(θ_i)k`` would no longer collapse to a function of ``r_j − r_i`` and the
    relative-only property Eq. 3 asserts would break outright; keys are also stored and
    cached once per kv head, so they cannot physically carry a per-query-head rotation
    in the first place. Sharing one table across all heads makes q and k rotate
    identically by construction, which is what
    ``tests/test_wire_injection.test_relative_only_holds_under_gqa`` asserts end to end.
    Each head still keeps its own ``qᵀk`` entirely — no content is averaged across heads.

    **Monte-Carlo sample count differs by layer type — and the DEFAULT inverts the
    asymmetry.** THIS SECTION APPLIES TO ``vanilla=False`` ONLY: in vanilla mode no
    expectation is taken, so ``P`` is a parameter count, not a sample count, and none of
    the variance reasoning below is claimed. With ω shared across heads, the Step-1
    averaging comes from the ``P``
    rotary planes in the shared table (Eq. 8's sum over ``k``), plus independence
    across depth; the variance falls as ``1/P``. ``P`` is :func:`wire_rope_planes`,
    and it is NOT uniform across layers. gemma-4-31B::

        layer type   head_dim   rotate_nope_planes=False   =True
        global        512        64  (factor 0.25)          256
        sliding       256       128  (no factor)            128

    At the DEFAULT (``rotate_nope_planes=False``) the global layers are the WORSE
    estimator — 64 samples against the sliding layers' 128 — and flipping the flag to
    True reverses that to 256 vs 128. Which layers this actually bites depends on
    ``layer_scope``: at the default ``"dense"`` only global layers carry WIRE, so
    every active layer runs at 64 planes; ``"all"`` is where the 2× cross-layer-type
    asymmetry is live. For any model other than 31B read the number off
    :func:`wire_rope_planes`, not off this table.

    Be precise about what more planes buy: they tighten the estimate a layer's scores
    are built from, but **no single score equals the expectation** — one score is one
    draw and carries the full oscillatory cos/sin terms of Eq. 8.

    **Scaling in the number of nodes.** Nothing here allocates or parameterizes in
    ``N``: ω is ``[P, m]`` with ``P ≤ head_dim/2`` (a function of head width and GT
    width only), Ψ is ``[N, m]`` (linear), and the rotation is ``O(d)`` per token via
    Eq. 11. No ``N×N`` or ``S×S`` object is built at any point, which is both WIRE's
    §3.1 parameters-independent-of-graph-size property and what keeps it
    linear-attention compatible (§3.3), unlike bias-style RPEs. The one quantity that
    CAN drift with graph size is the angle magnitude ``θ = ω·Ψ_i``: cos/sin are
    periodic, so if ``‖Ψ‖`` grew with ``N`` the angles would wrap and the
    leading-order Theorem 3 reading would silently fail — presenting exactly as
    "worked at N=30, noise at N=100". :meth:`build_wire_signal` therefore MEASURES
    ``σ·max‖Ψ_i‖`` and ``σ·max‖Ψ_i−Ψ_j‖`` on every forward and clamps the latter to
    ``max_angle`` (below) rather than assuming the regime holds.

    **The clamp, and the one hazard it creates.** ``max_angle`` is the ONLY angle
    threshold; a former ``hard_max_angle`` that RAISED has been deleted, because no
    training run may be killed by the angle guard. :meth:`layer_scale_factor` returns
    ``min(1.0, max_angle / (scale_ℓ·span))`` and :meth:`layer_omega` applies it
    unconditionally. ``scale_ℓ`` is :meth:`layer_omega_scale` — ``|σ_ℓ|`` in the
    expectation arm, ``max_n‖Ω_ℓ[n]‖₂/√m`` in vanilla, normalised so ``max_angle``
    means the same thing on both sides of the switch. Inside the bound the factor is
    EXACTLY 1.0, so the identity case goes down the same code path, and the effective
    angle satisfies the bound by construction on every forward.

    **THE CLAMP IS KEPT IN VANILLA MODE. This is a DELIBERATE, VISIBLE departure from
    the paper — the one thing ``vanilla=True`` does not strip.** Neither the paper nor
    the reference implementation bounds ``θ`` in any way. It is kept anyway, for four
    reasons, and the reader is entitled to disagree with the trade:

    1. The standing requirement is mode-independent: no run may be killed by the angle
       guard, and the guard may never be violated. The clamp is the ONLY thing bounding
       angle magnitude. Removing it in vanilla mode would leave nothing at all.
    2. It costs nothing in fidelity in the healthy regime. ``layer_scale_factor``
       returns *exactly* ``1.0`` inside the bound — not approximately — so on any run
       that never saturates, vanilla mode is the paper's algorithm bit for bit. It is a
       guard that fires, not a term that is always present.
    3. The paper's implicit bound does not transfer. Its ``r`` is Laplacian eigenvector
       coordinates (``‖u_k‖₂ = 1``, so ``|u_k[i]| ≤ 1``); its default ``init_omega`` is
       ZERO, so θ starts at 0 and grows only as gradients push it. Here ``r`` is Ψ from a
       learned ``GraphTransformer`` whose FINAL ``SparseTransformerBlock`` is built with
       ``normalize=False`` (``gt.py:307``/``:457``) — nothing pins ‖Ψ‖, which is MEASURED
       at ≈1.1·√d_model (35.4 at d_model=1024) and is free to drift during training.
    4. Without a Taylor expansion the *justification* changes but the hazard does not.
       Vanilla mode has no expansion to protect, so the "Taylor guard" reading is void —
       but cos/sin are still 2π-periodic, so once ``|θ| ≫ π`` the map from Ψ-distance to
       attention modulation stops being monotone and ``∂/∂ω`` oscillates. That is an
       optimisation failure rather than a theory failure, and it presents identically:
       "worked at N=30, noise at N=100".

    What now bounds the angles in vanilla mode is therefore exactly what bounds them in
    the expectation arm — ``max_angle``, enforced by construction — and NOT any property
    of the paper's algorithm. The saturation hazard below applies unchanged to Ω.

    The cost is a **biased gradient once the clamp saturates.** The factor is
    detached, so ``ω = σ·s·ε`` has ``dω/dσ = s·ε ≠ 0`` and autograd dutifully reports
    a non-zero ``dL/dσ``. But in the saturated regime ``σ·s`` is pinned at
    ``max_angle/span`` *regardless of σ*: the loss is bit-identical across a 1000×
    range of raw σ, so the TRUE ``dL/dσ`` is zero and the reported value is an
    artifact of treating ``s`` as constant. Consequence: **σ can drift upward
    indefinitely with no corrective signal and no effect on the model, while training
    looks perfectly healthy.** The same holds verbatim for Ω in vanilla mode: ``s`` is
    detached there too, so ``ω = s·Ω`` reports a non-zero ``dL/dΩ`` while the applied
    angle is pinned. Nothing detects this automatically. ``wire/sigma_raw_max``
    diverging from ``wire/sigma_eff_max`` in :meth:`wire_telemetry` (equivalently
    ``wire/clamp_engaged == 1``) is the ONLY signal, and a one-shot ``RuntimeWarning``
    fires the first time it happens. A saturated run is NOT an A/B over σ — it is a
    fixed-angle run at ``max_angle``, and must be reported as one.

    **Decode: cached keys are re-rotated.** ``decode="rotate"`` (the default). The KV
    cache is written at ``modeling_gemma4.py:1274``, BEFORE this hook, so it stores
    UNROTATED keys — but ``r`` is a function of POSITION only (:meth:`build_wire_signal`
    reads the prompt injection map once), so re-applying ``R(θ_j)`` to key slot ``j`` at
    every step reconstructs exactly the key the prefill forward scored against. The
    score stays a function of ``r_j − r_i`` alone and Theorem 3's relative-only
    hypothesis is untouched. Generated positions are absent from the injection map ⇒
    ``r = 0`` ⇒ their keys and the decode query are bit-identical to stock — the same
    prompt-only decode wiring ``GraphAugmentedLLM`` and the mask family already use at
    eval, so this is consistent with ``injection_scope`` ``prompt_only`` /
    ``exclude_supervised`` exactly and with ``full_sequence`` up to the response-token
    asymmetry the whole repo shares. ``decode_consistent`` is NOT wired for WIRE
    (rejected in ``train_v3._validate_config`` and again in ``inference.py``).
    ``decode="skip"`` falls through to stock attention on cached steps — a labeled
    diagnostic in which WIRE is OFF for every generated token, never a result. The
    prompt forward (training, teacher-forced eval) is unaffected either way.

    The one scope that cannot decode is ``layer_scope="all"``: sliding-window layers
    crop their KV cache past the window, so key slot 0 stops being absolute position 0.
    :meth:`assert_decode_supported` raises for it BEFORE generating, naming the knob.

    **Config switches.** All of them, and nothing else, fork behavior. Hydra keys are
    ``gnn.wire_*`` (``experiments/base_config.yaml``), wired in
    ``architectures.build_model``::

        arg (config key)                   default    effect / what the other side does
        vanilla (wire_vanilla)             True       THE MODE SWITCH (see the top of
                                                      this docstring). True = the paper's
                                                      algorithm: one learnable ω table per
                                                      layer, no ε/σ, no expectation
                                                      machinery on the active path. False
                                                      = the ω = σ·ε expectation arm.
        vanilla_omega_init                 "zero"     ω init when vanilla=True. "zero"
          (wire_vanilla_omega_init)                   (reference default) ⇒ θ=0 ⇒ exact
                                                      identity at step 0. Also "uniform",
                                                      "exponential" — the reference
                                                      implementation's own three values.
                                                      Inert when vanilla=False.
        layer_scope (wire_layer_scope)     "dense"    WIRE on Gemma full_attention
                                                      (global) layers only. "all" = every
                                                      layer incl. sliding; "dense_top_half"
                                                      = deeper half of globals;
                                                      "dense_first" = shallowest global.
        sigma_init (wire_sigma_init)       0.01       initial σ_ℓ, every in-scope layer.
                                                      Larger ⇒ larger angle; scale as
                                                      1/√d_model (see Args). INERT when
                                                      vanilla=True (ω init is chosen by
                                                      vanilla_omega_init instead).
        freeze_sigma (wire_freeze_sigma)   False      the learnable frequency term trains,
                                                      inside the structural LR group. True
                                                      pins it and drops it from
                                                      structural_parameters() (still
                                                      checkpointed). Pins σ when
                                                      vanilla=False (= literal-Theorem-3
                                                      arm) and Ω when vanilla=True.
        omega_seed (wire_omega_seed)       0          seed of the frozen ε draw
                                                      (vanilla=False) or of the
                                                      uniform/exponential ω draw
                                                      (vanilla=True). Any other int ⇒ a
                                                      different draw. Inert under
                                                      vanilla_omega_init="zero", which is
                                                      deterministic.
        rotate_nope_planes                 False      rotate only the text-RoPE planes.
          (wire_rotate_nope_planes)                   True also rotates the NoPE planes
                                                      (4× more planes on 31B globals; see
                                                      the sample-count table above).
        max_angle (wire_max_angle)         1.0        clamp bound on σ_eff·max‖Ψ_i−Ψ_j‖.
                                                      Larger ⇒ weaker clamp and a walk out
                                                      of the small-angle regime; smaller ⇒
                                                      saturates sooner (see the hazard).
        pe_gain_init                       1.0        gate tanh(pe_gain), init ≈ 0.76.
                                                      0.0 ⇒ θ=0 ⇒ exact identity cold start.
        decode (wire_decode)               "rotate"   re-rotate cached prompt keys each
                                                      decode step. "skip" = WIRE off at
                                                      decode, diagnostic only. "error" is
                                                      a legacy value normalised to
                                                      "rotate" (warned, not silent).
        pe_node_features                   "random"   GT samples its own probes. Anything
                                                      else is rejected in __init__.

    **Known coverage gaps** (stated so they are not mistaken for verified behavior):

    - ``dense`` / ``dense_top_half`` / ``dense_first`` are exercised but NOT
      DIFFERENTIATED by the current test fixture: the 4-layer ``gemma4`` used in
      ``tests/test_wire_*.py`` has exactly one ``full_attention`` layer (transformers
      forces the last layer global), so all three scopes resolve to the same single
      layer. Their divergence on a real 31B stack is untested.
    - Multistage init (``loaders.load_pe_weights_into``) carries WIRE state only
      between runs whose WIRE config MATCHES: a checkpoint written under a different
      ``wire_layer_scope`` / ``wire_rotate_nope_planes`` / ``d_model`` raises rather
      than remapping ε across layer sets or plane counts. Carrying from a prior GT
      checkpoint loads Ψ only — the gate, ε and σ cold-start (reported by the loader),
      so such a stage is NOT a WIRE resume. Crossing the ``wire_vanilla`` boundary
      raises as well: ω = σ·ε and a free ω table are different parameterisations, so a
      checkpoint from one mode carries no frequencies the other can read, and silently
      cold-starting them while reporting a resume is exactly the failure this refuses.
    - ``pe_node_features="word_embeddings"`` is unsupported (rejected in ``__init__``
      and again in ``architectures.build_model``).
    - Decode re-rotation covers the DENSE scopes only; ``layer_scope="all"`` raises
      before generating (see above). Decoding under a sliding-window cache would need
      the cropped cache's absolute offset, which is layout-dependent (crop vs ring).
    - ``injection_scope='decode_consistent'`` (generated node mentions carrying the
      graph channel) is not wired for WIRE — it needs the mask family's q/kv split.

    **Deliberately omitted — do not add**: any pooled/accumulated cross-layer estimate
    or ``[N,N]``/``[S,S]`` score-level gate (that is a relative position encoding, the
    class of method WIRE exists to replace, and it forfeits linear-attention
    compatibility); ``pe_proj`` / ``pe_norm`` (rotation angles
    are not in the LLM's hidden space, and renormalizing ``r`` would silently rescale
    the ``‖r_i−r_j‖`` the theorem is stated in — ``max_angle`` is the principled scale
    control instead); ``v`` injection (WIRE is q/k only); ``pe_node_features=
    "word_embeddings"`` (not wired, fails loud); **dropping the sin term of Eq. 8** on
    either path (an ``E[sin] = 0`` approximation that exists nowhere in the paper or its
    reference implementation, destroys orthogonality and Eq. 3, and would make the
    paper's own ``init_omega="zero"`` default a gradient-dead start); per-query-head or
    per-KV-head ω (breaks Eq. 3 under GQA, or needs a granularity the cached keys cannot
    carry — see the frequencies section).

    Args:
        llm: base causal LLM (Gemma-4).
        pe_model: Ψ producer; ``forward(data) → [N, d_model]``.
        d_model: Ψ width ``m`` — also the ω table's input width.
        layer_scope: which layers carry WIRE (see :func:`resolve_mask_active_flags`).
        sigma_init: initial value of every learnable ``σ_ℓ``. Only ``σ·‖r_i−r_j‖``
            matters, and the guarantee is leading-order in it (``O(ω⁴)`` error), so this
            must START inside the small-angle regime. **Scale it with 1/√d_model**:
            ``‖Ψ_i‖`` is MEASURED at ≈ 1.1·√d_model (8.9 at d_model=64, 35.4 at
            d_model=1024) and flat in N, so ``max‖Ψ_i−Ψ_j‖`` is ~√d_model and σ=0.05
            would already put the angle at ~2.0 rad at d_model=1024 — past
            ``max_angle``. The default is calibrated for d_model=1024 (σ=0.01 ⇒ ~0.4
            rad). Note this scaling is EMPIRICAL, not structural: the GT's final
            SparseTransformerBlock is built with ``normalize=False``, so no LayerNorm
            pins the output magnitude and training may move it. ``max_angle`` is the
            actual guarantee.
        freeze_sigma: pin σ (literal-Theorem-3 A/B arm). Default False = σ trains.
        omega_seed: seed for the frozen ε draw (recorded in the run config).
        rotate_nope_planes: also rotate the NoPE planes of ``proportional``-rope
            layers. Default False — see :func:`wire_rope_planes`.
        max_angle: bound on the effective ``σ·max‖Ψ_i−Ψ_j‖``. Enforced by an exact,
            UNCONDITIONAL clamp (:meth:`layer_scale_factor`), so it holds by
            construction on every forward and can never fail a run — at the cost of
            the biased-gradient hazard documented above.
        pe_gain_init: gate ``tanh(pe_gain)``; 0.0 ⇒ θ=0 ⇒ exact identity cold start.
        decode: ``"error"`` (default) or ``"skip"`` — see :func:`_wire_attention_forward`.
        pe_node_features: must be ``"random"``; anything else is rejected here.
        vanilla: **DEFAULT True.** Run strictly the paper's algorithm — ω is one
            learnable ``[P, m]`` table per layer (§3.3's "one learnable instantiation")
            and every expectation/Monte-Carlo-motivated construct is off the active
            path. False restores the ``ω = σ·ε`` expectation arm unchanged. The angle
            clamp is the one guard retained in BOTH modes; see the clamp section for the
            reasoning and for what it costs. The two modes' checkpoints are NOT
            interchangeable (different frequency stores) and ``loaders`` refuses to mix
            them rather than cold-starting ω silently.
        vanilla_omega_init: ω initialisation when ``vanilla=True``; one of
            :data:`WIRE_VANILLA_OMEGA_INITS`. Default ``"zero"``, which is the reference
            implementation's own default and gives an exact-identity cold start.
    """

    def __init__(self, llm: nn.Module, pe_model: nn.Module, d_model: int,
                 layer_scope: str = "dense", sigma_init: float = 0.01,
                 freeze_sigma: bool = False, omega_seed: int = 0,
                 rotate_nope_planes: bool = False, max_angle: float = 1.0,
                 pe_gain_init: float = 1.0,
                 decode: str = "rotate", pe_node_features: str = "random",
                 vanilla: bool = True, vanilla_omega_init: str = "zero"):
        # Wrapper is not a registered HF architecture or MoE class; force "eager" so
        # PreTrainedModel doesn't reject SDPA/flash or expert-impl validation.
        config = copy.copy(llm.config)
        config._attn_implementation = "eager"  # ty: ignore[invalid-assignment]
        config._experts_implementation = "eager"  # ty: ignore[invalid-assignment]
        super().__init__(config)
        self.llm = llm

        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = llm.device
        self.pe_model = pe_model.to(device)

        if layer_scope not in MASK_LAYER_SCOPES:
            raise ValueError(f"layer_scope must be one of {MASK_LAYER_SCOPES}, got {layer_scope!r}")
        if decode in WIRE_DECODE_LEGACY:
            # Back-compat: "error" existed ONLY as a guard while decode-time key rotation
            # was unimplemented. It is implemented now, so checkpoints (and configs) that
            # recorded the old value evaluate under "rotate" instead of failing. Announced,
            # never silent — the recorded provenance value and the active one differ.
            new = WIRE_DECODE_LEGACY[decode]
            warnings.warn(
                f"wire_decode={decode!r} is a legacy value from before decode-time key "
                f"rotation existed; running as {new!r}. Set gnn.wire_decode={new!r} "
                "explicitly (or 'skip' for the WIRE-off-at-decode diagnostic).",
                DeprecationWarning, stacklevel=2)
            decode = new
        if decode not in WIRE_DECODE_MODES:
            raise ValueError(f"decode must be one of {WIRE_DECODE_MODES}, got {decode!r}")
        if pe_node_features != "random":
            raise ValueError(
                "WireGraphLLM currently supports only pe_node_features='random' "
                f"(word-embedding feature prep is not wired). Got {pe_node_features!r}.")
        if not sigma_init > 0:
            raise ValueError(f"sigma_init must be > 0, got {sigma_init}")
        if not max_angle > 0:
            raise ValueError(f"max_angle must be > 0, got {max_angle}")
        if vanilla_omega_init not in WIRE_VANILLA_OMEGA_INITS:
            raise ValueError(
                f"vanilla_omega_init must be one of {WIRE_VANILLA_OMEGA_INITS}, "
                f"got {vanilla_omega_init!r}")

        self._wire_vanilla = bool(vanilla)
        self._wire_vanilla_omega_init = vanilla_omega_init
        self._wire_d_model = int(d_model)
        self._wire_layer_scope = layer_scope
        self._wire_sigma_init = float(sigma_init)
        self._wire_freeze_sigma = bool(freeze_sigma)
        self._wire_omega_seed = int(omega_seed)
        self._wire_rotate_nope = bool(rotate_nope_planes)
        self._wire_max_angle = float(max_angle)
        self._wire_decode = decode
        self._pe_node_features = pe_node_features
        # max‖Ψ_i−Ψ_j‖ from the current forward; combined with σ to measure the angle.
        self._wire_psi_span: float | None = None
        self._wire_warned_angle: bool = False

        # Gate g = tanh(pe_gain) ∈ (-1,1) on the ANGLE; 0.0 ⇒ exact identity cold start.
        self.pe_gain = nn.Parameter(torch.tensor(float(pe_gain_init), device=device))

        # Per-forward state; read by the patched attention fns, disarmed after forward.
        self._wire_signal: torch.Tensor | None = None       # [B, seq, m]
        # Measured small-angle diagnostics (see build_wire_signal): the pair term is
        # the Theorem 3 expansion parameter; the row term catches Ψ-norm drift with N.
        self._wire_measured_angle: float | None = None      # σ·max‖Ψ_i−Ψ_j‖
        self._wire_measured_row_angle: float | None = None  # σ·max‖Ψ_i‖
        self._wire_modeling_module = None
        # Per-layer (cos, sin) of the PROMPT positions' angles, reused across the decode
        # steps of one generate() call (see decode_cos_sin). Cleared whenever the signal
        # is (re)armed, so it can never outlive the r it was built from.
        self._wire_decode_cos_sin: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        # Frequency stores. EXACTLY ONE of these is populated, selected by `vanilla`:
        #
        #   vanilla=True (DEFAULT, the paper's algorithm) — `_wire_omega` holds ONE
        #     LEARNABLE [P, m] table per in-scope layer, which IS the paper's ω. No ε,
        #     no σ: §3.3 takes "one learnable instantiation" rather than sampling and
        #     averaging, so there is nothing to reparameterise. `_wire_eps` and
        #     `_wire_sigma` stay EMPTY.
        #
        #   vanilla=False (the expectation arm) — ε: frozen unit-Gaussian directions,
        #     one [P, m] buffer per in-scope layer; σ: learnable per-layer scale;
        #     ω = σ_ℓ · ε_ℓ. `_wire_omega` stays EMPTY.
        #
        # All three are constructed unconditionally (empty in the unused mode) so the
        # checkpoint key set is IDENTICAL in both modes — a key written in one mode and
        # absent in the other is exactly the silent-corruption case the loader guards.
        self._wire_eps = nn.Module()
        self._wire_sigma = nn.ParameterDict()
        self._wire_omega = nn.ParameterDict()
        self._install_wire()

    def layer_scale_factor(self, layer_idx: int) -> float:
        """Detached scale ``s_ℓ ∈ (0, 1]`` pinning ``|σ_ℓ|·s_ℓ·span ≤ max_angle``.

        Applied UNCONDITIONALLY by :meth:`layer_omega` — ``min(1.0, max_angle/measured)``
        is exactly 1.0 inside the bound, so the identity case is the SAME code path — and
        the Taylor-expansion premise therefore holds by construction on every forward
        rather than being checked and reported. No run can be killed by it.

        ``span`` is the pairwise Ψ spread measured by :meth:`build_wire_signal`. When it
        is unset or zero (no graph armed, or a single-node / all-equal Ψ) the angle is
        identically zero and the factor is 1.0.

        HAZARD — the factor is detached, so ``ω = σ·s·ε`` gives ``dω/dσ = s·ε ≠ 0`` and
        autograd reports a non-zero ``dL/dσ`` even when saturated. It is BIASED: once
        ``s = max_angle/(|σ|·span)``, the product ``σ·s`` no longer depends on σ at all,
        the loss is bit-identical across a 1000× range of raw σ, and the true ``dL/dσ``
        is zero. σ can then drift upward forever with no corrective signal and no effect
        on the model. The only detection is ``wire/sigma_raw_max`` diverging from
        ``wire/sigma_eff_max`` in :meth:`wire_telemetry`.
        """
        span = self._wire_psi_span
        if span is None or span <= 0:
            return 1.0
        measured = self.layer_omega_scale(layer_idx) * span
        if measured <= 0.0:
            return 1.0
        return min(1.0, self._wire_max_angle / measured)

    def layer_omega_scale(self, layer_idx: int) -> float:
        """Detached per-layer scalar summarising ω's magnitude, in ONE convention.

        This is the quantity the clamp and every ``wire/sigma_*`` telemetry key are
        stated in, so both modes are directly comparable and ``max_angle`` means the
        same thing on either side of the switch::

            vanilla=False   |σ_ℓ|                     (ω = σ_ℓ·ε_ℓ, ε ~ N(0, I))
            vanilla=True    max_n‖ω_ℓ[n]‖₂ / √m       (ω_ℓ is the table itself)

        The ``√m`` in the vanilla form is what makes the two agree rather than differ by
        a factor of √m: under the reparameterisation ‖ε_n‖₂ ≈ √m, so
        ``max_n‖ω_n‖₂/√m ≈ |σ|``. Both are therefore a scale on the SAME
        ``σ·span`` proxy the clamp has always used, not a change of units.

        Note this is a proxy, not a bound: the true angle spread is
        ``max_n|ω_n·(Ψ_i−Ψ_j)| ≤ max_n‖ω_n‖₂·span``, which is ``√m`` times this. That
        was already true of ``|σ_ℓ|`` before vanilla mode existed and is preserved
        deliberately — changing it would silently re-scale every existing
        ``wire_max_angle`` in the shipped configs.
        """
        if self._wire_vanilla:
            om = self._wire_omega[str(layer_idx)].detach()
            return float(om.norm(dim=-1).max()) * (self._wire_d_model ** -0.5)
        return float(self._wire_sigma[str(layer_idx)].detach().abs())

    def layer_omega(self, layer_idx: int, device=None):
        """The clamped frequencies actually used. Shape ``[P, m]`` in BOTH modes::

            vanilla=True    ω_ℓ = s_ℓ · Ω_ℓ        (Ω_ℓ the learnable table)
            vanilla=False   ω_ℓ = σ_ℓ · s_ℓ · ε_ℓ

        ONE table for the whole layer, shared by every head, so the angles are computed
        once per layer and broadcast (see :func:`_wire_rotate`). Head sharing is not
        this repo's invention: §3.1 lists it explicitly ("one can share WIRE weights
        between layers or heads"), and it is the only granularity under which Eq. 3
        stays exact given Gemma's GQA — see the class docstring.

        ``s_ℓ`` is the detached, unconditional factor from :meth:`layer_scale_factor`;
        it is EXACTLY 1.0 inside the bound, so in the normal regime vanilla mode
        returns the learned table unmodified. The multiply is out-of-place, so this is
        autograd-safe; gradient reaches ``Ω`` / ``σ`` (scaled by ``s_ℓ``) — but read the
        bias warning on :meth:`layer_scale_factor` before trusting it under saturation.
        """
        if self._wire_vanilla:
            omega = self._wire_omega[str(layer_idx)]
            if device is not None:
                omega = omega.to(device)
            return self.layer_scale_factor(layer_idx) * omega
        eps = getattr(self._wire_eps, str(layer_idx))
        sigma = self._wire_sigma[str(layer_idx)]
        if device is not None:
            eps = eps.to(device)
            sigma = sigma.to(device)
        return (sigma * self.layer_scale_factor(layer_idx)) * eps

    def decode_cos_sin(self, layer_idx: int, omega, head_dim: int, device):
        """Cached ``(cos, sin)`` of the PROMPT positions' WIRE angles for ``layer_idx``.

        A position's graph phase is fixed the moment its node span is known
        (:meth:`build_wire_signal` reads the prompt injection map ONCE), so these are
        constant for the whole ``generate`` call. Computing them once instead of per
        decode step removes the ``[S, m] × [m, P]`` angle product from the inner loop —
        at S=2k, m=1024, P=64 that is ~131 MFLOP per active layer per token.

        Cached ONLY under ``no_grad``: a stored cos/sin would freeze ``ω`` (hence ``σ``)
        out of the autograd graph, so a grad-enabled cached step recomputes exactly
        rather than silently detaching. Decode always runs under ``no_grad``.
        """
        cached = self._wire_decode_cos_sin.get(layer_idx)
        if cached is not None and cached[0].device == device:
            return cached
        cos, sin = wire_cos_sin(self._wire_signal.to(device), omega, head_dim)
        if not torch.is_grad_enabled():
            self._wire_decode_cos_sin[layer_idx] = (cos, sin)
        return cos, sin

    def assert_decode_supported(self) -> None:
        """Fail loud BEFORE generating if a WIRE-active layer cannot be decoded.

        Sliding-window layers crop their KV cache once the sequence passes the window, so
        key slot 0 stops being absolute position 0 and the per-position graph phase can no
        longer be aligned (see :func:`_wire_attention_forward`). Every ``dense*`` scope
        excludes them by construction (:func:`resolve_mask_active_flags`), so this can only
        fire for ``gnn.wire_layer_scope='all'`` — and it fires here, not halfway through a
        rollout. ``wire_decode='skip'`` is exempt: WIRE is off at decode there anyway.
        """
        if self._wire_decode == "skip":
            return
        layers = self._decoder_layers()
        sliding = [i for i in self.active_layer_indices()
                   if getattr(layers[i].self_attn, "is_sliding", False)]
        if sliding:
            raise NotImplementedError(
                f"WireGraphLLM: layer_scope={self._wire_layer_scope!r} put WIRE on "
                f"sliding-window layers {sliding}, whose KV cache is cropped past the "
                "window — key slot 0 stops being absolute position 0 and the graph phase "
                "cannot be aligned to the cached keys. Train/evaluate with a dense "
                "gnn.wire_layer_scope ('dense', 'dense_top_half', 'dense_first'), or set "
                "gnn.wire_decode='skip' to run WIRE-off-at-decode as a labeled diagnostic.")

    def wire_telemetry(self) -> dict:
        """Per-forward scalars for the training logs (``GradientDebugCallback``, which
        merges this dict when the wrapped model exposes it).

        This is the ONLY instrument for the clamp-saturation hazard
        (:meth:`layer_scale_factor`): a clamped σ grows without bound while the
        effective σ stays pinned, so the parameter silently stops meaning anything
        while loss and gradient norms look healthy.

        Read it as::

            wire/sigma_raw_max  >  wire/sigma_eff_max   ⇒ SATURATED (σ is dead weight)
            wire/clamp_engaged  == 1                    ⇒ same, as a boolean
            wire/scale_min      <  1.0                  ⇒ same, as the worst factor
            wire/angle_raw_max                          angle that WOULD have applied
            wire/angle_eff_max  ≤ wire/max_angle        angle actually applied

        Per-layer ``wire/sigma_raw/L{i}`` and ``wire/sigma_eff/L{i}`` make the
        saturation attributable to specific layers. With no graph armed (``span is
        None``) only ``wire/psi_span`` (NaN), ``wire/max_angle`` and ``wire/vanilla``
        are returned.

        The key names say "sigma" in BOTH modes: they carry
        :meth:`layer_omega_scale`, which is ``|σ_ℓ|`` in the expectation arm and
        ``max_n‖ω_ℓ[n]‖₂/√m`` in vanilla — one convention, so a log is comparable
        across the switch. ``wire/vanilla`` records which mode produced the row, so a
        run can never be read under the wrong one.
        """
        span = self._wire_psi_span
        out = {
            "wire/psi_span": float(span) if span is not None else float("nan"),
            "wire/max_angle": self._wire_max_angle,
            "wire/vanilla": int(self._wire_vanilla),
        }
        if span is None:
            return out
        raw, eff, scales, clamped = [], [], [], 0
        for li in self.active_layer_indices():
            s = self.layer_scale_factor(li)
            sig = self.layer_omega_scale(li)
            raw.append(sig)
            eff.append(sig * s)
            scales.append(s)
            clamped += int(s < 1.0)
            out[f"wire/sigma_raw/L{li}"] = sig
            out[f"wire/sigma_eff/L{li}"] = sig * s
        if raw:
            out.update({
                "wire/sigma_raw_max": max(raw),
                "wire/sigma_raw_mean": sum(raw) / len(raw),
                "wire/sigma_eff_max": max(eff),
                "wire/sigma_eff_mean": sum(eff) / len(eff),
                "wire/scale_min": min(scales),
                "wire/angle_raw_max": max(raw) * span,      # what it WOULD have been
                "wire/angle_eff_max": max(eff) * span,      # what was actually applied
                "wire/clamp_engaged": int(clamped > 0),
                "wire/clamped_layers": clamped,
            })
        return out

    # ------------------------------------------------------------------ wiring

    def structural_parameters(self) -> list[nn.Parameter]:
        """Graph-side params for the boosted-LR group: the Ψ producer, the gate, and the
        layer's learnable frequency term **only when learnable** (under ``freeze_sigma``
        it carries ``requires_grad=False`` and must not enter an optimizer group).

        The frequency term is the ω table itself in vanilla mode (§3.3's "one learnable
        instantiation") and the per-layer scalar ``σ_ℓ`` in the expectation arm. ``ε`` is
        never here — it is a frozen persistent buffer, not a parameter."""
        params = list(self.pe_model.parameters()) + [self.pe_gain]
        if not self._wire_freeze_sigma:
            store = self._wire_omega if self._wire_vanilla else self._wire_sigma
            params += list(store.values())
        return params

    def _decoder_layers(self):
        """Return the decoder layer list (Gemma text-only: ``model.layers``; multimodal: ``get_decoder().layers``)."""
        base = getattr(self.llm, "model", None)
        if base is not None and hasattr(base, "layers"):
            return base.layers
        return self.llm.get_decoder().layers

    def _install_wire(self) -> None:
        """Route every layer through ``prism_wire``, flagging only the in-scope ones.

        The decoder layers share one ``config`` object so the impl name cannot be
        flipped per layer; out-of-scope layers carry ``_wire_active=False`` and
        delegate untouched (same pattern as ``LearnableGraphMaskLLM``).

        Note the additive family's bias-free-projection guard is deliberately NOT
        replicated: a rotation never projects Ψ through ``q_proj``/``k_proj``, so a
        projection bias cannot leak the graph channel onto non-graph tokens. ``r=0 ⇒
        θ=0 ⇒ identity`` holds regardless of the projections.
        """
        layers = self._decoder_layers()
        if len(layers) == 0:
            return
        mod, orig_attn_fn = _wire_resolve_orig_attn_fn(self, layers[0].self_attn)
        self._wire_modeling_module = mod

        active_flags = resolve_mask_active_flags(layers, self._wire_layer_scope)
        gen = torch.Generator(device="cpu").manual_seed(self._wire_omega_seed)
        device = self.pe_gain.device
        for idx, (layer, active) in enumerate(zip(layers, active_flags)):
            attn = layer.self_attn
            # KV-shared layers reuse an upstream layer's keys, which already carry THAT
            # layer's ω phase; rotating q here with a different ω would break the
            # relative-only property, so they are inactive entirely rather than q-only.
            # (Dead on gemma-4-31B: num_kv_shared_layers=0.)
            if getattr(attn, "is_kv_shared_layer", False):
                active = False
            object.__setattr__(attn, "_wire_model", self)
            attn._wire_orig_attn_fn = orig_attn_fn
            attn._wire_active = bool(active)
            attn.config._attn_implementation = _WIRE_IMPL
            if not active:
                continue
            planes = wire_rope_planes(attn, self._wire_rotate_nope)
            if planes <= 0:
                raise ValueError(
                    f"layer {idx}: wire_rope_planes computed {planes} rotatable planes "
                    "(partial_rotary_factor too small for this head_dim) — WIRE would "
                    "be a no-op there. Set rotate_nope_planes=True or exclude the layer.")
            if self._wire_vanilla:
                # PAPER FORM (§3.3): ω is ONE learnable table per layer, full stop.
                # There is no ε and no σ to reparameterise — the expectation the σ·ε
                # split exists to serve is never taken here.
                self._wire_omega[str(idx)] = nn.Parameter(
                    self._init_vanilla_omega(planes, gen).to(device),
                    requires_grad=not self._wire_freeze_sigma)
                continue
            # Reparameterized frequencies: ω = σ_ℓ · ε with ε ~ N(0, I) FROZEN and σ_ℓ
            # a learnable per-layer scalar. ω is then a genuine Gaussian sample (the
            # direction is fixed by the draw, Theorem 3's hypothesis) while the learned
            # quantity is exactly the σ that appears in qᵀk(1 − σ²R(i,j)/2). ε is a
            # persistent buffer so it is checkpointed rather than regenerated.
            eps = torch.randn(planes, self._wire_d_model, generator=gen)
            self._wire_eps.register_buffer(str(idx), eps.to(device), persistent=True)
            self._wire_sigma[str(idx)] = nn.Parameter(
                torch.tensor(self._wire_sigma_init, device=device),
                requires_grad=not self._wire_freeze_sigma)

    def _init_vanilla_omega(self, planes: int, gen) -> torch.Tensor:
        """Draw one ``[P, m]`` vanilla-mode ω table per :data:`WIRE_VANILLA_OMEGA_INITS`.

        Reproduces the reference implementation's ``init_omega_matrix``
        (cederikhoefs/Graph-RoPE, ``graphgps/layer/graphrope.py``) literally, including
        its default (``"zero"``) and the two quirks of its ``"exponential"`` branch
        documented on :data:`WIRE_VANILLA_OMEGA_INITS`. ``gen`` is the ``omega_seed``
        generator, so a seeded run is reproducible for the two random strategies; it is
        unused (and the seed inert) for ``"zero"``.
        """
        m = self._wire_d_model
        if self._wire_vanilla_omega_init == "zero":
            return torch.zeros(planes, m)
        if self._wire_vanilla_omega_init == "uniform":
            # torch's default nn.Linear init, which the reference's "uniform" branch
            # leaves in place: U(-1/sqrt(in_features), 1/sqrt(in_features)).
            bound = m ** -0.5
            return torch.empty(planes, m).uniform_(-bound, bound, generator=gen)
        # "exponential": rand(P, m) / 10000^(2i/P), i the plane index.
        rand_freqs = torch.rand(planes, m, generator=gen)
        i = torch.arange(planes, dtype=torch.float32).unsqueeze(1)
        return rand_freqs / (10000.0 ** (2.0 * i / planes))

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)  # defer to nn.Module first
        except AttributeError:
            return getattr(self.llm, name)

    # ------------------------------------------------------------------- signal

    def active_layer_indices(self) -> list[int]:
        """Decoder-layer indices carrying WIRE, in depth order.

        Read off whichever frequency store this mode populated (``_wire_omega`` in
        vanilla, ``_wire_sigma`` in the expectation arm); the other is empty.
        """
        store = self._wire_omega if self._wire_vanilla else self._wire_sigma
        return sorted(int(k) for k in store.keys())

    def build_wire_signal(self, graphs, injection_maps, seq_len: int, device,
                          permutation=None) -> torch.Tensor:
        """Assemble ``r`` ``[B, seq, m]`` — the gated Ψ placed at node-token spans.

        ``permutation`` is threaded to ``pe_model`` exactly as ``build_pe_signal`` does,
        for the eval-time permutation-equivariance check (Lemma 1: WIRE is equivariant
        to node reordering up to eigenvector sign flips / degenerate-subspace rotations).

        Ψ is gated (``·tanh(pe_gain)``) BEFORE the measurements below, so every angle
        quoted here is the one actually applied.

        Measures the small-angle premise per forward, on the ``[N, m]`` node rows (not
        per token — the check is O(N²) on a ~100-row matrix, not on the sequence):

        - ``σ·max‖Ψ_i − Ψ_j‖`` → ``_wire_psi_span`` / ``_wire_measured_angle``, with
          ``σ`` read via :meth:`layer_omega_scale` so both modes report one convention.
          In the expectation arm this is the Theorem 3 expansion parameter (the
          correction is ``ω²‖Ψ_i−Ψ_j‖²/2`` with ``O(ω⁴)`` error). In VANILLA mode there
          is no expansion to be the parameter OF — the same quantity is measured and
          clamped purely as a periodicity guard on the angle (see the class docstring's
          clamp section). Either way it is the ONLY quantity ``max_angle`` clamps.
        - ``σ·max‖Ψ_i‖`` → ``_wire_measured_row_angle``. Diagnostic only, never
          clamped and not surfaced in :meth:`wire_telemetry`. Recorded because cos/sin
          are periodic: if ``‖Ψ‖`` grew with graph size the angles would wrap and the
          leading-order reading would fail silently at large N.

        Scope of the clamp, stated exactly: ``span`` is the pairwise spread over NODE
        rows, so it bounds node↔node phase differences. Non-graph tokens carry ``r=0``,
        so a node↔non-node pair sees ``‖Ψ_i − 0‖`` = the ROW norm, which ``span`` does
        not bound — and on a single-node (or all-equal-Ψ) graph ``span`` is 0, so no
        clamp applies at all while the row angle is nonzero. In practice the two terms
        are the same order (‖Ψ_i‖ ≈ 1.1·√d_model, measured), but do not read
        ``max_angle`` as a bound on every angle appearing in the scores.

        The clamp itself lives in :meth:`layer_scale_factor` and is unconditional, so
        nothing here can fail a run; the ``RuntimeError`` below fires only if the clamp
        arithmetic is wrong, and the ``RuntimeWarning`` (once per model) announces
        saturation.
        """
        B = len(injection_maps)
        r = torch.zeros(B, seq_len, self._wire_d_model, device=device, dtype=torch.float32)
        worst_pair = 0.0
        worst_row = 0.0
        for b in range(B):
            pe = self.pe_model(graphs[b], permutation=permutation).float()   # [N, m]
            pe = pe * torch.tanh(self.pe_gain)
            with torch.no_grad():
                worst_row = max(worst_row, float(pe.norm(dim=-1).max()))
                if pe.shape[0] > 1:
                    worst_pair = max(worst_pair, float(torch.cdist(pe, pe).max()))
            wire_place_at_node_spans(r, b, pe, injection_maps[b], seq_len)
        # Ψ span for this forward; layer_scale_factor() combines it with each σ_ℓ.
        # EXACT pairwise max via cdist, not the O(N·m) bound 2·max‖Ψ_i−μ‖. The exact
        # form is tighter (the bound over-clamps by up to 2x, needlessly shrinking the
        # signal) and is cheap at the N this repo supports: MEASURED on CPU at m=1024,
        # 0.33 ms at N=100, 0.55 ms at N=250, 1.12 ms at N=500, 2.47 ms at N=1000
        # (the O(N·m) bound is 0.15-0.29 ms across the same range). Both are SOUND
        # guards — the bound can only over-clamp, never under-clamp — so switching to it
        # is safe if N ever grows enough for the quadratic term to matter.
        self._wire_psi_span = worst_pair
        with torch.no_grad():
            # layer_omega_scale() is the ONE convention both modes report in: |σ_ℓ| in
            # the expectation arm, max_n‖ω_ℓ[n]‖₂/√m in vanilla (see that method).
            sigmas = [self.layer_omega_scale(li) for li in self.active_layer_indices()]
        sigma_max = max(sigmas) if sigmas else 0.0
        # Pre-clamp diagnostics (what the angle WOULD have been).
        self._wire_measured_angle = sigma_max * worst_pair
        self._wire_measured_row_angle = sigma_max * worst_row
        # POST-CLAMP invariant. layer_scale_factor() pins every layer's effective angle
        # at or below max_angle unconditionally, so this can no longer be reached by any
        # config value or optimizer trajectory — it fires ONLY if the clamp arithmetic
        # itself is wrong. Kept as a live assertion precisely because the guarantee is
        # now structural rather than checked.
        active = self.active_layer_indices()
        eff_max = max((self.layer_omega_scale(li)
                       * self.layer_scale_factor(li) * worst_pair)
                      for li in active) if active else 0.0
        self._wire_effective_angle = eff_max
        if eff_max < self._wire_measured_angle * (1.0 - 1e-9) and not self._wire_warned_angle:
            self._wire_warned_angle = True
            warnings.warn(
                f"WIRE clamp engaged: raw angle {self._wire_measured_angle:.4f} exceeds "
                f"max_angle {self._wire_max_angle:.4f}; effective angle pinned to "
                f"{eff_max:.4f}. Training continues — but σ is now SATURATED, so it can "
                "keep growing with no effect on the model. Watch wire/sigma_raw_max vs "
                "wire/sigma_eff_max in the logs. Warned once per model.",
                RuntimeWarning, stacklevel=2)
        if eff_max > self._wire_max_angle * (1.0 + 1e-5):
            raise RuntimeError(
                "WIRE clamp implementation bug: post-clamp angle "
                f"{eff_max:.6f} exceeds max_angle {self._wire_max_angle:.6f} even though "
                "layer_scale_factor() is applied unconditionally. This is NOT a config "
                "or optimizer condition (σ can no longer reach it) — it means the clamp "
                "arithmetic in layer_scale_factor/layer_omega is wrong. "
                f"(span={worst_pair:.6f}, max|σ_raw|={sigma_max:.6g})")
        return r

    # ------------------------------------------------------------------ forward

    def _arm(self, graphs, injection_maps, seq_len, device, permutation=None):
        """Build ``r`` and publish it where the patched attention fns read it.

        Drops any cached decode angles first: they are derived from the PREVIOUS ``r``
        and must never outlive it.
        """
        self._wire_decode_cos_sin.clear()
        self._wire_signal = self.build_wire_signal(
            graphs, injection_maps, seq_len, device, permutation=permutation)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        graphs: Batch | None = None,
        injection_maps: list[dict[int, list[tuple[int, int]]]] | None = None,
        **kwargs,
    ):
        """Prompt forward with the graph channel armed.

        ``input_ids`` are passed straight through (no ``inputs_embeds`` path): WIRE
        never touches the embeddings, only q/k inside the attention layers. The signal
        is armed only when graphs, injection maps AND input_ids are all present, and
        explicitly cleared otherwise so a stale signal from a previous batch cannot
        leak into an ungrounded forward.
        """
        kwargs.pop("inputs_embeds", None)
        kwargs.pop("input_ids", None)
        if graphs is not None and injection_maps is not None and input_ids is not None:
            self._arm(graphs, injection_maps, input_ids.shape[1], input_ids.device)
        else:
            self._wire_signal = None
        try:
            return self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )
        finally:
            # Disarm unless under gradient checkpointing (backward recomputes the
            # attention forwards and must see the same signal; every forward rebuilds it).
            if not getattr(self.llm, "is_gradient_checkpointing", False):
                self._wire_signal = None
                self._wire_decode_cos_sin.clear()

    def generate_with_graph(self, input_ids=None, graphs=None, injection_maps=None,
                            permutation=None, **kwargs):
        """``generate`` with the graph channel armed for prefill AND every decode step.

        The signal is built ONCE over the prompt and stays armed for the whole rollout:
        the attention fn re-applies each prompt position's rotation to the cached keys
        every step, and generated positions carry ``r = 0`` (they are absent from the
        prompt injection map), exactly as ``injection_scope='prompt_only'`` trains them.
        See :func:`_wire_attention_forward` for the decode semantics and
        :meth:`assert_decode_supported` for the one scope that cannot be decoded.

        ``permutation`` is threaded to the Ψ producer for the eval-time
        permutation-equivariance sweep, exactly as ``GraphAugmentedLLM.build_pe_signal``
        does — WIRE is equivariant to node reordering (Lemma 1).
        """
        self.assert_decode_supported()
        self._arm(graphs, injection_maps, input_ids.shape[1], input_ids.device,
                  permutation=permutation)
        try:
            return self.llm.generate(input_ids=input_ids, **kwargs)
        finally:
            self._wire_signal = None
            self._wire_decode_cos_sin.clear()


def find_last_graph_scope(input_ids_b, tokenizer) -> int:
    """Token index where the last scene-graph block begins (injection scope).

    Only mentions at/after this index are injected (earlier ones live in ICL graphs).
    Matches ``scene graph: •`` (compact block signature + bullet) via per-token decode
    to avoid BPE merges; maps the last match's char offset to its token index.
    Returns 0 (inject whole sequence) when no compact block is found.
    Used by both training collator and eval so scope is consistent.
    """
    seq = list(map(int, input_ids_b))
    # Per-token decode for self-consistent char→token offset mapping.
    pieces = tokenizer.batch_decode(
        [[t] for t in seq], clean_up_tokenization_spaces=False
    )
    text = "".join(pieces)
    matches = list(re.finditer(r"scene graph:\s*•", text, flags=re.IGNORECASE))
    if not matches:
        return 0
    char_start = matches[-1].start()
    cum = 0
    for ti, piece in enumerate(pieces):
        if cum + len(piece) > char_start:
            return ti
        cum += len(piece)
    return 0


def node_token_variants(node_names, tokenizer) -> list[list[list[int]]]:
    """Per-node candidate token-ID sequences for injection matching.

    Returns both standalone and space-preceded tokenizations per node (BPE merges the
    leading space, giving a different id). Both forms are needed for 100% injection coverage.
    """
    return [
        [
            tokenizer.encode(name, add_special_tokens=False),
            tokenizer.encode(" " + name, add_special_tokens=False),
        ]
        for name in node_names
    ]


def has_match(input_ids_b: list[int], to_match:list[int],start_pos:int):
    """Check if `to_match` is present in `input_ids_b` at `start_pos`."""
    end_pos = min(start_pos + len(to_match),len(input_ids_b))
    return input_ids_b[start_pos:end_pos] == to_match

def build_injection_map(
    input_ids_b: list[int],
    node_token_seqs: list,
    scope_start: int = 0,
) -> dict[int, list[tuple[int, int]]]:
    """Build injection map {node_idx: [(start, end), ...]} from token IDs and node sequences.

    ``node_token_seqs[i]`` is a single token-ID list or a list of candidates (e.g. from
    :func:`node_token_variants`). Matches only at/after ``scope_start`` (ICL-graph exclusion).
    Resolved longest-first so shorter node names can't claim tokens belonging to longer ones.

    Returns:
        Dict mapping node index to a list of ``(start, end)`` token spans.
    """
    n = len(input_ids_b)
    # Collect every candidate match as (length, node_idx, start), longest-first.
    spans_all: list[tuple[int, int, int]] = []
    for nid, seqs in enumerate(node_token_seqs):
        variants = seqs if (seqs and isinstance(seqs[0], list)) else [seqs]
        for seq in variants:
            length = len(seq)
            if length == 0:
                continue
            for start in range(n - length + 1):
                if input_ids_b[start:start + length] == seq:
                    spans_all.append((length, nid, start))
    spans_all.sort(key=lambda x: x[0], reverse=True)

    claimed: set[int] = set()
    injection_map: dict[int, list[tuple[int, int]]] = {}
    for length, nid, start in spans_all:
        if start < scope_start:
            continue
        positions = range(start, start + length)
        if claimed.isdisjoint(positions):
            injection_map.setdefault(nid, []).append((start, start + length))
            claimed.update(positions)
    # Stable, deterministic span order per node.
    for nid in injection_map:
        injection_map[nid].sort()
    return injection_map


def _extendable_span(tokens: list[int], node_token_seqs: list) -> bool:
    """True iff some node-name token sequence strictly extends ``tokens`` — the
    mention is still ambiguous at its own final token (e.g. ``region_1`` vs
    ``region_10`` under digit-split tokenization)."""
    k = len(tokens)
    for seqs in node_token_seqs:
        variants = seqs if (seqs and isinstance(seqs[0], list)) else [seqs]
        for seq in variants:
            if len(seq) > k and seq[:k] == tokens:
                return True
    return False


def decode_style_query_map(
    injection_map: dict[int, list[tuple[int, int]]],
    answer_start: int,
    input_ids: list[int],
    node_token_seqs: list,
) -> dict[int, list[tuple[int, int]]]:
    """QUERY-role map under the decode-consistency rule (decode-time design note §3).

    Prompt-side spans (start < ``answer_start``) are kept whole; each answer-side
    span is reduced to the single position where its node id first becomes
    KNOWABLE at decode time:

    - unambiguous name: the span's final token (``e-1``) — the forward consuming
      the completing token can already commit the assignment;
    - name that is a token-prefix of another node's name (``_extendable_span``):
      the position AFTER the span (``e``) — only the next token resolves whether
      the mention stopped short or continues into the longer name. A span ending
      at the sequence end with no resolving position is dropped (matches decode,
      where generation ended before the ambiguity resolved).

    Pair with the FULL map in the key role (``key_injection_maps``): answer
    mentions always act as keys once complete (score-level bias needs no KV-cache
    surgery). Used by ``injection_scope='decode_consistent'`` training and the
    decode-style diagnostic; ``MaskDecodeInjector`` reproduces these positions
    step-by-step at generation (parity-tested).
    """
    out: dict[int, list[tuple[int, int]]] = {}
    for node_idx, spans in injection_map.items():
        kept = []
        for s, e in spans:
            if s < answer_start:
                kept.append((s, e))
            elif not _extendable_span(list(input_ids[s:e]), node_token_seqs):
                kept.append((e - 1, e))
            elif e < len(input_ids):
                kept.append((e, e + 1))
        if kept:
            out[node_idx] = sorted(kept)
    return out


def clamp_injection_map(
    injection_map: dict[int, list[tuple[int, int]]],
    scope_end: int,
) -> dict[int, list[tuple[int, int]]]:
    """Truncate an injection map to token positions strictly below ``scope_end``.

    Spans starting at/after ``scope_end`` are dropped; spans straddling the boundary
    are cut at it; nodes left with no spans are removed. Used by
    ``injection_scope='prompt_only'`` training (only prompt mentions carry the graph
    channel, matching generation, where decode steps receive no injection) and by
    the injection-ablation diagnostic (``prism.eval.injection_diag``).
    """
    clamped: dict[int, list[tuple[int, int]]] = {}
    for nid, spans in injection_map.items():
        kept = [(start, min(end, scope_end)) for start, end in spans if start < scope_end]
        if kept:
            clamped[nid] = sorted(kept)
    return clamped


def exclude_positions_from_injection_map(
    injection_map: dict[int, list[tuple[int, int]]],
    excluded: set[int],
) -> dict[int, list[tuple[int, int]]]:
    """Remove a set of token positions from every span of an injection map.

    Spans overlapping ``excluded`` are split into the maximal remaining sub-spans
    (a mention partially inside the excluded block keeps its outside tokens);
    nodes left with no spans are removed. Used by
    ``injection_scope='exclude_supervised'`` (e12): the graph channel must never
    be attached to loss-target positions, otherwise the supervised token's own
    query/key carries its label (e.g. edge-list reconstruction reading the target
    node's own psi at the predicted position instead of inferring the neighbor
    from the anchor node's features).
    """
    if not excluded:
        return {nid: sorted(spans) for nid, spans in injection_map.items()}
    result: dict[int, list[tuple[int, int]]] = {}
    for nid, spans in injection_map.items():
        kept: list[tuple[int, int]] = []
        for start, end in spans:
            run_start = None
            for pos in range(start, end):
                if pos in excluded:
                    if run_start is not None:
                        kept.append((run_start, pos))
                        run_start = None
                elif run_start is None:
                    run_start = pos
            if run_start is not None:
                kept.append((run_start, end))
        if kept:
            result[nid] = sorted(kept)
    return result


def bucketize_prompt(input_ids_b: list, node_token_seqs : list) -> defaultdict:
    """Map each node to the set of start positions where its token sequence appears.

    Args:
        input_ids_b: token IDs for a single sequence.
        node_token_seqs: per-node token-ID subsequences.

    Returns:
        defaultdict[int, set] of node index → start positions.
    """
    buckets = defaultdict(set)
    for p_idx, p_token in enumerate(input_ids_b):
        for node_idx, node_token_seq in enumerate(node_token_seqs):
            if has_match(input_ids_b, to_match=node_token_seq,start_pos=p_idx):
                buckets[node_idx].add(p_idx)
    return buckets


class MaskDecodeInjector:
    """Decode-time structural-mask extension for the mask archs (design note §2.2).

    Batch-size-1 generation only (eval generates one sample at a time). Register
    :meth:`pre_hook` as a forward pre-hook on ``model.llm`` for the duration of one
    ``generate`` call; remove the handle and clear ``model._decode_bias_row`` after.

    Per decode step (single-token forward), the hook:
      1. appends the consumed token to the generated suffix and re-derives node-mention
         spans over the suffix with ``build_injection_map`` (same longest-first,
         disjoint semantics as training); a span ending exactly at the suffix end is
         DEFERRED while any node-name token sequence strictly extends it (partial-
         mention ambiguity rule);
      2. tags the current query iff it is the FINAL token of a completed span
         (``decode_style`` semantics — the §3 consistency rule);
      3. arms ``model._decode_bias_row`` ([1, 1, 1, K]) with ``M[q, π_k(s)]`` over all
         key positions (prompt wiring + every completed suffix mention), −inf on
         non-adjacent node pairs, 0 elsewhere; untagged queries arm nothing.

    ``M`` is the same per-node-pair value the prefill bias uses: 0/−inf for
    ``GraphMaskLLM``, ``log(α + (1−α)·sim(ΨΨᵀ))`` (Ψ computed once at construction) for
    ``LearnableGraphMaskLLM``.
    """

    def __init__(self, model, pyg_graph, prompt_injection_map, prompt_len,
                 node_token_seqs, permutation=None):
        self.model = model
        self.node_token_seqs = node_token_seqs
        self.prompt_len = prompt_len
        self.generated: list[int] = []
        self._committed: set = set()
        self.device = next(model.parameters()).device
        # `permutation` must match the one build_structural_mask used for the prefill bias:
        # the decode rows index the SAME node axis, so a permuted prefill with an
        # unpermuted decode extension would mix two different node labellings mid-rollout.
        adj = model._node_adjacency(pyg_graph, self.device, permutation=permutation)
        if hasattr(model, "pe_model"):
            # LearnableGraphMaskLLM: learned relative-PE values on allowed pairs.
            with torch.no_grad():
                psi = model.pe_model(pyg_graph, permutation=permutation).float()
                if model._mask_psi_scale == "cosine":
                    psi = psi / psi.norm(dim=-1, keepdim=True).clamp_min(model._mask_eps)
                    sim = psi @ psi.t()
                else:
                    sim = (psi @ psi.t()) / (psi.shape[-1] ** 0.5)
            gate = (model._mask_alpha + (1.0 - model._mask_alpha) * sim).clamp_min(model._mask_eps)
            node_m = gate.log()  # log-gate: matches build_structural_mask's multiplicative fold
        else:
            node_m = torch.zeros_like(adj, dtype=torch.float32)
        neg = torch.finfo(torch.float32).min
        self.node_values = torch.where(adj, node_m, torch.full_like(node_m, neg))
        self.prompt_tok2node = tok2node_vector(prompt_injection_map, prompt_len,
                                               self.device)
        # Flattened variant list for the partial-mention ambiguity check.
        self._all_variants = [seq for seqs in node_token_seqs
                              for seq in (seqs if seqs and isinstance(seqs[0], list)
                                          else [seqs]) if seq]

    def _suffix_spans(self):
        """Completed, unambiguous suffix spans: {node: [(s, e)]}, suffix coordinates."""
        smap = build_injection_map(self.generated, self.node_token_seqs, scope_start=0)
        n = len(self.generated)
        out = {}
        for nid, spans in smap.items():
            kept = []
            for start, end in spans:
                if end == n and self._extendable(self.generated[start:end]):
                    continue                     # still ambiguous — defer assignment
                kept.append((start, end))
            if kept:
                out[nid] = kept
        return out

    def _extendable(self, toks):
        """True iff some node-name token sequence strictly extends ``toks``."""
        k = len(toks)
        return any(len(seq) > k and seq[:k] == toks for seq in self._all_variants)

    def pre_hook(self, module, args, kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is None or input_ids.shape[1] != 1:
            return                                # prefill (multi-token forward)
        if input_ids.shape[0] != 1:
            raise RuntimeError(
                "MaskDecodeInjector supports batch-size-1 generation only "
                f"(got batch {input_ids.shape[0]}; beam search is not wired).")
        self.generated.append(int(input_ids[0, 0]))
        prev_committed = self._committed
        spans = self._suffix_spans()
        self._committed = {(nid, sp) for nid, sps in spans.items() for sp in sps}
        p_suffix = len(self.generated) - 1
        # Query tag = the position where a span's node id first becomes knowable
        # (mirrors decode_style_query_map): a span completing unambiguously at the
        # current position tags NOW (end-1 == p); a span that was DEFERRED for
        # prefix-ambiguity and committed only on this step tags at this resolving
        # position (end-1 == p-1, newly committed).
        q_node = -1
        for nid, sps in spans.items():
            for start, end in sps:
                if end - 1 == p_suffix:
                    q_node = nid
                elif end - 1 == p_suffix - 1 and (nid, (start, end)) not in prev_committed:
                    q_node = nid
        if q_node < 0:
            self.model._decode_bias_row = None    # untagged query: bias row is all-zero
            return
        k_len = self.prompt_len + len(self.generated)
        tok2node_k = torch.full((k_len,), -1, dtype=torch.long, device=self.device)
        tok2node_k[:self.prompt_len] = self.prompt_tok2node
        for nid, sps in spans.items():
            for start, end in sps:
                tok2node_k[self.prompt_len + start:self.prompt_len + end] = nid
        row = torch.zeros(k_len, dtype=torch.float32, device=self.device)
        k_pos = (tok2node_k >= 0).nonzero(as_tuple=True)[0]
        row[k_pos] = self.node_values[q_node, tok2node_k[k_pos]]
        self.model._decode_bias_row = row.view(1, 1, 1, k_len)
        # Identity-RoPE parity: under decode_consistent, training zeroes the position of
        # exactly the query-tagged positions (decode_style_query_map), which are the ones
        # tagged here — so zero this step's position_id too, or the same token would carry
        # RoPE at decode and none in training. Only the CURRENT step's kwargs are
        # rewritten; HF's own position_ids counter in model_kwargs is untouched, so
        # subsequent steps keep advancing naturally (a zero written back there would
        # restart the whole sequence's positions).
        pos = kwargs.get("position_ids")
        if getattr(self.model, "_disable_graph_token_rope", False) and pos is not None:
            return args, {**kwargs, "position_ids": torch.zeros_like(pos)}
