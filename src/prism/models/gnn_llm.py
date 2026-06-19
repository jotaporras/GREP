import copy
import importlib
import math
import re
import warnings
from collections import defaultdict

import torch
from torch import nn
from torch_geometric.data import Batch, Data
import transformers.masking_utils as masking_utils
from transformers import AttentionInterface, PreTrainedModel

from prism.models.composite_graph import build_composite_graph
from prism.models.llama import disable_rope


# Name under which the graph-PE-injecting attention function is registered in
# transformers' global ``ALL_ATTENTION_FUNCTIONS``. ``GraphAugmentedLLM`` points
# the wrapped LLM's ``config._attn_implementation`` at this so every decoder layer
# dispatches through ``_prism_pe_attention_forward`` (see ``_install_pe_injection``).
_PRISM_PE_IMPL = "prism_pe"


def _prism_pe_attention_forward(module, query, key, value, attention_mask,
                                scaling=None, dropout=0.0, **kwargs):
    """Attention fn that injects the graph signal Ψ into the *post-RoPE* q/k/v.

    Registered as the ``"prism_pe"`` attention implementation. Every HF decoder
    calls its attention interface as ``fn(module, q, k, v, attn_mask, scaling=…,
    **kwargs)`` with q/k already q/k-normed, rotary-embedded and shaped
    ``[B, H, S, head_dim]``. We add the *unrotated* ``W_q·Ψ`` / ``W_k·Ψ`` (and
    ``W_v·Ψ``) here, then delegate to the LLM's original attention impl.

    Because every model family (Llama, Qwen2, gemma-4, …) hands the interface
    the same post-RoPE ``[B, H, S, d]`` tensors, this single function works
    across architectures — each model keeps its own q/k-norm, RoPE convention,
    scaling, sliding window and KV-sharing. With Ψ absent (or zero) the output is
    identical to stock attention.
    """
    model = getattr(module, "_prism_pe_model", None)
    psi = None if model is None else model._pe_signal
    # Inject only on the prompt forward whose Ψ length matches the query length;
    # cached single-token decode steps (S mismatch) fall through to stock attn.
    if psi is not None and psi.shape[0] == query.shape[0] and psi.shape[1] == query.shape[-2]:
        psi = psi.to(query.dtype)
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


class GraphAugmentedLLM(PreTrainedModel):  # ty:ignore[unsupported-base]
    """
    Graph-Augmented LLM (GREP-PRISM).

    Domain-agnostic: receives pre-computed injection maps that specify where
    to add graph positional encodings into the LLM input embeddings.

    The graph encoder (``pe_model``) can be any module with the interface
    ``forward(data) -> Tensor[n, d_model]``.  Two options are supported:
      - ``RandomGNNPositionalEncodings`` (R-PEARL only, no GT blocks)
      - ``GraphTransformer`` (full Sparse GT with R-PEARL inside)

    Positional-injection scheme — the LLM sees ``RoPE(X) + Ψ`` at every layer:
        ``X`` is the word-embedding matrix and ``Ψ`` the graph PE placed at the
        node-name token spans. RoPE is applied *inside every attention layer* to
        the projected query/key (``apply_rotary_pos_emb``), so adding Ψ to the
        residual stream would make the LLM compute ``RoPE(W·(X+Ψ))`` — Ψ spun by
        the *sequence-position* rotation, which is wrong for a *graph* positional
        code. (A residual-stream counter-rotation ``R_p^{-1}Ψ`` can't fix this:
        ``W_q``/``W_k`` don't commute with the per-head rotation, and one residual
        vector can't satisfy the q- and k-constraints at once.)

        Instead of touching the residual stream, we register a custom attention
        implementation (``"prism_pe"``, see ``_install_pe_injection``) and point the
        wrapped LLM's ``config._attn_implementation`` at it. The LLM runs its own
        native attention forward — its q/k-norm, RoPE convention, scaling, sliding
        window and KV-sharing all untouched — and where it hands the already-rotated
        ``[B, H, S, d]`` query/key/value to its attention function we add the
        *unrotated* graph term::

            q = RoPE(W_q · h) + W_q · Ψ
            k = RoPE(W_k · h) + W_k · Ψ
            v =      W_v · h  + W_v · Ψ     # value/content path ("content too")

        Ψ is projected through that layer's own (LoRA-adapted) q/k/v_proj, so the
        graph signal enters the query/key dot product *unrotated* — exact
        ``RoPE(X) + Ψ`` — at all layers. Routing through the attention *function*
        (rather than reimplementing each model family's forward) keeps this
        architecture-agnostic across Llama, Qwen2, gemma-4, …. Ψ is supplied per
        forward via ``self._pe_signal`` (``[B, seq, hidden]``); injection is skipped
        on cached single-token decode steps (seq mismatch), so only prompt tokens
        carry it.

    Args:
        llm (nn.Module): LLM to perform classical planning
        pe_model (nn.Module): R-PEARL positional-encodings model
        d_model (int): Dimensionality of the positional encodings
        eps (float): Lipschitz normalization epsilon for the projection head
    """

    def __init__(self, llm: nn.Module, pe_model: nn.Module,
                 d_model: int, eps: float = 1e-8, pe_gain_init: float = 1.0,
                 disable_graph_token_rope: bool = False, use_pe_norm: bool = True,
                 pe_node_features: str = "random"):
        # GraphAugmentedLLM is not a registered HF architecture, so
        # PreTrainedModel rejects SDPA/flash-attn.  Force "eager" on the
        # wrapper config — the inner self.llm keeps its own attn impl.
        config = copy.copy(llm.config)
        config._attn_implementation = "eager" # ty: ignore[invalid-assignment]
        # Likewise, the wrapper is not a registered MoE class, so PreTrainedModel
        # can't validate "grouped_mm" experts against it (it greps this file for
        # @use_experts_implementation and raises when absent). Force "eager" on the
        # wrapper config — the inner self.llm keeps its own experts impl.
        config._experts_implementation = "eager" # ty: ignore[invalid-assignment]
        super().__init__(config)
        self.llm = llm

        # Place pe_model and pe_proj on the same device as the LLM so PEFT
        # wrapping (which only touches LoRA target modules) doesn't leave them on CPU.
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = llm.device
        self.pe_model = pe_model.to(device)
        # Plain linear projection to the LLM hidden size — no spectral/Lipschitz
        # normalization. The learnable ``pe_gain`` gate (below) sets the injection
        # strength; the projection weights learn the rest.
        self.pe_proj = nn.Linear(
            d_model, llm.config.get_text_config().hidden_size, device=device)
        # Learnable gate on the PE injection: g = tanh(pe_gain) ∈ (-1,1). Lets the
        # model regulate how strongly (and with which sign) Ψ enters RoPE(X) + g·Ψ.
        # pe_gain_init=1.0 → g ≈ 0.76 (active from step 0). pe_gain_init=0.0 → g=0:
        # Ψ is fully off at init (forward == base LLM) and, because the structural
        # path is multiplied by tanh(pe_gain)=0, its parameters get zero gradient
        # until the gate itself moves — a true cold-start.
        self.pe_gain = nn.Parameter(torch.tensor(float(pe_gain_init), device=device))
        # Calibrated RMSNorm on the projected Ψ (VLM modality-connector best practice:
        # MoCa 2410.07167, 2512.08374, 2503.17349). A fresh pe_proj has an uncalibrated
        # output scale; injecting that raw into a frozen LLM's residual/attention stream
        # is the magnitude-mismatch that drives divergence. We RMS-normalize Ψ and rescale
        # it to the base model's own mean token-embedding RMS, so Ψ enters at text scale.
        # The norm sets the SCALE; pe_gain (gate) sets the RAMP — separate jobs. Loaded
        # checkpoints overwrite this weight; the init only matters for fresh training.
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

        # When True, graph (node-name) token spans are assigned position_id 0 so RoPE is
        # the identity there — node names carry no sequential rotation; their position is
        # meant to come from Ψ. See _graph_token_position_ids / forward.
        self._disable_graph_token_rope: bool = bool(disable_graph_token_rope)

        # R-PEARL input features. "random" => the GNN samples its own random probes
        # (data.x ignored). "word_embeddings" => build_pe_signal computes a per-node
        # feature (mean word-embedding of the node's name tokens) and feeds it as the
        # GNN's data.x; the pe_model must be built with node_feature_dim = hidden size.
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

    def _decoder_layers(self):
        """Return the LLM's decoder layer list (Llama/Qwen2: ``<CausalLM>.model.layers``)."""
        base = getattr(self.llm, "model", None)
        if base is not None and hasattr(base, "layers"):
            return base.layers
        return self.llm.get_decoder().layers

    def _install_pe_injection(self) -> None:
        """Route every self-attention layer through the ``prism_pe`` attention fn.

        Rather than reimplementing each model family's attention ``forward`` (which
        is Llama-specific and breaks on e.g. gemma-4's q/k-norm, single-tensor RoPE,
        sliding window and KV-sharing), we let the LLM run its native forward and
        only swap the *attention function* it dispatches to, via
        ``config._attn_implementation``. Each attention module records its original
        attention impl (so ``prism_pe`` can delegate to it after adding Ψ) plus a
        back-reference to this wrapper (so it can read ``self._pe_signal``). Set on
        instances (not the class) so it stays scoped to this LLM and survives PEFT,
        which swaps leaf Linears in place.
        """
        layers = self._decoder_layers()
        if len(layers) == 0:
            return
        first_attn = layers[0].self_attn
        mod = importlib.import_module(type(first_attn).__module__)
        # Invariant: only graph tokens may be modified. Ψ is zero at every non-graph
        # token, so the injected ``W·Ψ`` vanishes there — but ONLY if the q/k/v
        # projections are bias-free (``proj(0)=0``). A bias would add ``b`` to every
        # token, perturbing non-graph positions too. All supported LLMs (Llama/Qwen2/
        # gemma-4) use bias-free attention; fail loud if a future base model doesn't.
        for _name in ("q_proj", "k_proj", "v_proj"):
            _proj = getattr(first_attn, _name, None)
            if _proj is not None and getattr(_proj, "bias", None) is not None:
                raise ValueError(
                    f"{type(first_attn).__name__}.{_name} has a bias; prism_pe injection "
                    "assumes bias-free attention projections so non-graph tokens stay "
                    "untouched (Ψ=0 ⇒ W·Ψ=0). This base model needs a bias-aware injection."
                )
        # Capture the LLM's real attention impl ONCE, before mutating any config:
        # configs are typically shared across layers, so a per-layer read after the
        # first mutation would already see "prism_pe". Persisted for idempotent
        # re-install (e.g. if called again after a config reload).
        if not hasattr(self, "_prism_orig_attn_impl"):
            impl = first_attn.config._attn_implementation
            self._prism_orig_attn_impl = "eager" if impl == _PRISM_PE_IMPL else impl
        # Resolve the original attention fn to delegate to. transformers ≥5.12 uses
        # ``get_interface(impl, default)``; older versions subscript the registry and
        # special-case eager — support both so the Llama path keeps working on 5.0.x.
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
        # HF builds the causal/sliding mask from the impl *name*: ``create_causal_mask``
        # returns None for any impl not in the mask registry, which would silently
        # disable causal masking under "prism_pe". Register prism_pe's mask to mirror
        # the original impl's, so the model builds exactly the mask the delegated
        # attention fn expects (covers gemma-4's causal and sliding masks alike).
        mask_fns = getattr(masking_utils, "ALL_MASK_ATTENTION_FUNCTIONS", None)
        if mask_fns is not None and self._prism_orig_attn_impl in mask_fns._global_mapping:
            masking_utils.AttentionMaskInterface.register(
                _PRISM_PE_IMPL, mask_fns._global_mapping[self._prism_orig_attn_impl])

        for layer in layers:
            attn = layer.self_attn
            # Bypass nn.Module.__setattr__ for the wrapper back-reference: assigning
            # an nn.Module as an attribute would register it as a submodule and form
            # a cycle (attn → wrapper → llm → attn), double-counting parameters.
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
                # Per-node feature = mean word-embedding of the node's name tokens, taken
                # from the node's mention spans in this prompt. Fail loud if any graph node
                # has no mention (every example lists all nodes, so coverage is required).
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
                # Detach: features come from the (frozen) embedding table; gradient still
                # flows to the GNN/pe_proj/pe_gain downstream.
                g.x = feats.detach()
            pe = self.pe_proj(self.pe_model(g, permutation=permutation))  # [n, hidden_size]
            # Calibrated RMSNorm rescales Ψ to the base model's token-embedding scale
            # (see __init__); skipped for checkpoints trained without it (use_pe_norm=False).
            if self.pe_norm is not None:
                pe = self.pe_norm(pe)
            # Apply the learnable gate g = tanh(pe_gain) ∈ (-1, 1). The norm sets the
            # scale, the gate sets the (signed) ramp. inference.py calls this method
            # directly, so eval matches training.
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

        Non-graph tokens keep their natural ``arange`` index. Causality is unaffected
        (HF derives the causal mask from cache_position, not position_ids). Returned
        only when ``_disable_graph_token_rope`` is set; callers pass it to the LLM.
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

    def _augment_embeddings(
        self,
        input_ids: torch.Tensor,
        graphs: Batch,
        injection_maps: list[dict[int, list[tuple[int, int]]]],
    ) -> torch.Tensor:
        """Return plain token embeddings and arm Ψ for the patched attention layers.

        Unlike the legacy residual-stream injection, Ψ is no longer added to the
        embeddings here — it is registered on ``self._pe_signal`` and added *after*
        RoPE inside every attention layer (see ``_install_pe_injection``), so the
        LLM sees ``RoPE(X) + Ψ``. Returns the unmodified ``X`` embeddings to feed as
        ``inputs_embeds``.
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
            # Disarm Ψ so a later forward can't inject a stale signal. But under
            # gradient checkpointing the backward pass *recomputes* the attention
            # forwards and must see the same Ψ, so keep it armed in that case (it
            # is rebuilt/overwritten at the start of every forward anyway).
            if not getattr(self.llm, "is_gradient_checkpointing", False):
                self._pe_signal = None


class GatedInjection(nn.Module):
    """M7 — blend token embeddings with the Graph Transformer output via a gate.

    Produces the LLM ``inputs_embeds`` from the original token embeddings ``X``
    and the GT token-node outputs ``Y[V_Tx]`` (M6), through a single learnable
    gate that starts at ≈0 (R6 cold-start). Scene-node outputs are discarded;
    only the token rows reach the LLM.

    Modes (spec §4/M7):
      - ``"interpolate"`` (default): ``inputs_embeds = (1 - gate) * X + gate * Y``.
        The gate is the fraction by which the LLM's own embeddings are replaced
        by the structural output — at ``gate=0`` the LLM sees clean Llama
        embeddings, and structure ramps in as the gate grows.
      - ``"additive"``: ``inputs_embeds = X + gate * Y``.
      - ``"none"``: ``inputs_embeds = Y`` — the Graph Transformer output is fed
        straight to the LLM with **no gate and no mixing of the token embeddings
        X**. There is no learnable gate in this mode; the LLM must consume the
        structural representation directly. Pairing it with a LoRA warmup (the
        LLM frozen for the first N optimizer steps, see ``LoraWarmupCallback``)
        lets the GT learn to emit an LLM-consumable representation before the
        adapters start adapting to it. ``gate`` is registered as a fixed
        non-trainable buffer of 1.0 so the M11 diagnostics
        (``grep/structural_gate``, ``grep/contrib_ratio``) keep reading sensibly.

    No RoPE or positional transform is applied here (the LLM is RoPE-disabled, M8).

    Args:
        d_model (int): Embedding width (only used for ``gate_per_dim``).
        gate_init (float): Initial gate value (≈0 cold-start, R6). Ignored when
            ``injection_mode == "none"``.
        gate_per_dim (bool): Per-feature gate vector instead of a scalar. Ignored
            when ``injection_mode == "none"``.
        injection_mode (str): "interpolate" (default), "additive", or "none".
    """

    def __init__(self, d_model: int, gate_init: float = 0.0,
                 gate_per_dim: bool = False, injection_mode: str = "interpolate"):
        super().__init__()
        if injection_mode not in ("interpolate", "additive", "none"):
            raise ValueError(
                "injection_mode must be 'interpolate', 'additive', or 'none', "
                f"got {injection_mode!r}"
            )
        self.injection_mode = injection_mode
        if injection_mode == "none":
            # No learnable gate: the GT output replaces X outright. Keep a fixed
            # gate=1.0 buffer (not a Parameter) so callbacks/optimizer code that
            # reads ``injection.gate`` still works and reports the true full-strength
            # injection (contrib_ratio = ‖Y‖/‖X‖) without adding a trainable param.
            self.register_buffer("gate", torch.tensor(1.0))
        else:
            gate_shape = (d_model,) if gate_per_dim else ()
            self.gate = nn.Parameter(torch.full(gate_shape, float(gate_init)))

    def forward(self, X: torch.Tensor, Y_tx: torch.Tensor) -> torch.Tensor:
        """Combine token embeddings X with the gated GT token-node outputs Y[V_Tx].

        Args:
            X: token embeddings [c, d] (V_Tx rows only).
            Y_tx: GT token-node outputs [c, d] (Y[V_Tx]).

        Returns:
            inputs_embeds [c, d] for the LLM.
        """
        if self.injection_mode == "none":
            # Feed the GT output straight through: no gate, no mixing with X.
            return Y_tx
        gate = self.gate
        if self.injection_mode == "interpolate":
            return (1 - gate) * X + gate * Y_tx
        return X + gate * Y_tx


class CompositeGraphLLM(PreTrainedModel):  # ty:ignore[unsupported-base]
    """Composite-graph assembly: M6 Graph Transformer → M7 gate → M8 RoPE-disabled Llama.

    For each sequence: the token embeddings ``X`` become the directed-cycle node
    features, the Graph Transformer (``gt_model``) refines them over the augmented
    graph (R-PEARL Ψ fused as ``X_full + Ψ``), the token-node rows ``Y[V_Tx]`` are
    blended back with ``X`` through the cold-start gate (M7), and the result is fed
    as ``inputs_embeds`` to the RoPE-disabled Llama (M8). Scene-node outputs are
    discarded.

    Inputs match the existing ``SpineDataCollator`` contract — a PyG Batch of
    scene graphs (``graphs``) and per-sample ``injection_maps`` ({scene_node_idx →
    token spans}). For each sequence the composite graph G is assembled on the fly
    (M4): a directed cycle over the ``c`` token positions, the scene graph, and the
    cross-links from the injection map. Token embeddings X seed the cycle nodes,
    R-PEARL + the GT refine over G, and ``Y[V_Tx]`` is gated back into X (M7).

    Args:
        llm (nn.Module): Llama for causal LM (RoPE disabled here unless told not to).
        gt_model (nn.Module): GraphTransformer (M6); ``forward(data, token_embeddings,
            is_token)`` returns per-node features ``Y``.
        d_model (int): Embedding / GT width (must equal the LLM hidden size).
        gate_init, gate_per_dim, injection_mode: M7 gate settings (R6).
        disable_llm_rope (bool): Apply the M8 RoPE disable to ``llm`` (default True).
        cycle_weight, cycle_directed, crosslink_*: M4 composite-graph settings.
    """

    def __init__(self, llm: nn.Module, gt_model: nn.Module, d_model: int,
                 gate_init: float = 0.0, gate_per_dim: bool = False,
                 injection_mode: str = "interpolate", disable_llm_rope: bool = True,
                 cycle_weight: float = 1.0, cycle_directed: bool = True,
                 crosslink_weight: float = 1.0, crosslink_mention_to_node: bool = True,
                 crosslink_mention_clique: bool = True):
        config = copy.copy(llm.config)
        config._attn_implementation = "eager"  # ty: ignore[invalid-assignment]
        super().__init__(config)
        self.llm = disable_rope(llm) if disable_llm_rope else llm

        try:
            device = next(self.llm.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        self.gt_model = gt_model.to(device)
        self.injection = GatedInjection(
            d_model, gate_init=gate_init, gate_per_dim=gate_per_dim,
            injection_mode=injection_mode,
        ).to(device)

        self.cycle_weight = cycle_weight
        self.cycle_directed = cycle_directed
        self.crosslink_weight = crosslink_weight
        self.crosslink_mention_to_node = crosslink_mention_to_node
        self.crosslink_mention_clique = crosslink_mention_clique

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)  # defer to nn.Module first
        except AttributeError:
            return getattr(self.llm, name)

    def _composite_graph(self, scene, injection_map, c, device, permutation=None):
        """Assemble the composite graph G for one sequence (M4) on ``device``.

        With a ``permutation`` (transferability sweep), the scene nodes are
        relabeled — matching the legacy R-PEARL semantics of permuting over the
        scene graph's ``num_nodes`` — and the injection-map keys are remapped so
        each token still cross-links to the same scene entity. The token cycle is
        never permuted (sequence positions are fixed; RoPE is off).
        """
        scene_edge_index = scene.edge_index.to(device)
        n_scene = scene.num_nodes
        scene_edge_weight = getattr(scene, "edge_weight", None)
        if scene_edge_weight is None:
            scene_edge_weight = torch.ones(scene_edge_index.shape[1], device=device)
        else:
            scene_edge_weight = scene_edge_weight.to(device)
        if permutation is not None:
            scene_edge_index = permutation.apply(scene_edge_index, n_scene, device=device)
            perm = permutation.perm.to(device)
            injection_map = {int(perm[k]): v for k, v in injection_map.items()}
        return build_composite_graph(
            c, scene_edge_index, scene_edge_weight, n_scene, injection_map,
            cycle_weight=self.cycle_weight, cycle_directed=self.cycle_directed,
            crosslink_weight=self.crosslink_weight,
            crosslink_mention_to_node=self.crosslink_mention_to_node,
            crosslink_mention_clique=self.crosslink_mention_clique,
        )

    def _fuse_embeddings(self, input_ids, graphs, injection_maps, permutation=None,
                         return_token_pe=False, return_c_tok=False, return_cbias=False):
        """Build G per sequence and run M4→M6→M7, returning ``inputs_embeds`` [B, c, d].

        When ``return_token_pe`` is True, also returns the un-gated GT token-row
        output ``Y[V_Tx]`` stacked as ``[B, c, d]`` — the per-position structural
        code that ``InjectedCompositeGraphLLM`` injects into q/k/v at every layer.

        When ``return_c_tok`` is True, also returns the composite token-block
        covariance ``C_tok`` stacked as ``[B, c, c]``, scaled to ``‖X‖`` — the
        per-layer relative-position operator (``c_per_layer``). See :meth:`_compute_c_tok`.

        When ``return_cbias`` is True (Design D / c_bias), also returns the two
        per-sequence [B, c, c] additive-bias kernels ``(Ĉ, Ψ̃)`` from
        :meth:`_kernel_and_psi` — the covariance ``Ĉ`` (live per ``self.c_kernel``)
        and the first-moment Gram ``Ψ̃=ΨΨᵀ``. Both are graph-DEPENDENT (per sequence).
        """
        X = self.llm.get_input_embeddings()(input_ids)  # [B, c, d]
        device = X.device
        c = input_ids.shape[1]
        fused = []
        token_pe = [] if return_token_pe else None
        c_tok = [] if return_c_tok else None
        cb_C = [] if return_cbias else None
        cb_P = [] if return_cbias else None
        for b in range(input_ids.shape[0]):
            aug = self._composite_graph(graphs[b], injection_maps[b], c, device, permutation=permutation)
            aug_data = Data(
                x=torch.zeros(aug.num_nodes, 1, device=device),
                edge_index=aug.edge_index,
            )
            aug_data.edge_weight = aug.edge_weight
            # M6: token embeddings on V_Tx, zeros on V_Sc, fused with R-PEARL Ψ.
            Y = self.gt_model(aug_data, token_embeddings=X[b], is_token=aug.is_token)
            Ytok = Y[aug.is_token]
            # M7: blend X with the token-node outputs Y[V_Tx] through the gate.
            fused.append(self.injection(X[b], Ytok))
            if return_token_pe:
                token_pe.append(Ytok)
            if return_c_tok:
                c_tok.append(self._compute_c_tok(aug, aug_data, X[b]))
            if return_cbias:
                C_b, P_b = self._kernel_and_psi(aug, aug_data, c, device)
                cb_C.append(C_b)
                cb_P.append(P_b)
        # The GT runs in float32 (sparse sampled_addmm is fp32-only) and the gate
        # promotes the bf16 X to fp32, so cast inputs_embeds back to the LLM's dtype.
        # Training tolerated the fp32 mismatch under autocast; generate() has no
        # autocast, so the fp32 hidden state would hit the bf16 lm_head and raise
        # "expected scalar type Float but found BFloat16".
        inputs_embeds = torch.stack(fused, dim=0).to(X.dtype)
        out = [inputs_embeds]
        if return_token_pe:
            out.append(torch.stack(token_pe, dim=0))
        if return_c_tok:
            out.append(torch.stack(c_tok, dim=0))
        if return_cbias:
            out.append(torch.stack(cb_C, dim=0))
            out.append(torch.stack(cb_P, dim=0))
        return out[0] if len(out) == 1 else tuple(out)

    @staticmethod
    def _unit_diag(M, eps: float = 1e-12):
        """Correlation-normalize a PSD [c,c] kernel to ~unit diagonal — robust + fail-loud.

        Fixes the earlier blow-up (an absolute/max-relative floor of ~1e-30 could divide a
        degenerate or fp-non-PSD kernel by ~0 → Inf → NaN logits → the silent loss→0 / acc→0
        collapse). Now:
          • fail LOUD if the kernel is already non-finite — that is an UPSTREAM GT/R-PEARL
            divergence (lower structural_lr / inspect the GT), not a normalization issue;
          • clamp the diagonal ≥ 0 (PSD ⇒ ≥0; kills fp-negative variances);
          • floor RELATIVE TO THE MEAN variance (scale-invariant, but a near-zero-variance
            token — e.g. Ψ̃'s non-graph rows where ‖Ψ_t‖²≈0 — gets a bounded small-coupling
            row, not 1/√0);
          • hard-clamp entries to [-1, 1] so the additive bias is provably bounded.
        For a finite PSD input the result is finite and bounded by construction."""
        if not torch.isfinite(M).all():
            raise FloatingPointError(
                "[c_bias] non-finite covariance kernel before normalization — UPSTREAM "
                "GT/R-PEARL divergence (NaN/Inf), not a _unit_diag issue. Lower "
                "structural_lr_mult / check the GT; failing loud instead of collapsing to 0.")
        d = M.diagonal().clamp(min=0)
        floor = d.mean().clamp(min=eps) * 1e-3
        d = d.clamp(min=floor).sqrt()
        out = M / d[:, None] / d[None, :]
        return out.clamp(-1.0, 1.0)

    def _kernel_and_psi(self, aug, aug_data, c, device):
        """Design D per-sequence [c,c] additive-bias kernels ``(Ĉ, Ψ̃)``, unit-diagonal.

          - ``c_kernel="sampled"``  : Ĉ = E_q[ΦΦᵀ]−ΨΨᵀ and Ψ̃ = ΨΨᵀ from ONE probe pass
            on the FULL composite graph (captures the nonlinear R-PEARL response; non-
            mention tokens inherit scene context through the graph diffusion);
          - ``c_kernel="analytic"`` : Ĉ = H(S)H(S)* via the all-layer cascade taps and
            matrix powers on S=[[S_c,B],[Bᵀ,S_sc]]; Ψ̃ = ΨΨᵀ from the first moment Ψ.

        Both carry gradient to the R-PEARL filter.
        """
        pe = self.gt_model.pe_model
        if self.c_kernel == "analytic":
            C = self._analytic_c_from_gso(self._analytic_taps().to(device), aug.gso, c)
            psi = pe(aug_data)[:c]                             # Ψ token rows [c, d]
            Psi = psi @ psi.t()
        else:                                                 # "sampled" (default)
            C, Psi = pe.covariance_token_block(aug_data, c)
        return self._unit_diag(C), self._unit_diag(Psi)

    def _compute_c_tok(self, aug, aug_data, X_b):
        """Materialize the composite token-block covariance ``C_tok`` [c, c] (c_per_layer).

        Applies the AugR-PEARL second moment ``C = E_q[Φ(q)Φ(q)ᵀ]`` to the token-row
        identity, so ``C·e_j`` reads off column ``j`` of ``C`` without forming the N×N
        matrix; the token rows of the result are the token block ``C[V_Tx, V_Tx]``,
        which includes token→scene→token paths (the scene coupling the crosslinks
        build). Raw (un-gated), then scaled so ``C_tok·X`` matches ``X``'s mean row-norm.
        Deterministic (no learnable params). (The c_bias path uses the ANALYTIC
        :meth:`_analytic_c_tok` instead — see that method.)
        """
        device = X_b.device
        N = aug.num_nodes
        c = int(aug.is_token.sum())
        sig = torch.zeros(N, c, device=device, dtype=torch.float32)
        sig[aug.is_token] = torch.eye(c, device=device, dtype=torch.float32)
        C_full = self.gt_model.pe_model.second_moment_apply(
            aug_data, sig, scale_to_signal=False)        # [N, c] = C[:, V_Tx]
        C_tok = C_full[aug.is_token]                      # [c, c] = C[V_Tx, V_Tx]
        Xf = X_b.float()
        cur = (C_tok @ Xf).norm(dim=-1).mean().clamp(min=1e-12)
        target = Xf.norm(dim=-1).mean()
        return C_tok * (target / cur)

    @staticmethod
    def _analytic_c_row_from_taps(H, c, eps: float = 1e-12):
        """Analytic relative-position row ``c_row[δ] = c(δ)/c(0)`` from R-PEARL taps.

        The proof's point: the SAME taps that produce Ψ give the relative-position
        covariance analytically (no probes). For the directed circulant cycle ``S_c``
        (eigenvalues ``ω^k=e^{2πik/c}``) a graph filter ``h(S_c)=Σ_j H_j S_c^j`` has
        per-mode response ``ĥ(ω_k)=Σ_j H_j ω_k^j``; ``ρ_k=‖ĥ(ω_k)‖²≥0``,
        ``c(δ)=IDFT(ρ)``, normalized by ``c(0)=mean(ρ)>0``. ``H`` is ``[K+1, F]``.
        This is the cheap ``[c]`` part — used directly at decode (the full ``[c,c]`` is
        only needed for the prompt bias / value-mix). Gradient flows to ``H``.
        """
        K1 = H.shape[0]
        dev = H.device
        kk = torch.arange(c, device=dev, dtype=torch.float32)
        jj = torch.arange(K1, device=dev, dtype=torch.float32)
        ang = (2.0 * math.pi / c) * torch.outer(kk, jj)        # [c, K1]
        Hf = H.float()
        hr = torch.cos(ang) @ Hf                                # Re ĥ(ω_k)  [c, F]
        hi = torch.sin(ang) @ Hf                                # Im ĥ(ω_k)  [c, F]
        rho = (hr * hr + hi * hi).sum(-1)                       # ρ_k = ‖ĥ(ω_k)‖²  [c]
        c_vec = torch.fft.ifft(rho.to(torch.complex64)).real    # c(δ) = IDFT(ρ)  [c]
        return c_vec / c_vec[0].clamp(min=eps)                  # c_row [c], c_row[0]=1

    @classmethod
    def _analytic_c_from_taps(cls, H, c, eps: float = 1e-12):
        """Full analytic ``Ĉ`` [c,c] (circulant) + ``c_row`` [c] from R-PEARL taps.

        PSD by construction (ρ≥0), O(1) (the c(0) normalization cancels the no-grad
        β=1/F forward rescale), deterministic, and its gradient trains ``H``.
        """
        c_row = cls._analytic_c_row_from_taps(H, c, eps)
        kk = torch.arange(c, device=H.device)
        idx = (kk[:, None] - kk[None, :]) % c                  # (t−u) mod c
        return c_row[idx], c_row                               # Ĉ [c,c] , c_row [c]

    @staticmethod
    def _analytic_c_from_gso(H, gso, c, eps: float = 1e-9):
        """Full symmetric graph-covariance kernel ``Ĉ = H(S)H(S)*`` on the FULL
        composite GSO ``S = [[S_c, B],[Bᵀ, S_sc]]`` (cycle ⊕ crosslinks ⊕ scene),
        returning the token block ``[c, c]``.

        Route (a) — real matrix powers (NOT the bare-cycle DFT). With taps
        ``H`` ``[K+1, F]`` and tap Gram ``G_{kl}=⟨H_k,H_l⟩``::

            Ĉ_full = Σ_{k,l=0}^{K} G_{kl} · Sᵏ (Sˡ)ᵀ            ( = H(S)H(S)* )

        ``S`` is real, so this is real throughout (the directed/complex-spectrum view
        is the same operator — see _analytic_c_from_taps — but the full S is
        non-normal, so its eigenbasis is ill-conditioned; matrix powers are the
        stable realization). Exactly **symmetric** (G is symmetric) and **PSD** by
        construction (Σ_f H^{(f)}(S)H^{(f)}(S)ᵀ). Because the crosslinks make S
        non-circulant, S² routes token→mention→scene→mention→token, so Ĉ couples
        far-apart tokens that mention the same / graph-adjacent scene nodes —
        long-range, graph-structured relative position (vs. the bare-cycle ±K band).
        Gradient flows to ``H`` (G carries grad; S is a fixed graph constant).
        Correlation-normalized to unit diagonal so it is a bounded additive bias.
        """
        K1 = H.shape[0]
        S = (gso.to_dense() if gso.is_sparse else gso).to(torch.float32)
        N = S.shape[0]
        G = H.float() @ H.float().t()                          # [K1,K1] ⟨H_k,H_l⟩ (grad)
        Spow = [torch.eye(N, device=S.device, dtype=S.dtype)]  # S^0..S^K
        for _ in range(1, K1):
            Spow.append(Spow[-1] @ S)
        C_full = sum(G[k, l] * (Spow[k] @ Spow[l].t())
                     for k in range(K1) for l in range(K1))     # H(S)H(S)*  [N,N]
        Ct = C_full[:c, :c]                                    # token block
        d = Ct.diagonal().clamp(min=eps).sqrt()
        return Ct / d[:, None] / d[None, :]                    # unit-diagonal Ĉ [c,c]

    def _analytic_taps(self):
        """The R-PEARL filter's EFFECTIVE taps ``[L·K+1, F]`` from ALL L GCN layers
        (grad-carrying). By the GSP cascade theorem, stacked layers — each a degree-K
        polynomial in the same S — compose into one degree-LK filter whose taps are the
        discrete convolution of the per-layer tap sequences::

            H(S) = Π_ℓ H^(ℓ)(S) = Σ_{k=0}^{LK} S^k H̄_k ,   H̄ = h^(1) * h^(2) * … * h^(L)

        (channel-diagonal reduction: each layer's K+1 lins → [K+1, F] via mean over the
        input dim). Small (≈(LK+1)·F) so it moves to sharded devices cheaply."""
        convs = self.gt_model.pe_model.pe_gcn.convs
        per_layer = [torch.stack([lin.weight.reshape(lin.weight.shape[0], -1).mean(-1)
                                  for lin in conv.lins], dim=0)        # [K+1, F]
                     for conv in convs]
        bar = per_layer[0]                                            # cascade convolution
        for h in per_layer[1:]:
            out = bar.new_zeros(bar.shape[0] + h.shape[0] - 1, bar.shape[1])
            for i in range(bar.shape[0]):
                out[i:i + h.shape[0]] = out[i:i + h.shape[0]] + bar[i:i + 1] * h
            bar = out
        return bar                                                   # [L*K+1, F]

    def _analytic_c_tok(self, c, device):
        """Convenience: full analytic ``Ĉ`` on ``device`` (used by the debug callback)."""
        return self._analytic_c_from_taps(self._analytic_taps().to(device), c)

    # ----- decode-time composite-graph extension ("the brain grows") -------------
    # As each token generates, the composite graph grows: a cycle node + edge, plus a
    # crosslink/clique when an exact node-name token-sequence completes. We cache the
    # per-hop probe aggregates A_k=(Ŝᵏq) and the per-probe linearized R-PEARL embeddings
    # Φ, and compute ONLY the new node's row Ĉ[new,·]=⟨Φ_new,Φ_u⟩/m − Ψ_new·Ψ_u (verified
    # to cos≈0.97 vs full recompute; frozen-old-degree approximation). The new generated
    # token thus gets long-range, graph-structured position to the prompt tokens it (and
    # its mentioned/adjacent scene nodes) relate to. Caches are zero-allocated to max_seq.
    def decode_setup(self, aug, node_token_seqs, c, max_seq, m_dec=16, device=None):
        """Arm the decode-extension state from the prompt composite graph ``aug``."""
        device = device or next(self.parameters()).device
        Hbar = self._analytic_taps().to(device=device, dtype=torch.float32)   # [K1,F]
        K1, F = Hbar.shape
        gso = aug.gso.coalesce()
        idx, val = gso.indices().to(device), gso.values().to(torch.float32).to(device)
        N0 = aug.num_nodes
        Nmax_ = max_seq + aug.num_scene_nodes
        # frozen prompt-node degrees (rowsum of A+I); sized Nmax so generated nodes (which
        # become neighbors of later tokens via the cycle/clique) have a degree slot.
        deg = torch.zeros(Nmax_, device=device)
        deg[:N0] = torch.zeros(N0, device=device).index_add_(0, idx[0], torch.ones_like(val))
        # neighbor lists for the per-hop aggregate recursion
        nbrs = [[] for _ in range(N0)]
        for e in range(idx.shape[1]):
            nbrs[int(idx[0, e])].append((int(idx[1, e]), float(val[e])))
        Nmax = max_seq + aug.num_scene_nodes
        torch.manual_seed(0)                                  # deterministic decode probes
        q = torch.zeros(Nmax, m_dec, device=device)
        q[:N0] = torch.randn(N0, m_dec, device=device)
        # per-hop aggregates A_k = Ŝᵏ q on the prompt graph (dense-apply via the sparse GSO)
        A = torch.zeros(K1, Nmax, m_dec, device=device)
        A[0, :N0] = q[:N0]
        S = torch.sparse_coo_tensor(idx, val, (N0, N0)).coalesce()
        for k in range(1, K1):
            A[k, :N0] = torch.sparse.mm(S, A[k - 1, :N0])
        Phi = torch.zeros(Nmax, m_dec, F, device=device)
        Phi[:N0] = torch.einsum("kf,knm->nmf", Hbar, A[:, :N0])
        Psi = torch.zeros(Nmax, F, device=device)
        Psi[:N0] = Phi[:N0].mean(1)
        # token-node sequence order: cycle positions 0..c-1 (scene rows excluded from keys)
        self._decode_state = dict(
            Hbar=Hbar, A=A, Phi=Phi, Psi=Psi, deg=deg, nbrs=nbrs, m=m_dec, F=F, K1=K1,
            c=c, N0=N0, n_scene=aug.num_scene_nodes, node_token_seqs=node_token_seqs,
            token_nodes=list(range(c)),          # sequence-pos → node index (extends on decode)
            mentions={}, gen_ids=[], next_node=N0, prev_node=c - 1)
        self._pe_decode_row = None

    def decode_extend(self, token_id):
        """Append one generated token: extend the graph, compute its Φ row incrementally,
        and stash the decode bias row ``Ĉ[new, :key_len]`` on ``self._pe_decode_row``."""
        st = self._decode_state
        dev = st["Hbar"].device
        new = st["next_node"]
        st["gen_ids"].append(int(token_id))
        # --- exact node-name token-sequence match (suffix of the generated stream) ---
        v = None
        for node_idx, seq in st["node_token_seqs"]:
            L = len(seq)
            if L and len(st["gen_ids"]) >= L and st["gen_ids"][-L:] == list(seq):
                v = node_idx; break
        # --- new node's neighbors: cycle predecessor + (crosslink scene node + clique) ---
        nb = [(st["prev_node"], 1.0)]
        if v is not None:
            scene_node = st["c"] + v
            nb.append((scene_node, 1.0))
            for p in st["mentions"].get(v, []):           # clique to prior mentions of v
                nb.append((p, 1.0))
            st["mentions"].setdefault(v, []).append(new)
        deg_new = float(len(nb)) + 1.0                     # incident + self-loop
        A, Phi, Psi, deg, Hbar = st["A"], st["Phi"], st["Psi"], st["deg"], st["Hbar"]
        m, K1 = st["m"], st["K1"]
        A[0, new] = torch.randn(m, device=dev)             # fresh probe for the new node
        inv_new = deg_new ** -0.5
        for k in range(1, K1):
            acc = inv_new * inv_new * A[k - 1, new].clone()        # self-loop term
            for (j, w) in nb:
                acc = acc + (w * inv_new * float(deg[j].clamp(min=1).item()) ** -0.5) * A[k - 1, j]
            A[k, new] = acc
        Phi[new] = torch.einsum("kf,km->mf", Hbar, A[:, new])
        Psi[new] = Phi[new].mean(0)
        # degrees: register the new node (frozen-old approximation leaves neighbors as-is)
        if new < deg.shape[0]:
            deg[new] = deg_new
        st["token_nodes"].append(new)
        st["prev_node"] = new
        st["next_node"] = new + 1
        # --- decode bias row over the current key sequence (token nodes only) ---
        cols = torch.tensor(st["token_nodes"], device=dev)
        row = (torch.einsum("mf,umf->u", Phi[new], Phi[cols]) / m
               - Psi[new] @ Psi[cols].t())                  # Ĉ[new, key positions]
        self._pe_decode_row = row

    def decode_disarm(self):
        self._decode_state = None
        self._pe_decode_row = None

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        graphs: Batch | None = None,
        injection_maps: list[dict[int, list[tuple[int, int]]]] | None = None,
        **kwargs,
    ):
        inputs_embeds = self._fuse_embeddings(input_ids, graphs, injection_maps)
        kwargs.pop("inputs_embeds", None)
        kwargs.pop("input_ids", None)
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )


class InjectedCompositeGraphLLM(CompositeGraphLLM):
    """Composite-graph LLM that injects the GT code into q/k/v *inside attention* at
    every layer, in place of RoPE.

    Written in eval_unification's style (per-instance patched attention forward,
    ``_install_pe_injection`` / ``_make_injected_attention_forward`` / ``_pe_signal`` /
    ``_pe_inject_value`` / ``_decoder_layers``, model-agnostic via ``importlib``) so the
    two files read the same — but the *mathematics* is this branch's RoPE-replacement
    design, not e-u's ``RoPE(X)+Ψ``. Per attention layer ``l``::

        q_l = W^q_l · h_l + W_q · S
        k_l = W^k_l · h_l + W_k · S
        v_l = W^v_l · h_l + W_v · S            (if ``inject_v``)

    Differences from e-u, by design:
      - **RoPE is OFF** (``disable_llm_rope=True``, identity rotary): the injected code
        is the sole position signal, present at every depth — there is no native
        rotation on the content q/k.
      - **Dedicated projections** ``W_q``/``W_k``/``W_v`` (shared across layers, the
        "shared/raw" map) carry the code into the q/k/v spaces — *not* the LLM's own
        content projections.
      - The signal is the GT-refined composite second-moment code
        ``S = Y_tok = GT([X;Ψ] + C·[X;Ψ])[V_Tx]`` (un-gated; the M7 gate is on the
        layer-0 blend, below).
      - ``inputs_embeds`` is the **gated GT blend** ``M7(X, Y_tok)`` (the Layer-0
        injection), *then* the same code is re-injected into q/k/v at every layer.

    The code is supplied per forward via ``self._pe_signal`` ([B, seq, hidden]) and,
    matching e-u's plumbing, skipped on cached single-token decode steps (seq mismatch)
    — so only prompt tokens carry it. (With RoPE off, generated tokens then have no
    position; that is the known trade-off of replacing RoPE.)

    The attention forward is patched per-instance (faithful copy of the HF
    implementation); it survives PEFT, which swaps the leaf Linears in place.
    """

    def __init__(self, *args, inject_v: bool = True,
                 disable_llm_rope: bool = True, c_per_layer: bool = False,
                 c_bias: bool = False, c_kernel: str = "sampled",
                 use_scene_bias: bool = True, **kwargs):
        # RoPE OFF by default (disable_llm_rope=True): the injected code replaces RoPE.
        kwargs["disable_llm_rope"] = disable_llm_rope
        super().__init__(*args, **kwargs)

        # ``c_bias`` (Design D, docs/composite_graph_gt_rope_free_*): no RoPE, no rotary,
        # no q/k transform. C enters the attention as an ADDITIVE logit bias λ_C·Ĉ and the
        # values as a residual mix; the R-PEARL first-moment Gram Ψ̃=ΨΨᵀ enters as a second
        # additive bias λ_ψ·Ψ̃. ``c_kernel`` selects the live Ĉ:
        #   "sampled"  : Ĉ = E_q[ΦΦᵀ]−ΨΨᵀ on the full composite graph (probe sampling);
        #   "analytic" : Ĉ = H(S)H(S)* via all-layer cascade taps + matrix powers on S.
        # Selection (⟨q,k⟩) is preserved, so the c_per_layer collapse cannot occur.
        # (``use_scene_bias`` kept for back-compat with older checkpoints; S̃ is removed.)
        self.c_bias = c_bias
        self.c_kernel = c_kernel

        # ``c_per_layer``: instead of the additive code S=Y_tok in q/k/v, REPLACE the
        # post-RoPE query/key at every layer with the composite covariance operator
        # ``C_tok`` mixing across the sequence: ``q ← C_tok·q``, ``k ← C_tok·k`` (page-9
        # proof: the token block of C is the relative operator ``c(n-m)`` — RoPE made
        # literal in the q·k score, at every depth). C_tok is deterministic (no params)
        # and scaled to ‖X‖, so nothing extra is saved/loaded; the dedicated q/k/v
        # projections below are not created in this mode.
        self.c_per_layer = c_per_layer

        cfg = self.llm.config
        hidden = cfg.hidden_size
        n_heads = cfg.num_attention_heads
        n_kv = getattr(cfg, "num_key_value_heads", n_heads)
        head_dim = getattr(cfg, "head_dim", None) or (hidden // n_heads)
        try:
            device = next(self.llm.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        if not c_per_layer and not c_bias:
            # Dedicated shared projections from the d_model code into the q / k (/ v) spaces
            # (one set, reused at every layer). Default Linear init (std ≈ 1/√d_model) maps a
            # code of row-norm ≈‖X‖ to a per-element std comparable to the attention logits'
            # scale. No bias (a constant shift across positions carries no order). q gets
            # H·Dh, k/v get Hkv·Dh (GQA).
            self.pe_q_proj = nn.Linear(hidden, n_heads * head_dim, bias=False, device=device)
            self.pe_k_proj = nn.Linear(hidden, n_kv * head_dim, bias=False, device=device)
            if inject_v:
                self.pe_v_proj = nn.Linear(hidden, n_kv * head_dim, bias=False, device=device)

        if c_bias:
            # Design D gains (scalar, learnable). λ_C init 1.0 so the covariance bias is
            # ON from step 0; λ_ψ init 0.1 (R-PEARL first-moment Gram Ψ̃ enters the logits
            # — through R-PEARL, thesis-consistent); λ_V init 0.1 (residual value mix).
            # Saved in gnn_weights.pt.
            self.lam_c = nn.Parameter(torch.tensor(1.0, device=device))
            self.lam_psi = nn.Parameter(torch.tensor(0.1, device=device))
            self.lam_v = nn.Parameter(torch.tensor(0.1, device=device))
            # λ_C warmup ramp ∈ [0,1], set each training step by LamCWarmupCallback.
            # Non-persistent: defaults to 1.0 on load so eval/inference applies full λ_C.
            self.register_buffer("_lam_c_warmup", torch.tensor(1.0, device=device),
                                 persistent=False)
            # Additive bias needs a backend that adds a float `attention_mask` to the
            # scores before softmax. BOTH eager and sdpa do; only flash-attn can't. Use
            # SDPA (fused, memory-efficient) — eager materializes the full [B,H,c,c] score
            # tensor (~8.6 GB/layer at c=8192) and softmaxes in Python, which is the
            # c_bias-specific slowdown. SDPA falls back to its mem-efficient/math kernel
            # when given a custom additive mask, but never materializes all heads at once.
            self.llm.config._attn_implementation = "sdpa"  # ty: ignore[invalid-assignment]

        # Is the LLM sharded across >1 device (device_map="auto")? If so, c_bias arms the
        # small grad-carrying TAPS and each layer recomputes Ĉ on its own device, so the
        # R-PEARL taps stay FULLY trainable and only the tiny [K+1,F] tensor crosses device
        # streams (not the [c,c] kernel). On one GPU the kernel is precomputed once.
        self._llm_sharded = False
        dm = getattr(self.llm, "hf_device_map", None)
        if dm:
            real = {str(d) for d in dm.values() if d not in ("cpu", "disk", -1)}
            self._llm_sharded = len(real) > 1
        if c_bias and self._llm_sharded:
            # The taps + the 3 scalar gains λ_C/λ_S/λ_V live on one device but are applied
            # in every (sharded) layer, so their grads accumulate across streams — benign
            # (tiny tensors, negligible sync) and they MUST keep their gradient. Silence the
            # now-expected stream-mismatch warning (the costly [c,c] mismatch is gone).
            _setw = getattr(getattr(torch.autograd, "graph", None),
                            "set_warn_on_accumulate_grad_stream_mismatch", None)
            if _setw is not None:
                _setw(False)

        # Per-forward signals read by the patched attention forwards, set by
        # forward()/prepare_generation():
        #   _pe_signal    : additive code S=Y_tok ([B, seq, hidden]) — additive variant.
        #   _pe_C         : composite token covariance Ĉ ([B, seq, seq]) — c_per_layer / c_bias.
        #   _pe_Psi       : R-PEARL first-moment Gram Ψ̃=ΨΨᵀ ([B, seq, seq]) — c_bias.
        #   _pe_c_row     : analytic relative row c(·) ([B, seq]) for decode — c_bias.
        self._pe_signal = None
        self._pe_C = None
        self._pe_Psi = None
        self._pe_c_row = None
        self._pe_taps = None      # c_bias multi-GPU: grad-carrying taps, Ĉ recomputed per layer
        self._pe_cyc = None       # cycle length c for the analytic kernel
        self._decode_state = None # decode-time graph-extension state (set by decode_setup)
        self._pe_decode_row = None  # live decode bias row Ĉ[new, :key_len] (set per token)
        self._pe_inject_value = inject_v
        self._install_pe_injection()

    # ----- attention patch (ported from eval_unification, model-agnostic) -------
    def _decoder_layers(self):
        """Return the LLM's decoder layer list (Llama/Qwen2: ``<CausalLM>.model.layers``)."""
        base = getattr(self.llm, "model", None)
        if base is not None and hasattr(base, "layers"):
            return base.layers
        return self.llm.get_decoder().layers

    def _install_pe_injection(self):
        """Patch every self-attention forward so S is added post-RoPE in q/k/(v)."""
        for layer in self._decoder_layers():
            attn = layer.self_attn
            attn.forward = self._make_injected_attention_forward(attn)

    def _make_injected_attention_forward(self, attn):
        # Resolve the model-family helpers from the module that defines this attention
        # class (Llama, Qwen2, … share this layout).
        mod = importlib.import_module(type(attn).__module__)
        apply_rotary_pos_emb = mod.apply_rotary_pos_emb
        attn_fns = mod.ALL_ATTENTION_FUNCTIONS
        eager = mod.eager_attention_forward
        model = self  # captured for the per-forward signal

        def forward(hidden_states, position_embeddings=None, attention_mask=None,
                    past_key_values=None, **kwargs):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, attn.head_dim)

            query_states = attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            # --- replace RoPE: add the GT code through dedicated projections at every
            # layer. RoPE is off (identity rotary), so the rotation above is a no-op and
            # the code is the sole position signal. ---
            psi = model._pe_signal
            if (psi is not None and psi.shape[0] == hidden_states.shape[0]
                    and psi.shape[1] == hidden_states.shape[1]):
                def _proj(linear):
                    # Run on the projection's own device, then move to the (possibly
                    # sharded) q/k/v device — .to(dtype) alone would strand it on the
                    # init device under device_map=auto.
                    w = linear.weight
                    out = linear(psi.to(device=w.device, dtype=w.dtype))
                    out = out.to(device=query_states.device, dtype=query_states.dtype)
                    return out.view(hidden_shape).transpose(1, 2)
                query_states = query_states + _proj(model.pe_q_proj)
                key_states = key_states + _proj(model.pe_k_proj)
                if model._pe_inject_value:
                    value_states = value_states + _proj(model.pe_v_proj)
            # --- Design D (c_bias): per-layer analytic kernel Ĉ [c,c] / c_row [c] on THIS
            # layer's device. Single-GPU: reuse the precomputed _pe_C. Multi-GPU: recompute
            # from the small grad-carrying taps so the taps stay FULLY trainable while only
            # the tiny [K+1,F] tensor crosses device streams (not the [c,c] kernel). ---
            cb_C = cb_crow = None
            if model.c_bias:
                qdev = query_states.device
                seq = hidden_states.shape[1]
                if model._pe_taps is not None:                  # multi-GPU: recompute here
                    taps = model._pe_taps.to(qdev)
                    if seq == model._pe_cyc:
                        cb_C, cb_crow = model._analytic_c_from_taps(taps, model._pe_cyc)
                    elif seq == 1:
                        cb_crow = model._analytic_c_row_from_taps(taps, model._pe_cyc)
                else:                                           # single-GPU: precomputed
                    if model._pe_C is not None and model._pe_C.shape[1] == seq:
                        cb_C = model._pe_C[0].to(device=qdev, dtype=torch.float32)
                    if model._pe_c_row is not None:
                        cb_crow = model._pe_c_row[0].to(device=qdev)
            # residual Ĉ-value mix on the PROMPT (mixed V is then cached); q/k untouched.
            if model.c_bias and cb_C is not None:
                mixed = torch.einsum("nm,bhmd->bhnd", cb_C.to(value_states.device),
                                     value_states.float())
                # renorm the mixed values back to ‖v‖ (Ĉ is O(1)); λ_V tunes strength.
                mn = mixed.norm(dim=-1).mean().clamp(min=1e-12)
                vn = value_states.float().norm(dim=-1).mean()
                mixed = mixed * (vn / mn)
                lam_v = model.lam_v.to(device=value_states.device, dtype=value_states.dtype)
                value_states = value_states + lam_v * mixed.to(value_states.dtype)
            # --- c_per_layer: REPLACE q/k with the composite covariance mixing the
            # sequence, q ← C_tok·q, k ← C_tok·k (the proof's relative c(n-m) in the
            # score, every layer). Skipped on cached single-token decode (seq mismatch)
            # — only prompt tokens carry it, like the additive path. Value is content. ---
            C = model._pe_C
            if (not model.c_bias and C is not None and C.shape[0] == hidden_states.shape[0]
                    and C.shape[1] == hidden_states.shape[1]):
                # C_tok is built once on the embedding device; under device_map="auto"
                # each decoder layer can live on a different GPU, so move C onto this
                # layer's device (q/k) before the einsum.
                Cf = C.to(device=query_states.device, dtype=torch.float32)

                def _mix(t):  # t: [B, H, seq, head_dim]
                    # fp32 einsum: C_tok carries fine relative structure (the c(n-m)
                    # decay) that bf16 would crush before the renorm; the GT runs fp32
                    # anyway. Cast back to t's dtype at the end.
                    out = torch.einsum("bnm,bhmd->bhnd", Cf.to(t.device), t.float())
                    # RoPE-like scale stability: RoPE preserves ‖q‖ exactly; C_tok is a
                    # PSD covariance scaled to ‖X‖ at the layer-0 manifold, but Llama's
                    # residual-stream norm grows with depth, so rescale C_tok·t back to
                    # t's current mean row-norm. Keeps q/k — hence the attention logits —
                    # proportionate to the content at EVERY layer (the global scalar
                    # preserves C's per-position redistribution, the relative signal).
                    cur = out.norm(dim=-1).mean().clamp(min=1e-12)
                    tgt = t.float().norm(dim=-1).mean()
                    return (out * (tgt / cur)).to(t.dtype)

                query_states = _mix(query_states)
                key_states = _mix(key_states)
            # ------------------------------------------------------------------

            if past_key_values is not None:
                key_states, value_states = past_key_values.update(
                    key_states, value_states, attn.layer_idx)

            # --- Design D (c_bias): fold the additive λ_C·Ĉ + λ_ψ·Ψ̃ position+graph bias
            # into the attention_mask (eager/sdpa add it to the logits before softmax).
            # Built post-cache so the decode row spans all cached keys; ⟨q,k⟩ untouched. ---
            if model.c_bias:
                key_len = key_states.shape[2]
                dev = query_states.device
                # gains follow this layer's (possibly sharded) device; .float() alone
                # keeps them on the init device → mismatch under device_map=auto.
                lam_c = (model.lam_c * model._lam_c_warmup).to(device=dev, dtype=torch.float32)
                lam_psi = model.lam_psi.to(device=dev, dtype=torch.float32)
                bias = None
                if cb_C is not None:
                    # PROMPT: λ_C·Ĉ + λ_ψ·Ψ̃ [c,c] → [B,1,seq,seq]
                    b = lam_c * cb_C.to(dev)[None]                          # [1,c,c]
                    Psi = model._pe_Psi
                    if Psi is not None:
                        b = b + lam_psi * Psi[0].to(device=dev, dtype=torch.float32)[None]
                    bias = b.unsqueeze(1)
                elif query_states.shape[2] == 1 and model._pe_decode_row is not None:
                    # DECODE (graph-extended): the live row Ĉ[new, :key_len] from the
                    # composite graph GROWN over generated tokens (decode_extend). Pad/clip
                    # to key_len (the new token attends all cached keys).
                    row = model._pe_decode_row.to(device=dev, dtype=torch.float32)
                    if row.shape[0] < key_len:
                        row = torch.cat([row, row.new_zeros(key_len - row.shape[0])])
                    bias = (lam_c * row[:key_len])[None, None, None, :]
                elif cb_crow is not None and query_states.shape[2] == 1:
                    # DECODE fallback (no decode-extension armed): analytic c(·), offset mod c.
                    crow = cb_crow.to(dev)                                  # [c]
                    cyc = crow.shape[0]
                    off = (key_len - 1 - torch.arange(key_len, device=dev)) % cyc
                    bias = (lam_c * crow[off])[None, None, None, :]
                if bias is not None:
                    bias = bias.to(dtype=query_states.dtype)
                    if attention_mask is None:
                        # SDPA passes no mask and would apply is_causal internally; supplying
                        # an explicit mask disables that, so fold the causal triangle into the
                        # bias (prompt only — a single decode query attends all cached keys).
                        q_len = query_states.shape[2]
                        if q_len > 1:
                            causal = torch.triu(
                                torch.full((q_len, key_len), float("-inf"), device=dev,
                                           dtype=bias.dtype), diagonal=key_len - q_len + 1)
                            bias = bias + causal[None, None]
                        attention_mask = bias
                    else:
                        am = attention_mask[..., :key_len].to(device=dev, dtype=bias.dtype)
                        attention_mask = am + bias

            attn_impl = attn.config._attn_implementation
            attention_interface = (
                eager if attn_impl == "eager" else attn_fns[attn_impl]
            )
            attn_output, attn_weights = attention_interface(
                attn, query_states, key_states, value_states, attention_mask,
                dropout=0.0 if not attn.training else attn.attention_dropout,
                scaling=attn.scaling, **kwargs,
            )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = attn.o_proj(attn_output)
            return attn_output, attn_weights

        return forward

    # ----- signal / forward / generation ---------------------------------------
    def _arm_signals(self, input_ids, graphs, injection_maps, permutation=None):
        """Build ``inputs_embeds`` and arm the per-forward attention signal(s).

        ``inputs_embeds`` is the **gated GT blend** ``M7(X, Y_tok)`` (the Layer-0
        injection). Then, per mode, the patched attention is armed:

          - additive (default): ``self._pe_signal = Y_tok`` is re-injected into
            q/k/v at every layer;
          - ``c_per_layer``: ``self._pe_C = C_tok`` REPLACES q/k at every layer
            (``q ← C_tok·q``), the proof's relative ``c(n-m)`` in the score.

        The unused signal is cleared so a stale one can't leak across modes.
        """
        if self.c_bias:
            # Design D: additive λ_C·Ĉ + λ_ψ·Ψ̃ logit bias + residual λ_V·Ĉ value mix; no
            # q/k transform. Ĉ (covariance, live per self.c_kernel) and Ψ̃=ΨΨᵀ are the
            # per-sequence [B,c,c] kernels from the FULL composite graph S=[[S_c,B],[Bᵀ,S_sc]].
            B, c = input_ids.shape[0], input_ids.shape[1]
            self._pe_signal = None
            self._pe_cyc = c
            inputs_embeds, c_hat, psi_t = self._fuse_embeddings(
                input_ids, graphs, injection_maps, permutation=permutation,
                return_cbias=True)
            self._pe_C = c_hat                                   # [B, c, c]
            self._pe_Psi = psi_t                                 # [B, c, c]
            self._pe_taps = None
            # Decode fallback row (analytic c(·)); the graph-extension cache supersedes it
            # once decode-time extension is armed (see _decode_state).
            self._pe_c_row = self._analytic_c_row_from_taps(
                self._analytic_taps().to(input_ids.device), c).unsqueeze(0).expand(B, -1)
        elif self.c_per_layer:
            inputs_embeds, c_tok = self._fuse_embeddings(
                input_ids, graphs, injection_maps, permutation=permutation,
                return_c_tok=True)
            self._pe_C = c_tok
            self._pe_Psi = None
            self._pe_c_row = None
            self._pe_signal = None
        else:
            inputs_embeds, token_pe = self._fuse_embeddings(
                input_ids, graphs, injection_maps, permutation=permutation,
                return_token_pe=True)
            self._pe_signal = token_pe
            self._pe_C = None
            self._pe_Psi = None
            self._pe_c_row = None
        return inputs_embeds

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                graphs=None, injection_maps=None, **kwargs):
        inputs_embeds = self._arm_signals(input_ids, graphs, injection_maps)
        kwargs.pop("inputs_embeds", None)
        kwargs.pop("input_ids", None)
        try:
            return self.llm(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                labels=labels, **kwargs)
        finally:
            # Disarm the signals unless gradient checkpointing will recompute the
            # attention forwards in backward (they must see the same signal; it is
            # rebuilt every forward anyway). Mirrors eval_unification.
            if not getattr(self.llm, "is_gradient_checkpointing", False):
                self._pe_signal = None
                self._pe_C = None
                self._pe_Psi = None
                self._pe_c_row = None
                self._pe_taps = None

    def prepare_generation(self, input_ids, graphs, injection_maps, permutation=None):
        """Arm the in-attention injection and return ``inputs_embeds``.

        The caller hands the returned embeds to ``self.llm.generate`` and must reset
        ``self._pe_signal``/``self._pe_C`` to None afterwards. Injection auto-skips
        cached decode steps, so only prompt tokens carry the signal.
        """
        return self._arm_signals(
            input_ids, graphs, injection_maps, permutation=permutation)


def find_last_graph_scope(input_ids_b, tokenizer) -> int:
    """Token index where the last scene-graph block begins (injection scope).

    Only mentions at/after this index belong to the last (query) graph; earlier
    matches live inside ICL-example graphs and must be ignored so the query
    graph's labels don't cross-link into ICL regions. Spec R10 locks this:
    "infra inputs only the last (query) graph and scopes injection after it
    completes."

    Robust text match (vs. token-subsequence matching, which silently failed on
    this corpus: ``:`` merges with the following character so the encoded
    ``"scene graph:"`` marker never appeared as a token subsequence). We decode
    per-token, anchor on the COMPACT block signature ``scene graph:`` immediately
    followed by the bullet ``•`` — a prose ``"the scene graph"`` mention inside
    the model's reasoning is NOT followed by a bullet, so it can't move the
    scope past the node lists — and map the last match's char offset back to its
    token index. Returns 0 (whole sequence eligible) when no compact block is
    present, e.g. the verbose JSON prompt — unchanged behavior for that path.

    Used by BOTH the training collator (``SpineDataCollator``) and eval
    (``GraphAugmentedInMemoryLLM``) so the composite graph is assembled with the
    same scope in train and eval.
    """
    seq = list(map(int, input_ids_b))
    # Per-token decode; offsets are computed from the SAME pieces we search, so
    # the char->token mapping is self-consistent regardless of BPE quirks.
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

    A node name tokenizes differently with a leading space (the space merges
    into the first sub-word, changing its id) than standalone. In the compact
    block a name appears space-preceded in the ``• … nodes:``/comma lists and
    standalone right after ``[`` in the edge lists, so we match BOTH forms; this
    is what gets injection to 100% (every node has at least its space-preceded
    list mention bound). Feed the result straight to :func:`build_injection_map`.
    """
    return [
        [
            tokenizer.encode(name, add_special_tokens=False),
            tokenizer.encode(" " + name, add_special_tokens=False),
        ]
        for name in node_names
    ]


def has_match(input_ids_b: list[int], to_match:list[int],start_pos:int):
    """ 
    For a single sequence, check if `to_match` is present at `start_pos`
    """
    end_pos = min(start_pos + len(to_match),len(input_ids_b))
    return input_ids_b[start_pos:end_pos] == to_match

def build_injection_map(
    input_ids_b: list[int],
    node_token_seqs: list,
    scope_start: int = 0,
) -> dict[int, list[tuple[int, int]]]:
    """Build a pre-computed injection map from token IDs and node token sequences.

    Returns the ``{node_idx: [(start, end), ...]}`` format expected by
    ``GraphAugmentedLLM._augment_embeddings``.

    ``node_token_seqs[i]`` is EITHER a single token-ID list (one tokenization of
    the node name — back-compatible) OR a list of candidate token-ID lists
    (multiple tokenizations, e.g. standalone + space-preceded from
    :func:`node_token_variants`). Matching any candidate binds the node, which is
    what reaches 100% coverage: a name tokenizes differently after ``", "`` than
    after ``"["``, and a single encoding misses one of those forms.

    Two refinements (M3):

    - **Scope (``scope_start``):** only matches starting at/after ``scope_start``
      are kept, so labels that also appear in earlier ICL-example graphs are
      ignored and PE lands only on the last (query) graph block.
    - **Longest-first matching:** spans are resolved longest-first and claim the
      token positions they cover, so a label that is a token prefix of a longer
      one (``barn_shed_1`` inside ``barn_shed_11``), or a shorter variant, can't
      steal a longer match's tokens.

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


def bucketize_prompt(input_ids_b: list, node_token_seqs : list) -> defaultdict:
    """
    Helper function for associating node token sequences with their positions
    in a tokenized prompt. Uses parallel iteration through the token list.

    Args:
        input_ids_b (list): Flat list of token IDs for a single sequence.
        node_token_seqs (list): Per-node list of token-ID subsequences.

    Returns:
        buckets (defaultdict[int, set]): Mapping from node index to the set of
            start positions where that node's token sequence appears.
    """
    # Get map of words to token locations.
    buckets = defaultdict(set)
    for p_idx, p_token in enumerate(input_ids_b):
        for node_idx, node_token_seq in enumerate(node_token_seqs):
            if has_match(input_ids_b, to_match=node_token_seq,start_pos=p_idx):
                buckets[node_idx].add(p_idx)
    return buckets
