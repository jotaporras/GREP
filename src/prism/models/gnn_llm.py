import copy
import importlib
import math
import re
from collections import defaultdict

import torch
from torch import nn
from torch_geometric.data import Batch, Data
import transformers.masking_utils as masking_utils
from transformers import AttentionInterface, PreTrainedModel

from prism.models.composite_graph import build_composite_graph
from prism.models.llama import disable_rope


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
    bias = None if model is None else model._struct_bias
    q_len = query.shape[-2]
    k_len = key.shape[-2]
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
            am = attention_mask[..., :k_len].to(device=query.device, dtype=query.dtype)
            attention_mask = am + bias
    return module._graph_mask_orig_attn_fn(
        module, query, key, value, attention_mask,
        scaling=scaling, dropout=dropout, **kwargs)


AttentionInterface.register(_GRAPH_MASK_IMPL, _graph_mask_attention_forward)


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
                 use_edges: bool = True):
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
        # Per-forward additive attention bias [B, 1, seq, seq]; read by the patched
        # attention layers, set in forward / inference, disarmed afterwards.
        self._struct_bias: torch.Tensor | None = None
        self._install_graph_mask()

    def structural_parameters(self) -> list[nn.Parameter]:
        """Parameter-free architecture: no graph params (only the LLM/LoRA train)."""
        return []

    def _decoder_layers(self):
        """Return the LLM's decoder layer list (Llama/Qwen2: ``<CausalLM>.model.layers``)."""
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

        for layer in layers:
            attn = layer.self_attn
            # Bypass nn.Module.__setattr__ to avoid registering a submodule cycle (attn→wrapper→llm→attn).
            object.__setattr__(attn, "_graph_mask_model", self)
            attn._graph_mask_orig_attn_fn = orig_attn_fn
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

    def _node_adjacency(self, g, device) -> torch.Tensor:
        """Boolean ``[N, N]`` node adjacency — True where two nodes may attend.

        Built from ``edge_index`` with self-loops (a node always sees itself and its
        own repeated mentions), optional symmetrization, and ``(A+I)^k`` reachability
        for ``k_hops > 1``. Mirrors the undirected adjacency the GNN/GT consume.
        """
        N = g.num_nodes
        adj = torch.zeros(N, N, dtype=torch.bool, device=device)
        ei = getattr(g, "edge_index", None)
        # use_edges=False ⇒ edgeless ablation: skip edge_index entirely, leaving only
        # the self-loops below, so all node↔(other-node) attention is blocked.
        if self._mask_use_edges and ei is not None and ei.numel() > 0:
            ei = ei.to(device)
            adj[ei[0], ei[1]] = True
            if self._mask_symmetrize:
                adj = adj | adj.t()
        adj.fill_diagonal_(True)  # self-loops: same node (and its repeats) always visible
        if self._mask_use_edges and self._mask_k_hops > 1:
            reach = adj.clone()
            f_adj = adj.float()
            power = adj.clone()
            for _ in range(self._mask_k_hops - 1):
                power = (power.float() @ f_adj) > 0
                reach = reach | power
            adj = reach
        return adj

    def build_structural_mask(self, seq_len, graphs, injection_maps, device, dtype=None):
        """Additive attention bias ``[B, 1, seq, seq]`` — 0 allowed, ``finfo.min`` blocked.

        ``bias[b,0,i,j] = finfo.min`` iff tokens i and j BOTH belong to graph nodes
        AND those nodes are non-adjacent (within ``k_hops``). Every other entry
        (node↔non-node, non-node↔non-node, same node, adjacent) stays 0. Because it
        is ADDED to the model's causal/sliding mask, blocking only ever removes
        already-causal pairs. Each node-token row keeps BOS (a non-node) and its own
        diagonal, so no row is fully masked (no softmax NaN).
        """
        if dtype is None:
            dtype = self.llm.get_input_embeddings().weight.dtype
        B = len(injection_maps)
        neg = torch.finfo(dtype).min
        bias = torch.zeros(B, 1, seq_len, seq_len, device=device, dtype=dtype)
        for b in range(B):
            g = graphs[b]
            # token position -> node id (-1 for non-node tokens). Spans are disjoint
            # (build_injection_map dedups longest-first), so each token maps to one node.
            tok2node = torch.full((seq_len,), -1, dtype=torch.long, device=device)
            for node_idx, spans in injection_maps[b].items():
                for start, end in spans:
                    end = min(end, seq_len)
                    if start < end:
                        tok2node[start:end] = node_idx
            node_pos = (tok2node >= 0).nonzero(as_tuple=True)[0]
            if node_pos.numel() == 0:
                continue
            adj = self._node_adjacency(g, device)        # [N, N] bool
            nid = tok2node[node_pos]                      # node id for each node-token
            allowed = adj[nid][:, nid]                    # [P, P] bool over node-token pairs
            blocked = ~allowed
            if blocked.any():
                bi, bj = blocked.nonzero(as_tuple=True)
                bias[b, 0, node_pos[bi], node_pos[bj]] = neg
        return bias

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        graphs: Batch | None = None,
        injection_maps: list[dict[int, list[tuple[int, int]]]] | None = None,
        **kwargs,
    ):
        kwargs.pop("inputs_embeds", None)
        kwargs.pop("input_ids", None)
        # Arm the structural bias for the patched attention layers. No graph (e.g. a
        # non-graph batch) ⇒ plain causal LLM.
        if graphs is not None and injection_maps is not None and input_ids is not None:
            self._struct_bias = self.build_structural_mask(
                input_ids.shape[1], graphs, injection_maps, input_ids.device)
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


class GraphAugmentedLLM(PreTrainedModel):  # ty:ignore[unsupported-base]
    """Graph-Augmented LLM: injects graph PE Ψ post-RoPE into q/k/v at every layer.

    Receives pre-computed injection maps; pe_model: ``forward(data) → Tensor[n, d_model]``.

    Injection scheme — ``RoPE(X) + Ψ`` at every layer via the ``"prism_pe"`` custom
    attention impl; Ψ added to the already-rotated q/k/v so it is unrotated in the score::

        q = RoPE(W_q · h) + W_q · Ψ
        k = RoPE(W_k · h) + W_k · Ψ
        v =      W_v · h  + W_v · Ψ

    Ψ projected through each layer's own (LoRA-adapted) q/k/v_proj. Architecture-agnostic
    (Llama, Qwen2, gemma-4). ``self._pe_signal`` [B, seq, hidden]; injection skipped on
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
        """Return the LLM's decoder layer list (Llama/Qwen2: ``<CausalLM>.model.layers``)."""
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
                pe = self.pe_norm(pe)
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


class GatedInjection(nn.Module):
    """Cold-start gate blending token embeddings with the Graph Transformer output.

    Produces the LLM ``inputs_embeds`` from the original token embeddings ``X``
    and the Graph Transformer token-node outputs ``Y[V_Tx]``, through a single
    learnable gate that starts at ≈0 (cold-start). Scene-node outputs are
    discarded; only the token rows reach the LLM.

    Modes:
      - ``"interpolate"`` (default): ``inputs_embeds = (1 - gate) * X + gate * Y``.
        The gate is the fraction by which the LLM's own embeddings are replaced
        by the structural output — at ``gate=0`` the LLM sees clean Llama
        embeddings, and structure ramps in as the gate grows.
      - ``"additive"``: ``inputs_embeds = X + gate * Y``.
      - ``"none"``: ``inputs_embeds = Y`` — GT output fed straight; no gate, no X mixing.
        ``gate`` is a fixed 1.0 buffer (non-trainable) so gate diagnostics still read.

    No RoPE or positional transform is applied here (the LLM runs with RoPE disabled).

    Args:
        d_model (int): Embedding width (only used for ``gate_per_dim``).
        gate_init (float): Initial gate value (≈0 cold-start). Ignored when
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
            # No learnable gate: fix gate=1.0 buffer (not a Parameter) so gate diagnostics still work.
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
    """Composite-graph model: Graph Transformer (fusion) → cold-start gate → RoPE-disabled LLM.

    For each sequence: builds composite graph G (directed cycle over c token positions +
    scene graph + cross-links), runs GT over G with token embeddings X on cycle nodes,
    blends token-row outputs Y[V_Tx] with X through the cold-start gate, and feeds
    ``inputs_embeds`` to the RoPE-disabled LLM. Scene-node outputs are discarded.

    Args:
        llm: causal LLM (RoPE disabled unless disable_llm_rope=False).
        gt_model: GraphTransformer; ``forward(data, token_embeddings, is_token) → Y [N, d]``.
        d_model: embedding / GT width (must equal LLM hidden size).
        gate_init, gate_per_dim, injection_mode: cold-start gate settings.
        disable_llm_rope: disable RoPE on llm (default True).
        cycle_weight, cycle_directed, crosslink_*: composite-graph edge settings.
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

    def structural_parameters(self) -> list[nn.Parameter]:
        """Graph-side parameters: the GraphTransformer and the gated injection."""
        return list(self.gt_model.parameters()) + list(self.injection.parameters())

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
        """Assemble the composite graph G for one sequence on ``device``.

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
        """Assemble composite graph → GT → gate; returns ``inputs_embeds`` [B, c, d].

        Optional extras (stacked across batch):
        - ``return_token_pe``: un-gated GT token-row outputs Y[V_Tx] [B, c, d].
        - ``return_c_tok``: composite token covariance C_tok [B, c, c], scaled to ‖X‖.
        - ``return_cbias``: (Ĉ, Ψ̃=ΨΨᵀ) [B, c, c] additive-bias kernels (c_bias mode).
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
            # Graph Transformer (fusion): token embeddings on V_Tx, zeros on V_Sc, fused with R-PEARL Ψ.
            Y = self.gt_model(aug_data, token_embeddings=X[b], is_token=aug.is_token)
            Ytok = Y[aug.is_token]
            # Cold-start gate: blend X with the token-node outputs Y[V_Tx].
            fused.append(self.injection(X[b], Ytok))
            if return_token_pe:
                token_pe.append(Ytok)
            if return_c_tok:
                c_tok.append(self._compute_c_tok(aug, aug_data, X[b]))
            if return_cbias:
                C_b, P_b = self._kernel_and_psi(aug, aug_data, c, device)
                cb_C.append(C_b)
                cb_P.append(P_b)
        # GT runs fp32 (sparse sampled_addmm is fp32-only), gate promotes bf16 X to fp32;
        # cast back to LLM dtype or generate() raises "expected Float but found BFloat16".
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
        """Correlation-normalize a PSD [c,c] kernel to unit diagonal.

        Fails loud on non-finite input (upstream GT/R-PEARL divergence). Clamps diagonal ≥ 0,
        floors variance relative to mean (avoids 1/√0 for near-zero rows), clamps entries
        to [-1, 1] so the additive bias is bounded.
        """
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
        """Per-sequence [c,c] additive-bias kernels ``(Ĉ, Ψ̃)`` for c_bias mode, unit-diagonal.

        - ``"sampled"``: Ĉ = E_q[ΦΦᵀ]−ΨΨᵀ and Ψ̃ = ΨΨᵀ from one probe pass on the full graph.
        - ``"analytic"``: Ĉ = H(S)H(S)* via cascade taps + matrix powers on S=[[S_c,B],[Bᵀ,S_sc]].
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
        """Composite token-block covariance C_tok [c, c] for c_per_layer.

        C_tok = C[V_Tx, V_Tx] from R-PEARL second-moment applied to token-row identity.
        Includes token→scene→token paths. Scaled so C_tok·X matches X's mean row-norm.
        Deterministic (no learnable params).
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
        """Analytic relative-position row c_row[δ] = c(δ)/c(0) from R-PEARL taps H [K+1, F].

        For the directed circulant cycle S_c: ĥ(ω_k)=Σ_j H_j ω_k^j, ρ_k=‖ĥ(ω_k)‖²,
        c(δ)=IDFT(ρ), normalized by c(0). Returns [c] row; gradient flows to H.
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
        """Full analytic Ĉ [c,c] (circulant) + c_row [c] from taps. PSD (ρ≥0), gradient to H."""
        c_row = cls._analytic_c_row_from_taps(H, c, eps)
        kk = torch.arange(c, device=H.device)
        idx = (kk[:, None] - kk[None, :]) % c                  # (t−u) mod c
        return c_row[idx], c_row                               # Ĉ [c,c] , c_row [c]

    @staticmethod
    def _analytic_c_from_gso(H, gso, c, eps: float = 1e-9):
        """Graph-covariance kernel Ĉ = H(S)H(S)* on the full composite GSO S = [[S_c,B],[Bᵀ,S_sc]].

        Ĉ_full = Σ_{k,l} G_{kl}·Sᵏ(Sˡ)ᵀ, where G_{kl}=⟨H_k,H_l⟩; returns token block [c,c].
        Symmetric and PSD by construction; non-circulant S couples far-apart tokens that share
        graph-adjacent scene nodes. Correlation-normalized to unit diagonal. Gradient flows to H.
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
        """Effective cascade taps H̄ [L·K+1, F] from all L GCN layers (grad-carrying).

        Stacked layers compose into H(S) = Σ_k S^k H̄_k via discrete convolution
        H̄ = h^(1) * … * h^(L). Channel-diagonal: each layer's K+1 lins → [K+1, F].
        """
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
    # Each generated token adds a cycle node; exact node-name match adds a crosslink/clique.
    # Incremental update: cache A_k=(Ŝᵏq) and Φ, compute only the new row
    # Ĉ[new,·]=⟨Φ_new,Φ_u⟩/m − Ψ_new·Ψ_u (frozen-old-degree approximation, cos≈0.97).
    def decode_setup(self, aug, node_token_seqs, c, max_seq, m_dec=16, device=None):
        """Arm the decode-extension state from the prompt composite graph ``aug``."""
        device = device or next(self.parameters()).device
        Hbar = self._analytic_taps().to(device=device, dtype=torch.float32)   # [K1,F]
        K1, F = Hbar.shape
        gso = aug.gso.coalesce()
        idx, val = gso.indices().to(device), gso.values().to(torch.float32).to(device)
        N0 = aug.num_nodes
        Nmax_ = max_seq + aug.num_scene_nodes
        # prompt-node degrees (A+I rowsum); Nmax-sized to include generated nodes.
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
    """Composite-graph LLM that injects the GT code into q/k/v inside attention at every layer.

    Per attention layer ``l``::

        q_l = W^q_l · h_l + W_q · S
        k_l = W^k_l · h_l + W_k · S
        v_l = W^v_l · h_l + W_v · S    (if inject_v)

    - RoPE is OFF (identity rotary): S = Y_tok is the sole position signal.
    - Dedicated shared projections W_q/W_k/W_v (not the LLM's own) inject S at every layer.
    - ``inputs_embeds`` is the gated GT blend gate(X, Y_tok) (Layer-0 injection).
    - ``self._pe_signal`` [B, seq, hidden]; skipped on cached decode steps (seq mismatch).
    - Attention forward patched per-instance; survives PEFT.
    """

    def __init__(self, *args, inject_v: bool = True,
                 disable_llm_rope: bool = True, c_per_layer: bool = False,
                 c_bias: bool = False, c_kernel: str = "sampled",
                 use_scene_bias: bool = True, **kwargs):
        # RoPE OFF by default (disable_llm_rope=True): the injected code replaces RoPE.
        kwargs["disable_llm_rope"] = disable_llm_rope
        super().__init__(*args, **kwargs)

        # c_bias: additive logit biases λ_C·Ĉ + λ_ψ·Ψ̃=ΨΨᵀ + residual λ_V·Ĉ value mix;
        # no q/k transform. c_kernel selects Ĉ: "sampled" (probe) or "analytic" (taps).
        self.c_bias = c_bias
        self.c_kernel = c_kernel

        # c_per_layer: REPLACE post-RoPE q/k with C_tok·q, C_tok·k at every layer
        # (q ← C_tok·q; deterministic, scaled to ‖X‖). No dedicated projections in this mode.
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
            # Dedicated shared projections (one set reused at every layer); no bias.
            self.pe_q_proj = nn.Linear(hidden, n_heads * head_dim, bias=False, device=device)
            self.pe_k_proj = nn.Linear(hidden, n_kv * head_dim, bias=False, device=device)
            if inject_v:
                self.pe_v_proj = nn.Linear(hidden, n_kv * head_dim, bias=False, device=device)

        if c_bias:
            # Scalar learnable gains: λ_C init 1.0 (bias active from step 0), λ_ψ and λ_V init 0.1.
            self.lam_c = nn.Parameter(torch.tensor(1.0, device=device))
            self.lam_psi = nn.Parameter(torch.tensor(0.1, device=device))
            self.lam_v = nn.Parameter(torch.tensor(0.1, device=device))
            # λ_C warmup ramp ∈ [0,1], non-persistent (defaults to 1.0 on load).
            self.register_buffer("_lam_c_warmup", torch.tensor(1.0, device=device),
                                 persistent=False)
            # Use SDPA: c_bias adds a float mask to logits; flash-attn can't; eager
            # materializes the full [B,H,c,c] score matrix which is prohibitive.
            self.llm.config._attn_implementation = "sdpa"  # ty: ignore[invalid-assignment]

        # Multi-GPU: each sharded layer recomputes Ĉ from the small [K+1,F] taps locally.
        # Single-GPU: kernel precomputed once in _arm_signals.
        self._llm_sharded = False
        dm = getattr(self.llm, "hf_device_map", None)
        if dm:
            real = {str(d) for d in dm.values() if d not in ("cpu", "disk", -1)}
            self._llm_sharded = len(real) > 1
        if c_bias and self._llm_sharded:
            # Taps + scalar gains accumulate grads across device streams; silence the
            # expected stream-mismatch warning (tiny tensors, negligible sync cost).
            _setw = getattr(getattr(torch.autograd, "graph", None),
                            "set_warn_on_accumulate_grad_stream_mismatch", None)
            if _setw is not None:
                _setw(False)

        # Per-forward signals set by forward()/prepare_generation():
        #   _pe_signal [B, seq, hidden]: additive GT code S=Y_tok.
        #   _pe_C [B, seq, seq]: composite covariance Ĉ (c_per_layer / c_bias).
        #   _pe_Psi [B, seq, seq]: first-moment Gram Ψ̃=ΨΨᵀ (c_bias).
        #   _pe_c_row [B, seq]: analytic relative row c(·) for decode (c_bias).
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

    def structural_parameters(self) -> list[nn.Parameter]:
        """Base graph-side params plus the optional dedicated q/k/v code projections
        (pe_qk_injection mode) and the scalar c_bias gains λ_C/λ_ψ/λ_V."""
        params = super().structural_parameters()
        for name in ("pe_q_proj", "pe_k_proj", "pe_v_proj"):
            mod = getattr(self, name, None)
            if mod is not None:
                params += list(mod.parameters())
        for name in ("lam_c", "lam_psi", "lam_v"):
            p = getattr(self, name, None)
            if p is not None:
                params.append(p)
        return params

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
        # Resolve model-family helpers (attention fns, rotary) from the attention class's module.
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

            # Add GT code S=Y_tok through dedicated projections (RoPE is off; identity rotary above).
            psi = model._pe_signal
            if (psi is not None and psi.shape[0] == hidden_states.shape[0]
                    and psi.shape[1] == hidden_states.shape[1]):
                def _proj(linear):
                    # Project on the weight's device, then move to q/k/v device (device_map=auto).
                    w = linear.weight
                    out = linear(psi.to(device=w.device, dtype=w.dtype))
                    out = out.to(device=query_states.device, dtype=query_states.dtype)
                    return out.view(hidden_shape).transpose(1, 2)
                query_states = query_states + _proj(model.pe_q_proj)
                key_states = key_states + _proj(model.pe_k_proj)
                if model._pe_inject_value:
                    value_states = value_states + _proj(model.pe_v_proj)
            # c_bias: resolve per-layer Ĉ [c,c] / c_row [c]. Single-GPU: reuse _pe_C.
            # Multi-GPU: recompute from taps on this layer's device (small [K+1,F] only).
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
            # c_bias: residual Ĉ-value mix on prompt (cached); renorm to ‖v‖.
            if model.c_bias and cb_C is not None:
                mixed = torch.einsum("nm,bhmd->bhnd", cb_C.to(value_states.device),
                                     value_states.float())
                mn = mixed.norm(dim=-1).mean().clamp(min=1e-12)
                vn = value_states.float().norm(dim=-1).mean()
                mixed = mixed * (vn / mn)
                lam_v = model.lam_v.to(device=value_states.device, dtype=value_states.dtype)
                value_states = value_states + lam_v * mixed.to(value_states.dtype)
            # c_per_layer: REPLACE q/k with C_tok·q, C_tok·k at every layer.
            # Skipped on cached decode (seq mismatch); value is content.
            C = model._pe_C
            if (not model.c_bias and C is not None and C.shape[0] == hidden_states.shape[0]
                    and C.shape[1] == hidden_states.shape[1]):
                # Move C_tok to this layer's device (device_map=auto may shard).
                Cf = C.to(device=query_states.device, dtype=torch.float32)

                def _mix(t):  # t: [B, H, seq, head_dim]
                    # fp32 einsum (bf16 crushes fine c(n-m) decay); renorm to t's mean
                    # row-norm for scale stability across depths.
                    out = torch.einsum("bnm,bhmd->bhnd", Cf.to(t.device), t.float())
                    cur = out.norm(dim=-1).mean().clamp(min=1e-12)
                    tgt = t.float().norm(dim=-1).mean()
                    return (out * (tgt / cur)).to(t.dtype)

                query_states = _mix(query_states)
                key_states = _mix(key_states)
            # ------------------------------------------------------------------

            if past_key_values is not None:
                key_states, value_states = past_key_values.update(
                    key_states, value_states, attn.layer_idx)

            # c_bias: fold λ_C·Ĉ + λ_ψ·Ψ̃ into attention_mask (added to logits pre-softmax).
            # Built post-cache so decode row spans all cached keys.
            if model.c_bias:
                key_len = key_states.shape[2]
                dev = query_states.device
                # Move gains to this layer's device (not just dtype, device_map=auto).
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
                    # DECODE (graph-extended): live row Ĉ[new, :key_len], padded/clipped to key_len.
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
                        # SDPA with no mask uses is_causal internally; supplying one disables it,
                        # so fold the causal triangle into the bias (prompt only).
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
        """Build ``inputs_embeds`` (gated GT blend) and arm per-forward signal(s) per mode:
        - additive: ``self._pe_signal = Y_tok`` injected into q/k/v at every layer;
        - c_per_layer: ``self._pe_C = C_tok`` replaces q/k (``q ← C_tok·q``);
        - c_bias: ``self._pe_C``, ``self._pe_Psi``, ``self._pe_c_row`` set for additive bias.
        Unused signals are cleared to avoid stale leakage.
        """
        if self.c_bias:
            # c_bias: additive λ_C·Ĉ + λ_ψ·Ψ̃ bias + λ_V value mix; per-sequence [B,c,c] kernels.
            B, c = input_ids.shape[0], input_ids.shape[1]
            self._pe_signal = None
            self._pe_cyc = c
            inputs_embeds, c_hat, psi_t = self._fuse_embeddings(
                input_ids, graphs, injection_maps, permutation=permutation,
                return_cbias=True)
            self._pe_C = c_hat                                   # [B, c, c]
            self._pe_Psi = psi_t                                 # [B, c, c]
            self._pe_taps = None
            # Decode fallback: analytic c(·) row; superseded if decode_setup is armed.
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
            # Disarm signals; keep armed under gradient checkpointing (backward recomputes).
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
