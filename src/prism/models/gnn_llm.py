import copy
import importlib
import re
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
