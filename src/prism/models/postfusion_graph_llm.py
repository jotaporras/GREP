"""Post-fusion (late/output) graph attention — an EXPERIMENTAL, self-contained arch.

Motivation: the input-injection (:class:`GraphAugmentedLLM`) and attention-mask
(:class:`GraphMaskLLM` / :class:`LearnableGraphMaskLLM`) families both fold the graph
signal *inside* the transformer, so it propagates through the KV cache to the answer.
Post-fusion instead fuses the graph AFTER the LLM, right before the vocabulary head::

    Y      = LLM(input_ids)                 # [B, T, H] final hidden states
    Psi    = pe_model(graph)                # [N, P] per-node embeddings (a standalone GT)
    Y_tilde = Y + tanh(gate) * CrossAttn(Q=Y, K=Psi, V=Psi)
    logits = VocabHead(Y_tilde)

Cross-attention (not a plain additive placement at node-token spans) is deliberate: the
vocab head is position-wise, so the ONLY way the graph can reach the token that predicts
the next word is to be *written into that token's own hidden vector*. Attention is the one
operation that moves information across positions, so every output token — including each
autoregressively generated one — queries the N node embeddings fresh. This is what makes
the fusion non-trivial at generation time (a bare `Y + Psi` at node spans would vanish
under KV-cached decoding; see the design notes in the e10 discussion).

This module is intentionally isolated from ``gnn_llm.py``: the idea may not pan out, and
keeping it separate makes it a clean delete (drop this file + the ``postfusion_graph_llm``
branches tagged in architectures/train_v3/loaders/inference/evaluate).

Gemma-4 only in practice, but nothing here is Gemma-specific — the fusion is a forward
pre-hook on the LLM's output-embedding (lm_head) module, so HF's own loss / logit-softcap
/ generation machinery is untouched.
"""
import copy

import torch
from torch import nn
from torch_geometric.data import Batch
from transformers import PreTrainedModel


class GraphCrossAttention(nn.Module):
    """Multi-head cross-attention: LLM output tokens (queries) read graph nodes (keys/values).

    ``forward(hidden, psi, node_mask)`` with ``hidden`` ``[B, T, H]`` (query, the LLM's
    final hidden states), ``psi`` ``[B, N, P]`` (padded per-graph node embeddings), and
    ``node_mask`` ``[B, N]`` bool (True = real node, False = padding). Returns the graph
    context ``[B, T, H]`` (pre-gate). Runs in fp32 (N is tiny; keeps the small fusion
    numerically clean under bf16 autocast).

    NOTE on the projection names (``to_q``/``to_k``/``to_v``/``proj_out``): they deliberately
    avoid the LoRA target suffixes (``q_proj``/``k_proj``/``v_proj``/``o_proj``). PEFT matches
    target modules by ``key.endswith('.'+target)``, so a submodule literally named ``q_proj``
    here would get a LoRA adapter *and* be frozen — silently mis-training the fusion. The
    ``to_*`` names never end with a target suffix, so PEFT leaves them alone.

    Args:
        hidden_size: LLM hidden width H (query dim and output dim).
        pe_dim: node-embedding width P (key/value input dim, = GT ``d_model``).
        num_heads: attention heads; must divide ``hidden_size``.
        dropout: attention dropout (train only).
    """

    def __init__(self, hidden_size: int, pe_dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = int(num_heads)
        self.head_dim = hidden_size // num_heads
        self.dropout = float(dropout)
        # Bias-free so an all-padding (impossible here, but defensive) or zero query stays clean.
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_k = nn.Linear(pe_dim, hidden_size, bias=False)
        self.to_v = nn.Linear(pe_dim, hidden_size, bias=False)
        self.proj_out = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor, psi: torch.Tensor,
                node_mask: torch.Tensor) -> torch.Tensor:
        B, T, H = hidden.shape
        N = psi.shape[1]
        q = self.to_q(hidden).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B,h,T,d]
        k = self.to_k(psi).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)      # [B,h,N,d]
        v = self.to_v(psi).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)      # [B,h,N,d]
        # Bool key-padding mask [B,1,1,N]: True = attend. Broadcast over heads and queries.
        # Every graph has >=1 real node, so no query row is fully masked (softmax safe).
        attn_mask = node_mask[:, None, None, :]
        ctx = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0)             # [B,h,T,d]
        ctx = ctx.transpose(1, 2).reshape(B, T, H)
        return self.proj_out(ctx)


class PostFusionGraphLLM(PreTrainedModel):  # ty:ignore[unsupported-base]
    """LLM whose FINAL hidden states cross-attend the graph before the vocab head.

    The base LLM is run unchanged; a forward pre-hook on its output-embedding (lm_head)
    module rewrites the head's input from ``Y`` to ``Y + tanh(gate)*CrossAttn(Y, Psi)``.
    Because the hook sits on lm_head — which HF calls once per forward, outside gradient
    checkpointing and on every generation step — the fusion affects the loss during
    training and every generated token at eval, while leaving HF's loss / logit-softcapping
    / ``generate`` code paths intact. q/k/v of the LLM are never touched (KV-sharing is
    irrelevant).

    The learnable side (:meth:`structural_parameters`, reported for the boosted-LR group
    and grad-re-enabled by ``GraphSFTTrainer``) is the standalone Graph Transformer
    ``pe_model``, the cross-attention ``fusion``, and the scalar ``gate``. ``gate`` starts
    at ``gate_init`` (default 0 ⇒ ``tanh(0)=0`` ⇒ cold start: the model IS the base LLM at
    init, and the fusion ramps in).

    Args:
        llm: base causal LLM (Gemma-4 12B / 31B).
        pe_model: standalone Graph Transformer; ``forward(graph) → [N, P]`` (P = ``pe_dim``).
        pe_dim: node-embedding width P (= GT ``d_model``).
        num_heads: cross-attention heads (must divide the LLM hidden size).
        gate_init: initial value of the scalar gate (0.0 ⇒ cold start).
        dropout: cross-attention dropout.
    """

    def __init__(self, llm: nn.Module, pe_model: nn.Module, pe_dim: int,
                 num_heads: int = 8, gate_init: float = 0.0, dropout: float = 0.0):
        # Wrapper is not a registered HF architecture or MoE class; force "eager" so
        # PreTrainedModel doesn't reject SDPA/flash or expert-impl validation.
        config = copy.copy(llm.config)
        config._attn_implementation = "eager"  # ty: ignore[invalid-assignment]
        config._experts_implementation = "eager"  # ty: ignore[invalid-assignment]
        super().__init__(config)
        self.llm = llm

        # Place the graph-side modules on the LLM's device so PEFT doesn't leave them on CPU.
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = llm.device
        self.pe_model = pe_model.to(device)
        hidden = llm.config.get_text_config().hidden_size
        self.fusion = GraphCrossAttention(hidden, pe_dim, num_heads=num_heads,
                                          dropout=dropout).to(device)
        # Scalar cold-start gate g = tanh(gate). gate_init=0 ⇒ g=0 ⇒ Ψ off at init;
        # trains (tanh'(0)=1). Kept fp32 (the fusion runs fp32).
        self.gate = nn.Parameter(torch.tensor(float(gate_init), device=device))

        # Per-forward graph context (psi_pad [B,Nmax,P], node_mask [B,Nmax]); read by the
        # lm_head pre-hook, set in forward / inference, disarmed afterward.
        self._graph_ctx: tuple[torch.Tensor, torch.Tensor] | None = None
        self._install_postfusion_hook()

    def structural_parameters(self) -> list[nn.Parameter]:
        """Graph-side params for the boosted-LR group: GT + cross-attention + gate."""
        return list(self.pe_model.parameters()) + list(self.fusion.parameters()) + [self.gate]

    def _lm_head_module(self) -> nn.Module:
        """The module HF calls to map hidden states → logits (the fusion attach point)."""
        head = self.llm.get_output_embeddings()
        if head is None:
            head = getattr(self.llm, "lm_head", None)
        if head is None:
            raise ValueError(
                "PostFusionGraphLLM needs the base LLM to expose an output-embedding "
                "(lm_head) module to attach the fusion pre-hook; none found.")
        return head

    def _apply_fusion(self, hidden: torch.Tensor, psi_pad: torch.Tensor,
                      node_mask: torch.Tensor) -> torch.Tensor:
        """``hidden + tanh(gate)*CrossAttn(hidden, Psi)`` in fp32, cast back to ``hidden``'s dtype."""
        orig_dtype = hidden.dtype
        # Disable autocast so the small fusion runs cleanly in fp32 (mirrors the GT's own
        # fp32 sparse ops); psi_pad / fusion params / gate are already fp32.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            h = hidden.float()
            ctx = self.fusion(h, psi_pad, node_mask)
            fused = h + torch.tanh(self.gate.float()) * ctx
        return fused.to(orig_dtype)

    def _install_postfusion_hook(self) -> None:
        """Register the persistent lm_head forward pre-hook that folds in the graph context.

        When ``self._graph_ctx`` is None (non-graph batch / disarmed) the hook is a no-op,
        so it's safe to leave installed. Persistent registration survives PEFT because
        lm_head is not a LoRA target (the module object is unchanged by ``get_peft_model``).
        """
        head = self._lm_head_module()

        def _pre_hook(module, args):
            ctx = self._graph_ctx
            if ctx is None or not args:
                return None
            hidden = args[0]
            psi_pad, node_mask = ctx
            # Batch mismatch (defensive) ⇒ pass through untouched. Query length T may be the
            # full prompt (training / prefill) or 1 (cached decode) — both fuse fine.
            if not isinstance(hidden, torch.Tensor) or hidden.dim() != 3 \
                    or hidden.shape[0] != psi_pad.shape[0]:
                return None
            fused = self._apply_fusion(hidden, psi_pad, node_mask)
            return (fused,) + tuple(args[1:])

        self._postfusion_handle = head.register_forward_pre_hook(_pre_hook)

    def build_graph_context(self, graphs, device):
        """Padded node embeddings + mask for a batch of graphs.

        Returns ``(psi_pad [B, Nmax, P] fp32, node_mask [B, Nmax] bool)`` where row ``b``
        holds ``pe_model(graphs[b])`` in its first ``N_b`` slots. Ψ carries grad to the GT.
        Returns ``None`` if there are no nodes at all (fusion becomes a no-op).
        """
        psis = [self.pe_model(graphs[b]).float() for b in range(len(graphs))]  # each [N_b, P]
        n_max = max((p.shape[0] for p in psis), default=0)
        if n_max == 0:
            return None
        pe_dim = psis[0].shape[1]
        B = len(psis)
        psi_pad = torch.zeros(B, n_max, pe_dim, device=device, dtype=torch.float32)
        node_mask = torch.zeros(B, n_max, dtype=torch.bool, device=device)
        for b, psi in enumerate(psis):
            n = psi.shape[0]
            psi_pad[b, :n] = psi.to(device)
            node_mask[b, :n] = True
        return psi_pad, node_mask

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)  # defer to nn.Module first
        except AttributeError:
            return getattr(self.llm, name)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        graphs: Batch | None = None,
        injection_maps: list[dict[int, list[tuple[int, int]]]] | None = None,
        **kwargs,
    ):
        # injection_maps is accepted (the collator passes it) but unused: cross-attention
        # lets every token read every node, so no token→node span map is needed.
        kwargs.pop("inputs_embeds", None)
        kwargs.pop("input_ids", None)
        # Arm the graph context for the lm_head pre-hook. No graph ⇒ plain causal LLM.
        if graphs is not None and input_ids is not None:
            self._graph_ctx = self.build_graph_context(graphs, input_ids.device)
        else:
            self._graph_ctx = None
        try:
            return self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )
        finally:
            # lm_head runs during this forward (it is NOT recomputed in backward under
            # gradient checkpointing, unlike the attention forwards), so disarming here is
            # safe — autograd already holds the fusion's saved tensors for the backward pass.
            self._graph_ctx = None
