import contextlib
import warnings

import torch
from torch import nn, Tensor
from torch.distributions import Cauchy, Normal, StudentT
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import Data
from torch_geometric.utils import (add_self_loops, coalesce, softmax,
                                   to_scipy_sparse_matrix)

from prism.models.r_pearl import RandomGNNPositionalEncodings
from prism.models.utils import SparseCSRDropout

warnings.filterwarnings("ignore", ".*Sparse CSR tensor support is in beta state.*")


class SparseGraphAttention(nn.Module):
    """Single-layer sparse graph-transformer attention over k-hop neighborhoods.

    Uses manual gather/scatter (not PyG ``MessagePassing.propagate``) to avoid a
    bf16 autocast scatter-kernel interaction that crashes on some GPUs.

    Args:
        d_model (int): Input/output feature dimension
        heads (int): Number of attention heads
        dropout (float): Attention weight dropout
    """

    def __init__(self, d_model: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.head_dim = d_model // heads
        self.d_model = d_model

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        # 1/sqrt(head_dim) attention scale.
        self.register_buffer("scale", torch.tensor(self.head_dim, dtype=torch.float).rsqrt())

        self.dropout: nn.Module = nn.Dropout(dropout)
        self.attn_dropout: nn.Module = SparseCSRDropout(dropout)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        N = x.shape[0]

        q = self.W_Q(x).view(N, self.heads, self.head_dim).permute(1, 0, 2)
        k = self.W_K(x).view(N, self.heads, self.head_dim).permute(1, 0, 2)
        v = self.W_V(x).view(N, self.heads, self.head_dim).permute(1, 0, 2)

        values = torch.ones(edge_index.shape[1], device=x.device, dtype=x.dtype)
        A = torch.sparse_coo_tensor(
            indices=edge_index, values=values, size=(N, N), device=x.device
        ).coalesce().to_sparse_csr()

        out = self._mha_sparse_attention(q, k, v, A)
        out = self.dropout(self.W_O(out))
        return out

    def _mha_sparse_attention(self, QX: torch.Tensor, KX: torch.Tensor,
                              VX: torch.Tensor, A_csr: torch.Tensor) -> torch.Tensor:
        """Sparse multi-head attention using CSR adjacency with per-head sampled_addmm.

        Args:
            QX: Query tensor of shape (H, N, Fh).
            KX: Key tensor of shape (H, N, Fh).
            VX: Value tensor of shape (H, N, Fh).
            A_csr: Sparse CSR adjacency of shape (N, N).

        Returns:
            Aggregated tensor of shape (N, H*Fh).
        """
        return _SafeBatchedSparseAttn.apply(
            QX, KX, VX, A_csr, self.scale, self.attn_dropout, self.training
        )


class _SafeBatchedSparseAttn(torch.autograd.Function):
    """Per-head sparse attention, memory-bounded.

    Forward and backward loop over heads so peak memory is one head's N×N sparsity
    pattern plus one ``[N, F_head]`` fp32 working copy, rather than all H at once
    (which OOMs on the composite graph, N ≈ thousands). The fp32 cast is per-head
    and only for the sparse score op (``sampled_addmm`` is fp32-only); saved
    tensors keep their original (e.g. bf16) dtype.
    """

    @staticmethod
    def _head_attn(qi, ki, vi, A_csr_f, scale, attn_dropout, training):
        """One head: sparse masked ``softmax(QKᵀ·scale) @ V`` over A's pattern.

        ``qi/ki/vi`` are ``[N, F_head]`` (any float dtype). Returns ``[N, F_head]``
        in fp32. ``A_csr_f`` is the fp32 CSR adjacency (shared across heads). The
        fp32 score tensor is local and freed when the head finishes.
        """
        N = qi.shape[0]
        # Sparse ops (sampled_addmm, sparse.mm) are fp32-only; disable autocast so a
        # surrounding bf16 context can't recast the operands and raise on BFloat16.
        with torch.autocast(device_type=qi.device.type, enabled=False):
            unnormalized = torch.sparse.sampled_addmm(
                input=A_csr_f, mat1=qi.float(), mat2=ki.float().T, beta=0.0
            )
            if training and attn_dropout is not None:
                unnormalized = attn_dropout(unnormalized)
            crow = unnormalized.crow_indices()
            col = unnormalized.col_indices()
            row_counts = crow[1:] - crow[:-1]
            row_index = torch.arange(N, device=qi.device).repeat_interleave(row_counts)
            # Scaled dot-product scores with per-neighborhood (per-row) softmax.
            attn_alpha = softmax(src=unnormalized.values() * scale, index=row_index,
                                 dim=0, num_nodes=N)
            B = torch.sparse_csr_tensor(crow, col, attn_alpha, size=(N, N))
            return torch.sparse.mm(B, vi.float())  # [N, F_head], fp32

    @staticmethod
    def forward(ctx, QX, KX, VX, A_csr, scale, attn_dropout, training):
        H, N, F_head = QX.shape
        orig_dtype = QX.dtype
        A_csr_f = A_csr.float()  # cast the (binary) adjacency once, shared by heads

        # Accumulate per-head outputs; only one head's fp32 working set is live.
        outs = [
            _SafeBatchedSparseAttn._head_attn(
                QX[i], KX[i], VX[i], A_csr_f, scale, attn_dropout, training)
            for i in range(H)
        ]
        attn_out = (torch.stack(outs, dim=0)        # [H, N, F_head]
                    .permute(1, 0, 2)               # [N, H, F_head]
                    .reshape(N, H * F_head)
                    .to(orig_dtype))

        # Save original-dtype tensors; backward recasts per head, mirroring forward.
        ctx.save_for_backward(QX, KX, VX, A_csr_f, scale)
        ctx.orig_dtype = orig_dtype
        ctx.attn_dropout = attn_dropout
        ctx.is_training = training
        return attn_out

    @staticmethod
    def backward(ctx, grad_output):
        QX, KX, VX, A_csr_f, scale = ctx.saved_tensors
        H, N, F_head = QX.shape

        # Reshape grad to (H, N, F_head).
        grad_out = grad_output.float().reshape(N, H, F_head).permute(1, 0, 2)

        grad_QX = torch.zeros_like(QX)
        grad_KX = torch.zeros_like(KX)
        grad_VX = torch.zeros_like(VX)

        # Per-head backward: recompute each head's attention with grad enabled,
        # driving gradients through the same primitive the forward uses.
        for i in range(H):
            with torch.enable_grad():
                qi = QX[i].detach().float().requires_grad_(True)
                ki = KX[i].detach().float().requires_grad_(True)
                vi = VX[i].detach().float().requires_grad_(True)
                out_h = _SafeBatchedSparseAttn._head_attn(
                    qi, ki, vi, A_csr_f, scale, ctx.attn_dropout, ctx.is_training)
            out_h.backward(grad_out[i])
            grad_QX[i] = qi.grad.to(QX.dtype)
            grad_KX[i] = ki.grad.to(KX.dtype)
            grad_VX[i] = vi.grad.to(VX.dtype)

        # 7 inputs to forward: QX, KX, VX, A_csr, scale, attn_dropout, training.
        return grad_QX, grad_KX, grad_VX, None, None, None, None


class SparseTransformerBlock(nn.Module):
    """
    One transformer block: attention + residual + norm + FFN + residual + norm.

    Args:
        d_model (int): Feature dimension.
        heads (int): Number of attention heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model: int, heads: int = 4, dropout: float = 0.1,
                 normalize: bool = True):
        super().__init__()
        self.attn = SparseGraphAttention(d_model, heads=heads, dropout=dropout)
        self.dropout: nn.Module = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
        )
        # normalize=False keeps the block magnitude-preserving: the two LayerNorms
        # are skipped and the residual adds carry through. Used for the final block.
        self.normalize = normalize
        self.norms = nn.ModuleList([])
        if normalize:
            self.norms.append(nn.LayerNorm(d_model))
            self.norms.append(nn.LayerNorm(d_model))


    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # Attention + residual, FFN + residual, each with optional LayerNorm
        # (skipped when normalize=False, leaving the block magnitude-preserving).
        h = self.attn(x, edge_index)
        x = self.norms[0](x + self.dropout(h)) if self.normalize else x + self.dropout(h)
        h = self.ffn(x)
        x = self.norms[1](x + self.dropout(h)) if self.normalize else x + self.dropout(h)
        return x


class GraphTransformer(nn.Module):
    """Full Graph Transformer with R-PEARL positional encodings.

    Pipeline: R-PEARL(graph) -> PE ⊕ node_features -> stacked SparseTransformerBlocks -> output

    With token embeddings supplied (fusion), node features are ``H0 = X_full + Psi``:
    ``X_full`` holds the token embeddings on the cycle (token) nodes and zeros on
    scene nodes, ``Psi`` is the R-PEARL encoding. No gate here — the cold-start gate
    is applied at the LLM input. Only the token-node output rows are used.

    With token embeddings omitted, it falls back to a pure R-PEARL PE generator
    (the ``rpearl_gt_llm`` path).

    Args:
        num_layers (int): Number of transformer blocks.
        pe_hidden_channels (int): R-PEARL GCN hidden dimension.
        pe_num_layers (int): R-PEARL GCN depth.
        d_model (int): Transformer feature dimension.
        heads (int): Number of attention heads.
        num_samples (int): R-PEARL sample count M.
        dropout (float): Dropout rate.
        k_pe (int): TAGConv polynomial order for R-PEARL.
        k_gt (int): Hop radius for sparse attention neighborhoods.
        eps (float): Retained for config back-compat (unused).
        use_layer_norm (bool): Retained for config back-compat; norms are always LayerNorm.
        pe_pool (str): WHERE the probe expectation E_q is taken.
            ``"pe"`` (default): inside R-PEARL — Ψ = Φ(E_q[Φ(q; S, H)], G, T). The blocks
            see the first moment only.
            ``"gt"``: here, after the blocks — Ψ = E_q[Φ(Φ(q; S, H), G, T)]. The M probe
            responses are pushed through the blocks as one block-diagonal M-copy graph
            (probes independent, weights shared) and averaged last. Costs M× the block
            activations and M× the k-hop sparsity pattern; see forward().
        fuse_node_features (bool): When True the blocks compute Φ(X + P; T): semantic
            node features X = ``Linear(data.x)`` are ADDED to the probe PE
            P = E_q[Φ(q; S, H)]. R-PEARL keeps its RANDOM-probe path, so
            ``node_feature_dim`` here is the ``data.x`` width, NOT a switch to the
            deterministic encoder (that is ``node_feature_dim`` with this False).
            Requires ``node_feature_dim``.
        directed (bool): Forwarded to the R-PEARL probe backbone (MagNet when True);
            see :class:`RandomGNNPositionalEncodings`. The GT blocks are unaffected.
        learn_r (bool): Forwarded to the same backbone — learn MagNet's charge r per
            layer; ``directed`` only. See :class:`RandomGNNPositionalEncodings`.
        hidden_norm (str): Forwarded to the same backbone — MagNet's inter-layer
            normalization, ``"none"`` (default) or ``"global_rms"``; ``directed`` only.
        cache_pe (bool): Reuse Ψ across consecutive forwards over the SAME ``Data``
            object; see :meth:`_probe_pe`. Off by default — every current consumer
            passes a fresh graph per forward, and a stale cache would be a silent
            train/eval drift. Callers that set it own :meth:`invalidate_cache`.
    """

    def __init__(self, num_layers: int, pe_hidden_channels: int,
                 pe_num_layers: int, d_model: int, heads: int = 4, num_samples: int = 30,
                 dropout: float = 0.1, k_pe: int = 3, k_gt: int = 3,
                 eps: float = 1e-8, use_layer_norm: bool = True,
                 probe_distribution: str = "gaussian",
                 max_gather_rows: int = 2_000_000,
                 fixed_seed_mode: bool = False, fixed_seed_value: int = 0,
                 pe_readout: str = "mean",
                 center_second_moment: bool = True,
                 node_feature_dim: int = None,
                 pe_pool: str = "pe",
                 fuse_node_features: bool = False,
                 directed: bool = False,
                 learn_r: bool = True,
                 hidden_norm: str = "none",
                 cache_pe: bool = False):
        super().__init__()
        if pe_readout not in ("mean", "second_moment"):
            raise ValueError(
                f"pe_readout must be 'mean' or 'second_moment', got {pe_readout!r}"
            )
        if pe_pool not in ("pe", "gt"):
            raise ValueError(f"pe_pool must be 'pe' or 'gt', got {pe_pool!r}")
        if pe_pool == "gt" and pe_readout != "mean":
            raise ValueError(
                "pe_pool='gt' takes E_q outside the blocks and therefore requires the "
                f"first-moment readout (pe_readout='mean'), got {pe_readout!r}: the "
                "second moment C is itself a probe expectation and cannot be deferred."
            )
        if pe_pool == "gt" and node_feature_dim is not None:
            raise ValueError(
                "pe_pool='gt' needs a probe axis to defer E_q over; semantic node "
                "features (node_feature_dim set) are deterministic. Use pe_pool='pe'."
            )
        if fuse_node_features and node_feature_dim is None:
            raise ValueError(
                "fuse_node_features=True computes Φ(X + P; T) and needs node_feature_dim "
                "— the width of data.x it projects to d_model."
            )

        self.k_hops = k_gt
        self.fuse_node_features = fuse_node_features
        self.heads = heads
        self.num_layers = num_layers
        self.d_model = d_model
        self.eps = eps
        # PE readout used in fusion: "mean" (first moment Ψ) or "second_moment"
        # (covariance C applied over the composite graph). See forward() for the
        # exact H0.
        self.pe_readout = pe_readout
        # Where E_q is taken: "pe" (inside R-PEARL) or "gt" (after the blocks).
        self.pe_pool = pe_pool

        # Set up R-PEARL Positional Encoder and Transformer Blocks.
        self.pe_model = RandomGNNPositionalEncodings(
            pe_hidden_channels=pe_hidden_channels, pe_num_layers=pe_num_layers, d_model=d_model,
            num_samples=num_samples, dropout=dropout, k=k_pe, eps=eps, use_layer_norm=use_layer_norm,
            probe_distribution=probe_distribution,
            max_gather_rows=max_gather_rows,
            fixed_seed_mode=fixed_seed_mode, fixed_seed_value=fixed_seed_value,
            center_second_moment=center_second_moment,
            # When fusing, X is added HERE (self.input_proj) instead of being consumed
            # by R-PEARL, so the probe path stays live and P = E_q[Φ(q; S, H)].
            node_feature_dim=None if fuse_node_features else node_feature_dim,
            directed=directed,
            learn_r=learn_r,
            hidden_norm=hidden_norm,
        )
        self.input_proj = nn.Linear(node_feature_dim, d_model) if fuse_node_features else None
        # Final block is norm-free (normalize=False) so its output magnitude
        # survives for the output gate; earlier blocks keep LayerNorm.
        self.blocks = nn.ModuleList([
            SparseTransformerBlock(
                d_model, heads=heads, dropout=dropout,
                normalize=(i < num_layers - 1),
            ) for i in range(num_layers)
        ])
        # Learnable scalar output gate: g = tanh(output_gain) ∈ (-1, 1), init ≈ 0.76.
        # Lets the model scale the structural output (0 recovers the base LLM).
        self.output_gain = nn.Parameter(torch.tensor(1.0))
        # Ψ cache; see _probe_pe. Plain attributes, so nothing enters the state dict.
        self.cache_pe = cache_pe
        self._pe_cache = self._pe_graph = None

    def _probe_pe(self, pe_data, **kwargs) -> Tensor:
        """Ψ for ``pe_data``, reusing the cached tensor when ``cache_pe`` is set.

        Ψ is a function of the TOPOLOGY alone — R-PEARL's probes never read ``data.x`` —
        so a caller sweeping several node-feature matrices over one graph (an
        autoregressive prefill, a rollout) can pay for it once. Two consequences the
        caller owns. The cached tensor keeps its autograd graph, so every loss that
        consumes it must be covered by ONE backward; a second backward raises
        ``Trying to backward through the graph a second time`` rather than quietly
        using stale gradients. And the whole sweep then shares ONE probe draw, which
        is a single Monte-Carlo estimate of E_q reused across the sweep rather than an
        independent one per forward. Call :meth:`invalidate_cache` after that backward.

        The cache is keyed on the identity of the ``Data`` it was built from, so a new
        graph misses on its own; only a weight update needs the explicit invalidation.
        """
        if not self.cache_pe:
            return self.pe_model(pe_data, **kwargs)
        if self._pe_cache is None or self._pe_graph is not pe_data:
            self._pe_graph = pe_data
            self._pe_cache = self.pe_model(pe_data, **kwargs)
        return self._pe_cache

    def invalidate_cache(self) -> None:
        """Drop the cached Ψ; a no-op when ``cache_pe`` is off (see :meth:`_probe_pe`)."""
        self._pe_cache = self._pe_graph = None

    @torch.no_grad()
    def _expand_edge_index(self, edge_index: Tensor, num_nodes: int, k_hops: int = 1) -> Tensor:
        """Expand the edge index to the ≤k-hop neighborhood via sparse (A+I)^k.

        Args:
            edge_index (LongTensor): edge indices [2, E]
            num_nodes (int): Number of nodes
            k_hops (int | None): Override hop radius. Defaults to self.k_hops.
        """
        k = k_hops if k_hops is not None else self.k_hops

        # (A + I) so the matrix power reaches all nodes within k hops, not only
        # those reachable in exactly k steps.
        edge_idx_self, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        # Binary adjacency; re-binarized after each matmul so entries stay 0/1.
        values = torch.ones(edge_idx_self.shape[1], device=edge_idx_self.device)
        adj = torch.sparse_coo_tensor(
            edge_idx_self, values, (num_nodes, num_nodes)
        ).coalesce()

        reachable = adj
        for _ in range(k - 1):
            reachable = torch.sparse.mm(reachable, adj).coalesce()
            reachable = torch.sparse_coo_tensor(
                reachable.indices(),
                torch.ones(reachable._nnz(), device=reachable.device),
                reachable.shape,
            ).coalesce()

        expanded_edge_index = reachable.indices()
        return coalesce(expanded_edge_index, num_nodes=num_nodes)

    def forward(self, data, token_embeddings=None, is_token=None, permutation=None) -> Tensor:
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = data.x.device
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)

        edge_index = data.edge_index
        if permutation is not None:
            edge_index = permutation.apply(edge_index, data.x.size(0), device=device)
            pe_data = Data(x=data.x, edge_index=edge_index)
            if getattr(data, "edge_weight", None) is not None:
                pe_data.edge_weight = data.edge_weight
        else:
            pe_data = data

        # Input fusion (no gate here — the cold-start gate is at the LLM input):
        #   "mean":          H0 = X_full + Ψ   (Ψ = first moment; X_full = token
        #                    embeddings on cycle rows, 0 on scene).
        #   "second_moment": H0 = seeded + C·seeded, seeded = [X on token rows ;
        #                    Ψ on scene rows], C = E_q[Φ(q)Φ(q)ᵀ] applied over the
        #                    composite graph (never formed). Tokens get the second
        #                    moment (the first collapses on the symmetric cycle);
        #                    scene rows keep their first-moment PE.
        num_nodes = pe_data.x.size(0)
        # pe_pool="gt" defers E_q: R-PEARL hands back the [M, N, d_model] probe stack
        # Φ(q^(s); S, H) and the expectation is taken after the blocks (see below).
        pe_kw = {} if self.pe_pool == "pe" else {"pool": False}
        if token_embeddings is None:
            # Legacy pure-PE-generator path (rpearl_gt_llm): first moment is the PE.
            x = self._probe_pe(pe_data, **pe_kw)
            if self.fuse_node_features:
                # Φ(X + P; T): semantic features X = input_proj(data.x) added to the
                # probe PE P before the blocks. Both fp32 on this path.
                x = self.input_proj(pe_data.x.float()) + x
        else:
            x_full = torch.zeros(num_nodes, self.d_model, device=device, dtype=torch.float32)
            rows = slice(0, token_embeddings.shape[0]) if is_token is None else is_token
            x_full[rows] = token_embeddings.to(device=device, dtype=torch.float32)
            if self.pe_readout == "second_moment":
                psi = self._probe_pe(pe_data)            # first moment Ψ ∈ [N, d_model]
                seeded = x_full.clone()
                if is_token is not None:
                    # scene rows ← first-moment Ψ; token rows stay the verbal embeddings X.
                    seeded[~is_token] = psi[~is_token].to(seeded.dtype)
                cx = self.pe_model.second_moment_apply(pe_data, seeded)  # C·seeded
                x = seeded + cx.to(seeded.dtype)          # H0 = seeded + C·seeded
            else:
                # Broadcasts over the probe axis when pe_pool="gt" (x_full is [N, d]).
                x = x_full + self._probe_pe(pe_data, **pe_kw)

        # Precompute k-hop neighborhood diffusions.
        if permutation is not None:
            khop_edge_index = self._expand_edge_index(edge_index, num_nodes)
        else:
            if not hasattr(data, '_khop_edge_index'):
                data._khop_edge_index = self._expand_edge_index(edge_index, num_nodes)
            khop_edge_index = data._khop_edge_index

        # pe_pool="gt": stack the M probe copies into ONE block-diagonal graph of M*N
        # nodes (same offset trick as R-PEARL's batched GCN), so the blocks run once with
        # shared weights and the probes stay exactly independent — no cross-probe edge is
        # created. Peak cost is M× the [N, d_model] activations and M× nnz(A_k).
        m_probes = x.shape[0] if x.dim() == 3 else None
        if m_probes is not None:
            x = x.reshape(m_probes * num_nodes, -1)
            offsets = (torch.arange(m_probes, device=device) * num_nodes).view(-1, 1, 1)
            khop_edge_index = ((khop_edge_index.unsqueeze(0) + offsets)
                               .permute(1, 0, 2).reshape(2, -1))

        # In the fusion path, run the blocks in the LLM's low-precision dtype:
        # eval/generate has no autocast, so the dense [N, d_model] activations would
        # otherwise sit in fp32 and OOM on the composite graph (N ≈ thousands). The
        # fp32-only sparse score op still casts locally (see _head_attn). The
        # PE-generator path (token_embeddings is None) stays fp32.
        amp_dtype = None
        if token_embeddings is not None and token_embeddings.dtype in (torch.float16, torch.bfloat16):
            amp_dtype = token_embeddings.dtype
        if amp_dtype is not None:
            x = x.to(amp_dtype)
            amp_ctx = torch.autocast(device_type=device.type, dtype=amp_dtype)
        else:
            amp_ctx = contextlib.nullcontext()

        with amp_ctx:
            # During training, activation-checkpoint each block so its dense
            # [N, d_model] activations are recomputed in backward instead of all
            # retained at once (the GT's dominant memory term). use_reentrant=False
            # preserves RNG state and the autocast dtype. Skipped at eval (no grad).
            for block in self.blocks:
                if self.training and torch.is_grad_enabled():
                    x = checkpoint(block, x, khop_edge_index, use_reentrant=False)
                else:
                    x = block(x, khop_edge_index)
            if m_probes is not None:
                # E_q — the ONLY probe aggregation, taken here instead of inside R-PEARL:
                # Ψ = E_q[Φ(Φ(q; S, H), G, T)]. Symmetric in the probe set, so Ψ stays a
                # Monte-Carlo estimator. Averaged in fp32: a bf16 mean over M terms loses
                # the precision the pe_pool="pe" path gets for free (it pools in fp32).
                x = x.view(m_probes, num_nodes, -1).float().mean(dim=0).to(x.dtype)
            # Learnable output gate (no-op when token_embeddings is None).
            if token_embeddings is not None:
                x = x * torch.tanh(self.output_gain).to(x.dtype)
        return x


class SemanticGraphTransformer(nn.Module):
    """Graph Transformer over semantic node features — NO R-PEARL, no random probes.

    The node features (e.g. the mean LLM word-embedding of each node's name, supplied as
    ``data.x`` [N, node_feature_dim] by ``GraphAugmentedLLM.build_pe_signal``) are projected
    to ``d_model`` and refined by stacked sparse k-hop attention blocks. Output is [N, d_model],
    gated by ``tanh(output_gain)``. A drop-in ``pe_model`` for ``GraphAugmentedLLM`` (the
    ``gt_llm`` architecture); it REQUIRES semantic node features (there is no probe fallback).

    Args:
        node_feature_dim (int): input feature width (the LLM text hidden size).
        d_model (int): transformer working dimension.
        num_layers (int): number of sparse transformer blocks.
        heads (int): attention heads.
        dropout (float): dropout.
        k_gt (int): hop radius for the sparse attention neighborhoods.
    """

    def __init__(self, node_feature_dim: int, d_model: int, num_layers: int,
                 heads: int = 4, dropout: float = 0.1, k_gt: int = 3):
        super().__init__()
        self.k_hops = k_gt
        self.d_model = d_model
        self.input_proj = nn.Linear(node_feature_dim, d_model)
        # Blocks 0..L-2 normalize (LayerNorm); the final block is norm-free so the output
        # magnitude survives for the learnable output gate (mirrors GraphTransformer).
        self.blocks = nn.ModuleList([
            SparseTransformerBlock(d_model, heads=heads, dropout=dropout,
                                   normalize=(i < num_layers - 1))
            for i in range(num_layers)
        ])
        self.output_gain = nn.Parameter(torch.tensor(1.0))

    @torch.no_grad()
    def _expand_edge_index(self, edge_index: Tensor, num_nodes: int) -> Tensor:
        """≤k-hop neighborhood via sparse (A+I)^k (binarized each step)."""
        edge_idx_self, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        values = torch.ones(edge_idx_self.shape[1], device=edge_idx_self.device)
        adj = torch.sparse_coo_tensor(edge_idx_self, values, (num_nodes, num_nodes)).coalesce()
        reachable = adj
        for _ in range(self.k_hops - 1):
            reachable = torch.sparse.mm(reachable, adj).coalesce()
            reachable = torch.sparse_coo_tensor(
                reachable.indices(),
                torch.ones(reachable._nnz(), device=reachable.device),
                reachable.shape,
            ).coalesce()
        return coalesce(reachable.indices(), num_nodes=num_nodes)

    def forward(self, data, permutation=None) -> Tensor:
        if permutation is not None:
            raise NotImplementedError(
                "permutation-equivariance eval is not supported for SemanticGraphTransformer "
                "(node features would also need permuting); pass permutation=None."
            )
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = data.x.device
        x = data.x.to(device=device, dtype=torch.float32)          # [N, node_feature_dim]
        edge_index = data.edge_index.to(device)
        x = self.input_proj(x)                                     # [N, d_model]
        khop_edge_index = self._expand_edge_index(edge_index, x.size(0))
        for block in self.blocks:
            if self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, khop_edge_index, use_reentrant=False)
            else:
                x = block(x, khop_edge_index)
        return x * torch.tanh(self.output_gain).to(x.dtype)


class NavigatorPE(nn.Module):
    """PE stage of the notebook navigator — the probe-PE Graph Transformer, NO AGT.

    Ψ = ``pe_gt(graph)``. This is the PE half of
    notebooks/2026-06-28 e9_gnn_navigation.ipynb's ``GNNShortestPathNavigator`` (its ``gnn``
    submodule; checkpoint ``path_navigator_gt.pt``) with the autoregressive head
    deliberately absent: the AGT (``SemanticGraphTransformer`` + classifier +
    autoregressive ``generate``) lives in :class:`NavigatorGT`, which extends this class.

    Also the base of :class:`TwoStagePE`, the LEGACY Ψ = SemanticGT(PE_GT(·)) producer
    kept only so pre-split checkpoints still reload (see that class).

    A drop-in ``pe_model`` for the Ψ-consuming architectures; the state dict keys are
    ``pe_gt.*``, so a plain ``GraphTransformer`` checkpoint needs the prefix strip in
    ``loaders._load_psi_producer_state``.

    Args:
        pe_gt: a ``GraphTransformer`` (random-probe PE) -> [N, d_model].
    """

    def __init__(self, pe_gt: "GraphTransformer"):
        super().__init__()
        self.pe_gt = pe_gt

    def forward(self, data, permutation=None) -> Tensor:
        return self.pe_gt(data, permutation=permutation)           # [N, d_model]


class TwoStagePE(NavigatorPE):
    """LEGACY two-stage Ψ producer: ``Ψ = SemanticGT(PE_GT(graph))``.

    This is what ``NavigatorPE`` used to be, before the PE/AGT split moved the Semantic
    GT into :class:`NavigatorGT`. It is retained under its own name for ONE reason:
    checkpoints trained with ``gnn.pe_gt_from`` AND ``gnn.semantic_gt_from`` (the e13
    navigator arms) carry ``pe_model.pe_gt.*`` + ``pe_model.semantic_gt.*`` and their Ψ
    IS the two-stage composition. Rebuilding those as a bare ``GraphTransformer`` would
    evaluate a DIFFERENT function under the same weights, so the topology is reproduced
    exactly instead. New runs should leave ``gnn.semantic_gt_from`` null (Ψ = PE GT
    alone); ``loaders._load_psi_producer_state`` fails loud if a checkpoint and a rebuild
    disagree about which of the two it is.

    Args:
        pe_gt: a ``GraphTransformer`` (random-probe PE) -> [N, d_model].
        semantic_gt: a ``SemanticGraphTransformer`` consuming [N, d_model]; its
            ``node_feature_dim`` must equal ``pe_gt.d_model``.
    """

    def __init__(self, pe_gt: "GraphTransformer", semantic_gt: "SemanticGraphTransformer"):
        super().__init__(pe_gt)
        self.semantic_gt = semantic_gt

    def forward(self, data, permutation=None) -> Tensor:
        pe = self.pe_gt(data, permutation=permutation)             # [N, d_model]
        edge_index = data.edge_index
        if permutation is not None:
            # Apply the SAME relabelling the PE GT applied, so the head refines Ψ over the
            # graph the PE was computed on. Feeding the original edge_index here would mix a
            # permuted-topology PE with unpermuted topology and silently break equivariance.
            # The permutation is pre-applied rather than passed down: SemanticGraphTransformer
            # rejects the kwarg because it cannot permute semantic node FEATURES — here the
            # features ARE the PE, already in original node order, so only edges move.
            edge_index = permutation.apply(edge_index, pe.size(0), device=pe.device)
        feed = Data(x=pe, edge_index=edge_index)                   # seed = 0 -> head input = PE
        return self.semantic_gt(feed)                              # [N, d_model]


class NavigatorGT(NavigatorPE):
    """Autoregressive shortest-path navigator — the notebook's ``GNNShortestPathNavigator``
    (notebooks/2026-06-28 e9_gnn_navigation.ipynb §3) rebuilt on this repo's GT blocks.

    Extends :class:`NavigatorPE` (the PE stage: ``pe_gt``, a probe-PE ``GraphTransformer``)
    with the AUTOREGRESSIVE head — the ``SemanticGraphTransformer`` ``semantic_gt`` (the
    notebook's ``head``), a per-node scoring ``classifier``, and an autoregressive
    :meth:`generate`. The AGT lives HERE, not in ``NavigatorPE``: ``NavigatorPE`` emits
    Ψ = ``pe_gt(graph)`` and nothing else. The head reads ``graph.x + PE`` where
    ``graph.x`` is the navigation SEED: start and
    goal are provided EXPLICITLY, not parsed from text — the goal node carries a tag at
    ``goal_loc`` and the current node a tag at the running step count, both offset by a
    sinusoidal step code, exactly as the notebook seeds them. R-PEARL probes are
    x-independent, so the PE is cached per graph.

    Decoding uses the notebook's *blurry-vision* mask: the next node is chosen among the
    unvisited nodes within ``mask_hops`` BFS hops of the current one (not just its direct
    neighbours), so the emitted route may contain multi-hop jumps by construction.

    .. warning:: **Tag-scale discrepancy in the source notebook (not fixed here).**
       Three different tag scales appear around the suite8 weights:

       * ``GNNShortestPathNavigator.__init__`` sets ``self.STD = target_var ** 1/2``.
         Python binds ``**`` tighter than ``/``, so this is ``(target_var ** 1) / 2``
         = ``0.01 / 2`` = **0.005**, NOT ``sqrt(0.01)`` = 0.1. Almost certainly an
         operator-precedence slip for ``target_var ** 0.5``. ``generate`` uses it.
       * The suite8 TRAINING loop (notebook cell 79) tags with the module-scope
         ``STD`` = **0.1** — that is the scale the weights were actually fitted under.
       * This class previously drew the tags from ``StudentT(df, loc, SCALE)`` with
         ``SCALE = sqrt(target_var*(df-2)/df)`` = **0.0775**, a family the notebook
         never uses.

       The default (``tag_dist='normal'``, ``tag_scale=None``) reproduces the notebook
       CLASS verbatim, i.e. 0.005 — so decoding runs at 1/20th the tag scale the weights
       saw in training and is off-distribution by construction. Set ``tag_scale: 0.1``
       in the config to decode at the training scale, or ``tag_dist: studentt`` for the
       previous repo arm. See ``experiments/e9_navigator_gt.yaml``.

    Load the notebook checkpoint (``path_navigator.pt`` = ``navigator.state_dict()``, keys
    ``gnn.*`` / ``head.*`` / ``classifier.*``) with :meth:`from_pretrained`, which remaps
    ``gnn.``→``pe_gt.`` and ``head.``→``semantic_gt.``.

    Args:
        pe_gt: a ``GraphTransformer`` (random-probe PE) -> [N, d_model].
        semantic_gt: a ``SemanticGraphTransformer`` consuming [N, d_model].
        max_length: hard cap on rollout length (the notebook's ``MAX_LENGTH``).
        mask_hops: blurry-vision radius, in BFS hops, of the per-step candidate set.
        tag_dist: goal / step tag family — ``'normal'`` (the notebook) or ``'studentt'``.
        target_var: notebook ``target_var``. Drives BOTH ``STD`` (normal arm, via the
            notebook's ``target_var ** 1/2``) and ``SCALE`` (studentt arm).
        tag_scale: explicit override of the normal arm's ``STD``; ``None`` = notebook value.
        df: Student's-T dof, and ``SCALE = sqrt(target_var * (df - 2) / df)``. Kept for
            back-compat with configs that parameterise the tags by ``df``.
        mu: retained for notebook/config parity (unused — step order rides on the
            sinusoidal PE, not on the tag location).
        pe_base: base of the sinusoidal step code (the notebook's ``PE_BASE``).
        base_loc / base_scale: Normal prior seeded on every non-tagged node.
        goal_loc: location of the goal node's tag.
        cauchy_scale: scale of the ``Cauchy(base_loc, ·)`` [N, 1] prior ``generate``
            restores ``graph.x`` to on exit (the notebook's ``CAUCHY_SCALE`` cleanup).
    """

    def __init__(self, pe_gt: "GraphTransformer", semantic_gt: "SemanticGraphTransformer",
                 max_length: int = 128, mask_hops: int = 3, df: float = 5.0,
                 target_var: float = 0.01, mu: float = 1.0, pe_base: float = 10000.0,
                 base_loc: float = 0.0, base_scale: float = 0.1, goal_loc: float = -5.0,
                 tag_dist: str = "normal", tag_scale: float = None,
                 cauchy_scale: float = 1.0):
        super().__init__(pe_gt)
        # The AGT head is owned by this class (NavigatorPE is the PE stage alone).
        self.semantic_gt = semantic_gt
        if tag_dist not in ("normal", "studentt"):
            raise ValueError(
                f"tag_dist must be 'normal' (the notebook) or 'studentt', got {tag_dist!r}")
        self.MAX_LENGTH = max_length
        self.MASK_HOPS = mask_hops
        self.TAG_DIST = tag_dist
        self.DF = df
        # Notebook parity, verbatim: GNNShortestPathNavigator.__init__ writes
        #   self.STD = target_var ** 1/2
        # which Python parses as (target_var ** 1) / 2 -> 0.005, not sqrt(0.01) = 0.1.
        # Reproduced deliberately (see the class docstring warning); `tag_scale` overrides.
        self.STD = (target_var ** 1 / 2) if tag_scale is None else float(tag_scale)
        self.SCALE = (target_var * (df - 2.0) / df) ** 0.5
        self.MU = mu
        self.PE_BASE = pe_base
        self.BASE_LOC = base_loc
        self.BASE_SCALE = base_scale
        self.GOAL_LOC = goal_loc
        self.CAUCHY_SCALE = cauchy_scale
        self.shape = pe_gt.d_model                       # PE / seed feature width
        self.classifier = nn.Linear(self.shape, 1)
        # PE cache keyed on graph identity (R-PEARL probes ignore data.x).
        self.graph = None
        self.cached_pe = None

    def _device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def invalidate_cache(self) -> None:
        self.graph = None
        self.cached_pe = None

    def forward(self, graph: Data, permutation=None) -> Tensor:
        # Cache PE per graph: R-PEARL probes are x-independent, so re-seeding graph.x
        # across rollout steps does not change the PE.
        if self.cached_pe is None or not self.cached_pe.any() or self.graph is not graph:
            self.graph = graph
            self.cached_pe = self.pe_gt(graph, permutation=permutation)   # [N, d_model]
        # Fresh Data (not graph.clone()): the head reads only x / edge_index, and the
        # rollout calls this once per step — cloning the [N, N] hop matrix each time is pure cost.
        feed = Data(x=graph.x + self.cached_pe, edge_index=graph.edge_index)
        return self.classifier(self.semantic_gt(feed))                    # [N, 1]

    @staticmethod
    @torch.no_grad()
    def hop_matrix(graph: Data, device=None) -> Tensor:
        """All-pairs BFS hop counts ``[N, N]`` (``inf`` where unreachable).

        Topology only — never edge weights — so a metric distance can never reach the
        blurry-vision mask (the notebook's ``graph.hops``).
        """
        from scipy.sparse.csgraph import shortest_path
        adj = to_scipy_sparse_matrix(graph.edge_index.cpu(), num_nodes=graph.num_nodes)
        hops = shortest_path(adj, method="D", unweighted=True, directed=False)
        return torch.as_tensor(hops, dtype=torch.float, device=device)

    @torch.no_grad()
    def generate(self, graph: Data, node1: int, node2: int) -> Tensor:
        """Autoregressively walk ``node1 -> node2`` under blurry-vision + visited masking.

        Mirrors the notebook: seed every node ~N(base_loc, base_scale), tag the goal at
        ``goal_loc`` and the current node at the running ``count``, both offset by the
        sinusoidal step code ``sin(count / PE_BASE^(2⌊i/2⌋/D) + (i mod 2)·π/2)``; at each
        step score all nodes, mask to the unvisited nodes within ``MASK_HOPS`` BFS hops of
        the current one, and take the argmax. The walk is capped at the longest simple
        path (``N - 1`` hops), and ``graph.x`` is restored to the notebook's [N, 1] Cauchy
        prior on exit. Returns the ordered node-index path as a ``[len, 1]`` LongTensor.

        The tag family/scale is ``TAG_DIST`` / ``STD`` (or ``DF``/``SCALE`` on the
        ``studentt`` arm) — see the class docstring for the suite8 scale discrepancy.
        """
        device = self._device()
        N, D = graph.num_nodes, self.shape
        # Cached on the Data (once per graph) — the mask is topology-only, so it is
        # invariant across rollouts on the same scene graph.
        hops = getattr(graph, "hops", None)
        if hops is None:
            hops = graph.hops = self.hop_matrix(graph, device)
        hops = hops.to(device)

        p = lambda v: torch.tensor(v, device=device, dtype=torch.float)
        pc = lambda v: torch.tensor(float(v), dtype=torch.float)      # CPU-side parameter
        # Goal / step tag sampler over a plain float `loc`. 'normal' is the notebook class
        # verbatim; 'studentt' keeps the earlier repo arm for configs parameterised by `df`.
        if self.TAG_DIST == "studentt":
            # `aten::_standard_gamma` (StudentT -> Chi2 -> Gamma) has no MPS kernel: draw on
            # CPU and move. The tag is per-rollout scratch, never differentiated, so this is
            # math-identical and removes the PYTORCH_ENABLE_MPS_FALLBACK=1 requirement.
            df, scale = pc(self.DF), pc(self.SCALE)
            tag = lambda loc: StudentT(df, loc=pc(loc), scale=scale).sample((1, D)).to(device)
        else:
            std = p(self.STD)
            tag = lambda loc: Normal(loc=p(loc), scale=std).sample((1, D))
        # Loop-invariant halves of the positional code (per-coordinate wavelength / phase).
        pe_scale = self.PE_BASE ** (2 * (torch.arange(D, device=device) // 2) / D)
        pe_phase = (torch.arange(D, device=device) % 2) * (torch.pi / 2)

        graph.x = Normal(loc=p(self.BASE_LOC), scale=p(self.BASE_SCALE)).sample((N, D))
        graph.x[node2] = tag(self.GOAL_LOC) + torch.sin(pe_phase)

        count = 1.0
        preds = [node1]
        visited = {node1}
        max_hops = min(self.MAX_LENGTH, N - 1)   # cap at the longest simple path
        while not (preds[-1] == node2 or len(preds) > max_hops):
            c = preds[-1]
            graph.x[c] = tag(count) + torch.sin(count / pe_scale + pe_phase)
            allowed = hops[c] <= self.MASK_HOPS
            if visited:
                allowed[torch.as_tensor(sorted(visited), device=allowed.device)] = False
            if not bool(allowed.any()):
                break
            logits = self(graph).T.masked_fill(~allowed.unsqueeze(0), float("-inf"))
            nxt = int(logits.argmax(dim=1))
            preds.append(nxt)
            visited.add(nxt)
            count += 1.0

        # Notebook cleanup: hand `graph.x` back as the [N, 1] Cauchy prior the other
        # notebook heads (detector / SPD) read. The seed is per-rollout scratch either way.
        # Drawn on CPU: `aten::cauchy_` has no MPS kernel (same reasoning as the tag draw).
        graph.x = Cauchy(loc=pc(self.BASE_LOC),
                         scale=pc(self.CAUCHY_SCALE)).sample((N, 1)).to(device)
        return torch.tensor([preds], device=device).T

    @classmethod
    def from_pretrained(cls, state_dict_path: str, *, gt_kwargs: dict,
                        semantic_kwargs: dict, map_location="cpu",
                        **nav_kwargs) -> "NavigatorGT":
        """Build a ``NavigatorGT`` and load the notebook's full-navigator state dict.

        ``state_dict_path`` is ``navigator.state_dict()`` (``path_navigator.pt``): keys
        ``gnn.*`` (the GraphTransformer), ``head.*`` (the SemanticGraphTransformer) and
        ``classifier.*``. They are remapped onto this class's ``pe_gt.*`` / ``semantic_gt.*``
        / ``classifier.*`` and loaded strictly (a mismatch means ``gt_kwargs`` /
        ``semantic_kwargs`` do not reproduce the trained submodules). ``nav_kwargs`` are the
        decode-policy arguments of ``__init__`` (``max_length``, ``mask_hops``, ``df``, …).
        """
        model = cls(GraphTransformer(**gt_kwargs),
                    SemanticGraphTransformer(**semantic_kwargs),
                    **nav_kwargs)
        raw = torch.load(state_dict_path, map_location=map_location)
        remap = {}
        for k, v in raw.items():
            if k.startswith("gnn."):
                remap["pe_gt." + k[len("gnn."):]] = v
            elif k.startswith("head."):
                remap["semantic_gt." + k[len("head."):]] = v
            else:
                remap[k] = v                                 # classifier.*
        missing, unexpected = model.load_state_dict(remap, strict=False)
        # Buffers registered on submodules (e.g. attention scale) are non-persistent and
        # legitimately "missing"; unexpected keys mean a genuine architecture mismatch.
        real_missing = [k for k in missing if not k.endswith(".scale")]
        if real_missing or unexpected:
            raise RuntimeError(
                f"NavigatorGT load from {state_dict_path} mismatched the config "
                f"(missing={real_missing}, unexpected={list(unexpected)}); gt_kwargs / "
                f"semantic_kwargs must reproduce the trained submodules exactly.")
        model.eval()
        return model


# ----------------------------------------------------------------------------
# Ψ-producer factory (shared by the mask / rotation architectures)
# ----------------------------------------------------------------------------
def build_psi_producer(cfg, node_feature_dim: int = None) -> nn.Module:
    """Build the Ψ producer for ``learnable_graph_mask`` / ``wire_llm`` from a gnn config.

    ONE construction site, used by training (``architectures.build_planner_model``) and
    by the eval-time checkpoint rebuild (``loaders.graph_augmented_llm_from_pretrained``)
    for BOTH architectures, so a train/eval topology drift is structurally impossible.

    Args:
        cfg: any mapping over the ``gnn`` section — an OmegaConf ``DictConfig`` at train
            time, the flat ``train_config.json`` dict at eval time. Read keys:
            ``gt_num_layers``/``gt_heads``/``k_gt``/``d_model``/``dropout``/``eps``/
            ``use_layer_norm``/``pe_hidden_channels``/``pe_num_layers``/``num_samples``/
            ``k_pe``, plus the navigator switches ``pe_gt_from``/``semantic_gt_from``.
            ``pe_pool`` and ``directed`` are OPTIONAL, defaulting to ``"pe"`` (E_q
            inside R-PEARL) and ``False`` (undirected GCN backbone) — the behaviour
            every existing run and checkpoint was trained with; the fallbacks keep
            checkpoints written before those keys existed reloading as themselves.
        node_feature_dim: semantic input width for the standalone GT (``None`` = random
            probes). Ignored in two-stage mode: the notebook's PE GT is probe-based.

    Returns:
        A standalone :class:`GraphTransformer` — Ψ = GT(graph) — for every current run.
        ``gnn.pe_gt_from`` only says WHICH weights ``loaders.load_navigator_pe_into``
        then pours into it (the notebook's ``path_navigator_gt.pt``); it does NOT change
        the topology. This is the PE stage of :class:`NavigatorPE`: the AGT
        (``SemanticGraphTransformer``) is NOT part of Ψ — it belongs to
        :class:`NavigatorGT`.

        The ONE exception is ``gnn.semantic_gt_from``: it is LEGACY and selects
        :class:`TwoStagePE` (Ψ = SemanticGT(PE_GT(graph))), the pre-split producer, so
        checkpoints from the e13 navigator arms keep reloading as the function they were
        trained as. New runs must leave it null.

    In the legacy two-stage arm the head takes its ``num_layers``/``heads``/``dropout``/
    ``k_gt`` from the SAME ``gt_*`` keys as the PE GT and its ``node_feature_dim`` from
    ``d_model``, because the notebook builds it from one ``model_hparams`` dict
    (e9_gnn_navigation.ipynb, ``GNNShortestPathNavigator.__init__``). The strict load in
    ``loaders.load_navigator_pe_into`` fails loudly if a checkpoint disagrees, so this is a
    checked assumption, not a silent one. Weights are NOT loaded here — this returns the
    topology only.
    """
    pe_gt_from, semantic_gt_from = cfg.get("pe_gt_from"), cfg.get("semantic_gt_from")
    if semantic_gt_from and not pe_gt_from:
        raise ValueError(
            "gnn.semantic_gt_from is set but gnn.pe_gt_from is not: the legacy two-stage Ψ "
            "producer needs BOTH (Ψ = SemanticGT(PE_GT(graph))). Leave semantic_gt_from "
            "null — Ψ is the PE GT alone (gnn.pe_gt_from) for every current run.")
    navigator = bool(pe_gt_from and semantic_gt_from)
    pe_gt = GraphTransformer(
        num_layers=cfg["gt_num_layers"],
        pe_hidden_channels=cfg["pe_hidden_channels"],
        pe_num_layers=cfg["pe_num_layers"],
        d_model=cfg["d_model"],
        heads=cfg["gt_heads"],
        num_samples=cfg["num_samples"],
        dropout=cfg["dropout"],
        k_pe=cfg["k_pe"],
        k_gt=cfg["k_gt"],
        eps=cfg["eps"],
        use_layer_norm=cfg["use_layer_norm"],
        node_feature_dim=None if navigator else node_feature_dim,
        pe_pool=cfg.get("pe_pool", "pe"),
        directed=cfg.get("directed", False),
        # Absent -> False, the value every pre-learn_r checkpoint was trained with;
        # a recorded True is required for the strict load to find the r_logit keys.
        learn_r=cfg.get("learn_r", False),
    )
    if not navigator:
        return pe_gt
    semantic_gt = SemanticGraphTransformer(
        node_feature_dim=cfg["d_model"], d_model=cfg["d_model"],
        num_layers=cfg["gt_num_layers"], heads=cfg["gt_heads"],
        dropout=cfg["dropout"], k_gt=cfg["k_gt"],
    )
    return TwoStagePE(pe_gt, semantic_gt)
