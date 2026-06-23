import contextlib
import warnings

import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, coalesce, softmax

from prism.models.r_pearl import RandomGNNPositionalEncodings
from prism.models.utils import SparseCSRDropout

warnings.filterwarnings("ignore", ".*Sparse CSR tensor support is in beta state.*")


class SparseGraphAttention(nn.Module):
    """
    Single-layer sparse graph transformer attention (Eq. 13 of Porras-Valenzuela).

    Computes scaled dot-product attention restricted to k-hop neighborhoods.

    Uses manual gather/scatter instead of PyG MessagePassing.propagate to
    avoid JIT-compiled propagate + autocast bf16 scatter kernel interactions
    that trigger CUDA device-side assertions on some GPU/driver combinations.

    Args:
        d_model (int): Input/output feature dimension
        heads (int): Number of attention heads
        dropout (float): Attention weight dropout
    """

    def __init__(self, d_model: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()

        # Store preliminary information.
        self.heads = heads
        self.head_dim = d_model // heads
        self.d_model = d_model

        # Instantiate this attention layer's query, key, and value matrices.
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

        # Register a scale factor for the attention scores.
        self.register_buffer("scale", torch.tensor(self.head_dim, dtype=torch.float).rsqrt())

        # Register dropout and the output linear map.
        self.dropout: nn.Module = nn.Dropout(dropout)
        self.attn_dropout: nn.Module = SparseCSRDropout(dropout)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        N = x.shape[0]

        # Gather the query, key, and value projections for input signal x.
        q = self.W_Q(x).view(N, self.heads, self.head_dim).permute(1, 0, 2)
        k = self.W_K(x).view(N, self.heads, self.head_dim).permute(1, 0, 2)
        v = self.W_V(x).view(N, self.heads, self.head_dim).permute(1, 0, 2)

        values = torch.ones(edge_index.shape[1], device=x.device, dtype=x.dtype)
        A = torch.sparse_coo_tensor(
            indices=edge_index, values=values, size=(N, N)
        ).coalesce().to_sparse_csr()

        # Perform sparse multi-head self-attention.
        out = self._mha_sparse_attention(q, k, v, A)

        # Apply the output projection and dropout.
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
    """Per-head sparse attention (memory-bounded).

    Both the forward and backward loop over heads. The previous forward batched
    all heads into one ``(H*N, H*N)`` block-diagonal ``sampled_addmm``; that
    materialised H copies of the N×N sparsity pattern (and H times the score
    nnz), an H-fold memory blow-up that OOMs on the composite graph, where
    N = token-cycle length ≈ thousands. Looping caps peak memory at a single
    head's N×N pattern plus one ``[N, F_head]`` fp32 working copy.

    The fp32 cast happens **per head and only for the sparse score op**
    (``torch.sparse.sampled_addmm`` is fp32-only). The saved tensors keep their
    original (e.g. bf16) dtype, so the backward stash is half the size of the
    old fp32 copies, and the dense ``[H, N, F_head]`` features never exist in
    fp32 all at once.
    """

    @staticmethod
    def _head_attn(qi, ki, vi, A_csr_f, scale, attn_dropout, training):
        """One head: sparse masked ``softmax(QKᵀ·scale) @ V`` over A's pattern.

        ``qi/ki/vi`` are ``[N, F_head]`` (any float dtype). Returns ``[N, F_head]``
        in fp32. ``A_csr_f`` is the fp32 CSR adjacency (shared across heads). The
        fp32 score tensor is local and freed when the head finishes.
        """
        N = qi.shape[0]
        # The sparse ops (sampled_addmm, sparse.mm) are fp32-only. .float() casts
        # the operands, but a surrounding bf16 autocast (the Graph-Transformer fusion
        # eval path) would still recast them to bf16 and raise "not implemented for
        # BFloat16", so
        # autocast is explicitly disabled for this fp32-only region.
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

        # Save the original-dtype tensors (half the size of fp32 copies); the
        # backward recasts per head, mirroring the forward.
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

        # Per-head backward: recompute the same head attention with grad enabled.
        # (Sparse autograd on the block-diagonal path is unreliable, so we drive
        # the gradient through the per-head primitive that the forward also uses.)
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

        # Set up Attention, Dropout, Feed-Forward Network, and LayerNorm layers.
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
        # Compute attention, (optional) norm, FFN, and another (optional) norm.
        # When self.normalize is False the residual adds carry through unnormalized
        # so the block preserves magnitude.
        h = self.attn(x, edge_index)
        x = self.norms[0](x + self.dropout(h)) if self.normalize else x + self.dropout(h)
        h = self.ffn(x)
        x = self.norms[1](x + self.dropout(h)) if self.normalize else x + self.dropout(h)
        return x


class GraphTransformer(nn.Module):
    """
    Full Graph Transformer with R-PEARL positional encodings.

    Pipeline: R-PEARL(graph) -> PE ⊕ node_features -> stacked SparseTransformerBlocks -> output

    Graph Transformer fusion (when token embeddings are supplied): node features are
    ``H0 = X_full + Psi``, where ``X_full`` carries the token embeddings on the
    directed-cycle (token) nodes and zeros on the scene nodes, and ``Psi`` is the
    R-PEARL encoding (the transferability-paper form ``X + Psi_G``). No gate lives
    here — the cold-start gate is applied at the LLM input as
    ``inputs_embeds = X + gate * Y[V_Tx]``. Only the token-node rows of the output
    are used downstream.

    When token embeddings are omitted the module keeps its legacy behavior — a
    pure R-PEARL-fed PE generator — so existing ``rpearl_gt_llm`` callers are
    unaffected.

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
    """

    def __init__(self, num_layers: int, pe_hidden_channels: int,
                 pe_num_layers: int, d_model: int, heads: int = 4, num_samples: int = 30,
                 dropout: float = 0.1, k_pe: int = 3, k_gt: int = 3,
                 eps: float = 1e-8, use_layer_norm: bool = True,
                 probe_distribution: str = "gaussian", m_test: int = None,
                 max_gather_rows: int = 2_000_000,
                 fixed_seed_mode: bool = False, fixed_seed_value: int = 0,
                 pe_readout: str = "mean",
                 center_second_moment: bool = True,
                 node_feature_dim: int = None):
        super().__init__()
        if pe_readout not in ("mean", "second_moment"):
            raise ValueError(
                f"pe_readout must be 'mean' or 'second_moment', got {pe_readout!r}"
            )

        # Register preliminary information.
        self.k_hops = k_gt
        self.heads = heads
        self.num_layers = num_layers
        self.d_model = d_model
        self.eps = eps
        # R-PEARL readout fed into the Graph Transformer fusion:
        #   "mean"          : H0 = X_full + Ψ,  Ψ = E_q[Φ(q)]   (first moment)
        #   "second_moment" : H0 = seeded + C·seeded, seeded = [X on token ; Ψ on scene],
        #                     C = E_q[Φ(q)Φ(q)ᵀ] applied over the full composite graph
        #                     (the proof's circulant c(n-m)); see
        #                     RandomGNNPositionalEncodings.second_moment_apply.
        self.pe_readout = pe_readout

        # Set up R-PEARL Positional Encoder and Transformer Blocks.
        self.pe_model = RandomGNNPositionalEncodings(
            pe_hidden_channels=pe_hidden_channels, pe_num_layers=pe_num_layers, d_model=d_model,
            num_samples=num_samples, dropout=dropout, k=k_pe, eps=eps, use_layer_norm=use_layer_norm,
            probe_distribution=probe_distribution, m_test=m_test,
            max_gather_rows=max_gather_rows,
            fixed_seed_mode=fixed_seed_mode, fixed_seed_value=fixed_seed_value,
            center_second_moment=center_second_moment,
            node_feature_dim=node_feature_dim,
        )
        # Blocks 0..L-2 keep their LayerNorms (stability through depth); the final
        # block is norm-free (normalize=False) so the output magnitude survives for
        # the learnable output gate below.
        self.blocks = nn.ModuleList([
            SparseTransformerBlock(
                d_model, heads=heads, dropout=dropout,
                normalize=(i < num_layers - 1),
            ) for i in range(num_layers)
        ])
        # Learnable tanh(g) gate on the GT output: g = tanh(output_gain) ∈ (-1, 1).
        # Lets the model dial the structural output up or down (toward 0 recovers the
        # base LLM) instead of pinning it to a fixed scale. Init output_gain=1.0 →
        # g ≈ 0.76. A single scalar, transferable, saved with gt_model.
        self.output_gain = nn.Parameter(torch.tensor(1.0))

    @torch.no_grad()
    def _expand_edge_index(self, edge_index: Tensor, num_nodes: int, k_hops: int = 1) -> Tensor:
        """
        Expands the edge index to the ≤k-hop neighborhood via sparse (A+I)^k.

        Uses sparse COO matrix power for O(nnz) computation, suitable for
        graphs with 1,000+ nodes.

        Args:
            edge_index (LongTensor): edge indices [2, E]
            num_nodes (int): Number of nodes
            k_hops (int | None): Override hop radius. Defaults to self.k_hops.
        """
        k = k_hops if k_hops is not None else self.k_hops

        # Convert edge index to a Sparse Tensor. Build (A + I) so that exponentiation
        # captures all nodes within k hops, not just nodes reachable in exactly k steps.
        edge_idx_self, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        # Build binary sparse adjacency (A + I) for reachability via repeated
        # sparse matmul. Binarize after each step so entries stay 0/1.
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

        # Return the coalesced edge index.
        return coalesce(expanded_edge_index, num_nodes=num_nodes)

    def forward(self, data, token_embeddings=None, is_token=None, permutation=None) -> Tensor:
        # Move input data to the Graph Transformer's device.
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

        # Graph Transformer input fusion. No gate here — the cold-start gate sits at the LLM input (see
        # gated_injection.GatedInjection).
        #   pe_readout="mean"          : H0 = X_full + Ψ, Ψ = E_q[Φ(q)] (first moment;
        #                                X_full = token embeddings on cycle rows, 0 on scene).
        #   pe_readout="second_moment" : H0 = seeded + C·seeded, where
        #                                seeded = [X on token rows ; Ψ_scene on scene rows]
        #                                and C = E_q[Φ(q)Φ(q)ᵀ] is the relative-position
        #                                operator (proof's c(n-m)) applied over the entire
        #                                composite graph (C·seeded = E_q[Φ(Φᵀ seeded)], C
        #                                never formed). Tokens get the second moment (the
        #                                first moment collapses on the vertex-transitive
        #                                cycle); scene rows carry their first-moment PE so
        #                                scene–scene structure propagates through C.
        if token_embeddings is None:
            # Legacy pure-PE-generator path (rpearl_gt_llm): first moment is the PE.
            x = self.pe_model(pe_data)
        else:
            num_nodes = pe_data.x.size(0)
            x_full = torch.zeros(num_nodes, self.d_model, device=device, dtype=torch.float32)
            rows = slice(0, token_embeddings.shape[0]) if is_token is None else is_token
            x_full[rows] = token_embeddings.to(device=device, dtype=torch.float32)
            if self.pe_readout == "second_moment":
                psi = self.pe_model(pe_data)              # first moment Ψ ∈ [N, d_model]
                seeded = x_full.clone()
                if is_token is not None:
                    # scene rows ← first-moment Ψ; token rows stay the verbal embeddings X.
                    seeded[~is_token] = psi[~is_token].to(seeded.dtype)
                cx = self.pe_model.second_moment_apply(pe_data, seeded)  # C·seeded
                x = seeded + cx.to(seeded.dtype)          # H0 = seeded + C·seeded
            else:
                x = x_full + self.pe_model(pe_data)

        # Precompute k-hop neighborhood diffusions.
        if permutation is not None:
            khop_edge_index = self._expand_edge_index(edge_index, x.size(0))
        else:
            if not hasattr(data, '_khop_edge_index'):
                data._khop_edge_index = self._expand_edge_index(edge_index, x.size(0))
            khop_edge_index = data._khop_edge_index

        # Run the transformer blocks in the LLM's low-precision dtype when token
        # embeddings are supplied (the Graph-Transformer fusion path). Training already runs the
        # GT under a bf16 autocast, but eval/generate has no autocast, so the GT
        # would otherwise hold the dense [N, 4096] attention/FFN activations in
        # fp32 — on the composite graph (N = token-cycle length ≈ thousands) that
        # OOMs. The fp32-only sparse score op casts locally (see
        # _SafeBatchedSparseAttn._head_attn), so only the dense projections/FFN
        # drop to bf16. The legacy PE-generator path (token_embeddings is None)
        # keeps full fp32, unchanged.
        amp_dtype = None
        if token_embeddings is not None and token_embeddings.dtype in (torch.float16, torch.bfloat16):
            amp_dtype = token_embeddings.dtype
            # CPU autocast supports bf16 only; never request fp16 autocast on CPU.
            if amp_dtype == torch.float16 and device.type == "cpu":
                amp_dtype = None
        if amp_dtype is not None:
            x = x.to(amp_dtype)
            amp_ctx = torch.autocast(device_type=device.type, dtype=amp_dtype)
        else:
            amp_ctx = contextlib.nullcontext()

        with amp_ctx:
            # Run signal through all Transformer Blocks. During training, activation-
            # checkpoint each block so its dense [N, d_model] attention/FFN activations
            # are recomputed in the backward pass instead of all being retained at once
            # — the GT's dominant training-memory term. use_reentrant=False
            # preserves RNG state (dropout masks match on recompute) and carries the
            # autocast dtype into the recomputed forward. At eval (no grad) this is
            # skipped: checkpoint would only add a recompute with nothing to save.
            for block in self.blocks:
                if self.training and torch.is_grad_enabled():
                    x = checkpoint(block, x, khop_edge_index, use_reentrant=False)
                else:
                    x = block(x, khop_edge_index)
            # Learnable tanh(g) output gate; the model dials the structural output's
            # magnitude itself (no ad-hoc embedding-scale matching). No-op shape-wise
            # when token_embeddings is None.
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
