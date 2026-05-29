import warnings

import torch
from torch import nn, Tensor
from torch.nn.utils.parametrizations import spectral_norm
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, coalesce, softmax


def _maybe_spectral_norm(linear: nn.Module, enabled: bool) -> nn.Module:
    """Wrap ``linear`` in spectral norm when ``enabled``, else return it as-is.

    In the M6 fusion path the GT node features carry the LLM token embeddings X
    (semantic content). Per the spec, that path must NOT be spectrally normalized
    — only PE-side operators are — so the Q/K/V/O and FFN linears are left bare
    when token embeddings are fused. In the legacy PE-generator path (no token
    embeddings) spectral norm stays on to preserve the transferability guarantees.
    """
    return spectral_norm(linear) if enabled else linear

from prism.models.r_pearl import RandomGNNPositionalEncodings
from prism.models.utils import LipschitzNorm, SparseCSRDropout

warnings.filterwarnings("ignore", ".*Sparse CSR tensor support is in beta state.*")


class SparseGraphAttention(nn.Module):
    """
    Single-layer sparse graph transformer attention (Eq. 13 of Porras-Valenzuela).

    Computes scaled dot-product attention restricted to k-hop neighborhoods.
    Q, K, V projections are spectrally normed to enforce Assumption 2.

    Uses manual gather/scatter instead of PyG MessagePassing.propagate to
    avoid JIT-compiled propagate + autocast bf16 scatter kernel interactions
    that trigger CUDA device-side assertions on some GPU/driver combinations.

    Q, K, V projections are spectrally normed (to enforce Assumption 2) only when
    ``spectral_norm_linears`` is set; in the M6 fusion path it is disabled so the
    X-carrying path is not distorted.

    Args:
        d_model (int): Input/output feature dimension
        heads (int): Number of attention heads
        dropout (float): Attention weight dropout
        spectral_norm_linears (bool): Spectrally normalize Q/K/V/O (default True).
    """

    def __init__(self, d_model: int, heads: int = 4, dropout: float = 0.1,
                 spectral_norm_linears: bool = True):
        super().__init__()

        # Store preliminary information.
        self.heads = heads
        self.head_dim = d_model // heads
        self.d_model = d_model

        # Create Lipschitz constants for the Q, K, and V matrices.
        self.c_q = nn.Parameter(torch.tensor(1.0))
        self.c_k = nn.Parameter(torch.tensor(1.0))
        self.c_v = nn.Parameter(torch.tensor(1.0))

        # Instantiate this attention layer's query, key, and value matrices.
        self.W_Q = _maybe_spectral_norm(nn.Linear(d_model, d_model, bias=False), spectral_norm_linears)
        self.W_K = _maybe_spectral_norm(nn.Linear(d_model, d_model, bias=False), spectral_norm_linears)
        self.W_V = _maybe_spectral_norm(nn.Linear(d_model, d_model, bias=False), spectral_norm_linears)

        # Register a scale factor for the attention scores.
        self.register_buffer("scale", torch.tensor(self.head_dim, dtype=torch.float).rsqrt())

        # Register dropout and the output linear map.
        self.dropout: nn.Module = nn.Dropout(dropout)
        self.attn_dropout: nn.Module = SparseCSRDropout(dropout)
        self.W_O = _maybe_spectral_norm(nn.Linear(d_model, d_model, bias=False), spectral_norm_linears)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        N = x.shape[0]

        # Gather the query, key, and value projections for input signal x.
        q = (self.W_Q(x) * self.c_q.clamp(0, 1)).view(
            N, self.heads, self.head_dim).permute(1, 0, 2)
        k = (self.W_K(x) * self.c_k.clamp(0, 1)).view(
            N, self.heads, self.head_dim).permute(1, 0, 2)
        v = (self.W_V(x) * self.c_v.clamp(0, 1)).view(
            N, self.heads, self.head_dim).permute(1, 0, 2)

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
    """Block-diagonal sampled_addmm forward, per-head backward.

    Forward batches all heads into one (H*N, H*N) sampled_addmm for speed.
    Backward loops per-head to avoid unreliable sparse autograd on the
    block-diagonal path.
    """

    @staticmethod
    def forward(ctx, QX, KX, VX, A_csr, scale, attn_dropout, training):
        # Initialize variables.
        H, N, F_head = QX.shape
        orig_dtype = QX.dtype

        # sampled_addmm only supports float32 — cast up front.
        QX = QX.float()
        KX = KX.float()
        VX = VX.float()

        # Flatten heads into the node dimension.
        QX_flat = QX.reshape(H * N, F_head)
        KX_flat = KX.reshape(H * N, F_head)
        VX_flat = VX.reshape(H * N, F_head)

        # Build block-diagonal CSR: H copies of A_csr along the diagonal.
        crow = A_csr.crow_indices()
        col = A_csr.col_indices()
        block_crow = torch.cat([
            crow[:1],
            *(crow[1:] + i * col.shape[0] for i in range(H))
        ])
        block_col = torch.cat([col + i * N for i in range(H)])
        block_vals = torch.ones(H * col.shape[0], device=QX.device, dtype=QX.dtype)

        A_block = torch.sparse_csr_tensor(
            block_crow, block_col, block_vals, size=(H * N, H * N)
        )

        # Compute the batched sampled_addmm multi-head attention.
        unnormalized = torch.sparse.sampled_addmm(
            input=A_block, mat1=QX_flat, mat2=KX_flat.T, beta=0.0
        )

        # Apply sparse dropout and clean up the tensor dimensions.
        if training and attn_dropout is not None:
            unnormalized = attn_dropout(unnormalized)
        crow_b = unnormalized.crow_indices()
        col_b = unnormalized.col_indices()
        row_counts = crow_b[1:] - crow_b[:-1]
        row_index = torch.arange(H * N, device=QX.device).repeat_interleave(row_counts)

        # Scaled dot-product attention scores with neighborhood softmax.
        attn_scores = unnormalized.values() * scale
        attn_alpha = softmax(src=attn_scores, index=row_index, dim=0, num_nodes=H * N)

        # Construct the sparse CSR attention tensor.
        B = torch.sparse_csr_tensor(
            crow_indices=crow_b, col_indices=col_b, values=attn_alpha, size=(H * N, H * N)
        )

        # Multiply by the value tensor.
        out_flat = torch.sparse.mm(B, VX_flat)

        # Reshape the output.
        attn_out = out_flat.reshape(H, N, F_head).permute(1, 0, 2).reshape(N, H * F_head)
        attn_out = attn_out.to(orig_dtype)

        # Save for backward (keep float32 copies for backward sampled_addmm).
        ctx.save_for_backward(QX, KX, VX, A_csr.float(), scale)
        ctx.orig_dtype = orig_dtype
        ctx.attn_dropout = attn_dropout
        ctx.is_training = training

        return attn_out

    @staticmethod
    def backward(ctx, grad_output):
        QX, KX, VX, A_csr, scale = ctx.saved_tensors
        H, N, F_head = QX.shape

        # Reshape grad to (H, N, F_head).
        grad_out = grad_output.float().reshape(N, H, F_head).permute(1, 0, 2)

        grad_QX = torch.zeros_like(QX)
        grad_KX = torch.zeros_like(KX)
        grad_VX = torch.zeros_like(VX)

        # Per-head backward for reliable gradients.
        for i in range(H):
            with torch.enable_grad():
                qi = QX[i].detach().requires_grad_(True)
                ki = KX[i].detach().requires_grad_(True)
                vi = VX[i].detach().requires_grad_(True)

                # Compute sparse dot-product scores for head i.
                unnormalized = torch.sparse.sampled_addmm(
                    input=A_csr, mat1=qi, mat2=ki.T, beta=0.0
                )

                # Apply sparse attention-score dropout.
                if ctx.is_training and ctx.attn_dropout is not None:
                    unnormalized = ctx.attn_dropout(unnormalized)

                # Extract CSR structure for neighborhood softmax.
                crow_b = unnormalized.crow_indices()
                col_b = unnormalized.col_indices()
                row_counts = crow_b[1:] - crow_b[:-1]
                row_index = torch.arange(N, device=qi.device).repeat_interleave(row_counts)

                # Scaled dot-product attention scores with neighborhood softmax.
                attn_scores = unnormalized.values() * scale
                attn_alpha = softmax(src=attn_scores, index=row_index, dim=0, num_nodes=N)

                # Construct the sparse CSR attention tensor.
                B_h = torch.sparse_csr_tensor(
                    crow_indices=crow_b, col_indices=col_b, values=attn_alpha, size=(N, N)
                )

                # Multiply by the value tensor.
                out_h = torch.sparse.mm(B_h, vi)

            out_h.backward(grad_out[i])
            grad_QX[i] = qi.grad
            grad_KX[i] = ki.grad
            grad_VX[i] = vi.grad

        # 7 inputs to forward: QX, KX, VX, A_csr, scale, attn_dropout, training.
        orig_dtype = ctx.orig_dtype
        return grad_QX.to(orig_dtype), grad_KX.to(orig_dtype), grad_VX.to(orig_dtype), None, None, None, None


class SparseTransformerBlock(nn.Module):
    """
    One transformer block: attention + residual + norm + FFN + residual + norm.

    Args:
        d_model (int): Feature dimension.
        heads (int): Number of attention heads.
        dropout (float): Dropout rate.
        eps (float): Lipschitz normalization epsilon.
    """

    def __init__(self, d_model: int, heads: int = 4, dropout: float = 0.1, use_layer_norm: bool = False, eps: float = 1e-8,
                 spectral_norm_linears: bool = True):
        super().__init__()

        # Set up Attention, Dropout, Feed-Forward Network, and
        # Lipschitz Normalizer layers of Transformer Block.
        self.attn = SparseGraphAttention(d_model, heads=heads, dropout=dropout,
                                         spectral_norm_linears=spectral_norm_linears)
        self.dropout: nn.Module = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            _maybe_spectral_norm(nn.Linear(d_model, d_model), spectral_norm_linears),
            nn.LeakyReLU(),
            _maybe_spectral_norm(nn.Linear(d_model, d_model), spectral_norm_linears),
        )
        self.norms = nn.ModuleList([])
        if use_layer_norm:
            self.norms.append(LipschitzNorm(d_model, eps=eps))
            self.norms.append(LipschitzNorm(d_model, eps=eps))
        else:
            self.norms.append(nn.BatchNorm1d(d_model))
            self.norms.append(nn.BatchNorm1d(d_model))


    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # Compute attention, Lipschitz norm, FFN,
        # and another independent Lipschitz norm.
        h = self.attn(x, edge_index)
        x = self.norms[0](x + self.dropout(h))
        h = self.ffn(x)
        x = self.norms[1](x + self.dropout(h))
        return x


class GraphTransformer(nn.Module):
    """
    Full Graph Transformer with R-PEARL positional encodings.

    Pipeline: R-PEARL(graph) -> PE ⊕ node_features -> stacked SparseTransformerBlocks -> output

    M6 fusion (when token embeddings are supplied): node features are
    ``H0 = X_full + Psi``, where ``X_full`` carries the token embeddings on the
    directed-cycle (token) nodes and zeros on the scene nodes, and ``Psi`` is the
    R-PEARL encoding (the transferability-paper form ``X + Psi_G``). No gate lives
    here — the M7 cold-start gate is applied at the LLM input as
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
        eps (float): Global Lipschitz epsilon.
        use_layer_norm (bool): LipschitzNorm vs BatchNorm.
    """

    def __init__(self, num_layers: int, pe_hidden_channels: int,
                 pe_num_layers: int, d_model: int, heads: int = 4, num_samples: int = 30,
                 dropout: float = 0.1, k_pe: int = 3, k_gt: int = 3,
                 eps: float = 1e-8, use_layer_norm: bool = True,
                 probe_distribution: str = "gaussian", m_test: int = None,
                 fixed_seed_mode: bool = False, fixed_seed_value: int = 0,
                 spectral_norm_linears: bool = True):
        super().__init__()

        # Register preliminary information.
        self.k_hops = k_gt
        self.heads = heads
        self.num_layers = num_layers
        self.d_model = d_model

        # Set up R-PEARL Positional Encoder, Transformer Blocks, and Output Lipschitz Normalizer.
        # spectral_norm_linears is disabled by the M6 fusion path (token embeddings
        # fused) so the X-carrying attention/FFN linears are not spectrally normalized;
        # the PE-side R-PEARL projection keeps its own spectral norm regardless.
        self.pe_model = RandomGNNPositionalEncodings(
            pe_hidden_channels=pe_hidden_channels, pe_num_layers=pe_num_layers, d_model=d_model,
            num_samples=num_samples, dropout=dropout, k=k_pe, eps=eps, use_layer_norm=use_layer_norm,
            probe_distribution=probe_distribution, m_test=m_test,
            fixed_seed_mode=fixed_seed_mode, fixed_seed_value=fixed_seed_value,
        )
        self.blocks = nn.ModuleList([
            SparseTransformerBlock(
                d_model, heads=heads, dropout=dropout, use_layer_norm=use_layer_norm, eps=eps,
                spectral_norm_linears=spectral_norm_linears,
            ) for _ in range(num_layers)
        ])
        self.output_norm = LipschitzNorm(d_model, eps=eps)

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

        # R-PEARL positional encoding Ψ over the augmented graph.
        psi = self.pe_model(pe_data)  # [N, d_model]

        # M6 input fusion: H0 = X_full + Ψ, where X_full carries the token
        # embeddings on the directed-cycle (token) nodes and zeros on the scene
        # nodes. No gate here — the M7 gate sits at the LLM input (see
        # gated_injection.GatedInjection). When no token embeddings are supplied
        # the module behaves as a pure PE generator (legacy rpearl_gt_llm path).
        if token_embeddings is None:
            x = psi
        else:
            x_full = torch.zeros_like(psi)
            rows = slice(0, token_embeddings.shape[0]) if is_token is None else is_token
            x_full[rows] = token_embeddings.to(device=psi.device, dtype=psi.dtype)
            x = x_full + psi

        # Precompute k-hop neighborhood diffusions.
        if permutation is not None:
            khop_edge_index = self._expand_edge_index(edge_index, x.size(0))
        else:
            if not hasattr(data, '_khop_edge_index'):
                data._khop_edge_index = self._expand_edge_index(edge_index, x.size(0))
            khop_edge_index = data._khop_edge_index

        # Run signal through all Transformer Blocks.
        for block in self.blocks:
            x = block(x, khop_edge_index)

        # Apply Output Lipschitz Normalizer.
        x = self.output_norm(x)
        return x
