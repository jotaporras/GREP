import warnings

import torch
from torch import nn, Tensor
from torch.nn.utils.parametrizations import spectral_norm
from torch.utils.checkpoint import checkpoint
from torch_geometric.utils import add_self_loops, coalesce, softmax
from torch_sparse import SparseTensor

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

        # Create Lipschitz constants for the Q, K, and V matrices.
        self.c_q = nn.Parameter(torch.tensor(1.0))
        self.c_k = nn.Parameter(torch.tensor(1.0))
        self.c_v = nn.Parameter(torch.tensor(1.0))

        # Instantiate this attention layer's query, key, and value matrices.
        self.W_Q = spectral_norm(nn.Linear(d_model, d_model, bias=False))
        self.W_K = spectral_norm(nn.Linear(d_model, d_model, bias=False))
        self.W_V = spectral_norm(nn.Linear(d_model, d_model, bias=False))

        # Register a scale factor for the attention scores.
        self.register_buffer("scale", torch.tensor(self.head_dim, dtype=torch.float).rsqrt())

        # Register dropout and the output linear map.
        self.dropout: nn.Module = nn.Dropout(dropout)
        self.attn_dropout: nn.Module = SparseCSRDropout(dropout)
        self.W_O = spectral_norm(nn.Linear(d_model, d_model, bias=False))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        N = x.shape[0]

        # Gather the query, key, and value projections for input signal x.
        q = (self.W_Q(x) * self.c_q.clamp(0, 1)).view(
            N, self.heads, self.head_dim).permute(1, 0, 2)
        k = (self.W_K(x) * self.c_k.clamp(0, 1)).view(
            N, self.heads, self.head_dim).permute(1, 0, 2)
        v = (self.W_V(x) * self.c_v.clamp(0, 1)).view(
            N, self.heads, self.head_dim).permute(1, 0, 2)

        values = torch.ones(edge_index.shape[1], device=x.device)
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
        block_vals = torch.ones(H * col.shape[0], device=QX.device)

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

        # Save for backward.
        ctx.save_for_backward(QX, KX, VX, A_csr, scale)
        ctx.attn_dropout = attn_dropout
        ctx.training = training

        return attn_out

    @staticmethod
    def backward(ctx, grad_output):
        QX, KX, VX, A_csr, scale = ctx.saved_tensors
        H, N, F_head = QX.shape

        # Reshape grad to (H, N, F_head).
        grad_out = grad_output.reshape(N, H, F_head).permute(1, 0, 2)

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
                if ctx.training and ctx.attn_dropout is not None:
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
        return grad_QX, grad_KX, grad_VX, None, None, None, None


class SparseTransformerBlock(nn.Module):
    """
    One transformer block: attention + residual + norm + FFN + residual + norm.

    Args:
        d_model (int): Feature dimension.
        heads (int): Number of attention heads.
        dropout (float): Dropout rate.
        eps (float): Lipschitz normalization epsilon.
    """

    def __init__(self, d_model: int, heads: int = 4, dropout: float = 0.1, use_layer_norm: bool = False, eps: float = 1e-8):
        super().__init__()

        # Set up Attention, Dropout, Feed-Forward Network, and
        # Lipschitz Normalizer layers of Transformer Block.
        self.attn = SparseGraphAttention(d_model, heads=heads, dropout=dropout)
        self.dropout: nn.Module = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            spectral_norm(nn.Linear(d_model, d_model)),
            nn.LeakyReLU(),
            spectral_norm(nn.Linear(d_model, d_model)),
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
                 eps: float = 1e-8, use_layer_norm: bool = True):
        super().__init__()

        # Register preliminary information.
        self.k_hops = k_gt
        self.heads = heads
        self.num_layers = num_layers

        # Set up R-PEARL Positional Encoder, Transformer Blocks, and Output Lipschitz Normalizer.
        self.pe_model = RandomGNNPositionalEncodings(
            pe_hidden_channels=pe_hidden_channels, pe_num_layers=pe_num_layers, d_model=d_model,
            num_samples=num_samples, dropout=dropout, k=k_pe, eps=eps, use_layer_norm=use_layer_norm
        )
        self.blocks = nn.ModuleList([
            SparseTransformerBlock(
                d_model, heads=heads, dropout=dropout, use_layer_norm=use_layer_norm, eps=eps
            ) for _ in range(num_layers)
        ])
        self.output_norm = LipschitzNorm(d_model, eps=eps)

    @torch.no_grad()
    def _expand_edge_index(self, edge_index: Tensor, num_nodes: int, k_hops: int = None) -> Tensor:
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

        # Use Sparse Tensor adjacency for reachability.
        adj = SparseTensor.from_edge_index(
            edge_idx_self, sparse_sizes=(num_nodes, num_nodes)
        )

        # Precompute k-hop reachable Sparse Tensor graph.
        reachable = adj
        for _ in range(k - 1):
            reachable = reachable @ adj

        # Extract edge indices from SparseTensor. torch_sparse's SparseTensor
        # uses .coo() -> (row, col, val), not .coalesce().indices().
        row, col, _ = reachable.coo()
        expanded_edge_index = torch.stack([row, col], dim=0)

        # Return the coalesced edge index.
        return coalesce(expanded_edge_index, num_nodes=num_nodes)

    def forward(self, data) -> Tensor:
        # Move input data to the Graph Transformer's device.
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = data.x.device
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)

        # Gather positional encodings as input to the Graph Transformer.
        x = self.pe_model(data)

        # Precompute k-hop neighborhood diffusions.
        if not hasattr(data, '_khop_edge_index'):
            data._khop_edge_index = self._expand_edge_index(data.edge_index, x.size(0))
        edge_index = data._khop_edge_index

        # Run signal through all Transformer Blocks.
        for block in self.blocks:
            x = block(x, edge_index)

        # Apply Output Lipschitz Normalizer.
        x = self.output_norm(x)

        # Return output.
        return x