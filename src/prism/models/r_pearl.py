import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import Data

from prism.models import gcn
from prism.models.utils import LipschitzNorm


class RandomGNNPositionalEncodings(nn.Module):
    """
    Random graph positional encodings (R-PEARL).

    Args:
        pe_hidden_channels (int): Hidden dimension for the GCN
        pe_num_layers (int): Number of layers in the GCN
        d_model (int): Output dimension
        num_samples (int): Number of random samples (M) to use
        dropout (float): Dropout rate of the GCN associated.
        k (int): Convolution depth of the GCN.
        use_layer_norm (bool): Whether to use Lipschitz layer normalization
        eps (float): The Lipschitz constant for layer normalization
    """

    def __init__(self,
        pe_hidden_channels,
        pe_num_layers,
        d_model,
        num_samples=30,
        dropout=0.1,
        k: int = 3,
        eps=1e-8,
        use_layer_norm=True,
    ):
        super().__init__()
        # Create a GCN that takes 1-dimensional random features
        self.pe_gcn = gcn.GCN(
            1, pe_hidden_channels, pe_num_layers,
            skip_connection=True, dropout=dropout, k=k, eps=eps
        )
        # Add a final projection to ensure output is d_model dimensions
        self.output_projection = spectral_norm(nn.Linear(pe_hidden_channels, d_model))
        
        # 1/√F scaling to satisfy PEARL Assumption 4.2 (β = 1/F) after spectral norm.
        # Spectral norm constrains ‖W‖_op ≤ 1; this additional scaling brings the
        # effective operator norm closer to 1/F as required by Theorem 4.3.
        self.register_buffer('_dim_scale', torch.tensor(d_model ** -0.5))
        self.dropout = nn.Dropout(dropout)
        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            self.norm = LipschitzNorm(d_model, eps=eps)
        else:
            self.norm = nn.BatchNorm1d(d_model)
        self.M = num_samples
        self.eps = eps

    def _batched_gcn_forward(self, Q: torch.Tensor, edge_index: torch.Tensor,
                             num_nodes: int, device=None) -> torch.Tensor:
        """Process all M random samples through the GCN in a single batched call.

        Creates M copies of the graph, each with a different random feature column,
        and processes them as a single PyG Batch for GPU-parallel execution.

        Args:
            Q: Random features [num_nodes, M].
            edge_index: Graph edge indices [2, num_edges].
            num_nodes: Number of nodes in the graph.

        Returns:
            Pooled positional encodings [num_nodes, d_model].
        """

        # Stack Q columns as a single [M*N, 1] feature with repeated edge_index.
        x_stacked = Q.T.reshape(-1, 1)

        # Build batch edge_index by offsetting
        offsets = torch.arange(self.M, device=device) * num_nodes
        edge_batch = edge_index.unsqueeze(0) + offsets.view(-1, 1, 1)
        edge_batch = edge_batch.permute(1, 0, 2).reshape(2, -1)
        batch_data = Data(x=x_stacked, edge_index=edge_batch, num_nodes=self.M * num_nodes)

        # Single batched GCN forward pass over all M copies.
        pe_all = self.pe_gcn(batch_data)
        pe_all = self.dropout(pe_all)
        pe_all = self.output_projection(pe_all) * self._dim_scale

        # Reshape [M*N, d_model] -> [M, N, d_model] and mean-pool over samples.
        pe_all = pe_all.view(self.M, num_nodes, -1)
        pooled_pe = pe_all.mean(dim=0)
        return pooled_pe

    def forward(self, data, permutation=None):
        # Move input data to the model's device.
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = data.x.device
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        X, edge_index = data.x, data.edge_index

        num_nodes = X.shape[0]
        if permutation is not None:
            edge_index = permutation.apply(edge_index, num_nodes, device=device)

        # Rademacher random signals: i.i.d. ±1 with equal probability.
        # Satisfies PEARL §4 requirements E[q]=0 and E[q^p]=1 for all even p,
        # unlike Gaussian which only satisfies p=2.
        Q = torch.randint(0, 2, (num_nodes, self.M), device=device, dtype=torch.float) * 2 - 1

        # Gradient checkpoint: recompute the batched GCN on backward to save memory.
        # The dummy tensor ensures at least one input requires grad (needed by checkpoint).
        dummy = Q.new_ones(1, requires_grad=True)
        pooled_pe = checkpoint(
            lambda q, ei, d, dev: self._batched_gcn_forward(q, ei, num_nodes, device=dev),
            Q, edge_index, dummy, device,
            use_reentrant=False,
        )

        pooled_pe = self.norm(pooled_pe)
        return pooled_pe

