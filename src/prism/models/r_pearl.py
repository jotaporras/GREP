import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import Data

from prism.models.gcn import GCN


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
        use_layer_norm (bool): Whether to use layer normalization
    """

    def __init__(self,
        pe_hidden_channels,
        pe_num_layers,
        d_model,
        num_samples=30,
        dropout=0.1,
        k: int = 3,
        use_layer_norm=True,
    ):
        super().__init__()
        # Create a GCN that takes 1-dimensional random features
        self.pe_gcn = GCN(
            1, pe_hidden_channels, pe_num_layers, skip_connection=True, dropout=dropout, k=k
        )
        # Add a final projection to ensure output is d_model dimensions
        self.output_projection = nn.Linear(pe_hidden_channels, d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(d_model)
        else:
            self.batch_norm = nn.BatchNorm1d(d_model)
        self.M = num_samples

    def forward(self, data):
        # Move input data to the model's device.
        device = next(self.parameters()).device
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        X, edge_index = data.x, data.edge_index

        # Generate random node embeddings for positional encoding
        num_nodes = X.shape[0]
        generator = torch.Generator(device=device).manual_seed(
            hash(tuple(data.edge_index.flatten().tolist())) % (2**31)
        )
        Q = torch.randn((num_nodes, self.M), generator=generator, device=device)

        # Process random embeddings individually through GCN
        P_m = []

        for i in range(self.M):

            def _pe_block(q_col, edge_idx, _dummy):
                q_data = Data(x=q_col.unsqueeze(-1), edge_index=edge_idx)
                pe_local = self.pe_gcn(q_data)
                pe_local = self.dropout(pe_local)
                pe_local = self.output_projection(pe_local)
                return pe_local

            dummy = Q.new_ones(1, requires_grad=True, device=device)
            pe = checkpoint(_pe_block, Q[:, i], edge_index, dummy, use_reentrant=False)
            P_m.append(pe)
        # checkpoint

        P = torch.stack(P_m, dim=-1)
        pooled_pe = P.mean(dim=-1)
        if self.use_layer_norm:
            pooled_pe = self.layer_norm(pooled_pe)
        else:
            pooled_pe = self.batch_norm(pooled_pe)
        return pooled_pe
