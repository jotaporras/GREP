from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import TAGConv


class GCN(nn.Module):
    """
    A simple TAG-based graph convolutional backbone that returns node embeddings.

    Args:
        in_channels (int): Number of input features per node
        hidden_channels (int): Number of hidden features per node
        num_layers (int): Number of convolution layers (must be >= 2)
        skip_connection (bool): Whether to use skip connections
        use_batch_norm (bool): Whether to use batch normalization
        k (int): Order of TAGConv polynomial (K)

    Returns:
        torch.Tensor: Node embeddings of shape [num_nodes, hidden_channels]
    """

    def __init__(self,
        in_channels,
        hidden_channels,
        num_layers,
        skip_connection=False,
        use_batch_norm=False,
        dropout=0.5,
        k: int = 3,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("GCN requires at least 2 layers.")

        self.convs = nn.ModuleList()
        self.k = k
        self.convs.append(TAGConv(in_channels, hidden_channels, K=self.k))
        self.norms = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(TAGConv(hidden_channels, hidden_channels, K=self.k))
            self.norms.append(nn.LayerNorm(hidden_channels))
            if use_batch_norm:
                self.norms.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(TAGConv(hidden_channels, hidden_channels, K=self.k))
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.skip_connection = skip_connection
        self.embedding_dim = hidden_channels

    def forward(self, data: Data):
        """
        Forward pass through the GCN.

        Args:
            data (Data): PyTorch Geometric Data object containing node features (x)
                        and edge indices (edge_index)

        Returns:
            torch.Tensor: Output node embeddings [num_nodes, hidden_channels]
        """
        device = next(self.parameters()).device
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        x0, edge_index = data.x, data.edge_index
        x_prev = x0
        x = x0.clone()
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x_prev, edge_index)
            if i < len(self.norms):
                x = self.norms[i](x)
            x = self.relu(x)
            x = self.dropout(x)
            if self.skip_connection and i > 0:
                x = x + x_prev
            x_prev = x
        x = self.convs[-1](x, edge_index)
        return x
