import torch
from torch import nn


class LipschitzNorm(nn.Module):
    """Normalizes the layer norm to be Lipschitz.

    Parameters
    ----------
    dim : int
        The dimension of the input tensor.
    eps : float
        The epsilon value for the Lipschitz constant.
    x : torch.Tensor
        The data passed into the forward method of the normalizer.
    """
    def __init__(self, dim, eps=1e-6, device=None):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim, device=device))
        self.eps = eps
        self.device = device

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        return self.g.clamp(0, 1) * (x / norm)