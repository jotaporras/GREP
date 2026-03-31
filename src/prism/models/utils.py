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


class SparseCSRDropout(nn.Module):
    def __init__(self, p: float = 0.5, set_to_neg_inf: bool = False):
        """
        Args:
            p: Dropout probability.
            set_to_neg_inf: If True, set the dropped out values to -inf for logit-space dropout.
        """
        super().__init__()
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1]")
        self.p = p
        self.set_to_neg_inf = set_to_neg_inf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_sparse_csr:
            raise TypeError("Input must be a sparse CSR tensor")
        if not self.training or self.p == 0.0:
            return x

        keep = 1.0 - self.p
        v = x.values()
        if self.set_to_neg_inf:
            crow = x.crow_indices()
            N = x.size(0)
            row_counts = crow[1:] - crow[:-1]
            row_index = torch.arange(N, device=v.device).repeat_interleave(row_counts)
            mask = torch.rand_like(v) < keep
            kept_per_row = utils.scatter(
                mask.to(v.dtype), index=row_index, dim=0, reduce="sum", dim_size=N
            )
            need_force = (kept_per_row == 0) & (row_counts > 0)
            rand_vals = torch.rand_like(v)
            max_per_row = utils.scatter(
                rand_vals, row_index, dim=0, reduce="max", dim_size=N
            )
            is_force = need_force[row_index] & (rand_vals == max_per_row[row_index])
            mask = mask | is_force
            new_values = v.masked_fill(~mask, float("-inf"))
        else:
            if keep == 0.0:
                new_values = torch.zeros_like(v)
            else:
                mask = (torch.rand_like(v) < keep).to(v.dtype)
                new_values = (v * mask) / keep

        return torch.sparse_csr_tensor(
            crow_indices=x.crow_indices(),
            col_indices=x.col_indices(),
            values=new_values,
            size=x.size(),
            dtype=x.dtype,
            device=x.device,
        )
