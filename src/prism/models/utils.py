import random
from copy import deepcopy

import torch
from torch import nn


class Permutation:
    """Records a reproducible node-index permutation for equivariance experiments.

    Created with a fixed seed; the actual permutation array is generated
    on first call to ``apply`` and stored for reporting.
    """

    def __init__(self, seed: int):
        self.seed = seed
        self.perm = None

    def apply(self, edge_index: torch.Tensor, num_nodes: int, device: torch.device = None) -> torch.Tensor:
        device = device or edge_index.device
        if self.perm is None or self.perm.shape[0] != num_nodes:
            g = torch.Generator(device="cpu").manual_seed(self.seed)
            self.perm = torch.randperm(num_nodes, generator=g, device="cpu").to(device)
        return self.perm[edge_index]

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "num_nodes": len(self.perm) if self.perm is not None else None,
            "permutation": self.perm.cpu().tolist() if self.perm is not None else None,
        }

    def __repr__(self):
        if self.perm is None:
            return f"Permutation(seed={self.seed})"
        return f"Permutation(seed={self.seed}, n={len(self.perm)}, perm={self.perm.cpu().tolist()})"


def permute_graph_dict(graph_dict: dict, seed: int) -> dict:
    """Shuffle node and edge list ordering in a graph dict for text equivariance testing."""
    rng = random.Random(seed)
    out = deepcopy(graph_dict)
    for key in ("objects", "regions", "object_connections", "region_connections"):
        if key in out and isinstance(out[key], list):
            rng.shuffle(out[key])
    return out


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

    def forward_values(self, v: torch.Tensor, row_index: torch.Tensor,
                       num_rows: int) -> torch.Tensor:
        """The same dropout in nnz-space, for callers that hold the values themselves.

        :meth:`forward` has to rebuild a CSR around the result, and differentiating
        through that reconstruction materializes a DENSE ``[N, N]`` gradient — which is
        fatal once ``N`` is ``M·|V|`` (the block-diagonal probe stack under
        ``pe_pool='gt'``). A caller that only needs the values stays here, and no sparse
        tensor ever enters the autograd graph.

        Args:
            v: the ``[nnz]`` values.
            row_index: the row each value belongs to, ``[nnz]``.
            num_rows: the number of rows, for the per-row rescue below.
        """
        if not self.training or self.p == 0.0:
            return v

        keep = 1.0 - self.p
        if self.set_to_neg_inf:
            mask = torch.rand_like(v) < keep
            kept_per_row = utils.scatter(
                mask.to(v.dtype), index=row_index, dim=0, reduce="sum", dim_size=num_rows
            )
            row_counts = utils.scatter(
                torch.ones_like(v), index=row_index, dim=0, reduce="sum", dim_size=num_rows
            )
            need_force = (kept_per_row == 0) & (row_counts > 0)
            rand_vals = torch.rand_like(v)
            max_per_row = utils.scatter(
                rand_vals, row_index, dim=0, reduce="max", dim_size=num_rows
            )
            is_force = need_force[row_index] & (rand_vals == max_per_row[row_index])
            return v.masked_fill(~(mask | is_force), float("-inf"))
        if keep == 0.0:
            return torch.zeros_like(v)
        mask = (torch.rand_like(v) < keep).to(v.dtype)
        return (v * mask) / keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_sparse_csr:
            raise TypeError("Input must be a sparse CSR tensor")
        if not self.training or self.p == 0.0:
            return x

        crow = x.crow_indices()
        N = x.size(0)
        row_index = torch.arange(
            N, device=x.values().device).repeat_interleave(crow[1:] - crow[:-1])
        return torch.sparse_csr_tensor(
            crow_indices=crow,
            col_indices=x.col_indices(),
            values=self.forward_values(x.values(), row_index, N),
            size=x.size(),
            dtype=x.dtype,
            device=x.device,
        )
