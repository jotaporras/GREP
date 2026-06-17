import warnings

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import TAGConv

from prism.models.utils import LipschitzNorm


@torch.no_grad()
def _weight_spectral_norm(W: torch.Tensor, n_iters: int = 5, eps: float = 1e-12) -> torch.Tensor:
    """Largest singular value ‖W‖₂ of a [out, in] weight, via n-step power iteration.

    No-grad (a measured constant, like the σ used by weight-norm / spectral-norm).
    """
    out_dim, in_dim = W.shape
    v = torch.randn(in_dim, device=W.device, dtype=W.dtype)
    v = v / (v.norm() + eps)
    for _ in range(n_iters):
        u = W @ v
        u = u / (u.norm() + eps)
        v = W.t() @ u
        v = v / (v.norm() + eps)
    return (W @ v).norm()


def _operator_spectral_norm(op, in_shape, device, dtype, n_iters: int = 5,
                            eps: float = 1e-12) -> torch.Tensor:
    """True 2-norm of a LINEAR map ``op: R^{in_shape} → R^{out}`` via power iteration.

    The adjoint ``opᵀ`` is obtained by a vector–Jacobian product (autograd), so this
    works for ``H(S)·x = Σ_k S̄^k x H_k`` (the actual graph-filter operator, with the
    real symmetric-normalized S̄) without forming or transposing it by hand. Used by
    STEP-3 logging to MEASURE ‖H(S)‖₂ rather than assume it.
    """
    v = torch.randn(*in_shape, device=device, dtype=dtype)
    v = v / (v.norm() + eps)
    for _ in range(n_iters):
        v = v.detach().requires_grad_(True)
        w = op(v)
        wn = (w / (w.norm() + eps)).detach()
        (vt,) = torch.autograd.grad((w * wn).sum(), v)
        v = (vt / (vt.norm() + eps)).detach()
    with torch.no_grad():
        return op(v).norm() / (v.norm() + eps)


class GCN(nn.Module):
    """
    A simple TAG-based graph convolutional backbone that returns node embeddings.

    Filter normalization (R-PEARL invariant, β = 1/F):
        Each layer is a graph filter ``H(S)·x = Σ_{k=0}^{K} S̄^k x H_k`` with
        ``H_k = conv.lins[k]`` and OUTPUT width ``F = conv.out_channels`` (read from
        the layer, never hardcoded). Because S̄ is symmetric-degree-normalized
        (‖S̄‖₂ ≤ 1), ``‖H(S)‖₂ ≤ Σ_k ‖S̄^k‖₂‖H_k‖₂ ≤ Σ_k ‖H_k‖₂``. We MEASURE
        ``Σ_k ‖H_k‖₂`` (power iteration) each forward and apply a no-grad rescale
        ``r = min(1, (1/F)/Σ_k‖H_k‖₂)`` to the layer output (weight-norm style, not a
        post-hoc clamp), so the differentiable bound — hence ‖H(S)‖₂ — is ≤ 1/F. This
        REPLACES the old ξ (spectral_norm, ‖·‖₂≤1) + fixed 1/F ``_dim_scale``, which
        only gave ‖H(S)‖₂ ≤ (K+1)/F and was never measured. ``filter_norm_report``
        measures the true ‖H(S)‖₂ on the actual S̄; ``assert_filter_bounds`` is the
        fail-loud check (strict raises, lenient warns).

    Args:
        in_channels (int): Number of input features per node
        hidden_channels (int): Number of hidden features per node (= F, output width)
        num_layers (int): Number of convolution layers (must be >= 2)
        skip_connection (bool): Whether to use skip connections
        use_batch_norm (bool): Whether to use batch normalization
        k (int): Order of TAGConv polynomial (K)
        eps (float): The Lipschitz constant for layer normalization

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
        eps: float = 1e-8,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("GCN requires at least 2 layers.")

        self.convs = nn.ModuleList()
        self.k = k
        self.eps = eps
        self.convs.append(TAGConv(in_channels, hidden_channels, K=self.k))
        self.norms = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(TAGConv(hidden_channels, hidden_channels, K=self.k))
            if use_batch_norm:
                self.norms.append(nn.BatchNorm1d(hidden_channels))
            else:
                self.norms.append(LipschitzNorm(hidden_channels, eps=eps))
        self.convs.append(TAGConv(hidden_channels, hidden_channels, K=self.k))
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.skip_connection = skip_connection
        self.embedding_dim = hidden_channels

        # β = 1/F filter-norm enforcement (R-PEARL only). No spectral_norm parametrization
        # and no fixed _dim_scale: the measured per-forward rescale below does both jobs.
        self.pi_steps = 5                 # power-iteration steps (STEP 3)
        self.filter_norm_tol = 1e-3       # slack on ‖H(S)‖₂ ≤ 1/F
        self.strict_filter_norm = False   # default LENIENT (warn) in training; tests set True
        self._filter_diag: dict = {}      # per-layer measured σ-sum / rescale / bound
        self._last_graph = None           # last (edge_index, edge_weight, N) for STEP-3 measurement

    def _conv_rescale(self, conv, idx: int) -> torch.Tensor:
        """No-grad rescale r enforcing Σ_k ‖r·H_k‖₂ ≤ 1/F for one TAGConv layer."""
        F = conv.out_channels                      # output width, read from the layer
        target = 1.0 / F
        with torch.no_grad():
            sigma_sum = sum(_weight_spectral_norm(lin.weight, self.pi_steps)
                            for lin in conv.lins)
            r = torch.clamp(torch.as_tensor(target, device=sigma_sum.device,
                                            dtype=sigma_sum.dtype)
                            / sigma_sum.clamp(min=1e-12), max=1.0)
            eff = float(r * sigma_sum)             # = Σ_k ‖r·H_k‖₂  (≥ true ‖H(S)‖₂)
        self._filter_diag[idx] = {"sigma_sum": float(sigma_sum), "r": float(r),
                                  "F": F, "target": target, "bound": eff}
        if eff > target + self.filter_norm_tol:
            msg = (f"[R-PEARL β=1/F] layer {idx}: Σ‖r·H_k‖₂={eff:.3e} > 1/F={target:.3e} "
                   f"(F={F})")
            if self.strict_filter_norm:
                raise AssertionError(msg)
            warnings.warn(msg)
        return r.to(conv.lins[0].weight.dtype)

    def forward(self, data: Data):
        """
        Forward pass through the GCN.

        Args:
            data (Data): PyTorch Geometric Data object containing node features (x)
                        and edge indices (edge_index)

        Returns:
            torch.Tensor: Output node embeddings [num_nodes, hidden_channels]
        """
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = data.x.device
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        edge_weight = getattr(data, "edge_weight", None)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)
        x0, edge_index = data.x, data.edge_index
        # capture the last S̄ so the debug callback can MEASURE true ‖H(S)‖₂ on it (STEP 3).
        self._last_graph = (edge_index, edge_weight, x0.shape[0])
        x_prev = x0
        x = x0.clone()
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x_prev, edge_index, edge_weight) * self._conv_rescale(conv, i)
            if i < len(self.norms):
                x = self.norms[i](x)
            x = self.relu(x)
            x = self.dropout(x)
            if self.skip_connection and i > 0:
                x = x + x_prev
            x_prev = x
        last = len(self.convs) - 1
        x = self.convs[-1](x, edge_index, edge_weight) * self._conv_rescale(self.convs[-1], last)
        return x

    # ---- STEP 3: measured, fail-loud invariant ----------------------------------
    def filter_norm_report(self, edge_index, edge_weight, num_nodes):
        """Per-layer MEASURED ‖H(S)‖₂ (true operator norm on the actual S̄) vs target 1/F.

        Returns ``{layer_idx: {measured, target, ratio, bound}}``. Uses the same no-grad
        rescale the forward applies, so the reported norm is the operator the LLM sees.
        """
        report = {}
        for idx, conv in enumerate(self.convs):
            r = self._conv_rescale(conv, idx)

            def op(v, _conv=conv, _r=r):
                return _conv(v, edge_index, edge_weight) * _r

            with torch.enable_grad():
                measured = _operator_spectral_norm(
                    op, (num_nodes, conv.in_channels),
                    edge_index.device, conv.lins[0].weight.dtype, self.pi_steps)
            target = 1.0 / conv.out_channels
            report[idx] = {"measured": float(measured), "target": target,
                           "ratio": float(measured) / target,
                           "bound": self._filter_diag.get(idx, {}).get("bound", float("nan"))}
        return report

    def assert_filter_bounds(self, strict: bool = True, tol: float = None,
                             report: dict = None):
        """Fail-loud check that every measured ‖H(S)‖₂ ≤ 1/F + tol. strict raises; else warns.

        Pass a ``report`` from :meth:`filter_norm_report` (true norm) or fall back to the
        per-forward bound in ``self._filter_diag``.
        """
        tol = self.filter_norm_tol if tol is None else tol
        src = report if report is not None else {
            i: {"measured": d["bound"], "target": d["target"]}
            for i, d in self._filter_diag.items()}
        for idx, d in src.items():
            if d["measured"] > d["target"] + tol:
                msg = (f"[R-PEARL β=1/F] layer {idx}: ‖H(S)‖₂={d['measured']:.3e} > "
                       f"1/F+tol={d['target']:.3e}+{tol:.1e}")
                if strict:
                    raise AssertionError(msg)
                warnings.warn(msg)
