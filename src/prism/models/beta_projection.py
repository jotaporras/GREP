r"""Post-step :math:`\beta` projection enforcing R-PEARL's bounded-filter assumption.

PEARL (`arXiv:2502.01122 <https://arxiv.org/abs/2502.01122>`_) Assumption 4.2
requires the linear operators :math:`\mathbf{H}(\mathbf{S}) = \sum_{k=0}^{K-1}
h_k \mathbf{S}^k` of each layer to be bounded, :math:`\Vert \mathbf{H}
(\mathbf{S}) \Vert \le \beta`; Theorem 4.3 (sample complexity) and Corollary 4.6
(stability) are both stated at :math:`C_\sigma = 1` and :math:`\beta = 1/F`,
where :math:`F` is the hidden width of the layer.

:class:`~prism.models.magnet.MagChebConv` realizes :math:`\mathbf{H}
(\mathbf{S}) = \sum_k T_k(\mathbf{S}) \mathbf{H}_k` on a shift operator with
:math:`\mathrm{spec}(\mathbf{S}) \subseteq [-1, 1]`, where :math:`T_k` is the
:math:`k`-th Chebyshev polynomial and :math:`\mathbf{H}_k` the :math:`k`-th tap
weight. On that spectrum :math:`\Vert T_k(\mathbf{S}) \Vert \le 1`, so

.. math::
    \Vert \mathbf{H}(\mathbf{S}) \Vert \le \sum_k \Vert \mathbf{H}_k \Vert_2

and the assumption is discharged by bounding the tap norms alone — no
eigendecomposition of :math:`\mathbf{S}`, and no dependence on the graph.

The bound is entered by a PROJECTION after the optimizer step, not by a
weight-norm reparameterization: the optimizer sees the raw weights, and the
constraint set is re-entered afterwards by one Euclidean rescale. That rescale
is a single GLOBAL scalar per layer. Per-row, per-channel or per-node scaling is
forbidden — it is no longer a filter of the form :math:`\sum_k h_k \mathbf{S}^k`
and so falls outside the parametrization Corollary 4.4 counts structures with.

CAVEAT on the value of :math:`\beta`. The paper's :math:`\mathbf{H}(\mathbf{S})`
has SCALAR coefficients :math:`h_k` — Assumption 4.1's preamble counts
:math:`F_l \cdot F_{l-1}` such filters per layer, and Corollary 4.6 says "each
layer consists of :math:`F^2` Lipschitz continuous filters". So :math:`\beta =
1/F` bounds each of the :math:`F^2` SCALAR filters, which bounds the assembled
:math:`F \times F` block operator by :math:`F \beta = 1` (each layer
non-expansive). The default here instead bounds the BLOCK operator by
:math:`1/F`, which is the stricter reading and contracts by :math:`1/F` per
layer; pass ``budget=1.0`` for the layer-non-expansive reading.
"""
import torch
from torch import nn

from prism.models.magnet import MagChebConv


def _spectral_norm(weight: torch.Tensor) -> torch.Tensor:
    r"""Largest singular value of one tap, :math:`\Vert \mathbf{H}_k \Vert_2`."""
    # `aten::_linalg_svd` has no MPS kernel: a tap is [F, G] and this detour is exact.
    src = weight.cpu() if weight.device.type == 'mps' else weight
    return torch.linalg.matrix_norm(src, ord=2).to(weight.device)


def beta_slack(conv: MagChebConv, budget: float = None) -> torch.Tensor:
    r"""The factor by which ``conv`` overshoots its budget,
    :math:`s = \sum_k \Vert \mathbf{H}_k \Vert_2 / \beta`.

    :math:`s \le 1` is the measured invariant behind Assumption 4.2. ``budget``
    is :math:`\beta`, defaulting to Corollary 4.6's :math:`1/F` read off
    ``conv.out_channels`` — :math:`F` is never hard-coded. ``ord=2`` is the true
    spectral norm (largest singular value), not a power-iteration estimate.
    """
    beta = 1.0 / conv.out_channels if budget is None else budget
    taps = torch.stack([_spectral_norm(lin.weight) for lin in conv.lins])
    return taps.sum() / beta


@torch.no_grad()
def project_beta_(module: nn.Module, budget: float = None) -> dict:
    r"""Rescale every :class:`MagChebConv` under ``module`` into
    :math:`\Vert \mathbf{H}(\mathbf{S}) \Vert \le \beta`, in place.

    Returns the PRE-projection slack per layer so a caller logs the measured
    quantity instead of assuming the invariant; a value :math:`> 1` is a layer
    that had left the constraint set. Divides the tap weights and the layer bias
    by the one scalar :math:`s`, which scales the pre-activation output by
    exactly :math:`1/s`. The charge ``r_logit`` is untouched: it parameterizes
    :math:`\mathbf{S}`, not :math:`\mathbf{H}`, so scaling it would change the
    operator rather than its norm. modReLU's bias lives on
    :class:`~prism.models.magnet.MagNet`, outside the filter, and is likewise
    untouched.
    """
    slack = {}
    for name, conv in module.named_modules():
        if not isinstance(conv, MagChebConv):
            continue
        s = beta_slack(conv, budget)
        slack[name] = float(s)
        if s <= 1.0:
            continue
        # Leaf parameters under no_grad: the same in-place rescale an optimizer does.
        for lin in conv.lins:
            lin.weight.div_(s)
        # self.lins are bias=False, so the layer bias is the only additive term.
        if conv.bias is not None:
            conv.bias.div_(s)
    return slack


def register_beta_projection(optimizer, module: nn.Module, enabled: bool = True,
                             budget: float = None):
    """Re-project ``module`` after every ``optimizer.step()``.

    ``enabled=False`` returns ``None`` having touched nothing, so the flag off is
    bit-identical to never importing this module. Otherwise projects ONCE at
    registration as well, so the invariant holds at the first forward rather than
    only from step 1 (glorot-initialized taps violate it by orders of magnitude).

    Returns the :class:`~torch.utils.hooks.RemovableHandle` of the step hook.
    """
    if not enabled:
        return None
    project_beta_(module, budget)
    return optimizer.register_step_post_hook(
        lambda opt, args, kwargs: project_beta_(module, budget))
