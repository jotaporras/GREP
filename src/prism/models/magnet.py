r"""Magnetic Chebyshev convolution for directed graphs.

:class:`MagChebConv` is adapted from
:class:`torch_geometric.nn.conv.ChebConv` (PyTorch Geometric, MIT License,
Copyright (c) PyG Team) and implements the operator introduced in Zhang et
al., "MagNet: A Neural Network for Directed Graphs", NeurIPS 2021
(arXiv:2102.11391).
"""
import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn.conv import ChebConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import OptTensor
from torch_geometric.utils import coalesce, remove_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes


def modrelu(x: Tensor, bias: Tensor, eps: float = 1e-12) -> Tensor:
    r"""Applies the modReLU function of the `"Unitary Evolution Recurrent
    Neural Networks" <https://arxiv.org/abs/1511.06464>`_ paper element-wise.

    .. math::
        \sigma(z) = \mathrm{ReLU}(|z| + b) \frac{z}{|z|}

    In contrast to :math:`\mathbb{C}\mathrm{ReLU}`, the phase of :math:`z` is
    left untouched, so that :math:`\sigma` commutes with the gauge
    :math:`z \mapsto e^{i \gamma} z` and gates the magnitude alone. The
    learnable bias :math:`b` is the radius of the deadzone; at :math:`b = 0`
    the function is the identity.
    """
    magnitude = (x.real.square() + x.imag.square() + eps).sqrt()
    b = -nn.functional.softplus(bias)
    return x * (magnitude + b).relu() / magnitude


def clin(lin: Linear, x: Tensor) -> Tensor:
    r"""Applies a real-valued linear transformation to a complex tensor."""
    return torch.complex(lin(x.real), lin(x.imag))


def global_rms(x: Tensor, eps: float = 1e-12) -> Tensor:
    r"""The single scalar :math:`\sqrt{\mathbb{E}\left[|z|^2\right]}`, the mean
    taken over every node AND every channel of :math:`\mathbf{X}`.

    Being a function of the moduli alone, it is invariant under the gauge
    :math:`z \mapsto e^{i \gamma} z` and under conjugation, so dividing by it
    commutes with both and leaves :class:`MagChebConv`'s gauge property intact.
    It is returned detached, so the division is a fixed scalar rather than a
    learned reparameterization: the layer stays a bounded linear map whose norm
    folds into the :math:`\beta` accounting of
    :mod:`prism.models.beta_projection`. There is deliberately no affine part —
    a learnable scale or shift would restore exactly the per-channel freedom
    that makes :class:`~torch.nn.LayerNorm` non-equivariant here. ``eps`` floors
    an all-zero input (an isolated node, an empty graph) away from ``0/0``.
    """
    return (x.real.square() + x.imag.square()).mean().sqrt().clamp_min(eps).detach()


class MagChebConv(ChebConv):
    r"""The magnetic Chebyshev spectral graph convolutional operator from the
    `"MagNet: A Neural Network for Directed Graphs"
    <https://arxiv.org/abs/2102.11391>`_ paper.

    .. math::
        \mathbf{Y}^{(l)} = \sum_{k=1}^{K} \mathbf{Z}_k(\mathbf{X}^{(l)};\,
        \mathbf{\mathcal{L}}) \mathbf{\Theta}_k

    where :math:`\mathbf{Z}_k` is computed recursively by

    .. math::
        \mathbf{Z}_1 &= \mathbf{X}

        \mathbf{Z}_2 &= \mathbf{S} \mathbf{X}

        \mathbf{Z}_k &= 2 \cdot \mathbf{S}
        \mathbf{Z}_{k-1} - \mathbf{Z}_{k-2}

    and :math:`\mathbf{S} = \mathbf{\hat{L}}^{(r)}` denotes the scaled
    normalized magnetic Laplacian, built from the normalized magnetic
    adjacency matrix

    .. math::
        \mathbf{\bar{H}}^{(r)} \coloneqq \mathbf{D}_s^{-1/2} \mathbf{A}_s
        \mathbf{D}_s^{-1/2} \odot \exp \left( i \mathbf{\Theta}^{(r)} \right)

    where :math:`\mathbf{A}_s = \frac{1}{2} (\mathbf{A} + \mathbf{A}^{\top})`
    denotes the symmetrized adjacency matrix, :math:`\mathbf{D}_s` its degree
    matrix, and :math:`\odot` the Hadamard product. Since
    :math:`\mathbf{\Theta}^{(r)}` is skew-symmetric,
    :math:`\mathbf{\bar{H}}^{(r)}` is Hermitian with
    :math:`\mathrm{spec}(\mathbf{\bar{H}}^{(r)}) \subseteq [-1, 1]`, so
    :math:`\mathbf{L}^{(r)} = \mathbf{I} - \mathbf{\bar{H}}^{(r)}` has
    :math:`\lambda_{\max} = 2` (Zhang et al., Theorem 2) and the identity
    cancels exactly,

    .. math::
        \mathbf{\hat{L}}^{(r)} = \frac{2 \mathbf{L}^{(r)}}{\lambda_{\max}}
        - \mathbf{I} = - \mathbf{\bar{H}}^{(r)}

    so that no rescaling by :math:`\lambda_{\max}` is required, no self-loop
    entries are needed, and :math:`\mathrm{spec}(\mathbf{\hat{L}}^{(r)})
    \subseteq [-1, 1]` as the Chebyshev recursion demands. The bound holds
    block-wise, so a disjoint union of graphs is scaled exactly as each graph
    would be on its own. Setting :obj:`shift` to :obj:`"adjacency"` filters
    over :math:`\mathbf{\bar{H}}^{(r)}` instead; since
    :math:`T_k(-x) = (-1)^k T_k(x)`, the two span the same family of filters
    and differ only by a sign on the odd-order taps.

    In contrast to :class:`~torch_geometric.nn.conv.ChebConv`, the shift
    operator is complex-valued, so that :math:`\mathbf{X}^{\prime}` aggregates
    from both :math:`\{ v : (u, v) \in \mathcal{E} \}` and
    :math:`\{ v : (v, u) \in \mathcal{E} \}`, with a phase difference between
    the two. The filter weights :math:`\mathbf{H}_{k} \in \mathbb{R}^
    {F \times G}`remain real-valued and are shared across the real and
    imaginary channels, so that aggregating along :math:`\mathbf{S}` or
    :math:`\mathbf{S}^{\top}` only conjugates :math:`\mathbf{X}^{\prime}` and
    leaves the network unchanged. The :obj:`normalization`, :obj:`batch` and
    :obj:`lambda_max` arguments of :class:`~torch_geometric.nn.conv.ChebConv`
    are unused, since the shift operator is scaled by construction.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        K (int): Chebyshev filter size :math:`K`.
        normalization (str, optional): Unused, see above.
            (default: :obj:`"sym"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        r (float, optional): The charge parameter :math:`r \in [0, 0.25]`, or
            its initial value if :obj:`learn_r` is set. (default: :obj:`0.25`)
        learn_r (bool, optional): If set to :obj:`True`, :math:`r` is learned
            jointly with the remaining parameters, constrained to
            :math:`[0, 0.25]` by a sigmoid reparameterization.
            (default: :obj:`False`)
        phase (str, optional): The phase matrix :math:`\mathbf{\Theta}^{(r)}`
            (default: :obj:`"binary"`):

            1. :obj:`"binary"`: :math:`\mathbf{\Theta}^{(r)} = 2 \pi r \,
            \mathrm{sgn}(\mathbf{A} - \mathbf{A}^{\top})`, taken after
            parallel edges are summed, so that a repeated edge does not
            widen the phase.

            2. :obj:`"weight"`: :math:`\mathbf{\Theta}^{(r)} = 2 \pi r
            (\mathbf{A} - \mathbf{A}^{\top})`

            The latter follows the paper, which assumes an unweighted graph,
            and wraps around :math:`2 \pi` whenever an edge weight exceeds
            :math:`r^{-1}`.
        shift (str, optional): The shift operator :math:`\mathbf{S}`, either
            :obj:`"laplacian"` for :math:`\mathbf{\hat{L}}^{(r)}` or
            :obj:`"adjacency"` for :math:`\mathbf{\bar{H}}^{(r)} =
            -\mathbf{\hat{L}}^{(r)}`, see above. (default: :obj:`"laplacian"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F)`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*,
          batch vector :math:`(|\mathcal{V}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, G)`
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        normalization: Optional[str] = 'sym',
        bias: bool = True,
        r: float = 0.126,
        learn_r: bool = True,
        phase: str = 'binary',
        shift: str = 'laplacian',
        **kwargs,
    ):
        super().__init__(in_channels, out_channels, K, normalization, bias,
                         **kwargs)

        assert phase in ['binary', 'weight'], 'Invalid phase'
        assert shift in ['laplacian', 'adjacency'], 'Invalid shift'

        self.phase, self.shift, self.r_const = phase, shift, r
        self.r_logit = nn.Parameter(
            torch.tensor(min(max(r, 1e-3), 0.249) / 0.25).logit(),
        ) if learn_r else None

    @property
    def r(self) -> Tensor:
        r"""The charge parameter :math:`r`, constrained to :math:`[0, 0.25]`.
        A hard clamp would leave no gradient at the upper end.
        """
        return self.r_const if self.r_logit is None else 0.25 * self.r_logit.sigmoid()

    def __norm__(
        self,
        edge_index: Tensor,
        num_nodes: Optional[int],
        edge_weight: OptTensor,
        normalization: Optional[str],
        lambda_max: OptTensor = None,
        dtype: Optional[int] = None,
        batch: OptTensor = None,
    ):
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), dtype=dtype,
                                     device=edge_index.device)

        row, col = edge_index[0], edge_index[1]
        sgn = edge_weight if self.phase == 'weight' else torch.ones_like(edge_weight)
        edge_index = torch.stack([torch.cat([row, col]), torch.cat([col, row])])
        edge_attr = torch.stack([edge_weight.repeat(2), torch.cat([sgn, -sgn])], 1)
        edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes)

        edge_index, edge_weight = gcn_norm(edge_index, edge_attr[:, 0] / 2,
                                           num_nodes, add_self_loops=False)
        asym = edge_attr[:, 1].sign() if self.phase == 'binary' else edge_attr[:, 1]
        theta = (2 * math.pi) * self.r * asym
        if self.shift == 'laplacian':  # L_hat = -H_bar, the identity cancels
            edge_weight = -edge_weight

        return edge_index, torch.complex(edge_weight * theta.cos(),
                                         edge_weight * theta.sin())

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: OptTensor = None,
        batch: OptTensor = None,
        lambda_max: OptTensor = None,
    ) -> Tensor:

        edge_index, norm = self.__norm__(
            edge_index,
            x.size(self.node_dim),
            edge_weight,
            self.normalization,
            lambda_max,
            dtype=x.real.dtype,
            batch=batch,
        )

        Tx_0 = x
        Tx_1 = x  # Dummy.
        out = clin(self.lins[0], Tx_0)

        # propagate_type: (x: Tensor, norm: Tensor)
        if len(self.lins) > 1:
            Tx_1 = self.propagate(edge_index, x=x, norm=norm)
            out = out + clin(self.lins[1], Tx_1)

        for lin in self.lins[2:]:
            Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm)
            Tx_2 = 2. * Tx_2 - Tx_0
            out = out + clin(lin, Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out


class MagNet(nn.Module):
    r"""The magnetic graph neural network from the `"MagNet: A Neural Network
    for Directed Graphs" <https://arxiv.org/abs/2102.11391>`_ paper, stacking
    :class:`MagChebConv` layers interleaved with the complex non-linearity
    :func:`modrelu`

    .. math::
        \mathbf{X}^{(\ell)} = \sigma \Big( \mathbf{Y}^{(\ell - 1)}
        \Big)

    followed by an unwind layer, which separates the real and imaginary parts
    of the final representation before applying a linear transformation

    .. math::
        \mathbf{X}^{\prime} = \mathbf{W} \left[
        \mathrm{Re}\Big(\mathbf{X}^{(L)}\Big) \, \Vert \,
        \mathrm{Im}\Big(\mathbf{X}^{(L)}\Big) \right]

    For :math:`r = 0` the phase matrix vanishes and the model reduces to
    :class:`~torch_geometric.nn.conv.ChebConv` on the symmetrized graph. Every
    layer shares one charge, so :obj:`learn_r` fits a single :math:`r`, while
    each holds its own modReLU bias.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample, which is also the
            size of each output sample.
        num_layers (int): Number of message passing layers, at least
            :obj:`2`.
        skip_connection (bool, optional): If set to :obj:`True`, adds residual
            connections between the hidden layers. (default: :obj:`False`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.5`)
        k (int): Chebyshev filter size :math:`K`. (default: :obj:`3`)
        r (float, optional): The charge parameter :math:`r \in [0, 0.25]`, or
            its initial value if :obj:`learn_r` is set. (default: :obj:`0.25`)
        learn_r (bool, optional): If set to :obj:`True`, :math:`r` is learned
            jointly with the remaining parameters, constrained to
            :math:`[0, 0.25]` by a sigmoid reparameterization.
            (default: :obj:`False`)
        phase (str, optional): The phase matrix, see
            :class:`MagChebConv`. (default: :obj:`"binary"`)
        shift (str, optional): The shift operator, see
            :class:`MagChebConv`. (default: :obj:`"laplacian"`)
        hidden_norm (str, optional): Normalization between the hidden layers
            (default: :obj:`"none"`):

            1. :obj:`"none"`: none at all. Scale is controlled by the
            :math:`\beta` projection of :mod:`prism.models.beta_projection`,
            which bounds every layer's operator norm directly.

            2. :obj:`"global_rms"`: divide by :func:`global_rms`, ONE detached
            non-affine scalar per layer.

    Shapes:
        - **input:** a :class:`~torch_geometric.data.Data` or
          :class:`~torch_geometric.data.Batch` object holding node features
          :math:`(|\mathcal{V}|, F_{in})`, edge indices
          :math:`(2, |\mathcal{E}|)`, optional edge weights
          :math:`(|\mathcal{E}|)` and an optional batch vector
          :math:`(|\mathcal{V}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F_{hidden})`
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        skip_connection: bool = False,
        dropout: float = 0.5,
        k: int = 3,
        r: float = 0.126,
        learn_r: bool = True,
        phase: str = 'binary',
        shift: str = 'laplacian',
        hidden_norm: str = 'none',
    ):
        super().__init__()

        assert num_layers >= 2, 'MagNet requires at least 2 layers'
        assert hidden_norm in ['none', 'global_rms'], 'Invalid hidden_norm'

        dims = [in_channels] + [hidden_channels] * (num_layers - 1)
        self.convs = nn.ModuleList([
            MagChebConv(i, hidden_channels, k, r=r, learn_r=learn_r,
                        phase=phase, shift=shift) for i in dims
        ])
        for conv in self.convs[1:]:  # one charge for the whole network
            conv.r_logit = self.convs[0].r_logit
        self.hidden_norm = hidden_norm
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.full((hidden_channels,), -4.6))  # softplus(-4.6) ≈ 0.01
             for _ in range(num_layers - 1)])
        self.unwind = nn.Linear(2 * hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.skip_connection = skip_connection
        self.embedding_dim = hidden_channels

    @property
    def r(self) -> Tensor:
        r"""The charge parameter :math:`r`, shared by every layer."""
        return self.convs[0].r

    def forward(self, data: Data) -> Tensor:
        device = next(self.parameters()).device
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        edge_weight = getattr(data, 'edge_weight', None)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)
        batch = getattr(data, 'batch', None)
        if batch is not None:
            batch = batch.to(device)

        x = x_prev = x + 0j
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x_prev, edge_index, edge_weight, batch)
            if self.hidden_norm == 'global_rms' and i < len(self.convs) - 2:
                x = x / global_rms(x)
            x = modrelu(x, self.biases[i]) * self.dropout(torch.ones_like(x.real))
            if self.skip_connection and i > 0:
                x = x + x_prev
            x_prev = x

        x = self.convs[-1](x, edge_index, edge_weight, batch)
        return self.unwind(torch.cat([x.real, x.imag], dim=-1))