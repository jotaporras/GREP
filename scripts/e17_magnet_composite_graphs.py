#!/usr/bin/env python
# --- headless shims (scripts/_e17_headless.py) -------------------------------------
import os as _os


def get_ipython():                      # defined only inside a kernel
    return None


def display(*args, **kwargs):           # a kernel builtin; print outside one
    for a in args:
        print(a)


def _env(name, default):
    """Env override for a notebook literal, so a smoke run needs no edit."""
    return type(default)(_os.environ.get(name, default))
# ----------------------------------------------------------------------------------
# coding: utf-8

# # E17 MagNet Composite Graphs
# 
# Author: Arush Arora

# ## Introduction

# The current codebase focuses on delivering text $\mathbf{X}$ and positional encodings $\Psi$ through separate channels to the LLM, which seems to be confounding their interleaving process. $\Psi$, the GREPs, are defined as follows:
# 
# $$\Phi = \Phi\big(\mathbf{q};\, \mathbf{S}, \mathcal{H}\big) \qquad \mathbf{P} \coloneqq \mathbb{E}_{\mathbf{q} \sim \mathcal{N}(0,\, \mathbf{I}_D)}\big[\Phi\big] \qquad \mathbf{C} \coloneqq \mathbb{E}_{\mathbf{q}}\big[\Phi\Phi^\top\big] - \mathbf{P}\mathbf{P}^\top$$
# 
# $$\mathbf{\Psi} = \Phi\big(\mathbf{X} + \mathbf{P};\, \mathcal{T}\big) \quad \text{or} \quad \mathbf{\Psi} = \Phi\bigg((\mathbf{I}_{n + c} + \mathbf{C})\begin{bmatrix}\mathbf{X} \\ \mathbf{P}\end{bmatrix};\, \mathcal{T}\bigg)$$
# 
# In this experiment, we wish to work with the Composite Graph paradigm to determine whether the MagNet architecture from the paper [“MagNet: A Neural Network for Directed Graphs” (Zhang et al., 2021)](https://arxiv.org/pdf/2102.11391) can encode the directional spectral information essential to the full Composite Graphs architecture to correct the issues that were appearing during the latest iteration of the experiment in June.
# 
# Specifically, we propose the following instantiation of $\mathbf{S}$ per the paper, the normalized complex Hermitian adjacency matrix $\bar{\mathbf{H}}^{(r)}$, which can also be seen as the Normalized Magnetic Adjacency:
# 
# $$\mathbf{S} = \bar{\mathbf{H}}^{(r)} \coloneqq \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2} \odot \exp\big(i\mathbf{\Theta}^{(r)}\big)$$
# 
# $$\mathbf{\Theta}^{(r)} \coloneqq 2 \pi r (\mathbf{A} - \mathbf{A}^\top), \quad r \ge 0$$
# 
# We thus define the Composite Graphs architecture as such:
# 
# $$\big|\mathcal{V}_\text{Tx}\big| = c, \qquad \big|\mathcal{V}_\text{Sc}\big| = n$$
# 
# $$\mathcal{G} = \big(\mathcal{V}_\text{Tx} \cup \mathcal{V}_\text{Sc}, \mathcal{E}_\text{Tx} \cup \mathcal{E}_\text{Tx} \cup \mathcal{V}_\text{Cross} \cup \mathcal{V}_\text{Mention})$$
# 
# $$\mathcal{E}_\text{Cross} = \big\{\{u, v\} : \text{$u$ is in the node label for $v$},\ u \in \mathcal{V}_\text{Tx},\ v \in \mathcal{V}_\text{Sc}\big\}$$
# 
# $$\mathcal{E}_\text{Mention} = \big\{\{u, v\} : \text{$u$ and $v$ are in same node labels},\ u, v \in \mathcal{V}_\text{Tx}\big\}$$

# ## Mathematical Overview

# ### The R-PEARL GNN

# The Random Positional Encoding (R-PEARL) GNN architecture is a PE generator that inputs white noise and processes it over an undirected graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{W})$. In this work, the graph is represented by an adjacency matrix $A$, and the GNN composes [Topology Adaptive Graph (TAG)](https://arxiv.org/abs/1710.10370) Convolutional Layers with pointwise nonlinearities (demodulators).

# #### Graph Convolutional Network (GNN)

# The code below establishes this project's implementation of a Graph Convolutional Network, which is the foundational architecture comprising R-PEARL. The equation to demonstrate the internal architecture of this NN as follows (in most cases, $\mathbf{P}(\cdot) = \mathbf{I}(\cdot)$, where $\mathbf{I}$ is the identity function):
# $$\Phi(\mathbf{X}, \mathbf{S}, \mathcal{H}) = \mathbf{X}^{(L)}$$
# $$\mathbf{X}^{(0)} = \mathbf{X} \qquad \mathbf{X}^{(l)} = \mathbf{P}\Bigg[\sigma\Bigg(\sum_{k = 0}^{K^{(l)} - 1} \mathbf{S}^k\mathbf{X}^{(l - 1)}\mathbf{H}_k^{(l)}\Bigg)\Bigg]$$

# #### Random Graph Positional Encodings (R-PEARL)

# The R-PEARL architecture extends on the GCN by instantiating it with simply one layer – a TAG Convolution and Demodulator. The mathematical equations below express the functionality of the R-PEARL network:
# 1. The white-noise matrix is sampled from the Gaussian distribution. $$\mathbf{Q} \in \mathbb{R}^{M \times N} \qquad \mathbf{Q} \sim \mathcal{N}(0, \mathbf{I}) \qquad \mathbf{Q} = \begin{bmatrix}
#   \mathbf{q}^{(0)} & \cdots & \mathbf{q}^{(m)} & \cdots & \mathbf{q}^{(M)}
#   \end{bmatrix}$$
# 
# 2. The R-PEARL network has row-vector parameter $\mathbf{H}^{(0)} \in \mathbb{R}^{1 \times D}$. It takes in each column of the white-noise matrix individually and produces a sample $\mathbf{P}^{(m)} \in \mathbb{R}^{N \times D}$, which are then pooled to form GREP $\mathbf{P}$:$$\mathbf{P}^{(m)} = \Phi\Big(\mathbf{q}^{(m)};\, \mathbf{S}, \mathcal{H}\Big) = \sigma\bigg(\sum_{k = 0}^{K = 1} \mathbf{S}^k\mathbf{q}^{(m)} {\mathbf{H}}_k\bigg)$$ $$\mathbf{P} = \mathbb{\hat{E}}\Big[\mathbf{p}^{(m)}\Big] = \frac{1}{M}\sum_{m = 1}^{M} \mathbf{P}^{(m)}$$

# ### Sparse Graph Transformer

# The Sparse Graph Transformer (hereafter named Graph Transformer or GT) follows the same architecture as that of a normal transformer, albeit that the attention mecahnism is modified to scope only over the $k$-hop neighborhood of the query node. The mathematical equations below express the functionality of the Graph Transformer:
# $$\mathbf{X}_L = \Phi\Big(\mathbf{X}_0 + \mathbb{\hat{E}}_{\mathbf{q \sim \mathcal{N}(0,\, \mathbf{I})}}\big[\Phi(\mathbf{q};\, S, \mathcal{H})\big];\, \mathcal{T}\Big)$$
# $$\mathbf{A}^{(h)}_l = \left[\frac{\exp\left[\left(\mathbf{Q}^{(h)}_l \mathbf{x}_{l-1,\ t}\right)^\top \left(\mathbf{K}^{(h)}_l \mathbf{X}_{l-1,\ U}\right)\right]}{\mathbf{1}^\top \exp\left[\left(\mathbf{Q}^{(h)}_l \mathbf{x}_{l-1,\ t}\right)^\top \left(\mathbf{K}^{(h)}_l \mathbf{X}_{l-1,\ U}\right)\right]}\right]^\top_{\begin{subarray}{l}t \in [N] \\[2.5pt] U = \mathcal{N}^{\le k}(t)\end{subarray}}$$
# $$\mathbf{Y}^{(h)}_l = \left(\mathbf{W}_o\right)^\top_l \mathbf{V}_l \mathbf{X}_{l - 1} \left(\mathbf{A}^{(h)}_l\right)^\top$$
# $$\mathbf{X}_l = \sigma\bigg(\sum_{h = 1}^H \mathbf{Y}^{(h)}_l\bigg)$$

# ### Transformer

# The Transformer architecture follows that of the Llama3.1-8B distilled PRISM model. First, the TXT file, containing the scene-graph data, is tokenized and embedded into matrices $E$ and $\tilde{X}$ as follows, where $V$ is the size of the vocabulary and $d$ is the embedding dimension.
# 
# $$\text{TXT Tokenized Data from GPT-4: } E = \begin{bmatrix}
# \mathbf{e}_1 & \mathbf{e}_2 & \overset{\mathbf{e}_t}{\cdots} & \mathbf{e}_T
# \end{bmatrix}^\top \qquad \mathbf{e}_t \in \mathbb{R}^V$$
# 
# $$\text{Embed: } X = \begin{bmatrix}
# \mathbf{x}_1 & \mathbf{x}_2 & \overset{\mathbf{x}_t}{\cdots} & \mathbf{x}_T
# \end{bmatrix}^\top \qquad \mathbf{x}_t \in \mathbb{R}^d$$
# 
# Next, the transformer operates using the equations below:
# 
# $$\mathbf{X} = \mathbf{\tilde{X}} + \mathbf{P}$$
# 
# $$\mathbf{Z}_{1:t}^{(L)} = \operatorname{Trf}\Big(\mathbf{X}_{1:t}, {\mathcal{T}}_l\Big) \qquad {\mathcal{T}}_l = \begin{bmatrix}
# \mathbf{Q}_l & \mathbf{K}_l & \mathbf{V}_l & \left(\mathbf{W}_o\right)_l
# \end{bmatrix}^\top \in \mathbb{R}^{4 \times T \times D}$$
# 
# $$\hat{\mathbf{Y}}_{t + 1} = \operatorname{Linear}\Big(\mathbf{Z}_{1:t}^{(L)}\Big) \in \mathbb{R}^V$$
# $$\text{Cross-Entropy Loss: } \mathcal{L}(E, \hat{\mathbf{Y}}) = \sum_t \sum_v e_{vt}\log{\hat{y}_t}$$

# ### Graph-Augmented LLM

# The last class that is needed to create the full GREP-PRISM architecture is the `GraphAugmentedLLM`, which simply implements the following equation as a Neural Network object in PyTorch's `torch.nn` module (referring to above equations for definitions).
# $$\mathbf{P} = \mathbb{\hat{E}}\Big[\mathbf{p}^{(m)}\Big] = \frac{1}{M}\sum_{m = 1}^{M} \Phi\Big(\mathbf{q}^{(m)}, \mathbf{S}, \mathcal{H}\Big)$$
# 
# $$\mathbf{X} = \mathbf{\tilde{X}} + \mathbf{P}$$
# 
# $${\mathbf{Z}}_{1:t}^{(L)} = \operatorname{Trf}\Big({\mathbf{X}}_{1:t}; \, \cdot \,\Big)$$

# ## Setup

# In[ ]:


# %env CUDA_VISIBLE_DEVICES=0
_ = None  # magic: 'load_ext', 'autoreload'
_ = None  # magic: 'autoreload', '2'


# In[ ]:


# Import modules.
import gc
import os
import glob
import math
import copy
import wandb
import torch
import random
import pickle
import sympy as sp
import numpy as np
import networkx as nx

from typing import Union, Optional

import torch
from torch import Tensor, nn
import matplotlib.pyplot as plt
from IPython.display import display
from torch_geometric.data import Data
from torch_geometric.nn import ChebConv
from torch.nn.utils import clip_grad_norm_
from torch_geometric.typing import OptTensor
from torch_geometric.utils import to_networkx
from torch_geometric.loader import DataLoader
from torch.distributions import Cauchy, Normal
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch_geometric.utils import to_dense_adj, to_networkx
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import coalesce, remove_self_loops

from prism.models.gt import GraphTransformer, SemanticGraphTransformer
from prism.models import gnn_llm
from prism.models.gnn_llm import (build_injection_map, find_last_graph_scope,
                                  node_token_variants)
from prism.data import compact_prompt, data, utils


# In[ ]:


# Weights & Biases setup. Mirrors prism.training.train_v3._setup_wandb (project /
# name / tags / group + full-config logging), adapted for this notebook's hand-written
# train loops. Each training stage gets its own run, grouped/tagged by GNN type so the
# R-PEARL and GT variants of the same stage line up on one W&B dashboard. The helpers
# introspect the live optimizer / scheduler / loss objects so EVERY hyperparameter is
# logged without hand-maintaining a list.
WANDB_PROJECT = 'e17-composite-graphs'


def optimizer_hparams(optimizer):
    """Every optimizer setting: class name, shared defaults, and per-param-group values
    (LRs, betas, eps, weight_decay, ...) with the parameter tensors stripped out."""
    return {
        'optimizer': type(optimizer).__name__,
        'optimizer_defaults': dict(optimizer.defaults),
        'param_groups': [
            {k: v for k, v in g.items() if k != 'params'}
            for g in optimizer.param_groups
        ],
    }


def scheduler_hparams(scheduler):
    """Every LR-scheduler setting (or {'scheduler': None} when unused)."""
    if scheduler is None:
        return {'scheduler': None}
    keys = ('mode', 'factor', 'patience', 'threshold', 'threshold_mode',
            'cooldown', 'min_lrs', 'eps')
    return {
        'scheduler': type(scheduler).__name__,
        **{k: getattr(scheduler, k) for k in keys if hasattr(scheduler, k)},
    }


def loss_hparams(loss_fn):
    """Loss class, reduction, and pos_weight (resolved to plain Python)."""
    out = {'loss_fn': type(loss_fn).__name__,
           'reduction': getattr(loss_fn, 'reduction', None)}
    pos_weight = getattr(loss_fn, 'pos_weight', None)
    if pos_weight is not None:
        out['pos_weight'] = (pos_weight.detach().cpu().tolist()
                             if torch.is_tensor(pos_weight) else pos_weight)
    return out


def init_wandb(stage, hparams):
    """Start a W&B run for a training `stage`.

    Logs the FULL run config: the GNN construction kwargs (`model_hparams`, set in the
    GNN-instantiation cell) plus every optimizer / scheduler / loss / batching
    hyperparameter the caller assembles in `hparams`. `model_type` selects R-PEARL vs
    GT and drives the run name / tag / group. Returns the run; `reinit=True` so
    successive stages in one notebook session each open a fresh run.
    """
    return wandb.init(
        project=WANDB_PROJECT,
        name=f'{stage}_{model_type}',
        tags=[stage, model_type],
        group=model_type,
        config={'model_type': model_type, 'stage': stage,
                'model': model_hparams, **hparams},
        reinit='return_previous',
    )


# In[ ]:


# Define a tensor rendering function.
def render_matrix(mat: torch.tensor, sig_figs: int = 3, decimals: int = 0):
    out = sp.Matrix(mat.detach().cpu().numpy())
    if sig_figs > 0:
        return sp.N(out, sig_figs)
    if decimals > 0:
        return out.applyfunc(lambda x: x.round(decimals))
    return out


# In[ ]:


# Standard options.
ex_path    = '../data/n_30/gen/nav100_n30_gemma_data/split/test_graphs'
train_path = '../data/n_30/gen/nav100_n30_gemma_data/split/train_graphs'
plan_path = '../data/n_30/gen/nav100_n30_gemma_data/generated_plans'
eval_path = '../data/n_100/gen/nav_n100_gemma_data/test_graphs'
save_path = '../data/pickle/e6_eval_graphs.pkl'
llm_path  = 'google/gemma-4-31B-it'
device    = 'cuda'
# The suite this run writes. suite1 is a finished run; never write into it.
suite     = _env('E17_SUITE', 'suite2')
save_path_gt = f'../outputs/e17_mag_gt/{suite}'
os.makedirs(save_path_gt, exist_ok=True)


# In[ ]:


# Setup eval infrastructure.
samples_by_graph, graph_file_by_name = data.load_samples_by_graph(ex_path)
graph_file = random.choice(list(samples_by_graph.keys()))
eval_data = samples_by_graph[graph_file]
eval_data = {graph_file: [random.choice(eval_data)]}


# In[ ]:


print(f'/Users/cyberlives/Documents/GitHub/GREP-PRISM/eval/render/revised/{graph_file}.html')


# ## Experiments

# ### §1 Pretraining a GNN to Classify Text-Scene Node Pairings

# We first hope to optimize a GNN (R-PEARL or Graph Transformer) to classify crosslink edges of the form $\{u, v\}$ where $u \in \mathcal{V}_\text{Tx}$ and $v \in \mathcal{V}_\text{Sc}$. Such a model will serve as a backbone pretrained model for fine-tuning on identifying node families given a node (including all other bucket text nodes, mention text nodes, and scene nodes). This architecture will instantiate the already implemented `GCN`, `RandomGNNPositionalEncodings`, and `GraphTransformer` architectures with the Magnetic Adjacency $\bar{\mathbf{H}}^{(r)}$ defined above.

# #### Model Definitions

# We first construct the MagNet architecture, mathematized below:

# ##### modReLU and Complex Linear Transformation

# $$z \in \mathbb{C} \qquad \mathbf{X} \in \mathbb{C}^{N \times D}$$
# 
# $$\sigma(z) = \mathrm{ReLU}\big(|z| - \mathrm{softplus}(b)\big) \frac{z}{|z|}$$
# 
# $$\mathrm{CLin}_\phi\big(\mathbf{X}\big) = \mathrm{Lin}_\phi\Big(\mathrm{Re}\big(\mathbf{X}\big)\Big) + i\bigg[\mathrm{Lin}_\phi\Big(\mathrm{Im}\big(\mathbf{X}\big)\Big)\bigg]$$
# 
# $$\textrm{GlobalRMS}(z) = \sqrt{\mathbb{E}\left[|z|^2\right]}$$

# In[ ]:


def modrelu(x: Tensor, bias: Tensor, eps: float = 1e-12) -> Tensor:
    """Applies the modReLU function"""
    magnitude = (x.real.square() + x.imag.square() + eps).sqrt()
    b = -nn.functional.softplus(bias)
    return x * (magnitude + b).relu() / magnitude


def clin(lin: Linear, x: Tensor) -> Tensor:
    """Applies a real-valued linear transformation to a complex tensor."""
    re, im = lin(x.real), lin(x.imag)
    if re.dtype == torch.bfloat16:
        re, im = re.float(), im.float()
    return torch.complex(re, im)


def global_rms(x: Tensor, eps: float = 1e-12) -> Tensor:
    """The single scalar :math:`\sqrt{\mathbb{E}\left[|z|^2\right]}`, the mean
    taken over every node AND every channel of :math:`\mathbf{X}`."""
    return (x.real.square() + x.imag.square()).mean().sqrt().clamp_min(eps).detach()


# ##### Magnetic Laplacian and Chebyshev Convolution

# $$\mathbf{A}_s = \tfrac{1}{2}\big(\mathbf{A} + \mathbf{A}^\top\big) \qquad \mathbf{D}_s = \operatorname{diag}\big(\mathbf{A}_s \mathbf{1}\big)$$
# 
# $$\mathbf{\bar{H}}^{(r)} \coloneqq \mathbf{D}_s^{-1/2} \mathbf{A}_s \mathbf{D}_s^{-1/2} \odot \exp \left( i \mathbf{\Theta}^{(r)} \right)$$
# 
# $$\mathbf{\Theta}^{(r)} = 2 \pi r \, \mathrm{sgn} (\mathbf{A} - \mathbf{A}^\top) \qquad r = \tfrac{1}{4}\sigma(\rho) \in \big[0,\, \tfrac{1}{4}\big]$$
# 
# $$\bar{\mathbf{L}}^{(r)} \coloneqq \mathbf{I}_N - \bar{\mathbf{H}}^{(r)} \qquad \lambda_{\max}\big(\bar{\mathbf{L}}^{(r)}\big) = 2$$
# 
# ---
# 
# $$\mathbf{S} = \mathbf{\hat{L}}^{(r)} = \frac{2 \bar{\mathbf{L}}^{(r)}}{\lambda_{\max}} - \mathbf{I} = -\mathbf{\bar{H}}^{(r)} \qquad \mathrm{spec}(\mathbf{S}) \subseteq [-1, 1]$$
# 
# $$\mathbf{Y}^{(\ell)} = \sum_{k=1}^{K} \mathrm{CLin}_{\mathbf{H}^{(\ell)}_k}\big(\mathbf{Z}_k\big) + \mathbf{b}^{(\ell)} \qquad \mathbf{H}_k \in \mathbb{R}^{F \times G}$$
# 
# $$\begin{align*}
# \mathbf{Z}_1 &= \mathbf{X} \\
# \mathbf{Z}_2 &= \mathbf{S} \mathbf{X} \\
# \mathbf{Z}_k &= 2 \cdot \mathbf{S} \mathbf{Z}_{k-1} - \mathbf{Z}_{k-2}
# \end{align*}$$

# In[ ]:


class MagChebConv(ChebConv):
    """The magnetic Chebyshev spectral graph convolutional operator."""
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


# ##### MagNet

# $$\Phi\Big(\mathbf{q};\, \mathbf{\hat{L}}^{(r)}, \mathcal{H}\Big) = \left[ \mathrm{Re}\Big(\mathbf{Y}^{(L)}\Big) \, \Vert \,
#         \mathrm{Im}\Big(\mathbf{Y}^{(L)}\Big) \right] \mathbf{W}_u \in \mathbb{R}^{N \times F}$$
# 
# $$\mathbf{X}^{(0)} = \mathbf{q} + 0i \qquad \mathbf{X}^{(\ell)} = \sigma \Big( \mathbf{Y}^{(\ell - 1)} \Big) + \mathbf{X}^{(\ell - 1)}\big[\ell > 1\big]$$

# In[ ]:


class MagNet(nn.Module):
    """The magnetic graph neural network."""
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


# #### Numerical Visualizations with SymPy

# We next wish to test out `MagNet` and verify that it produces coherent outputs. We will instantiate a raw MagNet and run it on all graphs.

# In[ ]:


# Prepare a graph from the data to be used in the GNN.
load_ex_graph = False

if load_ex_graph:
    with open(save_path, 'rb') as file:
        ex_graph = pickle.load(file)[4]
        N = ex_graph.num_nodes
else:
    ex_graph = utils.scene_graph_dict_to_pyg(eval_data[graph_file][0][2], 'binary')

    N, D = ex_graph.num_nodes, 1024
    ex_graph.edge_index = ex_graph.edge_index.to(device)

    EPS = 1e-12
    MAX_LENGTH = 128
    g = to_networkx(ex_graph, to_undirected=True)
    ex_graph.nxg = g
    all_pairs = dict(nx.all_pairs_dijkstra(g, weight=None))
    delta_max = max(len(path) for target in all_pairs.values() for path in target[1].values())
    paths = torch.full((N, N, delta_max if delta_max < MAX_LENGTH else MAX_LENGTH), -1).long()
    dist = torch.full((N, N), float('inf'))
    for u, (lengths_u, paths_u) in all_pairs.items():
        for v, p in paths_u.items():
            dist[u, v] = lengths_u[v]
            p = (
                torch.tensor(p, device=device).long() if len(p) < MAX_LENGTH 
                else torch.full((MAX_LENGTH,), -1, device=device).long()
            )
            paths[u, v, 0:len(p)] = p
            paths[v, u, 0:len(p)] = p.flip(0)
    dist.fill_diagonal_(EPS)
    ex_graph.diameter = delta_max
    ex_graph.paths = paths.to(device)
    ex_graph.dist = dist.to(device)

    # Topology only: BFS hop counts, never weighted, so metric distances in `dist`
    # can never reach the blurry-vision mask.
    hops = torch.full((N, N), float('inf'))
    for u, lengths_u in nx.all_pairs_shortest_path_length(g):
        for v, h in lengths_u.items():
            hops[u, v] = h
    assert torch.equal(hops[hops.isfinite()], hops[hops.isfinite()].round()), \
        'graph.hops must stay integral (unweighted BFS)'
    ex_graph.hops = hops.to(device)

    # Dense adjacency (bool).
    ex_graph.adj = to_dense_adj(
        ex_graph.edge_index, max_num_nodes=N
    ).squeeze(0).bool().to(device)

# Show the shortest paths matrix of a node in the graph.
node1 = random.randint(0, N - 1)
node2 = random.randint(0, N - 1)
render_matrix(ex_graph.paths[node1, node2][None, :], sig_figs=0)


# In[ ]:


# Instantiate THE GNN. One MagGT serves the whole notebook: §1 renders it untrained, §2
# trains it for link prediction, §3 fine-tunes it on the covariance metric. `model_type` /
# `model_hparams` are exposed at module scope so init_wandb can log the GNN config.
def create_gnn(model_type: str, **overrides):
    global model_hparams
    model_hparams = dict(
        num_layers=3,
        pe_hidden_channels=256,
        pe_num_layers=5,
        d_model=1024,
        heads=8,
        num_samples=_env('E17_NUM_SAMPLES', 320),
        max_probe_rows=16384,
        dropout=0.1,
        k_pe=3,
        k_gt=2,
        eps=1e-6,
        use_layer_norm=True,
        pe_pool='gt',
        directed=True
    )
    model_hparams.update(overrides)
    gnn = GraphTransformer(**model_hparams)
    gnn.out_features = gnn.d_model
    return gnn


model_type = 'gt'
gnn = create_gnn(model_type).to(device)
gnn


# In[ ]:


# Define a function to assemble the composite graph of a scene graph and its prompt.
def build_composite_graph(
    source,
    tokenizer,
    tasks=None,
    plan_files=(),
    edge_weights: str = 'binary',
    include_edges: bool = False,
    include_tools: bool = False,
    context_window: int = 2048,
    cycle_weight: float = 1.0,
    cycle_causal: bool = False,
    crosslink_weight: float = 0.1,
    anchor: bool = True,
    anchor_weight: float = 10.0,
    device=device,
) -> Data:
    """
    Assembles the composite graph of a scene graph and the text it is served to the
    LLM in.

    Only the TOKENIZATION half lives here: the edges themselves are laid down by
    `gnn_llm.build_composite_graph`, the same function `MagCompGraphLLM` calls, so the
    notebook and the trained architecture cannot drift apart. The defaults are the
    experiment's: `context_window` is `mask_cycle_size`, and the weights are
    `mask_cycle_weight` / `mask_crosslink_weight` / `mask_anchor_weight`.

    Args:
        source: an eval data_gen file, a scene graph dict, or an already-built PyG
            scene graph (`ex_graph`, which carries its own `raw_scene_graph`).
        tokenizer: the LLM tokenizer, whose tokenization of the prompt is V_Tx.
        tasks: the task strings the eval prompt states; read off `source` when it is
            a data_gen file, and required otherwise.
        plan_files: the generated plans of that same graph, as a glob or a list of
            paths. Given, their conversation is the text layer; omitted, the eval
            prompt is (generation turn open, no answers).
        edge_weights: 'binary' for a plain adjacency or 'gaussian' for the train-time
            affinity, as in `scene_graph_dict_to_pyg`.
        include_edges: state the edge bullets in the scene graph block of the text.
        include_tools: document the SPINE API and keep action-list plans in the text.
        context_window: the cap on c, counted from the last scene-graph block on.
        cycle_weight / crosslink_weight / anchor_weight: W on each edge family.
        cycle_causal: transpose E_Tx, so that a token reads its PREDECESSOR.
        anchor: bond t_0 → a → v_0, which is what keeps G one component.
        device: the device the assembled edges are built on.

    Returns:
        A PyG `Data` over c + n_Sc (+ 1) nodes, text first, carrying `edge_index`,
        `edge_weight`, `is_token`, `num_token_nodes`, `num_scene_nodes`, and this
        notebook's own `injection_map`, `node_names`, and windowed `input_ids`.
    """
    if isinstance(source, Data):
        graph_dict, scene = source.raw_scene_graph, source
    else:
        payload = utils.try_load_json(source) if isinstance(source, str) else source
        graph_dict = payload.get('graph', payload)
        if tasks is None and 'tasks' in payload:
            tasks = [t['task'] for t in payload['tasks']]
        scene = utils.scene_graph_dict_to_pyg(graph_dict, edge_weights=edge_weights)

    # V_Tx is the tokenized plan conversation when plans are given, else the eval prompt.
    rollouts = [utils.try_load_json(p) for p in
                (sorted(glob.glob(plan_files)) if isinstance(plan_files, str) else plan_files)]
    if rollouts:
        messages = compact_prompt.assemble_training_conversation(
            rollouts, include_edges=include_edges, include_tools=include_tools)
    elif tasks is not None:
        messages = compact_prompt.format_eval_messages(
            graph_dict, tasks, include_edges=include_edges, include_tools=include_tools)
    else:
        raise ValueError('an eval composite graph needs `tasks` (or a data_gen file carrying them)')
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=not rollouts, return_dict=False)

    # τ, and the crosslinks the LLM's own injection map scopes from it. The map stays in
    # FULL-sequence coordinates; the builder shifts and clamps it into the window.
    scope_start = find_last_graph_scope(input_ids, tokenizer)
    node_token_seqs = node_token_variants(scene.node_names, tokenizer)
    injection_map = build_injection_map(input_ids, node_token_seqs,
                                        scope_start=scope_start)

    # E_Cross runs V_Sc → V_Tx and never back: one-way is what leaves the phase
    # Θ^(r) = 2πr on exactly the edges that bind text to scene, and a symmetric pair
    # would cancel it. Not a knob here, as it is not one in `MagCompGraphLLM`.
    composite = gnn_llm.build_composite_graph(
        scene, input_ids, injection_map, scope_start=scope_start,
        context_window=context_window, device=device,
        cycle_weight=cycle_weight, cycle_causal=cycle_causal,
        crosslink_weight=crosslink_weight,
        crosslink_mention_to_node=False, crosslink_bidirectional=True,
        anchor=anchor, anchor_weight=anchor_weight,
        # The variants close the map the same way training and eval do: a label the text
        # is still writing is not a mention (`defer_open_mentions`).
        node_token_seqs=node_token_seqs,
    )
    c = composite.num_token_nodes
    composite.scope_start = scope_start
    composite.injection_map = injection_map
    composite.node_names = scene.node_names
    composite.input_ids = input_ids[scope_start:scope_start + c]

    # An uncrosslinked scene node would be a component of its own but for the anchor.
    linked = [k for k, v in injection_map.items()
              if any(s < scope_start + c for s, e in v)]
    print(f"Composite Graph: {c} text nodes, {composite.num_scene_nodes} scene nodes"
          f"{' + anchor' if anchor else ''}, {composite.edge_index.shape[1]} edges "
          f"| Crosslinked: {len(linked)}/{composite.num_scene_nodes} "
          f"| Window: {c}/{context_window} | Preamble dropped: {scope_start}")
    return composite


# In[ ]:


# Build the composite graphs of the loaded example and feed them to an untrained GNN.
tokenizer = AutoTokenizer.from_pretrained(llm_path)

# The eval arm reads the data_gen file alone; the training arm adds its generated plans.
eval_composite = build_composite_graph(
    graph_file_by_name[graph_file], tokenizer, device=device
)
# ONE plan, as `generate_data` builds them: globbing every plan gives c ~ 2900, a
# conversation five times longer than anything §2 or §3 trains on, and pe_pool='gt' runs
# the blocks over M*N nodes — 320 * 2912 does not fit a 24 GB card. The demo has to sit
# in the regime the stages actually train in.
train_composite = build_composite_graph(
    graph_file_by_name[graph_file], tokenizer, device=device,
    plan_files=sorted(glob.glob(
        f"{plan_path}/sample_{graph_file.split('_')[-1]}_*.json"))[:1],
)

# `directed=True` swaps R-PEARL's TAGConv backbone for MagNet, so S = H̄^(r); untrained.
gnn = gnn.to(device).eval()

for name, composite in (('Eval', eval_composite), ('Train', train_composite)):
    with torch.no_grad():
        out = gnn(composite).to(device)

    # Ψ is shift-invariant on E_Tx alone, so text rows decorrelate only at the crosslinks.
    text_pe, scene_pe = out[composite.is_token], out[~composite.is_token]
    shift_cos = torch.cosine_similarity(text_pe[:-1], text_pe[1:], dim=1).mean()
    print(f"{name} Composite: \n Text PE: {text_pe.norm(dim=1).mean():>0.4f}, "
          f"Scene PE: {scene_pe.norm(dim=1).mean():>0.4f} | Shift Cosine: {shift_cos:>0.4f}")

    _, _, V = torch.pca_lowrank(out.cpu(), q=10, center=True)
    out = (out - out.mean(dim=0)) @ V.to(device)
    display(render_matrix(torch.cat([out[composite.is_token][:3],
                                     out[~composite.is_token][:3]])))


# ### §2 Pretraining a GNN to Classify Edge Existence

# We first hope to optimize a GNN (a Magnetic Graph Transformer) to classify whether an edge exists in the composite graph or not. Such a model will serve as a backbone pretrained model for fine-tuning on the covariance metric of §3. The equations to represent this procedure are below:
# $$\mathbf{H} = \mathbf{\Psi} = \Phi\Big(\mathbb{\hat{E}}_{\mathbf{q}}\big[\Phi(\mathbf{q};\, S, \mathcal{H})\big];\, \mathcal{T}\Big)$$
# $$\mathbf{\hat{y}}_{ij} = \text{MLP}\big[\mathbf{h}_i\ \Vert\ \mathbf{h}_j\ \Vert\ \mathbf{h}_i \odot \mathbf{h}_j\ \Vert\ |\mathbf{h}_i - \mathbf{h}_j\|\big] \in [0, 1]$$

# #### Model Definitions

# We first define the models.

# In[ ]:


# Define a class for edge detection and instantiate it.
class GNNEdgeDetector(nn.Module):
    """
    Simple class to detect whether Node 1 and Node 2 are connected by
    applying an MLP to the Graph Positional Encodings of both nodes concatenated.
    """
    def __init__(self, gnn: GraphTransformer):
        super(GNNEdgeDetector, self).__init__()
        self.gnn = gnn
        shape = self.gnn.out_features
        self.classifier = nn.Sequential(
            nn.Linear(4 * shape, shape),
            nn.LeakyReLU(),
            nn.Linear(shape, 1),
        )
        self.graph = Data(
            x=torch.empty((0, 0), dtype=torch.float),
            edge_index=torch.empty((2, 0), dtype=torch.long)
        )
        self.cached_pe = torch.zeros(size=(1, shape))

    def forward(self, graph: Data, node1: int, node2: int):
        if self.cached_pe is None or not self.cached_pe.any() or self.graph is not graph:
            self.graph = graph
            self.cached_pe = self.gnn(self.graph)
        hi, hj = self.cached_pe[node1], self.cached_pe[node2]
        return self.classifier(torch.cat((hi, hj, hi * hj, abs(hi - hj)), dim=0))

    def invalidate_cache(self):
        self.graph = None
        self.cached_pe = None


# Instantiate the class over THE notebook's GNN, so that §2 trains the same weights
# §1 rendered and §3 goes on to fine-tune.
detector = GNNEdgeDetector(gnn).to(device)


# #### Numeric Visualizations with SymPy

# Using the `render_matrix()` function defined at the very beginning of this notebook, we explore the procedure needed to preprocess a pre-training set for the GNN to reconstruct the composite graph adjacency given a composite PyTorch `Data` object.

# In[ ]:


# Sampling options and operators.
_p = lambda v: torch.tensor(v, device=device, dtype=torch.float)
BASE_LOC, BASE_SCALE, CAUCHY_SCALE = _p(0.0), _p(0.1), _p(1.0)


# In[ ]:


# Test out the Detector.
detector.eval()
node1, node2 = random.sample(range(train_composite.num_nodes), k=2)
with torch.no_grad():
    out = detector(train_composite, node1, node2).to(device)

print(node1, node2)
render_matrix(out.sigmoid())


# #### Pre-Training of GNN on Edge Incidence

# Next, we actually preprocess and train the GNN using the steps defined above.

# In[ ]:


# Init variables. The split is over SCENE graphs and over DIRECTORIES: train and val are
# drawn from `train_path`, test from the held-out `ex_path` graphs, so no plan of a
# training scene graph can reach the test set. The run is sized in SAMPLES, each of which
# is one (scene graph, plan) conversation and so one composite graph.
plans_per_graph = _env('E17_PLANS_PER_GRAPH', 10)
train_samples, val_samples, test_samples = (_env('E17_TRAIN_SAMPLES', 100),
                                            _env('E17_VAL_SAMPLES', 20),
                                            _env('E17_TEST_SAMPLES', 20))

# Configure the datasets. `graph_file_by_name` carries the eval graphs already; the two
# directories are disjoint, so merging lets `generate_data` resolve either split's keys.
train_by_graph, train_file_by_name = data.load_samples_by_graph(train_path)
graph_file_by_name = {**graph_file_by_name, **train_file_by_name}
keys = random.sample(list(train_by_graph.keys()), k=len(train_by_graph))
train_keys = keys[:train_samples // plans_per_graph]
val_keys = keys[len(train_keys):len(train_keys) + val_samples // plans_per_graph]
test_keys = random.sample(list(samples_by_graph.keys()), k=test_samples // plans_per_graph)


# Preprocess the data.
def sample_edges(graph):
    """The balanced four-class sample: 37.5% directed true, 37.5% directed false,
    12.5% undirected true, 12.5% undirected false.

    A pair is DIRECTED when exactly one of (u, v), (v, u) is an edge — E_Tx, E_Cross and
    E_A — and UNDIRECTED when both are, which on this graph is E_Sc alone. The directed
    false pairs are the REVERSALS of the directed true ones, so those two halves are the
    same node pairs in opposite order and nothing but Θ^(r) = 2πr sgn(A - Aᵀ) separates
    them; a direction-blind model scores 62.5% here and no more. The undirected false
    pairs are adjacent in NEITHER direction and are drawn by rejection, so no N x N pool
    is ever formed. E_Sc is the scarce class, so it sizes the sample at 8 |E_Sc|.
    """
    ei, N, dev = graph.edge_index, graph.num_nodes, graph.edge_index.device
    codes = ei[0] * N + ei[1]
    sym = torch.isin(ei[1] * N + ei[0], codes)
    dir_true, und_true = ei[:, ~sym], ei[:, sym]
    n_und = min(und_true.shape[1], dir_true.shape[1] // 3)
    n_dir = 3 * n_und

    directed = dir_true[:, torch.randperm(dir_true.shape[1], device=dev)[:n_dir]]
    undirected = und_true[:, torch.randperm(und_true.shape[1], device=dev)[:n_und]]
    false_und, need = [], n_und
    while need > 0:
        u = torch.randint(N, (2 * need,), device=dev)
        v = torch.randint(N, (2 * need,), device=dev)
        ok = (u != v) & ~torch.isin(u * N + v, codes) & ~torch.isin(v * N + u, codes)
        false_und.append(torch.stack([u[ok], v[ok]])[:, :need])
        need -= false_und[-1].shape[1]

    x = torch.cat([directed, torch.stack([directed[1], directed[0]]),
                   undirected] + false_und, dim=1)
    y = torch.cat([torch.ones(n_dir, device=dev), torch.zeros(n_dir, device=dev),
                   torch.ones(n_und, device=dev), torch.zeros(n_und, device=dev)])

    return x, y


def label_edges(graph):
    """Attaches the balanced four-class sample the detector is scored on."""
    graph.edges_x, graph.edges_y = sample_edges(graph)

    return graph


def generate_data(keys, **kwargs):
    graphs = []
    for key in keys:
        plans = sorted(glob.glob(f"{plan_path}/sample_{key.split('_')[-1]}_*.json"))
        for plan in plans[:plans_per_graph]:
            graph = build_composite_graph(graph_file_by_name[key], tokenizer,
                                          plan_files=[plan], device=device, **kwargs)
            graph.input_ids = torch.tensor(graph.input_ids, device=device)
            graphs.append(graph)

    for graph in graphs:
        graph.x = Cauchy(loc=BASE_LOC, scale=CAUCHY_SCALE).sample(
            (graph.num_nodes, 1)).to(device)
        label_edges(graph)

    return graphs


def reshuffle(graphs):
    """Function for reshuffling edge data during training."""
    for graph in graphs:
        label_edges(graph)

    return graphs


train_graphs = generate_data(train_keys)
val_graphs = generate_data(val_keys)
test_graphs = generate_data(test_keys)


# In[ ]:


# Train the GNNEdgeDetector to reconstruct the graph adjacency.
batch_size = 4
val_freq = 5
epochs = _env('E17_EDGE_EPOCHS', 150)
es_patience = 1
train_edges = True

def test_loop_edges(dataloader, model, loss_fn, wandb_prefix=None, epoch=None):
    model.to(device)
    model.eval()
    size = len(dataloader.dataset)
    order = torch.randperm(size)
    test_loss, correct = 0, 0
    tp = fp = fn = tn = 0

    with torch.no_grad():
        for idx in order.tolist():
            graph = dataloader.dataset[idx]
            preds = torch.stack([
                model(graph, graph.edges_x[0, k], graph.edges_x[1, k])
                for k in range(graph.edges_x.shape[1])
            ]).squeeze(-1).to(device)
            test_loss += loss_fn(preds, graph.edges_y).item()
            true = graph.edges_y.bool()
            pred = preds > 0
            tp += (pred & true).sum().item()
            fp += (pred & ~true).sum().item()
            fn += (~pred & true).sum().item()
            tn += (~pred & ~true).sum().item()
            correct += ((preds.sigmoid() > 0.5).float() == graph.edges_y).float().mean().item()

    test_loss /= size
    correct /= size
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    bal_acc = 0.5 * (recall + tn / (tn + fp + 1e-9))
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, F1: {f1:.3f} | P: {precision:.3f} "
          f"| R: {recall:.3f} | Bal Acc: {(100*bal_acc):.1f}% | Avg loss: {test_loss:>8f} \n")
    # Log eval metrics under the given split prefix (e.g. 'val') when a run is active.
    if wandb_prefix is not None and wandb.run is not None:
        log = {
            f'{wandb_prefix}/loss': test_loss,
            f'{wandb_prefix}/accuracy': correct,
            f'{wandb_prefix}/f1': f1,
            f'{wandb_prefix}/precision': precision,
            f'{wandb_prefix}/recall': recall,
            f'{wandb_prefix}/bal_acc': bal_acc,
        }
        if epoch is not None:
            log['epoch'] = epoch
        wandb.log(log)
        # Final test metrics: also surface as run-summary headline numbers.
        if wandb_prefix == 'test':
            wandb.run.summary.update({k: v for k, v in log.items() if k != 'epoch'})
    return test_loss, f1


def train_loop_edges(train_dataloader, val_dataloader, test_dataloader, model,
               loss_fn, optimizer, scheduler, batch_size=20, epochs=50):
    size = len(train_dataloader.dataset)
    model.to(device)
    model.train()
    val_loss: float = 0
    best_val, best_state, bad_runs = float('inf'), None, 0
    # Log the full run config: GNN (model_hparams) + optimizer/scheduler/loss/batching.
    run = init_wandb('edge_detection', {
        'batch_size': batch_size, 'epochs': epochs,
        'val_freq': val_freq, 'es_patience': es_patience,
        'plans_per_graph': plans_per_graph,
        **optimizer_hparams(optimizer),
        **scheduler_hparams(scheduler),
        **loss_hparams(loss_fn),
    })
    global_step = 0
    for i in range(epochs):
        # Validation loop.
        reshuffle(val_dataloader.dataset)
        if i % val_freq == 0:
            print(f"=============\nValidation #{i // val_freq + 1}\n=============")
            val_loss, _ = test_loop_edges(val_dataloader, model, loss_fn, wandb_prefix='val', epoch=i)
            if scheduler:
                scheduler.step(val_loss)
            if val_loss < best_val - 1e-3:
                best_val, bad_runs = val_loss, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad_runs += 1
                if bad_runs >= es_patience:
                    print(f"Early stop at epoch {i} (best val {best_val:>8f})")
                    break
            model.train()

        print(f"=============\nEpoch #{i + 1}\n=============")
        optimizer.zero_grad()
        pending = 0
        order = torch.randperm(size)
        reshuffle(train_dataloader.dataset)
        for j, idx in enumerate(order.tolist()):
            # Compute prediction and loss.
            graph = train_dataloader.dataset[idx]
            preds = torch.stack([
                model(graph, graph.edges_x[0, k], graph.edges_x[1, k])
                for k in range(graph.edges_x.shape[1])
            ]).squeeze(-1).to(device)
            loss = loss_fn(preds, graph.edges_y)

            # Backpropagation.
            (loss / batch_size).backward()
            pending += 1
            model.invalidate_cache()

            # Optimization and results.
            if pending == batch_size:
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                pending = 0

                global_step += 1
                loss, current = loss.item(), j
                wandb.log({
                    'train/loss': loss,
                    'train/lr': optimizer.param_groups[0]['lr'],
                    'epoch': i,
                    'global_step': global_step,
                })
                print(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        if pending:
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            pending = 0

    # Early stopping hatch.
    if best_state is not None:
        model.load_state_dict(best_state)

    # Test the finished model.
    test_loop_edges(test_dataloader, model, loss_fn, wandb_prefix='test')
    run.finish()


loss_fn = nn.BCEWithLogitsLoss()
train_dataloader = DataLoader(train_graphs, batch_size=batch_size)
val_dataloader = DataLoader(val_graphs, batch_size=batch_size)
test_dataloader = DataLoader(test_graphs, batch_size=batch_size)
if train_edges:
    optimizer = torch.optim.AdamW([
        {'params': detector.gnn.parameters(), 'lr': 3e-5},
        {'params': detector.classifier.parameters(), 'lr': 3e-4},
    ], betas=(0.9, 0.95), weight_decay=0.05)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    train_loop_edges(train_dataloader, val_dataloader, test_dataloader, detector,
                     loss_fn, optimizer, scheduler, batch_size=batch_size, epochs=epochs)
    torch.save(gnn.state_dict(), f'{save_path_gt}/mag_gt.pt')
    torch.save(detector.classifier.state_dict(), f'{save_path_gt}/detector.pt')
    del optimizer, scheduler
    gc.collect()


# #### Evaluation of Pre-Trained GNN on Edge Incidence

# We now test the trained model on the evaluation dataset. First, we we will render the output for clarity.

# In[ ]:


# Test out the Detector.
detector.eval().to(device)
detector.invalidate_cache()
node1, node2 = random.sample(range(train_composite.num_nodes), k=2)
with torch.no_grad():
    out = detector(train_composite, node1, node2)

print(node1, node2)
render_matrix(out.sigmoid())


# ### §3 Pretraining a GT to Reproduce the Composite Graph's Effective Resistances

# We now wish to optimize the pre-trained GNN (a Magnetic Graph Transformer) so that the second moment of its probe response reproduces the effective resistance between every pair of token nodes of the composite graph. Such a model will deliver a covariance whose token block $\mathbf{C}_\text{tok}$ carries the structure of $\mathcal{G}$ into the attention bias $\beta \mathbf{C}_\text{tok}$ of the introduction. The equations to represent this procedure are below:
# 
# $$\mathbf{\Phi}'(\mathbf{q}) = T\Big(\Phi\big(\mathbf{q};\, \mathbf{\hat{L}}^{(r)}, \mathcal{H}\big)\Big) \qquad \mathbf{\Psi} = \mathbb{\hat{E}}_{\mathbf{q} \sim \mathcal{N}(0,\, \mathbf{I})}\big[\mathbf{\Phi}'\big] \qquad \mathbf{C} = \mathbb{\hat{E}}_{\mathbf{q}}\big[\mathbf{\Phi}' \mathbf{\Phi}'^\top\big] - \mathbf{\Psi}\mathbf{\Psi}^\top$$
# 
# $$R^{(r)}_\text{eff}(u,\, v) \coloneqq \big(\mathbf{e}_u - \mathbf{e}_v\big)^{\mathsf{H}} \big(\mathbf{\bar{L}}^{(r)}\big)^{\dagger} \big(\mathbf{e}_u - \mathbf{e}_v\big) = \bar{L}^\dagger_{uu} + \bar{L}^\dagger_{vv} - 2\operatorname{Re}\big(\bar{L}^\dagger_{uv}\big)$$
# 
# $$\text{Mean-Squared Error Loss: } \mathcal{L}\big(\mathbf{C}_\text{tok}\big) = \sum_{n = 1}^{c}\sum_{m = 1}^{c} \Big(\mathbf{C}_{nn} + \mathbf{C}_{mm} - 2\mathbf{C}_{nm} - \alpha R^{(r)}_\text{eff}\big(u(n),\, v(m)\big)\Big)^2$$
# 
# The rows of $\mathbf{C}_\text{tok} = \mathbf{C}[{:}c,\, {:}c]$ are the token nodes under the text-first layout, so $u(n) = n$, and $R^{(r)}_\text{eff}$ is read off the FULL composite graph before the block is taken: a non-mention token still inherits scene context by diffusion through $\mathcal{E}_\text{Cross}$. Since $\mathbf{\Theta}^{(r)}$ is skew-symmetric, $\mathbf{\bar{L}}^{(r)}$ and its pseudoinverse are Hermitian, and the quadratic form above is therefore real: the imaginary part of $\bar{L}^\dagger_{uv}$ cancels against that of $\bar{L}^\dagger_{vu}$, which the literal transcription $\bar{L}^\dagger_{uu} + \bar{L}^\dagger_{vv} - 2\bar{L}^\dagger_{uv}$ does not. It is also non-negative, since $\operatorname{spec}\big(\mathbf{\bar{H}}^{(r)}\big) \subseteq [-1, 1]$ (Zhang et al., Theorem 2) gives $\mathbf{\bar{L}}^{(r)} \succeq 0$, so that both sides of the loss are squared distances. The quantity the loss fits is the variance across probes of the difference of two rows, which is blind to the first moment $\mathbf{\Psi}$:
# 
# $$\mathbf{C}_{nn} + \mathbf{C}_{mm} - 2\mathbf{C}_{nm} = \mathbb{\hat{E}}_{\mathbf{q}}\Big[\big\Vert\big(\mathbf{\Phi}'_n - \mathbf{\Phi}'_m\big) - \big(\boldsymbol{\psi}_n - \boldsymbol{\psi}_m\big)\big\Vert_2^2\Big] \ge 0$$
# 
# Specifically, we take the probe expectation OUTSIDE the Transformer blocks, so that $\mathbf{\Phi}' = T(\Phi)$ is the block output per probe and both moments are taken over it. $T$ is nonlinear, so $\mathbb{E}_\mathbf{q}\big[T(\Phi)\big] \ne T\big(\mathbb{E}_\mathbf{q}[\Phi]\big)$, and this is the already implemented `pe_pool = 'gt'` path of `covariance_token_block`, which chunks the probe axis rather than materializing $\mathbf{\Phi}' \in \mathbb{R}^{M \times N \times D}$. We hold $\alpha$ and the charge $r$ fixed for this stage: a learnable $\alpha$ would be minimized by $(\mathbf{C},\, \alpha) \to (\mathbf{0},\, 0)$, and a learnable $r$ would let the model move $R^{(r)}_\text{eff}$ rather than meet it. Centering leaves $\operatorname{rank}(\mathbf{C}) \le \min\big(N,\, (M - 1)D\big)$, which at $M = 32$ and $D = 1024$ is $31{,}744$, far above $N$, so the metric is not rank-limited and the cell below states the bound rather than assuming it.
# 
# We assemble the composite graph exactly as `MagCompGraphLLM` does, through the same `gnn_llm.build_composite_graph`. $\mathcal{V}_\text{Tx}$ is the window of at most $c_{\max} = 2048$ tokens counted from the last scene graph block on, since the system preamble and any ICL graphs are text no crosslink reaches. $\mathcal{E}_\text{Tx}$ is the directed cycle $i \to i + 1 \bmod c$, $\mathcal{E}_\text{Sc}$ carries both directions and therefore magnitude alone, and $\mathcal{E}_\text{Cross}$ runs $\mathcal{V}_\text{Sc} \to \mathcal{V}_\text{Tx}$ and never back, so that the arrowheads land on the cycle and the phase $\Theta^{(r)} = 2 \pi r$ survives on exactly the edges that bind text to scene; a symmetric pair would cancel it. The anchor bond $t_0 \to a \to v_0$ adds one node and two edges, which is what keeps $\mathcal{G}$ one component whatever $\mathcal{E}_\text{Cross}$ covers, and so keeps every $R^{(r)}_\text{eff}(u, v)$ finite without reshaping it as a fan to all of $\mathcal{V}_\text{Sc}$ would.

# In[ ]:


# Define the effective-resistance target and the covariance metric that must match it.
@torch.no_grad()
def magnetic_resistance(graph, conv) -> Tensor:
    """R_eff(u, v) = (e_u - e_v)^H (L̄^(r))† (e_u - e_v) over the whole composite graph.

    The shift operator is `conv`'s OWN: `MagChebConv.__norm__` returns -H̄^(r) under
    shift='laplacian', so L̄^(r) is the identity plus what it hands back, and the target
    is the resistance of the very operator the network filters over. Hermitian L̄^(r)
    makes the form real — Im(L̄†_uv) cancels against L̄†_vu — and L̄^(r) ⪰ 0 makes it a
    squared distance; the clamp only absorbs the rounding of that cancellation.
    """
    n = graph.num_nodes
    edge_index, shift = conv.__norm__(graph.edge_index, n, graph.edge_weight,
                                      conv.normalization, dtype=torch.float)
    lap = torch.eye(n, dtype=torch.complex128, device=graph.edge_index.device)
    lap[edge_index[0], edge_index[1]] += shift.to(torch.complex128)
    pinv = torch.linalg.pinv(lap, hermitian=True)
    diag = pinv.diagonal().real
    return (diag[:, None] + diag[None, :] - 2 * pinv.real).clamp_min(0).float()


def gram_distances(gram: Tensor) -> Tensor:
    """d²(n, m) = C_nn + C_mm - 2C_nm for every pair — the squared distance C_tok
    induces, and the quantity the attention bias reads up to the row-constant that
    the softmax cancels. `torch.cdist` would route it through a square root whose
    gradient is undefined on the zero diagonal."""
    diag = gram.diagonal()
    return (diag[:, None] + diag[None, :] - 2 * gram).clamp_min(0)


def probe_covariance(model, graph) -> Tensor:
    """C_tok = E_q[Φ'Φ'ᵀ] - ΨΨᵀ over the c token rows, for Φ' = T(Φ(q; L̄^(r), H)).

    This is the call `MagCompGraphLLM.covariance_token_block` makes, so the notebook
    and the trained architecture form the SAME matrix: `pe_pool='gt'` puts the blocks
    inside E_q, per probe, and the probe axis is chunked and checkpointed there rather
    than materialized as [M, N, D]. Probes run on the full composite graph, so the
    token rows keep reading the scene through E_Cross; the slice is taken after.
    """
    C_tok, _ = model.pe_model.covariance_token_block(
        graph, graph.num_token_nodes, pe_pool=model.pe_pool, gt=model)
    return C_tok


def check_composite(graph, c) -> None:
    """The four edge classes, asserted rather than assumed (A3-A7 of the design note)."""
    ei = {(int(u), int(v)) for u, v in graph.edge_index.t().tolist()}
    n_sc, a, tau = graph.num_scene_nodes, c + graph.num_scene_nodes, graph.scope_start
    # The injection map is kept in FULL-sequence coordinates; the rows are the window.
    mentions = {j: sorted({p - tau for s, e in spans
                           for p in range(max(s, tau), min(e, tau + c))})
                for j, spans in graph.injection_map.items()}
    assert graph.num_nodes == c + n_sc + 1, 'N = c + n_Sc + 1'
    assert all((i, (i + 1) % c) in ei and ((i + 1) % c, i) not in ei for i in range(c)), \
        'E_Tx must be the DIRECTED cycle i -> i + 1'
    assert all((c + j, p) in ei and (p, c + j) not in ei
               for j, ps in mentions.items() for p in ps), \
        'E_Cross must run V_Sc -> V_Tx and never back'
    assert {(u, v) for u, v in ei if a in (u, v)} == {(0, a), (a, c)}, \
        'E_A must be exactly t_0 -> a -> v_0'
    # A3: the four classes partition E, so no stray edge rides along unnoticed.
    scene_e = sum(1 for u, v in ei if c <= u < a and c <= v < a)
    assert graph.edge_index.shape[1] == (
        c + scene_e + sum(len(ps) for ps in mentions.values()) + 2), \
        '|E| = c + |E_Sc| + sum_j |M_j| + 2'


# C_tok is the encoding this stage shapes, over the SAME gnn §2 trained. pe_pool='gt' is
# what puts T inside E_q; R_eff is evaluated at the charge as it stands when the targets
# are built, so a charge that drifts far during training leaves them stale.
conv = gnn.pe_model.pe_gcn.convs[0]
assert model_hparams['directed'], 'the resistance target is defined by the magnetic shift'
assert model_hparams['pe_pool'] == 'gt', 'C is the second moment of T(Φ), not of Φ'

# Read off a TRAIN graph, not `graph_file`: that one is drawn from the eval pool
# `test_keys` samples, and α is a scalar the training loss uses. One plan, as
# `generate_data` builds them, so these diagnostics describe the regime §3 trains in.
assert train_keys, 'train_samples // plans_per_graph rounded to zero train keys'
res_key = train_keys[0]
res_composite = build_composite_graph(
    graph_file_by_name[res_key], tokenizer, device=device,
    plan_files=sorted(glob.glob(
        f"{plan_path}/sample_{res_key.split('_')[-1]}_*.json"))[:1],
)
c_res = res_composite.num_token_nodes
check_composite(res_composite, c_res)
R = magnetic_resistance(res_composite, conv)[:c_res, :c_res].double()
print(f"R_eff: {tuple(R.shape)} | Charge: {conv.r:.4f} | Asymmetry: {(R - R.T).abs().max():.2e} "
      f"| Diagonal: {R.diagonal().abs().max():.2e} | Min: {R.min():.4f}, "
      f"Mean: {R.mean():.4f}, Max: {R.max():.4f}")

# The charge margin δ(r, c) = dist(2rc, Z), as `ChargeDegeneracyCallback` reports it:
# at δ = 0 the cycle's eigenvalues collide and the direction of E_Tx is destroyed.
s = 2 * float(conv.r) * c_res
print(f"Charge margin: δ(r, c) = {min(s % 1, 1 - s % 1):.4f} at 2rc = {s:.3f}")

# C is a centered sample covariance of M probe responses of width D, so
# rank(C) ≤ min(N, (M - 1)D). The metric is rank-limited only if that bound bites.
rank = min(res_composite.num_nodes,
           (model_hparams['num_samples'] - 1) * model_hparams['d_model'])
print(f"Rank bound: min(N, (M - 1)D) = {rank} against N = {res_composite.num_nodes} "
      f"| {'NOT rank-limited' if rank >= res_composite.num_nodes else 'RANK-LIMITED'}")

# C_tok as §2 leaves it, to state the scale the loss starts from and the memory it costs.
with torch.no_grad():
    C = probe_covariance(gnn.eval(), res_composite).double()
assert C.trace() > 0, 'a zero C_tok would leave the metric with nothing to shape'

# α is the unit conversion between R_eff and the scale C_tok is written in, and it is
# MEASURED at §3's STARTING POINT — the §2-pretrained gnn, not an untrained one — rather
# than chosen. C's width is set by the blocks, which know nothing of R: at α = 1 the
# measured d²(C) sits about two orders of magnitude above αR_eff, and MSE would spend its
# budget collapsing that scale rather than shaping the structure the stage is for. Fixing
# α to the ratio the model already holds starts the loss on the structure instead.
# Measured on `res_composite` alone — one graph fixes a scalar to O(1).
ALPHA = (gram_distances(C).mean() / R.mean()).item()
assert math.isfinite(ALPHA) and ALPHA > 0, 'α must be a finite positive scale'
print(f"α: {ALPHA:.4f} (calibrated: d²(C) mean / R_eff mean at §3 start)")

print(f"C_tok: {tuple(C.shape)} | Trace: {C.trace():.4f} | Asymmetry: "
      f"{(C - C.T).abs().max():.2e} | d²(C) mean: {gram_distances(C).mean():.4f} "
      f"against αR_eff mean: {ALPHA * R.mean():.4f} "
      f"| Peak: {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB")
display(render_matrix(R[:5, :5]))


# In[ ]:


# Preprocess the data. The split is §2's, so no scene graph leaks across it, and R_eff is
# a function of the topology alone, so each graph carries its own target from here on.
def seed_resistance(graph, conv):
    """Attaches the token block of the effective resistance under `conv`'s shift.
    R_eff is a WITHIN-component quantity, so the anchor's claim — that the composite
    graph is one component — is asserted here rather than assumed."""
    c = graph.num_token_nodes
    check_composite(graph, c)
    plain = Data(edge_index=graph.edge_index, num_nodes=graph.num_nodes)
    assert nx.is_connected(to_networkx(plain, to_undirected=True)), \
        'the composite graph is disconnected, so its resistances are not comparable'
    graph.R = magnetic_resistance(graph, conv)[:c, :c]

    return graph


train_res = [seed_resistance(g, conv) for g in generate_data(train_keys)]
val_res = [seed_resistance(g, conv) for g in generate_data(val_keys)]
test_res = [seed_resistance(g, conv) for g in generate_data(test_keys)]

# One c×c fp32 target per graph is this stage's whole standing allocation; state it.
graphs_res = train_res + val_res + test_res
print(f"Resistance Targets: {len(train_res)}/{len(val_res)}/{len(test_res)} graphs "
      f"| Cycle: {min(g.num_token_nodes for g in graphs_res)}-"
      f"{max(g.num_token_nodes for g in graphs_res)} "
      f"| Held: {sum(g.R.numel() for g in graphs_res) * 4 / 2 ** 30:.2f} GiB")


# In[ ]:


# Train the Graph Transformer so that C_tok reproduces the composite graph's resistances.
batch_size = 4
val_freq = 5
epochs = _env('E17_RES_EPOCHS', 50)
es_patience = 5
train_resistance = True


def test_loop_resistance(dataloader, model, loss_fn, wandb_prefix=None, epoch=None):
    model['resistance'].to(device).eval()
    model['edges'].to(device).eval()
    size = len(dataloader.dataset)
    order = torch.randperm(size)
    test_loss = {'resistance': 0, 'edges': 0}
    correct, error_norm, trace = 0, 0, 0
    tp = fp = fn = tn = 0

    with torch.no_grad():
        for idx in order.tolist():
            graph = dataloader.dataset[idx]
            c_tok = probe_covariance(model['resistance'], graph)
            preds = {
                'resistance': gram_distances(c_tok),
                'edges': torch.stack([
                    model['edges'](graph, graph.edges_x[0, k], graph.edges_x[1, k])
                    for k in range(graph.edges_x.shape[1])
                ]).squeeze(-1).to(device)
            }

            target = ALPHA * graph.R
            test_loss['resistance'] += loss_fn['resistance'](preds['resistance'], target).item()
            error_norm += ((preds['resistance'] - target).norm() / target.norm()).item()
            trace += c_tok.trace().item()

            test_loss['edges'] += loss_fn['edges'](preds['edges'], graph.edges_y).item()
            true = graph.edges_y.bool()
            pred = preds['edges'] > 0
            tp += (pred & true).sum().item()
            fp += (pred & ~true).sum().item()
            fn += (~pred & true).sum().item()
            tn += (~pred & ~true).sum().item()
            correct += ((preds['edges'].sigmoid() > 0.5).float() == graph.edges_y).float().mean().item()

    test_loss['resistance'] /= size
    error_norm /= size
    trace /= size
    print(f"Test Error #1: \n Rel Err: {error_norm:.3f}, tr C_tok: {trace:.3f} "
          f"| Avg loss: {test_loss['resistance']:>8f} \n")

    test_loss['edges'] /= size
    correct /= size
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    bal_acc = 0.5 * (recall + tn / (tn + fp + 1e-9))
    print(f"Test Error #2: \n Accuracy: {(100*correct):>0.1f}%, F1: {f1:.3f} | P: {precision:.3f} "
          f"| R: {recall:.3f} | Bal Acc: {(100*bal_acc):.1f}% | Avg loss: {test_loss['edges']:>8f} \n")

    # Log eval metrics under the given split prefix (e.g. 'val') when a run is active.
    if wandb_prefix is not None and wandb.run is not None:
        log = {
            f'{wandb_prefix}/loss_resistance': test_loss['resistance'],
            f'{wandb_prefix}/error_norm': error_norm,
            f'{wandb_prefix}/c_tok_trace': trace,
            f'{wandb_prefix}/loss_edges': test_loss['edges'],
            f'{wandb_prefix}/accuracy': correct,
            f'{wandb_prefix}/f1': f1,
            f'{wandb_prefix}/precision': precision,
            f'{wandb_prefix}/recall': recall,
            f'{wandb_prefix}/bal_acc': bal_acc
        }
        if epoch is not None:
            log['epoch'] = epoch
        wandb.log(log)
        # Final test metrics: also surface as run-summary headline numbers.
        if wandb_prefix == 'test':
            wandb.run.summary.update({k: v for k, v in log.items() if k != 'epoch'})
    return test_loss['resistance'], error_norm


def train_loop_resistance(train_dataloader, val_dataloader, test_dataloader, model,
                          loss_fn, optimizer, scheduler, batch_size=20, epochs=50):
    size = len(train_dataloader.dataset)
    model['resistance'].to(device).train()
    model['edges'].to(device).train()
    val_loss: float = 0
    best_val, best_state, bad_runs = float('inf'), {}, 0
    # Log the full run config: GNN (model_hparams) + optimizer/scheduler/loss/batching.
    # The pinned charge and α belong there too, since they are what fix the target.
    run = init_wandb('resistance_regression', {
        'batch_size': batch_size, 'epochs': epochs,
        'val_freq': val_freq, 'es_patience': es_patience,
        'plans_per_graph': plans_per_graph,
        'charge': float(conv.r), 'alpha': ALPHA,
        **optimizer_hparams(optimizer),
        **scheduler_hparams(scheduler),
        **loss_hparams(loss_fn['resistance']),
        **loss_hparams(loss_fn['edges']),
    })
    clip_params = [p for group in optimizer.param_groups for p in group['params']]

    global_step = 0
    for i in range(epochs):
        # Validation loop.
        if i % val_freq == 0:
            print(f"=============\nValidation #{i // val_freq + 1}\n=============")
            val_loss, _ = test_loop_resistance(val_dataloader, model, loss_fn, wandb_prefix='val', epoch=i)
            if scheduler:
                scheduler.step(val_loss)
            if val_loss < best_val - 1e-3:
                best_val, bad_runs = val_loss, 0
                best_state['resistance'] = copy.deepcopy(model['resistance'].state_dict())
                best_state['edges'] = copy.deepcopy(model['edges'].state_dict())
            else:
                bad_runs += 1
                if bad_runs >= es_patience:
                    print(f"Early stop at epoch {i} (best val {best_val:>8f})")
                    break
            model['resistance'].train()
            model['edges'].train()

        print(f"=============\nEpoch #{i + 1}\n=============")
        optimizer.zero_grad()
        pending = 0
        order = torch.randperm(size)
        reshuffle(train_dataloader.dataset)
        for j, idx in enumerate(order.tolist()):
            # Compute prediction and loss. The second moment IS the prediction — no
            # readout head stands between the probe response and the metric it carries.
            graph = train_dataloader.dataset[idx]
            preds = {
                'resistance': gram_distances(probe_covariance(model['resistance'], graph)),
                'edges': torch.stack([
                    model['edges'](graph, graph.edges_x[0, k], graph.edges_x[1, k])
                    for k in range(graph.edges_x.shape[1])
                ]).squeeze(-1).to(device)
            }
            loss = {
                'resistance': loss_fn['resistance'](preds['resistance'], ALPHA * graph.R),
                'edges': loss_fn['edges'](preds['edges'], graph.edges_y)
            }

            # Backpropagation.
            ((loss['resistance'] + loss['edges']) / batch_size).backward()
            pending += 1
            model['edges'].invalidate_cache()

            # Optimization and results.
            if pending == batch_size:
                clip_grad_norm_(clip_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                pending = 0

                global_step += 1
                current = j
                wandb.log({
                    'train/loss_resistance': loss['resistance'],
                    'train/loss_edges': loss['edges'],
                    'train/lr': optimizer.param_groups[0]['lr'],
                    'epoch': i,
                    'global_step': global_step,
                })
                print(f"Loss: {loss['resistance'].item():>7f} | {loss['edges'].item():>7f}  [{current:>5d}/{size:>5d}]")

        if pending:
            clip_grad_norm_(clip_params, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            pending = 0

    # Early stopping hatch.
    if best_state:
        model['resistance'].load_state_dict(best_state['resistance'])
        model['edges'].load_state_dict(best_state['edges'])

    # Test the finished model.
    test_loop_resistance(test_dataloader, model, loss_fn, wandb_prefix='test')
    run.finish()


# MSELoss is the stated sum divided by N², so graphs of different size weigh equally.
loss_fn = {
    'resistance': nn.MSELoss(),
    'edges': nn.BCEWithLogitsLoss()
}
train_dataloader = DataLoader(train_res, batch_size=batch_size)
val_dataloader = DataLoader(val_res, batch_size=batch_size)
test_dataloader = DataLoader(test_res, batch_size=batch_size)
if train_resistance:
    # pe_pool='gt' puts T inside E_q, so C_tok is a function of the blocks as well as of
    # the backbone and the loss reaches both; the classifier keeps §2's objective alive.
    optimizer = torch.optim.AdamW([
        {'params': gnn.parameters(), 'lr': 3e-5},
        {'params': detector.classifier.parameters(), 'lr': 3e-4},
    ], betas=(0.9, 0.95), weight_decay=0.05)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    train_loop_resistance(train_dataloader, val_dataloader, test_dataloader,
                          {'resistance': gnn, 'edges': detector}, loss_fn, optimizer,
                          scheduler, batch_size=batch_size, epochs=epochs)
    torch.save(gnn.state_dict(), f'{save_path_gt}/mag_gt.pt')
    torch.save(detector.classifier.state_dict(), f'{save_path_gt}/detector.pt')
    del optimizer, scheduler
    gc.collect()


# In[ ]:


torch.save(gnn.state_dict(), f'{save_path_gt}/mag_gt.pt')
torch.save(detector.classifier.state_dict(), f'{save_path_gt}/detector.pt')

