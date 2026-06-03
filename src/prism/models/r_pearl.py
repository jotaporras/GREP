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
        num_samples (int): Probe count M at train time (m_train).
        dropout (float): Dropout rate of the GCN associated.
        k (int): Convolution depth of the GCN.
        use_layer_norm (bool): Whether to use Lipschitz layer normalization
        eps (float): The Lipschitz constant for layer normalization
        probe_distribution (str): "gaussian" (N(0,I)) or "rademacher" (±1). Both
            satisfy E[q]=0 and unit second moment (R7).
        m_test (int): Probe count M at eval/test. Larger ⇒ lower-variance Monte
            Carlo estimate ⇒ reproducible-in-practice without a seed (R7).
            Defaults to ``num_samples`` when unset.
        fixed_seed_mode (bool): R7 determinism switch. False (default) resamples
            the probes every forward pass (train and eval). True re-seeds the RNG
            with ``fixed_seed_value`` on every forward so the probes — and hence
            Ψ — are identical across runs.
        fixed_seed_value (int): Seed used when ``fixed_seed_mode`` is True.
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
        probe_distribution: str = "gaussian",
        m_test: int = None,
        fixed_seed_mode: bool = False,
        fixed_seed_value: int = 0,
        max_probe_rows: int = 65536,
    ):
        super().__init__()
        if probe_distribution not in ("gaussian", "rademacher"):
            raise ValueError(
                f"probe_distribution must be 'gaussian' or 'rademacher', got {probe_distribution!r}"
            )
        # Multi-hop structure is only reachable when pe_num_layers * k >= 3 (M5).
        if pe_num_layers * k < 3:
            raise ValueError(
                f"pe_num_layers * k must be >= 3 to reach multi-hop structure, "
                f"got {pe_num_layers} * {k} = {pe_num_layers * k}"
            )
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
        # m_train / m_test probe counts (R7); self.M kept as the train alias.
        self.m_train = num_samples
        self.m_test = num_samples if m_test is None else m_test
        self.M = num_samples
        self.probe_distribution = probe_distribution
        self.fixed_seed_mode = fixed_seed_mode
        self.fixed_seed_value = fixed_seed_value
        # Cap on the number of [rows, d_model] entries materialised in one GCN
        # batch. The batched forward stacks `m` probe copies of the N-node graph
        # into a single [m*N, d_model] pass; at eval (m_test large) over the
        # composite graph (N = context_len + scene nodes) that tensor OOMs, so the
        # probes are split into chunks of `floor(max_probe_rows / N)` and the
        # per-chunk means are accumulated (the MC estimate is mean-over-probes, so
        # this is identical to a single pass — only the peak memory differs).
        self.max_probe_rows = max_probe_rows
        self.eps = eps

    def _sample_probes(self, num_nodes: int, m: int, device,
                       generator: torch.Generator = None) -> torch.Tensor:
        """Draw the [num_nodes, m] probe matrix Q from the configured distribution.

        Gaussian: N(0, I). Rademacher: i.i.d. ±1. Both have E[q]=0 and unit second
        moment, so they are valid R-PEARL probes (R7). Sampling is i.i.d. per node,
        keeping the encoder permutation-equivariant in distribution.
        """
        if self.probe_distribution == "gaussian":
            return torch.randn(num_nodes, m, device=device, generator=generator)
        bits = torch.randint(0, 2, (num_nodes, m), device=device,
                             generator=generator, dtype=torch.float)
        return bits * 2 - 1

    def _batched_gcn_forward(self, Q: torch.Tensor, edge_index: torch.Tensor,
                             num_nodes: int, m: int, edge_weight: torch.Tensor = None,
                             device=None, pool: bool = True) -> torch.Tensor:
        """Process all m random samples through the GCN in a single batched call.

        Creates m copies of the graph, each with a different random feature column,
        and processes them as a single PyG Batch for GPU-parallel execution.

        Args:
            Q: Random features [num_nodes, m].
            edge_index: Graph edge indices [2, num_edges].
            num_nodes: Number of nodes in the graph.
            m: Number of probe samples (m_train at train, m_test at eval).
            edge_weight: Optional per-edge weights [num_edges] (E1 affinity);
                the same weights apply to every one of the m graph copies.

        Returns:
            Pooled positional encodings [num_nodes, d_model].
        """

        # Stack Q columns as a single [m*N, 1] feature with repeated edge_index.
        x_stacked = Q.T.reshape(-1, 1)

        # Build batch edge_index by offsetting
        offsets = torch.arange(m, device=device) * num_nodes
        edge_batch = edge_index.unsqueeze(0) + offsets.view(-1, 1, 1)
        edge_batch = edge_batch.permute(1, 0, 2).reshape(2, -1)
        batch_data = Data(x=x_stacked, edge_index=edge_batch, num_nodes=m * num_nodes)
        # edge_batch columns run in [copy, edge] order (i*E + e), so tiling the
        # per-edge weights m times keeps them aligned with their edges.
        if edge_weight is not None:
            batch_data.edge_weight = edge_weight.to(device).repeat(m)

        # Single batched GCN forward pass over all m copies.
        pe_all = self.pe_gcn(batch_data)
        pe_all = self.dropout(pe_all)
        pe_all = self.output_projection(pe_all) * self._dim_scale

        # Reshape [m*N, d_model] -> [m, N, d_model].
        pe_all = pe_all.view(m, num_nodes, -1)
        if not pool:
            # Per-probe responses Φ(q^(s)); the second-moment readout needs these
            # un-pooled (the first-moment readout means over them, below).
            return pe_all
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
        edge_weight = getattr(data, "edge_weight", None)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)

        num_nodes = X.shape[0]
        if permutation is not None:
            edge_index = permutation.apply(edge_index, num_nodes, device=device)

        # Probe count: m_train while training, m_test at eval (R7). Default mode
        # resamples q every forward; fixed_seed_mode re-seeds with a constant so q
        # (and hence Ψ) is identical across runs.
        m = self.m_train if self.training else self.m_test
        generator = None
        if self.fixed_seed_mode:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.fixed_seed_value)

        # Sample all m probes up front (Q is [N, m], 1 feature dim ⇒ tiny). The
        # memory blowup is the batched GCN activation [chunk*N, d_model], not Q,
        # so we chunk only the GCN pass over column slices of the *same* Q. The
        # probe set is therefore identical regardless of chunk size, so chunking
        # is bit-transparent (and R7 reproducibility is unaffected).
        Q = self._sample_probes(num_nodes, m, device, generator)

        # chunk = how many probes fit under max_probe_rows; train (small N, small
        # m_train) keeps chunk == m (single pass, unchanged), eval (large N, large
        # m_test) splits so [chunk*N, d_model] stays bounded.
        chunk = max(1, min(m, self.max_probe_rows // max(1, num_nodes)))
        pooled_sum = None
        for start in range(0, m, chunk):
            Qc = Q[:, start:start + chunk]
            mc = Qc.shape[1]
            # Gradient checkpoint (train only): recompute the batched GCN on
            # backward to save memory. The dummy gives checkpoint a grad input.
            if torch.is_grad_enabled():
                dummy = Qc.new_ones(1, requires_grad=True)
                pooled = checkpoint(
                    lambda q, ei, d, dev, _mc=mc: self._batched_gcn_forward(
                        q, ei, num_nodes, _mc, edge_weight=edge_weight, device=dev),
                    Qc, edge_index, dummy, device,
                    use_reentrant=False,
                )
            else:
                pooled = self._batched_gcn_forward(
                    Qc, edge_index, num_nodes, mc, edge_weight=edge_weight, device=device)
            # _batched_gcn_forward returns the mean over its mc probes; weight by
            # mc and divide by m at the end to recover the global mean exactly.
            contrib = pooled * mc
            pooled_sum = contrib if pooled_sum is None else pooled_sum + contrib
        pooled_pe = pooled_sum / m

        pooled_pe = self.norm(pooled_pe)
        return pooled_pe

    def second_moment_apply(self, data, signal: torch.Tensor) -> torch.Tensor:
        """Apply the probe second moment ``C = E_q[Φ(q)Φ(q)ᵀ]`` to ``signal``.

        Returns ``C·signal ∈ [N, d_model]`` WITHOUT ever forming the ``[N, N]``
        matrix ``C``, via the associativity

            C·s = E_q[ Φ(q) ( Φ(q)ᵀ s ) ].

        ``C`` is the AugR-PEARL second moment — the deterministic, time-invariant
        circulant autocorrelation ``[C]_{nm} = c(n-m)`` over the token cycle
        (page-10 proof: ``C = F* diag(ρ) F``, ``ρ_k = |h(ω^k)|²``) — a *relative*-
        position operator that does NOT collapse on the vertex-transitive cycle the
        way the first moment ``E_q[Φ]`` does. ``Φ(q)`` runs over the entire composite
        graph, so ``C`` spans token+scene; when ``signal`` is non-zero on every node
        (token rows = X, scene rows = first-moment Ψ) the full structure
        (scene–scene, scene–token, token–token) propagates.

        Used by the Graph Transformer as ``H0 = signal + C·signal`` (the second-
        moment readout). Size-robust / transferable (no size-locked parameter, no
        factoring). Probe sampling, chunking, fixed-seed determinism and gradient
        checkpointing mirror ``forward``; only the pooled statistic differs.

        Args:
            data: PyG Data with the (already-permuted, if any) composite-graph edges.
            signal: [N, d_model] signal to apply ``C`` to (the GT's ``seeded``).

        Returns:
            ``C·signal`` in [N, d_model], LipschitzNorm-normalized like ``forward``.
        """
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = data.x.device
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        edge_index = data.edge_index
        edge_weight = getattr(data, "edge_weight", None)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)
        num_nodes = data.x.shape[0]
        # The GCN runs in fp32; apply the whole second moment in fp32.
        s = signal.to(device=device, dtype=torch.float32)

        m = self.m_train if self.training else self.m_test
        generator = None
        if self.fixed_seed_mode:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.fixed_seed_value)
        Q = self._sample_probes(num_nodes, m, device, generator)
        chunk = max(1, min(m, self.max_probe_rows // max(1, num_nodes)))

        def _chunk_apply(Qc, ei, _dummy=None):
            mc = Qc.shape[1]
            # Per-probe responses P_s = Φ(q^(s)) ∈ [N, d] (pool=False keeps them
            # un-meaned); accumulate Σ_s P_s (P_sᵀ s) so the [d, d] inner product
            # is the only dense intermediate — the N×N matrix is never formed.
            P = self._batched_gcn_forward(
                Qc, ei, num_nodes, mc, edge_weight=edge_weight, device=device, pool=False
            )  # [mc, N, d]
            out = None
            for si in range(P.shape[0]):
                Ps = P[si]                                   # [N, d]
                contrib = Ps @ (Ps.transpose(0, 1) @ s)      # [N, d] @ ([d, N] @ [N, d])
                out = contrib if out is None else out + contrib
            return out

        acc = None
        for start in range(0, m, chunk):
            Qc = Q[:, start:start + chunk]
            if torch.is_grad_enabled():
                dummy = Qc.new_ones(1, requires_grad=True)
                contrib = checkpoint(_chunk_apply, Qc, edge_index, dummy, use_reentrant=False)
            else:
                contrib = _chunk_apply(Qc, edge_index)
            acc = contrib if acc is None else acc + contrib
        result = acc / m
        # Scale C·signal to the SIGNAL's magnitude rather than to unit norm. A unit
        # LipschitzNorm here would leave C·signal at ~4% of H0 = signal + C·signal
        # (token rows of `signal` are X at ~‖X‖), drowning out the relative-position
        # operator that — with RoPE off — is the model's only source of token order.
        # Matching the mean row-norm of C·signal to that of `signal` gives the two H0
        # terms comparable strength via a single global scalar (per-row structure
        # preserved, transferable). self.norm (LipschitzNorm) is kept as a utility:
        # loads in old checkpoints and is the toggle point for unit-norm output.
        cur = result.float().norm(dim=-1).mean().clamp(min=1e-6)
        target = s.float().norm(dim=-1).mean()
        return result * (target / cur).to(result.dtype)

