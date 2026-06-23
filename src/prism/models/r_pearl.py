import warnings

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import Data

from prism.models import gcn


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
        use_layer_norm (bool): Retained for config back-compat; the readout is
            always a plain LayerNorm now.
        eps (float): Retained for config back-compat (unused).
        probe_distribution (str): "gaussian" (N(0,I)) or "rademacher" (±1). Both
            satisfy E[q]=0 and unit second moment.
        m_test (int): Probe count M at eval/test. Larger ⇒ lower-variance Monte
            Carlo estimate ⇒ reproducible-in-practice without a seed.
            Defaults to ``num_samples`` when unset.
        fixed_seed_mode (bool): Determinism switch. False (default) resamples
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
        max_gather_rows: int = 2_000_000,
        center_second_moment: bool = True,
        node_feature_dim: int = None,
    ):
        super().__init__()
        if probe_distribution not in ("gaussian", "rademacher"):
            raise ValueError(
                f"probe_distribution must be 'gaussian' or 'rademacher', got {probe_distribution!r}"
            )
        # Warn rather than raise: pe_num_layers*k < 3 gives limited multi-hop reach.
        if pe_num_layers * k < 3:
            warnings.warn(
                f"pe_num_layers * k = {pe_num_layers}*{k} = {pe_num_layers * k} < 3: "
                f"limited multi-hop reach (intentional for minimal/local PE probes)."
            )
        # node_feature_dim=None → random-probe encoder (1-D probes averaged over m samples).
        # When set → deterministic GCN over caller-supplied ``data.x``; no probes.
        self.node_feature_dim = node_feature_dim
        in_channels = 1 if node_feature_dim is None else node_feature_dim
        self.pe_gcn = gcn.GCN(
            in_channels, pe_hidden_channels, pe_num_layers,
            skip_connection=True, dropout=dropout, k=k
        )
        self.output_projection = nn.Linear(pe_hidden_channels, d_model)

        self.dropout = nn.Dropout(dropout)
        # ``use_layer_norm`` retained for config back-compat; always a plain LayerNorm.
        self.use_layer_norm = use_layer_norm
        self.norm = nn.LayerNorm(d_model)
        # Learnable tanh(g) gate applied to Ψ and C·s. g = tanh(output_gain) ∈ (-1,1);
        # init output_gain=1 → g ≈ 0.76. Scalar parameter, saved.
        self.output_gain = nn.Parameter(torch.tensor(1.0))
        # m_train / m_test probe counts; self.M kept as the train alias.
        self.m_train = num_samples
        self.m_test = num_samples if m_test is None else m_test 
        self.M = num_samples
        self.probe_distribution = probe_distribution
        self.fixed_seed_mode = fixed_seed_mode
        self.fixed_seed_value = fixed_seed_value
        # Cap on peak [chunk*N, d_model] during the batched GCN forward; splits
        # probes into chunks of floor(max_probe_rows/N) — same MC estimate.
        self.max_probe_rows = max_probe_rows
        # Companion cap on [chunk*E, channels] in the TAGConv message gather.
        # chunk = min(node-cap, edge-cap); the edge cap only binds for E >> N.
        self.max_gather_rows = max_gather_rows
        # Center E[ΦΦᵀ] into covariance C·s = E[Φ(Φᵀs)] − Ψ(Ψᵀs); removes the
        # rank-1 ΨΨᵀ bias that otherwise dominates E[ΦΦᵀ].
        self.center_second_moment = center_second_moment
        self.eps = eps

    def _sample_probes(self, num_nodes: int, m: int, device,
                       generator: torch.Generator = None) -> torch.Tensor:
        """Draw the [num_nodes, m] probe matrix Q from the configured distribution.

        Gaussian: N(0, I). Rademacher: i.i.d. ±1. Both have E[q]=0 and unit second
        moment, so they are valid R-PEARL probes. Sampling is i.i.d. per node,
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
            edge_weight: Optional per-edge weights [num_edges] (scene affinity weight);
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
        pe_all = self.output_projection(pe_all)

        # Reshape [m*N, d_model] -> [m, N, d_model].
        pe_all = pe_all.view(m, num_nodes, -1)
        if not pool:
            # Per-probe responses Φ(q^(s)); the second-moment readout needs these
            # un-pooled (the first-moment readout means over them, below).
            return pe_all
        pooled_pe = pe_all.mean(dim=0)
        return pooled_pe

    def _deterministic_forward(self, data, permutation=None):
        """Single GCN pass over caller-supplied semantic node features ``data.x``.

        Used when ``node_feature_dim`` is set: ``data.x`` is [N, node_feature_dim] (e.g.
        the mean word-embedding of each node's name). No random probes, no m-averaging —
        one deterministic forward. Mirrors the random path's post-GCN steps (projection,
        LayerNorm, tanh gate). The scene graph is small (N ~ tens), so no chunking /
        gradient-checkpointing is needed.
        """
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = data.x.device
        if permutation is not None:
            raise NotImplementedError(
                "permutation-equivariance eval is not supported with semantic node "
                "features (data.x would also need permuting); pass permutation=None."
            )
        x = data.x.to(device=device, dtype=torch.float32)
        edge_index = data.edge_index.to(device)
        edge_weight = getattr(data, "edge_weight", None)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)
        g = Data(x=x, edge_index=edge_index, num_nodes=x.shape[0])
        if edge_weight is not None:
            g.edge_weight = edge_weight
        out = self.output_projection(self.dropout(self.pe_gcn(g)))   # [N, d_model]
        out = self.norm(out)
        return out * torch.tanh(self.output_gain).to(out.dtype)

    def forward(self, data, permutation=None):
        if self.node_feature_dim is not None:
            return self._deterministic_forward(data, permutation=permutation)
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

        # m_train during training, m_test at eval; fixed_seed_mode re-seeds for reproducible Ψ.
        m = self.m_train if self.training else self.m_test
        generator = None
        if self.fixed_seed_mode:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.fixed_seed_value)

        # Sample all m probes upfront ([N, m], tiny). Chunk only the GCN pass
        # over column slices of Q; probe set is identical regardless of chunk size.
        Q = self._sample_probes(num_nodes, m, device, generator)

        # chunk = min(node-cap, edge-cap); train typically uses chunk=m (single pass).
        num_edges = edge_index.shape[1]
        chunk = max(1, min(m, self.max_probe_rows // max(1, num_nodes),
                           self.max_gather_rows // max(1, num_edges)))
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
            # weight by mc; divide by m at end to recover global mean.
            contrib = pooled * mc
            pooled_sum = contrib if pooled_sum is None else pooled_sum + contrib
        pooled_pe = pooled_sum / m

        pooled_pe = self.norm(pooled_pe)
        return pooled_pe * torch.tanh(self.output_gain).to(pooled_pe.dtype)

    def second_moment_apply(self, data, signal: torch.Tensor,
                            scale_to_signal: bool = True) -> torch.Tensor:
        """Apply the probe second moment ``C = E_q[Φ(q)Φ(q)ᵀ]`` to ``signal``.

        Returns ``C·signal ∈ [N, d_model]`` without forming the [N, N] matrix, via
            C·s = E_q[ Φ(q) ( Φ(q)ᵀ s ) ].
        Probe sampling, chunking, fixed-seed, and gradient checkpointing mirror
        ``forward``; only the pooled statistic differs.

        Args:
            data: PyG Data with (already-permuted, if any) composite-graph edges.
            signal: [N, d_model] signal to apply ``C`` to.
            scale_to_signal: when True (default) applies the learnable ``tanh(g)``
                gate. When False, returns raw ``C·signal`` with no gate (used to
                materialize the ``C`` operator for per-layer q/k injection).

        Returns:
            ``C·signal`` in [N, d_model], gated by ``tanh(output_gain)``.
        """
        if self.node_feature_dim is not None:
            raise NotImplementedError(
                "second_moment_apply is a random-probe readout and is incompatible with "
                "semantic node features (node_feature_dim set); use pe_readout='mean'."
            )
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
        num_edges = edge_index.shape[1]
        chunk = max(1, min(m, self.max_probe_rows // max(1, num_nodes),
                           self.max_gather_rows // max(1, num_edges)))

        def _chunk_apply(Qc, ei, _dummy=None):
            mc = Qc.shape[1]
            # Per-probe Φ(q^(s)) ∈ [N, d]; accumulate Σ_s Φ(Φᵀs) — [d,d] inner
            # product is the only dense intermediate, N×N is never formed.
            P = self._batched_gcn_forward(
                Qc, ei, num_nodes, mc, edge_weight=edge_weight, device=device, pool=False
            )  # [mc, N, d]
            out = None
            for si in range(P.shape[0]):
                Ps = P[si]                                   # [N, d]
                contrib = Ps @ (Ps.transpose(0, 1) @ s)      # [N, d] @ ([d, N] @ [N, d])
                out = contrib if out is None else out + contrib
            return out, P.sum(dim=0)                          # (Σ_s Φ Φᵀ s, Σ_s Φ)

        acc = None
        psi_acc = None
        for start in range(0, m, chunk):
            Qc = Q[:, start:start + chunk]
            if torch.is_grad_enabled():
                dummy = Qc.new_ones(1, requires_grad=True)
                contrib, psi_c = checkpoint(_chunk_apply, Qc, edge_index, dummy, use_reentrant=False)
            else:
                contrib, psi_c = _chunk_apply(Qc, edge_index)
            acc = contrib if acc is None else acc + contrib
            psi_acc = psi_c if psi_acc is None else psi_acc + psi_c
        result = acc / m                                     # E_q[Φ(Φᵀ s)] (uncentered second moment)
        if self.center_second_moment:
            # C·s = E[ΦΦᵀ]s − ΨΨᵀs; subtract rank-1 ΨΨᵀ bias so C carries position.
            # Ψ = E_q[Φ] (un-normed, same raw probes as the second-moment term).
            psi = psi_acc / m                                # E_q[Φ]  [N, d]
            result = result - psi @ (psi.transpose(0, 1) @ s)
        if not scale_to_signal:
            # Raw C·signal — no gate. Used when the caller scales C explicitly
            # (e.g. C_tok for per-layer q/k injection, scaled to ‖X‖ externally).
            return result.to(signal.dtype)
        # Learnable tanh(g) output gate.
        return result * torch.tanh(self.output_gain).to(result.dtype)

    def covariance_token_block(self, data, c):
        """Sampled centered covariance ``C = E_q[ΦΦᵀ] − ΨΨᵀ`` and first-moment Gram
        ``Ψ̃ = ΨΨᵀ``, returned as the TOKEN blocks ``[c, c]`` (the matrices, not C·s).

        Probes run on the FULL composite graph (so token rows co-vary through the
        crosslinks/scene — non-mention tokens inherit scene context via diffusion).
        Same probe sampling / chunking / fixed-seed semantics as ``forward`` /
        ``second_moment_apply``; gradient flows to the GCN. Returns ``(C_tok, Psi_tok)``.
        """
        if self.node_feature_dim is not None:
            raise NotImplementedError(
                "covariance_token_block is a random-probe readout and is incompatible "
                "with semantic node features (node_feature_dim set)."
            )
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = data.x.device
        ei = data.edge_index.to(device)
        ew = getattr(data, "edge_weight", None)
        if ew is not None:
            ew = ew.to(device)
        N = data.x.shape[0]
        m = self.m_train if self.training else self.m_test
        gen = None
        if self.fixed_seed_mode:
            gen = torch.Generator(device=device)
            gen.manual_seed(self.fixed_seed_value)
        Q = self._sample_probes(N, m, device, gen)
        chunk = max(1, min(m, self.max_probe_rows // max(1, N),
                           self.max_gather_rows // max(1, ei.shape[1])))
        # Covariance via centered outer products (1/m)Σ(Φ_s−Ψ)(Φ_s−Ψ)ᵀ — manifestly PSD,
        # avoids fp32 cancellation from un-centered E[ΦΦᵀ]−ΨΨᵀ.
        Pt_chunks = []
        for start in range(0, m, chunk):
            Qc = Q[:, start:start + chunk]
            P = self._batched_gcn_forward(Qc, ei, N, Qc.shape[1],
                                          edge_weight=ew, device=device, pool=False)  # [mc,N,F]
            Pt_chunks.append(P[:, :c, :])                           # token rows [mc, c, F]
        Pt = torch.cat(Pt_chunks, dim=0)                           # [m, c, F]
        psi = Pt.mean(dim=0)                                        # Ψ token rows [c, F]
        Ct = Pt - psi.unsqueeze(0)                                 # centered [m, c, F]
        C_tok = torch.einsum("mtf,muf->tu", Ct, Ct) / m            # PSD covariance [c, c]
        Psi_tok = psi @ psi.transpose(0, 1)                        # ΨΨᵀ token block [c, c]
        return C_tok, Psi_tok

