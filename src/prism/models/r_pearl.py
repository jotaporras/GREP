import warnings

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import Data

from prism.models import gcn, magnet


class RandomGNNPositionalEncodings(nn.Module):
    """
    Random graph positional encodings (R-PEARL).

    Args:
        pe_hidden_channels (int): Hidden dimension for the GCN
        pe_num_layers (int): Number of layers in the GCN
        d_model (int): Output dimension
        num_samples (int): Probe count M for the Monte-Carlo probe estimate
            (used for both train and eval).
        dropout (float): Dropout rate of the GCN associated.
        k (int): Convolution depth of the GCN.
        use_layer_norm (bool): Retained for config back-compat; the readout is
            always a plain LayerNorm now.
        eps (float): Retained for config back-compat (unused).
        probe_distribution (str): "gaussian" (N(0,I)) or "rademacher" (±1). Both
            satisfy E[q]=0 and unit second moment.
        fixed_seed_mode (bool): Determinism switch. False (default) resamples
            the probes every forward pass (train and eval). True re-seeds the RNG
            with ``fixed_seed_value`` on every forward so the probes — and hence
            Ψ — are identical across runs.
        fixed_seed_value (int): Seed used when ``fixed_seed_mode`` is True.
        directed (bool): Probe backbone. False (default) = ``gcn.GCN`` (TAGConv,
            undirected). True = ``magnet.MagNet``, the magnetic-Laplacian backbone
            that keeps edge direction as a phase; it symmetrizes A and encodes the
            asymmetry in Θ = 2πr·sgn(A − Aᵀ), so it is a no-op on graphs whose
            ``edge_index`` already carries both directions.
        hidden_norm (str): ``directed`` only — normalization between MagNet's hidden
            layers, ``"none"`` (default) or ``"global_rms"``. The pre-fix per-node
            LayerNorm is gone: it is not 1-Lipschitz (so it breaks PEARL Assumption
            4.2) and its affine shift is not conjugation-equivariant (so it breaks
            MagNet's gauge property). See :class:`~prism.models.magnet.MagNet`.
        learn_r (bool): ``directed`` only — learn the MagNet charge r (one per layer,
            sigmoid-constrained to [0, 0.25]) instead of pinning it at 0.25. Adds
            ``pe_gcn.convs.*.r_logit`` to the state dict, so a checkpoint trained with
            it set must be rebuilt with it set.
        phase (str): ``directed`` only — MagNet's phase matrix Θ^(r), ``"binary"``
            (2πr·sgn(A − Aᵀ)) or ``"weight"``. See :class:`~prism.models.magnet.MagChebConv`.
        shift (str): ``directed`` only — MagNet's shift operator, ``"laplacian"``
            (L̂^(r) = −H̄^(r)) or ``"adjacency"``. Same reference.
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
        fixed_seed_mode: bool = False,
        fixed_seed_value: int = 0,
        max_probe_rows: int = 65536,
        max_gather_rows: int = 2_000_000,
        center_second_moment: bool = True,
        node_feature_dim: int = None,
        directed: bool = False,
        learn_r: bool = True,
        hidden_norm: str = "none",
        phase: str = "binary",
        shift: str = "laplacian",
        readout_norm: str = "global_rms",
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
        # Same (in, hidden, layers, skip, dropout, k) contract and the same
        # Data -> [N, hidden] forward, so the backbone swap is local to this line.
        self.directed = directed
        # learn_r / hidden_norm / phase / shift are MagNet-only: TAGConv has no charge,
        # no phase matrix and no complex path.
        self.pe_gcn = (magnet.MagNet if directed else gcn.GCN)(
            in_channels, pe_hidden_channels, pe_num_layers,
            skip_connection=True, dropout=dropout, k=k,
            **({"learn_r": learn_r, "hidden_norm": hidden_norm,
                "phase": phase, "shift": shift} if directed else {})
        )
        self.output_projection = nn.Linear(pe_hidden_channels, d_model)

        self.dropout = nn.Dropout(dropout)
        # ``use_layer_norm`` retained for config back-compat; superseded by
        # ``readout_norm``, which selects the Ψ readout. Only "layer"/"rms" build a module.
        assert readout_norm in ("global_rms", "rms", "layer", "none"), "Invalid readout_norm"
        self.use_layer_norm, self.readout_norm = use_layer_norm, readout_norm
        self.norm = (nn.LayerNorm(d_model) if readout_norm == "layer" else
                     nn.RMSNorm(d_model, elementwise_affine=False)
                     if readout_norm == "rms" else nn.Identity())
        # Learnable tanh(g) gate applied to Ψ and C·s. g = tanh(output_gain) ∈ (-1,1);
        # init output_gain=1 → g ≈ 0.76. Scalar parameter, saved.
        self.output_gain = nn.Parameter(torch.tensor(1.0))
        # Single probe count M for the Monte-Carlo estimate (train and eval).
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
            m: Number of probe samples (``self.M``, train and eval).
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
        out = self._readout(out)
        return out * torch.tanh(self.output_gain).to(out.dtype)

    def _readout(self, x: torch.Tensor) -> torch.Tensor:
        r"""Normalize Ψ for the readout, per ``readout_norm``.

        ``"global_rms"`` (default) divides by ONE detached scalar over all nodes
        and channels, so Ψ's scale is bounded while the RELATIVE magnitudes across
        nodes survive — those carry the structure counted by PEARL Corollary 4.4,
        and they set the per-node angle scale under WIRE's φ_v = ⟨ω, Ψ_v⟩. The
        per-node alternatives pin ‖Ψ_v‖ to a constant and discard that channel:
        ``"layer"`` additionally centers across channels and carries an affine
        shift (not conjugation-equivariant, and not 1-Lipschitz, so it breaks
        PEARL Assumption 4.1 at C_σ = 1); ``"rms"`` is the milder per-node option.
        ``"none"`` leaves the scale unconstrained.
        """
        if self.readout_norm != "global_rms":
            return self.norm(x)
        scale = x.float().pow(2).mean().sqrt().clamp_min(self.eps)
        return x / scale.detach().to(x.dtype)

    def forward(self, data, permutation=None, pool: bool = True):
        """Ψ = E_q[Φ(q; S, H)] (``pool=True``), or the per-probe stack Φ(q^(s); S, H).

        ``pool=False`` returns [M, N, d_model] with the readout (LayerNorm + tanh gate)
        applied PER PROBE and the probe expectation NOT taken, so a caller can put E_q
        outside a later stage: Ψ = E_q[Φ(Φ(q; S, H), G, T)] instead of
        Ψ = Φ(E_q[Φ(q; S, H)], G, T). The aggregator the caller applies must be a
        symmetric function of the probe set (E_q is), or Ψ stops being a Monte-Carlo
        estimator of the expectation. See ``GraphTransformer(pe_pool='gt')``.
        """
        if self.node_feature_dim is not None:
            if not pool:
                raise NotImplementedError(
                    "pool=False is a random-probe path and is incompatible with semantic "
                    "node features (node_feature_dim set): there is no probe axis to pool."
                )
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

        # Probe count M (train and eval); fixed_seed_mode re-seeds for reproducible Ψ.
        m = self.M
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
        phi_chunks = []
        for start in range(0, m, chunk):
            Qc = Q[:, start:start + chunk]
            mc = Qc.shape[1]
            # Gradient checkpoint (train only): recompute the batched GCN on
            # backward to save memory. The dummy gives checkpoint a grad input.
            if torch.is_grad_enabled():
                dummy = Qc.new_ones(1, requires_grad=True)
                pooled = checkpoint(
                    lambda q, ei, d, dev, _mc=mc: self._batched_gcn_forward(
                        q, ei, num_nodes, _mc, edge_weight=edge_weight, device=dev, pool=pool),
                    Qc, edge_index, dummy, device,
                    use_reentrant=False,
                )
            else:
                pooled = self._batched_gcn_forward(
                    Qc, edge_index, num_nodes, mc, edge_weight=edge_weight, device=device,
                    pool=pool)
            if not pool:
                # [mc, N, d_model]; concatenated below. Peak memory is O(M*N*d) — the
                # chunk reduction that keeps it at O(N*d) only exists for pool=True.
                phi_chunks.append(pooled)
                continue
            # weight by mc; divide by m at end to recover global mean.
            contrib = pooled * mc
            pooled_sum = contrib if pooled_sum is None else pooled_sum + contrib
        pooled_pe = pooled_sum / m if pool else torch.cat(phi_chunks, dim=0)

        # LayerNorm is over d_model, so this is the same per-node readout in both
        # branches — applied to Ψ when pooled, to every Φ_s when not.
        pooled_pe = self._readout(pooled_pe)
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

        m = self.M
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

    def covariance_token_block(self, data, c, pe_pool: str = "pe", gt=None):
        """Sampled centered covariance ``C = E_q[Φ'Φ'ᵀ] − ΨΨᵀ`` and first-moment Gram
        ``Ψ̃ = ΨΨᵀ``, returned as the TOKEN blocks ``[c, c]`` (the matrices, not C·s).

        Probes run on the FULL composite graph (so token rows co-vary through the
        crosslinks/scene — non-mention tokens inherit scene context via diffusion).
        Same probe sampling / chunking / fixed-seed semantics as ``forward`` /
        ``second_moment_apply``; gradient flows to the GCN. Returns ``(C_tok, Psi_tok)``.

        ``pe_pool`` says what Φ' IS, exactly as it does for
        :class:`~prism.models.gt.GraphTransformer`:

        - ``"pe"`` (default, every pre-existing caller): ``Φ' = Φ(q; S, H)``, the probe
          response itself — C is read off the probe stack BEFORE any blocks.
        - ``"gt"``: ``Φ' = T(Φ(q; S, H))``, i.e. ``gt``'s transformer blocks applied PER
          PROBE, INSIDE the expectation, so both moments are taken over the block
          outputs. T is nonlinear, so ``E_q[T(Φ)] ≠ T(E_q[Φ])``: this is NOT T applied
          to Ψ, and that difference is the whole point of the option.

        The outer product is ``Φ'Φ'ᵀ`` and not ``Φ'Φ'ᴴ`` because ``MagNet.unwind`` has
        already split the complex representation into ``[Re ‖ Im]`` and returned a REAL
        tensor. Moving T ahead of that unwind would make Φ' complex and the Hermitian
        form the correct one — only ``Φ'Φ'ᴴ`` is invariant under the gauge z ↦ e^{iγ}z.

        Args:
            data: the composite-graph ``Data``, TOKEN rows first.
            c: number of leading token rows the returned blocks cover.
            pe_pool: ``"pe"`` or ``"gt"``, above.
            gt: the ``GraphTransformer`` whose blocks form Φ' = T(Φ). Required by — and
                only by — ``pe_pool="gt"``. Passed in rather than held as an attribute:
                this module is that GT's own submodule, so storing it would close a
                cycle in the module tree and duplicate every block in the state dict.
        """
        if self.node_feature_dim is not None:
            raise NotImplementedError(
                "covariance_token_block is a random-probe readout and is incompatible "
                "with semantic node features (node_feature_dim set)."
            )
        if pe_pool not in ("pe", "gt"):
            raise ValueError(f"pe_pool must be 'pe' or 'gt', got {pe_pool!r}")
        if (pe_pool == "gt") != (gt is not None):
            raise ValueError(
                f"pe_pool={pe_pool!r} was passed gt={type(gt).__name__ if gt is not None else None}: "
                "'gt' needs the GraphTransformer whose blocks form Φ' = T(Φ), and 'pe' must "
                "not be handed one — it would be ignored, and C would silently be the "
                "PRE-block covariance under a config that says otherwise."
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
        m = self.M
        gen = None
        if self.fixed_seed_mode:
            gen = torch.Generator(device=device)
            gen.manual_seed(self.fixed_seed_value)
        Q = self._sample_probes(N, m, device, gen)
        chunk = max(1, min(m, self.max_probe_rows // max(1, N),
                           self.max_gather_rows // max(1, ei.shape[1])))
        def _phi_tok(Qc):
            """Φ' token rows for one probe chunk, ``[mc, c, F]``."""
            P = self._batched_gcn_forward(Qc, ei, N, Qc.shape[1],
                                          edge_weight=ew, device=device, pool=False)  # [mc,N,F]
            if gt is not None:
                # Φ' = T(Φ(q)) — the blocks run per probe, INSIDE E_q, over the whole
                # composite (so the token rows keep reading the scene through the
                # crosslinks); the token slice below is taken after, never before.
                P = gt.apply_blocks(P, data)                        # [mc, N, F]
            return P[:, :c, :]

        def _chunked(fn, *args):
            """Σ over probe chunks of ``fn(Q_chunk, *args)``, recomputed on backward.

            Gradient checkpointing is what makes the reduction fit: without it EVERY
            chunk's MagNet and GT activations stay live at once, which is a multi-GiB
            standing allocation next to a quantized LLM. The dummy gives checkpoint a
            grad-requiring input (``Q`` is a constant).
            """
            acc = None
            for start in range(0, m, chunk):
                Qc = Q[:, start:start + chunk]
                if torch.is_grad_enabled():
                    dummy = Qc.new_ones(1, requires_grad=True)
                    part = checkpoint(fn, Qc, *args, dummy, use_reentrant=False)
                else:
                    part = fn(Qc, *args)
                acc = part if acc is None else acc + part
            return acc

        # Covariance via centered outer products (1/m)Σ(Φ'_s−Ψ)(Φ'_s−Ψ)ᵀ — manifestly PSD,
        # avoids fp32 cancellation from un-centered E[Φ'Φ'ᵀ]−ΨΨᵀ. TWO passes over the SAME
        # `Q` (drawn once above, so this is exact whatever fixed_seed_mode says): the first
        # accumulates Ψ [c, F], the second the Gram [c, c]. The [m, c, F] stack is NEVER
        # materialized — at m=320, c=1800, F=1024 it is 2.4 GB on its own, and the MagNet
        # gathers and GT block activations behind it are several times that again. The
        # price is 2 forwards + 2 backward recomputes; memory is what binds here, not FLOPs.
        def _sum_phi(Qc, _dummy=None):
            return _phi_tok(Qc).sum(dim=0)                          # [c, F]

        def _sum_outer(Qc, p, _dummy=None):
            Ct = _phi_tok(Qc) - p.unsqueeze(0)                      # centered [mc, c, F]
            return torch.einsum("mtf,muf->tu", Ct, Ct)              # [c, c]

        psi = _chunked(_sum_phi) / m                                # Ψ token rows [c, F]
        C_tok = _chunked(_sum_outer, psi) / m                       # PSD covariance [c, c]
        Psi_tok = psi @ psi.transpose(0, 1)                         # ΨΨᵀ token block [c, c]
        return C_tok, Psi_tok
