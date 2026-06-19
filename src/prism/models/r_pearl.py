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
        max_gather_rows: int = 2_000_000,
        center_second_moment: bool = True,
    ):
        super().__init__()
        if probe_distribution not in ("gaussian", "rademacher"):
            raise ValueError(
                f"probe_distribution must be 'gaussian' or 'rademacher', got {probe_distribution!r}"
            )
        # Multi-hop structure is only reachable when pe_num_layers * k >= 3 (M5). Warn
        # rather than raise so minimal/local PE probes (e.g. pe_num_layers=2, k=1) can run.
        if pe_num_layers * k < 3:
            warnings.warn(
                f"pe_num_layers * k = {pe_num_layers}*{k} = {pe_num_layers * k} < 3: "
                f"limited multi-hop reach (intentional for minimal/local PE probes)."
            )
        # Create a GCN that takes 1-dimensional random features
        self.pe_gcn = gcn.GCN(
            1, pe_hidden_channels, pe_num_layers,
            skip_connection=True, dropout=dropout, k=k
        )
        # Add a final projection to ensure output is d_model dimensions
        self.output_projection = nn.Linear(pe_hidden_channels, d_model)

        self.dropout = nn.Dropout(dropout)
        # ``use_layer_norm`` retained for config/back-compat; the readout is always a
        # plain LayerNorm now (the old Lipschitz/BatchNorm variants are gone).
        self.use_layer_norm = use_layer_norm
        self.norm = nn.LayerNorm(d_model)
        # Learnable tanh(g) gate on the R-PEARL output (first moment Ψ in forward() and
        # the second moment C·signal in second_moment_apply()). g = tanh(output_gain)
        # ∈ (-1, 1); init output_gain=1 → g ≈ 0.76. Lets the model scale the positional
        # signal up/down instead of it being fixed by the norm/rescale. Scalar, saved.
        self.output_gain = nn.Parameter(torch.tensor(1.0))
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
        # Companion cap on the TAGConv MESSAGE gather. The batched GCN materialises
        # x_j = [chunk * num_EDGES, channels] in message passing; this is the tensor
        # that OOMs, and it scales with edges, not nodes. ``max_probe_rows`` bounds
        # chunk*num_nodes and is blind to edge count, so a dense composite graph
        # (long token cycle + mention cliques, E >> N) blows past the node-based
        # bound. ``max_gather_rows`` bounds chunk*num_edges as well; chunk is the
        # min of the two caps. For ordinary graphs (E ~ N) the node cap binds and
        # chunk is unchanged (bit-identical output, same speed); only edge-dense
        # graphs — the ones that previously OOMed — get the smaller, fitting chunk.
        # Pure MC-estimate invariant: same probes, only the chunk grouping differs.
        self.max_gather_rows = max_gather_rows
        # Center the second-moment readout into a covariance (C·s = E[Φ(Φᵀs)] − Ψ(Ψᵀs)).
        # Required for the second moment to carry position: the nonlinear GCN gives Φ a
        # nonzero mean whose rank-1 outer product ΨΨᵀ otherwise dominates E[ΦΦᵀ].
        self.center_second_moment = center_second_moment
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
        pe_all = self.output_projection(pe_all)

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

        # chunk = how many probes fit under the node cap AND the edge (gather) cap;
        # train (small N/E, small m_train) keeps chunk == m (single pass, unchanged),
        # eval splits so both [chunk*N, d_model] and the [chunk*E, channels] message
        # gather stay bounded. The edge cap only binds for E >> N (dense composite
        # graphs); otherwise the node cap binds and chunk is unchanged.
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
            # _batched_gcn_forward returns the mean over its mc probes; weight by
            # mc and divide by m at the end to recover the global mean exactly.
            contrib = pooled * mc
            pooled_sum = contrib if pooled_sum is None else pooled_sum + contrib
        pooled_pe = pooled_sum / m

        pooled_pe = self.norm(pooled_pe)
        return pooled_pe * torch.tanh(self.output_gain).to(pooled_pe.dtype)

    def second_moment_apply(self, data, signal: torch.Tensor,
                            scale_to_signal: bool = True) -> torch.Tensor:
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
            scale_to_signal: when True (default) magnitude-match the result to the
                signal's mean row-norm and apply the learnable ``tanh(g)`` output
                gate — the GT readout path. When False, return the **raw centered
                covariance application** ``C·signal`` with no magnitude match and no
                gate, so a caller can extract the ``C`` operator (e.g. the token
                block ``C_tok`` for the per-layer q/k injection) and scale it
                explicitly (e.g. to ``‖X‖``).

        Returns:
            ``C·signal`` in [N, d_model], gated by ``tanh(output_gain)``.
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
        num_edges = edge_index.shape[1]
        chunk = max(1, min(m, self.max_probe_rows // max(1, num_nodes),
                           self.max_gather_rows // max(1, num_edges)))

        def _chunk_apply(Qc, ei, _dummy=None):
            mc = Qc.shape[1]
            # Per-probe responses P_s = Φ(q^(s)) ∈ [N, d] (pool=False keeps them
            # un-meaned); accumulate Σ_s P_s (P_sᵀ s) so the [d, d] inner product
            # is the only dense intermediate — the N×N matrix is never formed. Also
            # accumulate Σ_s P_s (the un-pooled first moment) so the readout can be
            # centered into a covariance (see below).
            P = self._batched_gcn_forward(
                Qc, ei, num_nodes, mc, edge_weight=edge_weight, device=device, pool=False
            )  # [mc, N, d]
            # fp32 outer products (the dominant matmul). [The PSD-stable fp64 path was
            # only needed by the old Monte-Carlo c_bias covariance kernel, which is now
            # ANALYTIC (_analytic_c_from_taps) — the remaining callers (GT H0 `C·seeded`
            # scaled to ‖signal‖; c_per_layer `C_tok` scaled to ‖X‖) are not PSD-asserted
            # and ran in fp32 fine. fp64 here cost ~30x on GPUs with throttled fp64.]
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
            # Center to the COVARIANCE: C·s = E[ΦΦᵀ]s − E[Φ]E[Φ]ᵀ s = E[Φ(Φᵀs)] − Ψ(Ψᵀs).
            # The proof's circulant C = h(S)h(S)* assumes the zero-mean linear response
            # Φ'=h(S)q; the nonlinear GCN gives Φ a nonzero mean whose outer product
            # ΨΨᵀ is a rank-1 DC term that otherwise dominates E[ΦΦᵀ] (collapsing C to
            # pure averaging). Subtracting it (Ψ = E_q[Φ], same raw probes) restores the
            # position-bearing covariance the proof guarantees. Ψ here is the un-normed
            # pooled Φ, matching the un-normed Φ in the second moment.
            psi = psi_acc / m                                # E_q[Φ]  [N, d]
            result = result - psi @ (psi.transpose(0, 1) @ s)
        if not scale_to_signal:
            # Raw centered covariance application C·signal — no magnitude match, no
            # output gate. Used to materialize the C operator (e.g. C_tok) for the
            # per-layer q/k replacement, which is scaled to ‖X‖ by the caller. The
            # gate is deliberately skipped here: a tanh(g)→0 would zero the operator,
            # which under a q←C·q replacement would collapse attention.
            return result.to(signal.dtype)
        # Learnable tanh(g) output gate; the model dials the relative-position
        # operator's strength itself (no ad-hoc magnitude matching to ‖signal‖).
        return result * torch.tanh(self.output_gain).to(result.dtype)

    def covariance_token_block(self, data, c):
        """Sampled centered covariance ``C = E_q[ΦΦᵀ] − ΨΨᵀ`` and first-moment Gram
        ``Ψ̃ = ΨΨᵀ``, returned as the TOKEN blocks ``[c, c]`` (the matrices, not C·s).

        Probes run on the FULL composite graph (so token rows co-vary through the
        crosslinks/scene — non-mention tokens inherit scene context via diffusion).
        Same probe sampling / chunking / fixed-seed semantics as ``forward`` /
        ``second_moment_apply``; gradient flows to the GCN. Returns ``(C_tok, Psi_tok)``.
        """
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
        # Collect the per-probe token-row responses (chunked GCN keeps the edge-gather
        # bounded), then form the covariance from CENTERED outer products. This is the
        # same C = E[ΦΦᵀ]−ΨΨᵀ but computed as (1/m)Σ(Φ_s−Ψ)(Φ_s−Ψ)ᵀ — manifestly PSD with
        # no catastrophic cancellation (the un-centered E[ΦΦᵀ]−ΨΨᵀ gives slightly-negative
        # diagonal variances in fp32, which the unit-diagonal normalization then blows up).
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

