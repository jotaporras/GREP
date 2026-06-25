"""Composite-graph LLM models: Graph-Transformer fusion + cold-start gate / in-attention injection.

Moved out of ``gnn_llm.py``. Holds the three composite-graph model wrappers:

- ``GatedInjection`` — cold-start gate blending token embeddings with the GT token-node output.
- ``CompositeGraphLLM`` — builds the composite graph per sequence, runs the GT (fusion),
  blends token rows via the gate, and feeds a RoPE-disabled LLM.
- ``InjectedCompositeGraphLLM`` — injects the GT code (or the composite covariance Ĉ) into
  q/k/v inside attention at every layer (``pe_qk_injection`` / ``c_per_layer`` / ``c_bias`` modes).

The first/second-moment readout machinery these consume (``covariance_token_block``,
``second_moment_apply``, ``pe_readout`` mean/second_moment) lives in ``r_pearl.py`` / ``gt.py``
and is intentionally left there. The composite-graph *assembly* primitive
(``build_composite_graph`` / ``CompositeGraph`` / GSO) lives in ``composite_graph.py``.
"""

import copy
import importlib
import math

import torch
from torch import nn
from torch_geometric.data import Batch, Data
from transformers import PreTrainedModel

from prism.models.composite_graph import build_composite_graph
from prism.models.llama import disable_rope


class GatedInjection(nn.Module):
    """Cold-start gate blending token embeddings with the Graph Transformer output.

    Produces the LLM ``inputs_embeds`` from the original token embeddings ``X``
    and the Graph Transformer token-node outputs ``Y[V_Tx]``, through a single
    learnable gate that starts at ≈0 (cold-start). Scene-node outputs are
    discarded; only the token rows reach the LLM.

    Modes:
      - ``"interpolate"`` (default): ``inputs_embeds = (1 - gate) * X + gate * Y``.
        The gate is the fraction by which the LLM's own embeddings are replaced
        by the structural output — at ``gate=0`` the LLM sees clean Llama
        embeddings, and structure ramps in as the gate grows.
      - ``"additive"``: ``inputs_embeds = X + gate * Y``.
      - ``"none"``: ``inputs_embeds = Y`` — GT output fed straight; no gate, no X mixing.
        ``gate`` is a fixed 1.0 buffer (non-trainable) so gate diagnostics still read.

    No RoPE or positional transform is applied here (the LLM runs with RoPE disabled).

    Args:
        d_model (int): Embedding width (only used for ``gate_per_dim``).
        gate_init (float): Initial gate value (≈0 cold-start). Ignored when
            ``injection_mode == "none"``.
        gate_per_dim (bool): Per-feature gate vector instead of a scalar. Ignored
            when ``injection_mode == "none"``.
        injection_mode (str): "interpolate" (default), "additive", or "none".
    """

    def __init__(self, d_model: int, gate_init: float = 0.0,
                 gate_per_dim: bool = False, injection_mode: str = "interpolate"):
        super().__init__()
        if injection_mode not in ("interpolate", "additive", "none"):
            raise ValueError(
                "injection_mode must be 'interpolate', 'additive', or 'none', "
                f"got {injection_mode!r}"
            )
        self.injection_mode = injection_mode
        if injection_mode == "none":
            # No learnable gate: fix gate=1.0 buffer (not a Parameter) so gate diagnostics still work.
            self.register_buffer("gate", torch.tensor(1.0))
        else:
            gate_shape = (d_model,) if gate_per_dim else ()
            self.gate = nn.Parameter(torch.full(gate_shape, float(gate_init)))

    def forward(self, X: torch.Tensor, Y_tx: torch.Tensor) -> torch.Tensor:
        """Combine token embeddings X with the gated GT token-node outputs Y[V_Tx].

        Args:
            X: token embeddings [c, d] (V_Tx rows only).
            Y_tx: GT token-node outputs [c, d] (Y[V_Tx]).

        Returns:
            inputs_embeds [c, d] for the LLM.
        """
        if self.injection_mode == "none":
            # Feed the GT output straight through: no gate, no mixing with X.
            return Y_tx
        gate = self.gate
        if self.injection_mode == "interpolate":
            return (1 - gate) * X + gate * Y_tx
        return X + gate * Y_tx


class CompositeGraphLLM(PreTrainedModel):  # ty:ignore[unsupported-base]
    """Composite-graph model: Graph Transformer (fusion) → cold-start gate → RoPE-disabled LLM.

    For each sequence: builds composite graph G (directed cycle over c token positions +
    scene graph + cross-links), runs GT over G with token embeddings X on cycle nodes,
    blends token-row outputs Y[V_Tx] with X through the cold-start gate, and feeds
    ``inputs_embeds`` to the RoPE-disabled LLM. Scene-node outputs are discarded.

    Args:
        llm: causal LLM (RoPE disabled unless disable_llm_rope=False).
        gt_model: GraphTransformer; ``forward(data, token_embeddings, is_token) → Y [N, d]``.
        d_model: embedding / GT width (must equal LLM hidden size).
        gate_init, gate_per_dim, injection_mode: cold-start gate settings.
        disable_llm_rope: disable RoPE on llm (default True).
        cycle_weight, cycle_directed, crosslink_*: composite-graph edge settings.
    """

    def __init__(self, llm: nn.Module, gt_model: nn.Module, d_model: int,
                 gate_init: float = 0.0, gate_per_dim: bool = False,
                 injection_mode: str = "interpolate", disable_llm_rope: bool = True,
                 cycle_weight: float = 1.0, cycle_directed: bool = True,
                 crosslink_weight: float = 1.0, crosslink_mention_to_node: bool = True,
                 crosslink_mention_clique: bool = True):
        config = copy.copy(llm.config)
        config._attn_implementation = "eager"  # ty: ignore[invalid-assignment]
        super().__init__(config)
        self.llm = disable_rope(llm) if disable_llm_rope else llm

        try:
            device = next(self.llm.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        self.gt_model = gt_model.to(device)
        self.injection = GatedInjection(
            d_model, gate_init=gate_init, gate_per_dim=gate_per_dim,
            injection_mode=injection_mode,
        ).to(device)

        self.cycle_weight = cycle_weight
        self.cycle_directed = cycle_directed
        self.crosslink_weight = crosslink_weight
        self.crosslink_mention_to_node = crosslink_mention_to_node
        self.crosslink_mention_clique = crosslink_mention_clique

    def structural_parameters(self) -> list[nn.Parameter]:
        """Graph-side parameters: the GraphTransformer and the gated injection."""
        return list(self.gt_model.parameters()) + list(self.injection.parameters())

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)  # defer to nn.Module first
        except AttributeError:
            return getattr(self.llm, name)

    def _composite_graph(self, scene, injection_map, c, device, permutation=None):
        """Assemble the composite graph G for one sequence on ``device``.

        With a ``permutation`` (transferability sweep), the scene nodes are
        relabeled — matching the legacy R-PEARL semantics of permuting over the
        scene graph's ``num_nodes`` — and the injection-map keys are remapped so
        each token still cross-links to the same scene entity. The token cycle is
        never permuted (sequence positions are fixed; RoPE is off).
        """
        scene_edge_index = scene.edge_index.to(device)
        n_scene = scene.num_nodes
        scene_edge_weight = getattr(scene, "edge_weight", None)
        if scene_edge_weight is None:
            scene_edge_weight = torch.ones(scene_edge_index.shape[1], device=device)
        else:
            scene_edge_weight = scene_edge_weight.to(device)
        if permutation is not None:
            scene_edge_index = permutation.apply(scene_edge_index, n_scene, device=device)
            perm = permutation.perm.to(device)
            injection_map = {int(perm[k]): v for k, v in injection_map.items()}
        return build_composite_graph(
            c, scene_edge_index, scene_edge_weight, n_scene, injection_map,
            cycle_weight=self.cycle_weight, cycle_directed=self.cycle_directed,
            crosslink_weight=self.crosslink_weight,
            crosslink_mention_to_node=self.crosslink_mention_to_node,
            crosslink_mention_clique=self.crosslink_mention_clique,
        )

    def _fuse_embeddings(self, input_ids, graphs, injection_maps, permutation=None,
                         return_token_pe=False, return_c_tok=False, return_cbias=False):
        """Assemble composite graph → GT → gate; returns ``inputs_embeds`` [B, c, d].

        Optional extras (stacked across batch):
        - ``return_token_pe``: un-gated GT token-row outputs Y[V_Tx] [B, c, d].
        - ``return_c_tok``: composite token covariance C_tok [B, c, c], scaled to ‖X‖.
        - ``return_cbias``: (Ĉ, Ψ̃=ΨΨᵀ) [B, c, c] additive-bias kernels (c_bias mode).
        """
        X = self.llm.get_input_embeddings()(input_ids)  # [B, c, d]
        device = X.device
        c = input_ids.shape[1]
        fused = []
        token_pe = [] if return_token_pe else None
        c_tok = [] if return_c_tok else None
        cb_C = [] if return_cbias else None
        cb_P = [] if return_cbias else None
        for b in range(input_ids.shape[0]):
            aug = self._composite_graph(graphs[b], injection_maps[b], c, device, permutation=permutation)
            aug_data = Data(
                x=torch.zeros(aug.num_nodes, 1, device=device),
                edge_index=aug.edge_index,
            )
            aug_data.edge_weight = aug.edge_weight
            # Graph Transformer (fusion): token embeddings on V_Tx, zeros on V_Sc, fused with R-PEARL Ψ.
            Y = self.gt_model(aug_data, token_embeddings=X[b], is_token=aug.is_token)
            Ytok = Y[aug.is_token]
            # Cold-start gate: blend X with the token-node outputs Y[V_Tx].
            fused.append(self.injection(X[b], Ytok))
            if return_token_pe:
                token_pe.append(Ytok)
            if return_c_tok:
                c_tok.append(self._compute_c_tok(aug, aug_data, X[b]))
            if return_cbias:
                C_b, P_b = self._kernel_and_psi(aug, aug_data, c, device)
                cb_C.append(C_b)
                cb_P.append(P_b)
        # GT runs fp32 (sparse sampled_addmm is fp32-only), gate promotes bf16 X to fp32;
        # cast back to LLM dtype or generate() raises "expected Float but found BFloat16".
        inputs_embeds = torch.stack(fused, dim=0).to(X.dtype)
        out = [inputs_embeds]
        if return_token_pe:
            out.append(torch.stack(token_pe, dim=0))
        if return_c_tok:
            out.append(torch.stack(c_tok, dim=0))
        if return_cbias:
            out.append(torch.stack(cb_C, dim=0))
            out.append(torch.stack(cb_P, dim=0))
        return out[0] if len(out) == 1 else tuple(out)

    @staticmethod
    def _unit_diag(M, eps: float = 1e-12):
        """Correlation-normalize a PSD [c,c] kernel to unit diagonal.

        Fails loud on non-finite input (upstream GT/R-PEARL divergence). Clamps diagonal ≥ 0,
        floors variance relative to mean (avoids 1/√0 for near-zero rows), clamps entries
        to [-1, 1] so the additive bias is bounded.
        """
        if not torch.isfinite(M).all():
            raise FloatingPointError(
                "[c_bias] non-finite covariance kernel before normalization — UPSTREAM "
                "GT/R-PEARL divergence (NaN/Inf), not a _unit_diag issue. Lower "
                "structural_lr_mult / check the GT; failing loud instead of collapsing to 0.")
        d = M.diagonal().clamp(min=0)
        floor = d.mean().clamp(min=eps) * 1e-3
        d = d.clamp(min=floor).sqrt()
        out = M / d[:, None] / d[None, :]
        return out.clamp(-1.0, 1.0)

    def _kernel_and_psi(self, aug, aug_data, c, device):
        """Per-sequence [c,c] additive-bias kernels ``(Ĉ, Ψ̃)`` for c_bias mode, unit-diagonal.

        - ``"sampled"``: Ĉ = E_q[ΦΦᵀ]−ΨΨᵀ and Ψ̃ = ΨΨᵀ from one probe pass on the full graph.
        - ``"analytic"``: Ĉ = H(S)H(S)* via cascade taps + matrix powers on S=[[S_c,B],[Bᵀ,S_sc]].
        Both carry gradient to the R-PEARL filter.
        """
        pe = self.gt_model.pe_model
        if self.c_kernel == "analytic":
            C = self._analytic_c_from_gso(self._analytic_taps().to(device), aug.gso, c)
            psi = pe(aug_data)[:c]                             # Ψ token rows [c, d]
            Psi = psi @ psi.t()
        else:                                                 # "sampled" (default)
            C, Psi = pe.covariance_token_block(aug_data, c)
        return self._unit_diag(C), self._unit_diag(Psi)

    def _compute_c_tok(self, aug, aug_data, X_b):
        """Composite token-block covariance C_tok [c, c] for c_per_layer.

        C_tok = C[V_Tx, V_Tx] from R-PEARL second-moment applied to token-row identity.
        Includes token→scene→token paths. Scaled so C_tok·X matches X's mean row-norm.
        Deterministic (no learnable params).
        """
        device = X_b.device
        N = aug.num_nodes
        c = int(aug.is_token.sum())
        sig = torch.zeros(N, c, device=device, dtype=torch.float32)
        sig[aug.is_token] = torch.eye(c, device=device, dtype=torch.float32)
        C_full = self.gt_model.pe_model.second_moment_apply(
            aug_data, sig, scale_to_signal=False)        # [N, c] = C[:, V_Tx]
        C_tok = C_full[aug.is_token]                      # [c, c] = C[V_Tx, V_Tx]
        Xf = X_b.float()
        cur = (C_tok @ Xf).norm(dim=-1).mean().clamp(min=1e-12)
        target = Xf.norm(dim=-1).mean()
        return C_tok * (target / cur)

    @staticmethod
    def _analytic_c_row_from_taps(H, c, eps: float = 1e-12):
        """Analytic relative-position row c_row[δ] = c(δ)/c(0) from R-PEARL taps H [K+1, F].

        For the directed circulant cycle S_c: ĥ(ω_k)=Σ_j H_j ω_k^j, ρ_k=‖ĥ(ω_k)‖²,
        c(δ)=IDFT(ρ), normalized by c(0). Returns [c] row; gradient flows to H.
        """
        K1 = H.shape[0]
        dev = H.device
        kk = torch.arange(c, device=dev, dtype=torch.float32)
        jj = torch.arange(K1, device=dev, dtype=torch.float32)
        ang = (2.0 * math.pi / c) * torch.outer(kk, jj)        # [c, K1]
        Hf = H.float()
        hr = torch.cos(ang) @ Hf                                # Re ĥ(ω_k)  [c, F]
        hi = torch.sin(ang) @ Hf                                # Im ĥ(ω_k)  [c, F]
        rho = (hr * hr + hi * hi).sum(-1)                       # ρ_k = ‖ĥ(ω_k)‖²  [c]
        c_vec = torch.fft.ifft(rho.to(torch.complex64)).real    # c(δ) = IDFT(ρ)  [c]
        return c_vec / c_vec[0].clamp(min=eps)                  # c_row [c], c_row[0]=1

    @classmethod
    def _analytic_c_from_taps(cls, H, c, eps: float = 1e-12):
        """Full analytic Ĉ [c,c] (circulant) + c_row [c] from taps. PSD (ρ≥0), gradient to H."""
        c_row = cls._analytic_c_row_from_taps(H, c, eps)
        kk = torch.arange(c, device=H.device)
        idx = (kk[:, None] - kk[None, :]) % c                  # (t−u) mod c
        return c_row[idx], c_row                               # Ĉ [c,c] , c_row [c]

    @staticmethod
    def _analytic_c_from_gso(H, gso, c, eps: float = 1e-9):
        """Graph-covariance kernel Ĉ = H(S)H(S)* on the full composite GSO S = [[S_c,B],[Bᵀ,S_sc]].

        Ĉ_full = Σ_{k,l} G_{kl}·Sᵏ(Sˡ)ᵀ, where G_{kl}=⟨H_k,H_l⟩; returns token block [c,c].
        Symmetric and PSD by construction; non-circulant S couples far-apart tokens that share
        graph-adjacent scene nodes. Correlation-normalized to unit diagonal. Gradient flows to H.
        """
        K1 = H.shape[0]
        S = (gso.to_dense() if gso.is_sparse else gso).to(torch.float32)
        N = S.shape[0]
        G = H.float() @ H.float().t()                          # [K1,K1] ⟨H_k,H_l⟩ (grad)
        Spow = [torch.eye(N, device=S.device, dtype=S.dtype)]  # S^0..S^K
        for _ in range(1, K1):
            Spow.append(Spow[-1] @ S)
        C_full = sum(G[k, l] * (Spow[k] @ Spow[l].t())
                     for k in range(K1) for l in range(K1))     # H(S)H(S)*  [N,N]
        Ct = C_full[:c, :c]                                    # token block
        d = Ct.diagonal().clamp(min=eps).sqrt()
        return Ct / d[:, None] / d[None, :]                    # unit-diagonal Ĉ [c,c]

    def _analytic_taps(self):
        """Effective cascade taps H̄ [L·K+1, F] from all L GCN layers (grad-carrying).

        Stacked layers compose into H(S) = Σ_k S^k H̄_k via discrete convolution
        H̄ = h^(1) * … * h^(L). Channel-diagonal: each layer's K+1 lins → [K+1, F].
        """
        convs = self.gt_model.pe_model.pe_gcn.convs
        per_layer = [torch.stack([lin.weight.reshape(lin.weight.shape[0], -1).mean(-1)
                                  for lin in conv.lins], dim=0)        # [K+1, F]
                     for conv in convs]
        bar = per_layer[0]                                            # cascade convolution
        for h in per_layer[1:]:
            out = bar.new_zeros(bar.shape[0] + h.shape[0] - 1, bar.shape[1])
            for i in range(bar.shape[0]):
                out[i:i + h.shape[0]] = out[i:i + h.shape[0]] + bar[i:i + 1] * h
            bar = out
        return bar                                                   # [L*K+1, F]

    def _analytic_c_tok(self, c, device):
        """Convenience: full analytic ``Ĉ`` on ``device`` (used by the debug callback)."""
        return self._analytic_c_from_taps(self._analytic_taps().to(device), c)

    # ----- decode-time composite-graph extension ("the brain grows") -------------
    # Each generated token adds a cycle node; exact node-name match adds a crosslink/clique.
    # Incremental update: cache A_k=(Ŝᵏq) and Φ, compute only the new row
    # Ĉ[new,·]=⟨Φ_new,Φ_u⟩/m − Ψ_new·Ψ_u (frozen-old-degree approximation, cos≈0.97).
    def decode_setup(self, aug, node_token_seqs, c, max_seq, m_dec=16, device=None):
        """Arm the decode-extension state from the prompt composite graph ``aug``."""
        device = device or next(self.parameters()).device
        Hbar = self._analytic_taps().to(device=device, dtype=torch.float32)   # [K1,F]
        K1, F = Hbar.shape
        gso = aug.gso.coalesce()
        idx, val = gso.indices().to(device), gso.values().to(torch.float32).to(device)
        N0 = aug.num_nodes
        Nmax_ = max_seq + aug.num_scene_nodes
        # prompt-node degrees (A+I rowsum); Nmax-sized to include generated nodes.
        deg = torch.zeros(Nmax_, device=device)
        deg[:N0] = torch.zeros(N0, device=device).index_add_(0, idx[0], torch.ones_like(val))
        # neighbor lists for the per-hop aggregate recursion
        nbrs = [[] for _ in range(N0)]
        for e in range(idx.shape[1]):
            nbrs[int(idx[0, e])].append((int(idx[1, e]), float(val[e])))
        Nmax = max_seq + aug.num_scene_nodes
        torch.manual_seed(0)                                  # deterministic decode probes
        q = torch.zeros(Nmax, m_dec, device=device)
        q[:N0] = torch.randn(N0, m_dec, device=device)
        # per-hop aggregates A_k = Ŝᵏ q on the prompt graph (dense-apply via the sparse GSO)
        A = torch.zeros(K1, Nmax, m_dec, device=device)
        A[0, :N0] = q[:N0]
        S = torch.sparse_coo_tensor(idx, val, (N0, N0)).coalesce()
        for k in range(1, K1):
            A[k, :N0] = torch.sparse.mm(S, A[k - 1, :N0])
        Phi = torch.zeros(Nmax, m_dec, F, device=device)
        Phi[:N0] = torch.einsum("kf,knm->nmf", Hbar, A[:, :N0])
        Psi = torch.zeros(Nmax, F, device=device)
        Psi[:N0] = Phi[:N0].mean(1)
        # token-node sequence order: cycle positions 0..c-1 (scene rows excluded from keys)
        self._decode_state = dict(
            Hbar=Hbar, A=A, Phi=Phi, Psi=Psi, deg=deg, nbrs=nbrs, m=m_dec, F=F, K1=K1,
            c=c, N0=N0, n_scene=aug.num_scene_nodes, node_token_seqs=node_token_seqs,
            token_nodes=list(range(c)),          # sequence-pos → node index (extends on decode)
            mentions={}, gen_ids=[], next_node=N0, prev_node=c - 1)
        self._pe_decode_row = None

    def decode_extend(self, token_id):
        """Append one generated token: extend the graph, compute its Φ row incrementally,
        and stash the decode bias row ``Ĉ[new, :key_len]`` on ``self._pe_decode_row``."""
        st = self._decode_state
        dev = st["Hbar"].device
        new = st["next_node"]
        st["gen_ids"].append(int(token_id))
        # --- exact node-name token-sequence match (suffix of the generated stream) ---
        v = None
        for node_idx, seq in st["node_token_seqs"]:
            L = len(seq)
            if L and len(st["gen_ids"]) >= L and st["gen_ids"][-L:] == list(seq):
                v = node_idx; break
        # --- new node's neighbors: cycle predecessor + (crosslink scene node + clique) ---
        nb = [(st["prev_node"], 1.0)]
        if v is not None:
            scene_node = st["c"] + v
            nb.append((scene_node, 1.0))
            for p in st["mentions"].get(v, []):           # clique to prior mentions of v
                nb.append((p, 1.0))
            st["mentions"].setdefault(v, []).append(new)
        deg_new = float(len(nb)) + 1.0                     # incident + self-loop
        A, Phi, Psi, deg, Hbar = st["A"], st["Phi"], st["Psi"], st["deg"], st["Hbar"]
        m, K1 = st["m"], st["K1"]
        A[0, new] = torch.randn(m, device=dev)             # fresh probe for the new node
        inv_new = deg_new ** -0.5
        for k in range(1, K1):
            acc = inv_new * inv_new * A[k - 1, new].clone()        # self-loop term
            for (j, w) in nb:
                acc = acc + (w * inv_new * float(deg[j].clamp(min=1).item()) ** -0.5) * A[k - 1, j]
            A[k, new] = acc
        Phi[new] = torch.einsum("kf,km->mf", Hbar, A[:, new])
        Psi[new] = Phi[new].mean(0)
        # degrees: register the new node (frozen-old approximation leaves neighbors as-is)
        if new < deg.shape[0]:
            deg[new] = deg_new
        st["token_nodes"].append(new)
        st["prev_node"] = new
        st["next_node"] = new + 1
        # --- decode bias row over the current key sequence (token nodes only) ---
        cols = torch.tensor(st["token_nodes"], device=dev)
        row = (torch.einsum("mf,umf->u", Phi[new], Phi[cols]) / m
               - Psi[new] @ Psi[cols].t())                  # Ĉ[new, key positions]
        self._pe_decode_row = row

    def decode_disarm(self):
        self._decode_state = None
        self._pe_decode_row = None

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        graphs: Batch | None = None,
        injection_maps: list[dict[int, list[tuple[int, int]]]] | None = None,
        **kwargs,
    ):
        inputs_embeds = self._fuse_embeddings(input_ids, graphs, injection_maps)
        kwargs.pop("inputs_embeds", None)
        kwargs.pop("input_ids", None)
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )


class InjectedCompositeGraphLLM(CompositeGraphLLM):
    """Composite-graph LLM that injects the GT code into q/k/v inside attention at every layer.

    Per attention layer ``l``::

        q_l = W^q_l · h_l + W_q · S
        k_l = W^k_l · h_l + W_k · S
        v_l = W^v_l · h_l + W_v · S    (if inject_v)

    - RoPE is OFF (identity rotary): S = Y_tok is the sole position signal.
    - Dedicated shared projections W_q/W_k/W_v (not the LLM's own) inject S at every layer.
    - ``inputs_embeds`` is the gated GT blend gate(X, Y_tok) (Layer-0 injection).
    - ``self._pe_signal`` [B, seq, hidden]; skipped on cached decode steps (seq mismatch).
    - Attention forward patched per-instance; survives PEFT.
    """

    def __init__(self, *args, inject_v: bool = True,
                 disable_llm_rope: bool = True, c_per_layer: bool = False,
                 c_bias: bool = False, c_kernel: str = "sampled",
                 use_scene_bias: bool = True, **kwargs):
        # RoPE OFF by default (disable_llm_rope=True): the injected code replaces RoPE.
        kwargs["disable_llm_rope"] = disable_llm_rope
        super().__init__(*args, **kwargs)

        # c_bias: additive logit biases λ_C·Ĉ + λ_ψ·Ψ̃=ΨΨᵀ + residual λ_V·Ĉ value mix;
        # no q/k transform. c_kernel selects Ĉ: "sampled" (probe) or "analytic" (taps).
        self.c_bias = c_bias
        self.c_kernel = c_kernel

        # c_per_layer: REPLACE post-RoPE q/k with C_tok·q, C_tok·k at every layer
        # (q ← C_tok·q; deterministic, scaled to ‖X‖). No dedicated projections in this mode.
        self.c_per_layer = c_per_layer

        cfg = self.llm.config
        hidden = cfg.hidden_size
        n_heads = cfg.num_attention_heads
        n_kv = getattr(cfg, "num_key_value_heads", n_heads)
        head_dim = getattr(cfg, "head_dim", None) or (hidden // n_heads)
        try:
            device = next(self.llm.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        if not c_per_layer and not c_bias:
            # Dedicated shared projections (one set reused at every layer); no bias.
            self.pe_q_proj = nn.Linear(hidden, n_heads * head_dim, bias=False, device=device)
            self.pe_k_proj = nn.Linear(hidden, n_kv * head_dim, bias=False, device=device)
            if inject_v:
                self.pe_v_proj = nn.Linear(hidden, n_kv * head_dim, bias=False, device=device)

        if c_bias:
            # Scalar learnable gains: λ_C init 1.0 (bias active from step 0), λ_ψ and λ_V init 0.1.
            self.lam_c = nn.Parameter(torch.tensor(1.0, device=device))
            self.lam_psi = nn.Parameter(torch.tensor(0.1, device=device))
            self.lam_v = nn.Parameter(torch.tensor(0.1, device=device))
            # λ_C warmup ramp ∈ [0,1], non-persistent (defaults to 1.0 on load).
            self.register_buffer("_lam_c_warmup", torch.tensor(1.0, device=device),
                                 persistent=False)
            # Use SDPA: c_bias adds a float mask to logits; flash-attn can't; eager
            # materializes the full [B,H,c,c] score matrix which is prohibitive.
            self.llm.config._attn_implementation = "sdpa"  # ty: ignore[invalid-assignment]

        # Multi-GPU: each sharded layer recomputes Ĉ from the small [K+1,F] taps locally.
        # Single-GPU: kernel precomputed once in _arm_signals.
        self._llm_sharded = False
        dm = getattr(self.llm, "hf_device_map", None)
        if dm:
            real = {str(d) for d in dm.values() if d not in ("cpu", "disk", -1)}
            self._llm_sharded = len(real) > 1
        if c_bias and self._llm_sharded:
            # Taps + scalar gains accumulate grads across device streams; silence the
            # expected stream-mismatch warning (tiny tensors, negligible sync cost).
            _setw = getattr(getattr(torch.autograd, "graph", None),
                            "set_warn_on_accumulate_grad_stream_mismatch", None)
            if _setw is not None:
                _setw(False)

        # Per-forward signals set by forward()/prepare_generation():
        #   _pe_signal [B, seq, hidden]: additive GT code S=Y_tok.
        #   _pe_C [B, seq, seq]: composite covariance Ĉ (c_per_layer / c_bias).
        #   _pe_Psi [B, seq, seq]: first-moment Gram Ψ̃=ΨΨᵀ (c_bias).
        #   _pe_c_row [B, seq]: analytic relative row c(·) for decode (c_bias).
        self._pe_signal = None
        self._pe_C = None
        self._pe_Psi = None
        self._pe_c_row = None
        self._pe_taps = None      # c_bias multi-GPU: grad-carrying taps, Ĉ recomputed per layer
        self._pe_cyc = None       # cycle length c for the analytic kernel
        self._decode_state = None # decode-time graph-extension state (set by decode_setup)
        self._pe_decode_row = None  # live decode bias row Ĉ[new, :key_len] (set per token)
        self._pe_inject_value = inject_v
        self._install_pe_injection()

    def structural_parameters(self) -> list[nn.Parameter]:
        """Base graph-side params plus the optional dedicated q/k/v code projections
        (pe_qk_injection mode) and the scalar c_bias gains λ_C/λ_ψ/λ_V."""
        params = super().structural_parameters()
        for name in ("pe_q_proj", "pe_k_proj", "pe_v_proj"):
            mod = getattr(self, name, None)
            if mod is not None:
                params += list(mod.parameters())
        for name in ("lam_c", "lam_psi", "lam_v"):
            p = getattr(self, name, None)
            if p is not None:
                params.append(p)
        return params

    # ----- attention patch (ported from eval_unification, model-agnostic) -------
    def _decoder_layers(self):
        """Return the LLM's decoder layer list (Llama/Qwen2: ``<CausalLM>.model.layers``)."""
        base = getattr(self.llm, "model", None)
        if base is not None and hasattr(base, "layers"):
            return base.layers
        return self.llm.get_decoder().layers

    def _install_pe_injection(self):
        """Patch every self-attention forward so S is added post-RoPE in q/k/(v)."""
        for layer in self._decoder_layers():
            attn = layer.self_attn
            attn.forward = self._make_injected_attention_forward(attn)

    def _make_injected_attention_forward(self, attn):
        # Resolve model-family helpers (attention fns, rotary) from the attention class's module.
        mod = importlib.import_module(type(attn).__module__)
        apply_rotary_pos_emb = mod.apply_rotary_pos_emb
        attn_fns = mod.ALL_ATTENTION_FUNCTIONS
        eager = mod.eager_attention_forward
        model = self  # captured for the per-forward signal

        def forward(hidden_states, position_embeddings=None, attention_mask=None,
                    past_key_values=None, **kwargs):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, attn.head_dim)

            query_states = attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            # Add GT code S=Y_tok through dedicated projections (RoPE is off; identity rotary above).
            psi = model._pe_signal
            if (psi is not None and psi.shape[0] == hidden_states.shape[0]
                    and psi.shape[1] == hidden_states.shape[1]):
                def _proj(linear):
                    # Project on the weight's device, then move to q/k/v device (device_map=auto).
                    w = linear.weight
                    out = linear(psi.to(device=w.device, dtype=w.dtype))
                    out = out.to(device=query_states.device, dtype=query_states.dtype)
                    return out.view(hidden_shape).transpose(1, 2)
                query_states = query_states + _proj(model.pe_q_proj)
                key_states = key_states + _proj(model.pe_k_proj)
                if model._pe_inject_value:
                    value_states = value_states + _proj(model.pe_v_proj)
            # c_bias: resolve per-layer Ĉ [c,c] / c_row [c]. Single-GPU: reuse _pe_C.
            # Multi-GPU: recompute from taps on this layer's device (small [K+1,F] only).
            cb_C = cb_crow = None
            if model.c_bias:
                qdev = query_states.device
                seq = hidden_states.shape[1]
                if model._pe_taps is not None:                  # multi-GPU: recompute here
                    taps = model._pe_taps.to(qdev)
                    if seq == model._pe_cyc:
                        cb_C, cb_crow = model._analytic_c_from_taps(taps, model._pe_cyc)
                    elif seq == 1:
                        cb_crow = model._analytic_c_row_from_taps(taps, model._pe_cyc)
                else:                                           # single-GPU: precomputed
                    if model._pe_C is not None and model._pe_C.shape[1] == seq:
                        cb_C = model._pe_C[0].to(device=qdev, dtype=torch.float32)
                    if model._pe_c_row is not None:
                        cb_crow = model._pe_c_row[0].to(device=qdev)
            # c_bias: residual Ĉ-value mix on prompt (cached); renorm to ‖v‖.
            if model.c_bias and cb_C is not None:
                mixed = torch.einsum("nm,bhmd->bhnd", cb_C.to(value_states.device),
                                     value_states.float())
                mn = mixed.norm(dim=-1).mean().clamp(min=1e-12)
                vn = value_states.float().norm(dim=-1).mean()
                mixed = mixed * (vn / mn)
                lam_v = model.lam_v.to(device=value_states.device, dtype=value_states.dtype)
                value_states = value_states + lam_v * mixed.to(value_states.dtype)
            # c_per_layer: REPLACE q/k with C_tok·q, C_tok·k at every layer.
            # Skipped on cached decode (seq mismatch); value is content.
            C = model._pe_C
            if (not model.c_bias and C is not None and C.shape[0] == hidden_states.shape[0]
                    and C.shape[1] == hidden_states.shape[1]):
                # Move C_tok to this layer's device (device_map=auto may shard).
                Cf = C.to(device=query_states.device, dtype=torch.float32)

                def _mix(t):  # t: [B, H, seq, head_dim]
                    # fp32 einsum (bf16 crushes fine c(n-m) decay); renorm to t's mean
                    # row-norm for scale stability across depths.
                    out = torch.einsum("bnm,bhmd->bhnd", Cf.to(t.device), t.float())
                    cur = out.norm(dim=-1).mean().clamp(min=1e-12)
                    tgt = t.float().norm(dim=-1).mean()
                    return (out * (tgt / cur)).to(t.dtype)

                query_states = _mix(query_states)
                key_states = _mix(key_states)
            # ------------------------------------------------------------------

            if past_key_values is not None:
                key_states, value_states = past_key_values.update(
                    key_states, value_states, attn.layer_idx)

            # c_bias: fold λ_C·Ĉ + λ_ψ·Ψ̃ into attention_mask (added to logits pre-softmax).
            # Built post-cache so decode row spans all cached keys.
            if model.c_bias:
                key_len = key_states.shape[2]
                dev = query_states.device
                # Move gains to this layer's device (not just dtype, device_map=auto).
                lam_c = (model.lam_c * model._lam_c_warmup).to(device=dev, dtype=torch.float32)
                lam_psi = model.lam_psi.to(device=dev, dtype=torch.float32)
                bias = None
                if cb_C is not None:
                    # PROMPT: λ_C·Ĉ + λ_ψ·Ψ̃ [c,c] → [B,1,seq,seq]
                    b = lam_c * cb_C.to(dev)[None]                          # [1,c,c]
                    Psi = model._pe_Psi
                    if Psi is not None:
                        b = b + lam_psi * Psi[0].to(device=dev, dtype=torch.float32)[None]
                    bias = b.unsqueeze(1)
                elif query_states.shape[2] == 1 and model._pe_decode_row is not None:
                    # DECODE (graph-extended): live row Ĉ[new, :key_len], padded/clipped to key_len.
                    row = model._pe_decode_row.to(device=dev, dtype=torch.float32)
                    if row.shape[0] < key_len:
                        row = torch.cat([row, row.new_zeros(key_len - row.shape[0])])
                    bias = (lam_c * row[:key_len])[None, None, None, :]
                elif cb_crow is not None and query_states.shape[2] == 1:
                    # DECODE fallback (no decode-extension armed): analytic c(·), offset mod c.
                    crow = cb_crow.to(dev)                                  # [c]
                    cyc = crow.shape[0]
                    off = (key_len - 1 - torch.arange(key_len, device=dev)) % cyc
                    bias = (lam_c * crow[off])[None, None, None, :]
                if bias is not None:
                    bias = bias.to(dtype=query_states.dtype)
                    if attention_mask is None:
                        # SDPA with no mask uses is_causal internally; supplying one disables it,
                        # so fold the causal triangle into the bias (prompt only).
                        q_len = query_states.shape[2]
                        if q_len > 1:
                            causal = torch.triu(
                                torch.full((q_len, key_len), float("-inf"), device=dev,
                                           dtype=bias.dtype), diagonal=key_len - q_len + 1)
                            bias = bias + causal[None, None]
                        attention_mask = bias
                    else:
                        am = attention_mask[..., :key_len].to(device=dev, dtype=bias.dtype)
                        attention_mask = am + bias

            attn_impl = attn.config._attn_implementation
            attention_interface = (
                eager if attn_impl == "eager" else attn_fns[attn_impl]
            )
            attn_output, attn_weights = attention_interface(
                attn, query_states, key_states, value_states, attention_mask,
                dropout=0.0 if not attn.training else attn.attention_dropout,
                scaling=attn.scaling, **kwargs,
            )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = attn.o_proj(attn_output)
            return attn_output, attn_weights

        return forward

    # ----- signal / forward / generation ---------------------------------------
    def _arm_signals(self, input_ids, graphs, injection_maps, permutation=None):
        """Build ``inputs_embeds`` (gated GT blend) and arm per-forward signal(s) per mode:
        - additive: ``self._pe_signal = Y_tok`` injected into q/k/v at every layer;
        - c_per_layer: ``self._pe_C = C_tok`` replaces q/k (``q ← C_tok·q``);
        - c_bias: ``self._pe_C``, ``self._pe_Psi``, ``self._pe_c_row`` set for additive bias.
        Unused signals are cleared to avoid stale leakage.
        """
        if self.c_bias:
            # c_bias: additive λ_C·Ĉ + λ_ψ·Ψ̃ bias + λ_V value mix; per-sequence [B,c,c] kernels.
            B, c = input_ids.shape[0], input_ids.shape[1]
            self._pe_signal = None
            self._pe_cyc = c
            inputs_embeds, c_hat, psi_t = self._fuse_embeddings(
                input_ids, graphs, injection_maps, permutation=permutation,
                return_cbias=True)
            self._pe_C = c_hat                                   # [B, c, c]
            self._pe_Psi = psi_t                                 # [B, c, c]
            self._pe_taps = None
            # Decode fallback: analytic c(·) row; superseded if decode_setup is armed.
            self._pe_c_row = self._analytic_c_row_from_taps(
                self._analytic_taps().to(input_ids.device), c).unsqueeze(0).expand(B, -1)
        elif self.c_per_layer:
            inputs_embeds, c_tok = self._fuse_embeddings(
                input_ids, graphs, injection_maps, permutation=permutation,
                return_c_tok=True)
            self._pe_C = c_tok
            self._pe_Psi = None
            self._pe_c_row = None
            self._pe_signal = None
        else:
            inputs_embeds, token_pe = self._fuse_embeddings(
                input_ids, graphs, injection_maps, permutation=permutation,
                return_token_pe=True)
            self._pe_signal = token_pe
            self._pe_C = None
            self._pe_Psi = None
            self._pe_c_row = None
        return inputs_embeds

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                graphs=None, injection_maps=None, **kwargs):
        inputs_embeds = self._arm_signals(input_ids, graphs, injection_maps)
        kwargs.pop("inputs_embeds", None)
        kwargs.pop("input_ids", None)
        try:
            return self.llm(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                labels=labels, **kwargs)
        finally:
            # Disarm signals; keep armed under gradient checkpointing (backward recomputes).
            if not getattr(self.llm, "is_gradient_checkpointing", False):
                self._pe_signal = None
                self._pe_C = None
                self._pe_Psi = None
                self._pe_c_row = None
                self._pe_taps = None

    def prepare_generation(self, input_ids, graphs, injection_maps, permutation=None):
        """Arm the in-attention injection and return ``inputs_embeds``.

        The caller hands the returned embeds to ``self.llm.generate`` and must reset
        ``self._pe_signal``/``self._pe_C`` to None afterwards. Injection auto-skips
        cached decode steps, so only prompt tokens carry the signal.
        """
        return self._arm_signals(
            input_ids, graphs, injection_maps, permutation=permutation)
