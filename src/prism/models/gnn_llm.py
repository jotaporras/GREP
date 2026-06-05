import copy
import importlib
from collections import defaultdict

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm
from torch_geometric.data import Batch, Data
from transformers import PreTrainedModel

from prism.models.composite_graph import build_composite_graph
from prism.models.llama import disable_rope
from prism.models.utils import LipschitzNorm


class GraphAugmentedLLM(PreTrainedModel):  # ty:ignore[unsupported-base]
    """
    Graph-Augmented LLM (GREP-PRISM).

    Domain-agnostic: receives pre-computed injection maps that specify where
    to add graph positional encodings into the LLM input embeddings.

    The graph encoder (``pe_model``) can be any module with the interface
    ``forward(data) -> Tensor[n, d_model]``.  Two options are supported:
      - ``RandomGNNPositionalEncodings`` (R-PEARL only, no GT blocks)
      - ``GraphTransformer`` (full Sparse GT with R-PEARL inside)

    Args:
        llm (nn.Module): LLM to perform classical planning
        pe_model (nn.Module): R-PEARL positional-encodings model
        d_model (int): Dimensionality of the positional encodings
        eps (float): Lipschitz normalization epsilon for the projection head
    """

    def __init__(self, llm: nn.Module, pe_model: nn.Module,
                 d_model: int, eps: float = 1e-8):
        # GraphAugmentedLLM is not a registered HF architecture, so
        # PreTrainedModel rejects SDPA/flash-attn.  Force "eager" on the
        # wrapper config — the inner self.llm keeps its own attn impl.
        config = copy.copy(llm.config)
        config._attn_implementation = "eager" # ty: ignore[invalid-assignment]
        super().__init__(config)
        self.llm = llm

        # Place pe_model and pe_proj on the same device as the LLM so PEFT
        # wrapping (which only touches LoRA target modules) doesn't leave them on CPU.
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = llm.device
        self.pe_model = pe_model.to(device)
        self.pe_proj = nn.Sequential(
            spectral_norm(nn.Linear(d_model, llm.config.hidden_size, device=device)),
            LipschitzNorm(llm.config.hidden_size, eps=eps, device=device),
        )
        
        # Learnable gate on the PE injection: g = tanh(pe_gain) ∈ (-1,1). Lets the
        # model regulate how strongly (and with which sign) Ψ enters the sum X + g·Ψ.
        # Init pe_gain = 1.0 → g ≈ 0.76, so Ψ is active from the first step (near the
        # token-embedding scale) and the optimizer can scale it down toward 0 or up
        # toward ±‖X‖. The init is deliberately nonzero: pe_gain=0 would give
        # tanh(0)=0 and switch the positional signal off entirely at the start.
        self.pe_gain = nn.Parameter(torch.tensor(1.0, device=device))

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)  # defer to nn.Module first
        except AttributeError:
            return getattr(self.llm, name)

    def _augment_embeddings(
        self,
        input_ids: torch.Tensor,
        graphs: Batch,
        injection_maps: list[dict[int, list[tuple[int, int]]]],
    ) -> torch.Tensor:
        """Compute GNN/GT-augmented input embeddings. Shared by forward() and generate().

        Args:
            input_ids: [B, seq_len] token IDs.
            graphs: PyG Batch (or list) of graphs, one per batch element.
            injection_maps: Per-batch-element dict mapping node index to a list
                of (start, end) token spans where that node's PE should be added.
        """
        # Clone so that in-place additions below don't corrupt the embedding table's
        # gradient. The clone itself is in the autograd graph, so gradients flow
        # through the additions back to pe_model / pe_proj normally.
        embeddings = (
            self.llm.get_input_embeddings()(input_ids)
                .clone()
                .to(input_ids.device)
        )  # [B, seq_len, d]

        for b in range(input_ids.shape[0]):
            pe = self.pe_proj(self.pe_model(graphs[b]))  # [n, hidden_size]
            # Scale Ψ to the token-embedding magnitude, then apply the learnable gate,
            # so the LLM sees RoPE(X + g·Ψ): a genuine sum the model can regulate.
            # pe_proj ends in a LipschitzNorm, so Ψ exits at ~unit norm — only ~4% of
            # ‖X‖ (≈24 for Llama) — and would be drowned out; matching it to the mean
            # token-embedding norm makes it a full-strength positional signal, and
            # g = tanh(pe_gain) ∈ (-1,1) (init ≈ 0.76) gates it: active from the start,
            # bounded by ‖X‖. Mirrored in inference.py so eval matches training.
            pe = pe * embeddings[b].norm(dim=-1).mean().detach() * torch.tanh(self.pe_gain)
            for node_idx, spans in injection_maps[b].items():
                for start, end in spans:
                    end = min(end, input_ids.shape[1])
                    if start < end:
                        embeddings[b, start:end] = embeddings[b, start:end] + pe[node_idx]

        return embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        graphs: Batch | None = None,
        injection_maps: list[dict[int, list[tuple[int, int]]]] | None = None,
        **kwargs,
    ):
        embeddings = self._augment_embeddings(input_ids, graphs, injection_maps)

        kwargs.pop("inputs_embeds", None)
        kwargs.pop("input_ids", None)

        return self.llm(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )


class GatedInjection(nn.Module):
    """M7 — blend token embeddings with the Graph Transformer output via a gate.

    Produces the LLM ``inputs_embeds`` from the original token embeddings ``X``
    and the GT token-node outputs ``Y[V_Tx]`` (M6), through a single learnable
    gate that starts at ≈0 (R6 cold-start). Scene-node outputs are discarded;
    only the token rows reach the LLM.

    Modes (spec §4/M7):
      - ``"interpolate"`` (default): ``inputs_embeds = (1 - gate) * X + gate * Y``.
        The gate is the fraction by which the LLM's own embeddings are replaced
        by the structural output — at ``gate=0`` the LLM sees clean Llama
        embeddings, and structure ramps in as the gate grows.
      - ``"additive"``: ``inputs_embeds = X + gate * Y``.
      - ``"none"``: ``inputs_embeds = Y`` — the Graph Transformer output is fed
        straight to the LLM with **no gate and no mixing of the token embeddings
        X**. There is no learnable gate in this mode; the LLM must consume the
        structural representation directly. Pairing it with a LoRA warmup (the
        LLM frozen for the first N optimizer steps, see ``LoraWarmupCallback``)
        lets the GT learn to emit an LLM-consumable representation before the
        adapters start adapting to it. ``gate`` is registered as a fixed
        non-trainable buffer of 1.0 so the M11 diagnostics
        (``grep/structural_gate``, ``grep/contrib_ratio``) keep reading sensibly.

    No RoPE or positional transform is applied here (the LLM is RoPE-disabled, M8).

    Args:
        d_model (int): Embedding width (only used for ``gate_per_dim``).
        gate_init (float): Initial gate value (≈0 cold-start, R6). Ignored when
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
            # No learnable gate: the GT output replaces X outright. Keep a fixed
            # gate=1.0 buffer (not a Parameter) so callbacks/optimizer code that
            # reads ``injection.gate`` still works and reports the true full-strength
            # injection (contrib_ratio = ‖Y‖/‖X‖) without adding a trainable param.
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
    """Composite-graph assembly: M6 Graph Transformer → M7 gate → M8 RoPE-disabled Llama.

    For each sequence: the token embeddings ``X`` become the directed-cycle node
    features, the Graph Transformer (``gt_model``) refines them over the augmented
    graph (R-PEARL Ψ fused as ``X_full + Ψ``), the token-node rows ``Y[V_Tx]`` are
    blended back with ``X`` through the cold-start gate (M7), and the result is fed
    as ``inputs_embeds`` to the RoPE-disabled Llama (M8). Scene-node outputs are
    discarded.

    Inputs match the existing ``SpineDataCollator`` contract — a PyG Batch of
    scene graphs (``graphs``) and per-sample ``injection_maps`` ({scene_node_idx →
    token spans}). For each sequence the composite graph G is assembled on the fly
    (M4): a directed cycle over the ``c`` token positions, the scene graph, and the
    cross-links from the injection map. Token embeddings X seed the cycle nodes,
    R-PEARL + the GT refine over G, and ``Y[V_Tx]`` is gated back into X (M7).

    Args:
        llm (nn.Module): Llama for causal LM (RoPE disabled here unless told not to).
        gt_model (nn.Module): GraphTransformer (M6); ``forward(data, token_embeddings,
            is_token)`` returns per-node features ``Y``.
        d_model (int): Embedding / GT width (must equal the LLM hidden size).
        gate_init, gate_per_dim, injection_mode: M7 gate settings (R6).
        disable_llm_rope (bool): Apply the M8 RoPE disable to ``llm`` (default True).
        cycle_weight, cycle_directed, crosslink_*: M4 composite-graph settings.
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
        """Assemble the composite graph G for one sequence (M4) on ``device``.

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
                         return_token_pe=False):
        """Build G per sequence and run M4→M6→M7, returning ``inputs_embeds`` [B, c, d].

        When ``return_token_pe`` is True, also returns the un-gated GT token-row
        output ``Y[V_Tx]`` stacked as ``[B, c, d]`` — the per-position structural
        code that ``InjectedCompositeGraphLLM`` injects into q/k/v at every layer.
        """
        X = self.llm.get_input_embeddings()(input_ids)  # [B, c, d]
        device = X.device
        c = input_ids.shape[1]
        fused = []
        token_pe = [] if return_token_pe else None
        for b in range(input_ids.shape[0]):
            aug = self._composite_graph(graphs[b], injection_maps[b], c, device, permutation=permutation)
            aug_data = Data(
                x=torch.zeros(aug.num_nodes, 1, device=device),
                edge_index=aug.edge_index,
            )
            aug_data.edge_weight = aug.edge_weight
            # M6: token embeddings on V_Tx, zeros on V_Sc, fused with R-PEARL Ψ.
            Y = self.gt_model(aug_data, token_embeddings=X[b], is_token=aug.is_token)
            Ytok = Y[aug.is_token]
            # M7: blend X with the token-node outputs Y[V_Tx] through the gate.
            fused.append(self.injection(X[b], Ytok))
            if return_token_pe:
                token_pe.append(Ytok)
        # The GT runs in float32 (sparse sampled_addmm is fp32-only) and the gate
        # promotes the bf16 X to fp32, so cast inputs_embeds back to the LLM's dtype.
        # Training tolerated the fp32 mismatch under autocast; generate() has no
        # autocast, so the fp32 hidden state would hit the bf16 lm_head and raise
        # "expected scalar type Float but found BFloat16".
        inputs_embeds = torch.stack(fused, dim=0).to(X.dtype)
        if return_token_pe:
            return inputs_embeds, torch.stack(token_pe, dim=0)
        return inputs_embeds

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
    """Composite-graph LLM that injects the GT code into q/k/v *inside attention* at
    every layer, in place of RoPE.

    Written in eval_unification's style (per-instance patched attention forward,
    ``_install_pe_injection`` / ``_make_injected_attention_forward`` / ``_pe_signal`` /
    ``_pe_inject_value`` / ``_decoder_layers``, model-agnostic via ``importlib``) so the
    two files read the same — but the *mathematics* is this branch's RoPE-replacement
    design, not e-u's ``RoPE(X)+Ψ``. Per attention layer ``l``::

        q_l = W^q_l · h_l + W_q · S
        k_l = W^k_l · h_l + W_k · S
        v_l = W^v_l · h_l + W_v · S            (if ``inject_v``)

    Differences from e-u, by design:
      - **RoPE is OFF** (``disable_llm_rope=True``, identity rotary): the injected code
        is the sole position signal, present at every depth — there is no native
        rotation on the content q/k.
      - **Dedicated projections** ``W_q``/``W_k``/``W_v`` (shared across layers, the
        "shared/raw" map) carry the code into the q/k/v spaces — *not* the LLM's own
        content projections.
      - The signal is the GT-refined composite second-moment code
        ``S = Y_tok = GT([X;Ψ] + C·[X;Ψ])[V_Tx]`` (un-gated; the M7 gate is on the
        layer-0 blend, below).
      - ``inputs_embeds`` is the **gated GT blend** ``M7(X, Y_tok)`` (the Layer-0
        injection), *then* the same code is re-injected into q/k/v at every layer.

    The code is supplied per forward via ``self._pe_signal`` ([B, seq, hidden]) and,
    matching e-u's plumbing, skipped on cached single-token decode steps (seq mismatch)
    — so only prompt tokens carry it. (With RoPE off, generated tokens then have no
    position; that is the known trade-off of replacing RoPE.)

    The attention forward is patched per-instance (faithful copy of the HF
    implementation); it survives PEFT, which swaps the leaf Linears in place.
    """

    def __init__(self, *args, inject_v: bool = True,
                 disable_llm_rope: bool = True, **kwargs):
        # RoPE OFF by default (disable_llm_rope=True): the injected code replaces RoPE.
        kwargs["disable_llm_rope"] = disable_llm_rope
        super().__init__(*args, **kwargs)

        cfg = self.llm.config
        hidden = cfg.hidden_size
        n_heads = cfg.num_attention_heads
        n_kv = getattr(cfg, "num_key_value_heads", n_heads)
        head_dim = getattr(cfg, "head_dim", None) or (hidden // n_heads)
        try:
            device = next(self.llm.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        # Dedicated shared projections from the d_model code into the q / k (/ v) spaces
        # (one set, reused at every layer). Default Linear init (std ≈ 1/√d_model) maps a
        # code of row-norm ≈‖X‖ to a per-element std comparable to the attention logits'
        # scale. No bias (a constant shift across positions carries no order). q gets
        # H·Dh, k/v get Hkv·Dh (GQA).
        self.pe_q_proj = nn.Linear(hidden, n_heads * head_dim, bias=False, device=device)
        self.pe_k_proj = nn.Linear(hidden, n_kv * head_dim, bias=False, device=device)
        if inject_v:
            self.pe_v_proj = nn.Linear(hidden, n_kv * head_dim, bias=False, device=device)

        # Per-forward graph code S = Y_tok ([B, seq, hidden]); read by the patched
        # attention forwards, set by forward()/prepare_generation().
        self._pe_signal = None
        self._pe_inject_value = inject_v
        self._install_pe_injection()

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
        # Resolve the model-family helpers from the module that defines this attention
        # class (Llama, Qwen2, … share this layout).
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

            # --- replace RoPE: add the GT code through dedicated projections at every
            # layer. RoPE is off (identity rotary), so the rotation above is a no-op and
            # the code is the sole position signal. ---
            psi = model._pe_signal
            if (psi is not None and psi.shape[0] == hidden_states.shape[0]
                    and psi.shape[1] == hidden_states.shape[1]):
                def _proj(linear):
                    out = linear(psi.to(linear.weight.dtype)).to(query_states.dtype)
                    return out.view(hidden_shape).transpose(1, 2)
                query_states = query_states + _proj(model.pe_q_proj)
                key_states = key_states + _proj(model.pe_k_proj)
                if model._pe_inject_value:
                    value_states = value_states + _proj(model.pe_v_proj)
            # ------------------------------------------------------------------

            if past_key_values is not None:
                key_states, value_states = past_key_values.update(
                    key_states, value_states, attn.layer_idx)

            attention_interface = attn_fns.get_interface(
                attn.config._attn_implementation, eager)
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
    def _build_signal(self, input_ids, graphs, injection_maps, permutation=None):
        """Return (inputs_embeds, code S=[B, seq, hidden]).

        ``inputs_embeds`` is the **gated GT blend** ``M7(X, Y_tok)`` (the Layer-0
        injection); the code ``S = Y_tok`` is then re-injected into q/k/v at every layer
        by the patched attention forward.
        """
        inputs_embeds, token_pe = self._fuse_embeddings(
            input_ids, graphs, injection_maps, permutation=permutation,
            return_token_pe=True)
        return inputs_embeds, token_pe

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                graphs=None, injection_maps=None, **kwargs):
        inputs_embeds, self._pe_signal = self._build_signal(input_ids, graphs, injection_maps)
        kwargs.pop("inputs_embeds", None)
        kwargs.pop("input_ids", None)
        try:
            return self.llm(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                labels=labels, **kwargs)
        finally:
            # Disarm S unless gradient checkpointing will recompute the attention
            # forwards in backward (they must see the same S; it is rebuilt every
            # forward anyway). Mirrors eval_unification.
            if not getattr(self.llm, "is_gradient_checkpointing", False):
                self._pe_signal = None

    def prepare_generation(self, input_ids, graphs, injection_maps, permutation=None):
        """Arm the in-attention injection and return ``inputs_embeds`` (plain X).

        The caller hands the returned X to ``self.llm.generate`` and must reset
        ``self._pe_signal = None`` afterwards. Injection auto-skips cached decode
        steps, so only prompt tokens carry S (generated tokens ride on RoPE).
        """
        inputs_embeds, self._pe_signal = self._build_signal(
            input_ids, graphs, injection_maps, permutation=permutation)
        return inputs_embeds


def find_last_graph_scope(input_ids_b, tokenizer) -> int:
    """Token index where the last scene-graph block begins (injection scope).

    Only mentions at/after this index belong to the last (query) graph; earlier
    matches live inside ICL-example graphs and must be ignored so the query
    graph's labels don't cross-link into ICL regions. Spec R10 locks this:
    "infra inputs only the last (query) graph and scopes injection after it
    completes."

    Used by BOTH the training collator (``SpineDataCollator``) and eval
    (``GraphAugmentedInMemoryLLM``) so the composite graph is assembled with the
    same scope in train and eval — otherwise the cross-link structure the model
    trains on differs from what it sees at inference (the divergence grows with
    ``n_icl_examples``). Returns 0 when no marker is found (whole sequence
    eligible), matching the previous default.
    """
    seq = list(map(int, input_ids_b))
    scope_start = 0
    for surface in ("scene graph:", " scene graph:", "Scene graph:", " Scene graph:"):
        marker = tokenizer.encode(surface, add_special_tokens=False)
        if not marker:
            continue
        for pos in range(len(seq) - len(marker) + 1):
            if seq[pos:pos + len(marker)] == marker:
                scope_start = max(scope_start, pos)
    return scope_start


def has_match(input_ids_b: list[int], to_match:list[int],start_pos:int):
    """ 
    For a single sequence, check if `to_match` is present at `start_pos`
    """
    end_pos = min(start_pos + len(to_match),len(input_ids_b))
    return input_ids_b[start_pos:end_pos] == to_match

def build_injection_map(
    input_ids_b: list[int],
    node_token_seqs: list[list[int]],
    scope_start: int = 0,
) -> dict[int, list[tuple[int, int]]]:
    """Build a pre-computed injection map from token IDs and node token sequences.

    Wraps ``bucketize_prompt`` and returns the ``{node_idx: [(start, end), ...]}``
    format expected by ``GraphAugmentedLLM._augment_embeddings``.

    Two refinements over the raw buckets (M3):

    - **Scope (``scope_start``):** only matches starting at/after ``scope_start``
      are kept, so node labels that also appear in earlier ICL-example graphs
      are ignored and PE lands only on the last (query) graph block. The caller
      computes the boundary (see ``GraphAugmentedInMemoryLLM._generate_tokens``);
      the default of 0 keeps the whole sequence eligible.
    - **Longest-first matching:** nodes are resolved from the longest token span
      down, claiming the token positions they cover. A label that is a token
      prefix of a longer one (``barn_shed_1`` inside ``barn_shed_11``) therefore
      can't steal the longer label's tokens, while a genuine standalone mention
      elsewhere is still picked up.

    Args:
        input_ids_b: Flat list of token IDs for a single sequence.
        node_token_seqs: Per-node list of token-ID subsequences
            (as returned by ``tokenizer.encode(node_names, add_special_tokens=False)``).
        scope_start: First token index eligible for matching.

    Returns:
        Dict mapping node index to a list of ``(start, end)`` token spans.
    """
    buckets = bucketize_prompt(input_ids_b, node_token_seqs)

    # Resolve longest labels first so a longer mention claims its tokens before
    # any shorter prefix label can match inside it.
    order = sorted(buckets, key=lambda nid: len(node_token_seqs[nid]), reverse=True)
    claimed: set[int] = set()
    injection_map: dict[int, list[tuple[int, int]]] = {}
    for nid in order:
        length = len(node_token_seqs[nid])
        spans = []
        for start in sorted(buckets[nid]):
            if start < scope_start:
                continue
            positions = range(start, start + length)
            if claimed.isdisjoint(positions):
                spans.append((start, start + length))
                claimed.update(positions)
        if spans:
            injection_map[nid] = spans
    return injection_map


def bucketize_prompt(input_ids_b: list, node_token_seqs : list) -> defaultdict:
    """
    Helper function for associating node token sequences with their positions
    in a tokenized prompt. Uses parallel iteration through the token list.

    Args:
        input_ids_b (list): Flat list of token IDs for a single sequence.
        node_token_seqs (list): Per-node list of token-ID subsequences.

    Returns:
        buckets (defaultdict[int, set]): Mapping from node index to the set of
            start positions where that node's token sequence appears.
    """
    # Get map of words to token locations.
    buckets = defaultdict(set)
    for p_idx, p_token in enumerate(input_ids_b):
        for node_idx, node_token_seq in enumerate(node_token_seqs):
            if has_match(input_ids_b, to_match=node_token_seq,start_pos=p_idx):
                buckets[node_idx].add(p_idx)
    return buckets
