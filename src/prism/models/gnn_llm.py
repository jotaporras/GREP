import copy
import importlib
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
from torch_geometric.data import Batch
from transformers import PreTrainedModel

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

    Positional-injection scheme — the LLM sees ``RoPE(X) + Ψ`` at every layer:
        ``X`` is the word-embedding matrix and ``Ψ`` the graph PE placed at the
        node-name token spans. RoPE is applied *inside every attention layer* to
        the projected query/key (``apply_rotary_pos_emb``), so adding Ψ to the
        residual stream would make the LLM compute ``RoPE(W·(X+Ψ))`` — Ψ spun by
        the *sequence-position* rotation, which is wrong for a *graph* positional
        code. (A residual-stream counter-rotation ``R_p^{-1}Ψ`` can't fix this:
        ``W_q``/``W_k`` don't commute with the per-head rotation, and one residual
        vector can't satisfy the q- and k-constraints at once.)

        Instead we patch each attention layer's forward (``_install_pe_injection``)
        to add Ψ *after* RoPE, in the projected query/key/value space::

            q = RoPE(W_q · h) + W_q · Ψ
            k = RoPE(W_k · h) + W_k · Ψ
            v =      W_v · h  + W_v · Ψ     # value/content path ("content too")

        Ψ is projected through that layer's own (LoRA-adapted) q/k/v_proj, so the
        graph signal enters the query/key dot product *unrotated* — exact
        ``RoPE(X) + Ψ`` — at all layers. Ψ is supplied per forward via
        ``self._pe_signal`` (``[B, seq, hidden]``); injection is skipped on cached
        single-token decode steps (seq mismatch), so only prompt tokens carry it.

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
        # model regulate how strongly (and with which sign) Ψ enters RoPE(X) + g·Ψ.
        # Init pe_gain = 1.0 → g ≈ 0.76, so Ψ is active from the first step (near the
        # token-embedding scale) and the optimizer can scale it down toward 0 or up
        # toward ±‖X‖. The init is deliberately nonzero: pe_gain=0 would give
        # tanh(0)=0 and switch the positional signal off entirely at the start.
        self.pe_gain = nn.Parameter(torch.tensor(1.0, device=device))

        # Per-forward graph signal Ψ ([B, seq, hidden]); read by the patched
        # attention forwards and set by _augment_embeddings / inference.
        self._pe_signal: torch.Tensor | None = None
        # Whether Ψ also enters the value/content path (v += W_v·Ψ).
        self._pe_inject_value: bool = True
        self._install_pe_injection()

    def _decoder_layers(self):
        """Return the LLM's decoder layer list (Llama/Qwen2: ``<CausalLM>.model.layers``)."""
        base = getattr(self.llm, "model", None)
        if base is not None and hasattr(base, "layers"):
            return base.layers
        return self.llm.get_decoder().layers

    def _install_pe_injection(self) -> None:
        """Patch every self-attention layer so Ψ is added *after* RoPE (RoPE(X)+Ψ).

        Each attention module's ``forward`` is replaced with a faithful copy of the
        HF implementation that additionally injects ``W_q·Ψ`` / ``W_k·Ψ`` into the
        post-rotary query/key (and ``W_v·Ψ`` into the value path when
        ``_pe_inject_value``). Ψ is projected through the same (LoRA-adapted)
        projections as the content stream, so the graph code enters the q·k score
        unrotated at all layers. Patching instance methods (not the class) keeps it
        scoped to this LLM and survives PEFT, which swaps leaf Linears in place.
        """
        for layer in self._decoder_layers():
            attn = layer.self_attn
            attn.forward = self._make_injected_attention_forward(attn)

    def _make_injected_attention_forward(self, attn):
        # Resolve the model-family helpers from the module that defines this
        # attention class (works for Llama, Qwen2, … which share this layout).
        mod = importlib.import_module(type(attn).__module__)
        apply_rotary_pos_emb = mod.apply_rotary_pos_emb
        attn_fns = mod.ALL_ATTENTION_FUNCTIONS
        eager = mod.eager_attention_forward
        model = self  # captured for the per-forward Ψ signal

        def forward(hidden_states, position_embeddings=None, attention_mask=None,
                    past_key_values=None, **kwargs):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, attn.head_dim)

            query_states = attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            # --- RoPE(X) + Ψ : add the graph signal AFTER the rotation ---------
            psi = model._pe_signal
            if (psi is not None and psi.shape[0] == hidden_states.shape[0]
                    and psi.shape[1] == hidden_states.shape[1]):
                psi = psi.to(query_states.dtype)
                query_states = query_states + attn.q_proj(psi).view(hidden_shape).transpose(1, 2)
                key_states = key_states + attn.k_proj(psi).view(hidden_shape).transpose(1, 2)
                if model._pe_inject_value:
                    value_states = value_states + attn.v_proj(psi).view(hidden_shape).transpose(1, 2)
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

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)  # defer to nn.Module first
        except AttributeError:
            return getattr(self.llm, name)

    def build_pe_signal(
        self,
        embeddings: torch.Tensor,
        graphs: Batch,
        injection_maps: list[dict[int, list[tuple[int, int]]]],
        permutation=None,
    ) -> torch.Tensor:
        """Assemble the graph signal Ψ ([B, seq, hidden]) placed at node-token spans.

        Ψ is consumed *inside* the patched attention layers (added post-RoPE), so
        here we only build the placed, scaled, gated signal — no rotation.

        Args:
            embeddings: [B, seq, hidden] token embeddings (for magnitude scaling).
            graphs: PyG Batch (or list) of graphs, one per batch element.
            injection_maps: Per-batch-element dict mapping node index to a list of
                (start, end) token spans where that node's PE should be placed.
            permutation: Optional node permutation passed to ``pe_model`` (used by
                eval-time permutation-equivariance checks; ``None`` in training).
        """
        B, seq_len, hidden = embeddings.shape
        psi = torch.zeros(B, seq_len, hidden, device=embeddings.device, dtype=embeddings.dtype)
        for b in range(B):
            pe = self.pe_proj(self.pe_model(graphs[b], permutation=permutation))  # [n, hidden_size]
            # Scale Ψ to the token-embedding magnitude, then apply the learnable gate.
            # pe_proj ends in a LipschitzNorm, so Ψ exits at ~unit norm and would be
            # drowned out; matching it to the mean token-embedding norm makes it a
            # full-strength positional signal, and g = tanh(pe_gain) ∈ (-1,1)
            # (init ≈ 0.76) gates it. Mirrored in inference.py so eval matches training.
            pe = pe * embeddings[b].norm(dim=-1).mean().detach() * torch.tanh(self.pe_gain)
            for node_idx, spans in injection_maps[b].items():
                for start, end in spans:
                    end = min(end, seq_len)
                    if start < end:
                        psi[b, start:end] = psi[b, start:end] + pe[node_idx].to(psi.dtype)
        return psi

    def _augment_embeddings(
        self,
        input_ids: torch.Tensor,
        graphs: Batch,
        injection_maps: list[dict[int, list[tuple[int, int]]]],
    ) -> torch.Tensor:
        """Return plain token embeddings and arm Ψ for the patched attention layers.

        Unlike the legacy residual-stream injection, Ψ is no longer added to the
        embeddings here — it is registered on ``self._pe_signal`` and added *after*
        RoPE inside every attention layer (see ``_install_pe_injection``), so the
        LLM sees ``RoPE(X) + Ψ``. Returns the unmodified ``X`` embeddings to feed as
        ``inputs_embeds``.
        """
        embeddings = (
            self.llm.get_input_embeddings()(input_ids)
                .clone()
                .to(input_ids.device)
        )  # [B, seq_len, d]
        self._pe_signal = self.build_pe_signal(embeddings, graphs, injection_maps)
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

        try:
            return self.llm(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )
        finally:
            # Disarm Ψ so a later forward can't inject a stale signal. But under
            # gradient checkpointing the backward pass *recomputes* the attention
            # forwards and must see the same Ψ, so keep it armed in that case (it
            # is rebuilt/overwritten at the start of every forward anyway).
            if not getattr(self.llm, "is_gradient_checkpointing", False):
                self._pe_signal = None


def has_match(input_ids_b: list[int], to_match:list[int],start_pos:int):
    """ 
    For a single sequence, check if `to_match` is present at `start_pos`
    """
    end_pos = min(start_pos + len(to_match),len(input_ids_b))
    return input_ids_b[start_pos:end_pos] == to_match

def build_injection_map(
    input_ids_b: list[int],
    node_token_seqs: list[list[int]],
) -> dict[int, list[tuple[int, int]]]:
    """Build a pre-computed injection map from token IDs and node token sequences.

    Convenience wrapper around ``bucketize_prompt`` that returns the
    ``{node_idx: [(start, end), ...]}`` format expected by
    ``GraphAugmentedLLM._augment_embeddings``.

    Args:
        input_ids_b: Flat list of token IDs for a single sequence.
        node_token_seqs: Per-node list of token-ID subsequences
            (as returned by ``tokenizer.encode(node_names, add_special_tokens=False)``).

    Returns:
        Dict mapping node index to a list of ``(start, end)`` token spans.
    """
    buckets = bucketize_prompt(input_ids_b, node_token_seqs)
    return {
        nid: [(s, s + len(node_token_seqs[nid])) for s in starts]
        for nid, starts in buckets.items()
    }


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
