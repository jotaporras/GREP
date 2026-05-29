import copy
from collections import defaultdict

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm
from torch_geometric.data import Batch, Data
from transformers import PreTrainedModel

from prism.models.augmented_graph import build_augmented_graph
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
        
        # Learnable scalar gain for PE injection magnitude (Audit Rec 3).
        # Replaces the F.normalize + target_norm heuristic which discarded
        # structurally meaningful norm differences between nodes.
        # Initialized small so PE doesn't overwhelm embeddings at the start.
        self.pe_gain = nn.Parameter(torch.tensor(0.00, device=device))

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
            pe = pe * torch.tanh(self.pe_gain)
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

    No RoPE or positional transform is applied here (the LLM is RoPE-disabled, M8).

    Args:
        d_model (int): Embedding width (only used for ``gate_per_dim``).
        gate_init (float): Initial gate value (≈0 cold-start, R6).
        gate_per_dim (bool): Per-feature gate vector instead of a scalar.
        injection_mode (str): "interpolate" (default) or "additive".
    """

    def __init__(self, d_model: int, gate_init: float = 0.0,
                 gate_per_dim: bool = False, injection_mode: str = "interpolate"):
        super().__init__()
        if injection_mode not in ("interpolate", "additive"):
            raise ValueError(
                f"injection_mode must be 'interpolate' or 'additive', got {injection_mode!r}"
            )
        self.injection_mode = injection_mode
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
        gate = self.gate
        if self.injection_mode == "interpolate":
            return (1 - gate) * X + gate * Y_tx
        return X + gate * Y_tx


class AugmentedGraphLLM(PreTrainedModel):  # ty:ignore[unsupported-base]
    """Augmented-graph assembly: M6 Graph Transformer → M7 gate → M8 RoPE-disabled Llama.

    For each sequence: the token embeddings ``X`` become the directed-cycle node
    features, the Graph Transformer (``gt_model``) refines them over the augmented
    graph (R-PEARL Ψ fused as ``X_full + Ψ``), the token-node rows ``Y[V_Tx]`` are
    blended back with ``X`` through the cold-start gate (M7), and the result is fed
    as ``inputs_embeds`` to the RoPE-disabled Llama (M8). Scene-node outputs are
    discarded.

    Inputs match the existing ``SpineDataCollator`` contract — a PyG Batch of
    scene graphs (``graphs``) and per-sample ``injection_maps`` ({scene_node_idx →
    token spans}). For each sequence the augmented graph G is assembled on the fly
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
        cycle_weight, cycle_directed, crosslink_*: M4 augmented-graph settings.
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

    def _augmented_graph(self, scene, injection_map, c, device, permutation=None):
        """Assemble the augmented graph G for one sequence (M4) on ``device``.

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
        return build_augmented_graph(
            c, scene_edge_index, scene_edge_weight, n_scene, injection_map,
            cycle_weight=self.cycle_weight, cycle_directed=self.cycle_directed,
            crosslink_weight=self.crosslink_weight,
            crosslink_mention_to_node=self.crosslink_mention_to_node,
            crosslink_mention_clique=self.crosslink_mention_clique,
        )

    def _fuse_embeddings(self, input_ids, graphs, injection_maps, permutation=None):
        """Build G per sequence and run M4→M6→M7, returning ``inputs_embeds`` [B, c, d]."""
        X = self.llm.get_input_embeddings()(input_ids)  # [B, c, d]
        device = X.device
        c = input_ids.shape[1]
        fused = []
        for b in range(input_ids.shape[0]):
            aug = self._augmented_graph(graphs[b], injection_maps[b], c, device, permutation=permutation)
            aug_data = Data(
                x=torch.zeros(aug.num_nodes, 1, device=device),
                edge_index=aug.edge_index,
            )
            aug_data.edge_weight = aug.edge_weight
            # M6: token embeddings on V_Tx, zeros on V_Sc, fused with R-PEARL Ψ.
            Y = self.gt_model(aug_data, token_embeddings=X[b], is_token=aug.is_token)
            # M7: blend X with the token-node outputs Y[V_Tx] through the gate.
            fused.append(self.injection(X[b], Y[aug.is_token]))
        return torch.stack(fused, dim=0)

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
