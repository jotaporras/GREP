import copy
from collections import defaultdict

import torch
from torch import nn
from transformers import PreTrainedModel
from torch_geometric.data import Batch
from torch.nn import functional as F

class GraphAugmentedLLM(PreTrainedModel):  # ty:ignore[unsupported-base]
    """
    Graph-Augmented LLM (GREP-PRISM).

    Domain-agnostic: receives pre-computed injection maps that specify where
    to add GNN positional encodings into the LLM input embeddings.

    Args:
        llm (nn.Module): LLM to perform classical planning.
        pe_model (nn.Module): R-PEARL positional-encodings model.
        pe_dim (int): Dimensionality of the positional encodings.
    """

    def __init__(self, llm: nn.Module, pe_model: nn.Module, pe_dim: int):
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
            nn.Linear(pe_dim, llm.config.hidden_size, device=device),
            nn.LayerNorm(llm.config.hidden_size, device=device),
        )

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
        """Compute GNN-augmented input embeddings. Shared by forward() and generate().

        Args:
            input_ids: [B, seq_len] token IDs.
            graphs: PyG Batch (or list) of graphs, one per batch element.
            injection_maps: Per-batch-element dict mapping node index to a list
                of (start, end) token spans where that node's PE should be added.
        """
        embeddings = (
            self.llm.get_input_embeddings()(input_ids)
                .clone()
                .to(input_ids.device)
        )  # [B, seq_len, d]

        for b in range(input_ids.shape[0]):
            pe = self.pe_proj(self.pe_model(graphs[b]))  # [n, hidden_size]
            # Rescale PE to match embedding norm. pe_proj ends with LayerNorm
            # which forces output to norm ≈ sqrt(d_model), while LLM embeddings
            # (pre-RMSNorm) are much smaller. Without rescaling, PE overwhelms.
            target_norm = embeddings[b].norm(dim=-1).mean().detach()
            pe = F.normalize(pe, dim=-1) * target_norm
            for node_idx, spans in injection_maps[b].items():
                for start, end in spans:
                    end = min(end, input_ids.shape[1])
                    embeddings[b, start:end] = embeddings[b, start:end, :] + pe[node_idx, :]

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
            pad_token_id=self.tokenizer.eos_token_id,
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
