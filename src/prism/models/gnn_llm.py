from collections import defaultdict

import torch
from torch import nn

class GraphAugmentedLLM(nn.Module):
    """
    Graph-Augmented LLM (GREP-PRISM).

    Args:
        llm (nn.Module): LLM to perform classical planning.
        pe_model (nn.Module): R-PEARL positional-encodings model.
        tokenizer: (nn.Module): Tokenizer associated with LLM.
    """

    def __init__(self, llm: nn.Module, pe_model: nn.Module, tokenizer: nn.Module, pe_dim: int):
        super().__init__()
        self.llm = llm
        self.config = llm.config
        self.tokenizer = tokenizer

        # Place pe_model and pe_proj on the same device as the LLM so PEFT
        # wrapping (which only touches LoRA target modules) doesn't leave them on CPU.
        device = next(llm.parameters()).device
        self.pe_model = pe_model.to(device)
        self.pe_proj = nn.Linear(pe_dim, llm.config.hidden_size, device=device)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)  # defer to nn.Module first
        except AttributeError:
            return getattr(self.llm, name)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        graphs: list | None = None,
        **kwargs,
    ):
        # input_ids: [B,seq_len]
        
        
        # Let's start with LLM embeddings since we can compute them in batch.
        embeddings = (
            self.llm.get_input_embeddings()(input_ids)
                .clone()
                .to(input_ids.device)
        ) # [B,seq_len,d]


        # Now to inject positional encodings.
        for b in range(input_ids.shape[0]):
            graph = graphs[b]

            node_token_seqs = self.tokenizer.encode(graph.node_names)
            bucket = bucketize_prompt(input_ids[b,:].tolist(), node_token_seqs)
    
            # Get positional encodings.
            pe = self.pe_proj(self.pe_model(graph))  # [n, hidden_size]
    
            for node_idx, match_idxes in bucket.items():
                for start in match_idxes:
                    max_len = input_ids.shape[1] #TO DO: worst case we'll inject PEs into some padded tokens but would be rare.
                    end = min(start + len(node_token_seqs[node_idx]), max_len)
                    embeddings[b,start:end] = embeddings[b,start:end,:] + pe[node_idx,:]

        kwargs.pop("inputs_embeds", None)
        kwargs.pop("input_ids", None)

        return self.llm(
            inputs_embeds=embeddings,
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

def bucketize_prompt(input_ids_b: list, node_token_seqs : list) -> defaultdict:
    """
    Helper function for associating full prompt words with their corresponding token indices.
    Uses parallel iteration through words alongside the token list.

    Args:
        input_ids (torch.Tensor): List of one-hot encodings for prompt.
        tokenizer (nn.Module): LLM tokenizer required to decode input IDs.

    Returns:
        bucket (dict): mappings for adding operation of positional encodings
            to respective tokens.
    """
    # Get map of words to token locations.
    buckets = defaultdict(set)
    for p_idx, p_token in enumerate(input_ids_b):
        for node_idx, node_token_seq in enumerate(node_token_seqs):
            if has_match(input_ids_b, to_match=node_token_seq,start_pos=p_idx):
                buckets[node_idx].add(p_idx)
    return buckets