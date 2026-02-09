import re
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
        pe_dim (int): Output dimension of pe_model (projected to llm.config.hidden_size).
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
        # Get the batch size for incoming data.
        batch_size = input_ids.shape[0]

        # Associate full words to token indices.
        buckets = self.bucketize_prompt(input_ids, self.tokenizer)

        # Get positional encodings per batch element, projected to LLM hidden dim.
        pos_encs = []
        pe: torch.Tensor | None = None
        for b in range(batch_size):
            graph = graphs[b]
            pe = self.pe_proj(self.pe_model(graph))  # (num_nodes, hidden_size)
            pos_enc = {}
            for i, word in enumerate(graph.node_names):
                pos_enc[word] = pe[i]
            pos_encs.append(pos_enc)

        # Clone so that the in-place-style writes don't break the autograd graph
        # of get_input_embeddings.
        embeddings = self.llm.get_input_embeddings()(input_ids).to(pe.device).clone()

        # Add positional encodings to embeddings.
        for b in range(batch_size):
            bucket = buckets[b]
            pos_enc = pos_encs[b]
            for word, token_indices in bucket.items():
                if word in pos_enc:
                    for pos in token_indices:
                        embeddings[b, pos, :] = embeddings[b, pos, :] + pos_enc[word]

        # Drop keys we're overriding so the LLM doesn't get duplicate arguments.
        kwargs.pop("inputs_embeds", None)
        kwargs.pop("input_ids", None)

        return self.llm(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    @classmethod
    def bucketize_prompt(cls, input_ids: torch.Tensor | None, tokenizer: nn.Module) -> list[defaultdict]:
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

        # Get prompt and token list.

        #tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze())
        buckets = []
        for b in range(input_ids.shape[0]):
            current_seq = input_ids[b,:]
            words = re.findall(rf'\b[\w_]+\b', tokenizer.decode(current_seq))
            tokens = tokenizer.convert_ids_to_tokens(current_seq)
            # Get map of words to token locations.
            bucket = defaultdict(list)
            j = 0
            for word in words:
                while not (word.startswith(tokens[j]) or tokens[j] in word or word in tokens[j]):
                    j += 1
                if word.startswith(tokens[j]):
                    bucket[word].append(j)
                    j += 1
                    while tokens[j] in bucket or word.endswith(tokens[j]):
                        bucket[word].append(j)
                        j += 1
                elif tokens[j] in word or word in tokens[j]:
                    bucket[word].append(j)
                    j += 1
            buckets.append(bucket)
        return buckets
