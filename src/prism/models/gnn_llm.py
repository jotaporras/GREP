import re
from collections import defaultdict

import torch
from torch import nn


from collections import defaultdict


class GraphAugmentedLLM(nn.Module):
    """
    Graph-Augmented LLM (GREP-PRISM).

    Args:
        llm (nn.Module): LLM to perform classical planning.
        pe_model (nn.Module): R-PEARL positional-encodings model.
        tokenizer: (nn.Module): Tokenizer associated with LLM.
    """

    def __init__(self, llm: nn.Module, pe_model: nn.Module, tokenizer: nn.Module):
        super().__init__()
        self.llm = llm
        self.pe_model = pe_model
        self.config = llm.config
        self.tokenizer = tokenizer

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
        # Associate full words to token indices.
        bucket = self.bucketize_prompt(input_ids, self.tokenizer)

        # Get positional encodings.
        graph = graphs[0]
        pe = self.pe_model(graph)
        pos_enc = defaultdict(lambda: torch.Tensor(size=pe.size(), device=pe.device))
        for i, word in enumerate(graph.node_names):
            pos_enc[word] = pe[i]

        # Add positional encodings to embeddings.
        embeddings = (
            self.llm.get_input_embeddings()(input_ids)
                .squeeze(0)
                .to(pe.device)
        )
        for word, token in bucket.items():
            if word in pos_enc:
                for pos in token:
                    embeddings[pos] += pos_enc[word]

        return self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    @classmethod
    def bucketize_prompt(cls, input_ids: torch.Tensor | None, tokenizer: nn.Module) -> defaultdict:
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
        words = re.findall(rf'\b[\w_]+\b',
                            tokenizer.decode(input_ids.squeeze()))
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze())

        # Get map of words to token locations.
        bucket = defaultdict(list)
        j = 0
        for word in words:
            while not (cls.has_prefix_suffix_match(word, tokens[j]) or tokens[j] in word or word in tokens[j]):
                j += 1
            if cls.has_prefix_suffix_match(word, tokens[j]):
                bucket[word].append(j)
                j += 1
                while tokens[j] in bucket or cls.has_prefix_suffix_match(tokens[j], word):
                    bucket[word].append(j)
                    j += 1
            elif tokens[j] in word or word in tokens[j]:
                bucket[word].append(j)
                j += 1
        return bucket

    @staticmethod
    def has_prefix_suffix_match(a: str, b: str) -> bool:
        """Returns True if any prefix of a matches any suffix of b."""
        # Check all possible prefixes of a against suffixes of b
        for i in range(1, len(a)+1):
            prefix_a = a[:i]
            for j in range(1, len(b)+1):
                suffix_b = b[-j:]
                if prefix_a == suffix_b:
                    return True
        return False
