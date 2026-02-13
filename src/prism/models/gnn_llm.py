import re
from bisect import bisect_left

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
        offset_mappings: torch.Tensor | list | None = None,
        labels: torch.Tensor | None = None,
        graphs: list | None = None,
        **kwargs,
    ):
        # Get the batch size for incoming data.
        batch_size = input_ids.shape[0]

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
            pos_enc = pos_encs[b]

            # Format offset_mappings as a list of tuples.
            offset_mappings[b] = list(map(
                tuple, offset_mappings[b].tolist()
                    if isinstance(offset_mappings[b], torch.Tensor) else offset_mappings[b]
            ))

            # Extract index-ranges of node names within prompt.
            prompt = self.tokenizer.decode(input_ids[b], skip_special_tokens=True)
            pattern = re.compile('|'.join(re.escape(node_name) for node_name in pos_enc))
            matches = pattern.finditer(prompt)

            # Find indices at which to inject NPEs using binary search for tuple ranges.
            bucket = {}
            for match in matches:
                x, y = match.span()
                A = offset_mappings[b]
                l = bisect_left(A, (x,))
                bucket[match.group()] = [i for i in range(l, len(A)) if A[i][1] <= y]

            # Inject NPEs into embeddings in list.
            for word, token_indices in bucket.items():
                if word in pos_encs[b]:
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
