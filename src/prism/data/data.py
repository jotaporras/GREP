import ast
import json
import re
from typing import Optional, no_type_check

import datasets
from torch_geometric.data import Batch
from transformers.data.data_collator import DataCollatorForLanguageModeling

from prism.data import compact_prompt
from prism.data import utils
from prism.models.gnn_llm import build_injection_map, find_last_graph_scope, node_token_variants


def node_index_columns(input_ids, node_token_seqs, *, scope_start, answer_start):
    """Partition node-name token positions into the two graph_acc index lists.

    ``build_injection_map`` finds every node-name span in the sequence; we split
    those positions into two DISJOINT groups by the assistant-turn boundary:

    * ``scene_idx``  — mentions in ``[scope_start, answer_start)``: the query
      scene-graph block (and any node named in the task prompt). ``scope_start``
      (from :func:`find_last_graph_scope`) drops mentions inside earlier ICL-example
      graphs, mirroring the injection scope used in training/eval.
    * ``answer_idx`` — mentions at/after ``answer_start``: node names the model emits
      in its final answer.

    Positions are sequence indices; with right-padding they're valid in the padded
    batch, so the trainer scatters them straight into a ``[B, S]`` mask.
    """
    spans = build_injection_map(input_ids, node_token_seqs, scope_start=0)
    positions = sorted({p for s in spans.values() for (a, b) in s for p in range(a, b)})
    scene_idx = [p for p in positions if scope_start <= p < answer_start]
    answer_idx = [p for p in positions if p >= answer_start]
    return scene_idx, answer_idx


def _find_subsequence(haystack, needle):
    """First (start, end) index span where `needle` occurs contiguously in `haystack`."""
    n, m = len(haystack), len(needle)
    if m == 0 or m > n:
        return None
    first = needle[0]
    for i in range(n - m + 1):
        if haystack[i] == first and haystack[i:i + m] == needle:
            return (i, i + m)
    return None


def edge_list_token_positions(full_text, input_ids, tokenizer):
    """Token positions covering the edge-list bullet block in a tokenized example.

    Locates the ``• Region Edges:`` … ``• Object Edges:`` span (emitted in the
    leading system message by ``compact_prompt._graph_block`` when edges are
    present) and returns the ``input_ids`` indices spanning it — the supervised
    target for the multistage Stage-2 edge-list-reconstruction loss
    (``loss_target='edge_list'``).

    Primary path: char span via the fast tokenizer's offset mapping, trusted only
    when re-encoding reproduces the chat-template ids exactly (true for the
    Gemma/Llama templates, whose special tokens round-trip as literal text).
    Fallback: contiguous subsequence match of the edge-block tokens. Returns ``[]``
    if the block can't be located (the trainer then leaves that example unmasked).
    """
    try:
        start_char = full_text.index("• Region Edges:")
    except ValueError:
        return []
    obj = full_text.find("• Object Edges:", start_char)
    if obj == -1:
        return []
    nl = full_text.find("\n", obj)
    end_char = nl if nl != -1 else len(full_text)

    input_ids = list(input_ids)
    # The edge bullets are plain text, but the char-span end can spill onto the
    # chat-template turn terminator (and on Llama the next role-header), which the
    # tokenizer emits as SPECIAL ids — never part of the edge list. Drop those so
    # the supervised span is exactly the edge tokens.
    special = set(getattr(tokenizer, "all_special_ids", []) or [])

    # Primary: offset mapping, trusted only when the ids round-trip exactly.
    try:
        enc = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
        if list(enc["input_ids"]) == input_ids:
            return [i for i, (a, b) in enumerate(enc["offset_mapping"])
                    if b > start_char and a < end_char and enc["input_ids"][i] not in special]
    except Exception:
        pass

    # Fallback: locate the edge-block token subsequence directly in input_ids.
    edge_text = full_text[start_char:end_char]
    for variant in (edge_text, "\n" + edge_text, " " + edge_text):
        needle = tokenizer(variant, add_special_tokens=False)["input_ids"]
        span = _find_subsequence(input_ids, needle)
        if span is not None:
            lo, hi = span
            while hi > lo and input_ids[hi - 1] in special:
                hi -= 1
            return list(range(lo, hi))
    return []


def assistant_token_positions(messages, input_ids, tokenizer):
    """Token positions of EVERY assistant turn in a tokenized example.

    Supervised target for ``loss_target='responses'`` (assistant-only loss).
    Multi-turn-correct (every assistant turn, not just the last) and
    template-agnostic — no ``{% generation %}`` template support needed, so it works
    for Gemma/Qwen/Llama alike.

    Anchored on each assistant turn's CONTENT, not on chat-template length
    arithmetic: ``len(apply_chat_template(messages[:i], ...))`` is NOT a stable token
    index into the full tokenization (Gemma's template ``| trim``s content and the
    independent re-render shifts boundaries), which silently truncated the first
    token(s) of every turn. Instead we locate the content's char span and map it to
    token indices the same robust way ``edge_list_token_positions`` does:

    * Primary: char span via the fast tokenizer's offset mapping, trusted only when
      re-encoding reproduces ``input_ids`` exactly.
    * Fallback: contiguous subsequence match of the content tokens.

    Each span is extended by the immediately following turn-terminator special token
    (e.g. Gemma ``<end_of_turn>``) so the model still learns to stop, but never into
    the next turn. A turn whose content can't be located is skipped (the trainer
    leaves it unmasked rather than masking the wrong tokens). The cumulative
    ``cursor`` makes repeated content match the correct (later) occurrence.
    """
    input_ids = list(input_ids)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    special = set(getattr(tokenizer, "all_special_ids", []) or [])

    # Primary path: offset mapping, trusted only when the ids round-trip exactly.
    try:
        enc = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc["offset_mapping"] if list(enc["input_ids"]) == input_ids else None
    except Exception:
        offsets = None

    positions = []
    cursor = 0
    for m in messages:
        if m.get("role") != "assistant":
            continue
        # Templates commonly `| trim` content; match the rendered (stripped) form.
        content = (m.get("content") or "").strip()
        if not content:
            continue
        cs = full_text.find(content, cursor)
        if cs == -1:
            continue
        ce = cs + len(content)
        cursor = ce

        if offsets is not None:
            span = [i for i, (a, b) in enumerate(offsets)
                    if b > cs and a < ce and input_ids[i] not in special]
        else:
            needle = tokenizer(content, add_special_tokens=False)["input_ids"]
            found = _find_subsequence(input_ids, needle)
            span = list(range(*found)) if found is not None else []
        if not span:
            continue

        # Include the single immediately-following terminator special (turn-end), so
        # the model learns to stop — but not a run that would reach the next header.
        end = span[-1] + 1
        if end < len(input_ids) and input_ids[end] in special:
            end += 1
        positions.extend(range(span[0], end))
    return positions


def preprocess_dataset(
    ds: datasets.Dataset,
    tokenizer,
    architecture: str,
    text_edge_list: str,
) -> datasets.Dataset:
    """Prepare a raw JSON dataset for training.

    Applies three transforms in order:
    1. Rename ``conversations`` → ``messages``.
    2. Translate to the compact format via ``spine_to_compact_messages``. The
       ``text_edge_list`` policy is resolved ONCE into
       ``include_edges = (text_edge_list == "present")`` and threaded uniformly to
       every architecture: when present, the LLM-facing scene-graph block carries
       the ``• Region Edges:`` / ``• Object Edges:`` bullets; when absent, the
       block lists node names only. For graph archs the GNN still ingests the FULL
       structural edges (parsed from the ORIGINAL messages by ``_parse_scene_graph``)
       regardless of this flag — the flag toggles only the LLM-facing text, which
       is what enables the "PE/mask + text edges" vs "PE/mask only" ablation.
    3. Tokenize with the chat template, keeping ``conversations`` and
       ``messages`` columns for the collator, then filter out examples that
       have no assistant turn.
    """
    @no_type_check
    def _tokenize(example):
        tokenized = tokenizer.apply_chat_template(
            example["messages"], tokenize=True, return_dict=True
        )
        tokenized["conversations"] = example["conversations"]
        tokenized["messages"] = example["messages"]
        return tokenized

    def _parse_scene_graph(example):
        full_text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        m = re.search(r"[Ss]cene graph:", full_text)
        start = full_text.index("{", m.end())
        tail = full_text[start:]
        try:
            sg, _ = json.JSONDecoder().raw_decode(tail)
        except json.JSONDecodeError:
            # Some rollouts serialize the scene graph as a Python repr
            # (single-quoted dict). Fall back to a safe literal parse over
            # the balanced-braces slice.
            depth, end = 0, None
            for i, ch in enumerate(tail):
                if ch == "{": depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end is None:
                raise
            sg = ast.literal_eval(tail[:end])
        all_names = [n["name"] for n in sg["objects"] + sg["regions"]]
        seen, duplicates = set(), set()
        for name in all_names:
            if name in seen:
                duplicates.add(name)
            seen.add(name)
        if duplicates:
            print(f"WARNING!!! Duplicate node labels found: {sorted(duplicates)}")
        example["scene_graph_dict"] = sg
        return example
    
    ds = ds.map(lambda e: {"messages": e["conversations"]})

    # Resolve the edge policy ONCE: `text_edge_list == "present"` is the single
    # source of truth for whether the LLM-facing scene-graph block carries edge
    # bullets. It is applied UNIFORMLY to every architecture (construction-time
    # inclusion — we never build edges then strip them). For graph archs the GNN
    # still ingests the full structural edges from the ORIGINAL messages below
    # regardless; this flag toggles only the text the LLM reads.
    include_edges = (text_edge_list == "present")

    is_graph_arch = architecture in ("rpearl_llm", "rpearl_gt_llm", "gt_llm", "graph_mask_llm", "composite_graph_gt")
    if is_graph_arch:
        # Parse the GNN's scene graph from the ORIGINAL messages first, so the
        # compact translation below (which only rewrites the LLM-facing text)
        # leaves the graph the GNN ingests untouched — the GNN always sees the
        # full connectivity, independent of `include_edges`.
        ds = ds.map(_parse_scene_graph)
        # Translate the verbose SPINE text to the compact format the LLM is
        # trained/evaluated on. `include_edges` gates ONLY the LLM-facing edge
        # bullets here; the GNN keeps the structural edges from `_parse_scene_graph`.
        def _translate_to_compact(example):
            example["messages"] = compact_prompt.spine_to_compact_messages(
                example["conversations"], include_edges=include_edges
            )
            return example
        ds = ds.map(_translate_to_compact)
    else:
        # Plain-LLM baseline: the SAME compact format as the graph archs. With
        # `include_edges` the scene-graph block carries the edge bullets so the
        # LLM (which has no GNN) can read connectivity from text; without it the
        # block lists node names only.
        #
        # Parse the scene graph from the ORIGINAL (verbose) messages first — the LLM
        # ingests no graph, but the node names feed the graph-token-accuracy metric
        # (`graph_acc/*`), so the baseline is comparable to the graph archs.
        ds = ds.map(_parse_scene_graph)
        def _translate_to_compact_llm(example):
            example["messages"] = compact_prompt.spine_to_compact_messages(
                example["conversations"], include_edges=include_edges
            )
            return example
        ds = ds.map(_translate_to_compact_llm)
    ds = ds.map(_tokenize)
    ds = ds.filter(lambda e: any(m.get("role") == "assistant" for m in e["messages"]))

    # Precompute the graph-token index columns for the graph_acc/* training metric.
    # These are a static function of each tokenized example, so they're computed
    # once here (not per-batch in the collator). With right-padding a token's index
    # is identical in the unpadded example and the padded batch, so plain index
    # lists suffice — the collator carries them through and the trainer scatters
    # them into a [B, S] mask.
    def _add_graph_token_indices(example):
        sg = example["scene_graph_dict"]
        names = [n["name"] for n in (sg.get("objects", []) + sg.get("regions", []))]
        variants = node_token_variants(names, tokenizer)
        input_ids = example["input_ids"]
        # `answer_start` is the exact prompt length when the final (assistant) turn is
        # dropped and the generation prompt re-added — the first answer token index.
        answer_start = min(
            len(tokenizer.apply_chat_template(
                example["messages"][:-1], tokenize=True,
                add_generation_prompt=True, return_dict=False)),
            len(input_ids))
        scene_idx, answer_idx = node_index_columns(
            input_ids, variants,
            scope_start=find_last_graph_scope(input_ids, tokenizer),
            answer_start=answer_start)
        example["scene_node_idx"] = scene_idx
        example["answer_node_idx"] = answer_idx
        example["answer_start"] = answer_start
        # Supervised-token spans for the masked loss_target modes (carried through
        # the collator, applied in the trainer's compute_loss):
        #   assistant_idx  -> loss_target='responses' (assistant-only loss)
        #   edge_list_idx  -> loss_target='edge_list'  (Stage-2 PE reconstruction)
        example["assistant_idx"] = assistant_token_positions(
            example["messages"], input_ids, tokenizer)
        # Edge bullets only exist in the text when include_edges; empty otherwise.
        if include_edges:
            full_text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
            example["edge_list_idx"] = edge_list_token_positions(full_text, input_ids, tokenizer)
        else:
            example["edge_list_idx"] = []
        return example
    ds = ds.map(_add_graph_token_indices)
    return ds


def remove_edge_list(decoded: str) -> str:
    """Remove the edge list (object_connections and region_connections) from
    a decoded prompt string containing a scene graph.

    Handles both single-quoted Python repr (training data) and double-quoted
    multiline JSON (SPINE ``GraphHandler.to_json_str`` with ``indent=2``).

    Parameters
    ----------
    decoded : str
        The full decoded prompt text that contains a scene graph with
        ``object_connections`` and ``region_connections`` entries.

    Returns
    -------
    str
        The prompt with both connection lists removed.
    """
    # Training data: single-quoted, single-line Python repr
    decoded = re.sub(
        r"'object_connections': .+?, 'region_connections': .+?, (?='robot_location'|\})",
        "", decoded,
    )
    # SPINE eval: double-quoted, multiline JSON (json.dumps with indent=2).
    # Keys are separated by ,\n<indent> rather than ", " so we use ,\s* between them.
    # Trailing comma is optional (absent when region_connections is the last key).
    decoded = re.sub(
        r'"object_connections":\s*.+?,\s*"region_connections":\s*.+?,?\s*(?="robot_location"|\})',
        "", decoded, flags=re.DOTALL,
    )
    return decoded


class TokenIndexCollator(DataCollatorForLanguageModeling):
    """Causal-LM collator that carries the precomputed graph-token index columns
    (``scene_node_idx`` / ``answer_node_idx``) through batching untouched.

    Those columns are static per example (built in ``preprocess_dataset``); with
    right-padding each token index stays valid in the padded batch, so this collator
    does NO graph logic — it only keeps the parent ``DataCollatorForLanguageModeling``
    from trying to tensorize the ragged int lists. Used directly for the plain-``llm``
    baseline and as the base for ``SpineDataCollator``. The trainer's graph-token
    accuracy metric reads the two columns and pops them before the model forward.
    """

    # Non-tensor / bookkeeping columns stripped before the parent pads the batch.
    _DROP_KEYS = {"conversations", "scene_graph", "scene_graph_dict", "messages",
                  "text", "full_text", "answer_start"}
    _PASSTHROUGH_KEYS = ("scene_node_idx", "answer_node_idx", "assistant_idx", "edge_list_idx")

    def __call__(self, features, return_tensors: Optional[str] = None):
        passthrough = {
            k: [f[k] for f in features]
            for k in self._PASSTHROUGH_KEYS if features and k in features[0]
        }
        drop = self._DROP_KEYS | set(self._PASSTHROUGH_KEYS)
        sanitized = [{k: v for k, v in f.items() if k not in drop} for f in features]
        batch = super().__call__(sanitized)
        batch.update(passthrough)
        return batch


class SpineDataCollator(TokenIndexCollator):
    """SPINE scene-graph collator.

    Expects ``scene_graph_dict`` to already be parsed in each example
    (by ``preprocess_dataset``).  Converts to PyG graphs, computes
    injection maps, and batches them alongside the padded token tensors —
    plus the graph-token index pass-through inherited from ``TokenIndexCollator``.
    """

    def _extract_graph(self, example):
        """Build PyG graph and injection map from a preprocessed example."""
        pyg_graph = utils.scene_graph_dict_to_pyg(example["scene_graph_dict"])
        # Standalone + space-preceded tokenizations per node so every list / edge
        # mention binds (100% injection); see node_token_variants.
        node_token_seqs = node_token_variants(pyg_graph.node_names, self.tokenizer)
        # Scope injection to the last (query) graph block, matching eval
        # (GraphAugmentedInMemoryLLM) and R10. Without this, training cross-links
        # the query graph's labels to their mentions across the *whole* prompt —
        # including the ICL examples — so the composite-graph structure the model
        # learns diverges from the scoped structure it sees at inference (and the
        # gap widens with more ICL examples).
        scope_start = find_last_graph_scope(example["input_ids"], self.tokenizer)
        injection_map = build_injection_map(
            example["input_ids"], node_token_seqs, scope_start=scope_start
        )
        return pyg_graph, injection_map

    def __call__(self, features, return_tensors: Optional[str] = None):
        """Attach parsed PyG graphs and injection maps for each example.

        Graphs are extracted from the raw features first (they still carry
        ``scene_graph_dict``); ``TokenIndexCollator`` then strips the non-tensor
        columns, pads the batch and re-attaches the graph-token index lists.
        """
        pyg_graphs = []
        injection_maps = []
        for example in features:
            pyg_graph, injection_map = self._extract_graph(example)
            pyg_graphs.append(pyg_graph)
            injection_maps.append(injection_map)

        batch = super().__call__(features)
        batch["graphs"] = Batch.from_data_list(pyg_graphs)
        batch["injection_maps"] = injection_maps
        return batch
