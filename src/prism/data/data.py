import ast
import glob
import json
import os
import re
from typing import Dict, List, Optional, Tuple, no_type_check

import datasets
from torch_geometric.data import Batch
from transformers.data.data_collator import DataCollatorForLanguageModeling

from prism.data import compact_prompt
from prism.data import utils
from prism.eval import evaluate
from prism.models.gnn_llm import (
    build_injection_map,
    clamp_injection_map,
    exclude_positions_from_injection_map,
    find_last_graph_scope,
    node_token_variants,
)


def load_eval_samples_by_graph(eval_data: str, num_graphs: int) -> Dict[str, List[evaluate.EvalSample]]:
    """Resolve `eval_data` (file, directory, or glob) into `{graph_name: [EvalSample]}`,
    truncated to the first `num_graphs` graphs (sorted by file stem).

    Used by the train-time periodic `EvalCallback` and the `no_train` zero-shot
    baseline so both score the same multi-graph held-out set. A single-file
    `eval_data` resolves to one graph (back-compatible). `num_graphs <= 0` keeps
    all resolved graphs.
    """
    samples_by_graph, _ = load_samples_by_graph(eval_data)
    if num_graphs and num_graphs > 0 and len(samples_by_graph) > num_graphs:
        kept = list(samples_by_graph.items())[:num_graphs]
        dropped = len(samples_by_graph) - num_graphs
        print(f"[eval] capping train-time eval to {num_graphs} of "
              f"{len(samples_by_graph)} graphs ({dropped} dropped)")
        samples_by_graph = dict(kept)
    return samples_by_graph


def load_samples_by_graph(target: str) -> Tuple[Dict[str, List[evaluate.EvalSample]], Dict[str, str]]:
    """Resolve `target` (file, directory, or glob) and load each graph JSON.

    Returns `(samples_by_graph, graph_file_by_name)`, both keyed by graph file
    stem (e.g. ``"data_gen_004"``): the per-graph `EvalSample` list ready for
    `evaluate.eval_model_multiple_graphs`, and the source path of each.

    Raises `SystemExit` if `target` resolves to zero matching files.
    """
    graph_files = _resolve_graph_files(target)

    samples_by_graph: Dict[str, List[evaluate.EvalSample]] = {}
    graph_file_by_name: Dict[str, str] = {}
    for gf in graph_files:
        stem = os.path.splitext(os.path.basename(gf))[0]
        with open(gf) as f:
            payload = json.load(f)
        samples_by_graph[stem] = evaluate.construct_eval_samples_from_dict(
            payload["graph"], payload["tasks"], graph_name=stem,
        )
        graph_file_by_name[stem] = gf
    return samples_by_graph, graph_file_by_name


def _resolve_graph_files(target: str) -> List[str]:
    """Expand `target` (single file, directory of JSONs, or glob) into a sorted list."""
    if os.path.isdir(target):
        files = sorted(glob.glob(os.path.join(target, "*.json")))
    elif any(ch in target for ch in ("*", "?", "[")):
        files = sorted(glob.glob(target))
    else:
        files = [target]
    if not files:
        raise SystemExit(f"No graph JSON files found at {target}")
    return files


def node_index_columns(input_ids, node_token_seqs, *, scope_start, answer_start):
    """Partition node-name token positions into scene_idx and answer_idx.

    - ``scene_idx``  — positions in ``[scope_start, answer_start)``: query scene-graph block.
      ``scope_start`` (from ``find_last_graph_scope``) excludes ICL-example graphs.
    - ``answer_idx`` — positions at/after ``answer_start``: node names in the model's answer.

    Positions are sequence indices valid in the right-padded batch (scattered into a ``[B, S]`` mask).
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
    """Token positions of the ``• Region Edges:`` … ``• Object Edges:`` block.

    Supervised target for ``loss_target='edge_list'`` (Stage-2 edge-list reconstruction).
    Primary: char-span via offset mapping (trusted when re-encode reproduces input_ids).
    Fallback: subsequence match. Returns ``[]`` if the block can't be located.
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
    """Token positions of every assistant turn (supervised target for ``loss_target='responses'``).

    Anchored on turn CONTENT (not chat-template length arithmetic — Gemma's ``| trim`` makes
    independent re-renders shift boundaries). Primary: offset-mapping char-span (trusted when
    re-encode reproduces input_ids). Fallback: subsequence match.

    Each span extends one token to include the turn-terminator (e.g. ``<end_of_turn>``) so the
    model learns to stop. Turns that can't be located are skipped. ``cursor`` advances past
    each found span so repeated content binds to the correct (later) occurrence.
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

    1. Rename ``conversations`` → ``messages``.
    2. Translate to compact format via ``spine_to_compact_messages``.
       ``text_edge_list`` is resolved once to ``include_edges``; for graph archs the GNN
       always reads structural edges from the ORIGINAL messages (flag only gates the LLM-facing text).
    3. Tokenize, filter out examples with no assistant turn, and precompute graph-token index columns.
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

    # text_edge_list=="present" gates edge bullets in the LLM-facing text only;
    # the GNN always reads structural edges from the ORIGINAL messages.
    include_edges = (text_edge_list == "present")

    is_graph_arch = architecture in ("rpearl_llm", "rpearl_gt_llm", "gt_llm", "graph_mask_llm", "composite_graph_gt")
    if is_graph_arch:
        ds = ds.map(_parse_scene_graph)
        # Translate SPINE text to compact format; include_edges gates LLM-facing edge bullets only.
        def _translate_to_compact(example):
            example["messages"] = compact_prompt.spine_to_compact_messages(
                example["conversations"], include_edges=include_edges
            )
            return example
        ds = ds.map(_translate_to_compact)
    else:
        # Plain-LLM: same compact format; include_edges gates text edge bullets.
        # Parse scene graph for the graph_acc/* metric even though the LLM ingests no graph.
        ds = ds.map(_parse_scene_graph)
        def _translate_to_compact_llm(example):
            example["messages"] = compact_prompt.spine_to_compact_messages(
                example["conversations"], include_edges=include_edges
            )
            return example
        ds = ds.map(_translate_to_compact_llm)
    ds = ds.map(_tokenize)
    ds = ds.filter(lambda e: any(m.get("role") == "assistant" for m in e["messages"]))

    # Precompute static graph-token index columns (once here, not per-batch).
    # With right-padding, token indices are identical in padded batches.
    def _add_graph_token_indices(example):
        sg = example["scene_graph_dict"]
        names = [n["name"] for n in (sg.get("objects", []) + sg.get("regions", []))]
        variants = node_token_variants(names, tokenizer)
        input_ids = example["input_ids"]
        # answer_start: prompt length with final assistant turn dropped + generation prompt re-added.
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
        # assistant_idx -> loss_target='responses'; edge_list_idx -> loss_target='edge_list'.
        example["assistant_idx"] = assistant_token_positions(
            example["messages"], input_ids, tokenizer)
        # edge_list_idx is empty when include_edges is False (no edge bullets in text).
        if include_edges:
            full_text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
            example["edge_list_idx"] = edge_list_token_positions(full_text, input_ids, tokenizer)
        else:
            example["edge_list_idx"] = []
        return example
    ds = ds.map(_add_graph_token_indices)
    return ds


def load_and_split_dataset(config, tokenizer):
    """Load, preprocess, and split training data according to ``config``.

    Handles three cases in priority order: explicit pre-split val file
    (``config.data.val_files``), random fraction split (``config.data.val_frac > 0``),
    or no validation set.  When ``config.data.debug`` is True, both train and val
    are downsampled to ``config.data.dataset_proportion`` before splitting.

    Args:
        config: TrainConfig (duck-typed) supplying ``data.train_files``, ``data.val_files``,
            ``data.val_frac``, ``data.debug``, ``data.dataset_proportion``, ``gnn.arch``,
            and ``data.text_edge_list``.
        tokenizer: tokenizer passed through to ``preprocess_dataset``.

    Returns:
        (train_dataset, eval_dataset) — ``eval_dataset`` is ``None`` when no
        validation set is configured.
    """
    full_dataset = datasets.load_dataset("json", data_files=[config.data.train_files], split="train")
    if config.data.debug:
        full_dataset = full_dataset.select(range(round(len(full_dataset) * config.data.dataset_proportion)))

    full_dataset = preprocess_dataset(
        full_dataset, tokenizer,
        architecture=config.gnn.arch,
        text_edge_list=config.data.text_edge_list,
    )

    if config.data.val_files:
        train_dataset = full_dataset
        eval_dataset = datasets.load_dataset("json", data_files=[config.data.val_files], split="train")
        if config.data.debug:
            eval_dataset = eval_dataset.select(range(round(len(eval_dataset) * config.data.dataset_proportion)))
        eval_dataset = preprocess_dataset(
            eval_dataset, tokenizer,
            architecture=config.gnn.arch,
            text_edge_list=config.data.text_edge_list,
        )
        print(f"Using pre-split val file: {len(train_dataset)} train / {len(eval_dataset)} eval")
    elif config.data.val_frac and config.data.val_frac > 0.0:
        dataset_size = len(full_dataset)
        val_size = int(dataset_size * config.data.val_frac)
        train_size = dataset_size - val_size
        split = full_dataset.train_test_split(
            test_size=val_size,
            train_size=train_size,
            seed=3407,
            shuffle=True,
        )
        train_dataset = split["train"]
        eval_dataset = split["test"]
        print(f"Dataset split: {len(train_dataset)} train / {len(eval_dataset)} eval")
    else:
        train_dataset = full_dataset
        eval_dataset = None
        print(f"Using all {len(full_dataset)} samples for training (no validation).")

    return train_dataset, eval_dataset


def remove_edge_list(decoded: str) -> str:
    """Remove ``object_connections`` and ``region_connections`` from a scene-graph prompt string.

    Handles single-quoted Python repr (training data) and double-quoted multiline JSON (SPINE eval).
    """
    # Single-quoted Python repr (training data).
    decoded = re.sub(
        r"'object_connections': .+?, 'region_connections': .+?, (?='robot_location'|\})",
        "", decoded,
    )
    # Double-quoted multiline JSON (SPINE eval); trailing comma optional.
    decoded = re.sub(
        r'"object_connections":\s*.+?,\s*"region_connections":\s*.+?,?\s*(?="robot_location"|\})',
        "", decoded, flags=re.DOTALL,
    )
    return decoded


class TokenIndexCollator(DataCollatorForLanguageModeling):
    """Causal-LM collator that passes precomputed graph-token index columns through batching untouched.

    Strips non-tensor columns and passes ``scene_node_idx`` / ``answer_node_idx`` /
    ``assistant_idx`` / ``edge_list_idx`` as ragged lists (the trainer pops them before forward).
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
    """Collator that builds PyG graphs and injection maps per example, then delegates to TokenIndexCollator.

    ``injection_scope`` (set by the trainer from ``config.data.injection_scope``)
    controls which token positions carry the graph channel during TRAINING forwards:

    - ``"full_sequence"`` (historical behavior): node mentions in the assistant
      answer are injected too — the positions being predicted carry the ground-truth
      node's PE/mask row, a channel that does not exist at generation (decode steps
      receive no injection; ``inference.py`` builds prompt-only maps).
    - ``"prompt_only"``: maps clamped at ``answer_start``, matching generation exactly.
    - ``"exclude_supervised"`` (e12): subtract the loss-target positions (the
      per-example index column named by ``supervised_positions_key``, e.g.
      ``edge_list_idx`` for ``loss_target='edge_list'``) so no supervised token
      carries its own node's channel. Enforces injection ∩ loss-target = ∅ even
      when the supervised block lives in the prompt (edge-list reconstruction).
    """

    injection_scope = "full_sequence"
    # Index column holding the supervised positions; the trainer sets it from
    # trainer.loss_target when injection_scope='exclude_supervised'.
    supervised_positions_key = None

    def _extract_graph(self, example):
        """Build PyG graph and injection map from a preprocessed example."""
        pyg_graph = utils.scene_graph_dict_to_pyg(example["scene_graph_dict"])
        node_token_seqs = node_token_variants(pyg_graph.node_names, self.tokenizer)
        # Scope to the last (query) graph block so ICL-example node mentions don't cross-link.
        scope_start = find_last_graph_scope(example["input_ids"], self.tokenizer)
        injection_map = build_injection_map(
            example["input_ids"], node_token_seqs, scope_start=scope_start
        )
        if self.injection_scope == "prompt_only":
            injection_map = clamp_injection_map(injection_map, example["answer_start"])
        elif self.injection_scope == "exclude_supervised":
            if self.supervised_positions_key is None:
                raise ValueError(
                    "injection_scope='exclude_supervised' requires "
                    "supervised_positions_key to be set (from trainer.loss_target)."
                )
            injection_map = exclude_positions_from_injection_map(
                injection_map, set(example[self.supervised_positions_key])
            )
        return pyg_graph, injection_map

    def __call__(self, features, return_tensors: Optional[str] = None):
        """Extract PyG graphs and injection maps, then delegate to TokenIndexCollator for padding."""
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
