"""Tests for the ``loss_target`` token-masking framework (assistant-only / edge-list loss).

Covers the four pieces added for e9 multistage training, all WITHOUT loading a model
(only CPU tensors and tiny synthetic tokenizers — a real tokenizer/model is exercised
separately by ``scripts/verify_loss_masking.py`` against the gemma stack):

* ``data.assistant_token_positions`` — multi-turn assistant span location, template-agnostic.
* ``data.edge_list_token_positions`` / ``_find_subsequence`` — the ``• Region/Object Edges:``
  block, including special-token exclusion at the span boundary.
* ``train_v2.LossTargetMixin`` — masks ``labels`` to the configured span, pops the index
  columns before the forward, and flips ``model_accepts_loss_kwargs`` so the masked CE is
  mean-over-kept (not under-normalized by the full-sequence ``num_items_in_batch``).
* MRO composition (mask BEFORE the pure-diagnostic accuracy mixin) and ``TrainConfig``
  validation.

All deterministic, CPU, no model weights.
"""
import sys
import warnings
from types import SimpleNamespace

sys.path.insert(0, "src")

import pytest
import torch

from prism.data.data import (
    _find_subsequence,
    assistant_token_positions,
    edge_list_token_positions,
    TokenIndexCollator,
)
from prism.training.train_v2 import (
    _LOSS_TARGET_COLUMN,
    BaselineSFTTrainer,
    GraphSFTTrainer,
    GraphTokenAccuracyMixin,
    LossTargetMixin,
    TrainConfig,
)


# ==========================================================================
# assistant_token_positions — multi-turn, content-anchored, template-agnostic
# ==========================================================================
class _WordChatTokenizer:
    """Faithful word-level mini-tokenizer exercising the content-anchored path.

    Renders a conversation as ``<bos> <role> {trimmed content} <eot> ... [<model>]``
    (the trailing ``<model>`` only with ``add_generation_prompt``), tokenizes on
    spaces, and exposes ``return_offsets_mapping`` + ``all_special_ids`` like a real
    fast tokenizer — so ``assistant_token_positions`` runs its true offset-mapping
    span logic (and terminator extension) with NO model and no real tokenizer. The
    ``| trim`` of content is mimicked so leading/trailing whitespace can't align the
    span by accident.
    """

    _SPECIAL = {"<bos>": 1, "<eot>": 2, "<system>": 3, "<user>": 4, "<model>": 5}
    all_special_ids = list(_SPECIAL.values())

    def _id(self, word):
        if word in self._SPECIAL:
            return self._SPECIAL[word]
        return 1000 + (hash(word) % 9000)            # stable per word, never special

    def render(self, messages, add_generation_prompt=False):
        parts = ["<bos>"]
        for m in messages:
            parts += [f"<{m['role']}>", m["content"].strip(), "<eot>"]
        if add_generation_prompt:
            parts.append("<model>")
        return " ".join(p for p in parts if p)

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids, offs, cur = [], [], 0
        for word in text.split(" "):
            a = text.index(word, cur)
            b = a + len(word)
            cur = b
            ids.append(self._id(word))
            offs.append((a, b))
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offs
        return out

    def apply_chat_template(self, messages, tokenize=True,
                            add_generation_prompt=False, return_dict=False):
        text = self.render(messages, add_generation_prompt)
        return self(text)["input_ids"] if tokenize else text


_MESSAGES = [
    {"role": "system", "content": "S"},
    {"role": "user", "content": "U"},
    {"role": "assistant", "content": "ans_a ans_b"},
    {"role": "user", "content": "q"},
    {"role": "assistant", "content": "ans_c ans_d ans_e"},
]
# Rendered tokens (word index -> token):
#  0 <bos> 1 <system> 2 S 3 <eot> | 4 <user> 5 U 6 <eot>
#  7 <model> 8 ans_a 9 ans_b 10 <eot> | 11 <user> 12 q 13 <eot>
#  14 <model> 15 ans_c 16 ans_d 17 ans_e 18 <eot>           (n=19)


def _ids():
    return _WordChatTokenizer().apply_chat_template(_MESSAGES, tokenize=True)


def test_assistant_positions_cover_content_and_terminator():
    pos = assistant_token_positions(_MESSAGES, _ids(), _WordChatTokenizer())
    # turn 2 -> content [8,9] + terminator 10; turn 4 -> content [15,16,17] + term 18.
    # The leading <model> header (7, 14) is NOT supervised; the <eot> IS (learn to stop).
    assert pos == [8, 9, 10, 15, 16, 17, 18]


def test_assistant_positions_do_not_leak_into_next_turn():
    # the terminator extension must stop at <eot>, never reach the next <user>/<model>.
    pos = assistant_token_positions(_MESSAGES, _ids(), _WordChatTokenizer())
    assert 11 not in pos and 14 not in pos          # next <user> header / next <model>


def test_assistant_positions_empty_without_assistant_turn():
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "hi there"}]
    tok = _WordChatTokenizer()
    assert assistant_token_positions(msgs, tok.apply_chat_template(msgs), tok) == []


def test_assistant_positions_subsequence_fallback_when_offsets_untrusted():
    # Force the offset path off (round-trip mismatch) -> _find_subsequence fallback.
    class _NoOffsetRoundtrip(_WordChatTokenizer):
        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
            out = super().__call__(text, add_special_tokens, return_offsets_mapping)
            if return_offsets_mapping:           # corrupt only the round-trip check input
                out["input_ids"] = [-1] + out["input_ids"]
            return out

    tok = _NoOffsetRoundtrip()
    pos = assistant_token_positions(_MESSAGES, _ids(), tok)
    # fallback finds the content token subsequence; terminator extension still applies.
    assert pos == [8, 9, 10, 15, 16, 17, 18]


# ==========================================================================
# _find_subsequence — contiguous match span
# ==========================================================================
def test_find_subsequence_basic():
    assert _find_subsequence([5, 1, 2, 3, 9], [1, 2, 3]) == (1, 4)


def test_find_subsequence_at_edges():
    assert _find_subsequence([1, 2, 3, 9], [1, 2]) == (0, 2)      # prefix
    assert _find_subsequence([9, 1, 2, 3], [2, 3]) == (2, 4)      # suffix


def test_find_subsequence_absent_and_degenerate():
    assert _find_subsequence([1, 2, 3], [4, 5]) is None
    assert _find_subsequence([1, 2], [1, 2, 3]) is None           # needle longer
    assert _find_subsequence([1, 2, 3], []) is None               # empty needle


# ==========================================================================
# edge_list_token_positions — • Region/Object Edges: block
# ==========================================================================
class _CharTokenizer:
    """Each character is one token with offset (i, i+1). One special id (999)."""

    all_special_ids = [999]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        out = {"input_ids": [ord(c) for c in text]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out


def test_edge_list_spans_region_through_object_block():
    full = "pre • Region Edges: r1-r2\n• Object Edges: o1-o2\nAFTER"
    ids = [ord(c) for c in full]
    pos = edge_list_token_positions(full, ids, _CharTokenizer())
    start = full.index("• Region Edges:")
    end = full.find("\n", full.index("• Object Edges:"))   # stop at the bullet's newline
    assert pos == list(range(start, end))


def test_edge_list_excludes_special_token_at_boundary():
    sentinel = chr(999)            # ord == 999 == _CharTokenizer special id
    full = f"x • Region Edges: a\n• Object Edges: b{sentinel}\nZ"
    ids = [ord(c) for c in full]
    pos = edge_list_token_positions(full, ids, _CharTokenizer())
    sent_idx = full.index(sentinel)
    assert sent_idx not in pos                 # special id dropped from the span
    assert (sent_idx - 1) in pos               # the real 'b' token before it is kept


def test_edge_list_missing_block_returns_empty():
    tok = _CharTokenizer()
    assert edge_list_token_positions("no edges here", [], tok) == []
    # region bullet present but object bullet missing -> still empty (incomplete block)
    only_region = "• Region Edges: a\nsomething else"
    assert edge_list_token_positions(only_region, [ord(c) for c in only_region], tok) == []


# ==========================================================================
# LossTargetMixin — label masking, column popping, normalization flag
# ==========================================================================
class _Recorder:
    """Stands in for the trainer below the mixin: records what reaches compute_loss."""

    def __init__(self):
        self.model_accepts_loss_kwargs = True
        self.seen_labels = None
        self.seen_keys = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self.seen_keys = set(inputs.keys())
        labels = inputs.get("labels")
        self.seen_labels = labels.clone() if labels is not None else None
        outputs = SimpleNamespace(logits=torch.zeros(1))
        return (torch.tensor(0.0), outputs) if return_outputs else torch.tensor(0.0)


class _LossTrainer(LossTargetMixin, _Recorder):
    def __init__(self, loss_target):
        _Recorder.__init__(self)
        self._set_loss_target(loss_target)


def test_set_loss_target_disables_loss_kwargs_only_when_masking():
    assert _LossTrainer("all").model_accepts_loss_kwargs is True
    assert _LossTrainer("responses").model_accepts_loss_kwargs is False
    assert _LossTrainer("edge_list").model_accepts_loss_kwargs is False


def test_compute_loss_masks_to_responses_and_pops_index_columns():
    t = _LossTrainer("responses")
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "labels": torch.tensor([[10, 11, 12, 13, 14]]),
        "assistant_idx": [[1, 3]],
        "edge_list_idx": [[0, 2, 4]],
    }
    loss = t.compute_loss(None, inputs, return_outputs=False)
    assert isinstance(loss, torch.Tensor) and loss.ndim == 0
    # only assistant positions survive; everything else -> -100
    assert t.seen_labels.tolist() == [[-100, 11, -100, 13, -100]]
    # BOTH index columns popped before the forward (never tensorized into the model)
    assert "assistant_idx" not in t.seen_keys
    assert "edge_list_idx" not in t.seen_keys


def test_compute_loss_edge_list_uses_edge_column():
    t = _LossTrainer("edge_list")
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "labels": torch.tensor([[10, 11, 12, 13, 14]]),
        "assistant_idx": [[0, 1]],          # ignored for edge_list target
        "edge_list_idx": [[2, 4]],
    }
    t.compute_loss(None, inputs)
    assert t.seen_labels.tolist() == [[-100, -100, 12, -100, 14]]


def test_compute_loss_all_target_leaves_labels_but_still_pops_columns():
    t = _LossTrainer("all")
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "labels": torch.tensor([[10, 11, 12]]),
        "assistant_idx": [[1]],
        "edge_list_idx": [[2]],
    }
    t.compute_loss(None, inputs)
    assert t.seen_labels.tolist() == [[10, 11, 12]]           # unmasked
    # index columns are popped regardless of target, so they never reach the model
    assert "assistant_idx" not in t.seen_keys
    assert "edge_list_idx" not in t.seen_keys


def test_mask_labels_multi_row():
    t = _LossTrainer("responses")
    inputs = {"labels": torch.tensor([[10, 11, 12], [20, 21, 22]])}
    t._mask_labels_to_positions(inputs, [[1], [0, 2]], "responses")
    assert inputs["labels"].tolist() == [[-100, 11, -100], [20, -100, 22]]


def test_mask_labels_empty_falls_back_to_full_sequence_with_warning():
    t = _LossTrainer("responses")
    inputs = {"labels": torch.tensor([[1, 2, 3]])}
    with pytest.warns(UserWarning, match="no supervised tokens"):
        t._mask_labels_to_positions(inputs, [[]], "responses")
    assert inputs["labels"].tolist() == [[1, 2, 3]]           # unchanged (no all-masked NaN)


# ==========================================================================
# MRO composition: mask (LossTargetMixin) BEFORE diagnostic (GraphTokenAccuracyMixin)
# ==========================================================================
class _DiagBase:
    """Stands in for SFTTrainer below BOTH mixins; records the labels the model sees."""

    def __init__(self):
        self.model_accepts_loss_kwargs = True
        self.seen_labels = None
        self.seen_keys = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self.seen_keys = set(inputs.keys())
        self.seen_labels = inputs["labels"].clone()
        seqlen = inputs["input_ids"].shape[1]
        outputs = SimpleNamespace(logits=torch.zeros(1, seqlen, 10))
        return (torch.tensor(0.0), outputs)


class _Composed(LossTargetMixin, GraphTokenAccuracyMixin, _DiagBase):
    def __init__(self):
        _DiagBase.__init__(self)
        self.args = SimpleNamespace(device=torch.device("cpu"), world_size=1)
        self._set_loss_target("responses")
        self._reset_token_acc()


def test_full_composition_masks_then_pops_all_columns_before_forward():
    c = _Composed()
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "labels": torch.tensor([[10, 11, 12, 13, 14]]),
        "assistant_idx": [[1, 3]],
        "edge_list_idx": [[0]],
        "scene_node_idx": [[2]],
        "answer_node_idx": [[4]],
    }
    loss = c.compute_loss(None, inputs)
    assert isinstance(loss, torch.Tensor) and loss.ndim == 0
    # base saw the MASKED labels -> LossTargetMixin ran before the forward
    assert c.seen_labels.tolist() == [[-100, 11, -100, 13, -100]]
    # every loss-target AND diagnostic index column popped before the model
    for col in ("assistant_idx", "edge_list_idx", "scene_node_idx", "answer_node_idx"):
        assert col not in c.seen_keys


def test_real_trainers_order_mask_before_diagnostic():
    for cls in (GraphSFTTrainer, BaselineSFTTrainer):
        names = [c.__name__ for c in cls.__mro__]
        assert names.index("LossTargetMixin") < names.index("GraphTokenAccuracyMixin") \
            < names.index("SFTTrainer")


# ==========================================================================
# Cross-module invariants and TrainConfig validation
# ==========================================================================
def test_loss_target_columns_are_carried_by_the_collator():
    # every column a loss target masks on MUST be passed through batching, else the
    # mixin would never see it. This guards adding a target without wiring the collator.
    assert set(_LOSS_TARGET_COLUMN.values()) <= set(TokenIndexCollator._PASSTHROUGH_KEYS)


def test_loss_target_column_map_is_stable():
    assert _LOSS_TARGET_COLUMN == {"responses": "assistant_idx", "edge_list": "edge_list_idx"}


def _cfg(**kw):
    return TrainConfig(name="t", checkpoint_dir="/tmp/x", data="d.json", **kw)


def test_config_rejects_unknown_loss_target():
    with pytest.raises(ValueError, match="loss_target"):
        _cfg(loss_target="bogus")


def test_config_edge_list_requires_text_edges_present():
    with pytest.raises(ValueError, match="text_edge_list"):
        _cfg(loss_target="edge_list", text_edge_list="none")


def test_config_accepts_all_valid_targets():
    for tgt in ("all", "responses", "edge_list"):
        assert _cfg(loss_target=tgt, text_edge_list="present").loss_target == tgt
