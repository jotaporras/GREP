"""Tests for the training-time graph-token accuracy metric (``graph_acc/*``).

Two pieces:

* ``data.node_index_columns`` — partitions node-name token positions into the
  disjoint ``scene_node_idx`` (query scene block / prompt) and ``answer_node_idx``
  (final answer) lists, dropping ICL-example mentions before the query scope.
* ``GraphTokenAccuracyMixin`` — teacher-forced next-token accuracy restricted to
  those positions. The subtle bit is the shift: a node token at sequence index ``p``
  is predicted by ``logits[p-1]``, so the mask is applied in the shifted target frame.

All deterministic, CPU, no model weights.
"""
import sys
from types import SimpleNamespace

sys.path.insert(0, "src")

import torch
from pytest import approx

from prism.data.data import node_index_columns
from prism.eval.evaluate import GraphTokenAccuracyMixin


# --------------------------------------------------------------------------
# node_index_columns: disjoint scene / answer partition by the answer boundary
# --------------------------------------------------------------------------
#   ids index: 0:99 1:5 2:6 3:100 4:101 5:102 6:5 7:6 8:7 9:5 10:6
# node A = token ids [5, 6] ⇒ mentions at positions (1,2), (6,7), (9,10).
IDS = [99, 5, 6, 100, 101, 102, 5, 6, 7, 5, 6]
SEQS = [[[5, 6]]]  # one node, one tokenization variant


def test_partition_splits_scene_and_answer_at_boundary():
    scene, answer = node_index_columns(IDS, SEQS, scope_start=1, answer_start=8)
    assert scene == [1, 2, 6, 7]   # mentions before the answer turn
    assert answer == [9, 10]       # mention inside the answer turn


def test_partition_drops_icl_mentions_before_scope():
    # scope_start past the first mention ⇒ it's an ICL-example mention, excluded.
    scene, answer = node_index_columns(IDS, SEQS, scope_start=3, answer_start=8)
    assert scene == [6, 7]
    assert answer == [9, 10]


def test_partition_absent_node_yields_empty():
    scene, answer = node_index_columns(IDS, [[[42, 43]]], scope_start=0, answer_start=8)
    assert scene == [] and answer == []


# --------------------------------------------------------------------------
# GraphTokenAccuracyMixin: shift-aligned accuracy + windowed flush
# --------------------------------------------------------------------------
class _Recorder:
    """Stands in for the Trainer below the mixin: records compute_loss inputs and
    the final logs dict."""

    def __init__(self):
        self.logged = None
        self.seen_keys = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self.seen_keys = set(inputs.keys())
        seqlen = inputs["input_ids"].shape[1]
        outputs = SimpleNamespace(logits=inputs["_logits"])
        assert seqlen  # silence unused
        return (torch.tensor(0.0), outputs)

    def log(self, logs, *args, **kwargs):
        self.logged = dict(logs)
        return None


class _Trainer(GraphTokenAccuracyMixin, _Recorder):
    def __init__(self):
        _Recorder.__init__(self)
        self.args = SimpleNamespace(device=torch.device("cpu"), world_size=1)


def _logits_predicting(targets_per_pos, vocab=10):
    """Build [1, S, V] logits whose argmax at t is targets_per_pos[t]."""
    s = len(targets_per_pos)
    logits = torch.zeros(1, s, vocab)
    for t, tok in enumerate(targets_per_pos):
        logits[0, t, tok] = 10.0
    return logits


def test_accumulate_shift_alignment_and_masks():
    t = _Trainer()
    t._reset_token_acc()
    # input_ids = [1,2,3,4,5]; predictions at t=0..3 target positions 1..4.
    # Make t=0,2,3 correct and t=1 wrong (predict 9 instead of 3).
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    logits = _logits_predicting([2, 9, 4, 5, 0])  # pos4 logit unused (shifted off)
    outputs = SimpleNamespace(logits=logits)
    # scene positions include 0 (must be skipped: no prediction targets index 0).
    t._accumulate_token_acc(outputs, input_ids, scene_idx=[[0, 1, 3]], answer_idx=[[2, 4, 5]])
    # scene p=1 -> preds[0]==2 ✓, p=3 -> preds[2]==4 ✓  (p=0 skipped)
    # answer p=2 -> preds[1]==9≠3 ✗, p=4 -> preds[3]==5 ✓  (p=5 out of range, skipped)
    assert t._gta == {"scene_c": 2, "scene_n": 2, "ans_c": 1, "ans_n": 2}


def test_accumulate_is_windowed_across_calls():
    t = _Trainer()
    t._reset_token_acc()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    outputs = SimpleNamespace(logits=_logits_predicting([2, 9, 4, 5, 0]))
    t._accumulate_token_acc(outputs, input_ids, scene_idx=[[1, 3]], answer_idx=[[2, 4]])
    t._accumulate_token_acc(outputs, input_ids, scene_idx=[[1, 3]], answer_idx=[[2, 4]])
    assert t._gta == {"scene_c": 4, "scene_n": 4, "ans_c": 2, "ans_n": 4}


def test_accumulate_handles_batch_of_two():
    t = _Trainer()
    t._reset_token_acc()
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    logits = torch.zeros(2, 5, 10)
    for tok, pos in zip([2, 9, 4, 5, 0], range(5)):
        logits[0, pos, tok] = 10.0          # row 0: t=1 wrong
    for tok, pos in zip([2, 3, 4, 5, 0], range(5)):
        logits[1, pos, tok] = 10.0          # row 1: all correct
    outputs = SimpleNamespace(logits=logits)
    t._accumulate_token_acc(outputs, input_ids, scene_idx=[[1, 3], [1, 3]],
                            answer_idx=[[2, 4], [2, 4]])
    # scene: row0 {p1✓,p3✓}, row1 {p1✓,p3✓} -> 4/4
    # answer: row0 {p2✗,p4✓}, row1 {p2✓,p4✓} -> 3/4
    assert t._gta == {"scene_c": 4, "scene_n": 4, "ans_c": 3, "ans_n": 4}


def test_accumulate_noop_when_no_indices():
    t = _Trainer()
    t._reset_token_acc()
    outputs = SimpleNamespace(logits=torch.zeros(1, 5, 10))
    t._accumulate_token_acc(outputs, torch.tensor([[1, 2, 3, 4, 5]]),
                            scene_idx=None, answer_idx=None)
    assert t._gta == {"scene_c": 0, "scene_n": 0, "ans_c": 0, "ans_n": 0}


def test_log_emits_ratios_and_resets():
    t = _Trainer()
    t._reset_token_acc()
    t._gta = {"scene_c": 2, "scene_n": 2, "ans_c": 1, "ans_n": 2}
    logs = {"loss": 0.5}
    t.log(logs)
    assert t.logged["graph_acc/scene_block"] == approx(1.0)
    assert t.logged["graph_acc/answer_nodes"] == approx(0.5)
    assert t.logged["loss"] == 0.5  # existing keys preserved
    assert t._gta == {"scene_c": 0, "scene_n": 0, "ans_c": 0, "ans_n": 0}  # reset


def test_log_skips_metric_with_no_counts():
    t = _Trainer()
    t._reset_token_acc()  # all zero
    t.log({"loss": 0.5})
    assert "graph_acc/scene_block" not in t.logged
    assert "graph_acc/answer_nodes" not in t.logged


def test_compute_loss_pops_index_columns_and_returns_loss():
    t = _Trainer()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    inputs = {
        "input_ids": input_ids,
        "_logits": _logits_predicting([2, 9, 4, 5, 0]),
        "scene_node_idx": [[1, 3]],
        "answer_node_idx": [[2, 4]],
    }
    loss = t.compute_loss(None, inputs, return_outputs=False)
    assert isinstance(loss, torch.Tensor) and loss.ndim == 0     # scalar loss only
    # index columns were popped before reaching the model / inner compute_loss
    assert "scene_node_idx" not in t.seen_keys
    assert "answer_node_idx" not in t.seen_keys
    # and the metric was accumulated
    assert t._gta["scene_n"] == 2 and t._gta["ans_n"] == 2
