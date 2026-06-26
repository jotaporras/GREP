"""Smoke test for the train_v2 refactor (commit 3f74290 and ancestors).

Verifies:
  1. All refactored modules import clean (dead imports in train_v2 removed).
  2. architectures.peft_tower_exclude — correct with/without tower modules.
  3. architectures.build_planner_model — error paths without needing a real LLM.
  4. evaluate.construct_eval_samples_from_dict — pure logic.
  5. evaluate._aggregate_multi_graph_eval — fold logic correct.
  6. evaluate.print_summary_table — runs without crash.
  7. evaluate.GraphTokenAccuracyMixin._accumulate_token_acc — tensor accumulation.
  8. data.load_and_split_dataset — 3 branch paths (no-val, val_frac, val_data)
     exercised via monkeypatching.

CPU-only; no LLM weights required.
"""
import sys
import types
import torch
import torch.nn as nn

PASS, FAIL = [], []

def check(name, cond, info=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {info}" if info else ""))

# ---------------------------------------------------------------------------
# 1. Module imports
# ---------------------------------------------------------------------------
try:
    from prism.training import train_v2
    check("train_v2 imports clean (dead imports removed)", True)
except Exception as e:
    check("train_v2 imports clean", False, str(e))
    sys.exit(1)

try:
    from prism.models import architectures
    check("architectures module imports clean", True)
except Exception as e:
    check("architectures module imports clean", False, str(e))
    sys.exit(1)

try:
    from prism.eval import evaluate
    check("evaluate module imports clean", True)
except Exception as e:
    check("evaluate module imports clean", False, str(e))

try:
    from prism.data import data
    check("data module imports clean", True)
except Exception as e:
    check("data module imports clean", False, str(e))

try:
    from prism.models import loaders as model_loaders
    check("loaders module imports clean", True)
except Exception as e:
    check("loaders module imports clean", False, str(e))

# Confirm the dead imports are gone
check(
    "gnn_llm not imported in train_v2 namespace",
    not hasattr(train_v2, "gnn_llm"),
)
check(
    "r_pearl_module not imported in train_v2 namespace",
    not hasattr(train_v2, "r_pearl_module"),
)
check(
    "gt_module not imported in train_v2 namespace",
    not hasattr(train_v2, "gt_module"),
)

# ---------------------------------------------------------------------------
# 2. peft_tower_exclude
# ---------------------------------------------------------------------------
class _Plain(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

class _WithTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_tower = nn.Linear(4, 4)
        self.text_layer = nn.Linear(4, 4)

plain = _Plain()
with_tower = _WithTower()

check("peft_tower_exclude: plain model → None",
      architectures.peft_tower_exclude(plain) is None)
exc = architectures.peft_tower_exclude(with_tower)
check("peft_tower_exclude: tower model → regex string",
      isinstance(exc, str) and "vision_tower" in exc, exc)
check("peft_tower_exclude: regex excludes tower module names",
      exc is not None and all(
          k in exc for k in ("vision_tower", "audio_tower")
      ))

# ---------------------------------------------------------------------------
# 3. architectures.build_planner_model — error paths
# ---------------------------------------------------------------------------
def _mock_config(**kw):
    return types.SimpleNamespace(**kw)

try:
    cfg = _mock_config(gnn=types.SimpleNamespace(arch="nonexistent_arch", pe_node_features="random"))
    mock_llm = types.SimpleNamespace(
        config=types.SimpleNamespace(
            get_text_config=lambda: types.SimpleNamespace(hidden_size=64)
        )
    )
    architectures.build_planner_model(cfg, mock_llm, None)
    check("build_planner_model: unknown arch raises ValueError", False, "no exception raised")
except ValueError as e:
    check("build_planner_model: unknown arch raises ValueError", True, str(e)[:60])

try:
    cfg = _mock_config(gnn=types.SimpleNamespace(arch="gt_llm", pe_node_features="random"))
    architectures.build_planner_model(cfg, mock_llm, None)
    check("build_planner_model: gt_llm + pe_node_features=random raises ValueError", False, "no exception")
except ValueError as e:
    check("build_planner_model: gt_llm + pe_node_features=random raises ValueError", True, str(e)[:60])

# ---------------------------------------------------------------------------
# 4. evaluate.construct_eval_samples_from_dict
# ---------------------------------------------------------------------------
graph_dict = {"objects": [{"name": "chair"}], "regions": []}
tasks = [
    {"task": "go to chair", "answer": "chair", "init_node": "entrance"},
    {"task": "find table", "answer": "table", "init_node": "hall",
     "acceptance_criterion": "node_reachable"},
]
samples = evaluate.construct_eval_samples_from_dict(graph_dict, tasks, "graph_001")
check("construct_eval_samples_from_dict: count matches tasks", len(samples) == 2)
check("construct_eval_samples_from_dict: graph_name stamped", samples[0].graph_name == "graph_001")
check("construct_eval_samples_from_dict: acceptance_criterion populated",
      samples[1].acceptance_criterion == "node_reachable")
check("construct_eval_samples_from_dict: no acceptance_criterion → None",
      samples[0].acceptance_criterion is None)

# ---------------------------------------------------------------------------
# 5. evaluate._aggregate_multi_graph_eval
# ---------------------------------------------------------------------------
from prism.eval import evaluate as _ev
dummy_sample = {
    "graph_name": "g", "idx": 0, "task": "t", "answer_key": "a",
    "response": None, "interaction_trace": [], "terminated_by": None,
    "formatted": True, "plan_keyword": True, "correct": True, "structured": False,
    "subjective_correct": None, "false_positive": False, "false_negative": False,
    "llm_judge_pass": None, "error": None, "traceback": None, "path_metrics": None,
}
summary_a = evaluate.GraphEvalResultSummary(
    name="graph_a", num_total=5, num_correct=4, accuracy=0.8,
    subjective_accuracy=None, num_judged=0, num_formatted=5, num_keyword=4,
    num_false_pos=0, num_false_neg=0, num_errors=0, elapsed_s=1.0, n_nodes=6,
    use_icl=True, permutation=None,
    samples=[dummy_sample] * 5,
)
summary_b = evaluate.GraphEvalResultSummary(
    name="graph_b", num_total=2, num_correct=1, accuracy=0.5,
    subjective_accuracy=None, num_judged=0, num_formatted=2, num_keyword=1,
    num_false_pos=0, num_false_neg=0, num_errors=0, elapsed_s=0.5, n_nodes=4,
    use_icl=True, permutation=None,
    samples=[dummy_sample] * 2,
)
results = {"graph_a": summary_a, "graph_b": summary_b}
log = evaluate._aggregate_multi_graph_eval(results, step=100, epoch=1.0)
check("_aggregate_multi_graph_eval: step/epoch stamped", log["step"] == 100 and log["epoch"] == 1.0)
check("_aggregate_multi_graph_eval: num_samples = 5+2", log["num_samples"] == 7)
# accuracy = (round(0.8*5) + round(0.5*2)) / 7 = (4+1)/7
expected_acc = (4 + 1) / 7
check("_aggregate_multi_graph_eval: accuracy = weighted micro-average",
      abs(log["accuracy"] - expected_acc) < 1e-6, f"{log['accuracy']:.4f} vs {expected_acc:.4f}")
check("_aggregate_multi_graph_eval: per_graph keys present",
      "graph_a" in log["per_graph"] and "graph_b" in log["per_graph"])
check("_aggregate_multi_graph_eval: num_graphs=2", log["num_graphs"] == 2)

# ---------------------------------------------------------------------------
# 6. evaluate.print_summary_table
# ---------------------------------------------------------------------------
try:
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        evaluate.print_summary_table([summary_a, summary_b])
    out = buf.getvalue()
    check("print_summary_table: runs without crash and has totals line",
          "TOTAL" in out, "")
except Exception as e:
    check("print_summary_table: runs without crash", False, str(e))

# ---------------------------------------------------------------------------
# 7. evaluate.GraphTokenAccuracyMixin._accumulate_token_acc
# ---------------------------------------------------------------------------
class _DummyTrainer(evaluate.GraphTokenAccuracyMixin):
    pass

trainer = _DummyTrainer()
trainer._reset_token_acc()

B, S, V = 2, 10, 50
logits = torch.zeros(B, S, V)
input_ids = torch.zeros(B, S, dtype=torch.long)
# token 3 is always predicted correctly by setting it at position 2
for b in range(B):
    input_ids[b, 3] = 7
    logits[b, 2, 7] = 10.0  # preds[2] = 7 = input_ids[3]

outputs = types.SimpleNamespace(logits=logits)
scene_idx = [[3], [3]]
answer_idx = [[3], [3]]
trainer._accumulate_token_acc(outputs, input_ids, scene_idx, answer_idx)
check("GraphTokenAccuracyMixin: scene_n accumulated",
      trainer._gta["scene_n"] == B * 1, f"scene_n={trainer._gta['scene_n']}")
check("GraphTokenAccuracyMixin: scene_c all correct",
      trainer._gta["scene_c"] == B, f"scene_c={trainer._gta['scene_c']}")
check("GraphTokenAccuracyMixin: ans_n accumulated",
      trainer._gta["ans_n"] == B * 1, f"ans_n={trainer._gta['ans_n']}")
check("GraphTokenAccuracyMixin: ans_c all correct",
      trainer._gta["ans_c"] == B, f"ans_c={trainer._gta['ans_c']}")

# Verify that wrong predictions are NOT counted as correct
logits2 = torch.zeros(B, S, V)  # all preds = 0, but input_ids[3] = 7 → wrong
trainer._reset_token_acc()
trainer._accumulate_token_acc(
    types.SimpleNamespace(logits=logits2), input_ids, scene_idx, answer_idx
)
check("GraphTokenAccuracyMixin: wrong preds → scene_c=0",
      trainer._gta["scene_c"] == 0, f"scene_c={trainer._gta['scene_c']}")

# ---------------------------------------------------------------------------
# 8. data.load_and_split_dataset — branch routing via monkeypatch
# ---------------------------------------------------------------------------
import json as _json, tempfile, os

# Write a minimal JSONL file (one record with 'conversations' key)
_conv = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]
_record = {"conversations": _conv}

# Monkeypatch data.preprocess_dataset to identity (avoids tokenizer dep)
import datasets as _datasets

_orig_preprocess = data.preprocess_dataset

def _identity_preprocess(ds, tokenizer, *, architecture, text_edge_list):
    return ds

data.preprocess_dataset = _identity_preprocess

try:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(_json.dumps(_record) + "\n")
        f.write(_json.dumps(_record) + "\n")
        tmp_path = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f2:
        f2.write(_json.dumps(_record) + "\n")
        tmp_val_path = f2.name

    # Branch A: no validation
    cfg_no_val = _mock_config(
        data=types.SimpleNamespace(
            train_files=tmp_path, val_files=None, val_frac=0.0,
            debug=False, dataset_proportion=1.0, text_edge_list="absent"),
        gnn=types.SimpleNamespace(arch="llm"),
    )
    train_ds, eval_ds = data.load_and_split_dataset(cfg_no_val, tokenizer=None)
    check("load_and_split_dataset: no-val branch → eval_ds is None", eval_ds is None)
    check("load_and_split_dataset: no-val branch → all 2 rows in train", len(train_ds) == 2)

    # Branch B: val_frac split
    cfg_frac = _mock_config(
        data=types.SimpleNamespace(
            train_files=tmp_path, val_files=None, val_frac=0.5,
            debug=False, dataset_proportion=1.0, text_edge_list="absent"),
        gnn=types.SimpleNamespace(arch="llm"),
    )
    train_ds2, eval_ds2 = data.load_and_split_dataset(cfg_frac, tokenizer=None)
    check("load_and_split_dataset: val_frac branch → eval_ds is not None", eval_ds2 is not None)
    check("load_and_split_dataset: val_frac branch → train+eval = 2",
          len(train_ds2) + len(eval_ds2) == 2, f"{len(train_ds2)}+{len(eval_ds2)}")

    # Branch C: explicit val_data file
    cfg_val = _mock_config(
        data=types.SimpleNamespace(
            train_files=tmp_path, val_files=tmp_val_path, val_frac=0.0,
            debug=False, dataset_proportion=1.0, text_edge_list="absent"),
        gnn=types.SimpleNamespace(arch="llm"),
    )
    train_ds3, eval_ds3 = data.load_and_split_dataset(cfg_val, tokenizer=None)
    check("load_and_split_dataset: val_data branch → train=2", len(train_ds3) == 2)
    check("load_and_split_dataset: val_data branch → eval_ds=1", len(eval_ds3) == 1)

    # Branch D: debug=True downsamples before split
    cfg_debug = _mock_config(
        data=types.SimpleNamespace(
            train_files=tmp_path, val_files=None, val_frac=0.0,
            debug=True, dataset_proportion=0.5, text_edge_list="absent"),
        gnn=types.SimpleNamespace(arch="llm"),
    )
    train_ds4, eval_ds4 = data.load_and_split_dataset(cfg_debug, tokenizer=None)
    check("load_and_split_dataset: debug branch → downsampled to 1 row",
          len(train_ds4) == 1, f"len={len(train_ds4)}")

finally:
    data.preprocess_dataset = _orig_preprocess
    os.unlink(tmp_path)
    os.unlink(tmp_val_path)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n================  SUMMARY  ================")
print(f"PASSED {len(PASS)} / {len(PASS) + len(FAIL)}")
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
