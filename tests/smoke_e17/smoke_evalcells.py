"""train_edges=False must run the §2 eval cell and log it to its OWN W&B stage."""
import os
from pathlib import Path

# Run from notebooks/: the notebook cells these scripts exec resolve their own ../data
# and ../outputs paths from there. Works from any cwd.
_ROOT = Path(__file__).resolve().parents[2]
os.chdir(_ROOT / 'notebooks')
NB = '2026-08-10 e17_magnet_composite_graphs.ipynb'
import json, os, torch
CELLS = json.load(open(NB))['cells']
ns = {'__name__': '__main__', 'display': lambda *a, **k: None}
stages = []
import wandb
_real = wandb.init
def spy(*a, **k):
    stages.append(k.get('name'))
    return _real(*a, **k)
wandb.init = spy

ORDER = [18, 19, 20, 21, 22, 40, 41, 42, 48, 51, 55, 56, 57]
SUBS = [("train_samples, val_samples, test_samples = 100, 20, 20",
         "train_samples, val_samples, test_samples = 10, 10, 10"),
        ("num_samples=320", "num_samples=32"),
        ("train_edges = True", "train_edges = False")]     # <-- the switch under test
for i in ORDER:
    src = ''.join(CELLS[i]['source'])
    src = '\n'.join(l for l in src.split('\n') if not l.startswith('%') and 'IPython' not in l)
    for a, b in SUBS:
        src = src.replace(a, b)
    if i == 42:
        exec(compile(src, f'<c{i}>', 'exec'), ns)
        from transformers import AutoTokenizer
        ns['tokenizer'] = AutoTokenizer.from_pretrained(ns['llm_path'])
        continue
    exec(compile(src, f'<c{i}>', 'exec'), ns)

print("\n================ EVAL-CELL SMOKE ================")
print("train_edges           :", ns['train_edges'])
print("W&B runs opened       :", stages)
assert len(stages) == 1, f"expected exactly ONE run, got {stages}"
assert 'edge_detection_eval' in stages[0], f"eval did not get its own stage: {stages[0]}"
assert 'edge_detection_gt' != stages[0], "eval reused the TRAINING run name"
print("distinct from training:", stages[0], "!= edge_detection_gt")
print("\nPASS: disabled objective ran eval into a separate W&B entry")
