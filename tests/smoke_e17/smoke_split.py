"""N=30 and N=100 must stay SEPARATE, and the transfer eval must get its own W&B entry."""
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
wandb.init = lambda *a, **k: (stages.append(k.get('name')), _real(*a, **k))[1]

def cw(t):
    return [i for i, c in enumerate(CELLS) if c['cell_type'] == 'code' and t in ''.join(c['source'])][0]
ORDER = [18, 19, 20, 21, 22, 40, 41, 42, 48, 51, cw('def sample_edges'),
         cw('# Train the GNNEdgeDetector'), cw('TRANSFER evaluation for §2')]
SUBS = [("train_samples, val_samples, test_samples = 100, 20, 20",
         "train_samples, val_samples, test_samples = 10, 10, 10"),
        ("num_samples=320", "num_samples=32"),
        ("train_edges = True", "train_edges = False")]   # skip training; load weights
for i in ORDER:
    src = ''.join(CELLS[i]['source'])
    src = '\n'.join(l for l in src.split('\n') if not l.startswith('%') and 'IPython' not in l)
    for a, b in SUBS: src = src.replace(a, b)
    exec(compile(src, f'<c{i}>', 'exec'), ns)
    if i == 42:
        from transformers import AutoTokenizer
        ns['tokenizer'] = AutoTokenizer.from_pretrained(ns['llm_path'])

t30, t100 = ns['test_graphs'], ns['n100_graphs']
print("\n================ SPLIT SEPARATION ================")
print(f"eval_n100 trigger      : {ns['eval_n100']}")
print(f"N=30  test_graphs      : {len(t30)} samples, scene nodes "
      f"{min(int(g.num_scene_nodes) for g in t30)}-{max(int(g.num_scene_nodes) for g in t30)}")
print(f"N=100 n100_graphs      : {len(t100)} samples, scene nodes "
      f"{min(int(g.num_scene_nodes) for g in t100)}-{max(int(g.num_scene_nodes) for g in t100)}")
assert max(int(g.num_scene_nodes) for g in t30) < 60, "an N=100 graph leaked into test_graphs"
assert min(int(g.num_scene_nodes) for g in t100) > 60, "an N=30 graph leaked into n100_graphs"
assert not (set(map(id, t30)) & set(map(id, t100))), "the two sets share objects"
print("no cross-contamination : OK (disjoint objects, disjoint size ranges)")
print(f"W&B entries opened     : {stages}")
assert len(stages) == 1 and 'TRANSFER_n100' in stages[0], f"bad stage: {stages}"
print("transfer has own entry : OK")
print("\nPASS: N=30 and N=100 are separate; transfer logs to its own W&B entry")
