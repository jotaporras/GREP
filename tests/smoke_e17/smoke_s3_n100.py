"""§3 must build R_eff targets on the N=100 set without the connectivity assert firing."""
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
def cell_with(txt):
    return [i for i, c in enumerate(CELLS)
            if c['cell_type'] == 'code' and txt in ''.join(c['source'])][0]
ORDER = [18, 19, 20, 21, 22, 40, 41, 42, 48, 51,
         cell_with('def sample_edges'), cell_with('conv = gnn.pe_model'),
         cell_with('train_res = ')]
SUBS = [("train_samples, val_samples, test_samples = 100, 20, 20",
         "train_samples, val_samples, test_samples = 10, 10, 10"),
        ("num_samples=320", "num_samples=32")]
for i in ORDER:
    src = ''.join(CELLS[i]['source'])
    src = '\n'.join(l for l in src.split('\n') if not l.startswith('%') and 'IPython' not in l)
    for a, b in SUBS: src = src.replace(a, b)
    exec(compile(src, f'<c{i}>', 'exec'), ns)
    if i == 42:
        from transformers import AutoTokenizer
        ns['tokenizer'] = AutoTokenizer.from_pretrained(ns['llm_path'])

tr = ns['test_res']
print("\n================ §3 ON N=100 ================")
print(f"test targets built : {len(tr)} of {len(ns['_test_all'])}")
assert len(tr) > 0
R = [g.R for g in tr]
import math
finite = all(torch.isfinite(r).all() for r in R)
print(f"all R_eff finite   : {finite}")
assert finite, "an infinite R_eff survived the filter"
print(f"R shape range      : {min(r.shape[0] for r in R)}-{max(r.shape[0] for r in R)} (c x c)")
print(f"R mean / max       : {sum(float(r.mean()) for r in R)/len(R):.2f} / {max(float(r.max()) for r in R):.2f}")
sym = max(float((r - r.T).abs().max()) for r in R)
diag = max(float(r.diagonal().abs().max()) for r in R)
print(f"symmetry / diag    : {sym:.2e} / {diag:.2e}  (both must be ~0)")
assert sym < 1e-4 and diag < 1e-4
held = sum(r.numel() for r in R) * 4 / 2**30
print(f"target memory      : {held:.2f} GiB")
print("\nPASS: §3 has finite, symmetric, zero-diagonal R_eff targets on N=100")
