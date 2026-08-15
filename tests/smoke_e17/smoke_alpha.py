"""alpha must be (a) stable across draws and (b) the SAME target scale on n_30 and n_100."""
import os
from pathlib import Path

# Run from notebooks/: the notebook cells these scripts exec resolve their own ../data
# and ../outputs paths from there. Works from any cwd.
_ROOT = Path(__file__).resolve().parents[2]
os.chdir(_ROOT / 'notebooks')
NB = '2026-08-10 e17_magnet_composite_graphs.ipynb'
import json, torch
CELLS = json.load(open(NB))['cells']
ns = {'__name__': '__main__', 'display': lambda *a, **k: None}
def cw(t):
    return [i for i, c in enumerate(CELLS) if c['cell_type'] == 'code' and t in ''.join(c['source'])][0]
ORDER = [18, 19, 20, 21, 22, 40, 41, 42, 48, 51, cw('def sample_edges'),
         cw('ALPHA_GRAPHS'), cw('train_res = ')]
SUBS = [("train_samples, val_samples, test_samples = 100, 20, 20",
         "train_samples, val_samples, test_samples = 40, 10, 10"),
        ("num_samples=320", "num_samples=32")]
for i in ORDER:
    src = ''.join(CELLS[i]['source'])
    src = '\n'.join(l for l in src.split('\n') if not l.startswith('%') and 'IPython' not in l)
    for a, b in SUBS: src = src.replace(a, b)
    exec(compile(src, f'<c{i}>', 'exec'), ns)
    if i == 42:
        from transformers import AutoTokenizer
        ns['tokenizer'] = AutoTokenizer.from_pretrained(ns['llm_path'])

tgt, A = ns['resistance_target'], ns['ALPHA']
print("\n================ ALPHA TRANSFER ================")
print(f"ALPHA (mean d2(C) over {ns['ALPHA_GRAPHS']} graphs) = {A:.2f}")

def stats(name, graphs):
    means = torch.stack([tgt(g).mean() for g in graphs]).double()
    rmean = torch.stack([g.R.mean() for g in graphs]).double()
    print(f"  {name:22s} n={len(graphs):3d}  target mean {means.mean():8.2f} "
          f"(sd {means.std():6.3f})   raw R mean {rmean.mean():8.2f} (sd {rmean.std():7.2f})")
    return float(means.mean())

m30 = stats("n_30  test_res", ns['test_res'])
m100 = stats("n_100 n100_res", ns['n100_res'])
print()
print(f"target scale ratio n_100 / n_30 : {m100/m30:.4f}   (1.000 = transfers exactly)")
assert abs(m100/m30 - 1.0) < 0.02, "the target scale still differs between corpora"
print("raw R mean ratio would have been : "
      f"{float(torch.stack([g.R.mean() for g in ns['n100_res']]).mean() / torch.stack([g.R.mean() for g in ns['test_res']]).mean()):.4f}"
      "   <- what the OLD global alpha had to absorb")
print("\nPASS: one alpha now writes the same target scale on both corpora")
