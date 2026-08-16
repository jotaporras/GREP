"""Is the suite2 MagE-GT agnostic to the cycle length max_seq_length would change?

Builds the SAME conversation at growing context_window caps and asks whether the encoding's
scale and the charge margin hold as c grows from the trained range into the 8192 range.
"""
import os
from pathlib import Path

# Run from notebooks/: the notebook cells these scripts exec resolve their own ../data
# and ../outputs paths from there. Works from any cwd.
_ROOT = Path(__file__).resolve().parents[2]
os.chdir(_ROOT / 'notebooks')
NB = '2026-08-10 e17_magnet_composite_graphs.ipynb'
import json, glob, torch
CELLS = json.load(open(NB))['cells']
ns = {'__name__': '__main__', 'display': lambda *a, **k: None}
def cw(t):
    return [i for i, c in enumerate(CELLS) if c['cell_type'] == 'code' and t in ''.join(c['source'])][0]
for i in [18, 19, 20, 21, 22, 40, 41, 42]:
    src = ''.join(CELLS[i]['source'])
    src = '\n'.join(l for l in src.split('\n') if not l.startswith('%') and 'IPython' not in l)
    src = src.replace("num_samples=320", "num_samples=32")
    exec(compile(src, f'<c{i}>', 'exec'), ns)
    if i == 42:
        from transformers import AutoTokenizer
        ns['tokenizer'] = AutoTokenizer.from_pretrained(ns['llm_path'])
# defs only: cut the cell at the calibration block, which needs data we don't build here
src = ''.join(CELLS[cw('def gram_distances')]['source'])
src = src[:src.index('# C_tok is the encoding this stage shapes')]
exec(compile(src, '<defs>', 'exec'), ns)

import os
gnn = ns['gnn']
sd = os.path.join(ns['load_path_gt'], 'mag_gt.pt')
gnn.load_state_dict(torch.load(sd, map_location=ns['device']))
gnn.eval()
print(f"loaded {sd}")

key = 'data_gen_' + sorted(os.path.basename(f) for f in
                           glob.glob(ns['ex_path'] + '/data_gen_*.json'))[0].split('_')[-1][:-5]
plans = sorted(glob.glob(f"{ns['plan_path']}/sample_{key.split('_')[-1]}_*.json"))
print(f"graph {key}, {len(plans)} plans available\n")
print(f"{'cap':>6} {'c':>6} {'delta(r,c)':>11} {'mean d2(C)':>12} {'trace C':>12}")
r = float(ns['conv'].r) if 'conv' in ns else 0.126
for cap, npl in ((1024, 2), (2048, 4), (4096, 8), (8192, 10)):
    g = ns['build_composite_graph'](ns['graph_file_by_name'][key], ns['tokenizer'],
                                    plan_files=plans[:npl], context_window=cap,
                                    device=ns['device'])
    c = g.num_token_nodes
    with torch.no_grad():
        C = ns['probe_covariance'](gnn, g).double()
    s = 2 * r * c
    print(f"{cap:>6} {c:>6} {min(s % 1, 1 - s % 1):>11.4f} "
          f"{float(ns['gram_distances'](C).mean()):>12.2f} {float(C.trace()):>12.1f}")
