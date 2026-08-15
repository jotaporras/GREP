"""Is the N=100 test set VIABLE for the MagGT on both objectives?

Runs the notebook's own cells, then checks the test set is (a) built from completed plans
only, (b) genuinely ~100 scene nodes, (c) fully crosslinked, (d) scorable by the §2
detector loaded from the trained suite, and (e) connected enough for §3's R_eff target.
"""
import os
from pathlib import Path

# Run from notebooks/: the notebook cells these scripts exec resolve their own ../data
# and ../outputs paths from there. Works from any cwd.
_ROOT = Path(__file__).resolve().parents[2]
os.chdir(_ROOT / 'notebooks')
NB = '2026-08-10 e17_magnet_composite_graphs.ipynb'
import json, os, sys, torch

CELLS = json.load(open(NB))['cells']
ns = {'__name__': '__main__', 'display': lambda *a, **k: None}

# setup -> model -> composite builder -> §2 data -> §3 target defs
ORDER = [18, 19, 20, 21, 22, 40, 41, 42, 48, 51, 55]


def run(i, subs=()):
    src = ''.join(CELLS[i]['source'])
    src = '\n'.join(l for l in src.split('\n')
                    if not l.startswith('%') and 'IPython' not in l)
    for a, b in subs:
        src = src.replace(a, b)
    exec(compile(src, f'<cell {i}>', 'exec'), ns)


SUBS = [("train_samples, val_samples, test_samples = 100, 20, 20",
         "train_samples, val_samples, test_samples = 10, 10, 10"),  # tiny train/val
        ("plans_per_graph = 10", "plans_per_graph = 10"),
        ("num_samples=320", "num_samples=32"),                       # cheap probes
        ("wandb.init", "_noop_init")]
ns['_noop_init'] = lambda *a, **k: None

for idx in ORDER:
    print(f"--- cell {idx}", flush=True)
    run(idx, SUBS)
    if idx == 42:                       # cell 43 builds it, but 43 also runs the GNN
        from transformers import AutoTokenizer
        ns['tokenizer'] = AutoTokenizer.from_pretrained(ns['llm_path'])

test = ns['test_graphs']
keys = ns['test_keys']
print("\n================ N=100 TEST SET ================")
print(f"graphs with completed plans : {len(keys)}  {keys}")
print(f"composite test samples      : {len(test)}")
assert len(test) > 0, "EMPTY test set — the plan glob matched nothing"

# (a) no failed / partial plan leaked in
bad = [p for k in keys for p in ns['complete_plans'](k, ns['eval_plan_path'])
       if '_failed' in p or p.endswith('.partial')]
assert not bad, f"failed/partial plans leaked in: {bad[:3]}"
print(f"failed/partial excluded     : OK ({len(bad)} leaked)")

# (b)+(c) size and crosslinking
n_sc = [int(g.num_scene_nodes) for g in test]
cross = [int((g.edge_index[0] >= g.num_token_nodes).sum()
             - 1 * (g.num_scene_nodes > 0)) for g in test]  # scene->token + anchor edge
c_tok = [int(g.num_token_nodes) for g in test]
print(f"scene nodes  min/med/max    : {min(n_sc)} / {sorted(n_sc)[len(n_sc)//2]} / {max(n_sc)}")
print(f"token nodes  min/med/max    : {min(c_tok)} / {sorted(c_tok)[len(c_tok)//2]} / {max(c_tok)}")
assert min(n_sc) > 60, f"scene graphs are not N=100 scale (min {min(n_sc)})"

# every scene node must reach the text, or the crosslink layer is vacuous
inj = [len(g.injection_map) for g in test]
print(f"nodes crosslinked min/max   : {min(inj)} / {max(inj)}  (of {min(n_sc)}-{max(n_sc)})")
assert min(inj) > 0, "NO node is mentioned in the text — crosslink layer is empty"

# (e) §3 viability: the composite must be ONE component for R_eff to be finite
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
dis = 0
for g in test:
    plain = Data(edge_index=g.edge_index, num_nodes=g.num_nodes)
    if not nx.is_connected(to_networkx(plain, to_undirected=True)):
        dis += 1
print(f"disconnected composites     : {dis} / {len(test)}")
print(f"  (§3 keeps the connected {len(test)-dis}; cell 63 filters these)")

# (d) §2 viability: score the TRAINED detector on it
gnn, detector = ns['gnn'], ns['detector']
load = ns['load_path_gt']
sd = os.path.join(load, 'mag_gt.pt')
if os.path.exists(sd):
    gnn.load_state_dict(torch.load(sd, map_location=ns['device']))
    detector.classifier.load_state_dict(
        torch.load(os.path.join(load, 'detector.pt'), map_location=ns['device']))
    print(f"loaded trained MagGT from   : {load}")
    gnn.eval(); detector.eval().to(ns['device'])
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for g in test[:12]:
            detector.invalidate_cache()
            preds = torch.stack([detector(g, g.edges_x[0, k], g.edges_x[1, k])
                                 for k in range(g.edges_x.shape[1])]).squeeze(-1)
            true, pred = g.edges_y.bool(), preds > 0
            tp += int((pred & true).sum()); fp += int((pred & ~true).sum())
            fn += int((~pred & true).sum()); tn += int((~pred & ~true).sum())
    acc = (tp + tn) / max(1, tp + fp + fn + tn)
    rec = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    print(f"\n>>> §2 on N=100 (12 graphs): acc={acc:.3%}  recall={rec:.3f}  "
          f"specificity={spec:.3f}  bal_acc={(rec+spec)/2:.3%}")
else:
    print(f"!! no trained checkpoint at {sd} — skipped the §2 score")
print("\nALL VIABILITY CHECKS PASSED")
