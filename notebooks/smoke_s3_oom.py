"""§3 ONLY, at the scale that OOM'd: does expandable_segments carry it through?

Loads §2 weights from suite3_smoke (train_edges=False), then TRAINS the resistance +
edge dual objective for one epoch over the real 100-graph training set at M=320 — the
exact path that died in train_loop_resistance with 8 GiB stranded. No LLM is loaded.
"""
import json, os, torch
NB = os.environ.get('AUDIT_NB', '2026-08-10 e17_magnet_composite_graphs.ipynb')
CELLS = json.load(open(NB))['cells']
ns = {'__name__': '__main__', 'display': lambda *a, **k: None}
def cw(t):
    return [i for i, c in enumerate(CELLS) if c['cell_type'] == 'code' and t in ''.join(c['source'])][0]

SUBS = [("load_suite     = 'suite2'", "load_suite     = 'suite3_smoke'"),
        ("train_edges = True", "train_edges = False"),      # load §2, don't retrain
        ("eval_n100 = True", "eval_n100 = False"),          # §3 only
        ("epochs = 50", "epochs = 1")]                      # one pass over 100 graphs
ORDER = [18, 19, 20, 21, 22, 40, 41, 42, 48, 51, cw('def sample_edges'),
         cw('# Train the GNNEdgeDetector'), cw('def gram_distances'),
         cw('train_res = '), cw('# Train the Graph Transformer')]
for i in ORDER:
    src = ''.join(CELLS[i]['source'])
    src = '\n'.join(l for l in src.split('\n') if not l.startswith('%') and 'IPython' not in l)
    for a, b in SUBS: src = src.replace(a, b)
    print(f"########## cell {i}", flush=True)
    exec(compile(src, f'<c{i}>', 'exec'), ns)
    if i == 42:
        from transformers import AutoTokenizer
        ns['tokenizer'] = AutoTokenizer.from_pretrained(ns['llm_path'])

print("\n================ §3 OOM SMOKE ================")
print(f"loaded §2 weights from : {ns['load_path_gt']}")
print(f"multiplier             : {ns['multiplier']}")
print(f"train/val/test graphs  : {len(ns['train_res'])}/{len(ns['val_res'])}/{len(ns['test_res'])}")
print(f"M                      : {ns['model_hparams']['num_samples']}")
print(f"peak CUDA allocated    : {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
print(f"peak CUDA reserved     : {torch.cuda.max_memory_reserved()/2**30:.2f} GiB")
frag = (torch.cuda.max_memory_reserved() - torch.cuda.max_memory_allocated()) / 2**30
print(f"reserved-but-unalloc   : {frag:.2f} GiB   (the OOM run stranded 7.99)")
print("\nPASS: §3 dual-objective training completed one epoch without OOM")
