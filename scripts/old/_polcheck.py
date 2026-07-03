import json, glob, re, sys
sys.path.insert(0, 'scripts')
from smoke_test_graph_solvable import build_graph, task_type

def answer_polarity(a):
    """Classify a non-path answer regex by its signature cue. Robust across the
    bare ``(?i)(yes|...)`` dialect and the canonical ``(?i)(?:\\byes\\b|...)``
    templates: affirm answers carry a ``yes`` cue, deny answers a ``cannot``/
    ``unreachable`` cue and never ``yes``."""
    if r"\byes\b" in a or "(yes" in a or "|yes" in a:
        return "affirm"
    if r"\bcannot\b" in a or r"\bunreachable\b" in a or "cannot" in a:
        return "deny"
    return "affirm"

disagree = 0
counts = {}
for f in glob.glob('data/gen/nav100_n30_gemma_data/split/*/data_gen_*.json'):
    d = json.load(open(f))
    G, nodes, _ = build_graph(d['graph'])
    for i, t in enumerate(d['tasks']):
        a = (t.get('answer', '') or '').strip()
        crit = t.get('acceptance_criterion', '')
        tt = task_type(a, crit, nodes)
        counts[tt] = counts.get(tt, 0) + 1
        if tt == 'path':
            continue
        ans_pol = answer_polarity(a)
        crit_pol = 'deny' if tt == 'deny' else 'affirm'
        if ans_pol != crit_pol:
            disagree += 1
            if disagree <= 12:
                print('DISAGREE', f.split('/')[-1], i, '| type=', tt, '| ans=', repr(a), '| crit=', crit[:90])
print('type counts:', counts)
print('total polarity disagreements:', disagree)
