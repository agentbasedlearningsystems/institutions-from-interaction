"""Debbie's check: within each society, do high-alpha buyers tolerate
higher midpoints? Spearman(alpha, midpoint) per world, latest board."""
import sys, os, glob, gzip, json
def load(p):
    op = gzip.open if p.endswith('.gz') else open
    try: return json.load(op(p, 'rt', errors='replace'))
    except Exception: return None
def rank(v):
    s = sorted(range(len(v)), key=lambda i: v[i]); r = [0]*len(v)
    for j, i in enumerate(s): r[i] = j
    return r
def spear(a, b):
    ra, rb = rank(a), rank(b); n = len(a)
    ma, mb = sum(ra)/n, sum(rb)/n
    num = sum((x-ma)*(y-mb) for x, y in zip(ra, rb))
    den = (sum((x-ma)**2 for x in ra) * sum((y-mb)**2 for y in rb)) ** 0.5
    return num/den if den else 0.0
for d in sorted(glob.glob(os.path.join(sys.argv[1], 'bigtree_basic_seed*')) + glob.glob(os.path.join(sys.argv[1], 'bigtree_alwayson_seed*'))):
    base = os.path.basename(d)
    if any(x in base for x in ('_bak', '_arch', '_aborted')): continue
    logs = sorted(glob.glob(os.path.join(d, 'logs', '*')), key=os.path.getmtime)
    if not logs: continue
    b = load(logs[-1])
    if not b: continue
    al, mp = [], []
    for a in b:
        for t in a.get('trades', []):
            if t.get('type') == 'buy' and t.get('alpha') is not None and t.get('midpoint') is not None:
                al.append(t['alpha']); mp.append(t['midpoint'])
    if len(al) >= 20:
        print(json.dumps({'world': base, 'n': len(al), 'spearman_alpha_midpoint': round(spear(al, mp), 3)}))
