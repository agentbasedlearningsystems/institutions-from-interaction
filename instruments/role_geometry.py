"""Task-family sign geometry per world: per-seller mean sign over sell
listings (last K boards); separation = mean cosine within
representation-role sellers (vectorSpace/clusterer primaries) minus mean
cosine between representation and anomaly-role sellers; permutation over
role labels."""
import sys, os, glob, gzip, json
import numpy as np

def load(p):
    op = gzip.open if p.endswith('.gz') else open
    try: return json.load(op(p, 'rt', errors='replace'))
    except Exception: return None

def world_stat(wdir, K=30, reps=2000):
    logs = sorted(glob.glob(os.path.join(wdir, 'logs', '*')), key=os.path.getmtime)[-K:]
    sellers = {}
    for p in logs:
        b = load(p)
        if not b: continue
        for a in b:
            lab = a.get('label', '')
            for t in a.get('trades', []):
                it = str(t.get('item', '')); sg = t.get('sign')
                if t.get('type') == 'sell' and isinstance(sg, list) and len(sg) >= 4:
                    d = sellers.setdefault(lab, {'signs': [], 'items': []})
                    d['signs'].append(sg); d['items'].append(it)
    rows = []
    for lab, d in sellers.items():
        from collections import Counter
        prim = Counter(i.split('_')[0] for i in d['items']).most_common(1)[0][0]
        if prim in ('vectorSpace', 'clusterer'): role = 'rep'
        elif prim == 'anomalyDetector': role = 'anom'
        else: continue
        rows.append((role, np.mean(np.array(d['signs'], float), axis=0)))
    reps_n = sum(1 for r, _ in rows if r == 'rep'); an_n = len(rows) - reps_n
    out = {'world': os.path.basename(wdir), 'rep_sellers': reps_n, 'anom_sellers': an_n}
    if reps_n < 3 or an_n < 2:
        out['verdict'] = 'sparse'; return out
    S = np.array([v for _, v in rows]); roles = np.array([r for r, _ in rows])
    nrm = np.linalg.norm(S, axis=1); nrm[nrm == 0] = 1
    C = (S / nrm[:, None]) @ (S / nrm[:, None]).T
    iu = np.triu_indices(len(rows), 1)
    def sep(rl):
        both_rep = (rl[:, None] == 'rep') & (rl[None, :] == 'rep')
        cross = (rl[:, None] != rl[None, :])
        w = C[iu][both_rep[iu]]; x = C[iu][cross[iu]]
        return float(w.mean() - x.mean()) if len(w) and len(x) else None
    d0 = sep(roles)
    rng = np.random.RandomState(3)
    cnt = 0
    for _ in range(reps):
        rp = roles.copy(); rng.shuffle(rp)
        dp = sep(rp)
        if dp is not None and dp >= d0: cnt += 1
    out['separation'] = round(d0, 4); out['perm_p'] = round((cnt + 1) / (reps + 1), 4)
    return out

if __name__ == '__main__':
    for d in sorted(glob.glob(os.path.join(sys.argv[1], 'bigtree_basic_seed*')) + glob.glob(os.path.join(sys.argv[1], 'bigtree_alwayson_seed*'))):
        base = os.path.basename(d)
        if any(x in base for x in ('_bak', '_arch', '_aborted')): continue
        try: print(json.dumps(world_stat(d)))
        except Exception as e: print(json.dumps({'world': base, 'error': str(e)[:60]}))
