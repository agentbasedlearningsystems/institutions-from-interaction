"""Task-family geometry, displayed signs only: per seller, its ONE
displayed sign (latest board in window); sellers grouped by primary
family (representation = vectorSpace/clusterer vs anomalyDetector);
separation = mean within-representation cosine minus
representation-vs-anomaly cosine; permutation over role labels."""
import sys, os, glob, gzip, json
import numpy as np
def load(p):
    op = gzip.open if p.endswith('.gz') else open
    try: return json.load(op(p, 'rt', errors='replace'))
    except Exception: return None
def world_stat(wdir, K=30, reps=2000):
    logs = sorted(glob.glob(os.path.join(wdir, 'logs', '*')), key=os.path.getmtime)[-K:]
    disp = {}; items = {}
    for p in logs:
        b = load(p)
        if not b: continue
        for a in b:
            lab = a.get('label', ''); sg = a.get('sign')
            sells = [str(t.get('item', '')) for t in a.get('trades', []) if t.get('type') == 'sell']
            if sells and isinstance(sg, list) and len(sg) >= 4:
                disp[lab] = sg
                items.setdefault(lab, []).extend(sells)
    rows = []
    from collections import Counter
    for lab, its in items.items():
        prim = Counter(i.split('_')[0] for i in its).most_common(1)[0][0]
        if prim in ('vectorSpace', 'clusterer'): rows.append(('rep', disp[lab]))
        elif prim == 'anomalyDetector': rows.append(('anom', disp[lab]))
    nr = sum(1 for r, _ in rows if r == 'rep'); na = len(rows) - nr
    out = {'world': os.path.basename(wdir), 'rep': nr, 'anom': na}
    if nr < 3 or na < 2:
        out['verdict'] = 'sparse'; return out
    S = np.array([v for _, v in rows], float)
    nrm = np.linalg.norm(S, axis=1); nrm[nrm == 0] = 1
    C = (S / nrm[:, None]) @ (S / nrm[:, None]).T
    roles = np.array([r for r, _ in rows])
    iu = np.triu_indices(len(rows), 1)
    def sep(rl):
        br = (rl[:, None] == 'rep') & (rl[None, :] == 'rep')
        cx = rl[:, None] != rl[None, :]
        w = C[iu][br[iu]]; x = C[iu][cx[iu]]
        return float(w.mean() - x.mean()) if len(w) and len(x) else None
    d0 = sep(roles)
    rng = np.random.RandomState(3); cnt = 0
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
