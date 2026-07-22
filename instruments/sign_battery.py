"""Sign battery per world — DISPLAYED signs only (Debbie's ruling,
July 21): (1) informativeness: MI between a seller's displayed sign
(agent top-level, 8-means quantized within window) and the family of
item it sells; (2) finest-grain: do DIFFERENT sellers of
settings-variants of one algorithm display more similar signs than
different sellers of unrelated same-role algorithms? Same-agent pairs
excluded everywhere (an agent has one displayed sign)."""
import sys, os, glob, gzip, json
import numpy as np

def load(p):
    op = gzip.open if p.endswith('.gz') else open
    try: return json.load(op(p, 'rt', errors='replace'))
    except Exception: return None

def family(item):
    parts = item.split('_'); return '_'.join(parts[:3]) if len(parts) >= 3 else item

def kmeans(X, k, iters=25, seed=0):
    rng = np.random.RandomState(seed)
    C = X[rng.choice(len(X), size=min(k, len(X)), replace=False)]
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        for j in range(len(C)):
            m = lab == j
            if m.any(): C[j] = X[m].mean(0)
    return lab

def world_stat(wdir, K=30, reps=300):
    logs = sorted(glob.glob(os.path.join(wdir, 'logs', '*')), key=os.path.getmtime)[-K:]
    uniq = {}
    for p in logs:
        b = load(p)
        if not b: continue
        for a in b:
            lab = a.get('label', '')
            disp = a.get('sign')
            if not (isinstance(disp, list) and len(disp) >= 4): continue
            for t in a.get('trades', []):
                it = str(t.get('item', ''))
                if t.get('type') == 'sell' and it:
                    uniq[(lab, it)] = (lab, it.split('_')[0], family(it), it, disp)
    rows = list(uniq.values())
    if len(rows) > 1200:
        rng0 = np.random.RandomState(7)
        idx = rng0.choice(len(rows), size=1200, replace=False)
        rows = [rows[i] for i in idx]
    out = {'world': os.path.basename(wdir), 'n_listings': len(rows)}
    if len(rows) < 15:
        out['verdict'] = 'sparse'; return out
    S = np.array([r[4] for r in rows], float)
    nrm = np.linalg.norm(S, axis=1); nrm[nrm == 0] = 1
    C = (S / nrm[:, None]) @ (S / nrm[:, None]).T
    agents = np.array([r[0] for r in rows]); roles = np.array([r[1] for r in rows])
    fams = np.array([r[2] for r in rows]); items = np.array([r[3] for r in rows])
    iu = np.triu_indices(len(rows), 1)
    sr = (roles[:, None] == roles[None, :])[iu]
    da = (agents[:, None] != agents[None, :])[iu]     # different sellers only
    di = (items[:, None] != items[None, :])[iu]
    Ciu = C[iu]
    def delta(f):
        sf = (f[:, None] == f[None, :])[iu]
        w = Ciu[sr & da & sf & di]; b = Ciu[sr & da & ~sf]
        if len(w) < 10 or len(b) < 10: return None, len(w), len(b)
        return float(w.mean() - b.mean()), len(w), len(b)
    d0, nw, nb = delta(fams)
    out['n_within'], out['n_between'] = nw, nb
    if d0 is not None:
        rng = np.random.RandomState(1); cnt = valid = 0
        for _ in range(reps):
            fp = fams.copy()
            for r in np.unique(roles):
                m = roles == r
                fp[m] = fp[m][rng.permutation(m.sum())]
            dp, _, _ = delta(fp)
            if dp is None: continue
            valid += 1
            if dp >= d0: cnt += 1
        out['delta'] = round(d0, 4)
        out['perm_p'] = round((cnt + 1) / (valid + 1), 4) if valid else None
    else:
        out['verdict'] = 'sparse-pairs'
    lab8 = kmeans(S / nrm[:, None], 8)
    fam_ids = {f: i for i, f in enumerate(sorted(set(fams)))}
    fv = np.array([fam_ids[f] for f in fams])
    joint = np.zeros((8, len(fam_ids)))
    for l, f in zip(lab8, fv): joint[l, f] += 1
    joint /= joint.sum()
    px = joint.sum(1, keepdims=True); py = joint.sum(0, keepdims=True)
    nz = joint > 0
    out['displayed_mi_bits'] = round(float((joint[nz] * np.log2(joint[nz] / (px @ py)[nz])).sum()), 3)
    return out

if __name__ == '__main__':
    for d in sorted(glob.glob(os.path.join(sys.argv[1], 'bigtree_basic_seed*')) + glob.glob(os.path.join(sys.argv[1], 'bigtree_alwayson_seed*'))):
        base = os.path.basename(d)
        if any(x in base for x in ('_bak', '_arch', '_aborted')): continue
        try: print(json.dumps(world_stat(d)))
        except Exception as e: print(json.dumps({'world': base, 'error': str(e)[:60]}))
