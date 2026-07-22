"""Finest-grain sign question: do two settings-variants of the same
algorithm carry more similar signs than two unrelated algorithms of the
same role? Per world: over last K boards, same-board sell-trade pairs,
cosine sim within-family (same brand+routine, different settings) vs
between-family (same role, different routine). Permutation test shuffles
family labels within board+role. Output one JSON row per world."""
import sys, os, glob, gzip, json, random, math

def load(p):
    op = gzip.open if p.endswith('.gz') else open
    try: return json.load(op(p, 'rt', errors='replace'))
    except Exception: return None

def cos(a, b):
    num = sum(x*y for x, y in zip(a, b))
    da = math.sqrt(sum(x*x for x in a)); db = math.sqrt(sum(x*x for x in b))
    return num/(da*db) if da > 0 and db > 0 else None

def family(item):
    parts = item.split('_')
    return '_'.join(parts[:3]) if len(parts) >= 3 else item  # role_brand_routine

def role(item):
    return item.split('_')[0]

def world_stat(wdir, K=30, reps=1000, rng=None):
    rng = rng or random.Random(0)
    logs = sorted(glob.glob(os.path.join(wdir, 'logs', '*')), key=os.path.getmtime)[-K:]
    uniq = {}
    for p in logs:
        b = load(p)
        if not b: continue
        for a in b:
            lab = a.get('label', '')
            for t in a.get('trades', []):
                it = str(t.get('item', ''))
                sg = t.get('sign')
                if t.get('type') == 'sell' and isinstance(sg, list) and len(sg) >= 4 and it:
                    uniq[(lab, it)] = (role(it), family(it), it, sg)
    listings = list(uniq.values())
    def contrast(rows):
        w, b = [], []
        for i in range(len(rows)):
            for j in range(i+1, len(rows)):
                r1, f1, i1, s1 = rows[i]; r2, f2, i2, s2 = rows[j]
                if r1 != r2: continue
                c = cos(s1, s2)
                if c is None: continue
                if f1 == f2 and i1 != i2: w.append(c)
                elif f1 != f2: b.append(c)
        return w, b
    within, between = contrast(listings)
    if len(within) < 10 or len(between) < 10:
        return {'world': os.path.basename(wdir), 'n_listings': len(listings),
                'n_within': len(within), 'n_between': len(between), 'verdict': 'sparse'}
    dw = sum(within)/len(within); db_ = sum(between)/len(between)
    delta = dw - db_
    count = 0
    byrole = {}
    for r, f, it, sg in listings: byrole.setdefault(r, []).append([f, it, sg])
    for _ in range(reps):
        shuffled = []
        for r, rows in byrole.items():
            fams = [x[0] for x in rows]; rng.shuffle(fams)
            shuffled += [(r, fp, x[1], x[2]) for x, fp in zip(rows, fams)]
        w2, b2 = contrast(shuffled)
        if len(w2) >= 10 and len(b2) >= 10 and (sum(w2)/len(w2) - sum(b2)/len(b2)) >= delta:
            count += 1
    p = (count + 1) / (reps + 1)
    return {'world': os.path.basename(wdir), 'n_listings': len(listings),
            'n_within': len(within), 'n_between': len(between),
            'mean_within': round(dw, 4), 'mean_between': round(db_, 4),
            'delta': round(delta, 4), 'perm_p': round(p, 4)}

if __name__ == '__main__':
    root = sys.argv[1]
    for d in sorted(glob.glob(os.path.join(root, 'bigtree_basic_seed*')) +
                    glob.glob(os.path.join(root, 'bigtree_alwayson_seed*'))):
        base = os.path.basename(d)
        if not base.replace('bigtree_basic_seed','').replace('bigtree_alwayson_seed','').isdigit():
            continue
        print(json.dumps(world_stat(d, rng=random.Random(base))))
