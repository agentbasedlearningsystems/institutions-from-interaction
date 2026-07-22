"""Software quality per world: per-want running-best score from the
report. Emits mean best-per-want (unfilled = 0) at own age and at a
given common age N, split entry/upper, plus age and want coverage."""
import sys, os, glob, json, csv

def tiers(cfg_path):
    c = json.load(open(cfg_path)); out = []
    for e in c.get('blackboard', []):
        for t in e.get('trades', []):
            if t.get('type') != 'buy': continue
            tests = [x for x in t.get('tests', []) if x.get('test')]
            datas = set(x.get('data') for x in tests if x.get('data'))
            out.append('upper' if (len(tests) >= 2 or len(datas) >= 2) else 'entry')
    return out

def world_stat(wdir, cfg, N):
    tr = tiers(cfg)
    best_all = {}; best_N = {}
    age = 0
    rep = os.path.join(wdir, 'reproduction_report.csv')
    if not os.path.exists(rep): return None
    for r in csv.reader(open(rep, errors='replace'), delimiter=';'):
        try: t = int(r[0]); a = int(r[1]); sc = float(r[5])
        except (ValueError, IndexError): continue
        age = max(age, t)
        if a >= len(tr) or sc <= 0: continue
        best_all[a] = max(best_all.get(a, 0.0), sc)
        if t <= N: best_N[a] = max(best_N.get(a, 0.0), sc)
    nw = len(tr)
    def mean_over(d, idxs): return sum(d.get(i, 0.0) for i in idxs) / max(len(idxs), 1)
    entry = [i for i in range(nw) if tr[i] == 'entry']; upper = [i for i in range(nw) if tr[i] == 'upper']
    return {'world': os.path.basename(wdir), 'age': age,
            'meanbest_own': round(mean_over(best_all, range(nw)), 4),
            'filled_own': sum(1 for i in range(nw) if best_all.get(i, 0) > 0),
            'meanbest_N': round(mean_over(best_N, range(nw)), 4),
            'entry_N': round(mean_over(best_N, entry), 4),
            'upper_N': round(mean_over(best_N, upper), 4),
            'filled_N': sum(1 for i in range(nw) if best_N.get(i, 0) > 0)}

if __name__ == '__main__':
    root, cfgdir, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
    for d in sorted(glob.glob(os.path.join(root, 'bigtree_*seed*'))):
        base = os.path.basename(d)
        if any(x in base for x in ('_bak', '_arch', '_aborted')): continue
        if not (base.startswith('bigtree_basic_seed') or base.startswith('bigtree_alwayson_seed') or base.startswith('bigtree_solo_seed')): continue
        cfg = os.path.join(cfgdir, base + '.json')
        if not os.path.exists(cfg): continue
        try:
            r = world_stat(d, cfg, N)
            if r: print(json.dumps(r))
        except Exception as e:
            print(json.dumps({'world': base, 'error': str(e)[:60]}))
