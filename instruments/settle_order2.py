"""First-settlement iteration per want tier, using the established report
schema: r[0]=iter, r[1]=agent index (buyers are a < NB=42), r[5]=score.
Tier per buyer index from the config blackboard: entry = one test,
multi = two tests, hard = 3+ tests or two data sources."""
import sys, os, glob, json, csv

def tiers(cfg_path):
    c = json.load(open(cfg_path))
    out = []
    for e in c.get('blackboard', []):
        for t in e.get('trades', []):
            if t.get('type') != 'buy': continue
            tests = [x for x in t.get('tests', []) if x.get('test')]
            datas = set(x.get('data') for x in tests if x.get('data'))
            if len(tests) >= 3 or len(datas) >= 2: out.append('hard')
            elif len(tests) == 2: out.append('multi')
            else: out.append('entry')
    return out

def world_stat(wdir, cfg_path, NB=42):
    tier = tiers(cfg_path)
    first = {}
    rep = os.path.join(wdir, 'reproduction_report.csv')
    if not os.path.exists(rep): return None
    with open(rep, errors='replace') as f:
        rd = csv.reader(f, delimiter=';'); next(rd, None)
        for r in rd:
            try: t = int(r[0]); a = int(r[1]); sc = float(r[5])
            except (ValueError, IndexError): continue
            if sc <= 0 or a >= NB: continue
            first.setdefault(a, t)
    out = {'world': os.path.basename(wdir)}
    meds = {}
    for tr in ('entry', 'multi', 'hard'):
        idx = [i for i in range(min(NB, len(tier))) if tier[i] == tr]
        vals = sorted(first[i] for i in idx if i in first)
        meds[tr] = vals[len(vals)//2] if vals else None
        out[tr] = {'settled': len(vals), 'total': len(idx), 'median_first': meds[tr]}
    e, m, h = meds['entry'], meds['multi'], meds['hard']
    out['ordering_holds'] = (e is not None and (m is None or e <= m)
                             and (h is None or (m is None or m <= h)) and (h is None or e <= h))
    return out

if __name__ == '__main__':
    root, cfgdir = sys.argv[1], sys.argv[2]
    for d in sorted(glob.glob(os.path.join(root, 'bigtree_*seed*'))):
        base = os.path.basename(d)
        if any(x in base for x in ('_bak', '_arch', '_aborted')): continue
        if not (base.startswith('bigtree_basic_seed') or base.startswith('bigtree_alwayson_seed') or base.startswith('bigtree_solo_seed')): continue
        cfg = os.path.join(cfgdir, base + '.json')
        if not os.path.exists(cfg): continue
        try:
            r = world_stat(d, cfg)
            if r: print(json.dumps(r))
        except Exception as ex:
            print(json.dumps({'world': base, 'error': str(ex)[:80]}))
