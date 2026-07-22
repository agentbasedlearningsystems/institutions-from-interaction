"""First-settlement iteration per want tier. Tiers from the world config:
entry = single-test want; multi = two tests; hard = three+ tests or
cross-corpus (two data sources). Report row: iteration; ...; want col."""
import sys, os, glob, json, csv

def tiers_from_config(cfg_path):
    c = json.load(open(cfg_path))
    tier = {}
    for e in c.get('blackboard', []):
        for t in e.get('trades', []):
            if t.get('type') != 'buy': continue
            item = t.get('item', '')
            tests = [x for x in t.get('tests', []) if x.get('test')]
            datas = set(x.get('data') for x in tests if x.get('data'))
            if len(tests) >= 3 or len(datas) >= 2: tier[item] = 'hard'
            elif len(tests) == 2: tier[item] = 'multi'
            else: tier[item] = 'entry'
    return tier

def world_stat(wdir, cfg):
    tier = tiers_from_config(cfg)
    first = {}
    rep = os.path.join(wdir, 'reproduction_report.csv')
    if not os.path.exists(rep): return None
    with open(rep, errors='replace') as f:
        for row in csv.reader(f, delimiter=';'):
            if len(row) < 9: continue
            try: it = float(row[0])
            except ValueError: continue
            want = None
            for cell in row[1:]:
                if cell in tier: want = cell; break
            if want and want not in first: first[want] = it
    out = {}
    for tr in ('entry', 'multi', 'hard'):
        vals = sorted(first[w] for w in first if tier.get(w) == tr)
        out[tr] = {'settled': len(vals),
                   'total': sum(1 for w in tier.values() if w == tr),
                   'median_first': round(vals[len(vals)//2], 1) if vals else None}
    out['world'] = os.path.basename(wdir)
    out['ordering_holds'] = bool(
        out['entry']['median_first'] is not None and (
            (out['multi']['median_first'] is None or out['entry']['median_first'] <= out['multi']['median_first'])
            and (out['hard']['median_first'] is None or (out['multi']['median_first'] or 0) <= (out['hard']['median_first'] or 1e18))))
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
        except Exception as e:
            print(json.dumps({'world': base, 'error': str(e)[:80]}))
