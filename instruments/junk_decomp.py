"""Junk decomposition per world: junk share (score<=0.05) of settles and
of score-mass, split by buyer tier (entry vs multi/hard)."""
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
def world_stat(wdir, cfg):
    tr = tiers(cfg)
    st = {'entry': [0, 0, 0.0], 'upper': [0, 0, 0.0]}
    rep = os.path.join(wdir, 'reproduction_report.csv')
    if not os.path.exists(rep): return None
    for r in csv.reader(open(rep, errors='replace'), delimiter=';'):
        try: a = int(r[1]); sc = float(r[5])
        except (ValueError, IndexError): continue
        if sc <= 0 or a >= len(tr): continue
        k = tr[a]
        st[k][0] += 1; st[k][2] += sc
        if sc <= 0.05: st[k][1] += 1
    out = {'world': os.path.basename(wdir)}
    tot = st['entry'][0] + st['upper'][0]
    junk = st['entry'][1] + st['upper'][1]
    mass = st['entry'][2] + st['upper'][2]
    if not tot: return None
    out['settles'] = tot
    out['junk_share'] = round(junk / tot, 3)
    out['entry_junk_share'] = round(st['entry'][1] / st['entry'][0], 3) if st['entry'][0] else None
    out['upper_junk_share'] = round(st['upper'][1] / st['upper'][0], 3) if st['upper'][0] else None
    junk_mass = 0.05 * junk
    out['junk_scoremass_share_max'] = round(junk_mass / mass, 4) if mass else None
    return out
if __name__ == '__main__':
    root, cfgdir = sys.argv[1], sys.argv[2]
    for d in sorted(glob.glob(os.path.join(root, 'bigtree_*seed*'))):
        base = os.path.basename(d)
        if any(x in base for x in ('_bak', '_arch', '_aborted')): continue
        if not (base.startswith('bigtree_basic_seed') or base.startswith('bigtree_alwayson_seed')): continue
        cfg = os.path.join(cfgdir, base + '.json')
        if not os.path.exists(cfg): continue
        try:
            r = world_stat(d, cfg)
            if r: print(json.dumps(r))
        except Exception as e:
            print(json.dumps({'world': base, 'error': str(e)[:60]}))
