"""End-window employment: learner-to-learner share of payments.
Roles from the latest board: an agent index whose trades include an
alpha-carrying buy is a human buyer; an agent with sell trades and no
alpha-buy is a learner. End window = last quarter of ledger time."""
import sys, os, glob, gzip, json, csv

def load(p):
    op = gzip.open if p.endswith('.gz') else open
    try: return json.load(op(p, 'rt', errors='replace'))
    except Exception: return None

def world_stat(wdir):
    logs = sorted(glob.glob(os.path.join(wdir, 'logs', '*')), key=os.path.getmtime)
    if not logs: return None
    b = load(logs[-1])
    if not b: return None
    buyers, learners = set(), set()
    for i, a in enumerate(b):
        trades = a.get('trades', [])
        if any(t.get('type') == 'buy' and t.get('alpha') is not None for t in trades):
            buyers.add(i)
        elif any(t.get('type') == 'sell' for t in trades):
            learners.add(i)
    pay = os.path.join(wdir, 'payments.csv')
    if not os.path.exists(pay): return None
    rows = []
    with open(pay, errors='replace') as f:
        r = csv.reader(f, delimiter=';')
        header = next(r, None)
        for row in r:
            try: rows.append((float(row[0]), int(row[1]), int(row[2]), float(row[3])))
            except (ValueError, IndexError): continue
    if len(rows) < 40: return {'world': os.path.basename(wdir), 'verdict': 'few payments', 'n': len(rows)}
    t_end = max(t for t, *_ in rows); t_cut = t_end * 0.75
    endw = [x for x in rows if x[0] >= t_cut]
    def shares(rs):
        ll_n = ll_v = tot_v = 0
        for t, buy, sell, price in rs:
            tot_v += price
            if buy in learners and sell in learners:
                ll_n += 1; ll_v += price
        return (ll_n / len(rs) if rs else 0.0, ll_v / tot_v if tot_v else 0.0, len(rs))
    ln, lv, n = shares(endw); fln, flv, fn = shares(rows)
    return {'world': os.path.basename(wdir), 'buyers': len(buyers), 'learners': len(learners),
            'endwin_ll_count_share': round(ln, 4), 'endwin_ll_value_share': round(lv, 4),
            'endwin_n': n, 'lifetime_ll_count_share': round(fln, 4),
            'lifetime_ll_value_share': round(flv, 4), 'lifetime_n': fn}

if __name__ == '__main__':
    import glob as g
    for d in sorted(g.glob(os.path.join(sys.argv[1], 'bigtree_*seed*'))):
        base = os.path.basename(d)
        if any(x in base for x in ('_bak', '_arch', '_aborted')): continue
        if not (base.startswith('bigtree_basic_seed') or base.startswith('bigtree_alwayson_seed') or base.startswith('bigtree_solo_seed')): continue
        r = world_stat(d)
        if r: print(json.dumps(r))
