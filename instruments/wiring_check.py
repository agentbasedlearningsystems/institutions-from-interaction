"""Wiring check: for each live world, verify from LOGS (not liveness):
fresh board (<15 min), utility buyers wired (alpha in live board),
prices moving (midpoint variance across boards), settles flowing
(report rows), param-walk draws (settings tokens in report), and for
solos: exactly one learner selling."""
import sys, os, glob, gzip, json, time, csv

def load(p):
    op = gzip.open if p.endswith('.gz') else open
    try: return json.load(op(p, 'rt', errors='replace'))
    except Exception: return None

def check(wdir):
    w = os.path.basename(wdir)
    logs = sorted(glob.glob(os.path.join(wdir, 'logs', '*')), key=os.path.getmtime)
    out = {'world': w, 'boards': len(logs)}
    if not logs:
        out['FLAG'] = 'no boards'; return out
    age = time.time() - os.path.getmtime(logs[-1])
    out['fresh_min'] = round(age / 60, 1)
    if age > 900: out['FLAG'] = 'stale board'
    bb = load(logs[-1])
    if bb is None:
        out['FLAG'] = 'unreadable board'; return out
    alphas = mids = 0
    for a in bb:
        for t in a.get('trades', []):
            if t.get('type') == 'buy' and t.get('alpha') is not None:
                alphas += 1
                if t.get('midpoint') is not None: mids += 1
    out['utility_buyers'] = alphas
    if len(logs) >= 2:
        b0 = load(logs[0])
        m0 = {}; m1 = {}
        for src, d in ((b0, m0), (bb, m1)):
            for a in src or []:
                for t in a.get('trades', []):
                    if t.get('type') == 'buy' and t.get('alpha') is not None:
                        d[a.get('label','')] = t.get('midpoint')
        moved = sum(1 for k in m1 if k in m0 and m0[k] is not None
                    and m1[k] is not None and abs(m0[k]-m1[k]) > 0.5)
        out['prices_moved'] = moved
    rep = os.path.join(wdir, 'reproduction_report.csv')
    rows = walks = 0
    if os.path.exists(rep):
        for r in csv.reader(open(rep, errors='replace'), delimiter=';'):
            rows += 1
            if len(r) > 8 and ('nclusters' in r[8] or 'comp' in r[8]
                              or 'creg' in r[8] or 'trees' in r[8]):
                walks += 1
    out['report_rows'] = rows
    out['param_walk_rows'] = walks
    sellers = set()
    for a in bb:
        if any(t.get('type') == 'sell' for t in a.get('trades', [])):
            sellers.add(a.get('label',''))
    out['sellers'] = len(sellers)
    return out

roots = sys.argv[1:]
for root in roots:
    for d in sorted(glob.glob(os.path.join(root, '*'))):
        if not os.path.isdir(d) or '_arch_' in d or '_bak' in d: continue
        base = os.path.basename(d)
        unit = 'sim-' + base.replace('_','-')
        if os.system(f'systemctl is-active --quiet {unit}') != 0: continue
        print(json.dumps(check(d)))
