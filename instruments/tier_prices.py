import sys, gzip, json, glob, os
w = sys.argv[1]
logs = sorted(glob.glob(os.path.join(w, 'logs', '*')), key=os.path.getmtime)
if len(logs) < 2:
    print('  too few boards'); sys.exit()
def load(p):
    op = gzip.open if p.endswith('.gz') else open
    try: return json.load(op(p, 'rt', errors='replace'))
    except Exception: return None
def tiers(bb):
    lo, hi = [], []
    for a in bb or []:
        lab = a.get('label', '')
        for t in a.get('trades', []):
            if t.get('type') == 'buy' and t.get('alpha') is not None:
                m = t.get('midpoint')
                if m is None: continue
                (lo if 'Baby' in lab else hi).append(m)
    def avg(x): return round(sum(x)/len(x), 1) if x else None
    return avg(lo), avg(hi)
b0, b1 = load(logs[0]), load(logs[-1])
e0, c0 = tiers(b0); e1, c1 = tiers(b1)
print(f'  entry-tier avg midpoint: {e0} -> {e1} | composite-tier: {c0} -> {c1}')
