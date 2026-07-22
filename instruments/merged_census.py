"""Merged division census: a world's CURRENT report + all its archived
segments (from ./archive_reports on fleet-2, rsynced local per box or
read via the volume host). Usage: merged_census.py <world_dir> <NB>
[reports_dir]. Counts settled deliveries per learner across ALL
segments; primary tool per learner; distinct trades + evenness."""
import sys, csv, glob, os, math, collections
w, NB = sys.argv[1].rstrip('/'), int(sys.argv[2])
rep_dir = sys.argv[3] if len(sys.argv) > 3 else './archive_reports'
world = os.path.basename(w)
paths = []
cur = os.path.join(w, 'reproduction_report.csv')
if os.path.exists(cur): paths.append(cur)
paths += sorted(glob.glob(os.path.join(rep_dir, world + '_arch_*_report.csv')))
per = collections.defaultdict(collections.Counter)
rows = 0
for p in paths:
    try:
        for r in csv.reader(open(p, errors='replace'), delimiter=';'):
            try:
                a = int(r[1]); sc = float(r[5])
            except (ValueError, IndexError):
                continue
            if sc <= 0 or a < NB or len(r) < 9 or 'OrderedDict' not in r[8]:
                continue
            tool = None
            for tok in r[8].split("'"):
                if tok.startswith('f') and '_' in tok:
                    tool = tok.split('_', 1)[1].split('_stop')[0]
                    break
            if tool:
                per[a][tool] += 1; rows += 1
    except Exception:
        continue
prof = collections.Counter()
for a, c in per.items():
    prof[c.most_common(1)[0][0]] += 1
n = sum(prof.values())
if n:
    probs = [v/n for v in prof.values()]
    H = -sum(p*math.log(p) for p in probs)
    even = H/math.log(len(prof)) if len(prof) > 1 else 0.0
    top = ', '.join(f'{t}:{c}' for t, c in prof.most_common(4))
    print(f'{world}: segments={len(paths)} settles={rows} earners={n} trades={len(prof)} evenness={even:.2f} -> {top}')
else:
    print(f'{world}: segments={len(paths)} settles={rows} earners=0')
