"""Disclosed interim peek (Debbie, July 22 ~1:20 AM): x272 alone at age
>= 300. Declared read applied to one daughter: last-5-iteration
tools-only census, TV similarity vs both frozen sources, fidelity =
sim(own=X) - sim(other=Y). Panel-complete unanimous-split stays primary."""
import csv
import json
import re
import sys

CHAIN = re.compile(r"'f\d+_([A-Za-z0-9_]+)'")
TOOL = ('clusterer', 'vectorSpace', 'anomalyDetector', 'preprocessor',
        'classifier', 'labeler', 'nearestNeighbors')

rep = sys.argv[1]
frozen = json.load(open('truth/cr3_frozen_profiles.json'))

rows = []
for r in csv.reader(open(rep, errors='replace'), delimiter=';'):
    try:
        t = int(r[0]); sc = float(r[5])
    except (ValueError, IndexError):
        continue
    if sc <= 0 or len(r) < 9 or 'OrderedDict' not in r[8]:
        continue
    rows.append((t, r[8]))
tmax = max(t for t, _ in rows)
cnt = {}
n_del = 0
for t, ch in rows:
    if t >= tmax - 4:
        n_del += 1
        for p in set(CHAIN.findall(ch)):
            if p.startswith(TOOL):
                cnt[p] = cnt.get(p, 0) + 1

def tv_sim(a, b):
    keys = set(a) | set(b)
    sa, sb = sum(a.values()), sum(b.values())
    if not sa or not sb:
        return 0.0
    tv = 0.5 * sum(abs(a.get(k, 0) / sa - b.get(k, 0) / sb) for k in keys)
    return 1.0 - tv

simX = tv_sim(cnt, frozen['source_X_bigtree_basic_seed227']['counts'])
simY = tv_sim(cnt, frozen['source_Y_bigtree_alwayson_seed211']['counts'])
print(json.dumps(dict(daughter='cr_3x272', age=tmax, window_deliveries=n_del,
                      tools=len(cnt), sim_own_X=round(simX, 3),
                      sim_other_Y=round(simY, 3),
                      fidelity=round(simX - simY, 3), census=cnt)))
