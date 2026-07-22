"""Merged TRUE division per world: current report + archived segments,
deduplicated (a checkpoint-resumed segment can replay iterations already
recorded by its predecessor), per-learner primary tool over the whole
deduped life, distinct trades + evenness (normalized entropy over
practitioners per trade). Matches Table A1's merged-census methodology.

Usage: merged_division.py <experiments_root> <NB> <arch_dir>
Prints one JSON line per bigtree world.
"""
import csv
import glob
import json
import math
import os
import re
import sys

CHAIN = re.compile(r"'f\d+_([A-Za-z0-9_]+)'")
TOOL = ('clusterer', 'vectorSpace', 'anomalyDetector', 'preprocessor',
        'classifier', 'labeler', 'nearestNeighbors')

root, NB, arch = sys.argv[1], int(sys.argv[2]), sys.argv[3]
for wdir in sorted(glob.glob(os.path.join(root, 'bigtree_*_seed2[0-9][0-9]'))):
    w = os.path.basename(wdir)
    if any(x in w for x in ('_bak', '_aborted')):
        continue
    paths = sorted(glob.glob(os.path.join(arch, w + '_arch_*_report.csv')))
    cur = os.path.join(wdir, 'reproduction_report.csv')
    if os.path.exists(cur):
        paths.append(cur)
    seen = set()
    per = {}
    n_rows = n_dup = 0
    age = 0
    for p in paths:
        try:
            f = open(p, errors='replace')
        except OSError:
            continue
        for r in csv.reader(f, delimiter=';'):
            try:
                t = int(r[0]); a = int(r[1]); sc = float(r[5])
            except (ValueError, IndexError):
                continue
            age = max(age, t)
            if sc <= 0 or a < NB or len(r) < 9 or 'OrderedDict' not in r[8]:
                continue
            key = (t, a, sc, r[8][:120])
            if key in seen:
                n_dup += 1
                continue
            seen.add(key)
            n_rows += 1
            for bare in set(CHAIN.findall(r[8])):
                if bare.startswith(TOOL):
                    per.setdefault(a, {}).setdefault(bare, 0)
                    per[a][bare] += 1
        f.close()
    primary = {}
    for a, tools in per.items():
        primary[a] = max(tools, key=tools.get)
    prac = {}
    for a, tool in primary.items():
        prac.setdefault(tool, 0)
        prac[tool] += 1
    n_learn = len(primary)
    n_trades = len(prac)
    if n_trades > 1:
        tot = sum(prac.values())
        H = -sum(c / tot * math.log(c / tot) for c in prac.values())
        even = H / math.log(n_trades)
    else:
        even = 0.0
    print(json.dumps(dict(world=w, segments=len(paths), settles=n_rows,
                          dup_dropped=n_dup, age=age, learners=n_learn,
                          trades=n_trades, evenness=round(even, 3))))
