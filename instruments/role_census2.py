"""Specialization roster: per LEARNING agent (id >= 6), the dominant
TOOL part (families: clusterer, vectorSpace, anomalyDetector,
preprocessor, classifier, labeler, nearestNeighbors — excluding test
roots and raw data terminals) across its settled products, with share;
then the society's roster grouped by specialization.

Usage: python3 role_census2.py <experiment_dir> [min_iter]
"""
import collections
import csv
import re
import sys

TOOL_FAMS = ('clusterer', 'vectorSpace', 'anomalyDetector',
             'preprocessor', 'classifier', 'labeler', 'nearestNeighbors')
CHAIN = re.compile(r"'f\d+_([A-Za-z0-9_]+)'")


def main(expdir, min_iter=2000):
    per = collections.defaultdict(collections.Counter)
    sold = collections.Counter()
    with open(f'{expdir}/reproduction_report.csv') as f:
        rd = csv.reader(f, delimiter=';')
        next(rd)
        for r in rd:
            try:
                t, a, b = int(r[0]), int(r[1]), float(r[5])
            except (ValueError, IndexError):
                continue
            if t < min_iter or b <= 0 or len(r) < 9 \
                    or 'OrderedDict' not in r[8] or len(r[8]) < 30:
                continue
            sold[a] += 1
            for bare in set(CHAIN.findall(r[8])):
                if bare.startswith(TOOL_FAMS):
                    per[a][bare] += 1
    roster = collections.defaultdict(list)
    for a in sorted(per):
        if a < 6:
            continue
        top = per[a].most_common(2)
        if not top:
            continue
        name, cnt = top[0]
        tot = sum(per[a].values())
        second = f' (+{top[1][0]})' if len(top) > 1 and top[1][1] > 0.3 * cnt else ''
        roster[name].append((a, cnt / tot, second))
    print(f'== {expdir}')
    for spec, agents in sorted(roster.items(), key=lambda x: -len(x[1])):
        ids = ', '.join(f'a{a}({share:.0%}){sec}' for a, share, sec in agents)
        print(f'  {len(agents):2d}x {spec:45s} {ids}')


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2000)
