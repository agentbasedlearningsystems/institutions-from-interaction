"""Split-half floor per the pre-registration: split a society's settled
products by iteration parity into two halves, compute each half's
within-family unigram proportions (same parse as measure_proportions),
and report the mean within-family TV between the halves — the
society's distance to itself, the measurement ceiling.

Usage: python3 split_half_floor.py <experiment_dir> [min_iter]
"""
import collections
import csv
import re
import sys

FAMILY_PREFIXES = ['data', 'vectorSpace', 'clusterer', 'anomalyDetector',
                   'classifier', 'preprocessor', 'test', 'labeler',
                   'nearestNeighbors', 'registry']
CHAIN = re.compile(r"'(f\d+_[A-Za-z0-9_]+)':\s*\[([^\]]*)\]")
NAME = re.compile(r"'f\d+_([A-Za-z0-9_]+)'")


def family_of(name):
    for f in FAMILY_PREFIXES:
        if name.startswith(f + '_') or name == f:
            return f
    return None


def half_props(expdir, min_iter, parity):
    uni = collections.defaultdict(collections.Counter)
    n = 0
    with open(f'{expdir}/reproduction_report.csv') as f:
        rd = csv.reader(f, delimiter=';')
        next(rd)
        for r in rd:
            try:
                t, b = int(r[0]), float(r[5])
            except (ValueError, IndexError):
                continue
            if (t < min_iter or b <= 0 or len(r) < 9
                    or 'OrderedDict' not in r[8] or len(r[8]) < 30
                    or t % 2 != parity):
                continue
            n += 1
            seen = set()
            for m in CHAIN.finditer(r[8]):
                parent = m.group(1)
                bare = parent.split('_', 1)[1]
                fam = family_of(bare)
                if fam and bare not in seen:
                    seen.add(bare)
                    uni[fam][bare] += 1
    props = {}
    for fam, ctr in uni.items():
        tot = sum(ctr.values())
        props[fam] = {k: c / tot for k, c in ctr.items()}
    return props, n


def main(expdir, min_iter=2000):
    a, na = half_props(expdir, min_iter, 0)
    b, nb = half_props(expdir, min_iter, 1)
    tvs = {}
    for fam, parts in a.items():
        if len(parts) < 2:
            continue
        keys = set(parts) | set(b.get(fam, {}))
        tvs[fam] = 0.5 * sum(abs(parts.get(k, 0.0) - b.get(fam, {}).get(k, 0.0))
                             for k in keys)
    mean_tv = sum(tvs.values()) / len(tvs) if tvs else float('nan')
    print(f'{expdir}: split-half floor (min_iter {min_iter}): '
          f'mean within-family TV {mean_tv:.4f} '
          f'(halves: {na} vs {nb} settles)')
    for fam, v in sorted(tvs.items(), key=lambda x: -x[1]):
        print(f'  {fam:24s} {v:.4f}')


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2000)
