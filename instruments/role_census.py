"""Role census: per LEARNING agent (ids >= 6), the dominant tool
family among the parents it sold in settled products past burn-in,
with its share — then the society's role composition (how many
learners hold each family as their role).

Usage: python3 role_census.py <experiment_dir> [min_iter]
"""
import collections
import csv
import re
import sys

FAMS = ['data', 'vectorSpace', 'clusterer', 'anomalyDetector',
        'classifier', 'preprocessor', 'test', 'labeler',
        'nearestNeighbors', 'registry']
CHAIN = re.compile(r"'(f\d+_[A-Za-z0-9_]+)':\s*\[([^\]]*)\]")


def family_of(name):
    for f in FAMS:
        if name.startswith(f + '_') or name == f:
            return f
    return None


def main(expdir, min_iter=2000):
    per = collections.defaultdict(collections.Counter)
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
            seen = set()
            for m in CHAIN.finditer(r[8]):
                bare = m.group(1).split('_', 1)[1]
                fam = family_of(bare)
                if fam and bare not in seen:
                    seen.add(bare)
                    per[a][fam] += 1
    roles = collections.Counter()
    for a in sorted(per):
        if a < 6:
            continue  # pinned humans excluded; learners only
        fam, cnt = per[a].most_common(1)[0]
        tot = sum(per[a].values())
        roles[fam] += 1
    comp = ' '.join(f'{f}:{c}' for f, c in roles.most_common())
    print(f'{expdir}: learner role composition -> {comp}')


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2000)
