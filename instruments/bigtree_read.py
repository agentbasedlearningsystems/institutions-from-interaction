"""Big-world reader: want coverage by buyer + division over true learners.
Usage: bigtree_read.py <expdir> [n_buyers]   (n_buyers: 18 diverse, 34 rich)"""
import csv, re, sys, collections, math
CHAIN = re.compile(r"'f\d+_([A-Za-z0-9_]+)'")
TOOL = ('clusterer','vectorSpace','anomalyDetector','preprocessor','classifier','labeler','nearestNeighbors')
def main(expdir, NB):
    labels, first, cnt = {}, {}, collections.Counter()
    per = collections.defaultdict(collections.Counter); n=junk=deep=0; last=0
    with open(expdir + "/reproduction_report.csv") as f:
        rd = csv.reader(f, delimiter=";"); next(rd)
        for r in rd:
            try: t=int(r[0]); a=int(r[1]); sc=float(r[5])
            except (ValueError, IndexError): continue
            last = max(last, t)
            if a not in labels and len(r) > 2: labels[a] = r[2].split(",")[0]
            if sc <= 0 or len(r) < 9 or "OrderedDict" not in r[8]: continue
            n += 1
            if sc <= 0.05: junk += 1
            parts = CHAIN.findall(r[8])
            if len(parts) > 6: deep += 1
            if a < NB:
                cnt[a] += 1; first.setdefault(a, t)
            for bare in set(parts):
                if bare.startswith(TOOL) and a >= NB: per[a][bare] += 1
    print(f'{expdir.split("/")[-1]}: age {last}, settles {n}, junk {junk/max(n,1):.0%}, deep {deep/max(n,1):.0%}')
    print("WANT COVERAGE (first settle iter, n settles):")
    for a in sorted(first, key=lambda x: first[x]):
        print(f"  iter {first[a]:>4}  x{cnt[a]:>4}  {labels.get(a,a)}")
    uncovered = [labels[a] for a in labels if a < NB and a not in first]
    print("WANTS WITH NO SETTLE YET:", len(uncovered))
    for u in sorted(uncovered): print("   -", u)
    specs = collections.Counter()
    for a, c in per.items():
        if c: specs[c.most_common(1)[0][0]] += 1
    tot = sum(specs.values()); even = 0.0
    if len(specs) > 1:
        even = -sum((v/tot)*math.log2(v/tot) for v in specs.values())/math.log2(len(specs))
    print(f"TRUE DIVISION (learners id>={NB}): {len(specs)} distinct over {tot} learners, evenness {even:.2f}")
    print("  ", ", ".join(f"{k}:{v}" for k, v in specs.most_common(8)))
main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 18)
