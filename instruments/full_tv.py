import csv, re, collections, itertools, glob, os
CHAIN = re.compile(r"'f\d+_([A-Za-z0-9_]+)'")
TOOL = ("clusterer","vectorSpace","anomalyDetector","preprocessor","classifier","labeler","nearestNeighbors")
prof, rows_n = {}, {}
for d in sorted(glob.glob("./merge_work/bigtree_*")):
    w = os.path.basename(d)
    f = os.path.join(d, "reproduction_report.csv")
    if not os.path.exists(f) or "solo" in w: continue
    per = collections.defaultdict(collections.Counter); n = 0
    for r in csv.reader(open(f, errors="replace"), delimiter=";"):
        try: a = int(r[1]); sc = float(r[5])
        except (ValueError, IndexError): continue
        if sc <= 0 or len(r) < 9 or "OrderedDict" not in r[8]: continue
        n += 1
        for bare in set(CHAIN.findall(r[8])):
            if bare.startswith(TOOL) and a >= 42: per[a][bare] += 1
    specs = collections.Counter()
    for a, c in per.items():
        if c: specs[c.most_common(1)[0][0]] += 1
    if sum(specs.values()) >= 15: prof[w] = specs; rows_n[w] = n
def tv(p1, p2):
    ks = set(p1) | set(p2); n1, n2 = sum(p1.values()), sum(p2.values())
    return 0.5*sum(abs(p1.get(k,0)/n1 - p2.get(k,0)/n2) for k in ks)
elders = sorted(rows_n, key=lambda w: -rows_n[w])[:8]
print("elders (merged settles):", [(w.split("seed")[1], rows_n[w]) for w in elders])
pairs = sorted(((tv(prof[a], prof[b]), a, b) for a, b in itertools.combinations(elders, 2)), reverse=True)
for t, a, b in pairs[:5]:
    sh = len(set(prof[a]) & set(prof[b]))
    sa, sb = a.split("seed")[1], b.split("seed")[1]
    print("TV=%.3f shared_trades=%d  %s vs %s" % (t, sh, sa, sb))
