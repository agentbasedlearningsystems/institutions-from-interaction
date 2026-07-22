import csv, re, sys, statistics
CHAIN = re.compile(r"'f\d+_[A-Za-z0-9_]+'")
def main(expdir, min_iter=0):
    rows = []
    with open(f'{expdir}/reproduction_report.csv') as f:
        rd = csv.reader(f, delimiter=';')
        next(rd)
        for r in rd:
            try:
                t = int(r[0]); b = float(r[5])
            except (ValueError, IndexError): continue
            if b <= 0 or len(r) < 9 or 'OrderedDict' not in r[8]: continue
            depth = len(CHAIN.findall(r[8]))
            rows.append((t, depth))
    rows.sort()
    n = len(rows)
    q1 = [d for _, d in rows[:n//4]]
    q4 = [d for _, d in rows[3*n//4:]]
    print(f'{expdir}: settles {n}; chain size mean first-quartile {statistics.mean(q1):.2f} -> last-quartile {statistics.mean(q4):.2f}; max ever {max(d for _, d in rows)}')
main(sys.argv[1])
