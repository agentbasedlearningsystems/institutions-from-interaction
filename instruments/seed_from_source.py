"""Seeder v3 (Debbie's conditional method): extract per-level conditional
tool-usage probabilities from a SOURCE society's paid chains, write them
as tree weights into a DAUGHTER config. Family roots stay balanced (the
family-root lesson); smoothing keeps every tool a small exploration lane.
Usage: seed_from_source.py <source_report.csv> <NB> <base_config> <seed> <out>"""
import csv, re, sys, json, collections
src_report, NB, base_path, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5]
CHAIN = re.compile(r"'f\d+_([A-Za-z0-9_]+)'")
counts = collections.Counter()
with open(src_report) as f:
    rd = csv.reader(f, delimiter=";"); next(rd)
    for r in rd:
        try: a = int(r[1]); sc = float(r[5])
        except (ValueError, IndexError): continue
        if sc <= 0 or len(r) < 9 or "OrderedDict" not in r[8]: continue
        for p in set(CHAIN.findall(r[8])):
            counts[p] += 1
base = json.load(open(base_path), object_pairs_hook=collections.OrderedDict)
tree = base['ontology']
K = 0.5   # smoothing: unused tools keep a lane
def leaf_name(path):
    return '_'.join(path)
def reweight(node, path):
    kids = [(k, v) for k, v in node.items() if isinstance(v, dict) and not k.startswith('_')]
    if not kids: return counts.get(leaf_name(path), 0)
    totals = {}
    for k, v in kids:
        totals[k] = reweight(v, path + [k])
    gross = sum(totals.values())
    n = len(kids)
    for k, v in kids:
        if path == []:   # family roots stay balanced (the seeder lesson)
            v['_weight'] = round(1.0 / n, 6)
        else:
            v['_weight'] = round((totals[k] + K) / (gross + K * n), 6)
    return gross
reweight(tree, [])
base['parameters']['seed'] = seed
base['parameters']['payment_ledger'] = True
base['parameters']['output_path'] = f'experiments/cr_{out}/'
base['parameters']['label'] = f'cr_{out}'
json.dump(base, open(f'/tmp/cr_{out}.json', 'w'), indent=1)
used = sum(1 for c in counts.values() if c > 0)
print(f'cr_{out}: seeded from {src_report.split("/")[-2]} ({used} tools used in source, {sum(counts.values())} paid part-uses)')
