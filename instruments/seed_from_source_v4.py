"""Seeder v4 (Debbie's July 20 ruling): at EVERY tree level including
family roots, weight = 0.10 * uniform + 0.90 * (end-window trade
appearances proportional). Census from the source's END WINDOW only:
paid deliveries in the last quarter of its iterations (min 500 iters).
Usage: seed_from_source_v4.py <source_report.csv> <base_config> <seed> <out>"""
import csv, re, sys, json, collections
src_report, base_path, seed, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
CHAIN = re.compile(r"'f\d+_([A-Za-z0-9_]+)'")
rows = []
with open(src_report, errors='replace') as f:
    rd = csv.reader(f, delimiter=';'); next(rd, None)
    for r in rd:
        try: t = int(r[0]); sc = float(r[5])
        except (ValueError, IndexError): continue
        if sc <= 0 or len(r) < 9 or 'OrderedDict' not in r[8]: continue
        rows.append((t, r[8]))
if not rows:
    sys.exit('no paid deliveries in source report')
t_max = max(t for t, _ in rows)
N_ITER = 5   # Debbie 7/20: the last n iterations, n like 5 - the standing state
t_cut = t_max - N_ITER
counts = collections.Counter()
n_end = 0
for t, chain in rows:
    if t < t_cut: continue
    n_end += 1
    for p in set(CHAIN.findall(chain)):
        counts[p] += 1
base = json.load(open(base_path), object_pairs_hook=collections.OrderedDict)
tree = base['ontology']
RAND = 0.10
def leaf_name(path): return '_'.join(path)
def tally(node, path):
    # returns (n_addresses, uses) for the subtree; every node is a
    # stoppable/sellable address, so interior appearances count too
    kids = [(k, v) for k, v in node.items() if isinstance(v, dict) and not k.startswith('_')]
    own = counts.get(leaf_name(path), 0) if path else 0
    if not kids: return 1, own
    nl, us = 1, own
    for k, v in kids:
        a, b = tally(v, path + [k]); v['_nl'] = a; v['_us'] = b
        nl += a; us += b
    return nl, us
L, G = tally(tree, [])
def reweight(node):
    kids = [(k, v) for k, v in node.items() if isinstance(v, dict) and not k.startswith('_')]
    if not kids: return
    masses = {}
    for k, v in kids:
        masses[k] = RAND * (v['_nl'] / L) + ((1.0 - RAND) * (v['_us'] / G) if G > 0 else RAND * 0)
        if G == 0: masses[k] = v['_nl'] / L
    tot = sum(masses.values())
    for k, v in kids:
        v['_weight'] = round(masses[k] / tot, 6) if tot > 0 else round(1.0 / len(kids), 6)
        reweight(v)
        del v['_nl']; del v['_us']
reweight(tree)
base['parameters']['seed'] = seed
base['parameters']['payment_ledger'] = True
base['parameters']['output_path'] = f'experiments/cr_{out}/'
base['parameters']['label'] = f'cr_{out}'
json.dump(base, open(f'/tmp/cr_{out}.json', 'w'), indent=1)
used = sum(1 for c in counts.values() if c > 0)
print(f'cr_{out}: end-window census t>={t_cut} of {t_max} ({n_end} paid deliveries, {used} tools, {sum(counts.values())} part-uses); 10% of TOTAL leaf space random + 90% end-window proportional')
