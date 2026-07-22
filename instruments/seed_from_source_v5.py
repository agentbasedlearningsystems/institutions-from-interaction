"""Seeder v5 (Debbie 7/20, tools-only ruling): census counts ONLY the
production-tool branches (what is traded); data/test branches keep their
natural uniform weights so seeded daughters match nulls outside the
traded space. Within the tool space: 10% uniform-over-tool-leaves + 90%
standing-state proportional (last 5 iterations). Defaults credit the
function, never its settings children.
Usage: seed_from_source_v5.py <source_report.csv> <base_config> <seed> <out>"""
import csv, re, sys, json, collections
src_report, base_path, seed, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
CHAIN = re.compile(r"'f\d+_([A-Za-z0-9_]+)'")
TOOL_ROOTS = ('preprocessor', 'clusterer', 'vectorSpace', 'classifier',
              'anomalyDetector', 'labeler', 'nearestNeighbors')
rows = []
with open(src_report, errors='replace') as f:
    rd = csv.reader(f, delimiter=';'); next(rd, None)
    for r in rd:
        try: t = int(r[0]); sc = float(r[5])
        except (ValueError, IndexError): continue
        if sc <= 0 or len(r) < 9 or 'OrderedDict' not in r[8]: continue
        rows.append((t, r[8]))
if not rows: sys.exit('no paid deliveries')
t_max = max(t for t, _ in rows)
N_ITER = 5
t_cut = t_max - N_ITER
counts = collections.Counter(); n_end = 0
for t, ch in rows:
    if t < t_cut: continue
    n_end += 1
    for p in set(CHAIN.findall(ch)):
        if p.startswith(TOOL_ROOTS): counts[p] += 1
base = json.load(open(base_path), object_pairs_hook=collections.OrderedDict)
tree = base['ontology']
RAND = 0.10
def leaf_name(path): return '_'.join(path)
def tally(node, path):
    kids = [(k, v) for k, v in node.items() if isinstance(v, dict) and not k.startswith('_')]
    own = counts.get(leaf_name(path), 0) if path else 0
    if not kids: return 1, own
    nl, us = 1, own
    for k, v in kids:
        a, b = tally(v, path + [k]); v['_nl'] = a; v['_us'] = b
        nl += a; us += b
    return nl, us
def uniformize(node):
    kids = [(k, v) for k, v in node.items() if isinstance(v, dict) and not k.startswith('_')]
    for k, v in kids:
        if '_weight' in v or True: v['_weight'] = round(1.0 / len(kids), 6)
        uniformize(v)
def clean(node):
    kids = [(k, v) for k, v in node.items() if isinstance(v, dict) and not k.startswith('_')]
    for k, v in kids:
        v.pop('_nl', None); v.pop('_us', None); clean(v)
root_kids = [(k, v) for k, v in tree.items() if isinstance(v, dict) and not k.startswith('_')]
n_root = len(root_kids)
tool_kids = [(k, v) for k, v in root_kids if k in TOOL_ROOTS]
L_t = G_t = 0
for k, v in tool_kids:
    a, b = tally(v, [k]); v['_nl'] = a; v['_us'] = b
    L_t += a; G_t += b
def reweight(node):
    kids = [(k, v) for k, v in node.items() if isinstance(v, dict) and not k.startswith('_')]
    if not kids: return
    masses = {}
    for k, v in kids:
        prop = (v['_us'] / G_t) if G_t > 0 else 0.0
        masses[k] = RAND * (v['_nl'] / L_t) + (1.0 - RAND) * prop
    tot = sum(masses.values())
    for k, v in kids:
        v['_weight'] = round(masses[k] / tot, 6) if tot > 0 else round(1.0 / len(kids), 6)
        reweight(v)
tool_mass_total = sum(RAND * (v['_nl'] / L_t) + (1.0 - RAND) * ((v['_us'] / G_t) if G_t else 0.0) for k, v in tool_kids)
nontool_share = (n_root - len(tool_kids)) / n_root
for k, v in root_kids:
    if k in TOOL_ROOTS:
        m = RAND * (v['_nl'] / L_t) + (1.0 - RAND) * ((v['_us'] / G_t) if G_t else 0.0)
        v['_weight'] = round((1.0 - nontool_share) * m / tool_mass_total, 6)
        reweight(v)
    else:
        v['_weight'] = round(1.0 / n_root, 6)
        uniformize(v)
for k, v in tool_kids: clean(v)
base['parameters']['seed'] = seed
base['parameters']['payment_ledger'] = True
base['parameters']['output_path'] = f'experiments/cr_{out}/'
base['parameters']['label'] = f'cr_{out}'
json.dump(base, open(f'/tmp/cr_{out}.json', 'w'), indent=1)
used = sum(1 for c in counts.values() if c > 0)
print(f'cr_{out}: tools-only census t>={t_cut} of {t_max} ({n_end} deliveries, {used} tools, {sum(counts.values())} tool-uses); data/test natural-uniform; 10% of tool space random')
