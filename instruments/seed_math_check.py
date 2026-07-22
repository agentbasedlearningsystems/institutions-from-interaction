"""The math: (1) build the v4 target mixture M = .10 U + .90 P over all
addresses from a source; (2) decode 6000 newborns through the REAL
engine under the seeded tree; (3) report total-variation distance of
expressed vs M over top addresses, and expressed vs the null tree."""
import sys, json, random, collections
sys.path.insert(0, './simulation')
sys.path.insert(0, '.')
sys.path.insert(0, '/tmp')
from SnetAgent import SnetAgent

class Shim(SnetAgent):
    def buyer_score_notification(self, *a, **k): pass
    def seller_score_notification(self, *a, **k): pass
    def payment_notification(self, *a, **k): pass
    def step(self, *a, **k): pass

def make_agent(cfg):
    a = Shim.__new__(Shim)
    a.p = dict(cfg['parameters']); a.o = cfg['ontology']
    return a

def sample(cfg, n, seed):
    ag = make_agent(cfg); rng = random.Random(seed)
    out = collections.Counter()
    for _ in range(n):
        vec = [rng.uniform(0, 1) for _ in range(60)]
        try: out[str(ag.ontology_item(vec))] += 1
        except Exception: pass
    return out

def rebalance(node):
    kids = [(k, v) for k, v in node.items() if isinstance(v, dict) and not k.startswith('_')]
    for k, v in kids:
        if '_weight' in v: v['_weight'] = 1.0 / len(kids)
        rebalance(v)

cfg = json.load(open(sys.argv[1]))
null = json.load(open(sys.argv[1])); rebalance(null['ontology'])
e_seed = sample(cfg, 6000, 11)
e_null = sample(null, 6000, 11)
def tv(c1, c2):
    keys = set(c1) | set(c2)
    n1, n2 = sum(c1.values()), sum(c2.values())
    return 0.5 * sum(abs(c1.get(k, 0)/n1 - c2.get(k, 0)/n2) for k in keys)
print('TV(seeded_expressed, null_expressed) =', round(tv(e_seed, e_null), 4))
print('top seeded-expressed:')
tot = sum(e_seed.values())
for k, v in e_seed.most_common(6): print('  %5.1f%% %s' % (100*v/tot, k[:60]))
print('top null-expressed:')
totn = sum(e_null.values())
for k, v in e_null.most_common(6): print('  %5.1f%% %s' % (100*v/totn, k[:60]))
