#!/usr/bin/env python3
"""cr_keyless.py <seeds_outward.json> <out.json> [--rng 1]   (7 Oct 2026, her yes to decision 28: the backward ablation)

The KEYLESS outward seed: the same practices as the outward seed (the rows of a mature learner's first sell section as
(type, item) pairs, dealt by share), with the KEY of every decoder item replaced by a key drawn at random from the keys
the source's practices used. The seed then carries the division of labour (who sells a class-2 decoder, who constructs
which class, who buys parts) and nothing of the solution to the hard-keys problem, which the learner must find again.
Her 16 July law of seeds (outward statistics only, never signs) and her 3 Oct program (ablate back to the minimal seed).
Run with the blank step size (make_cr_world --seed-sigma 0.1), so the seeded learner is as warm as a blank one.
"""
import json
import random
import re
import sys

ITEM = re.compile(r'^(decoder_[a-z]+_(?:c\d+|[A-H])_)(k\d+)$')
src, out = sys.argv[1], sys.argv[2]
rng = random.Random(int(sys.argv[sys.argv.index('--rng') + 1]) if '--rng' in sys.argv else 1)
seeds = json.load(open(src))
keys = sorted({ITEM.match(t.get('item', '')).group(2) for m in seeds.values() for t in m.get('trades', []) if ITEM.match(str(t.get('item', '')))})
changed = 0
for aid, m in seeds.items():
    for t in m.get('trades', []):
        mm = ITEM.match(str(t.get('item', '')))
        if mm:
            t['item'] = mm.group(1) + rng.choice(keys); changed += 1
    m['_role_keyless'] = 'keys drawn at random from %s; the roles are the source practice' % keys
    m.pop('sign', None)
json.dump(seeds, open(out, 'w'), indent=1)
print('keyless seed: %d learners, %d decoder items re-keyed at random from %s -> %s' % (len(seeds), changed, keys, out))
