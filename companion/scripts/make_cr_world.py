#!/usr/bin/env python3
"""make_cr_world.py - a coevolutionary-reconstruction world from a generated ladder config and a seeds file (7 Oct 2026).

Usage: make_cr_world.py <generated_config.json> <seeds.json> --label <name> --source <source run label>
                        --level presence|outward|half [--seed-sigma 0.02] [--out-dir ./]
Reads the generated world (make_ladder_world.py, with its own world seed), adds seed_messages (cr_seed_ladder.py),
seed_sigma (the seeded learners' first step; the engine adapts from there) and the provenance, renames the world to
cipher_s<seed>_<label>, and writes _cipher_s<seed>_<label>.json beside the input. Her 6 Aug rule: seeds set the
initial state only; nothing is pinned afterwards (RoleSim.py, the seed_messages branch).
"""
import collections
import json
import os
import sys

cfg_path, seeds_path = sys.argv[1], sys.argv[2]
a = sys.argv[3:]


def opt(n, d):
    return a[a.index(n) + 1] if n in a else d
label = opt('--label', None); source = opt('--source', None); level = opt('--level', None)
assert label and source and level in ('presence', 'outward', 'half', 'keyless'), 'need --label, --source, --level'   # keyless: 7 Oct, decision 28
sigma = float(opt('--seed-sigma', '0.02')); out_dir = opt('--out-dir', os.path.dirname(cfg_path) or '.')
world_seed = opt('--world-seed', None)      # a fresh RNG seed for the seeded world, never the source's

c = json.load(open(cfg_path), object_pairs_hook=collections.OrderedDict)
p = c['parameters']
seeds = json.load(open(seeds_path))
if world_seed is not None:
    p['seed'] = int(world_seed)
p['seed_messages'] = seeds
p['seed_sigma'] = sigma
lab = 'cipher_s%d_%s' % (int(p['seed']), label)
p['label'] = lab
p['output_path'] = 'experiments/%s/' % lab
p['_cr_provenance'] = ('COEVOLUTIONARY RECONSTRUCTION (7 Oct 2026, her program of 3 Oct and her word of 7 Oct 09:1x: the freed '
                       'cores go to reconstruction): level %s, %d learners seeded from the practices of %s (cr_seed_ladder.py: '
                       'the first sell section as (type, item) rows, dealt within capacity groups by share of learner-boards; '
                       'presence adds the practice\'s display-sign centroid, outward and half add no sign; keyless (7 Oct, her yes to decision 28) keeps the roles and draws every decoder key at random). Humans unseeded. '
                       'seed_sigma %s. Her 6 Aug rule: the initial state only.' % (level, len(seeds), source, sigma))
fn = os.path.join(out_dir, '_%s.json' % lab)
if os.path.exists(fn):
    raise SystemExit('exists, not overwriting: ' + fn)
json.dump(c, open(fn, 'w'), indent=1)
print('wrote %s: %d seeded learners, level %s, source %s' % (fn, len(seeds), level, source))
