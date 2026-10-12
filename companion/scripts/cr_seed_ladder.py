#!/usr/bin/env python3
"""cr_seed_ladder.py - coevolutionary reconstruction seeds for the cipher ladder world (7 Oct 2026).

Her program (EXPLAINED_20261003.md section 5; her law of seeds, 16 Jul; her 6 Aug rule: seeds are the initial state
only): measure a mature society as STATISTICS and seed a fresh world's starting point from them, never its genomes.
From the last --last boards of a source run this reads every learner's PRACTICE, the rows of its first sell section
(sell, construct and buy rows as (type, item) pairs, the item cut at '_stop'), the share of learner-boards each
practice holds within the learner's capacity group (3-row or 16-row learners, plans_by_learner cycled by agent number
as the engine does), and, for the ceiling only, the practice's display-sign centroid. It writes seed_messages for the
new world's learners, dealing practices within each group in proportion to their shares.

Levels (--level):
  presence   the ceiling, "more info than we have": every learner seeded with a practice AND its centroid sign (+noise)
  outward    outward statistics only: every learner seeded with a practice; NO sign (the sign starts random)
  half       outward, for every other learner of each group; the rest start blank
Humans are never seeded (their wants are re-learned; that is part of the read).

Usage: cr_seed_ladder.py <source_run_dir> --level presence|outward|half [--humans 11] [--last 2000] [--groups 3,16]
                         [--noise 0.03] [--min-share 0.05] [--no-buys] --out seeds.json [--roles roles.json]
"""
import collections
import json
import os
import re
import sys

import numpy as np


def agent_id(label):
    m = re.search(r'RoleSim Agent (\d+)', label or '')
    return int(m.group(1)) if m else None


def first_section(tr):
    """The rows of the first sell section before the stop codon, as (type, item) pairs; None if there is none."""
    for j, x in enumerate(tr):
        if x.get('type') == 'stop' or str(x.get('item', '')).startswith('stop'):
            return None
        if x.get('type') == 'sell':
            k = j + 1
            while k < len(tr) and tr[k].get('type') not in ('stop', 'sell') and not str(tr[k].get('item', '')).startswith('stop'):
                k += 1
            return tuple((r.get('type'), str(r.get('item', '')).split('_stop')[0]) for r in tr[j:k]
                         if r.get('type') in ('sell', 'construct', 'buy'))
    return None


def main():
    run = sys.argv[1].rstrip('/')
    a = sys.argv[2:]

    def opt(n, d):
        return a[a.index(n) + 1] if n in a else d
    level = opt('--level', None)
    assert level in ('presence', 'outward', 'half'), 'level must be presence, outward or half'
    H = int(opt('--humans', 11)); last = int(opt('--last', 2000))
    groups = [int(x) for x in opt('--groups', '3,16').split(',')]
    noise = float(opt('--noise', 0.03)); min_share = float(opt('--min-share', 0.05))
    out = opt('--out', 'seeds.json'); roles_out = opt('--roles', None)
    keep = ('sell', 'construct') if '--no-buys' in a else ('sell', 'construct', 'buy')
    group_of = lambda aid: groups[aid % len(groups)]           # the engine's and capacity_reads.py's rule

    logs = sorted(int(f[3:-4]) for f in os.listdir(os.path.join(run, 'logs')) if f.startswith('log') and f[3:-4].isdigit())
    window = [b for b in logs if b > max(logs) - last]
    holders = collections.defaultdict(list)                     # (group, practice) -> [(aid, sign floats)]
    learners = set()
    for L in window:
        try:
            msgs = json.load(open(os.path.join(run, 'logs', 'log%d.txt' % L)))
        except Exception:
            continue
        for m in msgs:
            aid = agent_id(m.get('label'))
            if aid is None or aid < H or not m.get('sign'):
                continue
            learners.add(aid)
            prac = first_section(m.get('trades', []))
            if not prac:
                continue
            x = [(float(v) + 1.0) / 2.0 for v in m['sign']]      # message sign [-1,1] -> genome float [0,1]
            holders[(group_of(aid), prac)].append((aid, x))
    if not holders:
        print('no practices found'); return 1
    by_group = collections.defaultdict(list)
    for (g, prac), v in holders.items():
        by_group[g].append((prac, v))
    roles = {}
    seeds = {}
    rng = np.random.RandomState(306)
    report = []
    for g in sorted(by_group):
        total = sum(len(v) for _, v in by_group[g])
        rl = []
        for prac, v in by_group[g]:
            share = len(v) / total
            if share < min_share:
                continue
            xs = np.array([x for _, x in v])
            rl.append(dict(practice=[list(p) for p in prac], share=share, holders=sorted(set(aid for aid, _ in v)),
                           agent_boards=len(v), centroid=xs.mean(axis=0).tolist(), spread=float(xs.std(axis=0).mean())))
        rl.sort(key=lambda r: -r['share'])
        roles[str(g)] = rl
        targets = sorted(aid for aid in learners if group_of(aid) == g)
        if level == 'half':
            targets = targets[::2]
        K = len(targets)
        tot = sum(r['share'] for r in rl) or 1.0
        quota = [r['share'] / tot * K for r in rl]
        counts = [int(q) for q in quota]
        for i in sorted(range(len(rl)), key=lambda i: -(quota[i] - counts[i]))[:K - sum(counts)]:
            counts[i] += 1
        it = iter(targets)
        for r, c in zip(rl, counts):
            for _ in range(c):
                aid = next(it)
                rows = [{'type': t, 'item': item} for t, item in r['practice'] if t in keep]
                rows.append({'type': 'stop'})
                msg = {'trades': rows, '_role': ['%s:%s' % (t, item) for t, item in r['practice']]}
                if level == 'presence':
                    msg['sign'] = np.clip(np.array(r['centroid']) + rng.normal(0, noise, len(r['centroid'])), 0.0, 1.0).tolist()
                seeds[str(aid)] = msg
            report.append('  group %2d rows: %-60s share %.2f holders %s -> %d seeds' % (
                g, ' '.join('%s:%s' % (t, item.replace('decoder_subst_', '')) for t, item in r['practice'])[:60], r['share'], r['holders'], c))
    json.dump(seeds, open(out, 'w'), indent=1)
    if roles_out:
        json.dump(dict(run=os.path.basename(run), window_boards=[min(window), max(window)], level=level, roles=roles),
                  open(roles_out, 'w'), indent=1)
    print('source %s, boards %d-%d, level %s: %d learners, %d seeded' % (os.path.basename(run), min(window), max(window), level,
                                                                       len(learners), len(seeds)))
    for line in report:
        print(line)
    return 0


if __name__ == '__main__':
    sys.exit(main())
