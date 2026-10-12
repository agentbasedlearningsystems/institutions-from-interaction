#!/usr/bin/env python3
"""cd_composition.py - complexification in a Cobb-Douglas world, by her definition: deliveries that HUMANS PAID FOR
(payment > 0, no zero-pay marker), split by the good the human's row asked for, and whether the product was assembled
by MORE THAN ONE AGENT (the seller's sold row plus the settled buys inside it, followed to other agents' rows).

Products are rebuilt as in cipher_reads.py (her 11 Sep ruling: the buyer's chosen offer names the seller and the sold
row; the product is that row up to the next sell or stop; a settled buy inside points at another agent's row; recurse).
Unlike cipher_reads.py, every buy row of a Cobb-Douglas human is read on its own, with its own good (cd_good).

Per window of boards (logged boards only), for one-layer, two-layer and four-layer goods:
  paid      human-paid deliveries per logged board
  multi     of those, passing through two or more agents (absolute per board, and the share); this includes
            middlemen who resell another agent's product unchanged
  composed  of those, ASSEMBLED by two or more agents: at least two agents each contribute a decoder layer (a sell or
            construct row naming a full leaf, c<class>_k<key>) - her complexification; resale chains do not count
  full      fully right (score >= 0.99) paid deliveries per board, and how many of those were composed
Usage: cd_composition.py <run_dir> [--window 2000] [--humans 11]
"""
import collections
import json
import os
import re
import sys

LEAF = re.compile(r'(?:c\d|[A-H])_k\d')   # c0_k1 or A_k1 (7 Oct: classes named by their corpus letters)


def agent_id(label):
    m = re.search(r'RoleSim Agent (\d+)', label or '')
    return int(m.group(1)) if m else None


def section(tr, j):
    k = j + 1
    while k < len(tr) and tr[k].get('type') not in ('stop', 'sell') and not str(tr[k].get('item', '')).startswith('stop'):
        k += 1
    return tr[j:k]


def main():
    a = sys.argv[1:]
    run = a[0].rstrip('/')
    W = int(a[a.index('--window') + 1]) if '--window' in a else 2000
    H = int(a[a.index('--humans') + 1]) if '--humans' in a else 11
    logs = os.path.join(run, 'logs')
    nums = sorted(int(f[3:-4]) for f in os.listdir(logs) if f.startswith('log') and f[3:-4].isdigit())
    depth = {'lA': 1, 'lB': 1, 'lC': 1, 'lD': 1, 'lAB': 2, 'lCD': 2, 'lABCD': 4}
    stats = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0, 0, 0, 0]))   # win -> depth -> [paid, multi, full, full_composed, composed]
    boards = collections.Counter()
    for N in nums:
        if N < 1:
            continue
        b = N - 1
        try:
            msgs = {agent_id(m.get('label')): m for m in json.load(open(os.path.join(logs, 'log%d.txt' % N)))}
        except Exception:
            continue
        w = b // W
        boards[w] += 1

        def tree(se, j, seen, d=0, contrib=None):
            m = msgs.get(se)
            if not m or d > 10:
                return
            tr = m.get('trades', [])
            if j >= len(tr):
                return
            sec = section(tr, j)
            seen.add(se)
            if contrib is not None:
                for x in sec:
                    if x.get('type') in ('sell', 'construct') and LEAF.search(str(x.get('item', ''))):
                        contrib[se].add(LEAF.search(str(x.get('item', ''))).group(0))
            for x in sec[1:]:
                if x.get('type') == 'buy' and x.get('chosen') is not None:
                    offs = x.get('offers') or []
                    if x['chosen'] < len(offs):
                        o = offs[x['chosen']]
                        tree(int(o['agent']), int(o['trades'][0]), seen, d + 1, contrib)

        for aid, m in msgs.items():
            if aid is None or aid >= H:
                continue
            for t in m.get('trades', []):
                if t.get('type') != 'buy' or t.get('chosen') is None or not t.get('cd_good'):
                    continue
                g = t['cd_good']
                if g not in depth:
                    continue
                try:
                    price = float(t.get('price', 0) or 0)
                except (TypeError, ValueError):
                    price = 0.0
                if price <= 0 or t.get('_zeropay'):
                    continue                               # her pool law: human-PAID deliveries only
                offs = t.get('offers') or []
                if t['chosen'] >= len(offs):
                    continue
                o = offs[t['chosen']]
                seen = set(); contrib = collections.defaultdict(set)
                tree(int(o['agent']), int(o['trades'][0]), seen, 0, contrib)
                composed = sum(1 for v in contrib.values() if v) >= 2
                sc = 0.0
                for tt in t.get('tests', []) or []:
                    for r in tt.get('results', []) or []:
                        if r.get('offer') == t['chosen']:
                            sc = max(sc, float(r.get('score', 0) or 0))
                s = stats[w][depth[g]]
                s[0] += 1
                multi = len(seen) >= 2
                s[1] += 1 if multi else 0
                s[4] += 1 if composed else 0
                if sc >= 0.99:
                    s[2] += 1
                    s[3] += 1 if composed else 0
    print('%s: complexification read (human-paid deliveries by the good asked for; logged boards only, windows of %d)'
          % (os.path.basename(run), W))
    for w in sorted(stats):
        nb = boards[w] or 1
        parts = []
        for d, name in ((1, 'one-layer'), (2, 'two-layer'), (4, 'whole')):
            p, mu, f, fc, c = stats[w][d]
            parts.append('%s: paid %.1f/board, through 2+ agents (%d/%d), ASSEMBLED by 2+ agents %.2f/board (%d/%d = %.0f%%), fully right %.2f/board of which assembled (%d/%d)'
                         % (name, p / nb, mu, p, c / nb, c, p, 100.0 * c / p if p else 0.0, f / nb, fc, f))
        print('  boards %d-%d (%d logged) | %s' % (w * W, w * W + W - 1, boards[w], ' | '.join(parts)))


if __name__ == '__main__':
    main()
