#!/usr/bin/env python3
"""ml_composition.py - complexification in the machine-learning ladder, by her definition (7 Oct 2026): deliveries that
HUMANS PAID FOR (payment > 0, no zero-pay marker), split by the good the human's row asked for (its cd_good: emb, lemb,
clf and the B-corpus twins), and whether the product was ASSEMBLED by more than one agent.

Products are rebuilt as in cd_composition.py (her 11 Sep ruling: the buyer's chosen offer names the seller and the sold
row; the product is that row up to the next sell or stop; a settled buy inside points at another agent's row; recurse).
A RUNG is the kind of an item's path (vectorSpace, labeler, classifier); an agent contributes a rung when a sell or
construct row of its section names one. "Assembled" = at least two agents each contribute a rung (resale chains do not
count). In this world a score is an F1 and never 1.0, so "scoring" (score > 0) replaces "fully right".

Per window of boards (logged boards only), per good:
  paid       human-paid deliveries per logged board
  through    of those, passing through two or more agents (n/d)
  ASSEMBLED  of those, assembled by two or more agents (n/d = share)
  scoring    paid deliveries with score > 0, of which assembled (n/d)
Usage: ml_composition.py <run_dir> [--window 2000] [--humans 11]
"""
import collections
import json
import os
import re
import sys

RUNGS = ('vectorSpace', 'labeler', 'classifier')


def agent_id(label):
    m = re.search(r'Agent (\d+)', str(label or ''))
    return int(m.group(1)) if m else None


def section(tr, j):
    """The rows of the product sold at row j: that row up to the next sell or stop."""
    out = [tr[j]]
    k = j + 1
    while k < len(tr) and tr[k].get('type') not in ('sell', 'stop') and not str(tr[k].get('item', '')).startswith('stop'):
        out.append(tr[k]); k += 1
    return out


def rung_of(item):
    head = str(item or '').split('_')[0]
    return head if head in RUNGS else None


def main():
    a = sys.argv[1:]
    run = a[0].rstrip('/')
    W = int(a[a.index('--window') + 1]) if '--window' in a else 2000
    H = int(a[a.index('--humans') + 1]) if '--humans' in a else 11
    logs = os.path.join(run, 'logs')
    nums = sorted(int(f[3:-4]) for f in os.listdir(logs) if f.startswith('log') and f[3:-4].isdigit())
    stats = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0, 0]))   # win -> good -> [paid, through, assembled, scoring, scoring_assembled, sell_only, own_constructs]
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
                    if x.get('type') in ('sell', 'construct'):
                        r = rung_of(x.get('_named') or x.get('item'))
                        if r:
                            contrib[se].add(r)
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
                g = str(t['cd_good'])
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
                assembled = sum(1 for v in contrib.values() if v) >= 2
                # how the SELLER's own section is built: the sell row alone (the designated defaults fill the parts),
                # or with its own construct rows (7 Oct: her capacity-arm question needs this)
                sm = msgs.get(int(o['agent'])); sec0 = section(sm.get('trades', []), int(o['trades'][0])) if sm and int(o['trades'][0]) < len(sm.get('trades', [])) else []
                sell_only = len(sec0) == 1
                own_constructs = any(x.get('type') == 'construct' for x in sec0[1:])
                sc = 0.0
                for tt in t.get('tests', []) or []:
                    for r in tt.get('results', []) or []:
                        if r.get('offer') == t['chosen']:
                            sc = max(sc, float(r.get('score', 0) or 0))
                s = stats[w][g]
                s[0] += 1
                s[1] += 1 if len(seen) >= 2 else 0
                s[2] += 1 if assembled else 0
                s[5] += 1 if sell_only else 0
                s[6] += 1 if (own_constructs and not assembled) else 0
                if sc > 0:
                    s[3] += 1
                    s[4] += 1 if assembled else 0
    print('%s: ML complexification read (human-paid deliveries by the good asked for; logged boards only, windows of %d)'
          % (os.path.basename(run), W))
    goods_order = ['emb', 'embB', 'lemb', 'lembB', 'clf', 'clfB']
    for w in sorted(stats):
        nb = boards[w] or 1
        parts = []
        for g in sorted(stats[w], key=lambda x: (goods_order.index(x) if x in goods_order else 99, x)):
            p, th, asm, sc, sca, so, oc = stats[w][g]
            parts.append('%s: paid %.1f/board, through 2+ agents (%d/%d), ASSEMBLED by 2+ agents (%d/%d = %.0f%%), scoring %d of which assembled (%d/%d), seller section = sell row alone (defaults fill) (%d/%d), own constructs (%d/%d)'
                         % (g, p / nb, th, p, asm, p, 100.0 * asm / p if p else 0.0, sc, sca, sc, so, p, oc, p))
        print('  boards %d-%d (%d logged) | %s' % (w * W, w * W + W - 1, boards[w], ' | '.join(parts)))


if __name__ == '__main__':
    main()
