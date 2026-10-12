#!/usr/bin/env python3
"""cipher_reads.py - leaf-level reads for a cipher world (C1), from its records.

Products are rebuilt exactly as program_shape.py does (her 2026-09-11 ruling: the
buyer's chosen offer names the seller and the sold row; the product is that row up to
the next sell/stop; a settled buy inside points at another agent's row; recurse).

Reads:
  A  learning curve: per generation, share of human boards served at or above a
     right layer, mean human score, share of human-paid deliveries that span more
     than one agent, share of full (1.0) deliveries.
  B  first full delivery per want: board, the leaves in the product, the agents.
     "two agents" = hiring; "one agent" = built alone.
  C  roles by LEAF over the last --last boards: per learner, its dominant sold leaf
     among human-paid sells, the share, income; how many learners are specialists.
  D  parts: learner-to-learner settlements per generation and what they paid.

Usage: cipher_reads.py <run_dir> --settles <settles.csv> [--gen 19] [--last 200] [--humans 7]
"""
import collections
import csv
import json
import os
import re
import sys


def agent_id(label):
    m = re.search(r'RoleSim Agent (\d+)', label or '')
    return int(m.group(1)) if m else None


def section(tr, j):
    k = j + 1
    while k < len(tr) and tr[k].get('type') not in ('stop', 'sell') \
            and not str(tr[k].get('item', '')).startswith('stop'):
        k += 1
    return tr[j:k]


def main():
    run = sys.argv[1].rstrip('/')
    args = sys.argv[2:]
    def opt(name, default):
        return args[args.index(name) + 1] if name in args else default
    settles = opt('--settles', None)
    G = int(opt('--gen', 19))
    last = int(opt('--last', 200))
    H = int(opt('--humans', 7))

    logs = {}
    def msgs(b):
        if b not in logs:
            try:
                logs[b] = {agent_id(m.get('label')): m
                           for m in json.load(open(os.path.join(run, 'logs', 'log%d.txt' % (b + 1))))}
            except Exception:
                logs[b] = {}
        return logs[b]

    def tree(b, se, j, seen, leaves, depth=0):
        m = msgs(b).get(se)
        if not m or depth > 10:
            return
        tr = m.get('trades', [])
        if j >= len(tr):
            return
        sec = section(tr, j)
        seen.add(se)
        for x in sec:
            if x.get('type') in ('sell', 'construct'):
                leaves.append(str(x.get('item', '')).split('_stop')[0])
        for x in sec[1:]:
            if x.get('type') == 'buy' and x.get('chosen') is not None:
                offs = x.get('offers') or []
                if x['chosen'] < len(offs):
                    o = offs[x['chosen']]
                    tree(b, int(o['agent']), int(o['trades'][0]), seen, leaves, depth + 1)

    def human_product(b, bu, se):
        """leaves and agents of what human bu bought from seller se on board b"""
        hm = msgs(b).get(bu)
        if not hm:
            return None, None
        for row in hm.get('trades', []):
            if row.get('type') == 'buy' and row.get('chosen') is not None:
                offs = row.get('offers') or []
                ch = row['chosen']
                if ch < len(offs) and int(offs[ch].get('agent')) == se:
                    seen, leaves = set(), []
                    tree(b, se, int(offs[ch]['trades'][0]), seen, leaves)
                    return leaves, seen
        return None, None

    # settlements
    S = []
    for r in csv.DictReader(open(settles), delimiter=';'):
        S.append(dict(b=int(float(r['time'])), bu=int(r['buyer']), sc=float(r['score']), paid=float(r['paid']), mid=float(r['midpoint'])))
    lastb = max(r['b'] for r in S)
    # payments: (board, buyer) -> seller
    pay = {}
    for r in csv.DictReader(open(os.path.join(run, 'payments.csv')), delimiter=';'):
        try:
            if float(r['price']) > 0:
                pay[(int(float(r['time'])), int(r['buyer']))] = int(r['seller'])
        except (ValueError, KeyError):
            pass
    cfg = json.load(open([os.path.join(run, f) for f in os.listdir(run) if f.endswith('.json')][0]))
    labels = {i: e.get('label', '') for i, e in enumerate(cfg['blackboard'])}
    depth_of = {i: int(re.search(r'(\d)-layer', labels[i]).group(1)) if re.search(r'(\d)-layer', labels[i]) else 1 for i in labels}

    # A. learning curve
    ngen = (lastb + 1) // G
    print('run %s: boards 0..%d, %d generations of %d' % (os.path.basename(run), lastb, ngen, G))
    print('A. per generation: human boards served >= 1/depth (a right layer) | mean human score | full (1.0) deliveries | multi-agent among human-paid')
    for g in range(ngen):
        b0, b1 = g * G, (g + 1) * G - 1
        hs = [r for r in S if b0 <= r['b'] <= b1 and r['bu'] < H]
        right = [r for r in hs if r['sc'] >= 0.99 / depth_of[r['bu']]]
        full = [r for r in hs if r['sc'] >= 0.99]
        multi = tot = 0
        if g % 5 == 4 or g == ngen - 1 or g < 3:
            for r in hs:
                if r['paid'] <= 0:
                    continue
                se = pay.get((r['b'], r['bu']))
                if se is None:
                    continue
                leaves, agents = human_product(r['b'], r['bu'], se)
                if agents is None:
                    continue
                tot += 1
                multi += 1 if len(agents) > 1 else 0
            ms = '%d/%d' % (multi, tot)
        else:
            ms = '-'
        print('  gen %3d: right (%d/%d) = %.0f%% | mean score %.3f | full %d | multi-agent %s'
              % (g, len(right), len(hs), 100.0 * len(right) / max(1, len(hs)),
                 sum(r['sc'] for r in hs) / max(1, len(hs)), len(full), ms))

    # A2. service at 1.0 by depth, per generation (the ladder read)
    depths = sorted(set(depth_of.values()))
    print('A2. per generation: share of buyer boards served in FULL (score 1.0), by depth of want')
    for g in range(ngen):
        b0, b1 = g * G, (g + 1) * G - 1
        if not (g % 5 == 4 or g < 3 or g == ngen - 1):
            continue
        parts = []
        for d in depths:
            hs = [r for r in S if b0 <= r['b'] <= b1 and r['bu'] < H and depth_of[r['bu']] == d]
            nb = sum(1 for bu in range(H) if depth_of[bu] == d) * G
            full = sum(1 for r in hs if r['sc'] >= 0.99)
            parts.append('%d-layer (%d/%d) = %.0f%%' % (d, full, nb, 100.0 * full / max(1, nb)))
        print('  gen %3d: %s' % (g, ' | '.join(parts)))

    # B. first full delivery per want
    print('B. first full (1.0) delivery per want, and the first at 1/depth:')
    for bu in range(H):
        hr = sorted((r for r in S if r['bu'] == bu), key=lambda r: r['b'])
        d = depth_of[bu]
        firsts = [('right', next((r for r in hr if r['sc'] >= 0.99 / d), None)),
                  ('full', next((r for r in hr if r['sc'] >= 0.99), None))]
        out = ['  buyer %d %s' % (bu, labels[bu][:48])]
        for tag, r in firsts:
            if r is None:
                out.append('%s: none' % tag)
                continue
            se = pay.get((r['b'], r['bu']))
            leaves, agents = human_product(r['b'], bu, se) if se is not None else (None, None)
            out.append('%s: board %d score %.3f seller %s agents %s leaves %s'
                       % (tag, r['b'], r['sc'], se, sorted(agents) if agents else '?', leaves))
        print(' | '.join(out))
    # all full deliveries: built alone vs hired
    alone = hired = 0
    for r in S:
        if r['bu'] < H and r['sc'] >= 0.99 and depth_of[r['bu']] >= 2:
            se = pay.get((r['b'], r['bu']))
            if se is None:
                continue
            leaves, agents = human_product(r['b'], r['bu'], se)
            if agents is None:
                continue
            if len(agents) > 1:
                hired += 1
            else:
                alone += 1
    print('  full deliveries on 2- and 3-layer wants: built by one agent %d, spanning several %d' % (alone, hired))

    # C. roles by leaf, last N boards
    b0 = lastb - last + 1
    per = collections.defaultdict(lambda: dict(sells=collections.Counter(), income=0.0))
    for r in S:
        if r['b'] < b0 or r['bu'] >= H or r['paid'] <= 0:
            continue
        se = pay.get((r['b'], r['bu']))
        if se is None:
            continue
        leaves, agents = human_product(r['b'], r['bu'], se)
        head = leaves[0] if leaves else '?'
        per[se]['sells'][head] += 1
        per[se]['income'] += r['paid']
    print('C. roles by leaf, boards %d-%d: learner | human-paid sells | dominant sold leaf (share) | income' % (b0, lastb))
    spec = 0
    for se in sorted(per):
        c = per[se]['sells']
        n = sum(c.values())
        leaf, k = c.most_common(1)[0]
        if n >= 5 and k / n >= 0.7:
            spec += 1
        print('  %3d | %3d | %-22s (%.2f) | %7.1f' % (se, n, leaf, k / n, per[se]['income']))
    print('  specialists (>=5 paid sells, >=70%% one leaf): %d of %d earning learners' % (spec, len(per)))

    # E. role stability: does a learner's head leaf persist from one board to the next?
    print('E. role stability over the last %d boards: per learner, share of consecutive boards with the SAME head leaf of its first sell section' % last)
    def head_leaf(m):
        tr = m.get('trades', [])
        for x in tr:
            if x.get('type') == 'stop' or str(x.get('item', '')).startswith('stop'):
                break
            if x.get('type') == 'sell':
                return str(x.get('item', '')).split('_stop')[0]
        return None
    stab = {}
    b0 = lastb - last + 1
    prev = {}
    same = collections.Counter(); tot = collections.Counter()
    for b in range(b0, lastb + 1):
        m = msgs(b)
        for a, msg in m.items():
            if a is None or a < H:
                continue
            h = head_leaf(msg)
            if a in prev:
                tot[a] += 1
                if h is not None and h == prev[a]:
                    same[a] += 1
            prev[a] = h
    vals = sorted(((same[a] / tot[a]) if tot[a] else 0.0, a) for a in tot)
    print('  median %.2f | learners above 0.8: %d of %d | lowest: %s' % (vals[len(vals) // 2][0] if vals else 0, sum(1 for v, a in vals if v >= 0.8), len(vals), ', '.join('%d:%.2f' % (a, v) for v, a in vals[:4])))

    # D. parts
    print('D. learner-to-learner settlements per generation: count | paid > 0 | tokens')
    for g in range(ngen):
        bb0, bb1 = g * G, (g + 1) * G - 1
        ls = [r for r in S if bb0 <= r['b'] <= bb1 and r['bu'] >= H]
        if g % 5 == 4 or g < 3 or g == ngen - 1:
            print('  gen %3d: %4d | %4d | %.1f' % (g, len(ls), sum(1 for r in ls if r['paid'] > 0), sum(r['paid'] for r in ls)))


if __name__ == '__main__':
    main()
