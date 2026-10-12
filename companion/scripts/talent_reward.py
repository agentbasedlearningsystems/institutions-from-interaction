#!/usr/bin/env python3
"""talent_reward.py <run_dir> [--window 2000] [--humans 11] [--min-deliveries 5]   (7 Oct 2026)

The SDSS corruption index (her 11 Aug staging; AIA_LLM_20260821: talent against reward) read on a ladder world, by
window of boards: TALENT = a learner's mean delivery score as a SELLER, from the logged boards (every chosen offer in a
buy row of any agent, the test result of that offer); REWARD = the learner's income over the same boards from
payments.csv (time;buyer;seller;price; every board). INDEX = 1 - Spearman(talent, reward) over learners with at least
--min-deliveries scored deliveries in the window: 0 when pay follows quality, 1 when they are unrelated, above 1 when
they run opposite. Also printed: the top earner's share of learners' income and its talent rank, and the suppliers'
view (learners whose deliveries went to other learners: their mean score and income share), so a monopoly that pays
its right-part suppliers 4 a part while taking 240 a whole shows as a low rank for high talent.
"""
import collections
import json
import os
import re
import sys


def agent_id(label):
    m = re.search(r'Agent (\d+)', str(label or ''))
    return int(m.group(1)) if m else None


def spearman(x, y):
    n = len(x)
    if n < 3:
        return None
    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i]); r = [0.0] * n; i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0 + 1
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = (sum((rx[i] - mx) ** 2 for i in range(n)) * sum((ry[i] - my) ** 2 for i in range(n))) ** 0.5
    return num / den if den > 0 else None


def main():
    a = sys.argv[1:]
    run = a[0].rstrip('/')
    W = int(a[a.index('--window') + 1]) if '--window' in a else 2000
    H = int(a[a.index('--humans') + 1]) if '--humans' in a else 11
    MIN = int(a[a.index('--min-deliveries') + 1]) if '--min-deliveries' in a else 5
    logs = os.path.join(run, 'logs')
    nums = sorted(int(f[3:-4]) for f in os.listdir(logs) if f.startswith('log') and f[3:-4].isdigit())
    scores = collections.defaultdict(lambda: collections.defaultdict(list))      # win -> seller -> [score]
    to_learners = collections.defaultdict(lambda: collections.defaultdict(list)) # win -> seller -> [score] for deliveries to learners
    for N in nums:
        if N < 1:
            continue
        b = N - 1; w = b // W
        try:
            msgs = json.load(open(os.path.join(logs, 'log%d.txt' % N)))
        except Exception:
            continue
        for m in msgs:
            bid = agent_id(m.get('label'))
            for t in m.get('trades', []):
                if t.get('type') != 'buy' or t.get('chosen') is None:
                    continue
                offs = t.get('offers') or []
                if t['chosen'] >= len(offs):
                    continue
                seller = int(offs[t['chosen']]['agent'])
                sc = None
                for tt in t.get('tests', []) or []:
                    for r in tt.get('results', []) or []:
                        if r.get('offer') == t['chosen']:
                            sc = max(sc or 0.0, float(r.get('score', 0) or 0))
                if sc is None:
                    continue
                scores[w][seller].append(sc)
                if bid is not None and bid >= H:
                    to_learners[w][seller].append(sc)
    income = collections.defaultdict(lambda: collections.defaultdict(float))     # win -> seller -> tokens
    from_learners = collections.defaultdict(lambda: collections.defaultdict(float))
    with open(os.path.join(run, 'payments.csv')) as fh:
        next(fh)
        for line in fh:
            p = line.rstrip('\n').split(';')
            if len(p) < 4:
                continue
            try:
                t = float(p[0]); buyer = int(p[1]); seller = int(p[2]); price = float(p[3])
            except ValueError:
                continue
            if price <= 0 or seller < H:
                continue
            w = int(t) // W
            income[w][seller] += price
            if buyer >= H:
                from_learners[w][seller] += price
    print('%s: talent against reward (SDSS index) by window of %d boards; talent = mean delivery score as seller on logged boards, reward = income from payments.csv'
          % (os.path.basename(run), W))
    for w in sorted(set(scores) | set(income)):
        sellers = [s for s in scores[w] if len(scores[w][s]) >= MIN and s >= H]
        tal = [sum(scores[w][s]) / len(scores[w][s]) for s in sellers]
        rew = [income[w].get(s, 0.0) for s in sellers]
        rho = spearman(tal, rew)
        tot = sum(income[w].values()) or 1.0
        top = max(income[w], key=lambda s: income[w][s]) if income[w] else None
        top_share = income[w][top] / tot if top is not None else 0.0
        top_rank = None
        if top in sellers and len(sellers) > 1:
            top_rank = 1 + sum(1 for s in sellers if (sum(scores[w][s]) / len(scores[w][s])) > (sum(scores[w][top]) / len(scores[w][top])))
        sup = [s for s in sellers if len(to_learners[w].get(s, [])) >= MIN]
        sup_tal = (sum(sum(to_learners[w][s]) / len(to_learners[w][s]) for s in sup) / len(sup)) if sup else None
        sup_share = sum(income[w].get(s, 0.0) for s in sup) / tot if sup else 0.0
        print('  boards %d-%d | learners with %d+ scored deliveries %d | Spearman(talent, reward) %s | INDEX 1-rho %s | top earner %s share %.2f, talent rank %s of %d'
              ' | suppliers to learners %d: mean part score %s, income share %.2f | learner-to-learner income share %.3f'
              % (w * W, w * W + W - 1, MIN, len(sellers), ('%.2f' % rho) if rho is not None else '-', ('%.2f' % (1 - rho)) if rho is not None else '-',
                 top, top_share, top_rank if top_rank is not None else '-', len(sellers), len(sup), ('%.2f' % sup_tal) if sup_tal is not None else '-', sup_share,
                 sum(from_learners[w].values()) / tot))


if __name__ == '__main__':
    main()
