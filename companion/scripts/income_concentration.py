#!/usr/bin/env python3
"""income_concentration.py <run_dir> <last_n_logs>   (7 Oct 2026, for her question: hierarchy against distribution)

Over the last N logged boards: each human buy row that settled (chosen offer, a recorded price) pays its price to the
seller it drew (first-level income, as buying_pays.py reads it); learner buy rows likewise pay learners. Per learner
(ids 11 and up, every one counted, zero included): income from humans, income from learners, and income from humans'
WHOLE purchases (cd_good hABCD). Prints one line: the Gini of total income over the learners present, the top-1 and
top-3 shares, how many learners had any whole income, the top whole-seller's share of whole income, and the share of
learner income that came from other learners (the part trade). Dead learners (birth-death worlds) count only if they
appear in the window.
"""
import collections
import glob
import json
import re
import sys

run, N = sys.argv[1].rstrip('/'), int(sys.argv[2])
logs = sorted(glob.glob(run + '/logs/log*.txt'), key=lambda f: int(re.sub(r'\D', '', f.split('/')[-1]) or 0))[-N:]


def aid(m):
    mm = re.search(r'RoleSim Agent (\d+)', m.get('label', ''))
    return int(mm.group(1)) if mm else None


human_inc = collections.Counter(); learner_inc = collections.Counter(); whole_inc = collections.Counter()
present = set(); first = last = None
for f in logs:
    try:
        msgs = json.load(open(f))
    except Exception:
        continue
    b = int(re.sub(r'\D', '', f.split('/')[-1]) or 0)
    first = b if first is None else first; last = b
    for m in msgs:
        a = aid(m)
        if a is None:
            continue
        if a >= 11 and m.get('trades'):
            present.add(a)
        for t in m.get('trades', []):
            if t.get('type') == 'stop':
                break
            if t.get('type') != 'buy' or t.get('chosen') is None or t.get('_refused'):
                continue
            offs = t.get('offers') or []
            if not (0 <= t['chosen'] < len(offs)):
                continue
            p = t.get('price')
            if p is None:
                continue
            seller = int(offs[t['chosen']]['agent'])
            if seller < 11 or float(p) <= 0:        # a zero payment (a wrong delivery) is not income and makes no seller
                continue
            if a < 11:
                human_inc[seller] += float(p)
                if t.get('cd_good') == 'hABCD':
                    whole_inc[seller] += float(p)
            else:
                learner_inc[seller] += float(p)


def gini(xs):
    xs = sorted(xs); n = len(xs); s = sum(xs)
    if n == 0 or s == 0:
        return float('nan')
    return sum((2 * (i + 1) - n - 1) * x for i, x in enumerate(xs)) / (n * s)


learners = sorted(present)
tot = {a: human_inc.get(a, 0.0) + learner_inc.get(a, 0.0) for a in learners}
vals = sorted(tot.values(), reverse=True); s = sum(vals) or 1.0
wh = sorted(whole_inc.values(), reverse=True); ws = sum(wh)
print('%s boards %s-%s | learners %d | gini %.2f | top1 %.2f top3 %.2f | whole sellers %d, top whole-seller share %s | '
      'income from learners (%.0f/%.0f) = %.2f | whole income %.0f of %.0f'
      % (run.split('/')[-1], first, last, len(learners), gini(vals), vals[0] / s if vals else 0, sum(vals[:3]) / s if vals else 0,
         len(wh), ('%.2f' % (wh[0] / ws)) if ws else '-', sum(learner_inc.values()), s, sum(learner_inc.values()) / s, ws, s))
