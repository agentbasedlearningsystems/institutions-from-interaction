"""Arms B/C declared analyses. Uptake by form (Fisher), spend-vs-income
(market), stuck-group contrasts within tier (stratified pairs only)."""
import json, glob, collections, random, sys
from math import comb
def read(run):
    cast = json.load(open(f'truth/het_cast_{run}.json'))
    tier = {k: v.split('-')[1] for k, v in cast.items()}
    income = collections.Counter(); spend = collections.Counter(); ss = collections.Counter()
    scores = collections.defaultdict(list); stuck = {}; periods = 0
    for line in open(f'runs/{run}/log.jsonl'):
        try: r = json.loads(line)
        except Exception: continue
        k = r.get('kind')
        if k == 'period_end': periods = max(periods, r.get('period', 0) or r.get('round', 0) or 0)
        elif k == 'sale':
            income[r['seller']] += r.get('price', 0)
            if (r.get('score') or 0) > 0: scores[r['seller']].append(r['score'])
        elif k == 'agent_trade':
            income[r['seller']] += r.get('paid', 0); spend[r['buyer']] += r.get('paid', 0)
        elif k == 'orch_submit' and (r.get('score') or 0) > 0:
            scores['A%s' % r['worker']].append(r['score'])
        elif k == 'sign_enforced':
            a = r.get('agent') or ('A%s' % r.get('worker'))
            ss[a] += r.get('cost', 0)
            sg = r.get('sign')
            if isinstance(sg, list) and len(sg) == 8 and abs(abs(sg[7]) - 1.0) < 1e-9:
                stuck.setdefault(a, sg[7])
    net = {a: income.get(a, 0) - spend.get(a, 0) for a in tier}
    prod = {a: sum(scores.get(a, [])) for a in tier}
    return dict(tier=tier, net=net, prod=prod, signspend=dict(ss), stuck=stuck, periods=periods)
def fisher(a, b, c, d):
    n = a + b + c + d; r1 = a + b; c1 = a + c
    return sum(comb(c1, x) * comb(n - c1, r1 - x) for x in range(a, min(r1, c1) + 1)) / comb(n, r1)
def pearson(x, y):
    n = len(x); mx, my = sum(x)/n, sum(y)/n
    num = sum((a-mx)*(b-my) for a, b in zip(x, y))
    den = (sum((a-mx)**2 for a in x) * sum((b-my)**2 for b in y)) ** 0.5
    return num/den if den else 0.0
def main(runs_by):
    print('=== B: costly-sign uptake by form ===')
    mk_b = mk_n = tm_b = tm_n = 0
    for form, runs in (('market', runs_by['Bm']), ('team', runs_by['Bt'])):
        for run in runs:
            try: d = read(run)
            except FileNotFoundError: continue
            for a in d['tier']:
                bought = d['signspend'].get(a, 0) > 0
                if form == 'market': mk_b += bought; mk_n += 1
                else: tm_b += bought; tm_n += 1
    print(f'market: {mk_b}/{mk_n} bought | team: {tm_b}/{tm_n} | Fisher one-sided p = {fisher(mk_b, mk_n-mk_b, tm_b, tm_n-tm_b):.2e}')
    xs, ys = [], []
    for run in runs_by['Bm']:
        try: d = read(run)
        except FileNotFoundError: continue
        for a in d['tier']: xs.append(d['signspend'].get(a, 0)); ys.append(d['net'][a])
    if xs: print(f'B market spend-vs-net pearson r = {pearson(xs, ys):.3f} (n={len(xs)})')
    print('=== C: stuck-group within-tier contrast (stratified pairs) ===')
    for form, key, val in (('market', 'Cm', 'net'), ('team', 'Ct', 'prod')):
        g1, g2 = [], []
        for run in runs_by[key]:
            try: d = read(run)
            except FileNotFoundError: continue
            if not d['stuck']: continue
            for t in ('haiku', 'sonnet', 'opus'):
                members = [a for a in d['tier'] if d['tier'][a] == t]
                v = d[val]
                order = sorted(members, key=lambda a: v.get(a, 0), reverse=True)
                rk = {a: i+1 for i, a in enumerate(order)}
                for a in members:
                    (g1 if d['stuck'].get(a) == 1.0 else g2).append(rk[a])
        if g1 and g2:
            d0 = sum(g1)/len(g1) - sum(g2)/len(g2)
            allv = g1 + g2; n1 = len(g1); rng = random.Random(2); cnt = 0
            for _ in range(20000):
                rng.shuffle(allv)
                if abs(sum(allv[:n1])/n1 - sum(allv[n1:])/len(allv[n1:])) >= abs(d0): cnt += 1
            print(f'{form}: +1 mean within-tier rank {sum(g1)/len(g1):.2f} vs -1 {sum(g2)/len(g2):.2f} (diff {d0:+.2f}, two-sided p = {cnt/20000:.3f}, n={len(g1)}+{len(g2)})')
        else:
            print(f'{form}: insufficient stratified data yet')
if __name__ == '__main__':
    full = dict(Bm=['cw_hetB_m1', 'cw_hetB_m2', 'cw_hetB_m3'], Bt=['cw_hetB_t1', 'cw_hetB_t2', 'cw_hetB_t3'],
                Cm=['cw_hetC_m2', 'cw_hetC_m3'], Ct=['cw_hetC_t2', 'cw_hetC_t3'])
    banked = dict(Bm=['cw_hetB_m1', 'cw_hetB_m2'], Bt=['cw_hetB_t1', 'cw_hetB_t2'],
                  Cm=['cw_hetC_m2'], Ct=['cw_hetC_t2'])
    main(banked if len(sys.argv) > 1 and sys.argv[1] == 'banked' else full)
