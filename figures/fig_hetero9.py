"""Merged mixed-cast figure for the CSSSA paper (replaces fig_armA.pdf
content, keeps the fig:armA label): (a) income rank by hidden tier in all
nine market societies, (b) the coordinator null on the arm A casts,
(c) arm B market costly-sign spend vs net income.
Prints the per-society top-earner audit used by the caption."""
import json
import collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TIER_COLOR = {'opus': '#c62828', 'sonnet': '#f9a825', 'haiku': '#78909c'}
TIER_LABEL = {'opus': 'large', 'sonnet': 'middle', 'haiku': 'small'}
OUT = ('./fig_armA9.pdf')


def read(run):
    cast = json.load(open(f'truth/het_cast_{run}.json'))
    tier = {k: v.split('-')[1] for k, v in cast.items()}
    income = collections.Counter()
    spend = collections.Counter()
    ss = collections.Counter()
    inter_in = collections.Counter()
    buyers_of = collections.defaultdict(set)
    scores = collections.defaultdict(list)
    for line in open(f'runs/{run}/log.jsonl'):
        try:
            r = json.loads(line)
        except Exception:
            continue
        k = r.get('kind')
        if k == 'sale':
            income[r['seller']] += r.get('price', 0)
        elif k == 'agent_trade':
            income[r['seller']] += r.get('paid', 0)
            spend[r['buyer']] += r.get('paid', 0)
            inter_in[r['seller']] += r.get('paid', 0)
            buyers_of[r['seller']].add(r['buyer'])
        elif k == 'orch_submit' and (r.get('score') or 0) > 0:
            scores['A%s' % r['worker']].append(r['score'])
        elif k == 'sign_enforced':
            a = r.get('agent') or ('A%s' % r.get('worker'))
            ss[a] += r.get('cost', 0)
    net = {a: income.get(a, 0) - spend.get(a, 0) for a in tier}
    prod = {a: sum(scores.get(a, [])) for a in tier}
    return dict(tier=tier, net=net, prod=prod, signspend=dict(ss),
                inter=inter_in, buyers=buyers_of)


M_RUNS = [('A1', 'cw_hetA_m1'), ('A2', 'cw_hetA_m2'), ('A3', 'cw_hetA_m3'),
          ('B1', 'cw_hetB_m1'), ('B2', 'cw_hetB_m2'), ('B3', 'cw_hetB_m3'),
          ('C1', 'cw_hetC_m1'), ('C2', 'cw_hetC_m2'), ('C3', 'cw_hetC_m3')]
T_RUNS = [('T1', 'cw_hetA_t1'), ('T2', 'cw_hetA_t2'), ('T3', 'cw_hetA_t3')]

fig, axes = plt.subplots(
    1, 3, figsize=(10.5, 3.2),
    gridspec_kw={'width_ratios': [1.9, 1.0, 1.0]})
axA, axT, axB = axes

JIT = {'opus': -0.18, 'sonnet': 0.0, 'haiku': 0.18}
for i, (lab, run) in enumerate(M_RUNS):
    d = read(run)
    order = sorted(d['net'], key=d['net'].get, reverse=True)
    rk = {a: j + 1 for j, a in enumerate(order)}
    top = order[0]
    t_top = d['tier'][top]
    inter_lead = max(d['inter'], key=d['inter'].get) if d['inter'] else None
    print(f"{lab} top earner: {top} tier={t_top} net={d['net'][top]:.0f} "
          f"| top intermediate seller: {inter_lead} "
          f"(agent buyers: {len(d['buyers'].get(top, set()))}, "
          f"intermediate share of top income: "
          f"{d['inter'].get(top, 0) / max(d['net'][top] + 1e-9, 1e-9):.2f})")
    for a in d['tier']:
        t = d['tier'][a]
        axA.plot(i + JIT[t], rk[a], 'o', ms=5,
                 color=TIER_COLOR[t], alpha=0.85, mew=0)
axA.axvline(2.5, color='0.85', lw=0.8)
axA.axvline(5.5, color='0.85', lw=0.8)
axA.set_xticks(range(9))
axA.set_xticklabels([lab for lab, _ in M_RUNS], fontsize=8)
axA.set_ylim(11.6, 0.4)
axA.set_yticks([1, 3, 5, 7, 9, 11])
axA.set_ylabel('income rank in society (1 = top)', fontsize=8)
axA.set_title('(a) market societies: income rank by hidden tier',
              fontsize=8.5)
axA.text(0.03, 0.05,
         'mean corr(tier, rank) $-0.81$\npermutation $p \\leq 10^{-5}$',
         transform=axA.transAxes, fontsize=7.5, va='bottom')
for t in ('opus', 'sonnet', 'haiku'):
    axA.plot([], [], 'o', ms=5, color=TIER_COLOR[t], label=TIER_LABEL[t])
axA.legend(fontsize=7, loc='lower right', frameon=False,
           handletextpad=0.2, borderaxespad=0.2)

for i, (lab, run) in enumerate(T_RUNS):
    d = read(run)
    for a in d['tier']:
        t = d['tier'][a]
        axT.plot(i + JIT[t], d['prod'][a], 'o', ms=5,
                 color=TIER_COLOR[t], alpha=0.85, mew=0)
axT.set_xticks(range(3))
axT.set_xticklabels([lab for lab, _ in T_RUNS], fontsize=8)
axT.set_ylabel('score-mass produced', fontsize=8)
axT.set_title('(b) coordinator, same casts', fontsize=8.5)
axT.text(0.05, 0.92, 'rank gap 0.12, $p = 0.47$',
         transform=axT.transAxes, fontsize=7.5, va='top')

n_pts = 0
for lab, run in M_RUNS[3:6]:
    d = read(run)
    for a in d['tier']:
        t = d['tier'][a]
        axB.plot(d['signspend'].get(a, 0), d['net'][a], 'o', ms=5,
                 color=TIER_COLOR[t], alpha=0.85, mew=0)
        n_pts += 1
axB.set_xlabel('costly-sign spend', fontsize=8)
axB.set_ylabel('net income', fontsize=8)
axB.set_title('(c) arm B market: costly display', fontsize=8.5)
axB.text(0.95, 0.92,
         f'$r = -0.26$, $n = {n_pts}$\nbuyers 18/33; team 0/33',
         transform=axB.transAxes, fontsize=7.5, va='top', ha='right')

for ax in axes:
    ax.tick_params(labelsize=7.5)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.6)
fig.savefig(OUT)
print('wrote', OUT)
