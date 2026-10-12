#!/usr/bin/env python3
"""category_meaning.py - do signs form CATEGORIES: groups of persons whose shared sign tells what they do?

Her definition (3 Oct 2026): culture is the existence of categories. The sign has mutual
information with what is done, and a GROUP of persons carries the sign, not one person. The
sign may change over time; what matters is the categories. Three earlier reads do not show
this and are not culture measures: role stability (one learner keeps its job from board to
board; no sign involved), drift (one learner's sign over time), and mutual information between
clustered signs and jobs over messages (high when each learner keeps its own sign and its own
job, which is individual recognition).

Everything is computed within one time window of --window boards (her snapshot law); signs are
free to change between windows.

  what is done  per message, its first sell section (the sell row to the next sell or stop
                row). HEAD = the sell row's item (the decoder sold); PRODUCT = the sorted items
                of the whole section (sell, construct and buy rows).
  units         each (learner, job) pair seen on at least --min-msgs boards of the window, with
                the learner's mean display sign over those boards. Persons, not messages, are
                the units: one learner's hundred messages are one observation of its sign.
  A. cross-person meaning (the culture read)
                each unit's job is guessed from its sign using only OTHER learners' units in
                the window: the job whose other-learner centroid is closest by cosine. A job
                held by one learner alone has no other-learner centroid, so that unit is a
                miss: a category needs two persons. Reads: accuracy (right/units) and the
                mutual information in bits between the true and the guessed job.
  B. clustered meaning (her snapshot dial, at the person level)
                the unit signs clustered by k-means on unit vectors with k = the number of jobs
                present in the window (her 16 Aug k ruling); mutual information in bits
                between cluster and job over units.
  Null for A and B: the jobs permuted among the window's units (--perms draws). Excess =
                observed - null mean; p = share of draws at or above the observed (smallest
                attainable 1/(perms+1)). A personal sign, however stable and however tied to
                its owner's job, cannot be read off other persons, so A and B fall to the null
                under individual recognition and rise only for shared signs.
  C. groups     jobs held by two or more learners in the window: how many learners each; mean
                cosine between different learners' unit signs within a job against across jobs.
  D. own-sign   (contrast, not culture) a single message's job guessed from every OTHER message
                of the window, the learner's own included. Individual recognition passes D.
  role stability (not culture) per learner, the share of consecutive boards with the same HEAD.

Usage: category_meaning.py <run_dir> [--window 100] [--stride 500] [--last 600]
                                      [--min-msgs 5] [--perms 200] [--sold-only] [--json out]
       category_meaning.py --selftest   (planted worlds: the reads must find categories where
                                         they were planted and only there)
"""
import collections
import csv
import json
import math
import os
import re
import sys

import numpy as np


def agent_id(label):
    m = re.search(r'RoleSim Agent (\d+)', label or '')
    return int(m.group(1)) if m else None


def strip(it):
    return str(it).split('_stop')[0]


def first_section(tr):
    for i, x in enumerate(tr):
        it = str(x.get('item', ''))
        if x.get('type') == 'stop' or it.startswith('stop'):
            return None
        if x.get('type') == 'sell':
            k = i + 1
            while k < len(tr) and tr[k].get('type') not in ('stop', 'sell') \
                    and not str(tr[k].get('item', '')).startswith('stop'):
                k += 1
            return tr[i:k]
    return None


def mi_bits(xs, ys):
    n = len(xs)
    if n == 0:
        return 0.0
    cxy = collections.Counter(zip(xs, ys)); cx = collections.Counter(xs); cy = collections.Counter(ys)
    return sum(c / n * math.log2(c * n / (cx[x] * cy[y])) for (x, y), c in cxy.items())


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1.0)


def cross_person(persons, U, jobs):
    """A: guess each unit's job from OTHER learners' units only; (accuracy, mutual information)."""
    labs = sorted(set(jobs), key=str); li = {L: k for k, L in enumerate(labs)}
    pers = sorted(set(persons)); pi = {p: k for k, p in enumerate(pers)}
    S = np.zeros((len(labs), U.shape[1])); N = np.zeros(len(labs))
    SP = np.zeros((len(pers), len(labs), U.shape[1])); NP = np.zeros((len(pers), len(labs)))
    for u, p, j in zip(U, persons, jobs):
        S[li[j]] += u; N[li[j]] += 1; SP[pi[p], li[j]] += u; NP[pi[p], li[j]] += 1
    right, guesses = 0, []
    for u, p, j in zip(U, persons, jobs):
        C = S - SP[pi[p]]; n = N - NP[pi[p]]
        norms = np.linalg.norm(C, axis=1)
        ok = (n > 0) & (norms > 1e-12)
        if not ok.any():
            guesses.append(None)
            continue
        sc = np.where(ok, (C @ u) / np.where(ok, norms, 1.0), -np.inf)
        g = labs[int(np.argmax(sc))]
        guesses.append(g); right += int(g == j)
    return right / len(jobs), mi_bits([str(j) for j in jobs], [str(g) for g in guesses])


def kmeans(U, k, rng, restarts=10, iters=60):
    n = len(U); best = None
    for _ in range(restarts):
        cent = [U[rng.randint(n)]]
        for _ in range(1, k):
            d2 = np.min([((U - c) ** 2).sum(1) for c in cent], axis=0)
            cent.append(U[rng.choice(n, p=d2 / d2.sum())] if d2.sum() > 0 else U[rng.randint(n)])
        C = np.array(cent)
        for _ in range(iters):
            lab = np.argmin(((U[:, None, :] - C[None]) ** 2).sum(2), axis=1)
            newC = np.array([U[lab == c].mean(0) if (lab == c).any() else C[c] for c in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        inertia = float(((U - C[lab]) ** 2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, lab.copy())
    return best[1]


def own_sign(Um, jobs):
    """D: a message's job from every other message of the window, its own learner included."""
    labs = sorted(set(jobs), key=str); li = {L: k for k, L in enumerate(labs)}
    J = np.array([li[j] for j in jobs])
    S = np.zeros((len(labs), Um.shape[1])); N = np.zeros(len(labs))
    np.add.at(S, J, Um); np.add.at(N, J, 1)
    dots = Um @ S.T                                      # messages x jobs
    norms = np.tile(np.linalg.norm(S, axis=1), (len(Um), 1))
    cnt = np.tile(N, (len(Um), 1))
    r = np.arange(len(Um))
    own = dots[r, J]
    dots[r, J] = own - 1.0                               # u.(S_j - u) for unit u
    norms[r, J] = np.sqrt(np.maximum(norms[r, J] ** 2 - 2 * own + 1.0, 0.0))
    cnt[r, J] -= 1
    ok = (cnt > 0) & (norms > 1e-12)
    sc = np.where(ok, dots / np.where(ok, norms, 1.0), -np.inf)
    return float((np.argmax(sc, axis=1) == J).mean())


def role_stability(recs):
    seq = collections.defaultdict(dict)
    for b, p, head, _prod, _s in recs:
        seq[p][b] = head
    vals = []
    for p, d in seq.items():
        pairs = [(d[b], d[b + 1]) for b in d if b + 1 in d]
        if pairs:
            vals.append(sum(1 for a, c in pairs if a == c) / len(pairs))
    return float(np.mean(vals)) if vals else float('nan')


def analyse(recs, min_msgs, perms, rng, label='head'):
    k = 2 if label == 'head' else 3
    by = collections.defaultdict(list)
    for r in recs:
        by[(r[1], r[k])].append(r[4])
    keys = [key for key, v in by.items() if len(v) >= min_msgs]
    if len(keys) < 3 or len(set(key[1] for key in keys)) < 2:
        return None
    persons = [key[0] for key in keys]; jobs = [key[1] for key in keys]
    U = unit(np.array([np.mean(by[key], axis=0) for key in keys]))
    acc, mi = cross_person(persons, U, jobs)
    kk = min(len(set(jobs)), len(keys))
    cl = [int(c) for c in kmeans(U, kk, rng)]
    cmi = mi_bits(cl, [str(j) for j in jobs])
    nacc, nmi, ncmi = [], [], []
    for _ in range(perms):
        pj = [jobs[i] for i in rng.permutation(len(jobs))]
        a, m = cross_person(persons, U, pj)
        nacc.append(a); nmi.append(m); ncmi.append(mi_bits(cl, [str(j) for j in pj]))
    holders = collections.defaultdict(set)
    for p, j in zip(persons, jobs):
        holders[j].add(p)
    multi = sorted((len(s) for s in holders.values() if len(s) >= 2), reverse=True)
    G = U @ U.T; within, across = [], []
    for a in range(len(jobs)):
        for b in range(a + 1, len(jobs)):
            if persons[a] != persons[b]:
                (within if jobs[a] == jobs[b] else across).append(float(G[a, b]))
    Um = unit(np.array([r[4] for r in recs])); jm = [r[k] for r in recs]
    own = own_sign(Um, jm)
    nown = [own_sign(Um, [jm[i] for i in rng.permutation(len(jm))]) for _ in range(20)]
    def p_of(obs, null):
        return (1 + sum(1 for v in null if v >= obs - 1e-12)) / (len(null) + 1)
    return dict(units=len(keys), learners=len(set(persons)), jobs=len(set(jobs)),
                acc=acc, acc_null=nacc, mi=mi, mi_null=nmi, cmi=cmi, cmi_null=ncmi,
                p_acc=p_of(acc, nacc), p_mi=p_of(mi, nmi), p_cmi=p_of(cmi, ncmi),
                holders=multi, within=(float(np.mean(within)) if within else float('nan')), n_within=len(within),
                across=(float(np.mean(across)) if across else float('nan')), n_across=len(across),
                own=own, own_null=float(np.mean(nown)), stab=role_stability(recs))


def line(tag, r):
    if r is None:
        return '  %-14s too few units' % tag
    return ('  %-14s units %2d (learners %2d, jobs %2d) | A cross-person: right (%d/%d) = %.2f vs null %.2f, '
            'MI %.2f bits vs null %.2f (excess %+.2f, p %.3f) | B clustered MI %.2f vs null %.2f (excess %+.2f, p %.3f) '
            '| C jobs with 2+ learners: %s; same-job cosine %.2f (n %d) vs other jobs %.2f (n %d) '
            '| D own-sign %.2f vs null %.2f | role stability %.2f'
            % (tag, r['units'], r['learners'], r['jobs'], round(r['acc'] * r['units']), r['units'], r['acc'],
               np.mean(r['acc_null']), r['mi'], np.mean(r['mi_null']), r['mi'] - np.mean(r['mi_null']), r['p_mi'],
               r['cmi'], np.mean(r['cmi_null']), r['cmi'] - np.mean(r['cmi_null']), r['p_cmi'],
               r['holders'] or 'none', r['within'], r['n_within'], r['across'], r['n_across'],
               r['own'], r['own_null'], r['stab']))


def summary(tag, rs, perms):
    rs = [r for r in rs if r is not None]
    if not rs:
        return '  %s: no window with enough units' % tag
    out = []
    for key, nk in (('mi', 'mi_null'), ('acc', 'acc_null'), ('cmi', 'cmi_null')):
        obs = float(np.mean([r[key] for r in rs]))
        null = np.mean([r[nk] for r in rs], axis=0)       # draw k averaged over windows
        p = (1 + int((null >= obs - 1e-12).sum())) / (len(null) + 1)
        out.append('%s %.2f vs null %.2f (excess %+.2f, p %.3f)' % (key, obs, float(null.mean()), obs - float(null.mean()), p))
    return '  %s, mean of %d windows: %s' % (tag, len(rs), ' | '.join(out))


def plant(kind, rng, N=16, K=8, boards=600, dim=16, noise=0.25, drift=0.02, switch=0.0):
    centers = rng.uniform(-1, 1, (K, dim)); personal = rng.uniform(-1, 1, (N, dim))
    codes = rng.uniform(-1, 1, (N, K, dim)); role = np.arange(N) % K
    recs = []
    for b in range(boards):
        centers = np.clip(centers + drift * rng.randn(K, dim), -1, 1)
        personal = np.clip(personal + drift * rng.randn(N, dim), -1, 1)
        codes = np.clip(codes + drift * rng.randn(N, K, dim), -1, 1)
        if switch > 0:
            sw = rng.rand(N) < switch
            role[sw] = rng.randint(0, K, int(sw.sum()))
        for p in range(N):
            j = int(role[p])
            base = {'categories': centers[j], 'individual': personal[p], 'own-codes': codes[p, j],
                    'half': centers[j] if j < K // 2 else personal[p]}.get(kind)
            if base is None:
                base = rng.uniform(-1, 1, dim)
            s = np.clip(base + noise * rng.randn(dim), -1, 1)
            recs.append((b, p, 'leaf%d' % j, ('leaf%d' % j,), s))
    return recs


def selftest(perms=200):
    rng = np.random.RandomState(7)
    cases = [('categories', 0.0, 'shared sign per job, stable jobs, signs drifting', True),
             ('categories', 0.01, 'shared sign per job, learners switching jobs', True),
             ('individual', 0.0, 'each learner its own sign (identity tag), stable jobs', False),
             ('own-codes', 0.01, 'each learner its own sign PER job, switching jobs', False),
             ('half', 0.0, 'four jobs with shared signs, four with personal signs', None),
             ('none', 0.0, 'a fresh random sign on every board', False)]
    print('planted worlds: 16 learners, 8 jobs, 600 boards, windows of 100 boards, sign noise 0.25, drift 0.02 per board')
    ok = True
    for kind, sw, desc, expect in cases:
        recs = plant(kind, rng, switch=sw)
        rs = [analyse([r for r in recs if w <= r[0] < w + 100], 5, perms, rng) for w in range(0, 600, 100)]
        print('%s (%s):' % (kind if not sw else kind + ', switching', desc))
        print(summary('head', rs, perms))
        print('    D own-sign %.2f vs null %.2f | role stability %.2f' % (
            np.mean([r['own'] for r in rs]), np.mean([r['own_null'] for r in rs]), np.mean([r['stab'] for r in rs])))
        obs = float(np.mean([r['mi'] for r in rs])); null = np.mean([r['mi_null'] for r in rs], axis=0)
        p = (1 + int((null >= obs - 1e-12).sum())) / (len(null) + 1)
        if expect is True and not p <= 0.01:
            ok = False; print('    FAIL: planted categories not found')
        if expect is False and p <= 0.05:
            ok = False; print('    FAIL: categories reported where none were planted')
    print('SELFTEST %s' % ('PASS' if ok else 'FAIL'))


def main():
    a = sys.argv[1:]
    if a and a[0] == '--selftest':
        return selftest()
    run = a[0].rstrip('/')
    def opt(n, d):
        return a[a.index(n) + 1] if n in a else d
    W = int(opt('--window', 100)); stride = int(opt('--stride', 500)); last = int(opt('--last', 600))
    mm = int(opt('--min-msgs', 5)); perms = int(opt('--perms', 200)); out = opt('--json', None)
    sold = '--sold-only' in a
    cfg = json.load(open([os.path.join(run, f) for f in os.listdir(run) if f.endswith('.json') and not f.startswith('_seeds')][0]))
    H = sum(1 for e in cfg['blackboard'] if e.get('distributes'))
    logs = os.path.join(run, 'logs')
    nb = max(int(f[3:-4]) for f in os.listdir(logs) if f.startswith('log') and f[3:-4].isdigit())
    paid = set()
    if sold:
        for r in csv.DictReader(open(os.path.join(run, 'payments.csv')), delimiter=';'):
            try:
                paid.add((int(float(r['time'])), int(r['seller'])))
            except (ValueError, KeyError):
                pass
    def recs_for(b0, b1):
        rs = []
        for b in range(b0, b1):
            try:
                msgs = json.load(open(os.path.join(logs, 'log%d.txt' % (b + 1))))
            except Exception:
                continue
            for m in msgs:
                aid = agent_id(m.get('label'))
                if aid is None or aid < H or not m.get('sign'):
                    continue
                if sold and (b, aid) not in paid:
                    continue
                sec = first_section(m.get('trades', []))
                if not sec:
                    continue
                head = strip(sec[0].get('item'))
                prod = tuple(sorted(strip(x.get('item')) for x in sec if x.get('type') in ('sell', 'construct', 'buy')))
                rs.append((b, aid, head, prod, np.array([float(v) for v in m['sign']])))
        return rs
    rng = np.random.RandomState(11)
    print('run %s: %d boards, %d humans, windows of %d boards, units need %d boards, %d null draws%s'
          % (os.path.basename(run), nb, H, W, mm, perms, ', SOLD messages only' if sold else ''))
    print('time course (HEAD = the decoder sold):')
    starts = list(range(0, nb - W + 1, stride))
    for s in starts:
        r = analyse(recs_for(s, s + W), mm, perms, rng, 'head')
        print(line('boards %d-%d' % (s, s + W - 1), r))
    res = {'head': [], 'product': []}
    print('last %d boards, windows of %d:' % (last, W))
    for s in range(nb - last, nb - W + 1, W):
        rr = recs_for(s, s + W)
        for lab in ('head', 'product'):
            r = analyse(rr, mm, perms, rng, lab)
            res[lab].append(r)
            print(line('%s %d-%d' % (lab, s, s + W - 1), r))
    for lab in ('head', 'product'):
        print(summary('%s (last %d boards)' % (lab, last), res[lab], perms))
    if out:
        json.dump({lab: [None if r is None else {k: v for k, v in r.items()} for r in res[lab]] for lab in res},
                  open(out, 'w'), default=float)


if __name__ == '__main__':
    main()
