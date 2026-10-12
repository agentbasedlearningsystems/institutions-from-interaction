#!/usr/bin/env python3
"""utility_curve.py <run_dir> [--window 1000]   (7 Oct 2026, her request: utility going up as a result of the signs)

From reproduction_report.csv (one line per agent per board: time;agent;label;utility;...), the humans' utility by window:
the mean over the window's boards of the mean over the 11 Cobb-Douglas humans of the utility the engine computed for
them that board (U = prod (0.3 + x_g)^(w_g / sum w) over the scores acquired on the board; a human that bought nothing
scores U = 0.3^1 = 0.3, the floor; there is no fixed ceiling).
Prints one line per window and a summary: the first window, the last full window, and the mean over the last 5 windows.
"""
import csv
import sys
import collections

run = sys.argv[1].rstrip('/')
W = int(sys.argv[sys.argv.index('--window') + 1]) if '--window' in sys.argv else 1000
by_win = collections.defaultdict(list)
with open(run + '/reproduction_report.csv') as fh:
    r = csv.reader(fh, delimiter=';')
    header = next(r)
    i_t, i_l, i_u = header.index('time'), header.index('label'), header.index('utility')
    for row in r:
        if len(row) <= i_u or 'human' not in row[i_l]:
            continue
        try:
            t = int(float(row[i_t])); u = float(row[i_u])
        except ValueError:
            continue
        by_win[t // W].append(u)
wins = sorted(by_win)
vals = [(w * W, sum(by_win[w]) / len(by_win[w]), len(by_win[w])) for w in wins]
name = run.split('/')[-1]
print('%s humans\' mean utility by window of %d boards: %s' % (name, W, ' '.join('%d:%.3f' % (s, u) for s, u, n in vals)))
full = [v for v in vals if v[2] >= 0.8 * 11 * W / 10]      # windows with most of their boards logged (one line per human per logged board)
if full:
    print('%s first window %.3f, last full window %.3f (boards %d+), mean of last 5 full windows %.3f' % (
        name, full[0][1], full[-1][1], full[-1][0], sum(v[1] for v in full[-5:]) / len(full[-5:])))
