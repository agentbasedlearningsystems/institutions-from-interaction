#!/usr/bin/env python3
"""start_rates.py - the starting rates of a cipher ladder world, read from its board logs (6 Oct 2026, evening).

Usage: CIPHER_P=4 CIPHER_SUBST_K=4 CIPHER_KEYSETS=1 CIPHER_EDGE_FREE=1 start_rates.py <world dir> --engine <dir>
Reads logs/log*.txt (every logged board). Humans are the entries with cd_good rows; the rest are learners. Reports:
  - learner buy rows: how deep the name goes (1 = decoder, 2 = subst, 3 = the class, 4 = the key), offers per row,
    the share of offers whose sell item is of the named class, refused draws, and the drawn section's classes;
  - learner sell and construct rows: exact leaf, early name (with or without a default substituted), bare stop;
  - sections (a sell row to the next sell or stop): the classes decoded with the right key, wrong-key layers, buy rows
    inside; exact right pairs {0,1} and {2,3}; right wholes; a right single.
Everything is a count over rows on logged boards; rates are given as (n/d).
"""
import collections
import json
import os
import re
import sys

a = sys.argv[1:]
eng = a[a.index('--engine') + 1] if '--engine' in a else None
if eng:
    a = [x for i, x in enumerate(a) if x != '--engine' and (i == 0 or a[i - 1] != '--engine')]
    sys.path.insert(0, eng)
import cipher_world as cw  # noqa: E402

RIGHT = {int(c): int(k) for c, (kind, k) in cw.CORPORA['hABCD'].items()}   # class -> the right key
d = a[0].rstrip('/')
logs = sorted((int(m.group(1)), f) for f in os.listdir(d + '/logs')
              for m in [re.match(r'^log(\d+)\.txt$', f)] if m)
LEAF = re.compile(r'^decoder_subst_(?:c(\d+)|([A-H]))_k(\d+)$')   # c0..c3 or A..D (7 Oct letters)


def depth(item):
    segs = str(item).split('_')
    return segs.index('stop') if 'stop' in segs else len(segs)


def layer(item):
    m = LEAF.match(str(item))
    if not m:
        return None
    cls = int(m.group(1)) if m.group(1) is not None else 'ABCDEFGH'.index(m.group(2))
    return (cls, int(m.group(3)))


buy_depth = collections.Counter()
offers_n = []
same_class = [0, 0]
refused = collections.Counter()
drawn_shape = collections.Counter()
sell_kind = collections.Counter()
sections = collections.Counter()
boards = 0
for n, f in logs:
    try:
        msgs = json.load(open(os.path.join(d, 'logs', f)))
    except Exception as e:  # noqa: BLE001
        print('skip', f, e)
        continue
    boards += 1
    is_human = [any('cd_good' in t for t in m.get('trades', [])) for m in msgs]

    def section_of(ai, ri):
        """classes right / wrong-key layers / buys of the section starting at sell row ri of agent ai"""
        rows = msgs[ai].get('trades', [])
        right, wrong, buys = [], 0, 0
        for r in rows[ri:]:
            if r is rows[ri] or r.get('type') == 'construct':
                lay = layer(r.get('item'))
                if lay is None:
                    continue
                c, k = lay
                if RIGHT.get(c) == k:
                    right.append(c)
                else:
                    wrong += 1
            elif r.get('type') == 'buy':
                buys += 1
            else:
                break
        return tuple(sorted(set(right))), wrong, buys

    for ai, m in enumerate(msgs):
        if is_human[ai]:
            continue
        rows = m.get('trades', [])
        for ri, r in enumerate(rows):
            t = r.get('type')
            if t == 'stop':
                break
            if t == 'buy':
                dep = depth(r.get('item'))
                buy_depth[dep] += 1
                offs = r.get('offers') or []
                offers_n.append(len(offs))
                named = str(r.get('item')).split('_')
                cls = named[2] if dep >= 3 else None
                for o in offs:
                    try:
                        s_item = msgs[o['agent']]['trades'][o['trades'][0]]['item']
                    except Exception:  # noqa: BLE001
                        continue
                    if cls is not None:
                        same_class[1] += 1
                        if str(s_item).startswith('decoder_subst_' + cls):
                            same_class[0] += 1
                if r.get('_refused'):
                    refused[r['_refused']] += 1
                dr = r.get('_draw')
                if dr is not None and offs and dr < len(offs):
                    o = offs[dr]
                    try:
                        drawn_shape[section_of(o['agent'], o['trades'][0])] += 1
                    except Exception:  # noqa: BLE001
                        drawn_shape['?'] += 1
            elif t in ('sell', 'construct'):
                it = str(r.get('item'))
                if it == 'stop':
                    sell_kind['bare stop'] += 1
                elif layer(it) is not None:
                    sell_kind['exact leaf' + (' (default for %s)' % r['_named'] if r.get('_named') else '')] += 1
                else:
                    sell_kind['early name, sells nothing'] += 1
                if t == 'sell':
                    sections[section_of(ai, ri)] += 1

print('boards read: %d (logs %s..%s)' % (boards, logs[0][0] if logs else '-', logs[-1][0] if logs else '-'))
print('right keys by class:', RIGHT)
tb = sum(buy_depth.values())
print('learner buy rows: %d; name depth: %s' % (tb, ', '.join('%d levels (%d/%d)' % (k, v, tb) for k, v in sorted(buy_depth.items()))))
if offers_n:
    print('offers per learner buy row: mean %.1f, min %d, max %d' % (sum(offers_n) / len(offers_n), min(offers_n), max(offers_n)))
print('offers of the named class, for rows naming a class: (%d/%d)' % tuple(same_class))
print('refused draws:', dict(refused))
print('drawn sections (right classes, wrong-key layers, buys inside): %s'
      % ', '.join('%s x%d' % (k, v) for k, v in drawn_shape.most_common(12)))
ts = sum(sell_kind.values())
print('learner sell+construct rows: %d; %s' % (ts, ', '.join('%s (%d/%d)' % (k, v, ts) for k, v in sell_kind.most_common())))
tsec = sum(sections.values())
print('sections: %d; shapes (right classes, wrong-key layers, buys inside): %s'
      % (tsec, ', '.join('%s x%d' % (k, v) for k, v in sections.most_common(14))))
pairs = sum(v for (rc, w, b), v in sections.items() if rc in ((0, 1), (2, 3)) and w == 0)
wholes = sum(v for (rc, w, b), v in sections.items() if rc == (0, 1, 2, 3) and w == 0)
singles = sum(v for (rc, w, b), v in sections.items() if len(rc) == 1 and w == 0)
print('exact right pairs {0,1}/{2,3}: (%d/%d); right wholes (own layers): (%d/%d); right singles: (%d/%d)'
      % (pairs, tsec, wholes, tsec, singles, tsec))
