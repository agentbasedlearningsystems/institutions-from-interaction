#!/usr/bin/env python3
"""make_ladder_world.py - the cipher ladder world (6 Oct 2026): complexification through signs, built to start.

Her direction (6 Oct): keep working toward complexification through signs with categories, and make the setup likely
to work before it runs. The math (PRICES_MATH_20261006.md and the ladder section of TWO_LAYER_KEYS_MATH_20261006.md):
  - under her 5 Oct ruling a learner's hire is one draw by sign over all sell rows, so a bought part is the right one
    with chance d, about (sellers of the part) / (offers on the board) while signs carry no categories;
  - a learner that can hold every layer never gains by buying one, so only learners that cannot build the whole buy;
  - such a learner assembles only if the whole is worth enough more than its best simple good:
        P_whole x s_own x d x s_part > P_two x s_two,
    and prices carry the humans' values only when kept money counts in their score (her yes to decision 1).
The world, on the fixed-money Cobb-Douglas world (11 humans, money 100 per board, learned price bands; 16 learners):
  keys     the substitution layer on 4 letter classes, 4 keys per class, one key set, class 3's key off the edge
           (CIPHER_SUBST_K=4, CIPHER_KEYSETS=1, CIPHER_EDGE_FREE=1); the goods hA hB hC hD, hAB hCD, hABCD
  weights  10 for a one-layer good, 30 for a two-layer good, and --whole (120 as her price list, or 240)
  money    humans_cd['money_value'] = --money-value (0.01): kept money counts in the same board's score
  capacity plans_by_learner [3, 16] (her suggestion of 6 Oct 05:00): even agent numbers 3 rows, odd numbers 16
  rulings  gates_after_sign and seller_name_defaults on (her 5 Oct ruling); defaults by her criterion, computed from the
           goods (with one key set, each class's default key is the key its goods need); the full method
Usage: CIPHER_P=4 CIPHER_SUBST_K=4 CIPHER_KEYSETS=1 CIPHER_EDGE_FREE=1 \
       make_ladder_world.py <gate dir> <name> <boards> --whole 240 --money-value 0.01 [--capacity 3,16] <seed> ...
       (--money-value 0 leaves the setting out; --capacity none leaves plans_by_learner out)
"""
import collections
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'simulation'))
import cipher_world as cw  # noqa: E402  (reads CIPHER_P, CIPHER_SUBST_K, CIPHER_KEYSETS and CIPHER_EDGE_FREE at import)

a = sys.argv[1:]


def opt(k, d):
    global a
    if k in a:
        i = a.index(k)
        v = a[i + 1]
        a = a[:i] + a[i + 2:]
        return v
    return d


WHOLE = int(opt('--whole', '240'))
MONEY_VALUE = float(opt('--money-value', '0.01'))
CAP = opt('--capacity', '3,16')
# The settings of the evening of 6 Oct 2026 (DEBUG_LADDER_20261006.md, section 9); each defaults to what the first
# ladder worlds ran with.
DEFAULTS = opt('--defaults', 'on')      # on: an early name sells the designated default (her 5 Oct ruling); off: it sells nothing (option A)
GATE = opt('--gate', 'after')           # after: one draw over all sell rows, then the checks (her 5 Oct ruling); category: the category gate before the draw (her 30 Sep rule; option C)
JUDGE = opt('--judge', 'lenient')       # lenient | strict (a damaged plain class zeroes the delivery)
WHOLE_AON = opt('--whole-aon', '0')     # 1: the whole pays only when all four classes are right
PRICES = opt('--prices', 'learned')     # learned: price bands, fixed money, kept money in the score if --money-value > 0; list: designed prices = the weights, parts at 4
NAME = opt('--name', 'one')             # one: the one-number name; many: one number per level (item_size 4; her point of 6 Oct 22:3x: the one-number name has no gradient at depth)
HTREE = opt('--humans-tree', 'off')     # on: her decision 18 (6 Oct 23:3x): a human row's good and want read as a tree (rung, then good within it), the want being the good's own
MAXSTD = float(opt('--maxstd', '0'))    # > 0: a ceiling on every gene's step (pycma maxstd; 7 Oct 00:5x diagnosis: neutral genes jitter far beyond the global step)
LOGSTDS = opt('--log-stds', 'off')      # on: one console line per generation per agent with the sampled per-gene steps
STICKY = float(opt('--sticky', '0'))
BW = float(opt('--buyers-weight', '1.0'))   # 1.0: the buyer's sought sign against the seller's displayed sign only (11 Aug); 0.5: the original engine's two-sided blend
BD = opt('--birth-death', 'off')          # off | d,b,cap : death per learner per board, births expected per board, population cap (her 6 Aug rule: newborns fresh from the broad prior; her 7 Oct yes)
DATA = opt('--data', 'goods')             # goods: the data branch holds this world's seven corpora only (7 Oct fix); all: every corpus of the module (the main set's state)
BRAIN = opt('--brain', 'cmaes')          # cmaes | llm:<model>[,<model>,<model>]:<ctx>[,<ctx>] - model tiers dealt by agent number (her 7 Oct ruling) - language-model learners (LLMAgent) with context budgets by learner, e.g. llm:deepseek-chat:2,16 or llm:stub:2,16 (plumbing test, no spend)
TPATH = int(opt('--test-path', '3'))     # one number per level of a test's path and its data path (her 7 Oct rule: no one-number paths anywhere); the main set ran with 1
SEEK = opt('--sells-seek', 'off')          # on: sell and construct rows decode their own sought sign (the original engine's two-sided match needs it with --buyers-weight 0.5)
CLASSN = opt('--class-names', 'letters')   # letters: decoder_subst_A_k2 names the class by its corpus letter (her 7 Oct question); numbers: c0..c3 (every world before 7 Oct 14:00)
POOL = opt('--pool', 'on')                 # on: standing wants with a pointer in each buy row; learners-off: each learner buy row carries its own sought sign and selectors (the original layout), humans keep one want per good (7 Oct, her word)    # > 0: the dead-zone reading of choice genes (RoleSim sticky_margin; 7 Oct 01:3x, section 17 of the DEBUG page)
assert DEFAULTS in ('on', 'off') and GATE in ('after', 'category') and JUDGE in ('lenient', 'strict') \
    and WHOLE_AON in ('0', '1') and PRICES in ('learned', 'list') and NAME in ('one', 'many'), 'bad option value'
G = a[0].rstrip('/') + '/'
name = a[1]
boards = int(a[2])
seeds = [int(s) for s in a[3:]]
assert cw.P == 4 and cw.SUBST_K == 4 and cw.KEYSETS == 1 and cw.EDGE_FREE, \
    'run with CIPHER_P=4 CIPHER_SUBST_K=4 CIPHER_KEYSETS=1 CIPHER_EDGE_FREE=1'
assert getattr(cw, 'LAYERS', 1) == 1, 'one layer per class'

GOODS = ['hA', 'hB', 'hC', 'hD', 'hAB', 'hCD', 'hABCD']
VALUE = {1: 10, 2: 30, 4: WHOLE}
WEIGHTS = {g: VALUE[len(cw.CORPORA[g])] for g in GOODS}

assert CLASSN in ('letters', 'numbers'), 'bad --class-names'
ont = cw.ontology(types=['subst'], classes=[0, 1, 2, 3], corpora=(GOODS if DATA == 'goods' else None), class_names=CLASSN)
demand = collections.Counter()
for g in GOODS:
    for c, (kind, key) in cw.CORPORA[g].items():
        demand[(kind, c, key)] += WEIGHTS[g]
dec = ont['decoder']


def kids(node):
    return [k for k, v in node.items() if isinstance(v, dict) and not k.startswith('_')]


table = []
kind_d = collections.Counter({kd: sum(v for (k2, _, _), v in demand.items() if k2 == kd) for kd in kids(dec)})
dec['_default'] = max(kids(dec), key=lambda kd: (kind_d[kd], -kids(dec).index(kd)))
for kd in kids(dec):
    node = dec[kd]
    cls = kids(node)
    cidx = {cn: i for i, cn in enumerate(cls)}      # the class's position in the tree, whatever its name (c0 or A)
    cdem = {cn: sum(v for (k2, c, _), v in demand.items() if k2 == kd and c == cidx[cn]) for cn in cls}
    node['_default'] = max(cls, key=lambda cn: (cdem[cn], -cls.index(cn)))
    for cn in cls:
        ks = kids(node[cn])
        kdem = {kn: demand.get((kd, cidx[cn], int(kn[1:])), 0) for kn in ks}
        node[cn]['_default'] = max(ks, key=lambda kn: (kdem[kn], -int(kn[1:])))
        table.append('%s_%s: default %s (demanded %d of %d)' % (kd, cn, node[cn]['_default'],
                                                              kdem[node[cn]['_default']], sum(kdem.values())))
print('defaults by her criterion (4 keys per class, one key set %s): decoder -> %s -> %s'
      % (cw._HK, dec['_default'], dec[dec['_default']]['_default']))
for t in table:
    print('  ' + t)


def cd_blackboard(n, rows_per_good):
    out = []
    for h in range(n):
        rows = [{'type': 'buy', 'item': 'decoder_stop',
                 'tests': [{'stophere': False, 'test': 'test_decoder_fractionCorrect',
                            'data': 'data_cipher_%s' % GOODS[r % len(GOODS)], 'hidden': False},
                           {'stophere': True}]} for r in range(rows_per_good * len(GOODS))]
        out.append({'type': 'RoleSim', 'distributes': True, 'cd_human': True,
                    'label': 'Cipher CD human %d (wants all %d goods)' % (h, len(GOODS)),
                    'trades': rows + [{'type': 'stop'}],
                    '_seed_provenance': 'her Cobb-Douglas scenario, 4 Oct 2026; the cipher ladder world, 6 Oct 2026'})
    return out


for seed in seeds:
    base = json.load(open(G + '_cipher_s%d_CD1.json' % seed), object_pairs_hook=collections.OrderedDict)
    c = copy.deepcopy(base)
    p = c['parameters']
    n_h = sum(1 for e in base['blackboard'] if e.get('cd_human'))
    c['blackboard'] = cd_blackboard(n_h, 3)            # 3 rows per good, 21 rows, as in the 7-good worlds
    c['ontology'] = ont
    p.pop('price_list', None)
    hcd = {'goods': GOODS, 'rows_per_good': 3, 'eps': 0.3, 'weights': WEIGHTS, 'money': 100}
    if MONEY_VALUE > 0:
        hcd['money_value'] = MONEY_VALUE
    if PRICES == 'list':
        # designed prices (as the LADSJP and LADA worlds of 6 Oct): humans pay the list price times score cubed, a
        # learner pays 4 for a part; no fixed money and no kept-money term
        hcd.pop('money', None)
        hcd.pop('money_value', None)
        hcd['prices'] = dict(WEIGHTS)
    if HTREE == 'on':
        hcd['good_tree'] = [[g for g in GOODS if len(cw.CORPORA[g]) == n] for n in (1, 2, 4)]
    p['humans_cd'] = hcd
    p['agent_parameters']['RoleSim']['cma_diagonal'] = False      # the full method (her ruling, 5 Oct)
    if MAXSTD > 0:
        p['agent_parameters']['RoleSim']['cma_maxstd'] = MAXSTD
    else:
        p['agent_parameters']['RoleSim'].pop('cma_maxstd', None)
    p['agent_parameters']['RoleSim']['cma_log_stds'] = (LOGSTDS == 'on' or MAXSTD > 0 or STICKY > 0)
    if STICKY > 0:
        p['agent_parameters']['RoleSim']['sticky_margin'] = STICKY
    else:
        p['agent_parameters']['RoleSim'].pop('sticky_margin', None)
    p['cipher_P'] = cw.P
    p['cipher_subst_k'] = cw.SUBST_K
    p['cipher_keysets'] = 1
    p['cipher_edge_free'] = True
    if CAP != 'none':
        p['plans_by_learner'] = [int(x) for x in CAP.split(',')]
        p['num_trade_plans'] = max(p['plans_by_learner'])
    p['gates_after_sign'] = (GATE == 'after')
    p['category_gate'] = (GATE == 'category')
    p['seller_name_defaults'] = (DEFAULTS == 'on')
    p['cipher_whole_aon'] = (WHOLE_AON == '1')
    p['cipher_strict_judge'] = (JUDGE == 'strict')
    if PRICES == 'list':
        p['price_list'] = {'decoder': 4, 'default': 4}
    p['buyers_weight'] = BW
    if BRAIN.startswith('llm:'):
        _, _model, _ctx = BRAIN.split(':')
        _models = _model.split(',')          # one model, or the MODEL TIERS dealt by agent number (her ruling 7 Oct, decision 24)
        p['random_agents'] = {'LLMAgent': 16}
        p['agent_parameters']['LLMAgent'] = {'model': _models[0], 'models_by_learner': _models,
                                             'context_boards_by_learner': [int(x) for x in _ctx.split(',')],
                                             'fitness_weights': {'agi_tokens': 1, 'buyer_score': 0, 'seller_score': 0},
                                             'max_tokens': 900, 'temperature': 0.3}
        p['_brain_provenance'] = ('LANGUAGE-MODEL LEARNERS (7 Oct 2026, her word that the LLM runs proceed beside the CMA-ES runs): the '
                                  'same world, humans, judge and draw; each learner is one model call per board (model %s) that writes its '
                                  'message, display sign, rows with items and prices, sought signs and tests, encoded through the authored-'
                                  'message path; the model tiers by learner (her ruling 7 Oct): %s; context budget %s by learner.' % (_model, _models, _ctx))
    p['test_path_size'] = TPATH
    if BD != 'off':
        _d, _b, _cap = BD.split(',')
        p['birth_death'] = True
        p['death_rate'] = float(_d); p['birth_rate'] = float(_b); p['max_learners'] = int(_cap)
        p['_birth_death_provenance'] = ('BIRTH AND DEATH (her 6 Aug 2026 rule, her 7 Oct yes: culture must outlast individuals): each learner dies '
                                        'with probability %s per board (mean life %.0f boards), newborns arrive at %s expected per board with fresh '
                                        'genomes from the broad prior, the population capped at %s; humans never die.' % (_d, 1.0 / float(_d), _b, _cap))
    if SEEK == 'on':
        p['sells_seek'] = True
    if POOL == 'learners-off':
        p['pool_wants_learners'] = False
    p['single_float_ontology'] = (NAME == 'one')
    p['item_size'] = 1 if NAME == 'one' else 4
    p['seed'] = seed                 # the world's RNG seed is the number in its name, whatever base config it was built from (7 Oct)
    p['max_iterations'] = boards
    p['log_every'] = 10
    lab = 'cipher_s%d_%s' % (seed, name)
    p['label'] = lab
    p['output_path'] = 'experiments/%s/' % lab
    p['_ladder_world_provenance'] = (
        '6 Oct 2026, the cipher ladder world: her 5 Oct ruling (draw by sign over all sell rows, then the checks; early '
        'names sell defaults by her criterion: %s); kept money counts in the humans\' score (money_value %s, her yes to '
        'decision 1); capacity %s (her suggestion of 6 Oct 05:00); weights %s; 4 keys per class, one key set %s, '
        'class 3 off the edge; the full method. Launch with CIPHER_P=4 CIPHER_SUBST_K=4 CIPHER_KEYSETS=1 '
        'CIPHER_EDGE_FREE=1.' % ('; '.join(table), MONEY_VALUE if MONEY_VALUE > 0 else 'off', CAP,
                                 json.dumps(WEIGHTS), cw._HK)
        + ' Evening settings (DEBUG_LADDER_20261006.md section 9): defaults %s; gate %s; judge %s; '
          'whole all-or-nothing %s; prices %s; name %s; humans\' good and want as a tree %s.'
          % (DEFAULTS, GATE, JUDGE, WHOLE_AON, PRICES, NAME, HTREE))
    fn = G + '_%s.json' % lab
    if os.path.exists(fn):
        raise SystemExit('exists, not overwriting: ' + fn)
    json.dump(c, open(fn, 'w'), indent=1)
    print('wrote %s (%.1f MB)' % (fn, os.path.getsize(fn) / 1e6))
