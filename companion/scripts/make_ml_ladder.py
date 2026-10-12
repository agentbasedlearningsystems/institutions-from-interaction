#!/usr/bin/env python3
"""make_ml_ladder.py - the machine-learning world under the ladder's design (7 Oct 2026; her word: the cipher world was
the toy built to support the ML world, so the conditions found there go to the ML world now).

Starts from a cipher ladder configuration written by make_ladder_world.py for the same seed (so every market law of the
ladder comes along: her 5 Oct rulings, the draw over every offer with the type checked after, ticket power 4, pay by
score cubed, buyer-only matching, the many-numbers name, 3-row and 16-row learners, the humans' tree, list prices) and
swaps in the machine-learning world: the cut tree of parts (two text cleaners, two vectorisers, the corpus labeler, six
classifiers), the three rungs per corpus (embedding scored by the probe, labelled embedding by the downstream F1, the
whole classifier by held-out F1), Cobb-Douglas humans wanting every rung with the tree over the rungs, and the ML
world's cache folder and row cap. The cipher settings are removed.

Usage: make_ml_ladder.py <gate dir with _cipher_s<seed>_<ladder>.json and _cleansed_template_seed306.json> <ladder name>
                         <new name> --seeds 312 313 ... [--corpora movieSentiment[,fakeNews:0.5]] [--nrows 1000]
                         [--weights 30,60,240] [--parts-price 4] [--rows-per-good 3]
"""
import collections
import copy
import json
import os
import sys

a = sys.argv[1:]
G = a[0].rstrip('/') + '/'
ladder = a[1]
name = a[2]


def opt(k, d):
    return a[a.index(k) + 1] if k in a else d
_tail = a[a.index('--seeds') + 1:] if '--seeds' in a else ['306']
seeds = []
for _s in _tail:                      # seeds run to the next option
    if _s.startswith('--'):
        break
    seeds.append(int(_s))
CORPORA = [(c.split(':')[0], float(c.split(':')[1]) if ':' in c else 1.0) for c in opt('--corpora', 'movieSentiment').split(',')]
corpus = CORPORA[0][0]
nrows = int(opt('--nrows', 1000))
W = [int(x) for x in opt('--weights', '30,60,240').split(',')]
PARTS_PRICE = float(opt('--parts-price', 4))
RPG = int(opt('--rows-per-good', 3))
SWS = opt('--shared-want-signs', 'off')       # 7 Oct 2026 night, the language-model culture debug: 'fixed' | 'initial' | 'off'.
# The 11 consumers' sought signs for the SAME good are unrelated random vectors (pairwise cosine ~0.1) and need thousands
# of boards of coevolution to align (517: whole 0.15 -> 0.46 by 27,000 boards); a 1,000-board language-model world cannot
# get there, so no sign can stand for a good. With this option every consumer's buy rows for good g carry the SAME
# random sought sign (one 16-vector per good, drawn once from a fixed seed, in genome floats [0,1]): 'fixed' pins it at
# every decode (a fixed buyer-side convention; sellers' signs free); 'initial' pins it at creation only (RoleSim.py strips
# the pin after the first decode, her 6 Aug rule: the initial state only) and the consumers evolve from the shared start.
import random as _random
_sw_rng = _random.Random(20261007)

T = json.load(open(G + '_cleansed_template_seed306.json'), object_pairs_hook=collections.OrderedDict)['ontology']


def meta(node):
    return collections.OrderedDict((k, copy.deepcopy(v)) for k, v in node.items() if k.startswith('_'))


def prune(node, keep):
    out = meta(node)
    for child, sub in keep.items():
        if child not in node:
            raise SystemExit('template has no child %r' % child)
        out[child] = copy.deepcopy(node[child]) if sub is None else prune(node[child], sub)
    return out


# the ML world's tree, exactly as make_ml_world.py cuts it (5 Oct 2026)
ont = collections.OrderedDict()
ont['test'] = prune(T['test'], {'classifier': {'f1heldout': None}, 'vectorSpace': {'probe': None}})
ont['test']['labeler'] = collections.OrderedDict([('_weight', 0.3)])
ont['test']['labeler']['downstreamF1'] = collections.OrderedDict([
    ('_args', [{'type': 'tuple', 'dtype': '(float32,int32)'}]), ('_return', [{'type': 'float'}]), ('_weight', 0.3),
    ('_deterministic', True), ('_comment', 'test_labeler_downstreamF1')])
ont['data'] = prune(T['data'], {'freetext': {c: None for c, _ in CORPORA}})
ont['preprocessor'] = prune(T['preprocessor'], {'freetext': {'strip': None, 'lemmatization': None}})
ont['labeler'] = prune(T['labeler'], {corpus: None})
ont['classifier'] = prune(T['classifier'], {'sklearn': {'LogisticRegression': {'creg01': None, 'creg1': None, 'creg10': None},
                                                        'ComplementNB': {'a001': None, 'a1': None, 'a10': None}}})
vs = meta(T['vectorSpace'])
vs['sklearn'] = meta(T['vectorSpace']['sklearn'])
for cls in ('TfidfVectorizer', 'HashingVectorizer'):
    vs['sklearn'][cls] = collections.OrderedDict([('_weight', 0.5)])
    vs['sklearn'][cls]['f300'] = collections.OrderedDict([
        ('_args', [{'type': 'list', 'dtype': 'str'}]), ('_return', [{'type': 'numpy.ndarray', 'dtype': 'float32'}]),
        ('_deterministic', True), ('_weight', 1.0), ('_comment', 'vectorSpace_sklearn_%s_f300' % cls)])
ont['vectorSpace'] = vs

# DESIGNATED DEFAULTS (her 5 Oct point 2: an early seller name sells the designated default, "designated in the
# ontology"; the ML defaults measured 5 Oct from 948 offline scores on four labelled corpora: lemmatization best in 4.3
# of 8 cases, TF-IDF in 12 of 12, complement naive Bayes with smoothing 0.01 in 9 of 24): one mark per level
ont['preprocessor']['_default'] = 'freetext'; ont['preprocessor']['freetext']['_default'] = 'lemmatization'
ont['vectorSpace']['_default'] = 'sklearn'; ont['vectorSpace']['sklearn']['_default'] = 'TfidfVectorizer'
ont['vectorSpace']['sklearn']['TfidfVectorizer']['_default'] = 'f300'; ont['vectorSpace']['sklearn']['HashingVectorizer']['_default'] = 'f300'
ont['labeler']['_default'] = corpus
ont['classifier']['_default'] = 'sklearn'; ont['classifier']['sklearn']['_default'] = 'ComplementNB'
ont['classifier']['sklearn']['ComplementNB']['_default'] = 'a001'; ont['classifier']['sklearn']['LogisticRegression']['_default'] = 'creg1'
ont['data']['_default'] = 'freetext'; ont['data']['freetext']['_default'] = corpus
ont['test']['_default'] = 'classifier'; ont['test']['classifier']['_default'] = 'f1heldout'
ont['test']['vectorSpace']['_default'] = 'probe'; ont['test']['labeler']['_default'] = 'downstreamF1'

# the rungs per corpus: embedding, labelled embedding, whole; the first corpus keeps the plain names
SPECS = {'emb': {'item': 'vectorSpace_stop', 'test': 'test_vectorSpace_probe'},
         'lemb': {'item': 'labeler_stop', 'test': 'test_labeler_downstreamF1'},
         'clf': {'item': 'classifier_stop', 'test': 'test_classifier_f1heldout'}}
BASE = ['emb', 'lemb', 'clf']
GOODS = []; WEIGHTS = {}; TREE = [[], [], []]
for ci, (c, f) in enumerate(CORPORA):
    for ri, g in enumerate(BASE):
        g2 = g if ci == 0 else g + c[0].upper()
        SPECS[g2] = dict(SPECS[g], data='data_freetext_' + c)
        GOODS.append(g2); WEIGHTS[g2] = int(round(W[ri] * f)); TREE[ri].append(g2)
for g in BASE:
    SPECS[g] = dict(SPECS[g], data='data_freetext_' + corpus)

n_cd = 11
bb = []
for h in range(n_cd):
    hrows = [{'type': 'buy', 'item': SPECS[g]['item'],
              'tests': [{'stophere': False, 'test': SPECS[g]['test'], 'data': SPECS[g]['data'], 'hidden': False},
                        {'stophere': True}]} for g in GOODS for _ in range(RPG)]
    if SWS != 'off' and h == 0:
        # The sign is written into the consumer's STANDING-WANT BLOCK by good (RoleSim.py), not into the rows: a row's
        # good is read from the consumer's own genes, so a sign pinned through a row would land in whatever good the
        # row happened to point at (the 7 Oct 23:3x smoke showed exactly that scramble).
        SHARED_SIGNS = {g: [round(_sw_rng.random(), 4) for _ in range(16)] for g in GOODS}
    bb.append({'type': 'RoleSim', 'distributes': True, 'cd_human': True,
               'label': 'ML CD human %d (wants all %d goods)' % (h, len(GOODS)), 'trades': hrows + [{'type': 'stop'}],
               '_seed_provenance': 'her Cobb-Douglas humans in the machine-learning world (5 Oct 2026), under the ladder design (7 Oct)'})

for seed in seeds:
    src = G + '_cipher_s%d_%s.json' % (seed, ladder)
    c = json.load(open(src), object_pairs_hook=collections.OrderedDict)
    p = c['parameters']
    c['ontology'] = ont
    c['blackboard'] = bb
    p['blackboard_agents'] = [1] * len(bb)
    p['random_agents'] = p.get('random_agents') or {'RoleSim': 16}      # the base decides the brain (CMA-ES or LLMAgent)
    p['humans_cd'] = collections.OrderedDict([
        ('goods', GOODS), ('rows_per_good', RPG), ('eps', 0.3), ('weights', WEIGHTS), ('prices', dict(WEIGHTS)),
        ('good_tree', TREE), ('good_specs', SPECS)])
    p['price_list'] = {'default': PARTS_PRICE}
    if SWS != 'off':
        p['humans_shared_want_signs'] = SWS
        p['humans_shared_want_signs_table'] = {g: list(SHARED_SIGNS[g]) for g in GOODS}
    p['data_nrows'] = nrows
    p['cache_path'] = ('shared_cache/ml_%s_rows%d_wn/' % (corpus, nrows) if len(CORPORA) == 1 else
                       'shared_cache/ml_%s_rows%d_wn2/' % ('+'.join(c for c, _ in CORPORA), nrows))
    for k in [k for k in p if k.startswith('cipher_')]:
        p.pop(k)
    for k in [k for k in p if k.startswith('_') and 'provenance' in k and k != '_ladder_world_provenance']:
        p.pop(k)
    p['seed'] = seed
    p['label'] = 'ml_s%d_%s' % (seed, name)
    p['output_path'] = 'experiments/%s/' % p['label']
    p['_ml_ladder_provenance'] = ('7 Oct 2026, the machine-learning world under the ladder design (her word: the cipher world was '
                                  'the toy built to support the ML world). Market laws from %s (make_ladder_world.py); the ML tree of 11 '
                                  'parts; rungs per corpus %s with weights %s as list prices; parts at %s; 11 Cobb-Douglas humans, '
                                  '%d rows per good, the good read as a tree over the rungs; corpora %s, %d rows.'
                                  % (ladder, BASE, json.dumps(WEIGHTS), PARTS_PRICE, RPG, [c for c, _ in CORPORA], nrows))
    fn = G + '_%s.json' % p['label']
    if os.path.exists(fn):
        raise SystemExit('exists, not overwriting: ' + fn)
    json.dump(c, open(fn, 'w'), indent=1)
    print('wrote %s: goods %s, tree %s, plans %s, item_size %s, prices %s' % (fn, GOODS, TREE, p.get('plans_by_learner'),
                                                                                p.get('item_size'), p['humans_cd']['prices']))
