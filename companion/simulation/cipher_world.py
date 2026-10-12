"""cipher_world.py - her July 2026 instrument as a registry module and an ontology.

Her 14 July words: "it can be completely contrived in the sense that we have multiple
cypher reqs, and we can control competition, and we can know exactly how long it will
take. but dont make them take unnecessarily long, ever." Keys are reusable capital,
delivery is a service not a key, layered ciphers give depth through distributed
ownership (DOMAIN_QUESTION_20261001.md section 3). Built 3 October 2026.

The world
---------
A message is a line of lowercase words. Its letters fall into P = 3 position classes
by their letter ordinal modulo 3 (non-letters pass through every layer unchanged and
count for nothing). A LAYER is a keyed cipher applied to one class:

  shift  k   Caesar shift by k (keys 3, 7, 11, 15, 19, 23)
  subst  k   a seeded permutation of the alphabet, k in 1..6
  vig    k   a Vigenere keyword, k in 1..6, advancing along the letters of the class

Layers on different classes commute, so a corpus enciphered with one, two or three
layers is decoded by applying the inverse layer of each class in any order; one
correct layer of three recovers a third of the enciphered letters. That is the
gradient by construction: a partly right chain scores partly, nothing scores unless a
layer is right.

A DECODER LEAF `decoder_<type>_c<class>_k<key>` applies the inverse of exactly one
layer with one key: 3 types x 3 classes x 6 keys = 54 leaves, each a routine that is
right for every corpus enciphered with that layer (reusable capital) and wrong, and
harmful, for every other class it is applied to.

A DATA TERMINAL `data_cipher_<corpus>` is the enciphered corpus (40 lines, seeded by
its name). The TEST `test_decoder_fractionCorrect` scores a delivery against the
corpus's hidden plaintext: the share of enciphered letters now correct. Letters that
were never enciphered are not scored, so a wrong layer applied to a plain class costs
nothing and a right layer always earns its share: a corpus with L layers scores 1/L
per right layer, 1.0 when all are undone, 0 for junk (every layer is a derangement,
so no wrong key earns coincidental credit). The stricter judge that also charges for
damage to plain letters was tried first (3 Oct) and set aside: it turns the one-layer
wants into needles, since any extra wrong layer zeroes the delivery.

The corpora share layers so that a tuned routine serves several wants:

  oneA   {c0: shift 7}                    oneB  {c1: subst 2}        oneC  {c2: vig 4}
  twoA   {c0: shift 7, c1: subst 2}       twoB  {c1: subst 2, c2: vig 4}
  threeA {c0: shift 7, c1: subst 2, c2: vig 4}
  threeB {c0: shift 11, c1: subst 5, c2: vig 1}     (a foreign key set, same depth)

The engine is not changed: products are chains, tests return [0, 1], the registry
calls are curried like every other product. The test reads the auditing buyer's
corpus from Registry.PORT_BUYER_DATA (the engine publishes it during an audition),
so it belongs in PORT_BUYER_KEYED, which _cipher_register in Registry.py does.
"""
import os
import random
import string

ALPHABET = string.ascii_lowercase
P = int(os.environ.get('CIPHER_P', '5'))   # position classes, set per world by the
# launcher (CIPHER_P) and checked against the config's cipher_P by MarketSim. It was a
# module constant edited 3 -> 4 (08:05) -> 5 (10:05) on 3 Oct, and because the delivery
# cache keyed cipher results by function names only, worlds run after each edit scored
# stale material cached under the old P (found 3 Oct afternoon). cache_salt() below now
# enters every cipher cache key.
SHIFT_KEYS = (3, 7, 11, 15, 19, 23)
SUBST_KEYS = (1, 2, 3, 4, 5, 6)
# HARD KEYS (her yes, 5 Oct 2026: "change the difficulty on a cipher ... make it worth more to hire than do it
# yourself"). CIPHER_SUBST_K = K gives the substitution layer K keys (each a seeded derangement), so a learner
# finds a given key with chance about 1/K per try and owns all the keys of a deep good rarely; the hard corpora
# hA..hABCD use four keys spread over 1..K. Unset: the module is exactly as before.
SUBST_K = int(os.environ.get('CIPHER_SUBST_K', '0') or 0)
if SUBST_K > 0:
    SUBST_KEYS = tuple(range(1, SUBST_K + 1))
VIG_WORDS = {1: 'by', 2: 'cedr', 3: 'delt', 4: 'ember', 5: 'flint', 6: 'grove'}   # no letter a: no zero shift, so no coincidental credit
LINES_PER_CORPUS = 40

CORPORA = {
    'oneA':   {0: ('shift', 7)},
    'oneB':   {1: ('subst', 2)},
    'oneC':   {2: ('vig', 4)},
    'twoA':   {0: ('shift', 7), 1: ('subst', 2)},
    'twoB':   {1: ('subst', 2), 2: ('vig', 4)},
    'threeA': {0: ('shift', 7), 1: ('subst', 2), 2: ('vig', 4)},
    'threeB': {0: ('shift', 11), 1: ('subst', 5), 2: ('vig', 1)},
    # The first-success world (3 Oct 2026, her order: a world the arithmetic says
    # will succeed): one class, one layer type, three keys. 3, 7 and 11 are chosen
    # so that no two of them add to a third (mod 26), so a two-row section is never
    # right by accident; a random one-row section is right one time in three.
    'eA': {0: ('shift', 3)},
    'eB': {0: ('shift', 7)},
    'eC': {0: ('shift', 11)},
    # The ladder world F2 (3 Oct 2026, her order: a whole society with signs,
    # completing each level of difficulty). One layer type (shift), three classes,
    # two keys per class (3 and 7: 3+3, 3+7, 7+7 are never 3 or 7 mod 26, so no
    # two-row section is right by accident). Six leaves; three of them are the
    # right parts, and every deeper want is built from those same three.
    'lA':   {0: ('shift', 3)},
    'lB':   {1: ('shift', 7)},
    'lC':   {2: ('shift', 3)},
    'lAB':  {0: ('shift', 3), 1: ('shift', 7)},
    'lBC':  {1: ('shift', 7), 2: ('shift', 3)},
    'lABC': {0: ('shift', 3), 1: ('shift', 7), 2: ('shift', 3)},
    # Depth four (M5, 3 Oct 08:05): the fourth class and the wants that need it.
    'lD':    {3: ('shift', 7)},
    'lCD':   {2: ('shift', 3), 3: ('shift', 7)},
    'lABCD': {0: ('shift', 3), 1: ('shift', 7), 2: ('shift', 3), 3: ('shift', 7)},
    # Depth five (M6, 3 Oct 10:05): the fifth class and the wants that climb to it.
    'lE':     {4: ('shift', 3)},
    'lABCDE': {0: ('shift', 3), 1: ('shift', 7), 2: ('shift', 3), 3: ('shift', 7), 4: ('shift', 3)},
}

# KEYS AWAY FROM THE EDGE (5 Oct 2026, the board-10,000 reads): with 4 and 8 keys the rule below put class 3's key on
# the last slice of the item number, which the search's folding at the bound makes easy to hold, so learners piled onto
# it. CIPHER_EDGE_FREE = 1 moves class 3's set-1 key inside (0.700 of the range instead of 0.912); the top slice is
# then a key no good needs. Unset: the module is exactly as before.
EDGE_FREE = bool(int(os.environ.get('CIPHER_EDGE_FREE', '0') or 0))
if SUBST_K > 0:
    # four keys spread over 1..K by a fixed rule, distinct for every K >= 4
    _HK = [1 + (SUBST_K * f) // 1000 for f in (137, 421, 689, (700 if EDGE_FREE else 912))]
    CORPORA.update({
        'hA': {0: ('subst', _HK[0])}, 'hB': {1: ('subst', _HK[1])},
        'hC': {2: ('subst', _HK[2])}, 'hD': {3: ('subst', _HK[3])},
        'hAB': {0: ('subst', _HK[0]), 1: ('subst', _HK[1])},
        'hCD': {2: ('subst', _HK[2]), 3: ('subst', _HK[3])},
        'hABCD': {0: ('subst', _HK[0]), 1: ('subst', _HK[1]), 2: ('subst', _HK[2]), 3: ('subst', _HK[3])},
    })
# VARIED KEYS (her yes, 5 Oct 2026: "do it by the math, fix the cipher world"; hiring must pay because tuning costs
# compute, not because space runs out). CIPHER_KEYSETS = 2 adds a second key on every class, so the same class needs a
# different key in different goods: the set-2 goods hA2..hABCD2 mirror the hard goods. A designated default (the key
# the goods demand most) is then right for some goods and wrong for others. Unset or 1: the module is exactly as before.
KEYSETS = int(os.environ.get('CIPHER_KEYSETS', '1') or 1)
if SUBST_K > 0 and KEYSETS >= 2:
    _HK2 = [1 + (SUBST_K * f) // 1000 for f in (271, 563, 812, 47)]
    assert all(a != b for a, b in zip(_HK, _HK2)), 'set-2 keys must differ from set-1 keys class by class'
    CORPORA.update({
        'hA2': {0: ('subst', _HK2[0])}, 'hB2': {1: ('subst', _HK2[1])},
        'hC2': {2: ('subst', _HK2[2])}, 'hD2': {3: ('subst', _HK2[3])},
        'hAB2': {0: ('subst', _HK2[0]), 1: ('subst', _HK2[1])},
        'hCD2': {2: ('subst', _HK2[2]), 3: ('subst', _HK2[3])},
        'hABCD2': {0: ('subst', _HK2[0]), 1: ('subst', _HK2[1]), 2: ('subst', _HK2[2]), 3: ('subst', _HK2[3])},
    })
# TWO LAYERS PER CLASS (6 Oct 2026, her question: why is complexification not easy where the problem can be made harder
# and the parts reusable?): a key that is hard to FIND but easy to KEEP. CIPHER_LAYERS = 2 enciphers each class of the
# varied-key goods with two substitution layers, so a class is read only by the right ORDERED PAIR of decoders
# (1 in K*K by chance), while each decoder is one of only 4*K leaves (a wide, holdable slice). The second layer's keys
# are in CORPORA2. Defined for K = 4 with the edge-free keys. Unset or 1: the module is exactly as before.
LAYERS = int(os.environ.get('CIPHER_LAYERS', '1') or 1)
CORPORA2 = {}
if LAYERS >= 2:
    assert SUBST_K == 4 and KEYSETS >= 2 and EDGE_FREE, 'CIPHER_LAYERS=2 is defined for CIPHER_SUBST_K=4, KEYSETS=2, EDGE_FREE=1'
    # first layers: _HK (set 1, edge-free: 1, 2, 3, 3) and _HK2 (set 2: 2, 3, 4, 1); second layers below. No needed
    # pair uses class 3's key 4 (the top edge of the item number), and the set-1 and set-2 pairs differ in every class.
    _L2 = [3, 4, 1, 2]
    _L2b = [4, 1, 2, 1]
    for _g, _layers in list(CORPORA.items()):
        if _g.startswith('h'):
            _second = _L2b if _g.endswith('2') else _L2
            CORPORA2[_g] = {c: ('subst', _second[c]) for c in _layers}


def required_leaves(name):
    """Every decoder leaf ('c<class>_k<key>') a good needs, both layers."""
    out = {'c%d_k%d' % (c, k) for c, (kind, k) in CORPORA[name].items()}
    out |= {'c%d_k%d' % (c, k) for c, (kind, k) in CORPORA2.get(name, {}).items()}
    return out


# THE MERGE (her yes, 6 Oct 2026 04:42, with the two layers): CIPHER_MERGE = 1 adds one two-input part,
# decoder_merge_c01(X, Y): the letters of classes 0 and 1 from X and every other letter from Y. It joins a product that
# reads classes 0 and 1 with one that reads classes 2 and 3, so a whole can be assembled from two bought halves by a
# learner that holds no decoder of its own ("sell merge, buy, buy": 3 rows). It holds no key, so it serves both key
# sets. Its kind takes MERGE_WEIGHT beside the substitution kind's 0.3, so the decoders' slices narrow by about 9%.
# Unset: the module is exactly as before.
MERGE = bool(int(os.environ.get('CIPHER_MERGE', '0') or 0))
MERGE_FROM_X = (0, 1)
MERGE_WEIGHT = 0.03


def _merge(X, Y):
    """classes 0 and 1 from X, the rest from Y; [] for anything not two texts of one shape"""
    if not isinstance(X, (list, tuple)) or not isinstance(Y, (list, tuple)) or len(X) != len(Y):
        return []
    out = []
    for x, y in zip(X, Y):
        if not isinstance(x, str) or not isinstance(y, str) or len(x) != len(y):
            return []
        n = 0
        chars = []
        for cx, cy in zip(x, y):
            lx, ly = cx in ALPHABET, cy in ALPHABET
            if lx != ly:
                return []
            if lx:
                chars.append(cx if n % P in MERGE_FROM_X else cy)
                n += 1
            else:
                chars.append(cx)
        out.append(''.join(chars))
    return out


if P >= 8:
    # DEPTH EIGHT (5 Oct 2026, the cipher's other difficulty setting): eight classes, keys 3 and 7 as in the
    # ladder, so every key stays easy to find and to keep, but the whole has 8 layers, more than the 6 rows a
    # learner's plan holds: it cannot be built alone (her 11 Sep condition: a team does better because a
    # learner is limited).
    CORPORA.update({
        'lF': {5: ('shift', 7)}, 'lG': {6: ('shift', 3)}, 'lH': {7: ('shift', 7)},
        'lEF': {4: ('shift', 3), 5: ('shift', 7)}, 'lGH': {6: ('shift', 3), 7: ('shift', 7)},
        'lEFGH': {4: ('shift', 3), 5: ('shift', 7), 6: ('shift', 3), 7: ('shift', 7)},
        'lABCDEFGH': {0: ('shift', 3), 1: ('shift', 7), 2: ('shift', 3), 3: ('shift', 7),
                      4: ('shift', 3), 5: ('shift', 7), 6: ('shift', 3), 7: ('shift', 7)},
    })

VOCAB = ('the market opens at dawn and the traders bring keys to the gate while '
         'every sign on the wall names a price that nobody reads twice until '
         'a buyer points and a seller answers with work that passes or fails '
         'under the lamp of the test so roles form where pointing meets making '
         'river stone bread salt copper wheel cloth grain iron lantern road'
         ).split()


_PERM_MEMO = {}


def _perm(k):
    """A seeded derangement of the alphabet (no letter maps to itself), so a
    wrong substitution never earns coincidental credit. Memoised (hard keys)."""
    if k in _PERM_MEMO:
        return _PERM_MEMO[k]
    _PERM_MEMO[k] = _perm_build(k)
    return _PERM_MEMO[k]


def _perm_build(k):
    r = random.Random(1000 + k)
    letters = list(ALPHABET)
    while True:
        r.shuffle(letters)
        if all(a != b for a, b in zip(letters, ALPHABET)):
            return ''.join(letters)


def _layer_maps(kind, key):
    """Return (encode, decode) functions on (letter_index, ordinal_in_class)."""
    if kind == 'shift':
        return (lambda i, j: (i + key) % 26, lambda i, j: (i - key) % 26)
    if kind == 'subst':
        # built on first use, so a world with K keys does not compute K permutations at start
        tab = {}

        def _tabs():
            if not tab:
                enc = [ALPHABET.index(c) for c in _perm(key)]    # plaintext index -> cipher index
                dec = [0] * 26
                for a, b in enumerate(enc):
                    dec[b] = a
                tab['e'], tab['d'] = enc, dec
            return tab
        return (lambda i, j: _tabs()['e'][i], lambda i, j: _tabs()['d'][i])
    if kind == 'vig':
        w = [ALPHABET.index(c) for c in VIG_WORDS[key]]
        n = len(w)
        return (lambda i, j: (i + w[j % n]) % 26, lambda i, j: (i - w[j % n]) % 26)
    raise ValueError(kind)


def _apply(lines, cls, fn):
    """Apply fn(letter_index, ordinal_in_class) to the letters of class cls."""
    out = []
    for line in lines:
        n = 0
        chars = []
        for ch in line:
            if ch in ALPHABET:
                if n % P == cls:
                    chars.append(ALPHABET[fn(ALPHABET.index(ch), n // P)])
                else:
                    chars.append(ch)
                n += 1
            else:
                chars.append(ch)
        out.append(''.join(chars))
    return out


def plaintext(name):
    r = random.Random(hash_seed(name))
    return [' '.join(r.choice(VOCAB) for _ in range(r.randint(8, 12)))
            for _ in range(LINES_PER_CORPUS)]


def hash_seed(name):
    h = 7
    for ch in name:
        h = (h * 131 + ord(ch)) % 1000003
    return h


def ciphertext(name):
    lines = plaintext(name)
    for cls, (kind, key) in CORPORA[name].items():
        enc, _ = _layer_maps(kind, key)
        lines = _apply(lines, cls, enc)
    # the second layer (CIPHER_LAYERS = 2) is applied after the first, so it is undone first: the chain
    # [sell decoder(k1), construct decoder(k2)] applies k2's decoder, then k1's
    for cls, (kind, key) in CORPORA2.get(name, {}).items():
        enc, _ = _layer_maps(kind, key)
        lines = _apply(lines, cls, enc)
    return lines


def score(delivery, name):
    """Fraction of the corpus's enciphered letters now correct; 0.0 for anything
    that is not a list of strings of the corpus's shape."""
    try:
        truth = plaintext(name)
        if not isinstance(delivery, (list, tuple)) or len(delivery) != len(truth):
            return 0.0
        enc_classes = set(CORPORA[name].keys())
        n_enc = correct = 0
        for d, t in zip(delivery, truth):
            if not isinstance(d, str) or len(d) != len(t):
                return 0.0
            n = 0
            for cd, ct in zip(d, t):
                if ct in ALPHABET:
                    if n % P in enc_classes:
                        n_enc += 1
                        if cd == ct:
                            correct += 1
                    n += 1
        if n_enc == 0:
            return 0.0
        _frac = max(0.0, min(1.0, correct / float(n_enc)))
        if SUBST_K > 0:
            # Hard keys: with many substitution keys, a wrong key unscrambles some common letters by coincidence
            # (mean 0.04, up to 0.31 of a layer over 200 wrong keys), a bumpy landscape the module's rule forbids
            # ("no wrong key earns coincidental credit"). So a layer counts only when its letters are all right,
            # and the score is the share of right layers: the additive gradient across layers is kept.
            per = {c: [0, 0] for c in enc_classes}
            for d, t in zip(delivery, truth):
                n = 0
                for cd, ct in zip(d, t):
                    if ct in ALPHABET:
                        if n % P in per:
                            per[n % P][0] += 1
                            per[n % P][1] += 1 if cd == ct else 0
                        n += 1
            right = sum(1 for tot, ok in per.values() if tot and ok >= 0.99 * tot)
            _frac = right / float(len(per))
        if os.environ.get('CIPHER_STRICT_JUDGE'):
            # THE STRICTER JUDGE (her yes, 6 Oct 2026 05:00, "sure you can try it"): a decoder applied to a class the
            # good leaves plain makes the delivery wrong. Each good then needs its own exact composition: a section
            # that reads more classes is not right for the smaller goods, so they no longer give a lone builder a
            # ladder to the whole. (The judge set aside on 3 Oct, when one-layer goods were single decoders.)
            for d, t in zip(delivery, truth):
                n = 0
                for cd, ct in zip(d, t):
                    if ct in ALPHABET:
                        if n % P not in enc_classes and cd != ct:
                            return 0.0
                        n += 1
        if os.environ.get('CIPHER_WHOLE_AON') and len(enc_classes) >= 4:
            # THE WHOLE ALL-OR-NOTHING (6 Oct 2026, proposed to her; off unless set): a whole of four or more classes
            # counts only when every class is right, so a lone builder has no partial credit to climb, while the one-
            # and two-class goods keep their own credit.
            return 1.0 if _frac >= 0.999 else 0.0
        if os.environ.get('CIPHER_ALL_OR_NOTHING'):
            # The gradient ablation (3 Oct 2026, her question: how much does the
            # additive score buy?): a delivery scores 1 only when every layer is
            # right, else 0. Partial decodes earn nothing.
            return 1.0 if _frac >= 0.999 else 0.0
        return _frac
    except Exception:
        return 0.0


def cache_salt():
    """A cache namespace for every cipher result: the hash of this module's source plus
    the position-class count, so a result computed under other material or other code
    can never be returned for this world."""
    import hashlib
    try:
        src = open(__file__, 'rb').read()
    except Exception:
        src = b''
    return 'cw-%s-P%d%s%s%s%s%s' % (hashlib.sha1(src).hexdigest()[:12], P,
                              '-aon' if os.environ.get('CIPHER_ALL_OR_NOTHING') else '',
                              ('-K%d' % SUBST_K) if SUBST_K > 0 else '',
                              ('-S%d' % KEYSETS) if (SUBST_K > 0 and KEYSETS >= 2) else '',
                              '-E' if (SUBST_K > 0 and EDGE_FREE) else '',
                              ('-L%d' % LAYERS if LAYERS >= 2 else '') + ('-W' if os.environ.get('CIPHER_WHOLE_AON') else '')
                              + ('-J' if os.environ.get('CIPHER_STRICT_JUDGE') else '') + ('-M' if MERGE else ''))


# ---- registry entries ------------------------------------------------------

def _make_decoder(kind, cls, key):
    _, dec = _layer_maps(kind, key)

    def decoder(X):
        if not isinstance(X, (list, tuple)):
            return []
        return _apply([s if isinstance(s, str) else '' for s in X], cls, dec)
    decoder.__name__ = 'decoder_%s_c%d_k%d' % (kind, cls, key)
    return decoder


def _make_data(name):
    def data():
        return ciphertext(name)
    data.__name__ = 'data_cipher_%s' % name
    return data


def test_decoder_fractionCorrect(X):
    """The want's judge. The auditing buyer's corpus is published by the engine
    in Registry.PORT_BUYER_DATA (data_cipher_<name>); without it nothing can be
    judged and the score is 0.0."""
    try:
        from simulation import Registry as _R
    except ImportError:
        import Registry as _R
    data = getattr(_R, 'PORT_BUYER_DATA', None) or ''
    name = str(data).replace('data_cipher_', '', 1)
    if name not in CORPORA:
        return 0.0
    return float(score(X, name))


def leaves():
    """(name, function) for every product leaf, data terminal and test."""
    out = []
    for kind, keys in (('shift', SHIFT_KEYS), ('subst', SUBST_KEYS), ('vig', tuple(VIG_WORDS))):
        for cls in range(P):
            for key in keys:
                f = _make_decoder(kind, cls, key)
                out.append((f.__name__, f))
                # THE CLASS NAMED BY ITS LETTER (her question 7 Oct 2026: "why can't the item path name the corpus in
                # the cipher world like it does in the ML world"): a corpus is named by the classes it contains (hA,
                # hAB), so a decoder named decoder_subst_A_k2 decodes the A cipher of any corpus with an A in its name,
                # as labeler_movieSentiment names its corpus. The same function under both names; the world's
                # ontology picks which name its tree shows (ontology(class_names='letters')).
                out.append((f.__name__.replace('_c%d_' % cls, '_%s_' % CLASS_LETTERS[cls]), f))
    for name in CORPORA:
        f = _make_data(name)
        out.append((f.__name__, f))
    if MERGE:
        def decoder_merge_c01(X, Y):
            return _merge(X, Y)
        out.append(('decoder_merge_c01', decoder_merge_c01))
    out.append(('test_decoder_fractionCorrect', test_decoder_fractionCorrect))
    return out


def entries(curr):
    return {name: curr(fn) for name, fn in leaves()}


# ---- ontology and blackboard -----------------------------------------------

_STR_LIST = [{'type': 'list', 'dtype': 'str'}]


def _node(args, **kw):
    d = {'_args': args, '_weight': 0.3, '_deterministic': True}
    d.update(kw)
    return d


CLASS_LETTERS = 'ABCDEFGH'


def ontology(types=None, classes=None, keys=None, corpora=None, class_names='numbers'):
    """The full tree by default; a world may restrict the layer types, the
    position classes and the keys its learners can express (the registry keeps
    every leaf; the tree decides what a genome can decode to). corpora: the
    data leaves a learner's test can name (7 Oct 2026: the ladder worlds had
    every corpus of the module in their data branch, 21 of them legacy ones
    that no good of the world uses, so a learner's test on the category
    data_cipher ran a bought part over all 28 and paid it coincidental letter
    credit; a world now names its own goods' corpora and nothing else)."""
    test = _node(_STR_LIST, _kind='test')
    test['decoder'] = _node(_STR_LIST)
    test['decoder']['fractionCorrect'] = _node(_STR_LIST, _return=[{'type': 'float'}],
                                               _comment='test_decoder_fractionCorrect')
    data = _node([], _kind='data')
    data['cipher'] = _node([])
    for name in CORPORA:
        if corpora is not None and name not in corpora:
            continue
        data['cipher'][name] = _node([], _return=_STR_LIST, _comment='data_cipher_%s' % name)
    dec = _node(_STR_LIST, _kind='product')
    for kind, kkeys in (('shift', SHIFT_KEYS), ('subst', SUBST_KEYS), ('vig', tuple(VIG_WORDS))):
        if types is not None and kind not in types:
            continue
        dec[kind] = _node(_STR_LIST)
        for cls in range(P):
            if classes is not None and cls not in classes:
                continue
            cn = CLASS_LETTERS[cls] if class_names == 'letters' else 'c%d' % cls
            dec[kind][cn] = _node(_STR_LIST)
            for key in kkeys:
                if keys is not None and key not in keys:
                    continue
                dec[kind][cn]['k%d' % key] = _node(_STR_LIST, _return=_STR_LIST,
                                                   _comment='decoder_%s_%s_k%d' % (kind, cn, key))
    if MERGE:
        dec['merge'] = _node(_STR_LIST + _STR_LIST)
        dec['merge']['_weight'] = MERGE_WEIGHT
        dec['merge']['c01'] = _node(_STR_LIST + _STR_LIST, _return=_STR_LIST, _comment='decoder_merge_c01')
    return {'test': test, 'data': data, 'decoder': dec}


PRICES = {1: 10, 2: 30, 3: 90, 4: 120, 5: 150, 8: 480}   # 8: the depth-eight whole (x4 per doubling, as 30 -> 120)


def blackboard(corpora=None, buyers_per_corpus=1):
    """One authored buyer per corpus by default, priced by depth (10, 30, 90:
    mine); a world may name its corpora and ask for several buyers each."""
    out = []
    for name, layers in CORPORA.items():
        if corpora is not None and name not in corpora:
            continue
        depth = len(layers)
        desc = ', '.join('c%d %s%d' % (c, k, v) for c, (k, v) in sorted(layers.items()))
        _n = buyers_per_corpus.get(name, 0) if isinstance(buyers_per_corpus, dict) else int(buyers_per_corpus)
        for copy_k in range(int(_n)):
          out.append({
            'type': 'RoleSim', 'distributes': True,
              'label': 'Cipher %d-layer: %s (%s)%s' % (depth, name, desc, (' buyer %d' % copy_k) if _n > 1 else ''),
              'trades': [
                  {'type': 'buy', 'item': 'decoder_stop', 'midpoint': PRICES[depth],
                   'tests': [{'stophere': False, 'test': 'test_decoder_fractionCorrect',
                              'data': 'data_cipher_%s' % name, 'hidden': False},
                             {'stophere': True}],
                   'alpha': 0.7, 'budget': 24.0},
                  {'type': 'stop'}],
              '_seed_provenance': 'cipher world, her July 2026 instrument, built 3 Oct 2026'})
    return out


if __name__ == '__main__':
    # Self-test: every corpus round-trips through its inverse layers; one right
    # layer of three scores a third; junk scores zero.
    for name, layers in CORPORA.items():
        if max(layers) >= P:
            continue                      # needs more position classes than this world has
        c = ciphertext(name)
        assert c != plaintext(name)
        d = c
        for cls, (kind, key) in layers.items():
            d = _make_decoder(kind, cls, key)(d)
        assert d == plaintext(name), name
        assert abs(score(d, name) - 1.0) < 1e-9
        assert score(c, name) == 0.0
        cls, (kind, key) = sorted(layers.items())[0]
        one = _make_decoder(kind, cls, key)(c)
        plain_cls = [x for x in range(P) if x not in layers]
        dmg = _make_decoder('shift', plain_cls[0], 3)(one) if plain_cls else None
        print('%-7s depth %d  round trip ok  junk %.3f  one right layer %.3f%s' % (
            name, len(layers), score(c, name), score(one, name),
            ('  right layer plus a wrong layer on a plain class %.3f' % score(dmg, name)) if dmg else ''))
    print(len(leaves()), 'registry entries;', len(blackboard()), 'authored buyers')
