"""The cipher world: complexification by dominance argument, on the
GCoN engine (spec: docs/cipher_world_spec.pdf in the llm_market repo).

Problems are keyed cipher classes. A class's key is a tuple of subkeys;
a decryption attempt names a full key guess (an ontology leaf whose
path spells the guess, so the genome's hierarchical decode searches
subkey-by-subkey — separable, graded search). Verification is instant
and graded: the fraction of subkeys correct. Once an agent's guess is
right it stays right — knowledge is capital, reusable at zero cost.

Tiers:
  tier 1: 1 subkey of 32   — solved solo, fast (the entry economy)
  tier 2: 2 subkeys of 24  — harder solo climbs
  tier 3: 1 subkey of 32 wrapped OVER a tier-1 class — the attempt takes
          the INNER plaintext as its input, so serving tier-3 demand
          requires owning or BUYING the inner decryption: every tier-3
          settle is structurally a supply chain (hiring, depth).

Secret keys are seeded constants: the world's ground truth, never in
any agent's reach except by search. The tests are the verifiers.
"""
import random as _random

# ---------------------------------------------------------- ground truth
_rng = _random.Random('cipher-world-keys-v1')
T1_S, T2_S, T3_S = 32, 24, 32
KEYS = {
    # tier 1: four classes, one subkey each
    't1c1': (_rng.randrange(T1_S),),
    't1c2': (_rng.randrange(T1_S),),
    't1c3': (_rng.randrange(T1_S),),
    't1c4': (_rng.randrange(T1_S),),
    # tier 2: four classes, two subkeys each
    't2c1': (_rng.randrange(T2_S), _rng.randrange(T2_S)),
    't2c2': (_rng.randrange(T2_S), _rng.randrange(T2_S)),
    't2c3': (_rng.randrange(T2_S), _rng.randrange(T2_S)),
    't2c4': (_rng.randrange(T2_S), _rng.randrange(T2_S)),
    # tier 3: four classes, one outer subkey each, wrapped over a tier-1
    # class (the inner layer whose plaintext the attempt needs)
    't3c1': (_rng.randrange(T3_S),),
    't3c2': (_rng.randrange(T3_S),),
    't3c3': (_rng.randrange(T3_S),),
    't3c4': (_rng.randrange(T3_S),),
}
T3_INNER = {'t3c1': 't1c1', 't3c2': 't1c2', 't3c3': 't1c3', 't3c4': 't1c4'}


def _guess_from_name(name, n_sub):
    """A decryptor leaf's name spells the guess: ..._k7 or ..._k7_k12."""
    parts = name.split('_')
    vals = [int(p[1:]) for p in parts if p.startswith('k') and p[1:].isdigit()]
    return tuple(vals[-n_sub:]) if len(vals) >= n_sub else None


def _score_guess(cls, guess):
    key = KEYS[cls]
    if not guess or len(guess) != len(key):
        return 0.0
    return sum(1 for g, k in zip(guess, key) if g == k) / float(len(key))


# ------------------------------------------------------------- the data
# Each class's "messages" are represented by their class marker; the
# attempt product carries (class, guess[, inner]) and the test verifies.
def _make_data(cls):
    def data():
        return ('CIPHERTEXT', cls)
    data.__name__ = 'data_cipher_' + cls
    return data


# --------------------------------------------------------- the attempts
def _make_attempt(cls, leafname, n_sub):
    guess = _guess_from_name(leafname, n_sub)

    def attempt(D):
        # D is the class's ciphertext data terminal.
        if not (isinstance(D, tuple) and len(D) == 2 and D[1] == cls):
            return None
        return ('PLAINTEXT_ATTEMPT', cls, guess)
    attempt.__name__ = leafname
    return attempt


def _make_t3_attempt(cls, leafname):
    guess = _guess_from_name(leafname, 1)
    inner_cls = T3_INNER[cls]

    def attempt(inner):
        # arity 1: the INNER product — a tier-1 plaintext attempt. Serving
        # this demand requires having (built or bought) the inner layer.
        if not (isinstance(inner, tuple) and inner[0] == 'PLAINTEXT_ATTEMPT'
                and inner[1] == inner_cls):
            return None
        inner_quality = _score_guess(inner_cls, inner[2])
        return ('PLAINTEXT_ATTEMPT_T3', cls, guess, inner_quality)
    attempt.__name__ = leafname
    return attempt


# ------------------------------------------------------------ the tests
def _make_test(cls, tier):
    def test(P):
        if tier < 3:
            if not (isinstance(P, tuple) and P[0] == 'PLAINTEXT_ATTEMPT'
                    and P[1] == cls):
                return ('', 0.0)
            return ('', _score_guess(cls, P[2]))
        if not (isinstance(P, tuple) and P[0] == 'PLAINTEXT_ATTEMPT_T3'
                and P[1] == cls):
            return ('', 0.0)
        outer = _score_guess(cls, P[2])
        # The full read requires BOTH layers: graded as the product of
        # the outer guess quality and the inner plaintext quality.
        return ('', outer * P[3])
    test.__name__ = 'test_cipher_' + cls
    return test


# ----------------------------------------------------------- the export
def build_cipher_registry(curr):
    reg = {}
    leaves = {}     # ontology construction info: family -> [leafnames]
    for cls in ('t1c1', 't1c2', 't1c3', 't1c4'):
        reg['data_cipher_' + cls] = curr(_make_data(cls))
        reg['test_cipher_' + cls] = curr(_make_test(cls, 1))
        fam = []
        for k in range(T1_S):
            name = f'decryptor_{cls}_k{k}'
            reg[name] = curr(_make_attempt(cls, name, 1))
            fam.append(name)
        leaves[cls] = fam
    for cls in ('t2c1', 't2c2', 't2c3', 't2c4'):
        reg['data_cipher_' + cls] = curr(_make_data(cls))
        reg['test_cipher_' + cls] = curr(_make_test(cls, 2))
        fam = []
        for a in range(T2_S):
            for b in range(T2_S):
                name = f'decryptor_{cls}_k{a}_k{b}'
                reg[name] = curr(_make_attempt(cls, name, 2))
                fam.append(name)
        leaves[cls] = fam
    for cls in ('t3c1', 't3c2', 't3c3', 't3c4'):
        reg['test_cipher_' + cls] = curr(_make_test(cls, 3))
        fam = []
        for k in range(T3_S):
            name = f'decryptor_{cls}_k{k}'
            reg[name] = curr(_make_t3_attempt(cls, name))
            fam.append(name)
        leaves[cls] = fam
    return reg, leaves
