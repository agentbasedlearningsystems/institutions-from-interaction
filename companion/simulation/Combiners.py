"""a8: the branch-capable catalogue - products that take TWO or THREE
inputs (her instruction, 2026-08-12: "we should have more than 18, is it
possible to find more quality and sensible programs that have higher
arity? even 3?").

Why this exists. The wave 19 record held 5,889 delivered programs and
not one of them branched: every single one was a straight chain. That
was not a discovery failure. The catalogue held 343 products of which
exactly ONE took two inputs, it sat in the family the market used twice
in a thousand iterations, and it threw on contact. Meanwhile fifteen
living agents wanted something that needs two inputs by their own stated
want - conjunctive tests, cross-corpus buyers, one asking for all three
corpora at once - and nothing in the world could serve them.

Every product here is a real method rather than filler, and they are
organised by WHERE the streams join, which is how this is described in
the literature: early fusion (the corpora themselves), feature fusion
(the vector spaces), representation fusion (the clusterings), decision
fusion (the classifiers), score fusion (the anomaly detectors), and
criterion fusion (the tests, which is literally what the CONJUNCTIVE
agents are named for).

Every one is deterministic. Every one returns None - a market failure,
never a crash - on missing, mis-shaped or degenerate input.

HER DIVERSITY LAW, kept from clusterer_consensus: "a combination of
clones is not a combination." A combiner handed two identical inputs
refuses. Memoisation makes the same chain bought twice byte-identical,
so without this the cheapest way to look composed would be to buy one
thing twice.

Types, as the cleaned catalogue declares them:
    text        list of str
    matrix      numpy.ndarray of float
    labels      (X, y)
    predictions (X, y, pred)
    scores      (score array, threshold)
    float       a test score
"""
import numpy as np


# ----------------------------------------------------------------- helpers
def _same(a, b):
    """Her diversity law: two identical inputs are not a combination."""
    try:
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return a.shape == b.shape and np.array_equal(a, b)
        return a is b or a == b
    except Exception:
        return False


def _mats(*xs):
    out = []
    for x in xs:
        if x is None or not isinstance(x, np.ndarray) or x.ndim != 2 or x.size == 0:
            return None
        out.append(np.asarray(x, dtype=np.float64))
    n = min(m.shape[0] for m in out)
    if n < 2:
        return None
    return [m[:n] for m in out]


def _labels(*xs):
    """each x is (X, y); returns (X, [y...]) aligned on rows"""
    Xs, ys = [], []
    for x in xs:
        if not isinstance(x, tuple) or len(x) < 2 or x[0] is None or x[1] is None:
            return None
        Xs.append(np.asarray(x[0]))
        ys.append(np.asarray(x[1]).ravel())
    n = min(min(a.shape[0] for a in Xs), min(len(v) for v in ys))
    if n < 2:
        return None
    return Xs[0][:n], [v[:n] for v in ys]


def _preds(*xs):
    """each x is (X, y, pred)"""
    Xs, ys, ps = [], [], []
    for x in xs:
        if not isinstance(x, tuple) or len(x) < 3 or x[2] is None:
            return None
        Xs.append(np.asarray(x[0]))
        ys.append(np.asarray(x[1]).ravel())
        ps.append(np.asarray(x[2]).ravel())
    n = min(min(a.shape[0] for a in Xs), min(len(v) for v in ps))
    if n < 2:
        return None
    return Xs[0][:n], ys[0][:n], [v[:n] for v in ps]


def _scores(*xs):
    ss, ts = [], []
    for x in xs:
        if not isinstance(x, tuple) or len(x) < 2 or x[0] is None:
            return None
        ss.append(np.asarray(x[0], dtype=np.float64).ravel())
        try:
            ts.append(float(x[1]))
        except (TypeError, ValueError):
            ts.append(0.0)
    n = min(len(v) for v in ss)
    if n < 2:
        return None
    return [v[:n] for v in ss], ts


def _overlap(*xs):
    """True if any two inputs share documents/rows. A fusion of clones is not
    a combination (the rule at the top of this file); merging a corpus with
    itself, or with a merge that already contains it, only multiplies rows -
    nested merges reached 5.6 million 'documents' (668 MB per delivery) and
    exhausted the machine on 2026-09-12. Sample-based: 64 rows of each input
    looked up in the others."""
    try:
        keys = []
        for x in xs:
            if isinstance(x, np.ndarray):
                keys.append(set(hash(r.tobytes()) for r in np.ascontiguousarray(x)))
            else:
                keys.append(set(hash(str(r)) for r in x))
        for i in range(len(xs)):
            xi = xs[i]
            sample = (list(xi[:64]) if not isinstance(xi, np.ndarray)
                      else list(np.ascontiguousarray(xi[:64])))
            for j in range(len(xs)):
                if i == j:
                    continue
                for r in sample:
                    h = hash(r.tobytes()) if isinstance(r, np.ndarray) else hash(str(r))
                    if h in keys[j]:
                        return True
        return False
    except Exception:
        return True

def _texts(*xs):
    out = []
    for x in xs:
        if not isinstance(x, list) or not x:
            return None
        out.append(x)
    return out


def _floats(*xs):
    out = []
    for x in xs:
        if x is None:
            return None
        try:
            out.append(float(x))
        except (TypeError, ValueError):
            return None
    return out


def _align(a, b):
    """Relabel b's cluster ids to agree with a where possible (a vote over
    two clusterings is meaningless while the ids are arbitrary)."""
    out = np.full(len(b), -1, dtype=np.int64)
    for lab in np.unique(b):
        m = b == lab
        if not m.any():
            continue
        vals, counts = np.unique(a[m], return_counts=True)
        out[m] = vals[int(np.argmax(counts))]
    return out


# -------------------------------------------------- EARLY FUSION (corpora)
def data_fusion_concat2(a, b):
    t = _texts(a, b)
    if t is None or _same(a, b):
        return None
    if _overlap(a, b):
        return None
    return list(t[0]) + list(t[1])


def data_fusion_concat3(a, b, c):
    t = _texts(a, b, c)
    if t is None or (_same(a, b) and _same(b, c)):
        return None
    if _overlap(a, b, c):
        return None
    return list(t[0]) + list(t[1]) + list(t[2])


def data_fusion_interleave2(a, b):
    t = _texts(a, b)
    if t is None or _same(a, b):
        return None
    out = []
    for i in range(max(len(t[0]), len(t[1]))):
        if i < len(t[0]):
            out.append(t[0][i])
        if i < len(t[1]):
            out.append(t[1][i])
    if _overlap(a, b):
        return None
    return out


def data_fusion_rowstack2(a, b):
    m = _mats(a, b)
    if m is None or _same(a, b) or m[0].shape[1] != m[1].shape[1]:
        return None
    if _overlap(a, b):
        return None
    return np.vstack([m[0], m[1]]).astype(np.float32)


def data_fusion_rowstack3(a, b, c):
    m = _mats(a, b, c)
    if m is None or (_same(a, b) and _same(b, c)):
        return None
    w = m[0].shape[1]
    if any(x.shape[1] != w for x in m):
        return None
    if _overlap(a, b, c):
        return None
    return np.vstack(m).astype(np.float32)


# ------------------------------------------------------- TEXT FUSION
def preprocessor_fusion_merge2(a, b):
    t = _texts(a, b)
    if t is None or _same(a, b):
        return None
    if _overlap(a, b):
        return None
    return list(t[0]) + list(t[1])


def preprocessor_fusion_merge3(a, b, c):
    t = _texts(a, b, c)
    if t is None or (_same(a, b) and _same(b, c)):
        return None
    if _overlap(a, b, c):
        return None
    return list(t[0]) + list(t[1]) + list(t[2])


def preprocessor_fusion_dedupUnion2(a, b):
    t = _texts(a, b)
    if t is None or _same(a, b):
        return None
    seen, out = set(), []
    for doc in list(t[0]) + list(t[1]):
        k = str(doc)
        if k not in seen:
            seen.add(k)
            out.append(doc)
    if _overlap(a, b):
        return None
    return out


# ---------------------------------------------------- FEATURE FUSION
def _fuse_cols(ms):
    return np.hstack(ms).astype(np.float32)


def vectorSpace_fusion_concat2(a, b):
    m = _mats(a, b)
    return None if m is None or _same(a, b) else _fuse_cols(m)


def vectorSpace_fusion_concat3(a, b, c):
    m = _mats(a, b, c)
    return None if m is None or (_same(a, b) and _same(b, c)) else _fuse_cols(m)


def _blend(ms, ws):
    w = np.asarray(ws, dtype=np.float64)
    w = w / w.sum()
    k = min(x.shape[1] for x in ms)
    return sum(wi * x[:, :k] for wi, x in zip(w, ms)).astype(np.float32)


def vectorSpace_fusion_mean2(a, b):
    m = _mats(a, b)
    return None if m is None or _same(a, b) else _blend(m, [1, 1])


def vectorSpace_fusion_mean3(a, b, c):
    m = _mats(a, b, c)
    return None if m is None or (_same(a, b) and _same(b, c)) else _blend(m, [1, 1, 1])


def vectorSpace_fusion_blend25(a, b):
    m = _mats(a, b)
    return None if m is None or _same(a, b) else _blend(m, [0.25, 0.75])


def vectorSpace_fusion_blend75(a, b):
    m = _mats(a, b)
    return None if m is None or _same(a, b) else _blend(m, [0.75, 0.25])


def vectorSpace_fusion_max2(a, b):
    m = _mats(a, b)
    if m is None or _same(a, b):
        return None
    k = min(x.shape[1] for x in m)
    return np.maximum(m[0][:, :k], m[1][:, :k]).astype(np.float32)


def vectorSpace_fusion_median3(a, b, c):
    m = _mats(a, b, c)
    if m is None or (_same(a, b) and _same(b, c)):
        return None
    k = min(x.shape[1] for x in m)
    return np.median(np.stack([x[:, :k] for x in m]), axis=0).astype(np.float32)


def vectorSpace_fusion_residual2(a, b):
    m = _mats(a, b)
    if m is None or _same(a, b):
        return None
    k = min(x.shape[1] for x in m)
    return (m[0][:, :k] - m[1][:, :k]).astype(np.float32)


def _l2(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def vectorSpace_fusion_normConcat2(a, b):
    m = _mats(a, b)
    return None if m is None or _same(a, b) else _fuse_cols([_l2(m[0]), _l2(m[1])])


def vectorSpace_fusion_reduceConcat2(a, b):
    """Join two views and compress the join - feature fusion followed by
    a reduction, the standard multi-view recipe."""
    m = _mats(a, b)
    if m is None or _same(a, b):
        return None
    from sklearn.decomposition import TruncatedSVD
    j = _fuse_cols(m)
    k = int(min(20, min(j.shape) - 1))
    if k < 2:
        return None
    return TruncatedSVD(n_components=k).fit_transform(j).astype(np.float32)


def vectorSpace_fusion_reduceConcat3(a, b, c):
    m = _mats(a, b, c)
    if m is None or (_same(a, b) and _same(b, c)):
        return None
    from sklearn.decomposition import TruncatedSVD
    j = _fuse_cols(m)
    k = int(min(20, min(j.shape) - 1))
    if k < 2:
        return None
    return TruncatedSVD(n_components=k).fit_transform(j).astype(np.float32)


def vectorSpace_fusion_cca2(a, b):
    """Canonical correlation of two views: the subspace on which the two
    representations agree."""
    m = _mats(a, b)
    if m is None or _same(a, b):
        return None
    from sklearn.cross_decomposition import CCA
    k = int(min(10, m[0].shape[1], m[1].shape[1], m[0].shape[0] - 1))
    if k < 1:
        return None
    u, v = CCA(n_components=k, max_iter=200).fit_transform(m[0], m[1])
    return np.hstack([u, v]).astype(np.float32)


def vectorSpace_fusion_procrustesMean2(a, b):
    """Rotate the second view onto the first, then average - a fusion
    that does not assume the two spaces share an orientation."""
    m = _mats(a, b)
    if m is None or _same(a, b):
        return None
    k = min(x.shape[1] for x in m)
    A, B = m[0][:, :k], m[1][:, :k]
    try:
        u, _, vt = np.linalg.svd(B.T @ A, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    return ((A + (B @ (u @ vt))) / 2.0).astype(np.float32)


# --------------------------------------------- REPRESENTATION FUSION
def _cons(Xy, ys):
    """co-association: agree where the clusterings agree, and give the
    disagreements their own class."""
    a, b = ys[0], ys[1]
    bb = _align(a, b)
    out = np.where(a == bb, a, -1).astype(np.int32)
    if len(np.unique(out)) < 2:
        return None
    return (Xy, out)


def clusterer_fusion_consensus2(a, b):
    L = _labels(a, b)
    if L is None or _same(a, b):
        return None
    return _cons(L[0], L[1])


def clusterer_fusion_agreementMask2(a, b):
    L = _labels(a, b)
    if L is None or _same(a, b):
        return None
    X, ys = L
    bb = _align(ys[0], ys[1])
    out = (ys[0] == bb).astype(np.int32)
    if len(np.unique(out)) < 2:
        return None
    return (X, out)


def _vote(ys):
    ref = ys[0]
    al = [ref] + [_align(ref, y) for y in ys[1:]]
    st = np.stack(al)
    out = np.zeros(st.shape[1], dtype=np.int32)
    for i in range(st.shape[1]):
        vals, counts = np.unique(st[:, i], return_counts=True)
        out[i] = vals[int(np.argmax(counts))]
    return out


def clusterer_fusion_vote3(a, b, c):
    L = _labels(a, b, c)
    if L is None or (_same(a, b) and _same(b, c)):
        return None
    X, ys = L
    out = _vote(ys)
    return None if len(np.unique(out)) < 2 else (X, out)


def clusterer_fusion_consensus3(a, b, c):
    L = _labels(a, b, c)
    if L is None or (_same(a, b) and _same(b, c)):
        return None
    X, ys = L
    ref = ys[0]
    al = [ref] + [_align(ref, y) for y in ys[1:]]
    agree = np.all(np.stack(al) == al[0], axis=0)
    out = np.where(agree, al[0], -1).astype(np.int32)
    return None if len(np.unique(out)) < 2 else (X, out)


def clusterer_fusion_coassoc2(a, b):
    """Cluster the co-association itself: points that the two clusterings
    put together stay together."""
    L = _labels(a, b)
    if L is None or _same(a, b):
        return None
    X, ys = L
    n = len(ys[0])
    if n > 1500:
        idx = np.arange(0, n, max(1, n // 1500))
        X2 = X[idx]
        ys = [y[idx] for y in ys]
    else:
        X2 = X
    co = ((ys[0][:, None] == ys[0][None, :]).astype(np.float64)
          + (ys[1][:, None] == ys[1][None, :]).astype(np.float64)) / 2.0
    from sklearn.cluster import AgglomerativeClustering
    k = int(max(2, min(8, len(np.unique(ys[0])))))
    lab = AgglomerativeClustering(n_clusters=k, metric='precomputed',
                                  linkage='average').fit_predict(1.0 - co)
    return (X2, lab.astype(np.int32))


def clusterer_fusion_coassoc3(a, b, c):
    L = _labels(a, b, c)
    if L is None or (_same(a, b) and _same(b, c)):
        return None
    X, ys = L
    n = len(ys[0])
    if n > 1200:
        idx = np.arange(0, n, max(1, n // 1200))
        X = X[idx]
        ys = [y[idx] for y in ys]
    co = sum((y[:, None] == y[None, :]).astype(np.float64) for y in ys) / float(len(ys))
    from sklearn.cluster import AgglomerativeClustering
    k = int(max(2, min(8, len(np.unique(ys[0])))))
    lab = AgglomerativeClustering(n_clusters=k, metric='precomputed',
                                  linkage='average').fit_predict(1.0 - co)
    return (X, lab.astype(np.int32))


# ---------------------------------------------------- LABEL FUSION
def labeler_fusion_vote2(a, b):
    L = _labels(a, b)
    if L is None or _same(a, b):
        return None
    X, ys = L
    bb = _align(ys[0], ys[1])
    out = np.where(ys[0] == bb, ys[0], -1).astype(np.int32)
    return None if len(np.unique(out)) < 2 else (X, out)


def labeler_fusion_vote3(a, b, c):
    L = _labels(a, b, c)
    if L is None or (_same(a, b) and _same(b, c)):
        return None
    X, ys = L
    out = _vote(ys)
    return None if len(np.unique(out)) < 2 else (X, out)


# -------------------------------------------------- DECISION FUSION
def _pred_out(X, y, p):
    return (X, y.astype(np.int32), p.astype(np.int32))


def classifier_fusion_vote3(a, b, c):
    P = _preds(a, b, c)
    if P is None or (_same(a, b) and _same(b, c)):
        return None
    X, y, ps = P
    out = np.full(len(y), -1, dtype=np.int32)
    held = np.all(np.stack([p >= 0 for p in ps]), axis=0)
    st = np.stack(ps)[:, held]
    if st.size == 0:
        return None
    vals = np.zeros(st.shape[1], dtype=np.int32)
    for i in range(st.shape[1]):
        v, c2 = np.unique(st[:, i], return_counts=True)
        vals[i] = v[int(np.argmax(c2))]
    out[held] = vals
    return _pred_out(X, y, out)


def classifier_fusion_majority3(a, b, c):
    """A genuine majority: two of three must agree, otherwise abstain.
    With two voters this cannot exist, which is why arity 3 earns its
    place rather than decorating."""
    P = _preds(a, b, c)
    if P is None or (_same(a, b) and _same(b, c)):
        return None
    X, y, ps = P
    out = np.full(len(y), -1, dtype=np.int32)
    held = np.all(np.stack([p >= 0 for p in ps]), axis=0)
    st = np.stack(ps)[:, held]
    if st.size == 0:
        return None
    vals = np.full(st.shape[1], -1, dtype=np.int32)
    for i in range(st.shape[1]):
        v, c2 = np.unique(st[:, i], return_counts=True)
        j = int(np.argmax(c2))
        if c2[j] >= 2:
            vals[i] = v[j]
    out[held] = vals
    return _pred_out(X, y, out)


def classifier_fusion_agreeOrAbstain2(a, b):
    """Answer only where both voters agree - precision bought with
    coverage, the honest two-voter combination."""
    P = _preds(a, b)
    if P is None or _same(a, b):
        return None
    X, y, ps = P
    out = np.where((ps[0] >= 0) & (ps[0] == ps[1]), ps[0], -1).astype(np.int32)
    return _pred_out(X, y, out)


def classifier_fusion_firstThenSecond2(a, b):
    """The second voter answers only where the first abstained - a
    cascade rather than a vote."""
    P = _preds(a, b)
    if P is None or _same(a, b):
        return None
    X, y, ps = P
    out = np.where(ps[0] >= 0, ps[0], ps[1]).astype(np.int32)
    return _pred_out(X, y, out)


def classifier_fusion_stack2(a, b):
    """Stacking: learn from the two voters' outputs which to trust
    where, instead of fixing the rule in advance."""
    P = _preds(a, b)
    if P is None or _same(a, b):
        return None
    X, y, ps = P
    held = (ps[0] >= 0) & (ps[1] >= 0) & (y >= 0)
    if held.sum() < 20:
        return None
    from sklearn.tree import DecisionTreeClassifier
    Z = np.stack([ps[0], ps[1]], axis=1)
    clf = DecisionTreeClassifier(max_depth=3, random_state=0)
    half = np.where(held)[0]
    cut = len(half) // 2
    if cut < 5:
        return None
    clf.fit(Z[half[:cut]], y[half[:cut]])
    out = np.full(len(y), -1, dtype=np.int32)
    out[half[cut:]] = clf.predict(Z[half[cut:]])
    return _pred_out(X, y, out)


# ------------------------------------------------------ SCORE FUSION
def _score_out(vals, ts):
    return (np.asarray(vals, dtype=np.float32), np.float32(float(np.mean(ts))))


def anomalyDetector_fusion_mean2(a, b):
    S = _scores(a, b)
    return None if S is None or _same(a, b) else _score_out((S[0][0] + S[0][1]) / 2.0, S[1])


def anomalyDetector_fusion_mean3(a, b, c):
    S = _scores(a, b, c)
    if S is None or (_same(a, b) and _same(b, c)):
        return None
    return _score_out(np.mean(np.stack(S[0]), axis=0), S[1])


def anomalyDetector_fusion_max2(a, b):
    S = _scores(a, b)
    if S is None or _same(a, b):
        return None
    return _score_out(np.maximum(S[0][0], S[0][1]), S[1])


def anomalyDetector_fusion_max3(a, b, c):
    S = _scores(a, b, c)
    if S is None or (_same(a, b) and _same(b, c)):
        return None
    return _score_out(np.max(np.stack(S[0]), axis=0), S[1])


def anomalyDetector_fusion_median3(a, b, c):
    """The median of three is the robust fusion, and it needs three
    inputs to exist at all."""
    S = _scores(a, b, c)
    if S is None or (_same(a, b) and _same(b, c)):
        return None
    return _score_out(np.median(np.stack(S[0]), axis=0), S[1])


def _ranks(v):
    order = np.argsort(np.argsort(v))
    return order / max(1.0, float(len(v) - 1))


def anomalyDetector_fusion_rankAvg2(a, b):
    """Average the RANKS, not the scores - two detectors on different
    scales can still be combined."""
    S = _scores(a, b)
    if S is None or _same(a, b):
        return None
    return _score_out((_ranks(S[0][0]) + _ranks(S[0][1])) / 2.0, S[1])


# -------------------------------------------------- CRITERION FUSION
def test_fusion_min2(a, b):
    """A conjunction: satisfy BOTH criteria. This is what the agents
    labelled CONJUNCTIVE are asking for by name."""
    f = _floats(a, b)
    return None if f is None else float(min(f))


def test_fusion_mean2(a, b):
    f = _floats(a, b)
    return None if f is None else float(sum(f) / 2.0)


def test_fusion_min3(a, b, c):
    f = _floats(a, b, c)
    return None if f is None else float(min(f))


def test_fusion_mean3(a, b, c):
    f = _floats(a, b, c)
    return None if f is None else float(sum(f) / 3.0)


def test_fusion_harmonic2(a, b):
    """The harmonic mean punishes a weak side harder than the arithmetic
    one - a demanding conjunction that still rewards balance."""
    f = _floats(a, b)
    if f is None or min(f) <= 0:
        return None
    return float(2.0 / (1.0 / f[0] + 1.0 / f[1]))


# --------------------------------------------------------- registration
TEXT = {"type": "list", "dtype": "str"}
MATRIX = {"type": "numpy.ndarray", "dtype": "float32"}
LABELS = {"type": "tuple", "dtype": "(float32,int32)"}
PREDS = {"type": "tuple", "dtype": "(float32,int32,int32)"}
SCORES = {"type": "tuple", "dtype": "(float32,float32)"}
FLOAT = {"type": "float"}

# name -> (function, arity, argument kind, return kind)
CATALOGUE = [
    ('data_fusion_concat2', data_fusion_concat2, 2, TEXT, TEXT),
    ('data_fusion_concat3', data_fusion_concat3, 3, TEXT, TEXT),
    ('data_fusion_interleave2', data_fusion_interleave2, 2, TEXT, TEXT),
    ('data_fusion_rowstack2', data_fusion_rowstack2, 2, MATRIX, MATRIX),
    ('data_fusion_rowstack3', data_fusion_rowstack3, 3, MATRIX, MATRIX),

    ('preprocessor_fusion_merge2', preprocessor_fusion_merge2, 2, TEXT, TEXT),
    ('preprocessor_fusion_merge3', preprocessor_fusion_merge3, 3, TEXT, TEXT),
    ('preprocessor_fusion_dedupUnion2', preprocessor_fusion_dedupUnion2, 2, TEXT, TEXT),

    ('vectorSpace_fusion_concat2', vectorSpace_fusion_concat2, 2, MATRIX, MATRIX),
    ('vectorSpace_fusion_concat3', vectorSpace_fusion_concat3, 3, MATRIX, MATRIX),
    ('vectorSpace_fusion_mean2', vectorSpace_fusion_mean2, 2, MATRIX, MATRIX),
    ('vectorSpace_fusion_mean3', vectorSpace_fusion_mean3, 3, MATRIX, MATRIX),
    ('vectorSpace_fusion_blend25', vectorSpace_fusion_blend25, 2, MATRIX, MATRIX),
    ('vectorSpace_fusion_blend75', vectorSpace_fusion_blend75, 2, MATRIX, MATRIX),
    ('vectorSpace_fusion_max2', vectorSpace_fusion_max2, 2, MATRIX, MATRIX),
    ('vectorSpace_fusion_median3', vectorSpace_fusion_median3, 3, MATRIX, MATRIX),
    ('vectorSpace_fusion_residual2', vectorSpace_fusion_residual2, 2, MATRIX, MATRIX),
    ('vectorSpace_fusion_normConcat2', vectorSpace_fusion_normConcat2, 2, MATRIX, MATRIX),
    ('vectorSpace_fusion_reduceConcat2', vectorSpace_fusion_reduceConcat2, 2, MATRIX, MATRIX),
    ('vectorSpace_fusion_reduceConcat3', vectorSpace_fusion_reduceConcat3, 3, MATRIX, MATRIX),
    ('vectorSpace_fusion_cca2', vectorSpace_fusion_cca2, 2, MATRIX, MATRIX),
    ('vectorSpace_fusion_procrustesMean2', vectorSpace_fusion_procrustesMean2, 2, MATRIX, MATRIX),

    ('clusterer_fusion_consensus2', clusterer_fusion_consensus2, 2, LABELS, LABELS),
    ('clusterer_fusion_consensus3', clusterer_fusion_consensus3, 3, LABELS, LABELS),
    ('clusterer_fusion_agreementMask2', clusterer_fusion_agreementMask2, 2, LABELS, LABELS),
    ('clusterer_fusion_vote3', clusterer_fusion_vote3, 3, LABELS, LABELS),
    ('clusterer_fusion_coassoc2', clusterer_fusion_coassoc2, 2, LABELS, LABELS),
    ('clusterer_fusion_coassoc3', clusterer_fusion_coassoc3, 3, LABELS, LABELS),

    ('labeler_fusion_vote2', labeler_fusion_vote2, 2, LABELS, LABELS),
    ('labeler_fusion_vote3', labeler_fusion_vote3, 3, LABELS, LABELS),

    ('classifier_fusion_vote3', classifier_fusion_vote3, 3, PREDS, PREDS),
    ('classifier_fusion_majority3', classifier_fusion_majority3, 3, PREDS, PREDS),
    ('classifier_fusion_agreeOrAbstain2', classifier_fusion_agreeOrAbstain2, 2, PREDS, PREDS),
    ('classifier_fusion_firstThenSecond2', classifier_fusion_firstThenSecond2, 2, PREDS, PREDS),
    ('classifier_fusion_stack2', classifier_fusion_stack2, 2, PREDS, PREDS),

    ('anomalyDetector_fusion_mean2', anomalyDetector_fusion_mean2, 2, SCORES, SCORES),
    ('anomalyDetector_fusion_mean3', anomalyDetector_fusion_mean3, 3, SCORES, SCORES),
    ('anomalyDetector_fusion_max2', anomalyDetector_fusion_max2, 2, SCORES, SCORES),
    ('anomalyDetector_fusion_max3', anomalyDetector_fusion_max3, 3, SCORES, SCORES),
    ('anomalyDetector_fusion_median3', anomalyDetector_fusion_median3, 3, SCORES, SCORES),
    ('anomalyDetector_fusion_rankAvg2', anomalyDetector_fusion_rankAvg2, 2, SCORES, SCORES),

    ('test_fusion_min2', test_fusion_min2, 2, FLOAT, FLOAT),
    ('test_fusion_mean2', test_fusion_mean2, 2, FLOAT, FLOAT),
    ('test_fusion_min3', test_fusion_min3, 3, FLOAT, FLOAT),
    ('test_fusion_mean3', test_fusion_mean3, 3, FLOAT, FLOAT),
    ('test_fusion_harmonic2', test_fusion_harmonic2, 2, FLOAT, FLOAT),
]


def entries(curr):
    """registry entries, curried the way every other product is"""
    return {name: curr(fn) for name, fn, _a, _i, _o in CATALOGUE}


def ontology_nodes(weight=0.02, branch_weight=0.15):
    """Catalogue declarations, NESTED as the catalogue is walked.

    A product name is read one level at a time - vectorSpace, then
    fusion, then blend75 - so the combiners hang under a 'fusion' node
    inside each family rather than as flat keys with underscores in
    them. That also groups them: reaching any combiner is one decision
    ("fuse") followed by which fusion.
    """
    out = {}
    for name, _fn, arity, kin, kout in CATALOGUE:
        parts = name.split('_')
        fam, mid, leaf = parts[0], parts[1], '_'.join(parts[2:])
        node = {"_weight": weight, "_deterministic": True,
                "_args": [dict(kin) for _ in range(arity)],
                "_return": [dict(kout)],
                "_comment": name}
        branch = out.setdefault(fam, {}).setdefault(
            mid, {"_weight": branch_weight, "_deterministic": True})
        branch[leaf] = node
    return out
