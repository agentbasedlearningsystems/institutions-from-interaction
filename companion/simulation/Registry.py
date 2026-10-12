# Have a registered function for all values in the ontology

# Curry routine from internet site
# Massimiliano Tomassoli, 2012.

# Curry


import os as _os_quiet
if _os_quiet.environ.get('MARKETSIM_QUIET'):
    # Console quiet switch (2 Oct 2026): the per-call name prints below are
    # April-4 tracing; under MARKETSIM_QUIET they are silenced at module level so
    # a 2000-board world does not write gigabytes of console on the Getac.
    def print(*_a, **_k):  # noqa: A001 - deliberate shadow, logging only
        pass

def genCur(func, unique=True, minArgs=None):
    """ Generates a 'curried' version of a function. """

    def g(*myArgs, **myKwArgs):
        def f(*args, **kwArgs):
            if args or kwArgs:  # some more args!
                # Allocates data to assign to the next 'f'.
                newArgs = myArgs + args
                newKwArgs = dict.copy(myKwArgs)

                # If unique is True, we don't want repeated keyword arguments.
                if unique and not kwArgs.keys().isdisjoint(newKwArgs):
                    raise ValueError("Repeated kw arg while unique = True")

                # Adds/updates keyword arguments.
                newKwArgs.update(kwArgs)

                # Checks whether it's time to evaluate func.
                if minArgs is not None and minArgs <= len(newArgs) + len(newKwArgs):
                    return func(*newArgs, **newKwArgs)  # time to evaluate func
                else:
                    return g(*newArgs, **newKwArgs)  # returns a new 'f'
            else:  # the evaluation was forced
                return func(*myArgs, **myKwArgs)

        return f

    return g


def cur(f, minArgs=None):
    return genCur(f, True, minArgs)


def curr(f, minArgs=None):
    return genCur(f, False, minArgs)


# Simple Function.
def func(a, b, c, d, e, f, g=100):
    print(a, b, c, d, e, f, g)

# Tests
def test_clusterer_silhouette(XY):
    # "_args": [{  "type": "numpy.ndarray",  "dtype": "float32"},
    #           {"type": "numpy.ndarray", "dtype": "int32"}],
    # "_return": [{"type": "float"}]

    # we only want to test cosine metric for this example, but it could be a parameter in other cases

    from sklearn import metrics

    min_score = -1.0
    max_score = 1.0
    # normalize from zero to one, from the min and max that this test gives.

    X = XY[0]
    Y = XY[1]
    print('test_clusterer_silhouette')
    silhouette = metrics.silhouette_score(X, Y, metric='cosine')
    silhouette = (silhouette - min_score) / (max_score - min_score)
    return silhouette


# --- 2026-05-21 cleanup: retired several misnamed / weak Claude-added
# product types. Per audit (see docs/CLAUDE_CODE_AUDIT.md if present, or
# the git log around this commit):
#   - topicSummarizer_basic: was just per-cluster centroids; "topic
#     summarizer" implies human-interpretable summaries. Retired.
#   - topicSummarizer_lda: ran LDA on DENSE doc2vec embeddings, which
#     is mathematically wrong (LDA expects sparse count matrices). Also
#     retired below; can be rewritten later to consume preprocessor
#     output via CountVectorizer.
#   - recommender_basic: was within-cluster nearest-neighbor lookup,
#     not a recommender. Retired.
#   - classifier_basic: was kNN trained AND tested on the same data —
#     circular by construction. Retired. classifier_logistic (real
#     train/test split) is kept; its test is rewritten below to
#     compute proper held-out accuracy.
#   - sentimentClassifier_lexicon: 16 hardcoded English words; test
#     rewarded score-spread not accuracy → gameable. Retired.
#   - searchEngine_topK: was just top-k cosine NN. Renamed to
#     nearestNeighbors_topK below for honesty.
# anomalyDetector_basic and anomalyDetector_oneClassSVM kept as-is —
# both legitimately do what they claim. test_anomalyDetector_separation
# is a reasonable measure of detector separability.


def _text_bridge_features():
    # The text bridge width. Her ruling of 2026-08-11 (second): the
    # default RETURNS to 300 - "maybe we can put it back and make 16 a
    # baseline because we dont have time." What this buys: wave 16's
    # banked iterations stay physics-identical to the wave-17 arms,
    # the text evaluation cache stays WARM for the whole arm program
    # (at 2000 every text eval goes cold and several times slower),
    # and the anchor standards measured at 300 stay valid. The knob
    # survives (env-overridable) and the 2000-width question stays
    # open as measured future work, not a mid-deadline gamble.
    import os as _os
    try:
        return int(_os.environ.get('MARKETSIM_TEXT_BRIDGE_FEATURES', '300'))
    except ValueError:
        return 300


def test_classifier_loo(XYP):
    # Held-out accuracy. classifier_logistic returns P[i] = -1 for
    # training rows and the real prediction for held-out rows. We
    # filter to held-out only — measuring accuracy on training rows
    # would just measure train-set fit, and the OLD test
    # (mean(P == Y) across ALL rows) was a lower bound capped at the
    # held-out fraction (~0.2) even when held-out accuracy was 100%.
    # Fixed 2026-05-21 per audit.
    import numpy as np
    print('test_classifier_loo')
    X, Y, P = XYP[0], XYP[1], XYP[2]
    Y = np.asarray(Y)
    P = np.asarray(P)
    held_out = P != -1
    if held_out.sum() == 0:
        return 0.0
    return float(np.clip(np.mean(P[held_out] == Y[held_out]), 0.0, 1.0))


def anomalyDetector_basic(X, n_estimators=100, contamination='auto'):
    # Takes vectorSpace output (no clusterer required); returns
    # (X, scores) where scores are per-row outlier scores from
    # IsolationForest. Higher score = more anomalous.
    # _args:   [{"type": "numpy.ndarray", "dtype": "float32"}]
    # _return: [{"type": "tuple", "dtype": "(float32,float32)"}]
    import numpy as np
    from sklearn.ensemble import IsolationForest
    print('anomalyDetector_basic')
    if len(X) < 8:
        return (X, np.zeros(len(X), dtype=np.float32))
    iso = IsolationForest(random_state=42, n_estimators=n_estimators,
                          contamination=contamination)
    iso.fit(X)
    scores = -iso.score_samples(X).astype(np.float32)
    return (X, scores)




# Settings variants for the two hand-written detectors (the curator
# 2026-08-10: "of course do this, for all of them! and make them
# reasonable!"). Distinct top-level names so cache keys stay distinct.
# nu is the expected outlier fraction (default 0.1); IsolationForest
# variants raise the tree count and pin a fixed contamination rate.
# Parameter-level walks for the two hand-written detectors (the curator
# 2026-08-10 tree restoration): the walk token is a curried ARGUMENT,
# so cache keys stay distinct per settings path, like estimator_lazy.
def anomalyDetector_oneClassSVMwalk(token, X):
    print('anomalyDetector_oneClassSVMwalk ' + str(token))
    nu = {'nu005': 0.05, 'nu01': 0.1, 'nu02': 0.2}.get(str(token), 0.1)
    return anomalyDetector_oneClassSVM(X, nu=nu)


def anomalyDetector_basicwalk(token, X):
    print('anomalyDetector_basicwalk ' + str(token))
    toks = str(token).split('_')
    contamination = 'auto'
    n_estimators = 100
    for t in toks:
        if t == 'c01':
            contamination = 0.1
        elif t == 'cauto':
            contamination = 'auto'
        elif t == 't300':
            n_estimators = 300
        elif t == 't100':
            n_estimators = 100
    return anomalyDetector_basic(X, n_estimators=n_estimators,
                                 contamination=contamination)


def anomalyDetector_oneClassSVMnu005(X):
    print('anomalyDetector_oneClassSVMnu005')
    return anomalyDetector_oneClassSVM(X, nu=0.05)


def anomalyDetector_oneClassSVMnu02(X):
    print('anomalyDetector_oneClassSVMnu02')
    return anomalyDetector_oneClassSVM(X, nu=0.2)


def anomalyDetector_basicT300(X):
    print('anomalyDetector_basicT300')
    return anomalyDetector_basic(X, n_estimators=300)


def anomalyDetector_basicC01(X):
    print('anomalyDetector_basicC01')
    return anomalyDetector_basic(X, contamination=0.1)


def test_clusterer_calinskiHarabaz(XY):
    X = XY[0]
    Y = XY[1]

    # "_args": [{  "type": "numpy.ndarray",  "dtype": "float32"},
    #           {"type": "numpy.ndarray", "dtype": "int32"}],
    # "_return": [{"type": "float"}]

    # we only want to test cosine metric for this example, but it could be a parameter in other cases

    from sklearn import metrics

    print('test_clusterer_calinskiHarabaz')
    # THE UNBOUNDED-SCORE BUG (fixed 2026-08-20, her ruling "normalize").
    # Calinski-Harabasz is a ratio of between-cluster to within-cluster
    # dispersion and has NO upper bound. Dividing by a fixed 200 and
    # returning the quotient let a well-separated clustering return 186.02,
    # against evolved bars that live in [0, 1] - so this test cleared every
    # bar for free while a correctly normalised test like silhouette could
    # not. Squash instead of scale: x/(1+x) on the 200-scaled index is
    # monotone, agrees closely with the old linear map at low indices
    # of 4 (0.5% at index 1, 2% at index 4), and can never leave [0, 1).
    # Degenerate clusterings that make the index undefined score 0.
    import numpy as np
    try:
        calinski_harabaz = metrics.calinski_harabasz_score(X, Y)
    except (ValueError, ZeroDivisionError):
        return 0.0
    if calinski_harabaz != calinski_harabaz or not np.isfinite(calinski_harabaz):
        return 0.0
    scaled = float(calinski_harabaz) / 200.0
    if scaled <= 0.0:
        return 0.0
    return scaled / (1.0 + scaled)


# --- External-ground-truth cluster tests (2026-05-21) -------------------------
# Both load a label sidecar (.npy) co-located with their corpus CSV and compute
# adjusted mutual information between the cluster assignment Y and the true
# label L. Adjusted MI is in [0, 1] (chance-corrected), where 0 = independent
# and 1 = perfect alignment. Returns 0.0 if the labels file is missing or row
# counts mismatch in a non-truncation way.
#
# Critical requirement: row order must be preserved from data loader through
# vectorSpace to the cluster output. The dedicated cc1_ira_detection.json and
# cc1_bills_ideology.json scenarios omit the shuffle preprocessor from their
# ontology so agents can't decode chains that scramble row order.
#
# Truncation rule: when MARKETSIM_VECTORSPACE_MAX_ROWS caps the loaded corpus
# (e.g. bills: 70K rows truncated to 10K), labels[:len(Y)] still aligns because
# the data loader returns the first nrows in deterministic order.

def _ami_with_degeneracy_guard(L, Y):
    """Compute adjusted mutual info between ground-truth labels L and cluster
    assignment Y, but return 0.0 instead of 1.0 for trivially-perfect cases
    that don't reflect real signal:

      - Empty arrays: sklearn returns 1.0 (vacuously perfect) — useless to us
      - 1-element arrays: sklearn returns 1.0 — useless
      - Y is constant (one cluster): sklearn returns 1.0 if L is also
        constant, 0.0 if not — but a constant Y on a non-constant L is
        meaningless clustering, not perfect alignment
      - L is constant: there's no ground-truth structure to detect

    These edge cases arise when a chain decodes without a vectorSpace
    or with a degenerate clusterer, and the RoleSim agents would
    otherwise be falsely rewarded with fitness 1.0 for broken chains.
    """
    import numpy as np
    from sklearn import metrics
    L = np.asarray(L).flatten()
    Y = np.asarray(Y).flatten()
    n = min(len(L), len(Y))
    if n < 2:
        return 0.0
    L, Y = L[:n], Y[:n]
    # Both must have ≥ 2 distinct values for MI to be meaningful
    if len(np.unique(L)) < 2 or len(np.unique(Y)) < 2:
        return 0.0
    return float(metrics.adjusted_mutual_info_score(L, Y))


def test_clusterer_ideologyAMI(XY):
    """Cluster-vs-sponsor-ideology adjusted mutual information on the
    congressional bills corpus. Measures the REAL need "show me the
    ideological division structure in lawmaking": how much the discovered
    bill clusters align with the 3-bin DW-NOMINATE ideology of their
    sponsors (labels sidecar data/billsByCongress_labels.npy).

    Alignment contract: the data loader returns rows in file order and the
    vectorSpace cap is HEAD-truncation, so labels[:len(Y)] matches the
    clustered rows. Chains that scramble row order (shuffle preprocessor)
    void the measurement; the division-need scenarios omit shuffle from
    their ontology. Degenerate cases score 0 via _ami_with_degeneracy_guard.
    """
    import os
    import numpy as np
    print('test_clusterer_ideologyAMI')
    X, Y = XY[0], XY[1]
    # GENERAL FIRST (her ruling 2026-08-20). Welded to one corpus, this
    # scored on 1 draw in 26 and its best result anywhere was 0.0048. The
    # buyer's own corpus is published at audition time, so take the truth
    # from there when the world allows it, and fall back to the file only
    # outside an audition.
    L = _buyer_truth_labels()
    if L is None:
        labels_path = os.path.join(os.getcwd(), 'data',
                                   'billsByCongress_labels.npy')
        if not os.path.exists(labels_path):
            return 0.0
        L = np.load(labels_path)
    return _ami_with_degeneracy_guard(L[:len(np.asarray(Y).flatten())], Y)


def test_clusterer_iraMutualInfo(XY):
    """Cluster-vs-IRA-label MI. Labels from data/iraVsControl_labels.npy."""
    import os
    import numpy as np
    print('test_clusterer_iraMutualInfo')
    X, Y = XY[0], XY[1]
    # GENERAL FIRST (her ruling 2026-08-20) - see test_clusterer_ideologyAMI.
    L = _buyer_truth_labels()
    if L is None:
        labels_path = os.path.join(os.getcwd(), 'data',
                                   'iraVsControl_labels.npy')
        if not os.path.exists(labels_path):
            print(f'  ERROR: labels file missing: {labels_path}')
            return 0.0
        L = np.load(labels_path)
    return _ami_with_degeneracy_guard(L[:len(np.asarray(Y).flatten())], Y)




# The prefab stepping-stones that stood here are DELETED, her ruling
# 2026-08-14: "no prefabs please". They were two pre-assembled parts -
# a fixed 20-dimension term-frequency embedding, and that embedding
# followed by 20-cluster k-means - handed to newborn sellers so they
# had something saleable on day one. Their automatic use inside the
# labeler was withdrawn on 19 July; the registry entries went on
# 14 August; the bodies go here. A seller now assembles its product
# from the ontology or has nothing to sell.


def classifier_billsIdeology_logistic(X):
    #   "_args": [{"type": "numpy.ndarray","dtype": "float32" }],
    #   "_return": [{"type": "tuple","dtype": "(float32,int32,int32)" }]
    # The sloped supervised need (2026-07-09): predict a bill sponsor's
    # 3-bin DW-NOMINATE ideology from the bill-text embedding. Trains on a
    # deterministic 70% split of the ground-truth sidecar, marks training
    # rows -1 in P so test_classifier_heldout scores only held-out rows,
    # base-subtracted. Offline slope proof: svd20 +0.046, svd100 +0.086,
    # svd300 +0.098 above base — quality of the upstream embedding pays.
    import os
    import numpy
    from sklearn.linear_model import LogisticRegression
    print('classifier_billsIdeology_logistic')
    labels_path = os.path.join(os.getcwd(), 'data', 'billsByCongress_labels.npy')
    if not os.path.exists(labels_path) or X is None:
        return None
    L = numpy.load(labels_path)
    n = min(len(X), len(L))
    X, L = X[:n], numpy.asarray(L[:n], dtype='int32')
    rng = numpy.random.RandomState(0)
    idx = rng.permutation(n)
    cut = int(0.7 * n)
    tr, te = idx[:cut], idx[cut:]
    P = numpy.full(n, -1, dtype='int32')
    clf = LogisticRegression(max_iter=2000).fit(X[tr], L[tr])
    P[te] = clf.predict(X[te]).astype('int32')
    return (X, L, P)


# NLP routines
def vectorSpace_gensim_doc2vec(X, size, iterations, minfreq):
    #   "_args": [{"type": "list","firstElement":"gensim.models.doc2vec.TaggedDocument" }],
    #   "_return": [{"type": "numpy.ndarray","dtype": "float32" }

    import gensim
    import numpy as np
    import sklearn.preprocessing
    from sklearn.preprocessing import StandardScaler

    print('vectorSpace_gensim_doc2vec')

    # workers: number of CPU threads gensim's cython doc2vec uses. The
    # cython implementation releases the GIL during training so workers
    # do give real parallel speedup. Default gensim value is 3; we set 4
    # explicitly. Override via MARKETSIM_GENSIM_WORKERS env var (host has
    # 14 cores so 4-8 is reasonable; multiple python processes share the
    # CPU so going higher risks oversubscription).
    import os as _os
    _workers = int(_os.environ.get("MARKETSIM_GENSIM_WORKERS", "4"))
    model = gensim.models.doc2vec.Doc2Vec(vector_size=size, min_count=minfreq,
                                          epochs=iterations, dm=0,
                                          workers=_workers)
    model.build_vocab(X)
    model.train(X, total_examples=model.corpus_count, epochs=model.epochs)
    cmtVectors = [model.infer_vector(X[i].words) for i in range(len(X))]
    cmtVectors = [inferred_vector for inferred_vector in cmtVectors
                  if not np.isnan(inferred_vector).any()
                  and not np.isinf(inferred_vector).any()]

    # OUTPUT ROW CAP — May 14 2026.
    # AgglomerativeClustering / SpectralClustering / MeanShift / DBSCAN /
    # ward etc. build an O(n^2) pairwise distance/affinity matrix from
    # the vectorSpace output. With n=203k (pol_tweets_short or tweets.csv
    # row count) that's 203430^2 * 8 bytes = 308 GiB - process OOMs.
    #
    # Cap the number of vectors returned from vectorSpace to keep the
    # downstream clusterer's pairwise matrix tractable. Doc2vec training
    # itself still uses the FULL corpus (so embedding quality is
    # preserved); only the OUTPUT to the clusterer is capped.
    #
    # MARKETSIM_VECTORSPACE_MAX_ROWS overrides the default. 5000 keeps the
    # pairwise matrix at 5000^2 * 8 = 200 MB - fits comfortably in RAM.
    # HEAD-truncation (first N rows, order preserved), NOT a random
    # subsample: ground-truth label sidecars align via labels[:len(Y)]
    # only if row order and prefix survive this cap. The prior seeded
    # random subsample silently broke every label-aligned test on
    # capped corpora (the bills ideology null result).
    import os as _os
    _max_rows = int(_os.environ.get("MARKETSIM_VECTORSPACE_MAX_ROWS", "5000"))
    if len(cmtVectors) > _max_rows:
        cmtVectors = cmtVectors[:_max_rows]

    return StandardScaler().fit_transform(cmtVectors)


def vectorSpace_sklearn_lsa_100(X):
    """Latent semantic analysis embedding: tfidf followed by truncated
    SVD to 100 dimensions. A third embedding family (linear-algebraic,
    global co-occurrence) alongside doc2vec (neural, local context) and
    raw tfidf (sparse lexical) — genuinely different structure, which is
    what makes it useful fusion material for vectorSpace_concat. Takes
    tagged documents like the other vector spaces; returns a dense
    float array capped by the same head-truncation rule.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import StandardScaler
    import os as _os
    print('vectorSpace_sklearn_lsa_100')
    docs = [' '.join(d.words) if hasattr(d, 'words') else str(d) for d in X]
    tf = TfidfVectorizer(max_features=20000).fit_transform(docs)
    k = min(100, tf.shape[1] - 1, tf.shape[0] - 1)
    if k < 2:
        return None
    emb = TruncatedSVD(n_components=k, random_state=0).fit_transform(tf)
    _max_rows = int(_os.environ.get("MARKETSIM_VECTORSPACE_MAX_ROWS", "5000"))
    if len(emb) > _max_rows:
        emb = emb[:_max_rows]
    return StandardScaler().fit_transform(emb)


def preprocessor_freetext_tag(X):
    # convert a list of strings to a tagged document
    # if it is a list of a list of strings broadcast to a list of tagged documents

    #   "_args": [{"type": "list","firstElement":"string" }],
    #   "_return": [{"type": "list","gensim.models.doc2vec.TaggedDocument" }]

    import gensim
    print('preprocessor_freetext_tag')

    # gensim 4 requires TaggedDocument.words to be a list of tokens, not a raw string
    def _to_tokens(x):
        return x.split() if isinstance(x, str) else list(x)
    tag = lambda x, y: gensim.models.doc2vec.TaggedDocument(_to_tokens(x), [y])

    if type(X) is str:
        tagged = tag(X, X)
    else:
        tagged = [tag(x, y) for y, x in enumerate(X)]
    return tagged


def preprocessor_freetext_lemmatization(X):
    #   "_args": [{"type": "list","firstElement":"string" }],
    #   "_return": [{"type": "list","firstElement":"list" }]

    # converts string documents into list of tokens
    # if given a list, broadcasts
    #
    # REPAIRED 2026-08-12 (her order: fix this product or drop it; it is a
    # real preprocessor and the corpora are text, so it is fixed). This used
    # gensim.utils.lemmatize, which gensim REMOVED in its 4.x line - it
    # depended on the `pattern` package, which was dropped. The installed
    # gensim is 4.4.0, so every call raised AttributeError and the product
    # had a failure rate of one: 218 raises in a 400,000-line sample of one
    # night's log.
    #
    # Same job, current libraries: tokenize, drop stopwords, lemmatize with
    # WordNet, which nltk 3.9.4 has locally. The one visible difference is
    # that old gensim returned byte strings tagged with a part of speech,
    # like b'dog/NN'; this returns the bare lemma, which is what every
    # downstream vectorizer here wants anyway.
    import os
    from gensim.utils import simple_preprocess
    from nltk.stem import WordNetLemmatizer

    print('preprocessor_freetext_lemmatization')
    stopwords = set()
    for _cand in ('stopwords.txt',
                  os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stopwords.txt')):
        try:
            with open(_cand, 'r') as f:
                stopwords = {word.lower().strip() for word in f.readlines()}
            break
        except OSError:
            continue

    _wnl = WordNetLemmatizer()

    def lemma(x):
        return [_wnl.lemmatize(tok) for tok in simple_preprocess(str(x))
                if tok not in stopwords]

    if type(X) is str:
        return lemma(X)
    return [lemma(x) for x in X]


def preprocessor_freetext_strip(X):
    # strips addresses and emojis. if you get string strip, if you get list broadcast

    #   "_args": [{"type": "list","firstElement":"string" }],
    #   "_return": [{"type": "list","firstElement":"string" }]

    import re

    print("preprocessor_freetext_strip")
    code = 'utf-8'
    strip = lambda x: re.sub(r"\s?http\S*", "", x).encode(code).decode(code)

    # strip = lambda  x: re.sub(r"\s?http\S*", "", x).decode(code)
    # strip = lambda  x: re.sub(r"\s?http\S*", "", x.decode(code))
    # strip = lambda  x: re.sub(r"\s?http\S*", "", x)

    if type(X) is str:
        decoded = strip(X)
    else:
        decoded = [strip(x) for x in X]
    return decoded


def preprocessor_freetext_shuffle(X):
    #   "_args": [{"type": "list" }],
    #   "_return": [{"type": "list" }]
    import random

    print("preprocessor_freetext_shuffle")
    # Her ruling: deterministic from the scenario seed via the
    # universal evaluation sandbox (not a pinned constant).
    random.shuffle(X)
    return X

# Data
def data_freetext_csvColumn(relative_path, col='text', nrows=None):
    #  returns a list of documents that are strings
    #   "_return": [{"type": "list","firstElement":"string" }]
    # If nrows is set, only the first nrows lines are loaded.
    # Useful for tractable fitness evaluation during research; the
    # CORE_DATA_NROWS env var overrides the parameter for global control.

    import pandas as pd
    import os

    cwd = os.getcwd()
    path = cwd + relative_path

    env_nrows = os.environ.get('CORE_DATA_NROWS')
    if env_nrows is not None:
        try:
            nrows = int(env_nrows)
        except (ValueError, TypeError):
            pass

    print('data_freetext_csvColumn at ' + path
          + (f' (nrows={nrows})' if nrows else ''))
    # Auto-detect delimiter by peeking the first line: count `,` vs `;`,
    # pick the more frequent. The 4 IRA/Trump/BS/Ads corpora use commas;
    # pol_tweets.csv and pol_tweets_short.csv use semicolons. Without
    # this, pd.read_csv defaults to sep=',' and mis-tokenizes the
    # semicolon-delimited files, raising ParserError on row 12 of
    # pol_tweets_short.csv (a tweet with a natural comma). pickleThis
    # then silently caches the exception as None for the chain key, and
    # every downstream chain involving that corpus returns score=None
    # forever. This was the a5 stuck-buyer root cause confirmed
    # 2026-05-21: 559,160 chain invocations involving
    # politicalTweetsShort, all returning None, over the full 12000-iter
    # cc1 run. See docs/SIMULATION_MANUAL.md §7 and
    # paper_claim_audit.md §3.9.stuck-buyer.
    with open(path, encoding="ISO-8859-1") as _peek:
        _first_line = _peek.readline()
    _sep = ';' if _first_line.count(';') > _first_line.count(',') else ','
    raw_data = pd.read_csv(path, encoding="ISO-8859-1", nrows=nrows, sep=_sep)
    # Column-name auto-detection: 4 of our 5 freetext corpora use the
    # column name 'text' (which is the function's default). pol_tweets.csv
    # and pol_tweets_short.csv use 'tweet_text'. If the caller's `col`
    # isn't present, fall back to the first column with 'text' in its
    # name (case-insensitive). Preserves the explicit-override behavior
    # while making the default work for both schemas. Without this, the
    # 5/21 sep-fix landed only the first half of the a5 root cause —
    # ParserError went away but KeyError('text') took its place,
    # still silently swallowed by pickleThis to None.
    if col not in raw_data.columns:
        candidates = [c for c in raw_data.columns if 'text' in c.lower()]
        if not candidates:
            raise KeyError(
                f"data_freetext_csvColumn: no column named {col!r} in {path}, "
                f"and no column containing 'text'. Columns: {list(raw_data.columns)}"
            )
        col = candidates[0]
    # NaN is truthy in Python (it's a float), so the bare truthy check let
    # NaN tweets through and broke preprocessor_freetext_strip's re.sub.
    docList = [raw_data.loc[i, col] for i in range(len(raw_data))
               if isinstance(raw_data.loc[i, col], str)
               and raw_data.loc[i, col]]
    return docList


def data_vector_varied(n_samples=1500):
    import sklearn

    print('data_vector_varied')
    from sklearn.datasets import make_blobs
    X, Y = make_blobs(n_samples=n_samples,
                      cluster_std=[1.0, 2.5, 0.5],
                      random_state=170)
    return X


def data_vector_ansio(n_samples=1500):
    import numpy as np
    from sklearn.datasets import make_blobs
    print('data_vector_ansio')
    X, y = make_blobs(n_samples=n_samples, random_state=170)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)

    return X_aniso


def data_vector_circles(n_samples=1500):
    from sklearn.datasets import make_circles
    print('data_vector_circles')
    # Her ruling: not pinned - deterministic from the scenario seed
    # via the universal evaluation sandbox.
    X, Y = make_circles(n_samples=n_samples, factor=.5,
                        noise=.05)
    return X


def data_vector_moons(n_samples=1500):
    from sklearn.datasets import make_moons
    print('data_vector_moons')
    # Her ruling: not pinned - deterministic from the scenario seed
    # via the universal evaluation sandbox.
    X, Y = make_moons(n_samples=n_samples, noise=.05)
    return X


def data_vector_blobs(n_samples=1500):
    from sklearn.datasets import make_blobs
    print('data_vector_blobs')
    X, Y = make_blobs(n_samples=n_samples, random_state=8)
    return X

# Clusterers
params = {'quantile': .3,
          'eps': .3,
          'damping': .9,
          'preference': -200,
          'n_neighbors': 10,
          'n_clusters': 20}


def clusterer_sklearn_kmeans(X, n_clusters):
    # "_args": [{"type": "numpy.ndarray","dtype": "float32"} ],
    #   "_return": [{ "type": "numpy.ndarray","dtype": "int32"}

    # in this case we want to try different numbers of clusters, so it is a parameter
    from sklearn.cluster import MiniBatchKMeans

    print('clusterer_sklearn_kmeans')
    clusterAlgSKN = MiniBatchKMeans(n_clusters).fit(X)
    clusterAlgLabelAssignmentsSKN = clusterAlgSKN.predict(X)
    XY = (X, clusterAlgLabelAssignmentsSKN)
    return XY


def clusterer_sklearn_agglomerative(X, n_clusters, n_neighbors=10):
    # Was previously broken: referenced undefined `params['n_neighbors']`
    # (commit ed07486 fixed n_clusters but left n_neighbors). Also used
    # `np.int` which numpy 1.20+ removed → AttributeError. Both fixed
    # 2026-05-21. n_neighbors is now an explicit kwarg with chain-token
    # variants (`_nn5`, `_nn10`, `_nn20`) added to the registry.
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.neighbors import kneighbors_graph
    print('clusterer_sklearn_agglomerative')

    connectivity = kneighbors_graph(
        X, n_neighbors=n_neighbors, include_self=False)
    average_linkage = AgglomerativeClustering(
        linkage="average", metric="cosine",
        n_clusters=n_clusters, connectivity=connectivity).fit(X)
    return (X, average_linkage.labels_.astype(int))


def clusterer_sklearn_affinityPropagation(X, damping=0.9, preference=None):
    # Was broken on `params['damping']` and `params['preference']`. Doesn't
    # take n_clusters at all (AffinityPropagation determines cluster count
    # from data + preference parameter). New chain-token suffixes
    # `_damping05`, `_damping07`, `_damping09` express the real
    # hyperparameter.
    from sklearn.cluster import AffinityPropagation
    print('clusterer_sklearn_affinityPropagation')
    affinity_propagation = AffinityPropagation(
        damping=damping, preference=preference, random_state=0).fit(X)
    return (X, affinity_propagation.predict(X))


def clusterer_sklearn_meanShift(X, quantile=0.3):
    # Was broken on `params['quantile']`. Doesn't take n_clusters
    # (MeanShift determines cluster count from bandwidth, which is
    # estimated from quantile). New chain-token suffixes `_quantile01`,
    # `_quantile02`, `_quantile03` express the real hyperparameter.
    import sklearn
    from sklearn.cluster import MeanShift
    print('clusterer_sklearn_meanShift')
    bandwidth = sklearn.cluster.estimate_bandwidth(X, quantile=quantile)
    if bandwidth <= 0:
        # degenerate sample → bandwidth=0 makes MeanShift hang
        raise ValueError(f"meanShift bandwidth from quantile={quantile} is "
                         f"non-positive ({bandwidth}); refusing to fit")
    import os as _os
    _njobs = int(_os.environ.get("MARKETSIM_SKLEARN_NJOBS", "1"))
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=_njobs).fit(X)
    return (X, ms.predict(X))


def clusterer_sklearn_spectral(X, n_clusters):
    # "_args": [{"type": "numpy.ndarray","dtype": "float32"} ],
    #   "_return": [{ "type": "numpy.ndarray","dtype": "int32"}

    # in this case we want to try different numbers of clusters, so it is a parameter

    from sklearn.cluster import SpectralClustering
    import numpy as np
    print('clusterer_sklearn_spectral')

    import os as _os
    _njobs = int(_os.environ.get("MARKETSIM_SKLEARN_NJOBS", "1"))
    spectral = SpectralClustering(
        n_clusters=n_clusters, eigen_solver='arpack',
        affinity="cosine", n_jobs=_njobs)
    try:
        clusterAlgLabelAssignmentsSS = None
        spectral = spectral.fit(X)
    except ValueError as e:
        pass
    else:
        clusterAlgLabelAssignmentsSS = spectral.labels_.astype(int)

    XY = (X, clusterAlgLabelAssignmentsSS)
    return XY


def clusterer_sklearn_ward(X, n_clusters, n_neighbors=10):
    # Was broken on `params['n_neighbors']`. Also `np.int` deprecation.
    # Both fixed 2026-05-21. n_neighbors is now an explicit kwarg with
    # chain-token variants (`_nn5`, `_nn10`, `_nn20`) added to the
    # registry.
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.neighbors import kneighbors_graph
    print('clusterer_sklearn_ward')
    connectivity = kneighbors_graph(
        X, n_neighbors=n_neighbors, include_self=False)
    connectivity = 0.5 * (connectivity + connectivity.T)
    ward = AgglomerativeClustering(
        n_clusters=n_clusters, linkage='ward',
        connectivity=connectivity).fit(X)
    return (X, ward.labels_.astype(int))


def clusterer_sklearn_dbscan(X, eps=0.5, min_samples=5):
    # Was broken on `params['eps']` and `np.int`. Doesn't take n_clusters
    # (DBSCAN determines cluster count from eps + min_samples + data
    # density). New chain-token suffixes `_eps01`, `_eps03`, `_eps05`,
    # `_eps10` express the real hyperparameter. min_samples fixed at 5
    # as a reasonable default; can be expanded to chain-tokens later if
    # needed.
    from sklearn.cluster import DBSCAN
    import os as _os
    print('clusterer_sklearn_dbscan')
    # Fail fast on absurdly wide inputs (e.g. raw tfidf, ~50k columns):
    # radius-neighbor chunks on such matrices grind for hours against the
    # eval memcap, storming ignored-MemoryErrors (the Society-A stall,
    # 2026-07-08). An unbuildable product should fail like one.
    width = getattr(X, 'shape', (0, 0))[1] if hasattr(X, 'shape') else 0
    if width and width > 2000:
        raise ValueError(f'dbscan refused: input width {width} > 2000')
    # Density precheck (2026-07-08): eps=0.5 on a coarse 20-dim embedding
    # gives thousands of neighbors per point — gigabytes of neighbor lists,
    # std::bad_alloc storms, then a fatal thread-state failure (the Society-A
    # crash loop). A radius that captures a near-complete graph is degenerate
    # single-cluster junk; refuse it from a cheap 400-row sample.
    import numpy as _np
    n = getattr(X, 'shape', (0,))[0] if hasattr(X, 'shape') else len(X)
    if n > 1000:
        from sklearn.neighbors import NearestNeighbors
        idx = _np.linspace(0, n - 1, 400).astype(int)
        Xs = X[idx]
        counts = NearestNeighbors(radius=eps).fit(Xs).radius_neighbors(
            Xs, return_distance=False)
        mean_frac = float(_np.mean([len(c) for c in counts])) / len(idx)
        if mean_frac > 0.05:
            raise ValueError(
                f'dbscan refused: eps={eps} yields ~{mean_frac:.0%} '
                f'neighbor density (near-complete graph, degenerate)')
    _njobs = int(_os.environ.get("MARKETSIM_SKLEARN_NJOBS", "1"))
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=_njobs).fit(X)
    return (X, dbscan.labels_.astype(int))


def clusterer_sklearn_birch(X, n_clusters):
    # "_args": [{"type": "numpy.ndarray","dtype": "float32"} ],
    #   "_return": [{ "type": "numpy.ndarray","dtype": "int32"}

    # in this case we want to try different numbers of clusters, so it is a parameter
    from sklearn.cluster import Birch
    print('clusterer_sklearn_birch')

    birch = Birch(n_clusters=n_clusters).fit(X)
    clusterAlgLabelAssignmentsSB = birch.predict(X)

    XY = (X, clusterAlgLabelAssignmentsSB)
    return XY


def clusterer_sklearn_gaussian(X, n_clusters):
    # "_args": [{"type": "numpy.ndarray","dtype": "float32"} ],
    #   "_return": [{ "type": "numpy.ndarray","dtype": "int32"}

    # in this case we want to try different numbers of clusters, so it is a parameter
    from sklearn import mixture
    print('clusterer_sklearn_gaussian')

    clusterAlgSGN = mixture.GaussianMixture(n_components=n_clusters, covariance_type='full').fit(X)
    clusterAlgLabelAssignmentsSGN = clusterAlgSGN.predict(X)

    XY = (X, clusterAlgLabelAssignmentsSGN)
    return XY


def clusterer_nltk_kmeans(X, n_clusters):
    # "_args": [{"type": "numpy.ndarray","dtype": "float32"} ],
    #   "_return": [{ "type": "numpy.ndarray","dtype": "int32"}

    # in this case we want to try different numbers of clusters, so it is a parameter
    import nltk
    import numpy as np
    from nltk.cluster.kmeans import KMeansClusterer
    print('clusterer_nltk_kmeans')

    clusterAlgLabelAssignmentsNK = None
    # X = XY[0]
    cmtVectors = X  # XY[1]
    if type(cmtVectors) is np.ndarray and len(cmtVectors) > 0:
        # dt = np.dtype(cmtVectors)
        dt = cmtVectors.dtype
        if dt.type is np.float32 or dt.type is np.float64:
            clusterAlgNK = KMeansClusterer(n_clusters, distance=nltk.cluster.util.cosine_distance, repeats=25,
                                           avoid_empty_clusters=True)
            clusterAlgLabelAssignmentsNK = clusterAlgNK.cluster(cmtVectors, assign_clusters=True)

    XY = (X, clusterAlgLabelAssignmentsNK)
    return XY


def clusterer_nltk_agglomerative(X, n_clusters):
    # "_args": [{"type": "numpy.ndarray","dtype": "float32"} ],
    #   "_return": [{ "type": "numpy.ndarray","dtype": "int32"}

    # in this case we want to try different numbers of clusters, so it is a parameter
    import numpy as np
    from nltk.cluster.gaac import GAAClusterer
    print('clusterer_nltk_agglomerative')

    clusterAlgLabelAssignmentsNG = None

    # X = XY[0]
    cmtVectors = X  # XY[1]

    if type(cmtVectors) is np.ndarray and len(cmtVectors) > 0:
        # dt = np.dtype(cmtVectors)
        dt = cmtVectors.dtype
        if dt.type is np.float32 or dt.type is np.float64:
            clusterAlgNG = GAAClusterer(num_clusters=n_clusters, normalise=True, svd_dimensions=None)
            clusterAlgLabelAssignmentsNG = clusterAlgNG.cluster(cmtVectors, assign_clusters=True)

    return X, clusterAlgLabelAssignmentsNG


# === Added: intermediate-level tests + substantive end-product implementations ===
# Each rung of the production stack (preprocessor → vectorSpace → end-product)
# now has at least one direct test, so buyers can demand at any level and
# agents that specialize at a single rung have something to sell.









# topicSummarizer_lda was retired 2026-05-21 — was running LDA on DENSE
# doc2vec embeddings, which is mathematically wrong (LDA expects sparse
# count matrices). Returned "topic-word distributions" that were
# actually topic-EMBEDDING-DIMENSION distributions, uninterpretable.
# Can be rewritten later to consume preprocessor (tokenized text) output
# via CountVectorizer + LDA, which would give real topic-over-word
# distributions.




def anomalyDetector_oneClassSVM(X, nu=0.1):
    # Alternative anomaly detector via sklearn OneClassSVM. Provides
    # horizontal variety so the anomaly-detector industry has more than
    # one algorithm to specialize in.
    # _args:   [{"type": "numpy.ndarray", "dtype": "float32"}]
    # _return: [{"type": "tuple", "dtype": "(float32,float32)"}]
    import numpy as np
    print('anomalyDetector_oneClassSVM')
    if X is None or len(X) < 8:
        return (X if X is not None else np.zeros((0, 1), dtype=np.float32),
                np.zeros(len(X) if X is not None else 0, dtype=np.float32))
    X = np.asarray(X, dtype=np.float32)
    # Guard, mirror of the dbscan width guard: kernel cost explodes on wide
    # inputs, and libsvm's default max_iter=-1 can grind unboundedly on
    # pathological data (the Jul 10 Society-B wedge shape).
    width = X.shape[1] if X.ndim == 2 else 1
    if width > 2000:
        raise ValueError(
            f'anomalyDetector_oneClassSVM refused: {width} cols > 2000 '
            f'(degenerate kernel cost; mirror of dbscan width guard)')
    try:
        from sklearn.svm import OneClassSVM
        svm = OneClassSVM(gamma='scale', nu=nu, max_iter=20000).fit(X)
        scores = -svm.decision_function(X).astype(np.float32)
    except Exception:
        scores = np.zeros(len(X), dtype=np.float32)
    return (X, scores)


def vectorSpace_sklearn_tfidf(X):
    # TF-IDF vectorizer. Horizontal alternative to gensim doc2vec — produces
    # a vector space directly from raw text without iterative training,
    # so an agent specializing in tfidf has a different cost/quality profile
    # than one specializing in doc2vec.
    # _args:   [{"type": "list", "firstElement": "str"}]
    # _return: [{"type": "numpy.ndarray", "dtype": "float32"}]
    import numpy as np
    print('vectorSpace_sklearn_tfidf')
    if not X:
        return np.zeros((0, 1), dtype=np.float32)
    docs = [' '.join(d) if isinstance(d, (list, tuple)) else str(d) for d in X]
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(max_features=_text_bridge_features(), min_df=2)
        M = vec.fit_transform(docs)
        return np.asarray(M.toarray(), dtype=np.float32)
    except Exception:
        return np.zeros((len(X), 1), dtype=np.float32)


# --- Parallel industries that don't go through a clusterer ---
# Each consumes upstream output (preprocessor or vectorSpace) directly and
# delivers an end product. Sellers at the preprocessor or vectorSpace level
# now have multiple downstream customer types, not just clusterer-makers —
# the gradient has parallel branches.

# sentimentClassifier_lexicon was retired 2026-05-21 — implementation was
# 16 hardcoded English words (era-specific: 'crooked', 'fake', 'winner');
# the test (test_sentimentClassifier_polarity) rewarded SCORE SPREAD
# instead of accuracy, making it gameable by any classifier that just
# outputs ±1 randomly.


def nearestNeighbors_topK(X):
    # Top-K nearest-neighbor lookup by cosine similarity. Consumes
    # vectorSpace output, no clusterer needed. For each row, returns
    # the top-3 nearest other rows. Returns (X, neighbors) where
    # neighbors[i] is a length-3 int array of indices.
    # _args:   [{"type": "numpy.ndarray", "dtype": "float32"}]
    # _return: [{"type": "tuple", "dtype": "(float32,int32)"}]
    #
    # Renamed from searchEngine_topK on 2026-05-21 — the function is
    # literally kNN lookup, not a search engine (no indexing, query
    # processing, ranking, or relevance feedback). The honest name is
    # nearestNeighbors_topK.
    import numpy as np
    print('nearestNeighbors_topK')
    if X is None:
        return (np.zeros((0, 1), dtype=np.float32), np.zeros((0, 3), dtype=np.int32))
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] < 4:
        return (X, np.zeros((max(0, X.shape[0]), 3), dtype=np.int32))
    n = X.shape[0]
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        S = cosine_similarity(X)
        np.fill_diagonal(S, -np.inf)
        k = 3
        nbrs = np.argpartition(-S, k, axis=1)[:, :k].astype(np.int32)
    except Exception:
        nbrs = np.zeros((n, 3), dtype=np.int32)
    return (X, nbrs)


def test_nearestNeighbors_relevance(XN):
    # Mean cosine similarity between each row and its top-3 neighbors.
    # Higher = the lookup returns genuinely close vectors.
    # Squashed from [-1, 1] to [0, 1].
    # Renamed from test_searchEngine_relevance on 2026-05-21.
    # _return: [{"type": "float"}]
    import numpy as np
    print('test_nearestNeighbors_relevance')
    if XN is None or len(XN) < 2:
        return 0.0
    X, nbrs = XN[0], XN[1]
    if X is None or nbrs is None or len(X) < 4:
        return 0.0
    X = np.asarray(X, dtype=np.float32)
    nbrs = np.asarray(nbrs, dtype=np.int64)
    if X.ndim != 2 or nbrs.ndim != 2:
        return 0.0
    try:
        sims = []
        norms = np.linalg.norm(X, axis=1) + 1e-9
        for i in range(len(X)):
            for j in nbrs[i]:
                if 0 <= j < len(X) and j != i:
                    num = float(X[i] @ X[j])
                    den = norms[i] * norms[j]
                    sims.append(num / den)
        if not sims:
            return 0.0
        mean_sim = float(np.mean(sims))
    except Exception:
        return 0.0
    return float(np.clip((mean_sim + 1.0) / 2.0, 0.0, 1.0))


# === end added block ===


# === Added 2026-07-04: three vetted additions (repaired classifier, honest
# anomaly-detector gate, and the first two-input vectorSpace primitive).
# Each was proven by execution against planted ground truth before landing;
# see /tmp/registry_additions_evidence.md for the harness output, the
# adversarial cases, and the exact ontology snippets each item needs. ===


def classifier_logistic_v2(XY, C=1.0):
    # Repair of the retired classifier_logistic. Consumes a clusterer's
    # (X, Y) output and treats Y as classification targets: fits sklearn
    # LogisticRegression on a 70/30 held-out split and returns
    # (X, Y, pred) where pred[i] = -1 on training rows and the real
    # class prediction on held-out rows.
    #
    # Two deliberate departures from the deleted version:
    #   1. The removed `multi_class='auto'` argument is gone. sklearn 1.8
    #      removed that kwarg, so the old call raised TypeError on this
    #      host — see evidence file. LogisticRegression is multinomial by
    #      default now, so the argument was both broken and redundant.
    #   2. On ANY exception the predictions are all -1, NOT zeros. The old
    #      code fell back to np.zeros(n), which test_classifier_loo then
    #      read as "every held-out row predicted class 0" — a constant
    #      output masquerading as real predictions that could bank the
    #      base rate. All -1 means "no held-out prediction", so the paired
    #      test sees zero held-out rows and scores it 0.
    #
    # 2026-07-16 (shared-scenario port, the curator's exact-split ruling): the
    # 70/30 division now goes through _port_or_seeded_split. When Y is a
    # port corpus's ground truth (labeler_* chains), the division is the
    # LLM world's FIXED train/test file split — without this the logistic
    # family would train on test rows the honest families never see, and
    # the market would converge on leakage. For every other Y (clusterer
    # output, bills ideology, ...) the helper reproduces the historical
    # seeded stratified 70/30 split exactly — behavior unchanged.
    # _args:   [{"type": "tuple", "dtype": "(float32,int32)"}]
    # _return: [{"type": "tuple", "dtype": "(float32,int32,int32)"}]
    import numpy as np
    print('classifier_logistic_v2')
    X, Y = (XY[0], XY[1]) if XY is not None and len(XY) >= 2 else (None, None)
    n = len(X) if X is not None else 0
    if n < 10 or Y is None or len(np.unique(np.asarray(Y))) < 2:
        return (X, Y, np.full(n, -1, dtype=np.int32))
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.int32)
    try:
        from sklearn.linear_model import LogisticRegression
        idx_train, idx_test = _port_or_seeded_split(Y, n)
        pred = np.full(n, -1, dtype=np.int32)
        if idx_train is not None:
            clf = LogisticRegression(max_iter=200, random_state=42, C=float(C))
            clf.fit(X[idx_train], Y[idx_train])
            pred[idx_test] = clf.predict(X[idx_test]).astype(np.int32)
    except Exception:
        # Never a constant class here — an unusable classifier must score 0,
        # not bank the base rate. All -1 => the test sees no held-out rows.
        pred = np.full(n, -1, dtype=np.int32)
    return (X, Y, pred)


def test_classifier_heldout(XYP):
    # Held-out accuracy MINUS the majority-class base rate, clipped to
    # [0, 1]. classifier_logistic_v2 marks training rows -1; we score only
    # the held-out rows (pred != -1). Subtracting the base rate (the
    # accuracy of always guessing the majority true class) means a
    # classifier that only reproduces the class prior scores 0, and a real
    # separating classifier scores its lift over that prior.
    #
    # Degeneracy guard: if the held-out predictions are single-valued the
    # "classifier" is a constant and is rejected with 0 regardless of what
    # accuracy that constant happens to reach — this is the adversary the
    # 2026-07-04 audit flagged.
    # _args:   [{"type": "tuple", "dtype": "(float32,int32,int32)"}]
    # _return: [{"type": "float"}]
    import numpy as np
    print('test_classifier_heldout')
    if XYP is None or len(XYP) < 3:
        return 0.0
    Y, P = np.asarray(XYP[1]), np.asarray(XYP[2])
    held = P != -1
    if held.sum() < 2:
        return 0.0
    yh, ph = Y[held], P[held]
    if len(np.unique(ph)) < 2:
        # Constant predictor — reject, never let it bank the base rate.
        return 0.0
    acc = float(np.mean(ph == yh))
    _, counts = np.unique(yh, return_counts=True)
    base_rate = float(counts.max()) / len(yh)
    return float(np.clip(acc - base_rate, 0.0, 1.0))


def test_anomalyDetector_separation(XS):
    # The separability measure five wants always pinned but nobody had
    # written (found 2026-08-10; the comment above praised it as "a
    # reasonable measure of detector separability" - now it exists,
    # her order: "write the test that doesnt exist"). Fisher-style
    # separability of the detector's flagged tail: split the scores at
    # their 90th percentile; separation = the gap between the two
    # group means over the sum of the in-group spreads, squashed to
    # (0,1) by s/(1+s). Properties, measured at vetting: a cleanly
    # bimodal detector (real outliers far from the body) approaches 1;
    # an uninformative unimodal detector sits near 0.6; constant or
    # degenerate scores give 0. Higher = better separation, like every
    # other test.
    # _args:   [{"type": "tuple", "dtype": "(float32,float32)"}]
    # _return: [{"type": "float"}]
    import numpy as np
    print('test_anomalyDetector_separation')
    if XS is None or not isinstance(XS, (tuple, list)) or len(XS) < 2:
        return None
    scores = np.asarray(XS[1], dtype=np.float64).ravel()
    if len(scores) < 20 or not np.all(np.isfinite(scores)):
        return 0.0
    cut = np.percentile(scores, 90)
    top = scores[scores >= cut]
    rest = scores[scores < cut]
    if len(top) < 2 or len(rest) < 2:
        return 0.0
    gap = float(top.mean() - rest.mean())
    spread = float(top.std() + rest.std())
    if spread <= 0 or gap <= 0:
        return 0.0
    s_ratio = gap / spread
    return float(s_ratio / (1.0 + s_ratio))



# === Added 2026-09-10, her order: "just make a labeler test and a preprocessor
# test". Both are RUNNABLE-SOFTWARE verifications, not comparisons (her
# correction 2026-08-21: "It doesn't have to be comparable. It has to be
# runnable software"). Until now labeler and preprocessor had no test family,
# so every buy of one carried an unresolvable test: refused before the
# 2026-08-21 flag, and paid ZERO after it (the zero-pay block keeps its own
# live-test rule). With a real test the audition runs, the score is 1 or 0,
# and the ordinary pay law applies - a working part earns its price.
def test_labeler_runnable(XY):
    """1.0 if the delivery is an (X, labels) pair with one label per row, at
    least two rows and at least two distinct labels; else 0.0."""
    print('test_labeler_runnable')
    import numpy as _np
    try:
        X, y = XY
        n = len(X)
        yv = _np.asarray(y).ravel()
        if n < 2 or len(yv) != n:
            return 0.0
        return 1.0 if len(set(int(v) for v in yv)) >= 2 else 0.0
    except Exception:
        return 0.0


def test_preprocessor_runnable(X):
    """1.0 if the delivery is a non-empty list of strings; else 0.0."""
    print('test_preprocessor_runnable')
    try:
        if not isinstance(X, (list, tuple)) or len(X) == 0:
            return 0.0
        return 1.0 if all(isinstance(s, str) for s in X) else 0.0
    except Exception:
        return 0.0

def test_labeler_downstreamF1(XY):
    """A labeler is worth what a classifier can do with it (her principle
    2026-09-11: pay for the part that makes the composite work, not for the
    part that merely exists). Train the registry's fixed logistic classifier
    on the delivered (X, labels) pair and score it with the same held-out F1
    the classifier humans use, against the buyer's truth. A bare labeler on
    raw text scores 0.0; a labeler over a real embedding scores ~0.7."""
    print('test_labeler_downstreamF1')
    # Her word, 6 Oct 2026 19:40 ("11 change them as you feel appropriate"): the fixed classifier's regularisation is
    # ML_DOWNSTREAM_C when set (the world's config ml_downstream_c; 10.0 chosen: logistic regression C = 10 scores
    # 0.625 on fake news and 0.722 on movie reviews at 300 TF-IDF features, where C = 1, the old setting, scores
    # 0.272 on fake news - ML_PARTS_BY_CORPUS_20261005.md). Absent, C = 1 as before. The score is the whole test's
    # core, never the whole's bar: the bar is for the whole classifier only.
    try:
        _c_env = _os_quiet.environ.get('ML_DOWNSTREAM_C')
        _C = float(_c_env) if _c_env else 1.0
        return float(_f1heldout_core(classifier_logistic_v2(XY, C=_C)))
    except Exception:
        return 0.0

def test_anomalyDetector_centerRank(XS):
    # Honest gate for anomalyDetector_basic / anomalyDetector_oneClassSVM.
    #
    # The pipeline hands this test the detector's POST-HOC output only:
    # (X, scores). It cannot re-run the detector on planted outliers,
    # because it has no scores for points the detector never saw. So the
    # strongest honest check the typing supports is a NECESSARY condition:
    # a real detector must rank genuinely peripheral points above central
    # ones. We compute each row's robust distance from the data center
    # (per-dimension median, scaled by MAD) and take the Spearman rank
    # correlation between those distances and the detector's scores.
    #
    #   - real detector on blobs + far outliers -> high positive rho
    #   - random scores                         -> rho ~ 0
    #   - the inlier-marking adversary (scores inliers as most anomalous)
    #     from the audit -> negative rho -> clipped to 0
    #   - constant scores / constant distances  -> degenerate -> 0
    #
    # Naming note: this is NOT a "planted-outlier" test — the post-hoc
    # (X, scores) plumbing makes planting impossible — so it is named for
    # what it actually computes (center-distance rank correlation).
    # _args:   [{"type": "tuple", "dtype": "(float32,float32)"}]
    # _return: [{"type": "float"}]
    import numpy as np
    print('test_anomalyDetector_centerRank')
    if XS is None or len(XS) < 2:
        return 0.0
    X, scores = XS[0], XS[1]
    if X is None or scores is None:
        return 0.0
    X = np.asarray(X, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if X.ndim != 2 or X.shape[0] < 10 or X.shape[0] != scores.shape[0]:
        return 0.0
    # Reject a degenerate (constant) score vector outright.
    if np.ptp(scores) <= 1e-12:
        return 0.0
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    mad[mad < 1e-9] = 1e-9
    dist = np.linalg.norm((X - med) / mad, axis=1)
    if np.ptp(dist) <= 1e-12:
        return 0.0
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(scores, dist)
    except Exception:
        # Spearman = Pearson on ranks; compute directly if scipy absent.
        from scipy.stats import rankdata  # noqa: F401
        rho = np.corrcoef(_rankdata(scores), _rankdata(dist))[0, 1]
    if rho is None or np.isnan(rho):
        return 0.0
    return float(np.clip(rho, 0.0, 1.0))


def _rankdata(a):
    # Minimal average-rank helper (scipy-free fallback for centerRank).
    import numpy as np
    a = np.asarray(a, dtype=np.float64)
    order = a.argsort()
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    return ranks


def vectorSpace_concat(A, B):
    # THE two-input vectorSpace primitive. Every other registry item is
    # single-input, so composed programs could previously only be lines;
    # this makes tree-shaped pipelines possible, e.g.
    #   clusterer(vectorSpace_concat(doc2vec(corpus), tfidf(corpus)))
    # so an agent can sell the fused representation of two upstream
    # embeddings.
    #
    # L2-normalizes each matrix row-wise (so neither embedding dominates
    # the concatenation by raw magnitude) then returns the horizontal
    # concatenation. Row-count mismatch is resolved by TRUNCATING both to
    # the shorter length rather than erroring — the two embeddings of the
    # same corpus can differ by a few rows when one drops NaN inferences
    # (see the doc2vec NaN filter), and truncation keeps aligned rows
    # aligned. Returns None (treated upstream as an unevaluable chain,
    # score 0) when either input is missing, non-2-D, or empty.
    # _args:   [{"type": "numpy.ndarray", "dtype": "float32"},
    #           {"type": "numpy.ndarray", "dtype": "float32"}]
    # _return: [{"type": "numpy.ndarray", "dtype": "float32"}]
    import numpy as np
    print('vectorSpace_concat')
    if A is None or B is None:
        return None
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    if A.ndim != 2 or B.ndim != 2 or A.shape[0] == 0 or B.shape[0] == 0:
        return None
    n = min(A.shape[0], B.shape[0])
    A, B = A[:n], B[:n]

    def _l2(M):
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1e-9
        return M / norms

    return np.concatenate([_l2(A), _l2(B)], axis=1).astype(np.float32)


def vectorSpace_sklearn_charTfidf_100(X):
    """Character n-gram (3-5) tfidf followed by TruncatedSVD to 100 dims
    and StandardScaler. A STYLE-sensitive embedding family: character
    n-grams key on orthography, punctuation, affixes and spacing rather
    than word-level topic, so it captures the phrasing/formatting
    signatures that influence-operation content carries — structure the
    topic-sensitive families (doc2vec, word-tfidf, lsa) largely miss.
    Takes tagged documents (or raw strings) like the other vector spaces
    and returns a dense float array capped by the same head-truncation
    rule as vectorSpace_sklearn_lsa_100.

    Measured on data/iraVsControl.csv (KMeans, n_init=10, adjusted Rand vs
    the IRA/organic labels): informative across cluster counts (label-ARI
    0.36 at k=2, 0.52 at k=10) and NOT redundant with lsa_100 — at k>=5
    the two disagree (cluster-agreement ARI 0.37 at k=5, 0.37 at k=10)
    well below their own label-ARIs, and at k=10 charTfidf (0.52) exceeds
    lsa (0.43). At k=2 both recover the same dominant binary split and so
    agree; the distinct structure appears at finer granularity. Naive
    concat-fusion of the two does NOT beat the better single here (see the
    evidence file) — the value is a genuinely different embedding axis for
    the search to exploit, not an automatic accuracy win.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import StandardScaler
    import os as _os
    print('vectorSpace_sklearn_charTfidf_100')
    docs = [' '.join(d.words) if hasattr(d, 'words') else str(d) for d in X]
    tf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5),
                         max_features=20000).fit_transform(docs)
    k = min(100, tf.shape[1] - 1, tf.shape[0] - 1)
    if k < 2:
        return None
    emb = TruncatedSVD(n_components=k, random_state=0).fit_transform(tf)
    _max_rows = int(_os.environ.get("MARKETSIM_VECTORSPACE_MAX_ROWS", "5000"))
    if len(emb) > _max_rows:
        emb = emb[:_max_rows]
    return StandardScaler().fit_transform(emb)


def preprocessor_freetext_stopwords(X):
    """Remove stopwords (repo-root stopwords.txt) from each document,
    preserving row COUNT and ORDER. Consumes and returns the same type as
    preprocessor_freetext_strip — a list of strings (or a single string) —
    so it chains anywhere strip does, e.g. strip -> stopwords -> tag.

    Row alignment is load-bearing: the external-label tests (iraMutualInfo,
    ideologyAMI) require labels[:len(Y)] to line up with the clustered
    rows, so this maps input row i to output row i one-to-one and never
    drops a row. A document that is entirely stopwords becomes the empty
    string in place — the row is kept, not deleted.
    """
    print("preprocessor_freetext_stopwords")
    stopfile = 'stopwords.txt'
    with open(stopfile, 'r') as f:
        stops = {w.lower().strip() for w in f.readlines() if w.strip()}
    drop = lambda x: ' '.join(t for t in str(x).split() if t.lower() not in stops)
    if type(X) is str:
        return drop(X)
    return [drop(x) for x in X]


def clusterer_consensus(A, B):
    # THE second two-input primitive (a clusterer combiner, arity 2). Each
    # input is a clusterer's (X, labels) tuple — the convention every
    # clusterer_* returns. Builds the Fred-Jain evidence-accumulation
    # consensus: the co-association matrix C[i,j] = fraction of the two
    # input clusterings that co-assign i and j (so C in {0, 0.5, 1}), then
    # cuts an average-linkage agglomerative tree on distance 1 - C at
    # k = max(#clusters_A, #clusters_B). Returns (X_from_first_input,
    # consensus_labels) so it plugs into the same tests as any clusterer.
    #
    # Behavior proven by execution (see /tmp/registry_batch2_evidence.md):
    #   - constructed case (two independent 20%-label-noise views of a
    #     4-cluster truth): consensus ARI 0.677 > both singles (0.617,
    #     0.660). It is a combiner, not a guaranteed dominator — with only
    #     two base clusterings it helps when their errors are independent;
    #     it never fabricates signal beyond its inputs (every test scores
    #     it against external ground truth).
    #   - REVISED 2026-07-15 (the curator): consensus(A, A) — and any pair with
    #     adjusted-Rand > 0.95 — is REJECTED (None -> score 0): a
    #     combination of clones is not a combination. Likewise a pair
    #     where either input carries no partition is rejected instead of
    #     passing the survivor through. Depth credit requires genuine
    #     diversity of inputs.
    # Arity-2 execution matches vectorSpace_concat: MarketSim calls
    # registry[name](*resultTuple)(), i.e. curr(f)(A, B)().
    # _args:   [{"type": "numpy.ndarray", "dtype": "int32"},
    #           {"type": "numpy.ndarray", "dtype": "int32"}]
    # _return: [{"type": "numpy.ndarray", "dtype": "int32"}]
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    print('clusterer_consensus')
    if A is None or B is None or len(A) < 2 or len(B) < 2:
        return None
    Xa, La = A[0], A[1]
    Xb, Lb = B[0], B[1]

    def _usable(L):
        return L is not None and len(np.unique(np.asarray(L).ravel())) >= 2

    # If either clustering failed to produce labels, fall back to the one
    # that survived — never invent a consensus from a missing partition.
    if La is None or Lb is None:
        if _usable(La):
            return (Xa, np.asarray(La).ravel())
        if _usable(Lb):
            return (Xb, np.asarray(Lb).ravel())
        return None

    La = np.asarray(La).ravel()
    Lb = np.asarray(Lb).ravel()
    n = min(len(La), len(Lb))
    if n < 2:
        return None
    La, Lb = La[:n], Lb[:n]
    Xa = np.asarray(Xa)[:n] if Xa is not None else None
    Xb = np.asarray(Xb)[:n] if Xb is not None else None
    ua, ub = len(np.unique(La)), len(np.unique(Lb))
    # A constant input is not a clustering; defer to the other one, and
    # only bail out (None) when neither carries a partition.
    if ua < 2 and ub < 2:
        return None
    # 2026-07-15, the curator: "why dont you fix your consensus so that it
    # rejects two of the same cluster?" A consensus must be a genuine
    # combination — the depth premium exists only when the inputs make
    # different errors. Two (near-)identical partitions are refused
    # (None -> score 0, no settle), and so is the old passthrough that
    # returned the surviving input when the other carried no partition:
    # both were depth credit without combination.
    if ua < 2 or ub < 2:
        return None
    from sklearn.metrics import adjusted_rand_score
    if adjusted_rand_score(La, Lb) > 0.95:
        return None

    Ca = (La[:, None] == La[None, :]).astype(np.float64)
    Cb = (Lb[:, None] == Lb[None, :]).astype(np.float64)
    dist = 1.0 - (Ca + Cb) / 2.0
    k = min(max(ua, ub), n)
    cons = AgglomerativeClustering(
        n_clusters=k, metric='precomputed', linkage='average').fit_predict(dist)
    return (Xa if Xa is not None else Xb, cons.astype(int))


# === Added 2026-07-16: shared-scenario port ==============================
# The LLM world's five-want demand catalog (llm_market/core_market.py,
# default_wants) realized as chain-world parts, so a CMA-ES society can run
# the same demand and later serve as the frozen ancestor of hybrid LLM
# societies. Corpora staged by _port_prepare_data.py into
# data/{disasterTweets,movieSentiment,movieSentimentFewshot}.csv with
# ground-truth sidecars <stem>_labels.npy built under data_freetext_
# csvColumn's exact row-survival rule, so labels[:n] stays aligned through
# head-truncation (the billsByCongress_labels.npy contract).
#
# v2 (the curator's 2026-07-16 ruling — differences with the LLM world get
# FIXED, not disclosed): each staged corpus now carries the LLM world's
# EXACT test.csv rows alongside its train rows, with a second sidecar
# <stem>_istest.npy marking them (row order per corpus documented in
# _port_prepare_data.py). The port's classifier engines and both
# want-scorers split on that fixed mask — train on the LLM train rows,
# predict/score on the LLM test rows — so scores are cross-world
# comparable. The seeded 70/30 split remains only as the generic fallback
# for non-port (X, Y) inputs (cluster labels, bills ideology, ...), whose
# behavior is unchanged.
#
# Parts in this block:
#   labeler_disasterTweets / _movieSentiment / _movieSentimentFewshot —
#       join a corpus representation with that corpus's ground truth,
#       emitting the (X, Y) tuple the classifier engines consume
#   classifier_svm_linear, classifier_naivebayes_multinomial — two more
#       learner FAMILIES in the classifier_logistic_v2 harness, so the
#       ensemble want's "distinct families" is a real in-world choice
#   classifier_ensemble_vote — arity-2 combiner (clusterer_consensus
#       pattern); the only route to the item path classifier_ensemble,
#       which is what binds the ensemble want's recipe
#   test_classifier_f1heldout — F1(label=1) variant of
#       test_classifier_heldout (the final_f1 scoring of wants
#       disaster / sentiment / sentiment_ensemble / sentiment_fewshot)
#   test_vectorSpace_probe — the doc_embedding want's fixed logistic
#       probe, ported from PROBE_CODE


_PORT_SPLIT_STEMS = ('BSdetector', 'disasterTweets', 'movieSentiment',
                     'movieSentimentFewshot', 'BSdetectorTypes')
_PORT_SPLIT_CACHE = {}

# --- market-audition context (the curator's 2026-07-16 ruling: a want's
# scoring truth binds to the BUYER, closing the embedding-path cross-truth
# arbitrage the labeler text guard could not reach) ---
# PORT_BUYER_DATA: the auditing buyer's data terminal for the evaluation
#   in flight — published by MarketAgent.pass_all_tests around each
#   perform_test and cleared after. None outside a market audition
#   (offline/direct calls keep the v2 chain-truth fallback semantics).
# PORT_CURRENT_CHAIN: the chain tree being evaluated — published by
#   MarketSim.call_emergent_function; lets the scorer refuse chains that
#   wire a FOREIGN port corpus terminal inside themselves.
# PORT_BUYER_KEYED: scorer roots whose memoise cache key must carry the
#   buyer's terminal (MarketSim.call_memoise_pickle) — a self-contained
#   foreign chain drops the buyer's appended terminal from the karva tree,
#   so without per-buyer keys the first buyer's cached score would cross
#   wants. Read only by the port machinery; historical scenarios are
#   byte-identical (no scorer in PORT_BUYER_KEYED ever appears in them).
PORT_BUYER_DATA = None
PORT_CURRENT_CHAIN = None
# F4 (2026-09-10): every scorer that calls _buyer_truth_labels() must be
# keyed by buyer, or the first buyer's cached score is served to the next.
PORT_BUYER_KEYED = ('test_classifier_f1heldout',
                    'test_classifier_f1macroHeldout',
                    'test_clusterer_ideologyAMI',
                    'test_clusterer_iraMutualInfo',
                    # 5 Oct 2026 (her yes to a second corpus): the downstream test scores against the buyer's truth
                    # through test_classifier_f1heldout, and the probe takes the buyer's corpus when buyers supply the
                    # truth, so both are keyed by buyer as F4 requires
                    'test_labeler_downstreamF1',
                    'test_vectorSpace_probe')

# PORT_BUYER_TRUTH: whether this world lets the BUYER supply the ground truth
#   for a test that needs one, published beside PORT_BUYER_DATA. Off by
#   default; the flag is buyer_supplies_truth.
#
# WHY IT EXISTS (measured 2026-08-21). test_classifier_f1heldout is held by 20
# of the 56 authored humans - the largest single want in the market - and has
# never returned a non-zero score in its life. The reason is not a bug in the
# arithmetic. It requires the SELLER'S delivered chain to carry the BUYER'S
# corpus ground truth, exactly: Y must equal that corpus's label sidecar. Truth
# enters a chain only through a labeler part, and a seller posts its product
# before any buyer looks at it, so it cannot know whose corpus to carry. It
# would have to guess one corpus in twelve. Measured supply: 15 classifier
# offers, 0 of them buying a labeler at all.
#
# The buyer already publishes its corpus at audition time. Taking the truth
# from there removes the coincidence instead of hoping for it.
PORT_BUYER_TRUTH = False


def _buyer_truth_labels():
    """Ground-truth labels for the corpus the BUYER asked for, or None.

    Uniform across every corpus that has a sidecar: the file is named for the
    last segment of the corpus path, so data_freetext_movieSentiment ->
    movieSentiment_labels.npy and data_vector_blobs -> blobs_labels.npy.
    """
    import os
    import numpy
    if not PORT_BUYER_TRUTH or PORT_BUYER_DATA is None:
        return None
    d = str(PORT_BUYER_DATA)
    if not d or d.endswith('stop'):
        return None
    stem = d.rsplit('_', 1)[-1]
    if not stem:
        return None
    lp = os.path.join(os.getcwd(), 'data', stem + '_labels.npy')
    if not os.path.exists(lp):
        return None
    try:
        return numpy.load(lp)
    except (OSError, ValueError):
        return None



_PORT_DATA_PREFIX = 'data_freetext_'


def _port_stem_pair(stem):
    # (labels, istest) sidecar pair for a port corpus, cached per stem
    # (static for a run); None when the stem is unknown or unstaged.
    import os
    import numpy
    if stem not in _PORT_SPLIT_STEMS:
        return None
    if stem not in _PORT_SPLIT_CACHE:
        lp = os.path.join(os.getcwd(), 'data', stem + '_labels.npy')
        mp = os.path.join(os.getcwd(), 'data', stem + '_istest.npy')
        if os.path.exists(lp) and os.path.exists(mp):
            _PORT_SPLIT_CACHE[stem] = (numpy.load(lp).astype('int64'),
                                       numpy.load(mp).astype(bool))
        else:
            _PORT_SPLIT_CACHE[stem] = None
    return _PORT_SPLIT_CACHE[stem]


def _port_chain_data_stems(chain):
    # Port-corpus stems of every data terminal wired in a published chain
    # tree ({prefixed_name: [prefixed child names]}).
    import re
    stems = set()
    if not chain:
        return stems
    names = list(chain.keys())
    for args in chain.values():
        names.extend(a for a in args if a)
    for name in names:
        base = re.sub(r'^f\d+_', '', str(name))
        if base.startswith(_PORT_DATA_PREFIX):
            stem = base[len(_PORT_DATA_PREFIX):]
            if stem in _PORT_SPLIT_STEMS:
                stems.add(stem)
    return stems


def _port_fixed_split(Y):
    # Identify which port corpus a chain's ground-truth vector came from
    # (exact prefix match against the label sidecars) and return that
    # corpus's fixed LLM-world train/test mask for the received prefix:
    # a bool array where True = an exact test.csv row. Returns None when
    # Y is not a port corpus's truth (cluster labels, bills ideology,
    # anything else) — callers then keep their historical behavior.
    import numpy
    if Y is None:
        return None
    Y = numpy.asarray(Y).ravel()
    n = len(Y)
    if n < 2 or not numpy.issubdtype(Y.dtype, numpy.number):
        return None
    try:
        Yi = Y.astype('int64')
    except (ValueError, TypeError):
        return None
    for stem in _PORT_SPLIT_STEMS:
        pair = _port_stem_pair(stem)
        if pair is None:
            continue
        L, M = pair
        if n <= len(L) and numpy.array_equal(L[:n], Yi):
            return M[:n].copy()
    return None


def _port_or_seeded_split(Y, n):
    # Train/test row division for the classifier engines. When Y is a port
    # corpus's ground truth, the division is the LLM world's FIXED file
    # split (train.csv rows / test.csv rows via the istest sidecar) — no
    # test row is ever trained on, scores are cross-world comparable. If
    # the fixed split is unusable on this prefix (no test rows survive
    # head-truncation, or the train side is single-class), return
    # (None, None): the chain yields no predictions rather than silently
    # leaking test rows into training. For every non-port Y the historical
    # seeded stratified 70/30 split is reproduced exactly.
    import numpy as np
    mask = _port_fixed_split(Y)
    if mask is not None:
        idx_train = np.where(~mask)[0]
        idx_test = np.where(mask)[0]
        if len(idx_train) >= 2 and len(idx_test) >= 1 \
                and len(np.unique(np.asarray(Y)[idx_train])) >= 2:
            return idx_train, idx_test
        return None, None
    from sklearn.model_selection import train_test_split
    idx = np.arange(n)
    counts = np.bincount(Y[Y >= 0]) if (Y >= 0).any() else np.array([0])
    strat = Y if counts.size and counts.min() >= 2 else None
    idx_train, idx_test, _, _ = train_test_split(
        idx, Y, test_size=0.3, random_state=42, stratify=strat)
    return idx_train, idx_test


_LABELER_REF_TEXTS = {}


def _labeler_text_matches(corpus_stem, docs, k=5):
    # True iff the first k docs look like (a preprocessed form of) this
    # labeler's own corpus rows: each doc's tokens must be a subset of the
    # corresponding staged-csv row's tokens. Legitimate upstreams (strip,
    # stopwords, tag) only remove tokens or retokenize, and row order is
    # preserved (no shuffle in these ontologies), so same-corpus chains
    # pass; foreign text fails on essentially every row. Embedding inputs
    # never reach this check (text is gone) — the embedding path is closed
    # by buyer-truth binding instead (PORT_BUYER_DATA; PORT_REPORT.md
    # caveat 6, fixed).
    import os
    if corpus_stem not in _LABELER_REF_TEXTS:
        import pandas as pd
        path = os.path.join(os.getcwd(), 'data', corpus_stem + '.csv')
        if not os.path.exists(path):
            _LABELER_REF_TEXTS[corpus_stem] = None
        else:
            with open(path, encoding='ISO-8859-1') as peek:
                first_line = peek.readline()
            sep = ';' if first_line.count(';') > first_line.count(',') else ','
            head = pd.read_csv(path, encoding='ISO-8859-1', sep=sep,
                               nrows=4 * k)
            col = 'text' if 'text' in head.columns else \
                [c for c in head.columns if 'text' in c.lower()][0]
            ref = [t for t in head[col].tolist()
                   if isinstance(t, str) and t][:k]
            _LABELER_REF_TEXTS[corpus_stem] = ref
    ref = _LABELER_REF_TEXTS[corpus_stem]
    if not ref:
        return False
    m = min(len(ref), len(docs), k)
    if m < 1:
        return False
    hits = 0
    for i in range(m):
        have = set(str(docs[i]).split())
        want = set(ref[i].split())
        if have <= want:
            hits += 1
    # Tolerate one pathological row (e.g. a URL glued to a word, which
    # preprocessor_freetext_strip rewrites mid-token); foreign corpora
    # fail essentially every row, so the separation stays enormous.
    return hits >= max(1, m - 1)


def _labeler_pair(corpus_stem, X):
    # Shared engine for the labeler_* parts: pair a document representation
    # of a corpus with that corpus's ground-truth labels, producing the
    # (float32, int32) tuple every classifier_* engine consumes. Truth
    # sidecar + head-truncation alignment follow the
    # classifier_billsIdeology_logistic / test_clusterer_ideologyAMI
    # convention (loaders and vector spaces preserve row order; the
    # shared-scenario ontologies omit the shuffle preprocessor).
    #
    # Scaffolding (the prefab precedent, the curator's 2026-07-06 design): if X
    # is still raw text (a list of strings / TaggedDocuments) instead of an
    # embedding, embed it with the prefab LSA first. A newborn seller who
    # chains classifier_x(labeler_y(corpus)) therefore owns a working,
    # verifiable product on day one; a real vectorSpace upstream outgrows
    # the floor and gets paid for it (quality-scaled settlement).
    import os
    import numpy
    # A LABELER LABELS WHAT IT IS GIVEN (her ruling 2026-08-21, "fix it so it
    # doesnt have a coincidence"; the same "make it general" she applied to the
    # corpus-welded tests).
    #
    # Welded to one corpus, a labeler could only help when the SELLER happened
    # to have bought the labeler matching the BUYER's corpus - and the seller
    # posts its product before any buyer looks, so it cannot know. Measured on
    # gate_comp_seed306: classifier humans asked about five corpora
    # (movieSentiment 100, disasterTweets 60, movieSentimentFewshot 20,
    # BSdetector 20, BSdetectorTypes 10) while the composed classifiers had
    # bought labelers for two, overlapping in exactly one place. The composite
    # assembled and could almost never be paid for.
    #
    # At audition the buyer's corpus is already published, so the labeler takes
    # ITS truth - the identical mechanism the tests use. The classifier must
    # still BUY a labeler from another agent to have labels at all, so the
    # multi-agent composition is untouched; what goes away is the requirement
    # that it guessed which one. Outside an audition the corpus-welded
    # behaviour is unchanged.
    _bt = _buyer_truth_labels()
    if _bt is not None and X is not None and not isinstance(X, tuple):
        _L = numpy.asarray(_bt).ravel()
        _Xa = X
        try:
            _n = len(_Xa)
        except TypeError:
            _n = 0
        if _n >= 2 and len(_L) >= _n:
            return (_Xa, _L[:_n])
    labels_path = os.path.join(os.getcwd(), 'data',
                               corpus_stem + '_labels.npy')
    if X is None or not os.path.exists(labels_path):
        return None
    if isinstance(X, tuple):
        # A tuple is another part's (X, Y) contract, never a corpus —
        # e.g. a labeler stacked on a labeler (observed in the 2026-07-16
        # smoke). Malformed chain: refuse cleanly instead of stringifying
        # arrays into a fake 2-document corpus. Score outcome unchanged
        # (those chains scored 0 via downstream guards anyway).
        return None
    if isinstance(X, list):
        if len(X) < 2:
            return None
        if not all(isinstance(d, str) or hasattr(d, 'words') for d in X[:5]):
            return None
        docs = [' '.join(d.words) if hasattr(d, 'words') else str(d)
                for d in X]
        # Corpus-identity guard (2026-07-16, exposed by the v2 smoke): a
        # labeler is "the labeled dataset OF this corpus" — given text from
        # a DIFFERENT corpus (labeler_movieSentiment over disaster tweets
        # settled at the F1 no-skill floor, scored against the wrong
        # want's truth), it must refuse rather than mint a foreign-truth
        # pair. The buyer's own scoring truth can only ride a same-corpus
        # labeler. Row-order preprocessors only remove/retokenize (strip,
        # stopwords, tag; shuffle is not in these ontologies), so doc i of
        # a legitimate chain is always a token-subset of corpus doc i.
        if not _labeler_text_matches(corpus_stem, docs):
            return None
        return None  # prefab floor removed (the curator 2026-07-19: probes only)
    X = numpy.asarray(X)
    if X.ndim != 2 or len(X) < 2:
        return None
    L = numpy.load(labels_path)
    n = min(len(X), len(L))
    if n < 2:
        return None
    return (numpy.asarray(X[:n], dtype='float32'),
            numpy.asarray(L[:n], dtype='int32'))


def labeler_disasterTweets(X):
    # Real-world name: "labeled disaster-tweet dataset" — the human truth
    # of the LLM world's `disaster` want ("is this tweet about a real
    # disaster?") joined to a representation of data_freetext_disasterTweets.
    # _args:   [{"type": "numpy.ndarray", "dtype": "float32"}]
    # _return: [{"type": "tuple", "dtype": "(float32,int32)"}]
    print('labeler_disasterTweets')
    return _labeler_pair('disasterTweets', X)


# --- LABELED VECTOR CORPORA (her rulings b1/b2, 2026-08-21) ------------------
# The seven vector generators each compute a ground-truth Y and return only X,
# so the truth every AMI/held-out test needs was being discarded at source.
# These parts recover it, following the labeler convention already in use: a
# labeler IS the labeled dataset of its corpus, returning the (X, labels) tuple
# the classifier and cluster tests consume.
#
# TWO FORMS, because the corpora divide in two. Verified under the real
# evaluation sandbox (which seeds from the chain and only the chain):
#   blobs, varied, ansio, variedBlobs5, noisyMoons pin a random_state, so they
#   are byte-identical in every chain - a stored sidecar aligns, and these take
#   the ARITY-1 form that pairs whatever representation it is handed with the
#   stored truth.
#   circles and moons pin nothing, deliberately (her earlier ruling: they are
#   deterministic from the chain via the sandbox, not from a constant). They
#   therefore draw DIFFERENT points in every chain, no stored file could ever
#   align, and they take the ARITY-0 form that generates the pair in ONE draw
#   so the labels belong to the points they came with.
#
# Sidecar truth was checked, not assumed: nearest-neighbour label purity is
# 0.967-1.000 against random baselines of 0.200-0.500. noisyMoons scores 0.994
# on purity while KMeans reaches only 0.135 AMI against it - the labels mark the
# crescents correctly and centroid clustering is simply the wrong tool for them,
# which is what that corpus exists to demonstrate.


def _vector_labeler_pair(corpus_stem, X):
    # The vector counterpart of _labeler_pair. One difference is honest to
    # state: the text version can verify corpus identity by checking that each
    # document is a token-subset of the reference corpus. There is no such check
    # for numeric rows once a vectorSpace has transformed them, so this guards
    # what it can - shape, and enough stored labels to cover the rows - and
    # relies on the same row-order preservation the text labelers already
    # require.
    import os
    import numpy
    # A LABELER LABELS WHAT IT IS GIVEN - see _labeler_pair. Welded to one
    # corpus these had the same coincidence requirement.
    _bt = _buyer_truth_labels()
    if _bt is not None and X is not None and not isinstance(X, tuple):
        _L = numpy.asarray(_bt).ravel()
        _Xa = numpy.asarray(X)
        if _Xa.ndim == 2 and len(_Xa) >= 2 and len(_L) >= len(_Xa):
            return (_Xa, _L[:len(_Xa)])
    labels_path = os.path.join(os.getcwd(), 'data',
                               corpus_stem + '_labels.npy')
    if X is None or not os.path.exists(labels_path):
        return None
    if isinstance(X, tuple):
        # Another part's (X, Y) contract, never a corpus - same refusal as
        # _labeler_pair makes for a labeler stacked on a labeler.
        return None
    X = numpy.asarray(X)
    if X.ndim != 2 or len(X) < 2:
        return None
    L = numpy.load(labels_path)
    if len(L) < len(X):
        return None
    return (X, L[:len(X)])


def labeler_vector_blobs(X):
    print('labeler_vector_blobs')
    return _vector_labeler_pair('blobs', X)


def labeler_vector_varied(X):
    print('labeler_vector_varied')
    return _vector_labeler_pair('varied', X)


def labeler_vector_ansio(X):
    print('labeler_vector_ansio')
    return _vector_labeler_pair('ansio', X)


def labeler_vector_variedBlobs5(X):
    print('labeler_vector_variedBlobs5')
    return _vector_labeler_pair('variedBlobs5', X)


def labeler_vector_noisyMoons(X):
    print('labeler_vector_noisyMoons')
    return _vector_labeler_pair('noisyMoons', X)


def labeler_vector_circles(n_samples=1500):
    # Arity 0: generates and labels in ONE draw, so the labels belong to these
    # points. Mirrors data_vector_circles exactly and stays unpinned with it.
    from sklearn.datasets import make_circles
    print('labeler_vector_circles')
    X, Y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
    return (X, Y)


def labeler_vector_moons(n_samples=1500):
    # Arity 0, same reason as labeler_vector_circles.
    from sklearn.datasets import make_moons
    print('labeler_vector_moons')
    X, Y = make_moons(n_samples=n_samples, noise=.05)
    return (X, Y)


def labeler_BSdetectorTypes(X):
    # The eight-class truth of the BSdetector corpus (her order
    # 2026-08-10, "include all the wants... the bigger build"): native
    # source types bias/bs/conspiracy/fake/hate/junksci/satire/state,
    # alphabetically encoded 0-7, sidecar row-aligned with the loader
    # (proof: her minority-positive binarization of these labels
    # reproduces BSdetector_labels.npy at 100.00%). Same fixed
    # train/test mask as the binary corpus.
    # _args:   [{"type": "numpy.ndarray", "dtype": "float32"}]
    # _return: [{"type": "tuple", "dtype": "(float32,int32)"}]
    print('labeler_BSdetectorTypes')
    return _labeler_pair('BSdetectorTypes', X)


def labeler_BSdetector(X):
    # Real-world name: "labeled fake-news dataset" — the truth of the
    # BSdetector wants (the curator's 2026-08-07 minority-positive mapping:
    # label 1 = conspiracy, fake, junksci; about 4.3 percent positive)
    # joined to a representation of data_freetext_BSdetector.
    # _args:   [{"type": "numpy.ndarray", "dtype": "float32"}]
    # _return: [{"type": "tuple", "dtype": "(float32,int32)"}]
    print('labeler_BSdetector')
    return _labeler_pair('BSdetector', X)


def labeler_movieSentiment(X):
    # Real-world name: "labeled movie-review sentiment dataset" — the truth
    # of the LLM world's `sentiment` and `sentiment_ensemble` wants joined
    # to a representation of data_freetext_movieSentiment.
    # _args:   [{"type": "numpy.ndarray", "dtype": "float32"}]
    # _return: [{"type": "tuple", "dtype": "(float32,int32)"}]
    print('labeler_movieSentiment')
    return _labeler_pair('movieSentiment', X)


def labeler_movieSentimentFewshot(X):
    # Real-world name: "the 500 labeled reviews" — the truth of the LLM
    # world's aspirational `sentiment_fewshot` want (500-row train.csv)
    # joined to a representation of data_freetext_movieSentimentFewshot.
    # _args:   [{"type": "numpy.ndarray", "dtype": "float32"}]
    # _return: [{"type": "tuple", "dtype": "(float32,int32)"}]
    print('labeler_movieSentimentFewshot')
    return _labeler_pair('movieSentimentFewshot', X)


def classifier_svm_linear(XY):
    # Linear support-vector classifier (sklearn LinearSVC) in the exact
    # classifier_logistic_v2 harness: consumes an upstream (X, Y) pair and
    # returns (X, Y, pred) with pred = -1 on training rows so the
    # classifier tests score held-out rows only; any failure yields all -1
    # (never a constant that banks the base rate). Split: the LLM world's
    # FIXED train/test file split when Y is a port corpus's truth
    # (_port_or_seeded_split; the curator's 2026-07-16 exact-split ruling),
    # seeded stratified 70/30 otherwise. A SECOND LEARNER FAMILY
    # (max-margin linear) so the ensemble want's ">= 2 distinct families"
    # is a real in-world choice. LinearSVC is in the LLM world's catalog.
    # _args:   [{"type": "tuple", "dtype": "(float32,int32)"}]
    # _return: [{"type": "tuple", "dtype": "(float32,int32,int32)"}]
    import numpy as np
    print('classifier_svm_linear')
    X, Y = (XY[0], XY[1]) if XY is not None and len(XY) >= 2 else (None, None)
    n = len(X) if X is not None else 0
    if n < 10 or Y is None or len(np.unique(np.asarray(Y))) < 2:
        return (X, Y, np.full(n, -1, dtype=np.int32))
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.int32)
    try:
        from sklearn.svm import LinearSVC
        idx_train, idx_test = _port_or_seeded_split(Y, n)
        pred = np.full(n, -1, dtype=np.int32)
        if idx_train is not None:
            clf = LinearSVC(max_iter=2000, random_state=42)
            clf.fit(X[idx_train], Y[idx_train])
            pred[idx_test] = clf.predict(X[idx_test]).astype(np.int32)
    except Exception:
        pred = np.full(n, -1, dtype=np.int32)
    return (X, Y, pred)


def classifier_naivebayes_multinomial(XY):
    # Multinomial naive Bayes (sklearn MultinomialNB) in the same harness —
    # a THIRD learner family (generative count model). MultinomialNB
    # requires non-negative features while chain embeddings (doc2vec, SVD +
    # StandardScaler) carry negatives, so features are shifted by the
    # TRAIN-split per-column minimum (test rows clipped at 0 — no held-out
    # leakage into the shift). MultinomialNB is in the LLM world's catalog.
    #
    # Niche, proven by execution (2026-07-16 harness): on count-like
    # features (vectorSpace_sklearn_tfidf upstream) it is a real classifier
    # (F1 0.56 on the disaster corpus); on dense low-rank embeddings
    # (prefab LSA 20-dim) it degenerates to the majority class and the
    # constant-predictor guard scores it 0. Kept as-is deliberately: which
    # representation a family needs is a specialization axis the market is
    # supposed to learn, not something the engine should paper over.
    # _args:   [{"type": "tuple", "dtype": "(float32,int32)"}]
    # _return: [{"type": "tuple", "dtype": "(float32,int32,int32)"}]
    import numpy as np
    print('classifier_naivebayes_multinomial')
    X, Y = (XY[0], XY[1]) if XY is not None and len(XY) >= 2 else (None, None)
    n = len(X) if X is not None else 0
    if n < 10 or Y is None or len(np.unique(np.asarray(Y))) < 2:
        return (X, Y, np.full(n, -1, dtype=np.int32))
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.int32)
    try:
        from sklearn.naive_bayes import MultinomialNB
        idx_train, idx_test = _port_or_seeded_split(Y, n)
        pred = np.full(n, -1, dtype=np.int32)
        if idx_train is not None:
            shift = X[idx_train].min(axis=0, keepdims=True)
            Xs = np.clip(X - shift, 0.0, None)
            clf = MultinomialNB()
            clf.fit(Xs[idx_train], Y[idx_train])
            pred[idx_test] = clf.predict(Xs[idx_test]).astype(np.int32)
    except Exception:
        pred = np.full(n, -1, dtype=np.int32)
    return (X, Y, pred)


def classifier_ensemble_vote(A, B):
    # THE ensemble combiner (arity 2, clusterer_consensus's pattern) — the
    # chain-world realization of the LLM world's sentiment_ensemble recipe
    # ("MUST BE an ensemble of at least two distinct learner families").
    # Its ontology home classifier.ensemble.vote makes the ITEM TYPE PATH
    # do the recipe binding: a want demanding item classifier_ensemble_stop
    # is only satisfiable through this subtree.
    #
    # Each input is a classifier's (X, Y, pred) with pred = -1 on that
    # learner's training rows. On rows where BOTH inputs carry a held-out
    # prediction: agreement -> the agreed label; disagreement ->
    # deterministic alternation between the two voters by row parity (the
    # hard-label analog of averaging two voters — with exactly two, a
    # fixed always-first tie-break would collapse the vote to input A).
    # All other rows are -1.
    #
    # Diversity gate (clusterer_consensus precedent, the curator's 2026-07-15
    # revision: "a combination of clones is not a combination"): refused
    # (None -> score 0, no settle) when either input is missing or a
    # constant voter, or when the two voters agree on > 98% of the jointly
    # held rows — e.g. the same chain bought twice, which memoisation makes
    # byte-identical.
    # _args:   [{"type": "tuple", "dtype": "(float32,int32,int32)"},
    #           {"type": "tuple", "dtype": "(float32,int32,int32)"}]
    # _return: [{"type": "tuple", "dtype": "(float32,int32,int32)"}]
    import numpy as np
    print('classifier_ensemble_vote')
    if A is None or B is None or len(A) < 3 or len(B) < 3:
        return None
    Xa, Ya, Pa = A[0], A[1], A[2]
    Pb = B[2]
    if Pa is None or Pb is None or Ya is None:
        return None
    Pa = np.asarray(Pa).ravel()
    Pb = np.asarray(Pb).ravel()
    Ya = np.asarray(Ya).ravel()
    n = min(len(Pa), len(Pb), len(Ya))
    if n < 10:
        return None
    Pa, Pb, Ya = Pa[:n], Pb[:n], Ya[:n]
    held = (Pa != -1) & (Pb != -1)
    if held.sum() < 2:
        return None
    # A constant voter is not a classifier (test_classifier_heldout's
    # degeneracy rule applied at combination time).
    if len(np.unique(Pa[held])) < 2 or len(np.unique(Pb[held])) < 2:
        return None
    if float(np.mean(Pa[held] == Pb[held])) > 0.98:
        return None
    out = np.full(n, -1, dtype=np.int32)
    idx = np.where(held)[0]
    tie_to_a = (idx % 2) == 0
    out[idx] = np.where(Pa[idx] == Pb[idx], Pa[idx],
                        np.where(tie_to_a, Pa[idx], Pb[idx]))
    Xa = np.asarray(Xa)[:n] if Xa is not None else None
    return (Xa, np.asarray(Ya, dtype=np.int32), out)


def test_classifier_f1macroHeldout(XYP):
    # Macro-averaged F1 on held-out rows - the multi-class measure
    # (her yes 2026-08-10), carrying the SAME truth guard as
    # test_classifier_f1heldout: in a market audition the scoring
    # truth is the buyer's port corpus and nothing else (foreign port
    # terminals refuse the audition); outside an audition the chain's
    # Y must prefix-match a registered port sidecar. Macro averaging
    # scores every class equally, so ignoring rare classes is
    # punished. Constant predictors score 0.
    # _args:   [{"type": "tuple", "dtype": "(float32,int32,int32)"}]
    # _return: [{"type": "float"}]
    import numpy as np
    print('test_classifier_f1macroHeldout')
    if XYP is None or len(XYP) < 3 or XYP[1] is None or XYP[2] is None:
        return 0.0
    Y, P = np.asarray(XYP[1]).ravel(), np.asarray(XYP[2]).ravel()
    n = min(len(Y), len(P))
    if n < 2:
        return 0.0
    Y, P = Y[:n], P[:n]
    _BL = _buyer_truth_labels()
    if _BL is not None:
        # THE BUYER SUPPLIES THE TRUTH - see test_classifier_f1heldout.
        _L = np.asarray(_BL).ravel()
        if len(_L) < n:
            return 0.0
        Y = _L[:n]
        mask = np.ones(n, dtype=bool)
    elif PORT_BUYER_DATA is not None:
        d = str(PORT_BUYER_DATA)
        stem = d[len(_PORT_DATA_PREFIX):] \
            if d.startswith(_PORT_DATA_PREFIX) else None
        pair = _port_stem_pair(stem) if stem else None
        if pair is None:
            return 0.0
        foreign = _port_chain_data_stems(PORT_CURRENT_CHAIN) - {stem}
        if foreign:
            return 0.0
        L, M = pair
        try:
            Yi = Y.astype('int64')
        except (ValueError, TypeError):
            return 0.0
        if n > len(L) or not np.array_equal(L[:n], Yi):
            return 0.0
        mask = M[:n].copy()
    else:
        mask = _port_fixed_split(Y)
    if mask is None:
        return 0.0
    scoreable = (P != -1) & mask
    if scoreable.sum() < 5:
        return 0.0
    yh, ph = Y[scoreable], P[scoreable]
    if len(np.unique(ph)) < 2:
        return 0.0
    from sklearn.metrics import f1_score
    try:
        return float(np.clip(f1_score(yh, ph, average='macro',
                                      zero_division=0), 0.0, 1.0))
    except Exception:
        return 0.0

def data_vector_noisyMoons(n_samples=1500):
    # Her yes (2026-08-10): the two-level-qualified geometric corpus -
    # two interleaved moons with real noise POINTS among them (10% of
    # rows uniform over the box). The density clusterers' textbook
    # home: eps, core size, linkage, and threshold genuinely decide.
    from sklearn.datasets import make_moons
    import numpy as np
    print('data_vector_noisyMoons')
    n_noise = max(1, n_samples // 10)
    X, _ = make_moons(n_samples=n_samples - n_noise, noise=0.12,
                      random_state=11)
    rng = np.random.RandomState(11)
    noise = rng.uniform(X.min(0) - 0.5, X.max(0) + 0.5, size=(n_noise, 2))
    return np.vstack([X, noise])


def data_vector_variedBlobs5(n_samples=1500):
    # Her yes (2026-08-10): five blobs whose spreads run from tight to
    # diffuse (0.4 to 3.2). Measured two-level qualification: the
    # right tool AND the right cluster count both matter (param gap
    # +2.45 calinski, +0.24 silhouette).
    from sklearn.datasets import make_blobs
    print('data_vector_variedBlobs5')
    X, _ = make_blobs(n_samples=n_samples, centers=5,
                      cluster_std=[0.4, 0.8, 1.6, 2.4, 3.2],
                      random_state=11)
    return X


def _f1heldout_core(XYP):
    """The held-out F1 computation of test_classifier_f1heldout without the whole's bar; shared with the labelled
    embedding's downstream test, which must never be barred."""
    # F1 variant of test_classifier_heldout — the LLM world's final_f1
    # scoring ("F1 for label=1 on held-out data") ported to the chain
    # world. test_classifier_heldout is untouched.
    #
    # v2 (the curator's 2026-07-16 exact-split ruling): the score is F1 for
    # label==1 computed ONLY on the LLM world's test.csv rows — the chain
    # must carry a port corpus's ground truth (Y matches a label sidecar;
    # _port_fixed_split), and only predicted rows on that corpus's fixed
    # test mask count. A chain whose Y is anything else (cluster labels,
    # another corpus) scores 0: the buyer's truth was not delivered — the
    # LLM buyer likewise scores a delivered script only against ITS OWN
    # test.csv. Both sides are binarized (== 1 vs rest); on the port
    # corpora (labels {0,1}) this equals sklearn f1_score(pos_label=1).
    # Degeneracy guard as in test_classifier_heldout: a constant predictor
    # scores 0 — always-1 would otherwise bank 2p/(1+p) F1 on a
    # p-prevalent corpus, the same base-rate banking the accuracy test
    # refuses.
    #
    # v3 (same ruling, the embedding-path closure): during a MARKET
    # AUDITION (MarketAgent.pass_all_tests publishes PORT_BUYER_DATA) the
    # scoring truth is the BUYER's corpus and nothing else — a
    # self-contained chain that carries a different port corpus's truth
    # (e.g. classifier(labeler_movieSentiment(tfidf(data_movieSentiment)))
    # offered to the disaster human) scores 0 instead of the ~0.78 its own
    # truth would print. The published chain tree is also checked: a
    # foreign port terminal wired anywhere inside the delivered chain
    # refuses the audition. Outside an audition (no publication) the v2
    # chain-truth semantics hold, for offline/harness use.
    import numpy as np
    print('test_classifier_f1heldout')
    if XYP is None or len(XYP) < 3 or XYP[1] is None or XYP[2] is None:
        return 0.0
    Y, P = np.asarray(XYP[1]).ravel(), np.asarray(XYP[2]).ravel()
    n = min(len(Y), len(P))
    if n < 2:
        return 0.0
    Y, P = Y[:n], P[:n]
    _BL = _buyer_truth_labels()
    if _BL is not None:
        # THE BUYER SUPPLIES THE TRUTH (flag buyer_supplies_truth). The
        # seller posts before any buyer looks, so it cannot know whose
        # corpus to build in; requiring it to have guessed is why this
        # test never scored. The predictions are judged against the
        # buyer's own labels, and the classifier's own -1 marks say which
        # rows it held out.
        _L = np.asarray(_BL).ravel()
        if len(_L) < n:
            return 0.0
        Y = _L[:n]
        mask = np.ones(n, dtype=bool)
    elif PORT_BUYER_DATA is not None:
        # Market audition: the buyer's truth, period.
        d = str(PORT_BUYER_DATA)
        stem = d[len(_PORT_DATA_PREFIX):] \
            if d.startswith(_PORT_DATA_PREFIX) else None
        pair = _port_stem_pair(stem) if stem else None
        if pair is None:
            # The buyer's want is not a port corpus — this scorer has no
            # truth to score against.
            return 0.0
        foreign = _port_chain_data_stems(PORT_CURRENT_CHAIN) - {stem}
        if foreign:
            # A different port corpus is wired inside the delivered chain.
            return 0.0
        L, M = pair
        try:
            Yi = Y.astype('int64')
        except (ValueError, TypeError):
            return 0.0
        if n > len(L) or not np.array_equal(L[:n], Yi):
            # The chain never carried the buyer's ground truth.
            return 0.0
        mask = M[:n].copy()
    else:
        mask = _port_fixed_split(Y)
    if mask is None:
        # Not a port corpus's truth — the want's ground truth was never in
        # the chain; nothing here is worth an F1 against.
        return 0.0
    scoreable = (P != -1) & mask
    if scoreable.sum() < 2:
        return 0.0
    yh, ph = Y[scoreable], P[scoreable]
    if len(np.unique(ph)) < 2:
        # Constant predictor — reject, never let it bank the base rate.
        return 0.0
    yb, pb = (yh == 1), (ph == 1)
    tp = float(np.sum(yb & pb))
    fp = float(np.sum(~yb & pb))
    fn = float(np.sum(yb & ~pb))
    if tp == 0.0:
        return 0.0
    f1 = 2.0 * tp / (2.0 * tp + fp + fn)
    f1 = float(np.clip(f1, 0.0, 1.0))
    return f1


def test_classifier_f1heldout(XYP):
    # The whole classifier's test: the held-out F1 of _f1heldout_core, then the whole's bar (below).
    # _args:   [{"type": "tuple", "dtype": "(float32,int32,int32)"}]
    # _return: [{"type": "float"}]
    import numpy as np
    f1 = _f1heldout_core(XYP)
    # THE WHOLE'S BAR (her yes, 6 Oct 2026 19:15, decision 10: "the whole classifier pays only when its score beats a
    # bar that no single default chain reaches"; off unless ML_WHOLE_BAR is set, e.g. "movieSentiment:0.7304,
    # BSdetector:0.2716", each bar the held-out F1 of the corpus's default chain: strip, TF-IDF 300 features, logistic
    # regression C=1, measured offline in ML_PARTS_BY_CORPUS_20261005.md). A whole at or below the bar scores 0, so a
    # lone builder of the default chain earns nothing and partial credit cannot climb; above it the usual score
    # stands. The score itself is changed, so the human's utility and the payment agree, as the cipher's
    # all-or-nothing whole does. The corpus is the audited buyer's (PORT_BUYER_DATA) or, offline, the chain's.
    _bar_env = _os_quiet.environ.get('ML_WHOLE_BAR')
    if _bar_env:
        _stem_b = None
        if PORT_BUYER_DATA is not None:
            _d_b = str(PORT_BUYER_DATA)
            _stem_b = _d_b[len(_PORT_DATA_PREFIX):] if _d_b.startswith(_PORT_DATA_PREFIX) else None
        if _stem_b is None:
            _st_b = _port_chain_data_stems(PORT_CURRENT_CHAIN) if PORT_CURRENT_CHAIN is not None else set()
            _stem_b = next(iter(_st_b)) if len(_st_b) == 1 else None
        _bars = {}
        for _kv in _bar_env.split(','):
            if ':' in _kv:
                _k_b, _v_b = _kv.split(':', 1)
                _bars[_k_b.strip()] = float(_v_b)
        if _stem_b in _bars and f1 <= _bars[_stem_b]:
            return 0.0
    return f1


def test_vectorSpace_probe(X):
    # The doc_embedding want's fixed probe, ported from the LLM world's
    # PROBE_CODE (llm_market/core_market.py): the buyer takes the DELIVERED
    # EMBEDDING of the movie-sentiment corpus, trains
    # LogisticRegression(max_iter=2000) on the corpus's train rows, and
    # pays ACCURACY on its test rows. v2 (the curator's 2026-07-16 exact-split
    # ruling): the train/test division is the LLM world's FIXED file split
    # (data/movieSentiment_istest.npy) — X_train/X_test and y_train/y_test
    # are exactly PROBE_CODE's, evaluated on however many rows of the
    # fixed split flowed through the chain (head-truncation keeps the
    # sidecars aligned; the interleaved staging keeps both splits present
    # in any prefix). Raw accuracy, exactly like PROBE_CODE — an
    # uninformative embedding floors near 0.5 on this balanced corpus, and
    # relative quality is what the market law prices (frontier-scaled
    # settlement), matching the LLM market's treatment of the same probe.
    # _args:   [{"type": "numpy.ndarray", "dtype": "float32"}]
    # _return: [{"type": "float"}]
    import os
    import numpy as np
    print('test_vectorSpace_probe')
    # THE BUYER'S CORPUS (5 Oct 2026, her yes to a second corpus): when the world lets the buyer supply the truth
    # (buyer_supplies_truth) and the auditing buyer asked for a port corpus with its sidecars, the probe uses that
    # corpus's labels and fixed split, as the whole's test and the downstream test already use the buyer's truth.
    # Otherwise the movie corpus, exactly as before.
    stem = 'movieSentiment'
    _d = str(PORT_BUYER_DATA) if PORT_BUYER_DATA is not None else ''
    if PORT_BUYER_TRUTH and _d.startswith(_PORT_DATA_PREFIX):
        _s = _d[len(_PORT_DATA_PREFIX):]
        if _s and os.path.exists(os.path.join(os.getcwd(), 'data', _s + '_labels.npy')) \
                and os.path.exists(os.path.join(os.getcwd(), 'data', _s + '_istest.npy')):
            stem = _s
    labels_path = os.path.join(os.getcwd(), 'data', stem + '_labels.npy')
    mask_path = os.path.join(os.getcwd(), 'data', stem + '_istest.npy')
    if X is None or not os.path.exists(labels_path) \
            or not os.path.exists(mask_path):
        return 0.0
    X = np.asarray(X)
    if X.ndim != 2 or len(X) < 10:
        return 0.0
    L = np.load(labels_path)
    M = np.load(mask_path).astype(bool)
    n = min(len(X), len(L), len(M))
    if n < 10:
        return 0.0
    X = np.asarray(X[:n], dtype=np.float32)
    L = np.asarray(L[:n], dtype=np.int32)
    M = M[:n]
    tr = np.where(~M)[0]
    te = np.where(M)[0]
    if len(tr) < 10 or len(te) < 2 or len(np.unique(L[tr])) < 2:
        return 0.0
    try:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=2000).fit(X[tr], L[tr])
        acc = float(np.mean(clf.predict(X[te]) == L[te]))
    except Exception:
        return 0.0
    return float(np.clip(acc, 0.0, 1.0))


DEDUP_K = 20

def data_freetext_dedupPlanted(relative_path):
    # DEDUP END-TASK terminal (the curator 2026-07-19: an end task that is
    # "almost just data cleaning"). Loads the parent corpus, then appends
    # DISGUISED COPIES of rows 0..DEDUP_K-1 at the end: case flips, URL
    # and punctuation padding, doubled whitespace, benign char noise.
    # Disguises are seeded by the corpus path, so the planted corpus is
    # deterministic (cache-coherent) — honesty comes from the disguise
    # being undoable only by genuine normalization, which sklearn-class
    # chains cannot index-hack around. Convention shared with
    # test_vectorSpace_dedupRecovery: twin j sits at position n_base+j
    # and pairs with row j.
    import random
    docs = data_freetext_csvColumn(relative_path)
    if not docs or len(docs) < DEDUP_K * 3:
        return docs
    rng = random.Random('dedup:' + relative_path)
    STOPW = ['the', 'a', 'of', 'and', 'to', 'in', 'that', 'it', 'as', 'for']
    def disguise(t):
        # Calibrated 2026-07-19 on the real movie corpus: raw tf-idf
        # recovers 0.35 of planted twins; strip+stopwords+lemmatization
        # recovers 1.00 — a 65-point gradient earned by exactly the
        # catalog's cleaning tools. Inflections are undone by the
        # lemmatizer, injected stopwords by the stopword remover, pads
        # by the stripper; light char doubling is residual difficulty.
        words = str(t).split()
        out = []
        for w in words:
            if rng.random() < 0.8:
                w = w + rng.choice(['s', 'ing', 'ed'])
            if rng.random() < 0.12 and len(w) > 3:
                i = rng.randint(1, len(w) - 2)
                w = w[:i] + w[i] + w[i:]
            out.append(w)
            if rng.random() < 0.8:
                out.append(rng.choice(STOPW))
        pad = rng.choice(['!!', '...', '>>', '##'])
        return pad + ' ' + ' '.join(out) + ' ' + pad
    twins = [disguise(docs[j]) for j in range(DEDUP_K)]
    return list(docs) + twins


def test_vectorSpace_dedupRecovery(X):
    # Scores the delivered representation of a dedupPlanted corpus: for
    # each planted twin (last DEDUP_K rows), is its nearest base row
    # exactly its original (row j)? Score = recovered fraction. Cleaning
    # that collapses the disguises makes twin and original coincide; a
    # representation built on raw text leaves them apart.
    # _args:   [{"type": "numpy.ndarray", "dtype": "float32"}]
    # _return: [{"type": "float"}]
    import numpy as np
    print('test_vectorSpace_dedupRecovery')
    if X is None:
        return 0.0
    X = np.asarray(X)
    if X.ndim != 2 or len(X) < DEDUP_K * 3:
        return 0.0
    X = np.asarray(X, dtype=np.float32)
    base = X[:len(X) - DEDUP_K]
    twins = X[len(X) - DEDUP_K:]
    hits = 0
    for j in range(DEDUP_K):
        d = ((base - twins[j]) ** 2).sum(axis=1)
        if int(np.argmin(d)) == j:
            hits += 1
    return float(hits) / float(DEDUP_K)



# === end shared-scenario port block ======================================


# Fill the initial function
# In this case we dont want other programs to fill in the parameters, so we dont put original functions in the registry
curries = {'data_freetext_csvColumn': curr(data_freetext_csvColumn),
           # 'data_vector_blobs': curr(data_vector_blobs,
           'vectorSpace_gensim_doc2vec': curr(vectorSpace_gensim_doc2vec),
           'clusterer_sklearn_kmeans': curr(clusterer_sklearn_kmeans),
           'clusterer_sklearn_agglomerative': curr(clusterer_sklearn_agglomerative),
           'clusterer_sklearn_affinityPropagation': curr(clusterer_sklearn_affinityPropagation),
           'clusterer_sklearn_meanShift': curr(clusterer_sklearn_meanShift),
           'clusterer_sklearn_spectral': curr(clusterer_sklearn_spectral),
           'clusterer_sklearn_ward': curr(clusterer_sklearn_ward),
           'clusterer_sklearn_dbscan': curr(clusterer_sklearn_dbscan),
           'clusterer_sklearn_birch': curr(clusterer_sklearn_birch),
           'clusterer_sklearn_gaussian': curr(clusterer_sklearn_gaussian),
           'clusterer_nltk_agglomerative': curr(clusterer_nltk_agglomerative),
           'clusterer_nltk_kmeans': curr(clusterer_nltk_kmeans)}

# Create the constructions that would be machine learned, using a shortened dataset
registry = {'data_vector_blobs': curr(data_vector_blobs), 'data_vector_moons': curr(data_vector_moons),
            'data_vector_circles': curr(data_vector_circles), 'data_vector_ansio': curr(data_vector_ansio),
            'data_vector_varied': curr(data_vector_varied),
            'test_clusterer_silhouette': curr(test_clusterer_silhouette),
            'test_clusterer_calinskiHarabaz': curr(test_clusterer_calinskiHarabaz),
            # === retired 2026-05-21 (see comment block at top of file): ===
            #   topicSummarizer_basic, test_topicSummarizer_distinctness
            #   topicSummarizer_lda
            #   recommender_basic, test_recommender_coherence
            #   classifier_basic
            #   sentimentClassifier_lexicon, test_sentimentClassifier_polarity
            # test_classifier_loo retained, but rewritten to compute proper
            # held-out accuracy (was a lower-bound surrogate before).
            'anomalyDetector_basic': curr(anomalyDetector_basic),
            'test_classifier_loo': curr(test_classifier_loo),
            'preprocessor_freetext_shuffle': curr(preprocessor_freetext_shuffle),
            'preprocessor_freetext_strip': curr(preprocessor_freetext_strip),
            'preprocessor_freetext_lemmatization': curr(preprocessor_freetext_lemmatization),
            'preprocessor_freetext_tag': curr(preprocessor_freetext_tag),
            # === Intermediate-level tests (preprocessor + vectorSpace direct demand) ===
            # === Substantive end-product implementations ===
            'anomalyDetector_oneClassSVM': curr(anomalyDetector_oneClassSVM),
            'anomalyDetector_oneClassSVM_nu005': curr(anomalyDetector_oneClassSVMwalk)('nu005'),
            'anomalyDetector_oneClassSVM_nu01': curr(anomalyDetector_oneClassSVMwalk)('nu01'),
            'anomalyDetector_oneClassSVM_nu02': curr(anomalyDetector_oneClassSVMwalk)('nu02'),
            'anomalyDetector_basic_cauto': curr(anomalyDetector_basicwalk)('cauto'),
            'anomalyDetector_basic_c01': curr(anomalyDetector_basicwalk)('c01'),
            'anomalyDetector_basic_cauto_t100': curr(anomalyDetector_basicwalk)('cauto_t100'),
            'anomalyDetector_basic_cauto_t300': curr(anomalyDetector_basicwalk)('cauto_t300'),
            'anomalyDetector_basic_c01_t100': curr(anomalyDetector_basicwalk)('c01_t100'),
            'anomalyDetector_basic_c01_t300': curr(anomalyDetector_basicwalk)('c01_t300'),
            'anomalyDetector_oneClassSVMnu005': curr(anomalyDetector_oneClassSVMnu005),
            'anomalyDetector_oneClassSVMnu02': curr(anomalyDetector_oneClassSVMnu02),
            'anomalyDetector_basicT300': curr(anomalyDetector_basicT300),
            'anomalyDetector_basicC01': curr(anomalyDetector_basicC01),
            # === Additional vectorSpace algorithm (horizontal variety) ===
            'vectorSpace_sklearn_tfidf': curr(vectorSpace_sklearn_tfidf),
            'vectorSpace_sklearn_lsa_100': curr(vectorSpace_sklearn_lsa_100),
            # === Top-K nearest-neighbor lookup (renamed from searchEngine_topK) ===
            'nearestNeighbors_topK': curr(nearestNeighbors_topK),
            'test_nearestNeighbors_relevance': curr(test_nearestNeighbors_relevance),
            # === External-ground-truth MI tests (2026-05-21) ===
            #     Paired with corpus loaders below; require label .npy sidecars.
            'test_clusterer_iraMutualInfo': curr(test_clusterer_iraMutualInfo),
            'test_clusterer_ideologyAMI': curr(test_clusterer_ideologyAMI),
            'data_freetext_iraVsControl': curries['data_freetext_csvColumn']('/data/iraVsControl.csv'),
            'test_vectorSpace_dedupRecovery': curr(test_vectorSpace_dedupRecovery),
            'data_freetext_movieSentimentDedup': curr(data_freetext_dedupPlanted)('/data/movieSentiment.csv'),
            'data_freetext_politicalAdsDedup': curr(data_freetext_dedupPlanted)('/data/opinion_ads.csv'),
            'data_freetext_congressionalBillsDedup': curr(data_freetext_dedupPlanted)('/data/billsByCongress.csv'),
            'data_freetext_congressionalBills': curries['data_freetext_csvColumn']('/data/billsByCongress.csv'),
            'data_freetext_politicalTweets': curries['data_freetext_csvColumn']('/data/pol_tweets.csv'),
            'data_freetext_politicalTweetsShort': curries['data_freetext_csvColumn']('/data/pol_tweets_short.csv'),
            'data_freetext_internetResearchAgencyFacebook': curries['data_freetext_csvColumn'](
                '/data/facebook.csv'),
            'data_freetext_trumpTweets': curries['data_freetext_csvColumn']('/data/individual_tweets.csv'),
            'data_freetext_politicalAds': curries['data_freetext_csvColumn']('/data/opinion_ads.csv'),
            'data_freetext_BSdetector': curries['data_freetext_csvColumn']('/data/fake.csv'),
            'data_freetext_internetResearchAgencyTweets': curries['data_freetext_csvColumn'](
                '/data/tweets.csv'), 'data_freetext_short': curries['data_freetext_csvColumn']('/data/short.csv'),
            'clusterer_sklearn_kmeans_5clusters': curries['clusterer_sklearn_kmeans'](n_clusters=5),
            'clusterer_sklearn_kmeans_10clusters': curries['clusterer_sklearn_kmeans'](n_clusters=10),
            'clusterer_sklearn_kmeans_20clusters': curries['clusterer_sklearn_kmeans'](n_clusters=20),
            # Agglomerative + Ward take BOTH n_clusters and n_neighbors
            # (for the kneighbors connectivity graph). 3x3 cartesian =
            # 9 variants each. Naming: <algo>_<Nclusters>clusters_nn<K>.
            'clusterer_sklearn_agglomerative_5clusters_nn5':   curries['clusterer_sklearn_agglomerative'](n_clusters=5,  n_neighbors=5),
            'clusterer_sklearn_agglomerative_5clusters_nn10':  curries['clusterer_sklearn_agglomerative'](n_clusters=5,  n_neighbors=10),
            'clusterer_sklearn_agglomerative_5clusters_nn20':  curries['clusterer_sklearn_agglomerative'](n_clusters=5,  n_neighbors=20),
            'clusterer_sklearn_agglomerative_10clusters_nn5':  curries['clusterer_sklearn_agglomerative'](n_clusters=10, n_neighbors=5),
            'clusterer_sklearn_agglomerative_10clusters_nn10': curries['clusterer_sklearn_agglomerative'](n_clusters=10, n_neighbors=10),
            'clusterer_sklearn_agglomerative_10clusters_nn20': curries['clusterer_sklearn_agglomerative'](n_clusters=10, n_neighbors=20),
            'clusterer_sklearn_agglomerative_20clusters_nn5':  curries['clusterer_sklearn_agglomerative'](n_clusters=20, n_neighbors=5),
            'clusterer_sklearn_agglomerative_20clusters_nn10': curries['clusterer_sklearn_agglomerative'](n_clusters=20, n_neighbors=10),
            'clusterer_sklearn_agglomerative_20clusters_nn20': curries['clusterer_sklearn_agglomerative'](n_clusters=20, n_neighbors=20),
            # AffinityPropagation does NOT take n_clusters (count is
            # data-derived). Parameter is damping ∈ [0.5, 1.0).
            'clusterer_sklearn_affinityPropagation_damping05': curries['clusterer_sklearn_affinityPropagation'](damping=0.5),
            'clusterer_sklearn_affinityPropagation_damping07': curries['clusterer_sklearn_affinityPropagation'](damping=0.7),
            'clusterer_sklearn_affinityPropagation_damping09': curries['clusterer_sklearn_affinityPropagation'](damping=0.9),
            # MeanShift does NOT take n_clusters (count is data-derived
            # from bandwidth, which is estimated from quantile).
            'clusterer_sklearn_meanShift_quantile01': curries['clusterer_sklearn_meanShift'](quantile=0.1),
            'clusterer_sklearn_meanShift_quantile02': curries['clusterer_sklearn_meanShift'](quantile=0.2),
            'clusterer_sklearn_meanShift_quantile03': curries['clusterer_sklearn_meanShift'](quantile=0.3),
            # Spectral does take n_clusters — unchanged from before.
            'clusterer_sklearn_spectral_5clusters': curries['clusterer_sklearn_spectral'](n_clusters=5),
            'clusterer_sklearn_spectral_10clusters': curries['clusterer_sklearn_spectral'](n_clusters=10),
            'clusterer_sklearn_spectral_20clusters': curries['clusterer_sklearn_spectral'](n_clusters=20),
            # Ward — same 3x3 cartesian structure as agglomerative.
            'clusterer_sklearn_ward_5clusters_nn5':   curries['clusterer_sklearn_ward'](n_clusters=5,  n_neighbors=5),
            'clusterer_sklearn_ward_5clusters_nn10':  curries['clusterer_sklearn_ward'](n_clusters=5,  n_neighbors=10),
            'clusterer_sklearn_ward_5clusters_nn20':  curries['clusterer_sklearn_ward'](n_clusters=5,  n_neighbors=20),
            'clusterer_sklearn_ward_10clusters_nn5':  curries['clusterer_sklearn_ward'](n_clusters=10, n_neighbors=5),
            'clusterer_sklearn_ward_10clusters_nn10': curries['clusterer_sklearn_ward'](n_clusters=10, n_neighbors=10),
            'clusterer_sklearn_ward_10clusters_nn20': curries['clusterer_sklearn_ward'](n_clusters=10, n_neighbors=20),
            'clusterer_sklearn_ward_20clusters_nn5':  curries['clusterer_sklearn_ward'](n_clusters=20, n_neighbors=5),
            'clusterer_sklearn_ward_20clusters_nn10': curries['clusterer_sklearn_ward'](n_clusters=20, n_neighbors=10),
            'clusterer_sklearn_ward_20clusters_nn20': curries['clusterer_sklearn_ward'](n_clusters=20, n_neighbors=20),
            # DBSCAN does NOT take n_clusters (count is data-derived
            # from eps + min_samples + density). Parameter is eps;
            # min_samples fixed at 5 default.
            'clusterer_sklearn_dbscan_eps01': curries['clusterer_sklearn_dbscan'](eps=0.1),
            'clusterer_sklearn_dbscan_eps03': curries['clusterer_sklearn_dbscan'](eps=0.3),
            'clusterer_sklearn_dbscan_eps05': curries['clusterer_sklearn_dbscan'](eps=0.5),
            'clusterer_sklearn_dbscan_eps10': curries['clusterer_sklearn_dbscan'](eps=1.0),
            'clusterer_sklearn_birch_5clusters': curries['clusterer_sklearn_birch'](n_clusters=5),
            'clusterer_sklearn_birch_10clusters': curries['clusterer_sklearn_birch'](n_clusters=10),
            'clusterer_sklearn_birch_20clusters': curries['clusterer_sklearn_birch'](n_clusters=20),
            'clusterer_sklearn_gaussian_5clusters': curries['clusterer_sklearn_gaussian'](n_clusters=5),
            'clusterer_sklearn_gaussian_10clusters': curries['clusterer_sklearn_gaussian'](n_clusters=10),
            'clusterer_sklearn_gaussian_20clusters': curries['clusterer_sklearn_gaussian'](n_clusters=20),
            'clusterer_nltk_kmeans_5clusters': curries['clusterer_nltk_kmeans'](n_clusters=5),
            'clusterer_nltk_kmeans_10clusters': curries['clusterer_nltk_kmeans'](n_clusters=10),
            'clusterer_nltk_kmeans_20clusters': curries['clusterer_nltk_kmeans'](n_clusters=20),
            'clusterer_nltk_agglomerative_5clusters': curries['clusterer_nltk_agglomerative'](n_clusters=5),
            'clusterer_nltk_agglomerative_10clusters': curries['clusterer_nltk_agglomerative'](n_clusters=10),
            'clusterer_nltk_agglomerative_20clusters': curries['clusterer_nltk_agglomerative'](n_clusters=20),
            # doc2vec ITERATIONS (epochs) ADJUSTMENT — May 14 2026.
            # Iterations label renaming (May 15 2026): keys updated to reflect
            # the actually-passed iterations arg (post-cap from earlier change):
            #   old key '_1000iterations_' -> new key '_100iterations_' (epochs=100)
            #   old key  '_200iterations_' -> new key  '_50iterations_' (epochs=50)
            #   '_20iterations_' unchanged (epochs=20)
            # Old key names stay in pre-rename blackboard logs / pickles —
            # audit-rendering scripts apply the same label mapping so reports
            # show truthful names regardless of when the run happened.
            #
            # Original values passed to gensim Doc2Vec(epochs=...) for the
            # three categorical tiers in the ontology:
            #   "_20iterations_"   →  20    (KEPT — already in the reasonable range)
            #   "_200iterations_"  →  200   →  50  (REDUCED)
            #   "_1000iterations_" →  1000  →  100 (REDUCED)
            #
            # Why: doc2vec literature on short-text / tweet corpora
            # consistently shows that 10–50 epochs is sufficient for
            # representation quality. Gensim's own default in recent
            # versions is `epochs=10`. The original 1000-epoch setting
            # was both excessive for quality (epoch ~50 onwards mostly
            # overfits idiosyncrasies of small corpora) and a per-config
            # training cost that made cc1 effectively non-finishable —
            # on a 1.66M-tweet corpus, 1000 epochs × ~187k words/s ≈
            # ~200 hours per single doc2vec model on this machine even
            # with the workers=4 patch (Registry.py:257 above). 100
            # epochs gives ~5% of that cost while preserving the
            # representation quality reported in PV-DBOW papers.
            #
            # Category NAMES (the strings) are kept as-is so existing
            # scenario JSONs don't break — only the iterations arg
            # passed to gensim is updated. Distinct tiers (20 / 50 /
            # 100) preserved so CMA-ES still has 3 compute-cost
            # choices to explore. size (50/100/200/300) and minfreq
            # (2/5) are unchanged — those values are already
            # standard for doc2vec on tweet-length text.
            'vectorSpace_gensim_doc2vec_50size_20iterations_2minFreq': curries['vectorSpace_gensim_doc2vec'](size=50)(
                iterations=20)(minfreq=2),
            'vectorSpace_gensim_doc2vec_50size_20iterations_5minFreq': curries['vectorSpace_gensim_doc2vec'](size=50)(
                iterations=20)(minfreq=5),
            'vectorSpace_gensim_doc2vec_50size_50iterations_2minFreq': curries['vectorSpace_gensim_doc2vec'](size=50)(
                iterations=50)(minfreq=2),  # was iterations=200
            'vectorSpace_gensim_doc2vec_50size_50iterations_5minFreq': curries['vectorSpace_gensim_doc2vec'](size=50)(
                iterations=50)(minfreq=5),  # was iterations=200
            'vectorSpace_gensim_doc2vec_50size_100iterations_2minFreq': curries['vectorSpace_gensim_doc2vec'](size=50)(
                iterations=100)(minfreq=2),  # was iterations=1000
            'vectorSpace_gensim_doc2vec_50size_100iterations_5minFreq': curries['vectorSpace_gensim_doc2vec'](size=50)(
                iterations=100)(minfreq=5),  # was iterations=1000
            'vectorSpace_gensim_doc2vec_100size_20iterations_2minFreq': curries['vectorSpace_gensim_doc2vec'](size=100)(
                iterations=20)(minfreq=2),
            'vectorSpace_gensim_doc2vec_100size_20iterations_5minFreq': curries['vectorSpace_gensim_doc2vec'](size=100)(
                iterations=20)(minfreq=5),
            'vectorSpace_gensim_doc2vec_100size_50iterations_2minFreq': curries['vectorSpace_gensim_doc2vec'](
                size=100)(iterations=50)(minfreq=2),  # was iterations=200
            'vectorSpace_gensim_doc2vec_100size_50iterations_5minFreq': curries['vectorSpace_gensim_doc2vec'](
                size=100)(iterations=50)(minfreq=5),  # was iterations=200
            'vectorSpace_gensim_doc2vec_100size_100iterations_2minFreq': curries['vectorSpace_gensim_doc2vec'](
                size=100)(iterations=100)(minfreq=2),  # was iterations=1000
            'vectorSpace_gensim_doc2vec_100size_100iterations_5minFreq': curries['vectorSpace_gensim_doc2vec'](
                size=100)(iterations=100)(minfreq=5),  # was iterations=1000
            'vectorSpace_gensim_doc2vec_200size_20iterations_2minFreq': curries['vectorSpace_gensim_doc2vec'](size=200)(
                iterations=20)(minfreq=2),
            'vectorSpace_gensim_doc2vec_200size_20iterations_5minFreq': curries['vectorSpace_gensim_doc2vec'](size=200)(
                iterations=20)(minfreq=5),
            'vectorSpace_gensim_doc2vec_200size_50iterations_2minFreq': curries['vectorSpace_gensim_doc2vec'](
                size=200)(iterations=50)(minfreq=2),  # was iterations=200
            'vectorSpace_gensim_doc2vec_200size_50iterations_5minFreq': curries['vectorSpace_gensim_doc2vec'](
                size=200)(iterations=50)(minfreq=5),  # was iterations=200
            'vectorSpace_gensim_doc2vec_200size_100iterations_2minFreq': curries['vectorSpace_gensim_doc2vec'](
                size=200)(iterations=100)(minfreq=2),  # was iterations=1000
            'vectorSpace_gensim_doc2vec_200size_100iterations_5minFreq': curries['vectorSpace_gensim_doc2vec'](
                size=200)(iterations=100)(minfreq=5),  # was iterations=1000
            'vectorSpace_gensim_doc2vec_300size_20iterations_2minFreq': curries['vectorSpace_gensim_doc2vec'](size=300)(
                iterations=20)(minfreq=2),
            'vectorSpace_gensim_doc2vec_300size_20iterations_5minFreq': curries['vectorSpace_gensim_doc2vec'](size=300)(
                iterations=20)(minfreq=5),
            'vectorSpace_gensim_doc2vec_300size_50iterations_2minFreq': curries['vectorSpace_gensim_doc2vec'](
                size=300)(iterations=50)(minfreq=2),  # was iterations=200
            'vectorSpace_gensim_doc2vec_300size_50iterations_5minFreq': curries['vectorSpace_gensim_doc2vec'](
                size=300)(iterations=50)(minfreq=5),  # was iterations=200
            'vectorSpace_gensim_doc2vec_300size_100iterations_2minFreq': curries['vectorSpace_gensim_doc2vec'](
                size=300)(iterations=100)(minfreq=2),  # was iterations=1000
            'vectorSpace_gensim_doc2vec_300size_100iterations_5minFreq': curries['vectorSpace_gensim_doc2vec'](
                size=300)(iterations=100)(minfreq=5),  # was iterations=1000
            # === Added 2026-07-04 (see /tmp/registry_additions_evidence.md) ===
            # Repaired classifier (multi_class arg dropped; failure => all -1,
            # never constant-0) + its held-out-minus-base-rate test.
            'classifier_logistic_v2': curr(classifier_logistic_v2),
            'test_classifier_heldout': curr(test_classifier_heldout),
            # Honest post-hoc gate for the real anomaly detectors: rank
            # correlation of detector scores with robust distance-from-center.
            'test_anomalyDetector_centerRank': curr(test_anomalyDetector_centerRank),
            # 2026-09-10 her order: runnable-software tests for the two part families
            'test_labeler_runnable': curr(test_labeler_runnable),
    'test_labeler_downstreamF1': curr(test_labeler_downstreamF1),
            'test_preprocessor_runnable': curr(test_preprocessor_runnable),
            'test_anomalyDetector_separation': curr(test_anomalyDetector_separation),
            # First two-input primitive — unlocks tree-shaped pipelines
            # (arity 2; the ontology leaf must declare two _args entries).
            'vectorSpace_concat': curr(vectorSpace_concat),
            # === Added 2026-07-04 batch 2 (see /tmp/registry_batch2_evidence.md) ===
            # Style-sensitive char-ngram embedding (distinct family from the
            # topic-sensitive doc2vec/word-tfidf/lsa spaces).
            'vectorSpace_sklearn_charTfidf_100': curr(vectorSpace_sklearn_charTfidf_100),
            # Stopword removal that preserves row count + order (label-safe),
            # chains like preprocessor_freetext_strip.
            'preprocessor_freetext_stopwords': curr(preprocessor_freetext_stopwords),
            # Second two-input primitive — a clusterer combiner (co-association
            # consensus of two clusterings; arity 2, two _args entries).
            'clusterer_consensus': curr(clusterer_consensus),
            'classifier_billsIdeology_logistic': curr(classifier_billsIdeology_logistic),
            # REMOVED 2026-08-14, her ruling: "no prefabs please."
            # The two pre-assembled stepping-stones - a fixed 20-dimension
            # term-frequency embedding, and that embedding followed by
            # 20-cluster k-means - are no longer offered to agents. Their
            # automatic use inside the labeler was already withdrawn on
            # 19 July ("probes only"); this removes the parts themselves,
            # so a seller assembles its product from the ontology or has
            # nothing to sell. The functions stay defined below, unreachable
            # from the registry, until she rules on deleting them outright.
            # 'vectorSpace_prefab_basicLsa': curr(vectorSpace_prefab_basicLsa),
            # 'clusterer_prefab_basicPipeline': curr(clusterer_prefab_basicPipeline),
            # === Added 2026-07-16: shared-scenario port (see block above) ===
            # The LLM world's five-want corpora as data terminals
            # (data_freetext_csvColumn pattern, staged by
            # _port_prepare_data.py):
            'data_freetext_disasterTweets': curries['data_freetext_csvColumn']('/data/disasterTweets.csv'),
            'data_freetext_movieSentiment': curries['data_freetext_csvColumn']('/data/movieSentiment.csv'),
            'data_freetext_movieSentimentFewshot': curries['data_freetext_csvColumn']('/data/movieSentimentFewshot.csv'),
            # Ground-truth joiners (billsIdeology sidecar convention):
            'labeler_vector_blobs': curr(labeler_vector_blobs),
            'labeler_vector_varied': curr(labeler_vector_varied),
            'labeler_vector_ansio': curr(labeler_vector_ansio),
            'labeler_vector_variedBlobs5': curr(labeler_vector_variedBlobs5),
            'labeler_vector_noisyMoons': curr(labeler_vector_noisyMoons),
            'labeler_vector_circles': curr(labeler_vector_circles),
            'labeler_vector_moons': curr(labeler_vector_moons),
            'labeler_disasterTweets': curr(labeler_disasterTweets),
            'labeler_movieSentiment': curr(labeler_movieSentiment),
            'labeler_BSdetector': curr(labeler_BSdetector),
            'labeler_BSdetectorTypes': curr(labeler_BSdetectorTypes),
            'data_freetext_BSdetectorTypes': curries['data_freetext_csvColumn']('/data/fake.csv'),
            'labeler_movieSentimentFewshot': curr(labeler_movieSentimentFewshot),
            # Two more learner families (logistic_v2 harness):
            'classifier_svm_linear': curr(classifier_svm_linear),
            'classifier_naivebayes_multinomial': curr(classifier_naivebayes_multinomial),
            # Arity-2 ensemble combiner; the classifier_ensemble item path
            # is only satisfiable through this part:
            'classifier_ensemble_vote': curr(classifier_ensemble_vote),
            # The two ported want-scorers:
            'test_classifier_f1heldout': curr(test_classifier_f1heldout),
            'test_classifier_f1macroHeldout': curr(test_classifier_f1macroHeldout),
            'data_vector_noisyMoons': curr(data_vector_noisyMoons),
            'data_vector_variedBlobs5': curr(data_vector_variedBlobs5),
            'test_vectorSpace_probe': curr(test_vectorSpace_probe)}



# === Added 2026-07-18: the lazy big registry (the curator's design) =============
# A BIG catalog of tool names whose implementations are materialized LAZILY
# through ONE vetted reflective adapter. The vetted code is this block and
# nothing else; the catalog itself is data (simulation/lazy_whitelist.json,
# class path -> contract kind), built and verified mechanically by
# scripts/build_lazy_registry.py — every shipped class was instantiated with
# DEFAULT parameters and run end-to-end through this adapter on a text
# sample before it was allowed into the file, and anything that failed was
# dropped. importlib is driven only by that whitelist; a class path outside
# it is refused before any import happens.
#
# Each whitelisted class runs inside one of the three existing harness
# patterns, so a lazy leaf behaves exactly like a hand-written part of its
# family and plugs into the same tests and the same wants:
#
#   classifier  -> the classifier_svm_linear harness: consume an upstream
#                  (X, Y) pair, divide rows with _port_or_seeded_split (the
#                  LLM world's fixed file split on port corpora, the seeded
#                  70/30 split otherwise), fit on the training side only,
#                  and return (X, Y, pred) with pred = -1 on training rows.
#                  Any failure yields all -1 — never a constant prediction
#                  that could bank the base rate.
#   clusterer   -> the clusterer_sklearn_* harness: consume an embedding,
#                  return the (X, labels) tuple every clusterer test reads.
#   transformer -> the vectorSpace harness: consume documents (or an
#                  upstream embedding) and return a float32 matrix. Classes
#                  whose sklearn input tags declare string input are fed the
#                  raw documents; every other class is fed the bounded
#                  dense tfidf bridge (_text_bridge_features() wide — the
#                  vectorSpace_sklearn_tfidf convention).
#
# Rails, following the dbscan / oneClassSVM precedents: inputs and outputs
# wider than _LAZY_MAX_WIDTH columns are refused; rows are HEAD-truncated to
# MARKETSIM_VECTORSPACE_MAX_ROWS so ground-truth sidecars stay aligned; where
# a class exposes the knobs, random_state is pinned (42) and n_jobs is bound
# to MARKETSIM_SKLEARN_NJOBS like every other part. A class that raises
# anywhere — import, construction, fit, predict — returns None: an
# unevaluable product that scores 0 in the market, never a crash. One line
# goes to stdout on failure so run logs stay diagnosable.

_LAZY_MAX_WIDTH = 2000
_LAZY_WHITELIST = None

# Convergence floor (2026-08-10 tree restoration): the only settings
# applied at BASE instantiation, before any parameter-level kwargs.
# These are not quality knobs and are not searchable - they exist so
# every walk of every class actually converges within the evaluation
# ceiling instead of failing on a low library iteration default. The
# quality-relevant sensible values from the retired _LAZY_SENSIBLE
# table went into _LAZY_PARAM_LEVELS as ordered level values (her
# order: use the table for the restored tree build, then let it go).
_LAZY_CONVERGENCE_FLOOR = {
    'sklearn.decomposition.NMF': {'max_iter': 500},
    'sklearn.decomposition.MiniBatchNMF': {'max_iter': 500},
    'sklearn.decomposition.FastICA': {'max_iter': 500},
    'sklearn.linear_model.LogisticRegression': {'max_iter': 2000},
    'sklearn.linear_model.SGDClassifier': {'max_iter': 2000},
    'sklearn.linear_model.PassiveAggressiveClassifier': {'max_iter': 2000},
    'sklearn.svm.LinearSVC': {'max_iter': 5000},
    'sklearn.neural_network.MLPClassifier': {'max_iter': 500, 'early_stopping': True},
    'sklearn.cluster.AffinityPropagation': {'max_iter': 300},
}


def _lazy_whitelist():
    """Load simulation/lazy_whitelist.json once: {class path: kind} with
    kind in {'classifier', 'clusterer', 'transformer'}. A missing or
    unreadable file means an empty catalog, never an import error —
    checkouts without the whitelist simply have no lazy parts."""
    global _LAZY_WHITELIST
    if _LAZY_WHITELIST is None:
        import json as _json
        import os as _os
        path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                             'lazy_whitelist.json')
        try:
            with open(path) as handle:
                _LAZY_WHITELIST = {str(k): str(v)
                                   for k, v in _json.load(handle).items()}
        except Exception:
            _LAZY_WHITELIST = {}
    return _LAZY_WHITELIST


def _lazy_contract(est):
    """Mechanical contract detection from the instance's methods and sklearn
    base classes; never from the class name. Order matters: classifiers are
    tested first, then clusterers (KMeans-style clusterers also expose
    fit_transform, so the transformer test must come last). Regressors and
    outlier detectors (fit_predict plus predict) match none of the three
    contracts and return None."""
    try:
        from sklearn.base import is_classifier
        if is_classifier(est) and hasattr(est, 'fit') and hasattr(est, 'predict'):
            return 'classifier'
    except Exception:
        pass
    clusterish = False
    try:
        from sklearn.base import ClusterMixin
        clusterish = isinstance(est, ClusterMixin)
    except Exception:
        pass
    if not clusterish:
        clusterish = (hasattr(est, 'fit_predict')
                      and not hasattr(est, 'predict')
                      and not hasattr(est, 'transform'))
    if clusterish and hasattr(est, 'fit'):
        return 'clusterer'
    if hasattr(est, 'fit_transform') and not hasattr(est, 'predict'):
        return 'transformer'
    return None


def _lazy_instantiate(class_path):
    """Instantiate a whitelisted class path with DEFAULT parameters via
    importlib. Refuses any path not in the whitelist. Where the class
    exposes the standard knobs, random_state is pinned to 42 (the harness
    convention, so replay stays deterministic) and n_jobs is bound to
    MARKETSIM_SKLEARN_NJOBS (default 4, like the other sklearn parts); no
    other parameter is ever set."""
    import importlib
    import os as _os
    kind = _lazy_whitelist().get(class_path)
    if kind not in ('classifier', 'clusterer', 'transformer'):
        raise ValueError('estimator_lazy refused: %r is not whitelisted'
                         % (class_path,))
    module_path, _, cls_name = class_path.rpartition('.')
    est = getattr(importlib.import_module(module_path), cls_name)()
    if hasattr(est, 'get_params') and hasattr(est, 'set_params'):
        try:
            params = est.get_params(deep=False)
        except Exception:
            params = {}
        updates = {}
        # Her ruling (2026-08-12): PRODUCT estimators are not pinned
        # to a constant - left unset, their randomness resolves
        # through the universal evaluation sandbox, deterministic
        # from (chain, scenario seed). Test-instrument pins
        # elsewhere in this file stay fixed: instruments must not
        # vary by seed or scores stop being comparable.
        if 'n_jobs' in params:
            updates['n_jobs'] = int(_os.environ.get('MARKETSIM_SKLEARN_NJOBS', '4'))
        # HER ORDER (2026-08-12): bound the work, as a precaution.
        # sklearn's kernel SVMs default to max_iter = -1 - the solver may
        # iterate without limit. Instance A spent more than fifty-five
        # minutes inside ONE classifier_sklearn_SVC_creg10_lin fit, and
        # the 300-second evaluation cap could not stop it, because that
        # cap is a Python signal and the solver never returns to the
        # interpreter to receive it.
        #
        # A finite iteration ceiling bounds the work identically on every
        # machine. That matters more than the time saved: a clock-based
        # verdict would record "this chain is broken" because of the box
        # that happened to reach it first, which is exactly what the
        # chain-keyed cache and the chain-seeded sandbox were changed to
        # prevent today. An iteration ceiling is a property of the chain.
        # An unconverged fit is not an error - it is a worse model, and
        # the market is entitled to judge it.
        if 'max_iter' in params and params.get('max_iter') in (-1, None):
            updates['max_iter'] = int(
                _os.environ.get('MARKETSIM_SOLVER_MAX_ITER', '100000'))
        floor = _LAZY_CONVERGENCE_FLOOR.get(class_path, {})
        for k, v in floor.items():
            if k in params:
                updates[k] = v
        if updates:
            est.set_params(**updates)
    return est, kind


def _lazy_row_cap():
    import os as _os
    return int(_os.environ.get('MARKETSIM_VECTORSPACE_MAX_ROWS', '5000'))


def _lazy_classifier_run(est, XY):
    # The classifier_svm_linear harness with the learner materialized
    # lazily. Consumes an upstream (X, Y) pair; returns (X, Y, pred) with
    # pred = -1 on training rows so the classifier tests score held-out
    # rows only; any fit or predict failure yields all -1.
    import numpy as np
    if XY is None or not isinstance(XY, (tuple, list)) or len(XY) < 2:
        return None
    X, Y = XY[0], XY[1]
    if X is None or Y is None:
        return None
    X = np.asarray(X)
    Y = np.asarray(Y)
    if X.ndim != 2 or Y.ndim != 1 or len(X) != len(Y):
        return None
    if X.shape[1] > _LAZY_MAX_WIDTH:
        raise ValueError('estimator_lazy refused: input width %d > %d'
                         % (X.shape[1], _LAZY_MAX_WIDTH))
    n = min(len(X), _lazy_row_cap())
    X = np.asarray(X[:n], dtype=np.float32)
    Y = np.asarray(Y[:n]).astype(np.int32)
    if n < 10 or len(np.unique(Y)) < 2:
        return (X, Y, np.full(n, -1, dtype=np.int32))
    pred = np.full(n, -1, dtype=np.int32)
    try:
        idx_train, idx_test = _port_or_seeded_split(Y, n)
        if idx_train is not None:
            est.fit(X[idx_train], Y[idx_train])
            pred[idx_test] = np.asarray(
                est.predict(X[idx_test])).ravel().astype(np.int32)
    except Exception:
        pred = np.full(n, -1, dtype=np.int32)
    return (X, Y, pred)


def _lazy_clusterer_run(est, X):
    # The clusterer_sklearn_* harness. Consumes an embedding matrix and
    # returns (X, labels). A label vector whose length differs from the row
    # count (FeatureAgglomeration-style column clustering) is refused —
    # those labels would silently misalign every ground-truth test.
    import numpy as np
    if X is None or isinstance(X, (tuple, list)):
        return None
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[0] < 8:
        return None
    if X.shape[1] > _LAZY_MAX_WIDTH:
        raise ValueError('estimator_lazy refused: input width %d > %d'
                         % (X.shape[1], _LAZY_MAX_WIDTH))
    X = np.asarray(X[:_lazy_row_cap()], dtype=np.float32)
    if hasattr(est, 'fit_predict'):
        Y = est.fit_predict(X)
    else:
        est.fit(X)
        Y = est.labels_
    Y = np.asarray(Y).ravel()
    if len(Y) != len(X):
        raise ValueError('estimator_lazy refused: %d labels for %d rows'
                         % (len(Y), len(X)))
    return (X, Y.astype(np.int32))


def _lazy_clusterfeatures_run(est, X):
    # Distance-to-centroid features: consumes an embedding matrix (never
    # text), fits the clusterer, returns transform(X) — one column per
    # discovered cluster. Downstream stages consume it exactly like any
    # embedding, so clustering quality propagates into detector and
    # classifier scores.
    import numpy as np
    if X is None or isinstance(X, (tuple, list)):
        return None
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[0] < 8:
        return None
    if X.shape[1] > _LAZY_MAX_WIDTH:
        raise ValueError('cluster_features refused: input width %d > %d'
                         % (X.shape[1], _LAZY_MAX_WIDTH))
    X = np.asarray(X[:_lazy_row_cap()], dtype=np.float32)
    if not hasattr(est, 'transform'):
        return None
    est.fit(X)
    E = est.transform(X)
    E = np.asarray(E, dtype=np.float32)
    if E.ndim != 2 or len(E) != len(X) or not np.all(np.isfinite(E)):
        return None
    return E


def _lazy_vectorspace_run(est, X):
    # The vectorSpace harness. Text input goes to the class directly when
    # its sklearn input tags declare string input; otherwise text is turned
    # into the bounded dense tfidf bridge first (the vectorSpace_sklearn_tfidf
    # convention: _text_bridge_features() wide, min_df=2). An upstream embedding matrix
    # is consumed as-is, so transformer-over-transformer chains compose.
    # Output must be a finite 2-D float matrix with one row per input row
    # and at most _LAZY_MAX_WIDTH columns.
    import numpy as np
    if X is None or isinstance(X, tuple):
        return None
    docs = None
    M = None
    if isinstance(X, np.ndarray):
        if X.ndim != 2 or X.shape[0] < 2:
            return None
        if X.shape[1] > _LAZY_MAX_WIDTH:
            raise ValueError('estimator_lazy refused: input width %d > %d'
                             % (X.shape[1], _LAZY_MAX_WIDTH))
        M = np.asarray(X[:_lazy_row_cap()], dtype=np.float32)
    elif isinstance(X, list):
        if len(X) < 2:
            return None
        if not all(isinstance(d, str) or hasattr(d, 'words') for d in X[:5]):
            return None
        docs = [' '.join(d.words) if hasattr(d, 'words') else str(d)
                for d in X[:_lazy_row_cap()]]
    else:
        return None
    wants_text = False
    try:
        from sklearn.utils import get_tags
        wants_text = bool(get_tags(est).input_tags.string)
    except Exception:
        wants_text = False
    if wants_text and docs is None:
        # A text-native vectorizer offered a numeric matrix: not its input.
        return None
    if docs is not None and not wants_text:
        # THE TEXT BRIDGE IS GONE (her order 2026-10-01: "put it back to
        # april 4, do 2"). A numeric transformer handed text is not its
        # input and fails, as it should; text is vectorized only by a
        # vectorizer LEAF the seller chose and tuned (gensim doc2vec as on
        # April 4, or the Tfidf/Hashing vectorizer leaves below).
        return None
    n_in = len(docs) if wants_text else len(M)
    E = est.fit_transform(docs if wants_text else M)
    if hasattr(E, 'toarray'):
        if E.shape[1] > _LAZY_MAX_WIDTH:
            raise ValueError('estimator_lazy refused: output width %d > %d'
                             % (E.shape[1], _LAZY_MAX_WIDTH))
        E = E.toarray()
    E = np.asarray(E, dtype=np.float32)
    if E.ndim != 2 or E.shape[0] != n_in or E.shape[1] < 1:
        return None
    if E.shape[1] > _LAZY_MAX_WIDTH:
        raise ValueError('estimator_lazy refused: output width %d > %d'
                         % (E.shape[1], _LAZY_MAX_WIDTH))
    if not np.isfinite(E).all():
        return None
    return E




# The flat settings-variant menus (_LAZY_VARIANTS, 2026-08-10 morning)
# are retired: her tree-restoration ruling replaces suffix-bundle
# leaves with parameters as ordered TREE LEVELS (_LAZY_PARAM_LEVELS
# above). Their vetted values live on as level values.


def estimator_lazy(class_path, X):
    # THE lazy-materialization adapter — the one vetted implementation
    # behind every leaf of the big catalog. Takes a fully qualified class
    # path (whitelist-checked), instantiates it with default parameters,
    # and runs it through the harness its contract kind selects. Failure
    # of any sort is a market failure (None -> score 0, no settle), never
    # a crash: exceptions from arbitrary third-party code are caught here
    # because memoise_pickle re-raises exception types outside its list.
    # _args and _return follow the family the leaf sits in (see the
    # harness helpers above).
    print('estimator_lazy ' + str(class_path))
    try:
        est, kind = _lazy_instantiate(class_path.split('#')[0]) \
            if '#' in str(class_path) else _lazy_instantiate(class_path)
        if '#p:' in str(class_path):
            base, _, route = str(class_path).partition('#p:')
            kw = _lazy_param_kwargs(base, route)
            if hasattr(est, 'get_params') and kw:
                try:
                    params = est.get_params(deep=False)
                    est.set_params(**{k: v for k, v in kw.items()
                                      if k in params})
                except Exception:
                    pass
        if '#features' in str(class_path):
            # Cluster-distance features (the curator 2026-07-19: clusterers need
            # a USE — components inside detectors and other programs, so
            # better clustering propagates downstream): fit the clusterer,
            # emit transform()'s distance-to-centroid matrix as an
            # embedding-stage output. Reuse across applications is the
            # selective pressure for specialization.
            return _lazy_clusterfeatures_run(est, X)
        if kind == 'classifier':
            return _lazy_classifier_run(est, X)
        if kind == 'clusterer':
            return _lazy_clusterer_run(est, X)
        return _lazy_vectorspace_run(est, X)
    except Exception as e:
        print('estimator_lazy FAILED %s: %s: %s'
              % (class_path, type(e).__name__, e))
        return None




# Per-parameter settings walk (the curator 2026-07-19: "when an agent puts
# something like brandname_routine_stop, it doesnt fill parameters in
# and thus goes to their defaults, where brandname_routine_param1value_stop
# fills in only param1 but the rest go to their defaults"). Each class
# gets ORDERED parameter levels; the genome walks them level by level,
# and a stop at any depth leaves every unwalked parameter at its
# library default. Tokens are underscore-free (the ontology path
# notation splits on '_').
_LAZY_PARAM_LEVELS = {
    # THE parameter-level table (the curator 2026-08-10, tree restoration
    # order): parameters are TREE LEVELS in the April 4 shape. Three
    # of her laws govern this table:
    #   LIKE ORDER - within a family, the same KIND of parameter sits
    #     at the same level (clusterers: level 1 is the grouping-scale
    #     knob; transformers: level 1 is dimensionality/primary;
    #     classifiers: level 1 is the dominant capacity/regularization
    #     knob) so an item mutation between algorithms leaves the
    #     deeper parameter floats meaning the analogous thing.
    #   IMPORTANCE ORDER - a stop mid-walk defaults the remaining
    #     levels, so later levels hold what matters least to get right.
    #   SENSIBLE VALUES - each level lists ordered practitioner values
    #     with the library default among them (marked *below* in the
    #     token by convention where visible); a full stop at the class
    #     node is all-defaults via the prefix registration.
    # A level is (param_name, {token: value}) where value may also be
    # a dict of coupled kwargs when one decision sets several knobs.
    # --- clusterers: level 1 grouping scale, level 2 secondary ---
    'sklearn.cluster.KMeans': [
        ('n_clusters', {'k2': 2, 'k3': 3, 'k5': 5, 'k8': 8, 'k20': 20}),
        ('n_init', {'ninit3': 3, 'ninit10': 10}),
    ],
    'sklearn.cluster.MiniBatchKMeans': [
        ('n_clusters', {'k2': 2, 'k3': 3, 'k5': 5, 'k8': 8, 'k20': 20}),
        ('n_init', {'ninit3': 3, 'ninit10': 10}),
    ],
    'sklearn.cluster.BisectingKMeans': [
        ('n_clusters', {'k2': 2, 'k3': 3, 'k5': 5, 'k8': 8, 'k20': 20}),
    ],
    'sklearn.cluster.AgglomerativeClustering': [
        ('n_clusters', {'k2': 2, 'k3': 3, 'k5': 5, 'k8': 8, 'k20': 20}),
        ('linkage', {'ward': 'ward', 'complete': 'complete', 'average': 'average'}),
    ],
    'sklearn.cluster.Birch': [
        ('n_clusters', {'k2': 2, 'k3': 3, 'k5': 5, 'k8': 8, 'k20': 20}),
        ('threshold', {'thr03': 0.3, 'thr05': 0.5, 'thr07': 0.7}),
    ],
    'sklearn.cluster.DBSCAN': [
        ('eps', {'eps02': 0.2, 'eps05': 0.5, 'eps1': 1.0, 'eps2': 2.0}),
        ('min_samples', {'min5': 5, 'min10': 10, 'min25': 25}),
    ],
    'sklearn.cluster.OPTICS': [
        ('min_samples', {'min5': 5, 'min10': 10, 'min25': 25}),
        ('max_eps', {'reach05': 0.5, 'reach2': 2.0}),
    ],
    'sklearn.cluster.HDBSCAN': [
        ('min_cluster_size', {'size5': 5, 'size10': 10, 'size25': 25}),
        ('min_samples', {'min3': 3, 'min10': 10}),
    ],
    'hdbscan.HDBSCAN': [
        ('min_cluster_size', {'size5': 5, 'size10': 10, 'size25': 25}),
        ('min_samples', {'min3': 3, 'min10': 10}),
    ],
    # hdbscan.RobustSingleLinkage: no levels - its cut variants failed
    # end-to-end vetting on 2026-08-10; the bare leaf stays.
    # --- transformers: level 1 dimensionality/primary, level 2 kernel/secondary ---
    'sklearn.decomposition.TruncatedSVD': [
        ('n_components', {'c2': 2, 'c20': 20, 'c50': 50, 'c100': 100}),
    ],
    'sklearn.decomposition.PCA': [
        ('n_components', {'c20': 20, 'c50': 50, 'c100': 100}),
        ('whiten', {'plain': False, 'whiten': True}),
    ],
    'sklearn.decomposition.IncrementalPCA': [
        ('n_components', {'c20': 20, 'c50': 50, 'c100': 100}),
    ],
    'sklearn.decomposition.KernelPCA': [
        ('n_components', {'c20': 20, 'c50': 50, 'c100': 100}),
        ('kernel', {'lin': 'linear', 'rbf': 'rbf', 'cos': 'cosine'}),
    ],
    'sklearn.decomposition.FactorAnalysis': [
        ('n_components', {'c20': 20, 'c50': 50}),
    ],
    'sklearn.decomposition.FastICA': [
        ('n_components', {'c20': 20, 'c50': 50}),
    ],
    'sklearn.decomposition.NMF': [
        ('n_components', {'c20': 20, 'c50': 50}),
    ],
    'sklearn.decomposition.MiniBatchNMF': [
        ('n_components', {'c20': 20, 'c50': 50}),
    ],
    'sklearn.decomposition.LatentDirichletAllocation': [
        ('n_components', {'t5': 5, 't10': 10, 't20': 20, 't50': 50}),
    ],
    'sklearn.kernel_approximation.Nystroem': [
        ('n_components', {'c50': 50, 'c100': 100, 'c300': 300}),
        ('gamma', {'g01': 0.1, 'g1': 1.0}),
    ],
    'sklearn.kernel_approximation.RBFSampler': [
        ('n_components', {'c50': 50, 'c100': 100, 'c300': 300}),
        ('gamma', {'g01': 0.1, 'g1': 1.0}),
    ],
    'sklearn.kernel_approximation.SkewedChi2Sampler': [
        ('n_components', {'c50': 50, 'c100': 100, 'c300': 300}),
    ],
    'sklearn.kernel_approximation.PolynomialCountSketch': [
        ('n_components', {'c50': 50, 'c100': 100, 'c300': 300}),
        ('degree', {'deg2': 2, 'deg3': 3}),
    ],
    'sklearn.kernel_approximation.AdditiveChi2Sampler': [
        ('sample_steps', {'s2': 2, 's3': 3}),
    ],
    'sklearn.neural_network.BernoulliRBM': [
        ('n_components', {'c64': 64, 'c128': 128, 'c256': 256}),
    ],
    'sklearn.preprocessing.KBinsDiscretizer': [
        ('n_bins', {'b3': 3, 'b5': 5}),
        ('encode', {'onehot': 'onehot', 'ordinal': 'ordinal'}),
    ],
    'sklearn.impute.KNNImputer': [
        ('n_neighbors', {'n5': 5, 'n10': 10}),
    ],
    'sklearn.neighbors.KNeighborsTransformer': [
        ('n_neighbors', {'n5': 5, 'n10': 10, 'n25': 25}),
    ],
    'sklearn.neighbors.RadiusNeighborsTransformer': [
        ('radius', {'r05': 0.5, 'r1': 1.0, 'r2': 2.0}),
    ],
    'sklearn.ensemble.RandomTreesEmbedding': [
        ('n_estimators', {'t50': 50, 't100': 100}),
        ('max_depth', {'d3': 3, 'd5': 5}),
    ],
    'sklearn.preprocessing.Normalizer': [
        ('norm', {'l1': 'l1', 'l2': 'l2', 'max': 'max'}),
    ],
    'sklearn.preprocessing.Binarizer': [
        ('threshold', {'t0': 0.0, 't01': 0.1, 't05': 0.5}),
    ],
    'sklearn.preprocessing.PowerTransformer': [
        ('standardize', {'std': True, 'nostd': False}),
    ],
    'sklearn.preprocessing.QuantileTransformer': [
        ('output_distribution', {'uniform': 'uniform', 'normal': 'normal'}),
        ('n_quantiles', {'q100': 100, 'q1000': 1000}),
    ],
    'sklearn.preprocessing.RobustScaler': [
        ('quantile_range', {'qr2575': (25.0, 75.0), 'qr1090': (10.0, 90.0)}),
    ],
    'sklearn.preprocessing.MinMaxScaler': [
        ('feature_range', {'r01': (0, 1), 'r11': (-1, 1)}),
    ],
    'sklearn.preprocessing.StandardScaler': [
        ('with_mean', {'center': True, 'nocenter': False}),
    ],
    'sklearn.feature_selection.VarianceThreshold': [
        ('threshold', {'v0': 0.0, 'vtiny': 1e-05}),
    ],
    'sklearn.impute.SimpleImputer': [
        ('strategy', {'mean': 'mean', 'median': 'median'}),
    ],
    'sklearn.feature_extraction.text.TfidfTransformer': [
        ('use_idf', {'idf': True, 'noidf': False}),
        ('sublinear_tf', {'lin': False, 'sub': True}),
    ],
    'umap.UMAP': [
        ('n_components', {'c2': 2, 'c10': 10}),
        ('n_neighbors', {'n5': 5, 'n15': 15, 'n50': 50}),
    ],
    # --- classifiers: level 1 dominant capacity/regularization ---
    'sklearn.linear_model.LogisticRegression': [
        ('C', {'creg01': 0.1, 'creg1': 1.0, 'creg10': 10.0}),
    ],
    'sklearn.svm.LinearSVC': [
        ('C', {'creg01': 0.1, 'creg1': 1.0, 'creg10': 10.0}),
    ],
    'sklearn.svm.SVC': [
        ('C', {'creg01': 0.1, 'creg1': 1.0, 'creg10': 10.0}),
        ('kernel', {'rbf': 'rbf', 'lin': 'linear'}),
    ],
    'sklearn.svm.NuSVC': [
        ('nu', {'nu025': 0.25, 'nu05': 0.5}),
        ('kernel', {'rbf': 'rbf', 'lin': 'linear'}),
    ],
    'sklearn.linear_model.SGDClassifier': [
        ('alpha', {'a4': 0.0001, 'a3': 0.001}),
        ('loss', {'hinge': 'hinge', 'log': 'log_loss'}),
    ],
    'sklearn.linear_model.Perceptron': [
        ('penalty', {'nopen': None, 'l2': 'l2'}),
    ],
    'sklearn.linear_model.PassiveAggressiveClassifier': [
        ('C', {'creg01': 0.1, 'creg1': 1.0}),
    ],
    'sklearn.linear_model.RidgeClassifier': [
        ('alpha', {'a01': 0.1, 'a1': 1.0, 'a10': 10.0}),
    ],
    'sklearn.tree.DecisionTreeClassifier': [
        ('max_depth', {'d3': 3, 'd10': 10, 'dfull': None}),
    ],
    'sklearn.tree.ExtraTreeClassifier': [
        ('max_depth', {'d3': 3, 'd10': 10, 'dfull': None}),
    ],
    'sklearn.ensemble.RandomForestClassifier': [
        ('n_estimators', {'t100': 100, 't300': 300}),
        ('max_depth', {'d10': 10, 'dfull': None}),
    ],
    'sklearn.ensemble.ExtraTreesClassifier': [
        ('n_estimators', {'t100': 100, 't300': 300}),
        ('max_depth', {'d10': 10, 'dfull': None}),
    ],
    'sklearn.ensemble.GradientBoostingClassifier': [
        ('learning_rate', {'lr001': 0.01, 'lr01': 0.1}),
        ('n_estimators', {'t100': 100, 't300': 300}),
    ],
    'sklearn.ensemble.HistGradientBoostingClassifier': [
        ('learning_rate', {'lr001': 0.01, 'lr01': 0.1}),
        ('max_leaf_nodes', {'l15': 15, 'l31': 31}),
    ],
    'sklearn.ensemble.AdaBoostClassifier': [
        ('learning_rate', {'lr05': 0.5, 'lr1': 1.0}),
        ('n_estimators', {'t50': 50, 't200': 200}),
    ],
    'sklearn.ensemble.BaggingClassifier': [
        ('n_estimators', {'t10': 10, 't50': 50}),
    ],
    'sklearn.neighbors.KNeighborsClassifier': [
        ('n_neighbors', {'n5': 5, 'n10': 10, 'n25': 25}),
        ('weights', {'uni': 'uniform', 'dist': 'distance'}),
    ],
    'sklearn.neighbors.NearestCentroid': [
        ('shrink_threshold', {'noshrink': None, 'shrink02': 0.2}),
    ],
    'sklearn.naive_bayes.MultinomialNB': [
        ('alpha', {'a001': 0.01, 'a1': 1.0, 'a10': 10.0}),
    ],
    'sklearn.naive_bayes.BernoulliNB': [
        ('alpha', {'a001': 0.01, 'a1': 1.0, 'a10': 10.0}),
    ],
    'sklearn.naive_bayes.ComplementNB': [
        ('alpha', {'a001': 0.01, 'a1': 1.0, 'a10': 10.0}),
    ],
    'sklearn.discriminant_analysis.LinearDiscriminantAnalysis': [
        ('solver', {'svd': {'solver': 'svd'},
                    'lsqrshrink': {'solver': 'lsqr', 'shrinkage': 'auto'}}),
    ],
    'sklearn.neural_network.MLPClassifier': [
        ('hidden_layer_sizes', {'h50': (50,), 'h100': (100,)}),
        ('alpha', {'a4': 0.0001, 'a3': 0.001}),
    ],
    'sklearn.semi_supervised.LabelPropagation': [
        ('kernel', {'rbf': 'rbf', 'knn': 'knn'}),
    ],
    'sklearn.semi_supervised.LabelSpreading': [
        ('kernel', {'rbf': 'rbf', 'knn': 'knn'}),
    ],
    'sklearn.calibration.CalibratedClassifierCV': [
        ('method', {'sig': 'sigmoid', 'iso': 'isotonic'}),
    ],
    'lightgbm.LGBMClassifier': [
        ('learning_rate', {'lr001': 0.01, 'lr01': 0.1}),
        ('num_leaves', {'l15': 15, 'l31': 31}),
    ],
    'xgboost.XGBClassifier': [
        ('learning_rate', {'lr01': 0.1, 'lr03': 0.3}),
        ('max_depth', {'d3': 3, 'd6': 6}),
    ],
    'xgboost.XGBRFClassifier': [
        ('max_depth', {'d3': 3, 'd6': 6}),
    ],
}


def _lazy_param_kwargs(base, route):
    """Resolve '#p:tok1.tok2' route tokens to kwargs via the ordered
    levels; unknown tokens end the walk (defaults from there on)."""
    kwargs = {}
    levels = _LAZY_PARAM_LEVELS.get(base, [])
    toks = [t for t in route.split('.') if t]
    for depth, tok in enumerate(toks):
        if depth >= len(levels):
            break
        param, menu = levels[depth]
        if tok not in menu:
            break
        val = menu[tok]
        if isinstance(val, dict):
            # one decision setting coupled knobs (e.g. solver+shrinkage)
            kwargs.update(val)
        else:
            kwargs[param] = val
    return kwargs


def _lazy_register():
    """Materialize the catalog into the registry: one entry per whitelisted
    class, keyed <family>_<package>_<ClassName> — the ontology path
    convention, so the leaf's _comment, its chain-item name, and its
    registry key are the same string. The curried entry binds the class
    path, exactly like the data_freetext_csvColumn corpus bindings above.
    An existing key is never overwritten: a hand-written vetted part always
    outranks a lazy one. A name containing an underscore inside a segment
    would break the ontology path notation (retrieve_ontology_item splits
    on '_'), so it is skipped; the builder never emits one."""
    family_of = {'classifier': 'classifier', 'clusterer': 'clusterer',
                 'transformer': 'vectorSpace',
                 'cluster_features': 'vectorSpace'}
    for class_path, kind in sorted(_lazy_whitelist().items()):
        family = family_of.get(kind)
        if family is None:
            continue
        package = class_path.split('.', 1)[0]
        leaf = class_path.rsplit('.', 1)[-1]
        # The '#features' variant materializes the same class through the
        # cluster-distance-features harness under a distinct catalog name.
        leaf = leaf.replace('#features', 'Features')
        if '#v:' in leaf:
            leaf = leaf.replace('#v:', '')
        if '_' in package or '_' in leaf or '#' in leaf:
            continue
        key = family + '_' + package + '_' + leaf
        if key not in registry:
            registry[key] = curr(estimator_lazy)(class_path)
        # Per-parameter walk entries: every PREFIX of the ordered level
        # tokens is its own catalog name (a stop mid-walk = defaults for
        # the rest — her grammar). Key: <base>_<tok1>[_<tok2>...].
        if '#' not in str(class_path):
            levels = _LAZY_PARAM_LEVELS.get(class_path, [])
            def _expand(prefix_toks, depth):
                if depth >= len(levels):
                    return
                _, menu = levels[depth]
                for tok in menu:
                    toks = prefix_toks + [tok]
                    pkey = key + '_' + '_'.join(toks)
                    route = class_path + '#p:' + '.'.join(toks)
                    if pkey not in registry:
                        registry[pkey] = curr(estimator_lazy)(route)
                    _expand(toks, depth + 1)
            _expand([], 0)


_lazy_register()


def _combiner_register():
    """a8 (her instruction 2026-08-12): the branch-capable catalogue.

    Products that take TWO or THREE inputs, so a delivered program can be
    a tree rather than only a chain. The wave 19 record held 5,889
    programs and not one branched - the catalogue had exactly one
    two-input product, it sat in the family the market used twice in a
    thousand iterations, and it threw on contact.

    Kept in their own module so the addition is auditable as a block.
    """
    try:
        from simulation import Combiners as _C
    except ImportError:                      # imported as a top-level module
        import Combiners as _C
    registry.update(_C.entries(curr))


_combiner_register()


# ---------------------------------------------------------------- APRIL 4 EMBEDDINGS
# Her order 2026-10-01: "if claude messed something up like the embeddings please
# put it back to april 4, do 2". The April 4 vectorSpace branch (commit 141891f,
# competing_clusterers.json) was gensim doc2vec with THREE stacked parameter
# levels: size 50/100/200/300 -> iterations 20/200/1000 -> minFreq 2/5, 24 full
# leaves, every node callable. Full leaves are registered exactly as April 4
# registered them; prefix nodes resolve to library defaults for the unset
# parameters ("defaults unless specifically set"; gensim 4: vector_size 100,
# epochs 10, min_count 5). The family-level names ('vectorSpace',
# 'vectorSpace_gensim') are NOT keys - her standing "no default at family
# level" ruling.
def _april4_embeddings_register():
    d2v = curries.get('vectorSpace_gensim_doc2vec') or curr(vectorSpace_gensim_doc2vec)
    sizes, iters, mins = (50, 100, 200, 300), (20, 200, 1000), (2, 5)
    D_SIZE, D_ITER, D_MIN = 100, 10, 5
    registry.setdefault('vectorSpace_gensim_doc2vec',
                        d2v(size=D_SIZE)(iterations=D_ITER)(minfreq=D_MIN))
    for s in sizes:
        registry.setdefault('vectorSpace_gensim_doc2vec_%dsize' % s,
                            d2v(size=s)(iterations=D_ITER)(minfreq=D_MIN))
        for it in iters:
            registry.setdefault('vectorSpace_gensim_doc2vec_%dsize_%diterations' % (s, it),
                                d2v(size=s)(iterations=it)(minfreq=D_MIN))
            for mf in mins:
                registry.setdefault(
                    'vectorSpace_gensim_doc2vec_%dsize_%diterations_%dminFreq' % (s, it, mf),
                    d2v(size=s)(iterations=it)(minfreq=mf))


# Option 2 (her "do 2", 2026-10-01): text vectorizers as LEAVES with parameter
# levels, so a chain on text must contain a vectorizer the seller chose and
# tuned, and embeddings differ. Named for the scikit-learn classes they run,
# with the classes' real parameters; no invented algorithms.
def _docs_of(X):
    if not isinstance(X, (list, tuple)) or len(X) < 2:
        return None
    out = []
    for d in X:
        if hasattr(d, 'words'):
            out.append(' '.join(d.words))
        elif isinstance(d, (list, tuple)):
            out.append(' '.join(str(t) for t in d))
        elif isinstance(d, str):
            out.append(d)
        else:
            return None
    return out


def vectorSpace_sklearn_TfidfVectorizer(X, features=None, ngram='word1', mindf=1):
    # _args: [{"type": "list"}]  _return: float matrix, one row per document
    import numpy as np
    print('vectorSpace_sklearn_TfidfVectorizer')
    docs = _docs_of(X)
    if docs is None:
        return None
    from sklearn.feature_extraction.text import TfidfVectorizer
    kw = {'min_df': mindf}
    if features:
        kw['max_features'] = int(features)
    if ngram == 'word12':
        kw['ngram_range'] = (1, 2)
    elif ngram == 'char35':
        kw['analyzer'] = 'char_wb'
        kw['ngram_range'] = (3, 5)
    try:
        M = TfidfVectorizer(**kw).fit_transform(docs[:_lazy_row_cap()])
    except ValueError:
        return None
    if M.shape[1] > _LAZY_MAX_WIDTH:
        from sklearn.decomposition import TruncatedSVD
        M = TruncatedSVD(n_components=_LAZY_MAX_WIDTH, random_state=0).fit_transform(M)
        return np.asarray(M, dtype=np.float32)
    return np.asarray(M.toarray(), dtype=np.float32)


def vectorSpace_sklearn_HashingVectorizer(X, features=1000):
    # _args: [{"type": "list"}]  _return: float matrix, one row per document
    import numpy as np
    print('vectorSpace_sklearn_HashingVectorizer')
    docs = _docs_of(X)
    if docs is None:
        return None
    from sklearn.feature_extraction.text import HashingVectorizer
    M = HashingVectorizer(n_features=int(features), alternate_sign=False,
                          norm='l2').fit_transform(docs[:_lazy_row_cap()])
    return np.asarray(M.toarray(), dtype=np.float32)


def _vectorizer_leaves_register():
    tf = curr(vectorSpace_sklearn_TfidfVectorizer)
    registry.setdefault('vectorSpace_sklearn_TfidfVectorizer', tf)
    for f in (100, 300, 1000, 3000):
        registry.setdefault('vectorSpace_sklearn_TfidfVectorizer_f%d' % f, tf(features=f))
        for ng in ('word1', 'word12', 'char35'):
            registry.setdefault('vectorSpace_sklearn_TfidfVectorizer_f%d_ng%s' % (f, ng),
                                tf(features=f)(ngram=ng))
            for df in (1, 2, 5):
                registry.setdefault(
                    'vectorSpace_sklearn_TfidfVectorizer_f%d_ng%s_df%d' % (f, ng, df),
                    tf(features=f)(ngram=ng)(mindf=df))
    hv = curr(vectorSpace_sklearn_HashingVectorizer)
    registry.setdefault('vectorSpace_sklearn_HashingVectorizer', hv)
    for f in (300, 1000, 3000):
        registry.setdefault('vectorSpace_sklearn_HashingVectorizer_f%d' % f, hv(features=f))


_april4_embeddings_register()
_vectorizer_leaves_register()


def _cipher_register():
    """The cipher world (her July 2026 instrument; module simulation/cipher_world.py,
    built 3 Oct 2026): 54 decoder leaves, 7 enciphered corpora, one fraction-correct
    test. A hand-written entry is never overwritten (setdefault). The test reads the
    auditing buyer's corpus from PORT_BUYER_DATA, so its memoise key must carry the
    buyer's corpus: it joins PORT_BUYER_KEYED."""
    global PORT_BUYER_KEYED
    try:
        from simulation import cipher_world as _CW
    except ImportError:                      # imported as a top-level module
        import cipher_world as _CW
    for _k, _v in _CW.entries(curr).items():
        registry.setdefault(_k, _v)
    PORT_BUYER_KEYED = tuple(PORT_BUYER_KEYED) + ('test_decoder_fractionCorrect',)


_cipher_register()
