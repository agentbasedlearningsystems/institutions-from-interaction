# Have a registered function for all values in the ontology

# Curry routine from internet site
# Massimiliano Tomassoli, 2012.

# Curry


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


def anomalyDetector_basic(X):
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
    iso = IsolationForest(random_state=42, contamination='auto')
    iso.fit(X)
    scores = -iso.score_samples(X).astype(np.float32)
    return (X, scores)




def test_clusterer_calinskiHarabaz(XY):
    X = XY[0]
    Y = XY[1]

    # "_args": [{  "type": "numpy.ndarray",  "dtype": "float32"},
    #           {"type": "numpy.ndarray", "dtype": "int32"}],
    # "_return": [{"type": "float"}]

    # we only want to test cosine metric for this example, but it could be a parameter in other cases

    from sklearn import metrics

    print('test_clusterer_calinskiHarabaz')
    min_score = 0
    max_score = 200

    calinski_harabaz = metrics.calinski_harabasz_score(X, Y)

    calinski_harabaz = (calinski_harabaz - min_score) / (max_score - min_score)
    return calinski_harabaz


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
# Truncation rule: when SNETSIM_VECTORSPACE_MAX_ROWS caps the loaded corpus
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
    or with a degenerate clusterer, and the SISTER agents would
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
    labels_path = os.path.join(os.getcwd(), 'data', 'billsByCongress_labels.npy')
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
    labels_path = os.path.join(os.getcwd(), 'data', 'iraVsControl_labels.npy')
    if not os.path.exists(labels_path):
        print(f'  ERROR: labels file missing: {labels_path}')
        return 0.0
    L = np.load(labels_path)
    return _ami_with_degeneracy_guard(L, Y)




# --- 2026-07-06 scaffolding: prefab stepping-stones (Debbie's design) ---
# Pre-assembled working parts: a newborn seller who stumbles on one has a
# working, verifiable product on day one. Measured floor on the IO need:
# 0.2334 AMI — NOT weak, because that corpus is easy once a chain executes
# at all (offline sweep: nearly any working pipeline lands 0.23-0.31).
# Outgrowability therefore comes from composition above the floor
# (consensus of diverse pipelines: 0.35-0.44 measured) and from the
# division need (~0.02 for everything; summit open). The May free-rider
# lesson is mitigated structurally: rivals CAN out-build this floor and
# buyer thermostats will demand it.

def vectorSpace_prefab_basicLsa(X):
    #   "_args": [{"type": "list","firstElement":"string" }],
    #   "_return": [{"type": "numpy.ndarray","dtype": "float32" }]
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    import sklearn.preprocessing
    print('vectorSpace_prefab_basicLsa')
    vec = TfidfVectorizer(stop_words='english', max_features=2000)
    M = vec.fit_transform(X)
    E = TruncatedSVD(n_components=20, random_state=0).fit_transform(M)
    return sklearn.preprocessing.normalize(E).astype('float32')


def clusterer_prefab_basicPipeline(X):
    #   "_args": [{"type": "list","firstElement":"string" }],
    #   "_return": [{"type": "tuple","dtype": "(float32,int32)" }]
    from sklearn.cluster import KMeans
    import numpy
    print('clusterer_prefab_basicPipeline')
    E = vectorSpace_prefab_basicLsa(X)
    Y = KMeans(n_clusters=20, n_init=3, random_state=0).fit_predict(E)
    return (E, numpy.asarray(Y, dtype='int32'))


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
    # explicitly. Override via SNETSIM_GENSIM_WORKERS env var (host has
    # 14 cores so 4-8 is reasonable; multiple python processes share the
    # CPU so going higher risks oversubscription).
    import os as _os
    _workers = int(_os.environ.get("SNETSIM_GENSIM_WORKERS", "4"))
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
    # SNETSIM_VECTORSPACE_MAX_ROWS overrides the default. 5000 keeps the
    # pairwise matrix at 5000^2 * 8 = 200 MB - fits comfortably in RAM.
    # HEAD-truncation (first N rows, order preserved), NOT a random
    # subsample: ground-truth label sidecars align via labels[:len(Y)]
    # only if row order and prefix survive this cap. The prior seeded
    # random subsample silently broke every label-aligned test on
    # capped corpora (the bills ideology null result).
    import os as _os
    _max_rows = int(_os.environ.get("SNETSIM_VECTORSPACE_MAX_ROWS", "5000"))
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
    _max_rows = int(_os.environ.get("SNETSIM_VECTORSPACE_MAX_ROWS", "5000"))
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
    import gensim

    print('preprocessor_freetext_lemmatization')
    stopfile = 'stopwords.txt'
    lemmatized = []
    with open(stopfile, 'r') as f:
        stopwords = {word.lower().strip() for word in f.readlines()}
        lemma = lambda x: [b.decode('utf-8') for b in gensim.utils.lemmatize(str(x), stopwords=frozenset(stopwords))]

        if type(X) is str:
            lemmatized = lemma(X)
        else:
            lemmatized = [lemma(x) for x in X]

    return lemmatized


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
    random.shuffle(X)
    return X

# Data
def data_freetext_csvColumn(relative_path, col='text', nrows=None):
    #  returns a list of documents that are strings
    #   "_return": [{"type": "list","firstElement":"string" }]
    # If nrows is set, only the first nrows lines are loaded.
    # Useful for tractable fitness evaluation during research; the
    # GCON_DATA_NROWS env var overrides the parameter for global control.

    import pandas as pd
    import os

    cwd = os.getcwd()
    path = cwd + relative_path

    env_nrows = os.environ.get('GCON_DATA_NROWS')
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
    X, Y = make_circles(n_samples=n_samples, factor=.5,
                        noise=.05)
    return X


def data_vector_moons(n_samples=1500):
    from sklearn.datasets import make_moons
    print('data_vector_moons')
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
    _njobs = int(_os.environ.get("SNETSIM_SKLEARN_NJOBS", "4"))
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
    _njobs = int(_os.environ.get("SNETSIM_SKLEARN_NJOBS", "4"))
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
    _njobs = int(_os.environ.get("SNETSIM_SKLEARN_NJOBS", "4"))
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




def anomalyDetector_oneClassSVM(X):
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
        svm = OneClassSVM(gamma='scale', nu=0.1, max_iter=20000).fit(X)
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
        vec = TfidfVectorizer(max_features=300, min_df=2)
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


def classifier_logistic_v2(XY):
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
    # 2026-07-16 (shared-scenario port, Debbie's exact-split ruling): the
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
            clf = LogisticRegression(max_iter=200, random_state=42)
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
    _max_rows = int(_os.environ.get("SNETSIM_VECTORSPACE_MAX_ROWS", "5000"))
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
    #   - REVISED 2026-07-15 (Debbie): consensus(A, A) — and any pair with
    #     adjusted-Rand > 0.95 — is REJECTED (None -> score 0): a
    #     combination of clones is not a combination. Likewise a pair
    #     where either input carries no partition is rejected instead of
    #     passing the survivor through. Depth credit requires genuine
    #     diversity of inputs.
    # Arity-2 execution matches vectorSpace_concat: SnetSim calls
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
    # 2026-07-15, Debbie: "why dont you fix your consensus so that it
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
# The LLM world's five-want demand catalog (llm_market/gcon_market.py,
# default_wants) realized as chain-world parts, so a CMA-ES society can run
# the same demand and later serve as the frozen ancestor of hybrid LLM
# societies. Corpora staged by _port_prepare_data.py into
# data/{disasterTweets,movieSentiment,movieSentimentFewshot}.csv with
# ground-truth sidecars <stem>_labels.npy built under data_freetext_
# csvColumn's exact row-survival rule, so labels[:n] stays aligned through
# head-truncation (the billsByCongress_labels.npy contract).
#
# v2 (Debbie's 2026-07-16 ruling — differences with the LLM world get
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


_PORT_SPLIT_STEMS = ('disasterTweets', 'movieSentiment',
                     'movieSentimentFewshot')
_PORT_SPLIT_CACHE = {}

# --- market-audition context (Debbie's 2026-07-16 ruling: a want's
# scoring truth binds to the BUYER, closing the embedding-path cross-truth
# arbitrage the labeler text guard could not reach) ---
# PORT_BUYER_DATA: the auditing buyer's data terminal for the evaluation
#   in flight — published by SnetAgent.pass_all_tests around each
#   perform_test and cleared after. None outside a market audition
#   (offline/direct calls keep the v2 chain-truth fallback semantics).
# PORT_CURRENT_CHAIN: the chain tree being evaluated — published by
#   SnetSim.call_emergent_function; lets the scorer refuse chains that
#   wire a FOREIGN port corpus terminal inside themselves.
# PORT_BUYER_KEYED: scorer roots whose memoise cache key must carry the
#   buyer's terminal (SnetSim.call_memoise_pickle) — a self-contained
#   foreign chain drops the buyer's appended terminal from the karva tree,
#   so without per-buyer keys the first buyer's cached score would cross
#   wants. Read only by the port machinery; historical scenarios are
#   byte-identical (no scorer in PORT_BUYER_KEYED ever appears in them).
PORT_BUYER_DATA = None
PORT_CURRENT_CHAIN = None
PORT_BUYER_KEYED = ('test_classifier_f1heldout',)

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
    # Scaffolding (the prefab precedent, Debbie's 2026-07-06 design): if X
    # is still raw text (a list of strings / TaggedDocuments) instead of an
    # embedding, embed it with the prefab LSA first. A newborn seller who
    # chains classifier_x(labeler_y(corpus)) therefore owns a working,
    # verifiable product on day one; a real vectorSpace upstream outgrows
    # the floor and gets paid for it (quality-scaled settlement).
    import os
    import numpy
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
        return None  # prefab floor removed (Debbie 2026-07-19: probes only)
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
    # (_port_or_seeded_split; Debbie's 2026-07-16 exact-split ruling),
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
    # Diversity gate (clusterer_consensus precedent, Debbie's 2026-07-15
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


def test_classifier_f1heldout(XYP):
    # F1 variant of test_classifier_heldout — the LLM world's final_f1
    # scoring ("F1 for label=1 on held-out data") ported to the chain
    # world. test_classifier_heldout is untouched.
    #
    # v2 (Debbie's 2026-07-16 exact-split ruling): the score is F1 for
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
    # AUDITION (SnetAgent.pass_all_tests publishes PORT_BUYER_DATA) the
    # scoring truth is the BUYER's corpus and nothing else — a
    # self-contained chain that carries a different port corpus's truth
    # (e.g. classifier(labeler_movieSentiment(tfidf(data_movieSentiment)))
    # offered to the disaster human) scores 0 instead of the ~0.78 its own
    # truth would print. The published chain tree is also checked: a
    # foreign port terminal wired anywhere inside the delivered chain
    # refuses the audition. Outside an audition (no publication) the v2
    # chain-truth semantics hold, for offline/harness use.
    # _args:   [{"type": "tuple", "dtype": "(float32,int32,int32)"}]
    # _return: [{"type": "float"}]
    import numpy as np
    print('test_classifier_f1heldout')
    if XYP is None or len(XYP) < 3 or XYP[1] is None or XYP[2] is None:
        return 0.0
    Y, P = np.asarray(XYP[1]).ravel(), np.asarray(XYP[2]).ravel()
    n = min(len(Y), len(P))
    if n < 2:
        return 0.0
    Y, P = Y[:n], P[:n]
    if PORT_BUYER_DATA is not None:
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
    return float(np.clip(f1, 0.0, 1.0))


def test_vectorSpace_probe(X):
    # The doc_embedding want's fixed probe, ported from the LLM world's
    # PROBE_CODE (llm_market/gcon_market.py): the buyer takes the DELIVERED
    # EMBEDDING of the movie-sentiment corpus, trains
    # LogisticRegression(max_iter=2000) on the corpus's train rows, and
    # pays ACCURACY on its test rows. v2 (Debbie's 2026-07-16 exact-split
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
    labels_path = os.path.join(os.getcwd(), 'data',
                               'movieSentiment_labels.npy')
    mask_path = os.path.join(os.getcwd(), 'data',
                             'movieSentiment_istest.npy')
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
    # DEDUP END-TASK terminal (Debbie 2026-07-19: an end task that is
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
            'vectorSpace_prefab_basicLsa': curr(vectorSpace_prefab_basicLsa),
            'clusterer_prefab_basicPipeline': curr(clusterer_prefab_basicPipeline),
            # === Added 2026-07-16: shared-scenario port (see block above) ===
            # The LLM world's five-want corpora as data terminals
            # (data_freetext_csvColumn pattern, staged by
            # _port_prepare_data.py):
            'data_freetext_disasterTweets': curries['data_freetext_csvColumn']('/data/disasterTweets.csv'),
            'data_freetext_movieSentiment': curries['data_freetext_csvColumn']('/data/movieSentiment.csv'),
            'data_freetext_movieSentimentFewshot': curries['data_freetext_csvColumn']('/data/movieSentimentFewshot.csv'),
            # Ground-truth joiners (billsIdeology sidecar convention):
            'labeler_disasterTweets': curr(labeler_disasterTweets),
            'labeler_movieSentiment': curr(labeler_movieSentiment),
            'labeler_movieSentimentFewshot': curr(labeler_movieSentimentFewshot),
            # Two more learner families (logistic_v2 harness):
            'classifier_svm_linear': curr(classifier_svm_linear),
            'classifier_naivebayes_multinomial': curr(classifier_naivebayes_multinomial),
            # Arity-2 ensemble combiner; the classifier_ensemble item path
            # is only satisfiable through this part:
            'classifier_ensemble_vote': curr(classifier_ensemble_vote),
            # The two ported want-scorers:
            'test_classifier_f1heldout': curr(test_classifier_f1heldout),
            'test_vectorSpace_probe': curr(test_vectorSpace_probe)}



# === Added 2026-07-18: the lazy big registry (Debbie's design) =============
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
#                  dense tfidf bridge (max_features=300, min_df=2 — the
#                  vectorSpace_sklearn_tfidf convention).
#
# Rails, following the dbscan / oneClassSVM precedents: inputs and outputs
# wider than _LAZY_MAX_WIDTH columns are refused; rows are HEAD-truncated to
# SNETSIM_VECTORSPACE_MAX_ROWS so ground-truth sidecars stay aligned; where
# a class exposes the knobs, random_state is pinned (42) and n_jobs is bound
# to SNETSIM_SKLEARN_NJOBS like every other part. A class that raises
# anywhere — import, construction, fit, predict — returns None: an
# unevaluable product that scores 0 in the market, never a crash. One line
# goes to stdout on failure so run logs stay diagnosable.

_LAZY_MAX_WIDTH = 2000
_LAZY_WHITELIST = None


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
    SNETSIM_SKLEARN_NJOBS (default 4, like the other sklearn parts); no
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
        if 'random_state' in params and params.get('random_state') is None:
            updates['random_state'] = 42
        if 'n_jobs' in params:
            updates['n_jobs'] = int(_os.environ.get('SNETSIM_SKLEARN_NJOBS', '4'))
        if updates:
            est.set_params(**updates)
    return est, kind


def _lazy_row_cap():
    import os as _os
    return int(_os.environ.get('SNETSIM_VECTORSPACE_MAX_ROWS', '5000'))


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
    # convention: max_features=300, min_df=2). An upstream embedding matrix
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
        from sklearn.feature_extraction.text import TfidfVectorizer
        M = np.asarray(
            TfidfVectorizer(max_features=300, min_df=2)
            .fit_transform(docs).toarray(), dtype=np.float32)
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


# Sensible per-class settings (Debbie 2026-07-19: "put sensible
# hyperparameter settings in all members, similar to the registry
# ontology of apr 4" — "this leaves more room for specialist
# knowledge"). Conventions mirrored from the hand-built entries:
# ~100 components for decompositions (the lsa_100 style), 20 clusters
# with n_init=3, raised iteration budgets, random_state=0 wherever
# accepted. Reference values for the per-parameter settings tree (the
# environment, so default-worlds and sensible-worlds share one code.
_LAZY_SENSIBLE = {
    'sklearn.decomposition.TruncatedSVD': {'n_components': 100},
    'sklearn.decomposition.PCA': {'n_components': 100},
    'sklearn.decomposition.IncrementalPCA': {'n_components': 100},
    'sklearn.decomposition.KernelPCA': {'n_components': 100},
    'sklearn.decomposition.SparsePCA': {'n_components': 100},
    'sklearn.decomposition.MiniBatchSparsePCA': {'n_components': 100},
    'sklearn.decomposition.FactorAnalysis': {'n_components': 100},
    'sklearn.decomposition.NMF': {'n_components': 100, 'max_iter': 500},
    'sklearn.decomposition.MiniBatchNMF': {'n_components': 100},
    'sklearn.decomposition.DictionaryLearning': {'n_components': 100},
    'sklearn.decomposition.MiniBatchDictionaryLearning': {'n_components': 100},
    'sklearn.decomposition.LatentDirichletAllocation': {'n_components': 100},
    'sklearn.decomposition.FastICA': {'n_components': 100, 'max_iter': 500},
    'sklearn.manifold.Isomap': {'n_components': 50, 'n_neighbors': 10},
    'sklearn.manifold.LocallyLinearEmbedding': {'n_components': 50, 'n_neighbors': 10},
    'sklearn.manifold.SpectralEmbedding': {'n_components': 50},
    'sklearn.manifold.TSNE': {'n_components': 3, 'perplexity': 30},
    'sklearn.kernel_approximation.Nystroem': {'n_components': 100},
    'sklearn.kernel_approximation.RBFSampler': {'n_components': 100},
    'sklearn.kernel_approximation.SkewedChi2Sampler': {'n_components': 100},
    'sklearn.kernel_approximation.PolynomialCountSketch': {'n_components': 100},
    'sklearn.cluster.KMeans': {'n_clusters': 20, 'n_init': 3},
    'sklearn.cluster.MiniBatchKMeans': {'n_clusters': 20, 'n_init': 3},
    'sklearn.cluster.BisectingKMeans': {'n_clusters': 20},
    'sklearn.cluster.Birch': {'n_clusters': 20},
    'sklearn.cluster.AgglomerativeClustering': {'n_clusters': 20},
    'sklearn.cluster.SpectralClustering': {'n_clusters': 20},
    'sklearn.cluster.AffinityPropagation': {'damping': 0.9, 'max_iter': 300},
    'sklearn.cluster.HDBSCAN': {'min_cluster_size': 10},
    'sklearn.linear_model.LogisticRegression': {'max_iter': 2000},
    'sklearn.linear_model.SGDClassifier': {'max_iter': 2000},
    'sklearn.linear_model.PassiveAggressiveClassifier': {'max_iter': 2000},
    'sklearn.svm.LinearSVC': {'max_iter': 5000},
    'sklearn.neural_network.MLPClassifier': {'max_iter': 500, 'early_stopping': True},
    'sklearn.ensemble.RandomForestClassifier': {'n_estimators': 200},
    'sklearn.ensemble.ExtraTreesClassifier': {'n_estimators': 200},
    'sklearn.neighbors.KNeighborsClassifier': {'n_neighbors': 10},
    'sklearn.neighbors.KNeighborsTransformer': {'n_neighbors': 10},
}


# Setting-variant menus (Debbie 2026-07-19: "wouldnt it be smart to
# have several choices for different scenarios?"). Each variant is its
# own catalog leaf; the market selects settings by what sells. Keys are
# '<class_path>#v:<Suffix>'; the suffix names the leaf (KMeansK8 etc.).
_LAZY_VARIANTS = {
    'sklearn.decomposition.TruncatedSVD': {'C20': {'n_components': 20}, 'C100': {'n_components': 100}, 'C300': {'n_components': 300}},
    'sklearn.decomposition.PCA': {'C20': {'n_components': 20}, 'C100': {'n_components': 100}, 'C300': {'n_components': 300}},
    'sklearn.decomposition.DictionaryLearning': {'C50': {'n_components': 50}, 'C100': {'n_components': 100}, 'C200': {'n_components': 200}},
    'sklearn.decomposition.LatentDirichletAllocation': {'T20': {'n_components': 20}, 'T100': {'n_components': 100}},
    'sklearn.kernel_approximation.Nystroem': {'C100': {'n_components': 100}, 'C300': {'n_components': 300}},
    'sklearn.kernel_approximation.RBFSampler': {'C100': {'n_components': 100}, 'C300': {'n_components': 300}},
    'sklearn.manifold.Isomap': {'C20': {'n_components': 20, 'n_neighbors': 10}, 'C50': {'n_components': 50, 'n_neighbors': 10}},
    'sklearn.manifold.LocallyLinearEmbedding': {'C20': {'n_components': 20, 'n_neighbors': 10}, 'C50': {'n_components': 50, 'n_neighbors': 10}},
    'sklearn.cluster.KMeans': {'K2': {'n_clusters': 2, 'n_init': 3}, 'K8': {'n_clusters': 8, 'n_init': 3}, 'K20': {'n_clusters': 20, 'n_init': 3}, 'K50': {'n_clusters': 50, 'n_init': 3}},
    'sklearn.cluster.MiniBatchKMeans': {'K2': {'n_clusters': 2, 'n_init': 3}, 'K8': {'n_clusters': 8, 'n_init': 3}, 'K20': {'n_clusters': 20, 'n_init': 3}, 'K50': {'n_clusters': 50, 'n_init': 3}},
    'sklearn.cluster.AgglomerativeClustering': {'K8': {'n_clusters': 8}, 'K20': {'n_clusters': 20}, 'K50': {'n_clusters': 50}},
    'sklearn.cluster.Birch': {'K8': {'n_clusters': 8}, 'K20': {'n_clusters': 20}, 'K50': {'n_clusters': 50}},
    'sklearn.cluster.BisectingKMeans': {'K8': {'n_clusters': 8}, 'K20': {'n_clusters': 20}, 'K50': {'n_clusters': 50}},
    'sklearn.cluster.HDBSCAN': {'M5': {'min_cluster_size': 5}, 'M10': {'min_cluster_size': 10}, 'M25': {'min_cluster_size': 25}},
    'sklearn.neighbors.KNeighborsClassifier': {'N5': {'n_neighbors': 5}, 'N10': {'n_neighbors': 10}, 'N25': {'n_neighbors': 25}},
    'sklearn.neighbors.KNeighborsTransformer': {'N5': {'n_neighbors': 5}, 'N10': {'n_neighbors': 10}, 'N25': {'n_neighbors': 25}},
    'sklearn.linear_model.LogisticRegression': {'Creg01': {'C': 0.1, 'max_iter': 2000}, 'Creg1': {'C': 1.0, 'max_iter': 2000}, 'Creg10': {'C': 10.0, 'max_iter': 2000}},
    'sklearn.neural_network.MLPClassifier': {'H100': {'hidden_layer_sizes': (100,), 'max_iter': 500, 'early_stopping': True}, 'H200x100': {'hidden_layer_sizes': (200, 100), 'max_iter': 500, 'early_stopping': True}},
    'sklearn.ensemble.RandomForestClassifier': {'T100': {'n_estimators': 100}, 'T300': {'n_estimators': 300}},
    'sklearn.ensemble.ExtraTreesClassifier': {'T100': {'n_estimators': 100}, 'T300': {'n_estimators': 300}},
}


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
        if '#v:' in str(class_path):
            base, _, suffix = str(class_path).partition('#v:')
            kw = dict(_LAZY_VARIANTS.get(base, {}).get(suffix, {}))
            if hasattr(est, 'get_params') and kw:
                try:
                    params = est.get_params(deep=False)
                    est.set_params(**{k: v for k, v in kw.items()
                                      if k in params})
                except Exception:
                    pass
        if '#features' in str(class_path):
            # Cluster-distance features (Debbie 2026-07-19: clusterers need
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




# Per-parameter settings walk (Debbie 2026-07-19: "when an agent puts
# something like brandname_routine_stop, it doesnt fill parameters in
# and thus goes to their defaults, where brandname_routine_param1value_stop
# fills in only param1 but the rest go to their defaults"). Each class
# gets ORDERED parameter levels; the genome walks them level by level,
# and a stop at any depth leaves every unwalked parameter at its
# library default. Tokens are underscore-free (the ontology path
# notation splits on '_').
_LAZY_PARAM_LEVELS = {
    'sklearn.cluster.KMeans': [
        ('n_clusters', {'nclusters2': 2, 'nclusters8': 8, 'nclusters20': 20, 'nclusters50': 50}),
        ('n_init', {'ninit3': 3, 'ninit10': 10}),
    ],
    'sklearn.cluster.MiniBatchKMeans': [
        ('n_clusters', {'nclusters2': 2, 'nclusters8': 8, 'nclusters20': 20, 'nclusters50': 50}),
    ],
    'sklearn.cluster.AgglomerativeClustering': [
        ('n_clusters', {'nclusters2': 2, 'nclusters8': 8, 'nclusters20': 20, 'nclusters50': 50}),
    ],
    'sklearn.cluster.Birch': [
        ('n_clusters', {'nclusters2': 2, 'nclusters8': 8, 'nclusters20': 20, 'nclusters50': 50}),
    ],
    'sklearn.decomposition.TruncatedSVD': [
        ('n_components', {'comp20': 20, 'comp100': 100, 'comp300': 300}),
    ],
    'sklearn.decomposition.PCA': [
        ('n_components', {'comp20': 20, 'comp100': 100, 'comp300': 300}),
    ],
    'sklearn.decomposition.NMF': [
        ('n_components', {'comp20': 20, 'comp100': 100, 'comp300': 300}),
    ],
    'sklearn.decomposition.DictionaryLearning': [
        ('n_components', {'comp50': 50, 'comp100': 100}),
    ],
    'sklearn.decomposition.LatentDirichletAllocation': [
        ('n_components', {'comp20': 20, 'comp100': 100, 'comp300': 300}),
    ],
    'sklearn.kernel_approximation.Nystroem': [
        ('n_components', {'comp20': 20, 'comp100': 100, 'comp300': 300}),
    ],
    'sklearn.linear_model.LogisticRegression': [
        ('C', {'creg01': 0.1, 'creg1': 1.0, 'creg10': 10.0}),
    ],
    'sklearn.svm.LinearSVC': [
        ('C', {'creg01': 0.1, 'creg1': 1.0, 'creg10': 10.0}),
    ],
    'sklearn.ensemble.RandomForestClassifier': [
        ('n_estimators', {'trees100': 100, 'trees200': 200, 'trees500': 500}),
    ],
    'sklearn.neural_network.MLPClassifier': [
        ('hidden_layer_sizes', {'h100': (100,), 'h200x100': (200, 100)}),
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
        kwargs[param] = menu[tok]
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
