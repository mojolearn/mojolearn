# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# ESTIMATOR TRANSFORM-FAMILY LANES (lane vdv-cov-est, 2026-11-24; plan docs/plans/VERIFY_THROUGH_DEVICE.md
# item C): `fit_transform`, `inverse_transform`, `partial_fit` and a bare `fit` that the inventory
# (docs/plans/verify_through_device/endpoints.tsv) left unmapped "OWN LANE".
#
# `fit_transform` is a fit endpoint, so each such lane FITS AFRESH: it calls the constructor through `ml` with
# the arguments of the base lane the table names, on that lane's fixture slice (cut to a light size where the
# base fits the whole fixture), and digests the one-call result. It never refits the base lane's object. An
# `inverse_transform` lane digests the inverse of the estimator's own transform output. A lane named
# `<base>-...` keeps the base lane's configuration; none of them calls the base lane (a fresh fit needs none),
# except `x-prep-label-encoder-inverse`, which extends it.
#
# Device-agnostic (the vd-verify contract): no `device=`, no vendor read. Every lane is an independent unit
# (own fixture slice, own model, nothing shared). Every reference is OWED to the release's one verification.
# Helpers carry the `_m_transform_` prefix; lane names are literal; nothing existing is rebound.


def _m_transform_h(*outs):
    """`_h` over the arrays of each output: a tuple or list of arrays (PLS scores, (mean, var) pairs) is
    flattened in order; a single array passes as is."""
    arrays = []
    for out in outs:
        if isinstance(out, (tuple, list)):
            arrays.extend(np.asarray(a) for a in out)
        else:
            arrays.append(np.asarray(out))
    return _h(*arrays)


def _m_transform_ragged(rows):
    """A list of label tuples (MultiLabelBinarizer.inverse_transform) as two int64 arrays: the lengths and
    the concatenated labels, so `_h` never sees an object array."""
    lengths = np.asarray([len(r) for r in rows], dtype=np.int64)
    flat = np.asarray([int(v) for r in rows for v in r], dtype=np.int64)
    return lengths, flat


# ---------------------------------------------------------------- clustering and kernel feature maps

@lane("x-cluster-bisecting-kmeans-fit-calls")
def _(ml, X, yc, yr, Xh=None):
    """BisectingKMeans.fit_transform and .fit_predict, each a fresh fit with x-cluster-bisecting-kmeans'
    arguments on its 2000 x 6 rows."""
    rows = np.ascontiguousarray(X[:2000, :6])
    T = ml.BisectingKMeans(n_clusters=5, n_init=2, random_state=3).fit_transform(rows)
    labels = ml.BisectingKMeans(n_clusters=5, n_init=2, random_state=3).fit_predict(rows)
    return _fit(dict(fit_transform=_m_transform_h(T), fit_predict=_m_transform_h(labels)))


@lane("x-cluster-minibatch-kmeans-fit-calls")
def _(ml, X, yc, yr, Xh=None):
    """MiniBatchKMeans.fit_transform and .fit_predict, fresh fits with x-cluster-minibatch-kmeans' arguments
    on its 3000 x 8 rows."""
    rows = np.ascontiguousarray(X[:3000, :8])
    T = ml.MiniBatchKMeans(n_clusters=6, batch_size=256, max_iter=5, random_state=3).fit_transform(rows)
    labels = ml.MiniBatchKMeans(n_clusters=6, batch_size=256, max_iter=5, random_state=3).fit_predict(rows)
    return _fit(dict(fit_transform=_m_transform_h(T), fit_predict=_m_transform_h(labels)))


@lane("kernel-ridge-laplacian-fit-transform")
def _(ml, X, yc, yr, Xh=None):
    """Nystroem.fit_transform with the laplacian kernel variant's arguments (`_kernel_variant`) on its 48
    scaled rows."""
    x = np.ascontiguousarray(X[:48, :4] * np.float32(0.125))
    T = ml.Nystroem(n_components=8, random_state=7, kernel="laplacian", gamma=0.5, coef0=0.25,
                    degree=3).fit_transform(x)
    return _fit(dict(fit_transform=_m_transform_h(T)))


@lane("rbf-sampler-fit-transform")
def _(ml, X, yc, yr, Xh=None):
    """RBFSampler.fit_transform with the rbf-sampler lane's arguments on 256 rows."""
    T = ml.RBFSampler(gamma=0.5, n_components=64, random_state=1).fit_transform(np.ascontiguousarray(X[:256]))
    return _fit(dict(fit_transform=_m_transform_h(T)))


@lane("x-neighbors-poly-sketch-fit-transform")
def _(ml, X, yc, yr, Xh=None):
    T = ml.PolynomialCountSketch(degree=3, gamma=0.5, coef0=1.0, n_components=64,
                                 random_state=3).fit_transform(np.ascontiguousarray(X[:512]))
    return _fit(dict(fit_transform=_m_transform_h(T)))


@lane("x-neighbors-additive-chi2-fit-transform")
def _(ml, X, yc, yr, Xh=None):
    A = np.abs(X[:512]).astype(np.float32)
    A[::7, 2] = np.float32(0.0)                         # zeros take the masked branch, as the base lane
    T = ml.AdditiveChi2Sampler(sample_steps=3).fit_transform(A)
    return _fit(dict(fit_transform=_m_transform_h(T)))


@lane("x-neighbors-skewed-chi2-fit-transform")
def _(ml, X, yc, yr, Xh=None):
    A = np.abs(X[:512]).astype(np.float32)
    T = ml.SkewedChi2Sampler(skewedness=0.5, n_components=64, random_state=5).fit_transform(A)
    return _fit(dict(fit_transform=_m_transform_h(T)))


@lane("x-neighbors-knn-imputer-fit-transform")
def _(ml, X, yc, yr, Xh=None):
    A = _neighbors_holes(X[:512])
    T = ml.KNNImputer(n_neighbors=5).fit_transform(A)
    W = ml.KNNImputer(n_neighbors=4, weights="distance", add_indicator=True).fit_transform(A)
    return _fit(dict(uniform=_m_transform_h(T), distance=_m_transform_h(W)))


@lane("umap-fit-transform")
def _(ml, X, yc, yr, Xh=None):
    """UMAP.fit_transform with the umap lane's arguments on its 1024 x 8 rows (the embedding of the fit)."""
    E = ml.UMAP(n_neighbors=8, n_components=2, n_epochs=8, random_state=3).fit_transform(
        np.ascontiguousarray(X[:1024, :8]))
    return _fit(dict(fit_transform=_m_transform_h(E)))


@lane("spectral-embedding-fit-transform")
def _(ml, X, yc, yr, Xh=None):
    """SpectralEmbedding.fit_transform with the spectral-embedding lane's k-NN graph arguments on its
    512 x 4 rows, fitted afresh."""
    E = ml.SpectralEmbedding(n_components=3, n_neighbors=10, random_state=3).fit_transform(
        np.ascontiguousarray(X[:512, :4]))
    return _fit(dict(fit_transform=_m_transform_h(E)))


# ---------------------------------------------------------------- decompositions

@lane("tsvd-inverse-fit-transform")
def _(ml, X, yc, yr, Xh=None):
    T = ml.TruncatedSVD(n_components=4).fit_transform(np.ascontiguousarray(X[:4096]))
    return _fit(dict(fit_transform=_m_transform_h(T)))


@lane("x-decomp-ipca-fit-calls")
def _(ml, X, yc, yr, Xh=None):
    """IncrementalPCA.fit_transform (x-decomp-ipca's arguments, 8000 rows in two batches) and partial_fit
    (two calls of 2000 rows, then the components, the running statistics and a transform)."""
    T = ml.IncrementalPCA(n_components=6, batch_size=4000).fit_transform(np.ascontiguousarray(X[:8000]))
    p = ml.IncrementalPCA(n_components=6)
    p.partial_fit(np.ascontiguousarray(X[:2000]))
    p.partial_fit(np.ascontiguousarray(X[2000:4000]))
    return _fit(dict(fit_transform=_m_transform_h(T),
                     partial_fit=_h(p.components_, p.mean_, p.var_, p.singular_values_, p.transform(X[:256]))))


@lane("x-decomp-lda-fit-calls")
def _(ml, X, yc, yr, Xh=None):
    """LatentDirichletAllocation.fit_transform (batch, x-decomp-lda's arguments) and partial_fit (the online
    learner over two 200-row calls), on the base lane's integer counts."""
    C = np.minimum(np.floor(np.abs(X[:400]) * np.float32(3)).astype(np.float32), np.float32(50))
    T = ml.LatentDirichletAllocation(n_components=4, max_iter=5, random_state=0).fit_transform(C)
    o = ml.LatentDirichletAllocation(n_components=3, learning_method="online", batch_size=100, random_state=1)
    o.partial_fit(C[:200])
    o.partial_fit(C[200:400])
    return _fit(dict(fit_transform=_m_transform_h(T), partial_fit=_h(o.components_, o.transform(C[:128]))))


@lane("x-decomp-pls-fit-calls")
def _(ml, X, yc, yr, Xh=None):
    """PLSRegression, PLSCanonical and CCA: fit_transform with x-decomp-pls' arguments and slices (the X and
    Y scores), then inverse_transform of those scores back to X (and Y)."""
    Yc = np.ascontiguousarray(X[:2000, 10:14])
    Ya = np.ascontiguousarray(X[:1000, 8:12])
    parts = {}
    for tag, est, x, y in (  # glue: the three PLS estimators of the base lane
            ("pls", ml.PLSRegression(n_components=3), X[:2000], yr[:2000]),
            ("canonical", ml.PLSCanonical(n_components=2), np.ascontiguousarray(X[:2000, :10]), Yc),
            ("cca", ml.CCA(n_components=2, max_iter=100), np.ascontiguousarray(X[:1000, :8]), Ya)):
        scores = est.fit_transform(x, y)
        parts[tag + "_fit_transform"] = _m_transform_h(scores)
        if isinstance(scores, (tuple, list)):
            back = est.inverse_transform(scores[0], Y=scores[1])
        else:
            back = est.inverse_transform(scores)
        parts[tag + "_inverse"] = _m_transform_h(back)
    return _fit(parts)


@lane("x-decomp-sparse-pca-fit-calls")
def _(ml, X, yc, yr, Xh=None):
    """SparsePCA and MiniBatchSparsePCA: fit_transform with x-decomp-sparse-pca's arguments, then
    inverse_transform of the codes."""
    S = np.ascontiguousarray(X[:300])
    s = ml.SparsePCA(n_components=4, alpha=1, max_iter=8, random_state=0)
    Ts = s.fit_transform(S)
    b = ml.MiniBatchSparsePCA(n_components=3, alpha=1, max_iter=4, batch_size=4, random_state=3)
    Tb = b.fit_transform(np.ascontiguousarray(X[:200]))
    return _fit(dict(sparse=_m_transform_h(Ts), sparse_inverse=_m_transform_h(s.inverse_transform(Ts)),
                     minibatch=_m_transform_h(Tb), minibatch_inverse=_m_transform_h(b.inverse_transform(Tb))))


@lane("x-decomp-srp-fit-calls")
def _(ml, X, yc, yr, Xh=None):
    """SparseRandomProjection.fit_transform with x-decomp-srp's inverse-components arguments, then
    inverse_transform of the projection."""
    t = ml.SparseRandomProjection(n_components=6, density=0.6, random_state=9, compute_inverse_components=True)
    T = t.fit_transform(np.ascontiguousarray(X[:512]))
    return _fit(dict(fit_transform=_m_transform_h(T), inverse=_m_transform_h(t.inverse_transform(T))))


@lane("x-decomp-dict-learning-inverse")
def _(ml, X, yc, yr, Xh=None):
    """inverse_transform of DictionaryLearning, MiniBatchDictionaryLearning and SparseCoder, each on its own
    codes, with x-decomp-dict-learning's arguments (the coder takes the learned dictionary)."""
    S = np.ascontiguousarray(X[:300])
    m = ml.DictionaryLearning(n_components=6, alpha=0.5, max_iter=8, random_state=0)
    code = m.fit_transform(S)
    mb = ml.MiniBatchDictionaryLearning(n_components=5, alpha=0.4, batch_size=64, max_iter=3, random_state=2,
                                        transform_algorithm="lasso_lars").fit(np.ascontiguousarray(X[:600]))
    mcode = mb.transform(S[:128])
    sc = ml.SparseCoder(m.components_, transform_algorithm="lasso_cd", transform_alpha=0.2)
    scode = sc.transform(S[:128])
    return _fit(dict(dict_learning=_m_transform_h(m.inverse_transform(code)),
                     minibatch=_m_transform_h(mb.inverse_transform(mcode)),
                     coder=_m_transform_h(sc.inverse_transform(scode))))


@lane("x-decomp-eigh-wide-kpca-fit")
def _(ml, X, yc, yr, Xh=None):
    """KernelPCA.fit then transform (the base lane reaches only fit_transform): the dense eigensolve at 72
    rows, rbf, and the fitted spectrum."""
    m = ml.KernelPCA(n_components=4, kernel="rbf", eigen_solver="dense").fit(np.ascontiguousarray(X[:72]))
    return _fit(dict(fit=_h(m.eigenvalues_, m.eigenvectors_), transform=_m_transform_h(m.transform(X[:72]))),
                m, lambda e: (e.transform(np.ascontiguousarray(Xh[:72])),))


@lane("x-isotonic-fit-transform")
def _(ml, X, yc, yr, Xh=None):
    T = ml.IsotonicRegression(y_min=-2.0).fit_transform(np.ascontiguousarray(X[:3000, 3]), yr[:3000])
    return _fit(dict(fit_transform=_m_transform_h(T)))


# ---------------------------------------------------------------- preprocessing

@lane("x-prep-binarizer-fit-transform")
def _(ml, X, yc, yr, Xh=None):
    T = ml.Binarizer(threshold=0.3).fit_transform(np.ascontiguousarray(X[:256]))
    return _fit(dict(fit_transform=_m_transform_h(T)))


@lane("x-prep-maxabs-scaler-fit-transform")
def _(ml, X, yc, yr, Xh=None):
    return _fit(dict(fit_transform=_m_transform_h(ml.MaxAbsScaler().fit_transform(np.ascontiguousarray(X[:4096])))))


@lane("x-prep-robust-scaler-fit-transform")
def _(ml, X, yc, yr, Xh=None):
    return _fit(dict(fit_transform=_m_transform_h(ml.RobustScaler().fit_transform(np.ascontiguousarray(X[:4096])))))


@lane("x-prep-normalizer-fit-transform")
def _(ml, X, yc, yr, Xh=None):
    rows = np.ascontiguousarray(X[:256])
    return _fit({nm: _m_transform_h(ml.Normalizer(norm=nm).fit_transform(rows)) for nm in ("l1", "l2", "max")})


@lane("x-prep-variance-threshold-fit-transform")
def _(ml, X, yc, yr, Xh=None):
    T = ml.VarianceThreshold(threshold=0.3).fit_transform(np.ascontiguousarray(X[:4096]))
    return _fit(dict(fit_transform=_m_transform_h(T)))


@lane("x-prep-simple-imputer-fit-transform")
def _(ml, X, yc, yr, Xh=None):
    Xm = _prep_with_nan(X[:4096])
    return _fit({s: _m_transform_h(ml.SimpleImputer(strategy=s).fit_transform(Xm))
                 for s in ("mean", "median", "most_frequent")})


@lane("x-prep-spline-options-fit-transform")
def _(ml, X, yc, yr, Xh=None):
    """SplineTransformer.fit_transform: the base lane's 'linear' extrapolation and its weighted quantile knots."""
    Xs = _prep_first_four(X[:2000])
    w = (np.arange(Xs.shape[0]) % 4).astype(np.float32)
    lin = ml.SplineTransformer(extrapolation="linear").fit_transform(Xs)
    wq = ml.SplineTransformer(knots="quantile", n_knots=6).fit_transform(Xs, sample_weight=w)
    return _fit(dict(linear=_m_transform_h(lin), weighted_quantile=_m_transform_h(wq)))


@lane("x-prep-encoder-categories-fit-transform")
def _(ml, X, yc, yr, Xh=None):
    """OrdinalEncoder and OneHotEncoder fit_transform with x-prep-encoder-categories' explicit category
    lists (one training value dropped, one unseen added, NaN last) on its first 1000 rows."""
    Xq = _prep_categorical_nan(X[:1000])
    cats = []
    for j in range(Xq.shape[1]):  # glue: the base lane's category lists, per column
        u = np.unique(Xq[:, j])
        num = [float(v) for v in u if v == v]
        num = sorted(num[:1] + num[2:] + [float(np.float32(max(num) + 7.5))])
        cats.append(num + ([float("nan")] if np.isnan(u).any() else []))
    o = ml.OrdinalEncoder(categories=cats, handle_unknown="use_encoded_value", unknown_value=-1).fit_transform(Xq)
    h = ml.OneHotEncoder(categories=cats, handle_unknown="ignore").fit_transform(Xq)
    return _fit(dict(ordinal=_m_transform_h(o), onehot=_m_transform_h(h)))


@lane("x-prep-inverse-transforms-fit-transform")
def _(ml, X, yc, yr, Xh=None):
    """fit_transform of the four x-prep-inverse-transforms estimators with its arguments: the quantile map
    (uniform), the Yeo-Johnson power map, k-means bins and the label binarizer over the gapped labels."""
    rows = np.ascontiguousarray(X[:2000])
    return _fit(dict(quantile=_m_transform_h(ml.QuantileTransformer(n_quantiles=200, random_state=5).fit_transform(rows)),
                     power=_m_transform_h(ml.PowerTransformer().fit_transform(rows)),
                     kbins=_m_transform_h(ml.KBinsDiscretizer(n_bins=6, encode="ordinal",
                                                              strategy="kmeans").fit_transform(rows)),
                     labels=_m_transform_h(ml.LabelBinarizer(neg_label=-1, pos_label=2).fit_transform(
                         _prep_labels(X[:2000])))))


@lane("x-prep-label-encoder-inverse")
def _(ml, X, yc, yr, Xh=None):
    """LabelEncoder.inverse_transform of the base lane's encoder on its own codes."""
    fit = LANES["x-prep-label-encoder"](ml, X, yc, yr, Xh)
    m = fit.est
    if type(m).__name__ != "LabelEncoder":
        raise AssertionError(f"m_transform: the base lane's estimator is {type(m).__name__}, not LabelEncoder")
    y = _prep_labels(X)
    codes = m.transform(y[:256])
    return _fit(dict(inverse=_m_transform_h(m.inverse_transform(codes))), m,
                lambda e: (np.asarray(e.inverse_transform(codes)),))


@lane("x-prep-multilabel-binarizer-fit-calls")
def _(ml, X, yc, yr, Xh=None):
    """MultiLabelBinarizer.fit_transform on the base lane's label sets (first 1000 rows), then
    inverse_transform of that indicator matrix back to label sets (lengths and labels as int64)."""
    ys = _prep_multilabel(X[:1000])
    m = ml.MultiLabelBinarizer()
    Y = m.fit_transform(ys)
    return _fit(dict(fit_transform=_m_transform_h(Y), inverse=_h(*_m_transform_ragged(m.inverse_transform(Y)))))


@lane("x-prep-mi-discrete-fit-transform")
def _(ml, X, yc, yr, Xh=None):
    """SelectKBest(mutual_info_classif, k=3).fit_transform on x-prep-mi-discrete's mixed matrix."""
    Xs, y3 = X[:1200, :6], _prep_three_class(X, yr)[:1200]
    Xm = np.array(Xs, dtype=np.float32)
    Xm[:, :3] = _prep_categorical(Xs[:, :3])
    return _fit(dict(fit_transform=_m_transform_h(ml.SelectKBest(ml.mutual_info_classif, k=3).fit_transform(Xm, y3))))


@lane("x-prep-score-edges-rfe-fit-transform")
def _(ml, X, yc, yr, Xh=None):
    """RFE.fit_transform with x-prep-score-edges' arguments (LDA, five features, step 3, coef_ importances)
    on 2048 rows."""
    Z = np.ascontiguousarray(X[:2048])
    y3 = _prep_three_class(Z, yr[:2048])
    T = ml.RFE(ml.LinearDiscriminantAnalysis(), n_features_to_select=5, step=3,
               importance_getter="coef_").fit_transform(Z, y3)
    return _fit(dict(fit_transform=_m_transform_h(T)))


@lane("x-prep-iterative-imputer-fit")
def _(ml, X, yc, yr, Xh=None):
    """IterativeImputer.fit, then transform (the base lane reaches only fit_transform), with its arguments."""
    Xm = _prep_with_nan(X[:3000, :8])
    m = ml.IterativeImputer(max_iter=4, min_value=-5.0, max_value=5.0).fit(Xm)
    return _fit(dict(transform=_m_transform_h(m.transform(Xm[:256])), n_iter=_h(np.array([m.n_iter_]))),
                m, lambda e: (e.transform(_prep_with_nan(Xh[:256, :8])),))


# ---------------------------------------------------------------- ensembles with a transform

@lane("trees-stacking-clf-fit-transform")
def _(ml, X, yc, yr, Xh=None):
    """StackingClassifier.fit_transform with trees-stacking-clf's main stack (three folds, passthrough) on
    its 7001 rows; the stacked held-out probabilities plus the passthrough columns."""
    rows = 7001
    m = ml.StackingClassifier([("dt", ml.DecisionTreeClassifier(max_depth=3)),
                               ("rf", ml.RandomForestClassifier(n_estimators=2, max_depth=3, random_state=3))],
                              final_estimator=ml.DecisionTreeClassifier(max_depth=3), cv=3, passthrough=True)
    return _fit(dict(fit_transform=_m_transform_h(m.fit_transform(X[:rows], yc[:rows]))))


@lane("trees-voting-reg-fit-transform")
def _(ml, X, yc, yr, Xh=None):
    m = ml.VotingRegressor([("dt", ml.DecisionTreeRegressor(max_depth=5)),
                            ("rf", ml.RandomForestRegressor(n_estimators=4, max_depth=5, random_state=3))],
                           weights=[3.0, 1.0])
    return _fit(dict(fit_transform=_m_transform_h(m.fit_transform(X[:4096], yr[:4096]))))


@lane("trees-random-embedding-fit-transform")
def _(ml, X, yc, yr, Xh=None):
    """RandomTreesEmbedding.fit_transform with trees-random-embedding's arguments on the first 4096 rows
    (the leaf one-hot of a fresh fit)."""
    T = ml.RandomTreesEmbedding(n_estimators=8, max_depth=4, random_state=7).fit_transform(X[:4096])
    return _fit(dict(fit_transform=_m_transform_h(T)))


@lane("m-voting-clf-calls")
def _(ml, X, yc, yr, Xh=None):
    """VotingClassifier.fit_transform (soft voting: the members' probabilities) and .score of that fit
    (the table's base, trees-stacking-clf, holds no voting classifier): a tree and a two-tree forest on the
    first 4096 rows."""
    v = ml.VotingClassifier([("dt", ml.DecisionTreeClassifier(max_depth=3)),
                             ("rf", ml.RandomForestClassifier(n_estimators=2, max_depth=3, random_state=3))],
                            voting="soft")
    T = v.fit_transform(X[:4096], yc[:4096])
    return _fit(dict(fit_transform=_m_transform_h(T), score=_h(np.float64(v.score(X[:256], yc[:256])))),
                v, lambda e: (e.predict_proba(Xh[:256]),))


@lane("m-stacking-reg-calls")
def _(ml, X, yc, yr, Xh=None):
    """StackingRegressor.fit_transform, .transform and .score (the table's base, trees-oob-cv-link, holds a
    calibrated classifier): trees-oob-cv-link's stack (a depth-4 member, explicit (train, test) pairs, a
    depth-3 final tree) on the first 2000 rows."""
    n = 2000
    pairs = [(np.arange(n)[np.arange(n) % 2 != f], np.arange(n)[np.arange(n) % 2 == f]) for f in (0, 1)]
    st = ml.StackingRegressor([("dt", ml.DecisionTreeRegressor(max_depth=4))],
                              final_estimator=ml.DecisionTreeRegressor(max_depth=3), cv=pairs)
    T = st.fit_transform(X[:n], yr[:n])
    return _fit(dict(fit_transform=_m_transform_h(T), transform=_m_transform_h(st.transform(X[:256])),
                     score=_h(np.float64(st.score(X[:256], yr[:256])))),
                st, lambda e: (e.transform(Xh[:256]),))


# ---------------------------------------------------------------- batch declarations (lane vdv-fin, 2026-11-24)
# fit_transform, fit_predict, partial_fit and the inverse of a fresh fit's own output are fit endpoints: the
# answer for a row depends on every row the call fits on, so a row alone or a split is a different fit, not a
# smaller batch of the same call. These lanes return no fitted estimator (`_fit(parts)`), and the transform of
# each fitted class keeps its batch part on the base lane the lane copies.
_batch_decl("n/a:fit-endpoint (fit_transform, fit_predict, partial_fit or the inverse of a fresh fit's own "
            "output: each row's answer depends on every row the call fits on, so a row alone is a different fit; "
            "the lane holds no estimator, and the fitted class's transform keeps its batch part on the base lane)",
            "x-cluster-bisecting-kmeans-fit-calls",
            "x-cluster-minibatch-kmeans-fit-calls",
            "kernel-ridge-laplacian-fit-transform",
            "rbf-sampler-fit-transform",
            "x-neighbors-poly-sketch-fit-transform",
            "x-neighbors-additive-chi2-fit-transform",
            "x-neighbors-skewed-chi2-fit-transform",
            "x-neighbors-knn-imputer-fit-transform",
            "umap-fit-transform",
            "spectral-embedding-fit-transform",
            "tsvd-inverse-fit-transform",
            "x-decomp-ipca-fit-calls",
            "x-decomp-lda-fit-calls",
            "x-decomp-pls-fit-calls",
            "x-decomp-sparse-pca-fit-calls",
            "x-decomp-srp-fit-calls",
            "x-decomp-dict-learning-inverse",
            "x-isotonic-fit-transform",
            "x-prep-binarizer-fit-transform",
            "x-prep-maxabs-scaler-fit-transform",
            "x-prep-robust-scaler-fit-transform",
            "x-prep-normalizer-fit-transform",
            "x-prep-variance-threshold-fit-transform",
            "x-prep-simple-imputer-fit-transform",
            "x-prep-spline-options-fit-transform",
            "x-prep-encoder-categories-fit-transform",
            "x-prep-inverse-transforms-fit-transform",
            "x-prep-multilabel-binarizer-fit-calls",
            "x-prep-mi-discrete-fit-transform",
            "x-prep-score-edges-rfe-fit-transform",
            "trees-stacking-clf-fit-transform",
            "trees-voting-reg-fit-transform",
            "trees-random-embedding-fit-transform")


def _m_transform_batch_label_inverse(ml, e, Xh):
    """LabelEncoder.inverse_transform on the codes of the held-out labels the encoder knows (the base lane's
    `_prep_batch_label_encoder` rows): a code's label is a lookup of that code alone."""
    labels = _prep_labels(Xh)
    known = labels[np.isin(labels, np.asarray(e.classes_))][:256]
    codes = np.asarray(e.transform(known))
    return [_BatchRows("inverse_transform", codes, lambda r: (np.asarray(e.inverse_transform(r)),))]


# The row-wise calls of the lanes that hold a fitted estimator, on the held-out rows their probe reads. The
# derived inputs go through a lambda so the prep.py helpers are looked up when the part runs (prep.py loads
# after this file).
_batch_decl(_rows_calls("transform", sl=slice(0, 72)), "x-decomp-eigh-wide-kpca-fit")
_batch_decl(_m_transform_batch_label_inverse, "x-prep-label-encoder-inverse")
_batch_decl(_rows_calls("transform", sl=slice(0, 256), prep=lambda Xh: _prep_nan_first_eight(Xh)),
            "x-prep-iterative-imputer-fit")
_batch_decl(_rows_calls("predict_proba", sl=slice(0, 256)), "m-voting-clf-calls")
_batch_decl(_rows_calls("transform", sl=slice(0, 256)), "m-stacking-reg-calls")
