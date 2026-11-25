# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# ESTIMATOR PREDICT-FAMILY LANES (lane vdv-cov-est, 2026-11-24; plan docs/plans/VERIFY_THROUGH_DEVICE.md item
# C): `fit_predict`, `forecast`, and the remaining estimator methods with a numerical result that the inventory
# (docs/plans/verify_through_device/endpoints.tsv) left unmapped "OWN LANE": decision functions, joint and log
# probabilities, score_samples, the tree read-backs (apply, predict_with_leaves, staged_predict, a saved tree),
# the GBDT read-backs, the time-series components, and a score where the same base holds other methods.
#
# Three shapes, each said in the lane's name and docstring:
#   `<base>-fit-calls`  fit_predict / forecast are fit endpoints: a FRESH fit with the base lane's constructor
#                       arguments on its fixture slice, never a refit of the base lane's object;
#   `<base>-methods`    calls the base lane, checks its fitted estimator is the endpoint's class, digests the
#                       methods on the base fit's own rows;
#   `m-<class>-...`     a full lane: the table's base lane holds no fitted estimator of the class (it returns
#                       none, another class, or is a many-fit lane a read-back does not need), so the lane fits
#                       one small model with the base lane's arguments.
#
# Device-agnostic (the vd-verify contract): no `device=`, no vendor read. Every lane is an independent unit
# (own fixture slice and model; an extension re-runs its base lane). Every reference is OWED to the release's
# one verification. Helpers carry the `_m_predict_` prefix; lane names are literal; nothing is rebound.


def _m_predict_est(fit, cls):
    """The base lane's fitted estimator, refused unless it is a `cls`."""
    est = fit.est
    if type(est).__name__ != cls:
        raise AssertionError(f"m_predict: the base lane's estimator is {type(est).__name__}, not {cls}")
    return est


def _m_predict_h(*outs):
    """`_h` over each output's arrays: tuples and lists (latent mean and variance, approximate_predict) are
    flattened in order, a dict by sorted key, a Python float becomes float64."""
    arrays = []
    for out in outs:
        if isinstance(out, dict):
            arrays.extend(np.asarray(out[k]) for k in sorted(out))
        elif isinstance(out, (tuple, list)):
            arrays.extend(np.asarray(a) for a in out)
        else:
            arrays.append(np.asarray(out))
    return _h(*arrays)


def _m_predict_forest_parts(m, rows, y=None):
    """The forest protocol's read-backs on `rows` (ForestProtocol: the native tree walk): leaf ids, the
    ordered raw scores with leaf ids, the eager prefix means, and the score when `y` is given."""
    parts = dict(apply=_m_predict_h(m.apply(rows)), with_leaves=_m_predict_h(m.predict_with_leaves(rows)),
                 staged=_m_predict_h(m.staged_predict(rows)))
    if y is not None:
        parts["score"] = _h(np.float64(m.score(rows, y)))
    return parts


def _m_predict_tree_file(m, rows):
    """A fitted tree's `save` round trip: every numeric member of the written `.npz`, by sorted name, and
    the loaded tree's predictions on `rows`."""
    with tempfile.TemporaryDirectory(prefix="ib-m-predict-tree-") as d:
        path = os.path.join(d, "tree.npz")
        m.save(path)
        with np.load(path, allow_pickle=False) as z:
            members = [np.array(z[k]) for k in sorted(z.files) if np.asarray(z[k]).dtype.kind in "biuf"]
        back = type(m).load(path)
    return _h(*members, np.asarray(back.predict(rows)))


# ---------------------------------------------------------------- fit_predict: fresh fits

@lane("agglomerative-fit-predict")
def _(ml, X, yc, yr, Xh=None):
    return _fit(dict(fit_predict=_m_predict_h(
        ml.AgglomerativeClustering(n_clusters=4).fit_predict(np.ascontiguousarray(X[:2000, :4])))))


@lane("dbscan-fit-predict")
def _(ml, X, yc, yr, Xh=None):
    return _fit(dict(fit_predict=_m_predict_h(
        ml.DBSCAN(eps=0.9, min_samples=5).fit_predict(np.ascontiguousarray(X[:6000, :4])))))


@lane("hdbscan-fit-predict")
def _(ml, X, yc, yr, Xh=None):
    return _fit(dict(fit_predict=_m_predict_h(
        ml.HDBSCAN(min_cluster_size=5).fit_predict(np.ascontiguousarray(X[:4000, :4])))))


@lane("spectral-fit-predict")
def _(ml, X, yc, yr, Xh=None):
    return _fit(dict(fit_predict=_m_predict_h(
        ml.SpectralClustering(n_clusters=4, random_state=3).fit_predict(np.ascontiguousarray(X[:512, :4])))))


@lane("gmm-fit-predict")
def _(ml, X, yc, yr, Xh=None):
    return _fit(dict(fit_predict=_m_predict_h(
        ml.GaussianMixture(n_components=4, max_iter=30, random_state=3).fit_predict(
            np.ascontiguousarray(X[:6000, :4])))))


@lane("iforest-fit-predict")
def _(ml, X, yc, yr, Xh=None):
    return _fit(dict(fit_predict=_m_predict_h(ml.IsolationForest(n_estimators=16, random_state=5).fit_predict(X))))


@lane("m-kmeans-fit-predict")
def _(ml, X, yc, yr, Xh=None):
    """KMeans.fit_predict with public-set-params' refit arguments on its 256 x 6 rows."""
    rows = np.ascontiguousarray(X[:256, :6], dtype=np.float32)
    return _fit(dict(fit_predict=_m_predict_h(
        ml.KMeans(n_clusters=4, n_init=1, max_iter=20, random_state=1).fit_predict(rows))))


@lane("x-cluster-affinity-propagation-fit-predict")
def _(ml, X, yc, yr, Xh=None):
    return _fit(dict(fit_predict=_m_predict_h(
        ml.AffinityPropagation(damping=0.7, random_state=3).fit_predict(np.ascontiguousarray(X[:400, 1:5])))))


@lane("x-cluster-bgmm-fit-predict")
def _(ml, X, yc, yr, Xh=None):
    return _fit(dict(fit_predict=_m_predict_h(
        ml.BayesianGaussianMixture(n_components=5, max_iter=30, random_state=3).fit_predict(
            np.ascontiguousarray(X[:2000, 1:5])))))


@lane("x-cluster-meanshift-fit-predict")
def _(ml, X, yc, yr, Xh=None):
    return _fit(dict(fit_predict=_m_predict_h(
        ml.MeanShift(cluster_all=False).fit_predict(np.ascontiguousarray(X[:1200, 1:4])))))


@lane("x-cluster-optics-fit-predict")
def _(ml, X, yc, yr, Xh=None):
    """OPTICS.fit_predict labels, plus the core and reachability distances that same call fitted.

    Lane w3-fix-10-s2, 2026-11-25: the labels alone cannot move under a one-ulp distance fault
    (coverage-gate-x_cluster-sq_dist_rows-cpu missed this lane); the two float vectors the call
    already computed carry the euclidean sqdist -> sqrt -> kth chain into the digest."""
    m = ml.OPTICS(min_samples=8)
    labels = m.fit_predict(np.ascontiguousarray(X[:1500, 1:5]))
    return _fit(dict(fit_predict=_m_predict_h(labels),
                     core=_m_predict_h(m.core_distances_), reach=_m_predict_h(m.reachability_)))


@lane("x-neighbors-louvain-fit-predict")
def _(ml, X, yc, yr, Xh=None):
    return _fit(dict(fit_predict=_m_predict_h(ml.Louvain(resolution=1.0).fit_predict(_neighbors_graph(X)))))


@lane("x-neighbors-ocsvm-fit-predict")
def _(ml, X, yc, yr, Xh=None):
    return _fit(dict(fit_predict=_m_predict_h(ml.OneClassSVM(nu=0.2).fit_predict(np.ascontiguousarray(X[:256])))))


@lane("x-decomp-robust-cov-fit-predict")
def _(ml, X, yc, yr, Xh=None):
    S = np.ascontiguousarray(X[:400, :6])
    return _fit(dict(fit_predict=_m_predict_h(
        ml.EllipticEnvelope(contamination=0.05, random_state=1, support_fraction=0.7).fit_predict(S))))


# ---------------------------------------------------------------- forecast: fit and forecast in one call

@lane("sequence-theta-forecast")
def _(ml, X, yc, yr, Xh=None):
    """AutoTheta, DynamicOptimizedTheta and Theta .forecast(y, 18) on sequence-theta's four series."""
    t = np.arange(96, dtype=np.float32)
    wave = np.sin(t * np.float32(2 * np.pi / 12)).astype(np.float32)
    c = np.ascontiguousarray(X[:96, 7], dtype=np.float32)
    y = np.stack([np.float32(20.0) + np.float32(4.0) * wave + np.float32(0.3) * c + np.float32(0.05) * t,
                  np.float32(4.0) * wave + np.float32(0.3) * c,
                  np.cumsum(c, dtype=np.float32), c]).astype(np.float32)
    return _fit(dict(auto=_m_predict_h(ml.AutoTheta(season_length=12).forecast(y, 18)),
                     dotm=_m_predict_h(ml.DynamicOptimizedTheta(season_length=12, alpha=0.3).forecast(y, 18)),
                     theta=_m_predict_h(ml.Theta(season_length=12).forecast(y, 18))))


@lane("public-theta-variants-forecast")
def _(ml, X, yc, yr, Xh=None):
    """DynamicTheta and OptimizedTheta .forecast(y, 9) on public-theta-variants' three series."""
    y = np.ascontiguousarray(X[:96, :3].T, dtype=np.float32)
    return _fit(dict(dynamic=_m_predict_h(ml.DynamicTheta(season_length=1, alpha=0.3).forecast(y, 9)),
                     optimized=_m_predict_h(ml.OptimizedTheta(season_length=1, alpha=0.3).forecast(y, 9))))


@lane("sequence-croston-forecast")
def _(ml, X, yc, yr, Xh=None):
    """CrostonClassic, CrostonOptimized and CrostonSBA .forecast(y, 4) on sequence-croston's six
    intermittent series."""
    c = np.ascontiguousarray(X[:120, :6].T, dtype=np.float32)
    y = np.where(c > np.float32(0.8), c + np.float32(1.0), np.float32(0.0)).astype(np.float32)
    y[4] = np.float32(0.0)
    y[5] = np.where(c[5] < np.float32(-1.0), c[5], y[5]).astype(np.float32)
    return _fit(dict(classic=_m_predict_h(ml.CrostonClassic().forecast(y, 4)),
                     optimized=_m_predict_h(ml.CrostonOptimized().forecast(y, 4)),
                     sba=_m_predict_h(ml.CrostonSBA().forecast(y, 4))))


@lane("sequence-ets-forecast")
def _(ml, X, yc, yr, Xh=None):
    """DampedETS and ETS(AAN) .forecast(y, 12) on sequence-ets' four 48-observation series, and the
    seasonal ETS(AAA) on the same series plus its wave."""
    t = np.arange(48, dtype=np.float32)
    c = np.ascontiguousarray(X[:48, 9], dtype=np.float32)
    y = np.stack([np.float32(5.0) + np.float32(0.3) * t + c, np.cumsum(c, dtype=np.float32),
                  np.float32(40.0) + c, c]).astype(np.float32)
    wave = (np.float32(3.0) * np.sin(np.float32(2.0 * np.pi / 12.0) * t)).astype(np.float32)
    ys = (y + wave).astype(np.float32)
    return _fit(dict(damped=_m_predict_h(ml.DampedETS().forecast(y, 12)),
                     aan=_m_predict_h(ml.ETS(model="AAN", damped=False).forecast(y, 12)),
                     aaa=_m_predict_h(ml.ETS(season_length=12, model="AAA", damped=False).forecast(ys, 12))))


@lane("m-autoarima-predict")
def _(ml, X, yc, yr, Xh=None):
    """AutoARIMA.predict (base sequence-autoarima keeps no estimator): its search and fit on its four
    200-observation series, then in-sample predictions and predictions through 12 steps past the end."""
    raw = np.ascontiguousarray(X[:200, 3:5].T, dtype=np.float32)
    walk = np.cumsum(raw, axis=1, dtype=np.float32)
    y = np.ascontiguousarray(np.concatenate([raw, walk]), dtype=np.float32)
    m = ml.AutoARIMA(y).search(d=range(2), p=range(2), q=range(2))
    m.fit()
    return _fit(dict(in_sample=_m_predict_h(m.predict()), through=_m_predict_h(m.predict(180, 212))))


@lane("m-prophet-add-seasonality")
def _(ml, X, yc, yr, Xh=None):
    """ProphetForecaster.add_seasonality (base sequence-prophet keeps no estimator): sequence-prophet's
    additive model over its 120 days with a custom 30.5-day seasonality of order 3 added before the fit;
    the fitted parameters and the 21-day forecast."""
    t = np.arange(120, dtype=np.float64) + 19000.0
    c = np.ascontiguousarray(X[:120, 13:15].T, dtype=np.float32)
    k = np.where(np.arange(120) < 60, np.float32(0.05), np.float32(-0.02)).astype(np.float32)
    trend = np.cumsum(k, dtype=np.float32) + np.float32(10.0)
    week = np.sin(np.arange(120, dtype=np.float32) * np.float32(2 * np.pi / 7)).astype(np.float32)
    hol = (np.arange(141) % 30 == 5).astype(np.float32)
    y = (trend + week + np.float32(3.0) * hol[:120] + np.float32(0.2) * c).astype(np.float32)
    tf = np.arange(120, 141, dtype=np.float64) + 19000.0
    m = ml.ProphetForecaster()
    m.add_seasonality("monthly", 30.5, 3)
    m.fit(t, y, holidays=hol[:120, None])
    return _fit(dict(params=_m_predict_h(m.params_), forecast=_m_predict_h(m.predict(tf, holidays=hol[120:, None]))))


@lane("holtwinters-methods")
def _(ml, X, yc, yr, Xh=None):
    """ExponentialSmoothing.score (the SSE), get_level, get_trend, get_season and predict(0, 152) (in
    sample, NaN before two periods, then the forecast) of the holtwinters lane's fit."""
    m = _m_predict_est(LANES["holtwinters"](ml, X, yc, yr, Xh), "ExponentialSmoothing")
    return _fit(dict(score=_m_predict_h(m.score()), level=_m_predict_h(m.get_level()),
                     trend=_m_predict_h(m.get_trend()), season=_m_predict_h(m.get_season()),
                     predict=_m_predict_h(m.predict(0, 152))),
                m, lambda e: (np.asarray(e.predict(128, 128 + FORECAST_HORIZON)),))


# ---------------------------------------------------------------- probabilistic models and density

@lane("gmm-methods")
def _(ml, X, yc, yr, Xh=None):
    """GaussianMixture.score, aic and bic of the gmm lane's fit on 256 of its rows."""
    m = _m_predict_est(LANES["gmm"](ml, X, yc, yr, Xh), "GaussianMixture")
    rows = np.ascontiguousarray(X[:256, :4])
    return _fit(dict(score=_h(np.float64(m.score(rows))), aic=_h(np.float64(m.aic(rows))),
                     bic=_h(np.float64(m.bic(rows)))),
                m, lambda e: (np.float64(e.score(np.ascontiguousarray(Xh[:256, :4]))),))


@lane("gpc-methods")
def _(ml, X, yc, yr, Xh=None):
    """GaussianProcessClassifier.score, latent_mean_and_variance and log_marginal_likelihood() (the only
    supported theta) of the gpc lane's fit, on its query rows 256:320."""
    m = _m_predict_est(LANES["gpc"](ml, X, yc, yr, Xh), "GaussianProcessClassifier")
    q = np.ascontiguousarray(X[256:320, :4])
    return _fit(dict(score=_h(np.float64(m.score(q, yc[256:320]))), latent=_m_predict_h(m.latent_mean_and_variance(q)),
                     lml=_h(np.float64(m.log_marginal_likelihood()))),
                m, lambda e: tuple(np.asarray(a) for a in e.latent_mean_and_variance(np.ascontiguousarray(Xh[:64, :4]))))


@lane("x-decomp-lda-perplexity")
def _(ml, X, yc, yr, Xh=None):
    m = _m_predict_est(LANES["x-decomp-lda"](ml, X, yc, yr, Xh), "LatentDirichletAllocation")
    C = np.minimum(np.floor(np.abs(X[:128]) * np.float32(3)).astype(np.float32), np.float32(50))
    return _fit(dict(perplexity=_h(np.float64(m.perplexity(C)))), m,
                lambda e: (np.float64(e.perplexity(np.minimum(np.floor(np.abs(Xh[:128]) * np.float32(3)),
                                                              np.float32(50)).astype(np.float32))),))


@lane("m-isomap-reconstruction-error")
def _(ml, X, yc, yr, Xh=None):
    """Isomap.reconstruction_error (base x-decomp-manifold is a many-fit lane): its first Isomap, eight
    neighbors and three components on 160 rows."""
    m = ml.Isomap(n_neighbors=8, n_components=3).fit(np.ascontiguousarray(X[:160]))
    return _fit(dict(error=_h(np.float64(m.reconstruction_error())), embedding=_m_predict_h(m.embedding_)))


@lane("cholesky-profile-jitter")
def _(ml, X, yc, yr, Xh=None):
    """Cholesky.profile_jitter of the cholesky lane's factor: the profile's fixed ridge (float64 bytes)."""
    c = _m_predict_est(LANES["cholesky"](ml, X, yc, yr, Xh), "Cholesky")
    return _fit(dict(profile_jitter=_h(np.float64(c.profile_jitter()))))


# ---------------------------------------------------------------- discriminants and naive Bayes

@lane("x-prep-lda-methods")
def _(ml, X, yc, yr, Xh=None):
    """LinearDiscriminantAnalysis.score, predict_joint_log_proba and predict_log_proba of x-prep-lda's
    three-class fit."""
    m = _m_predict_est(LANES["x-prep-lda"](ml, X, yc, yr, Xh), "LinearDiscriminantAnalysis")
    y3 = _prep_three_class(X, yr)[:256]
    return _fit(dict(score=_h(np.float64(m.score(X[:256], y3))), joint=_m_predict_h(m.predict_joint_log_proba(X[:256])),
                     log_proba=_m_predict_h(m.predict_log_proba(X[:256]))),
                m, lambda e: (np.asarray(e.predict_joint_log_proba(Xh[:256])),))


@lane("x-prep-qda-methods")
def _(ml, X, yc, yr, Xh=None):
    m = _m_predict_est(LANES["x-prep-qda"](ml, X, yc, yr, Xh), "QuadraticDiscriminantAnalysis")
    y3 = _prep_three_class(X, yr)[:256]
    return _fit(dict(score=_h(np.float64(m.score(X[:256], y3))), joint=_m_predict_h(m.predict_joint_log_proba(X[:256])),
                     log_proba=_m_predict_h(m.predict_log_proba(X[:256]))),
                m, lambda e: (np.asarray(e.predict_joint_log_proba(Xh[:256])),))


@lane("x-prep-gaussian-nb-methods")
def _(ml, X, yc, yr, Xh=None):
    m = _m_predict_est(LANES["x-prep-gaussian-nb"](ml, X, yc, yr, Xh), "GaussianNB")
    return _fit(dict(score=_h(np.float64(m.score(X[:256], yc[:256]))), joint=_m_predict_h(m.predict_joint_log_proba(X[:256]))),
                m, lambda e: (np.asarray(e.predict_joint_log_proba(Xh[:256])),))


@lane("x-prep-bernoulli-nb-methods")
def _(ml, X, yc, yr, Xh=None):
    m = _m_predict_est(LANES["x-prep-bernoulli-nb"](ml, X, yc, yr, Xh), "BernoulliNB")
    return _fit(dict(score=_h(np.float64(m.score(X[:256], yc[:256]))), joint=_m_predict_h(m.predict_joint_log_proba(X[:256]))),
                m, lambda e: (np.asarray(e.predict_joint_log_proba(Xh[:256])),))


@lane("x-prep-multinomial-nb-methods")
def _(ml, X, yc, yr, Xh=None):
    m = _m_predict_est(LANES["x-prep-multinomial-nb"](ml, X, yc, yr, Xh), "MultinomialNB")
    A = _prep_abs(X[:256])
    return _fit(dict(score=_h(np.float64(m.score(A, yc[:256]))), joint=_m_predict_h(m.predict_joint_log_proba(A))),
                m, lambda e: (np.asarray(e.predict_joint_log_proba(_prep_abs(Xh[:256]))),))


@lane("x-prep-complement-nb-methods")
def _(ml, X, yc, yr, Xh=None):
    m = _m_predict_est(LANES["x-prep-complement-nb"](ml, X, yc, yr, Xh), "ComplementNB")
    A = _prep_abs(X[:256])
    y3 = _prep_three_class(X, yr)[:256]
    return _fit(dict(score=_h(np.float64(m.score(A, y3))), joint=_m_predict_h(m.predict_joint_log_proba(A))),
                m, lambda e: (np.asarray(e.predict_joint_log_proba(_prep_abs(Xh[:256]))),))


@lane("x-prep-categorical-nb-methods")
def _(ml, X, yc, yr, Xh=None):
    m = _m_predict_est(LANES["x-prep-categorical-nb"](ml, X, yc, yr, Xh), "CategoricalNB")
    Xc = _prep_cat_codes(X)[:256]
    y3 = _prep_three_class(X, yr)[:256]
    return _fit(dict(score=_h(np.float64(m.score(Xc, y3))), joint=_m_predict_h(m.predict_joint_log_proba(Xc))),
                m, lambda e: (np.asarray(e.predict_joint_log_proba(_prep_cat_codes(Xh)[:256])),))


@lane("x-prep-score-edges-rfe-methods")
def _(ml, X, yc, yr, Xh=None):
    """RFE.score, decision_function and predict_proba of x-prep-score-edges' RFE over LDA (fitted on its
    4500 rows and three-class labels)."""
    m = _m_predict_est(LANES["x-prep-score-edges"](ml, X, yc, yr, Xh), "RFE")
    y3 = _prep_three_class(X[:4500], yr[:4500])[:256]
    return _fit(dict(score=_h(np.float64(m.score(X[:256], y3))), decision=_m_predict_h(m.decision_function(X[:256])),
                     proba=_m_predict_h(m.predict_proba(X[:256]))),
                m, lambda e: (np.asarray(e.decision_function(Xh[:256])), np.asarray(e.predict_proba(Xh[:256]))))


# ---------------------------------------------------------------- outlier scores and decision functions

@lane("iforest-decision-function")
def _(ml, X, yc, yr, Xh=None):
    m = _m_predict_est(LANES["iforest"](ml, X, yc, yr, Xh), "IsolationForest")
    return _fit(dict(decision=_m_predict_h(m.decision_function(X[:512]))), m,
                lambda e: (np.asarray(e.decision_function(Xh[:512])),))


@lane("x-neighbors-lof-decision-function")
def _(ml, X, yc, yr, Xh=None):
    # the base lane's estimator is its novelty=True fit, the one with decision_function
    m = _m_predict_est(LANES["x-neighbors-lof"](ml, X, yc, yr, Xh), "LocalOutlierFactor")
    return _fit(dict(decision=_m_predict_h(m.decision_function(X[512:768]))), m,
                lambda e: (np.asarray(e.decision_function(Xh[:256])),))


@lane("x-neighbors-ocsvm-score-samples")
def _(ml, X, yc, yr, Xh=None):
    m = _m_predict_est(LANES["x-neighbors-ocsvm"](ml, X, yc, yr, Xh), "OneClassSVM")
    return _fit(dict(score_samples=_m_predict_h(m.score_samples(X[:256]))), m,
                lambda e: (np.asarray(e.score_samples(Xh[:256])),))


@lane("x-sgd-ocsvm-score-samples")
def _(ml, X, yc, yr, Xh=None):
    m = _m_predict_est(LANES["x-sgd-ocsvm"](ml, X, yc, yr, Xh), "SGDOneClassSVM")
    return _fit(dict(score_samples=_m_predict_h(m.score_samples(X[:256]))), m,
                lambda e: (np.asarray(e.score_samples(Xh[:256])),))


@lane("m-robust-cov-methods")
def _(ml, X, yc, yr, Xh=None):
    """MinCovDet.get_precision, EllipticEnvelope.get_precision and EllipticEnvelope.score_samples (base
    x-decomp-robust-cov holds the envelope only): its two fits on its 400 x 6 rows."""
    S = np.ascontiguousarray(X[:400, :6])
    m = ml.MinCovDet(random_state=0).fit(S)
    e = ml.EllipticEnvelope(contamination=0.05, random_state=1, support_fraction=0.7).fit(S)
    return _fit(dict(mcd_precision=_m_predict_h(m.get_precision()), ee_precision=_m_predict_h(e.get_precision()),
                     ee_score_samples=_m_predict_h(e.score_samples(S[:128]))),
                e, lambda est: (np.asarray(est.score_samples(np.ascontiguousarray(Xh[:256, :6]))),))


@lane("trees-adaboost-clf-methods")
def _(ml, X, yc, yr, Xh=None):
    m = _m_predict_est(LANES["trees-adaboost-clf"](ml, X, yc, yr, Xh), "AdaBoostClassifier")
    return _fit(dict(score=_h(np.float64(m.score(X[:256], yc[:256]))), decision=_m_predict_h(m.decision_function(X[:256]))),
                m, lambda e: (np.asarray(e.decision_function(Xh[:256])),))


@lane("x-isotonic-methods")
def _(ml, X, yc, yr, Xh=None):
    """IsotonicRegression.score and transform of x-isotonic's fit (column 3, y_min=-2)."""
    m = _m_predict_est(LANES["x-isotonic"](ml, X, yc, yr, Xh), "IsotonicRegression")
    q = np.ascontiguousarray(X[:256, 3])
    return _fit(dict(score=_h(np.float64(m.score(q, yr[:256]))), transform=_m_predict_h(m.transform(q))),
                m, lambda e: (np.asarray(e.transform(np.ascontiguousarray(Xh[:256, 3]) * np.float32(1.5))),))


@lane("trees-voting-reg-methods")
def _(ml, X, yc, yr, Xh=None):
    m = _m_predict_est(LANES["trees-voting-reg"](ml, X, yc, yr, Xh), "VotingRegressor")
    return _fit(dict(score=_h(np.float64(m.score(X[:256], yr[:256]))), transform=_m_predict_h(m.transform(X[:256]))),
                m, lambda e: (np.asarray(e.transform(Xh[:256])),))


@lane("m-gridsearch-methods")
def _(ml, X, yc, yr, Xh=None):
    """model_selection.GridSearchCV.score and .transform (the table's base, public-scorer-factories, refits
    nothing): a three-fold search over LinearDiscriminantAnalysis' tol on 300 rows and three classes, refit
    on the best; the refit model's accuracy and its discriminant projection."""
    ms = ml.model_selection   # through the view: the running side (vdv-fragfix)
    rows = np.ascontiguousarray(X[:300])
    y3 = _prep_three_class(X[:300], yr[:300])
    s = ms.GridSearchCV(ml.LinearDiscriminantAnalysis(), {"tol": [1e-4, 1e-3]}, cv=3, n_jobs=1,
                        error_score="raise").fit(rows, y3)
    return _fit(dict(score=_h(np.float64(s.score(rows[:128], y3[:128]))), transform=_m_predict_h(s.transform(rows[:128]))),
                s.best_estimator_, lambda e: (np.asarray(e.transform(Xh[:128])),))


# ---------------------------------------------------------------- tree read-backs

@lane("et-clf-methods")
def _(ml, X, yc, yr, Xh=None):
    """ExtraTreesClassifier apply, predict_with_leaves, staged_predict and score of the et-clf lane's fit."""
    m = _m_predict_est(LANES["et-clf"](ml, X, yc, yr, Xh), "ExtraTreesClassifier")
    return _fit(_m_predict_forest_parts(m, X[:256], yc[:256]), m, lambda e: (np.asarray(e.apply(Xh[:256])),))


@lane("trees-dt-clf-methods")
def _(ml, X, yc, yr, Xh=None):
    """DecisionTreeClassifier apply, predict_with_leaves, staged_predict, score and the save round trip of
    the trees-dt-clf lane's fit."""
    m = _m_predict_est(LANES["trees-dt-clf"](ml, X, yc, yr, Xh), "DecisionTreeClassifier")
    parts = _m_predict_forest_parts(m, X[:256], yc[:256])
    parts["save"] = _m_predict_tree_file(m, X[:256])
    return _fit(parts, m, lambda e: (np.asarray(e.apply(Xh[:256])),))


@lane("m-dt-reg-methods")
def _(ml, X, yc, yr, Xh=None):
    """DecisionTreeRegressor apply, predict_with_leaves, staged_predict, score, get_depth and the save round
    trip (base trees-dt-random holds its classifier, and its random-splitter regressor refuses `save` by
    name): one best-split regressor, depth 5 (32 leaves at most), on the first 4096 rows.
    Lane w3-fix-9-s1 (2026-11-25): it asked max_leaf_nodes=24 with the default splitter='best', which the
    product refuses by name (bindings/trees_public_forest_door.mojo _rf_arguments: best-first growth lives on
    the random splitter's builder), so its clean run REFUSED on every side; the leaf bound is max_depth now."""
    m = ml.DecisionTreeRegressor(max_depth=5, random_state=5).fit(X[:4096], yr[:4096])
    parts = _m_predict_forest_parts(m, X[:256], yr[:256])
    parts["depth"] = _h(np.int64(m.get_depth()))
    parts["save"] = _m_predict_tree_file(m, X[:256])
    return _fit(parts, m, lambda e: (np.asarray(e.apply(Xh[:256])),))


@lane("m-et-reg-methods")
def _(ml, X, yc, yr, Xh=None):
    """ExtraTreesRegressor apply, predict_with_leaves, staged_predict and score (base
    saved-model-host-infer-trees keeps no estimator): its regressor's arguments on the first 4096 rows."""
    m = ml.ExtraTreesRegressor(n_estimators=16, max_depth=8, random_state=7).fit(X[:4096], yr[:4096])
    return _fit(_m_predict_forest_parts(m, X[:256], yr[:256]), m, lambda e: (np.asarray(e.apply(Xh[:256])),))


@lane("m-rf-methods")
def _(ml, X, yc, yr, Xh=None):
    """RandomForestClassifier and RandomForestRegressor apply, predict_with_leaves and staged_predict (base
    saved-model-host-infer holds a HostForest reader): its two forests' arguments on its first 4096 rows."""
    c = ml.RandomForestClassifier(n_estimators=16, max_depth=8, random_state=7).fit(X[:4096], yc[:4096])
    r = ml.RandomForestRegressor(n_estimators=16, max_depth=8, random_state=7).fit(X[:4096], yr[:4096])
    parts = {"clf_" + k: v for k, v in _m_predict_forest_parts(c, X[:256]).items()}
    parts.update({"reg_" + k: v for k, v in _m_predict_forest_parts(r, X[:256]).items()})
    return _fit(parts, c, lambda e: (np.asarray(e.apply(Xh[:256])),))


@lane("m-gbdt-readbacks")
def _(ml, X, yc, yr, Xh=None):
    """GradientBoosting.predict_classes, get_tree_leaf_counts and get_leaf_values, and the leaf read-backs of
    OrderedRMSE and ExperimentalTwoLevelFeatureFreq (their tables' base lanes, gbdt-cat-bootstrap,
    gbdt-binary-columns and gbdt-weighted-special, are many-fit lanes): a Logloss GradientBoosting at the
    recorded GBDT options, an OrderedRMSE over the fixture's hashed permutation and the FeatureFreq tree over
    the coded fixture, each on the first 2000 rows."""
    Xr, ycr, yrr = np.ascontiguousarray(X[:2000]), yc[:2000], yr[:2000]
    g = _gbdt(ml.GradientBoosting, n_estimators=8, max_depth=4, loss="Logloss").fit(Xr, ycr)
    perm = np.argsort(_hashed_uniform(Xr.shape[0], 1, "ordered-permutation").reshape(-1), kind="stable")
    o = ml.OrderedRMSE(n_estimators=8, max_depth=4).fit(Xr, yrr, permutation=perm)
    f = ml.ExperimentalTwoLevelFeatureFreq(sources=[0, 1], random_state=7).fit(_coded(Xr), yrr)
    return _fit(dict(classes=_m_predict_h(g.predict_classes(Xr[:256])),
                     gbdt_leaves=_m_predict_h(g.get_tree_leaf_counts(), g.get_leaf_values()),
                     ordered_leaves=_m_predict_h(o.get_tree_leaf_counts(), o.get_leaf_values()),
                     featurefreq_leaves=_m_predict_h(f.get_tree_leaf_counts(), f.get_leaf_values())),
                g, lambda e: (np.asarray(e.predict_classes(Xh[:256])),))


# ---------------------------------------------------------------- batch declarations (lane vdv-fin, 2026-11-24)
# fit_predict fits on the rows it labels (a clustering, a mixture, an outlier fit): a row alone or a split is a
# different fit, not a smaller batch of the same call; the lanes hold no estimator.
_batch_decl("n/a:fit-endpoint (fit_predict labels the rows it fits on, so each row's label depends on every row "
            "of the call; a row alone is a different fit; the lane holds no estimator)",
            "agglomerative-fit-predict",
            "dbscan-fit-predict",
            "hdbscan-fit-predict",
            "spectral-fit-predict",
            "gmm-fit-predict",
            "iforest-fit-predict",
            "m-kmeans-fit-predict",
            "x-cluster-affinity-propagation-fit-predict",
            "x-cluster-bgmm-fit-predict",
            "x-cluster-meanshift-fit-predict",
            "x-cluster-optics-fit-predict",
            "x-neighbors-louvain-fit-predict",
            "x-neighbors-ocsvm-fit-predict",
            "x-decomp-robust-cov-fit-predict")
# forecast(y, h) fits each series and forecasts it in one call; there are no input rows, and the horizon's
# length invariance is asked on the forecaster lanes that keep a fitted model (`_batch_forecast`).
_batch_decl("n/a:forecast-fit (forecast(y, h) fits every series and forecasts it in one call: no input rows and "
            "no fitted estimator; horizon length invariance is the batch part of the fitted forecaster lanes)",
            "sequence-theta-forecast",
            "public-theta-variants-forecast",
            "sequence-croston-forecast",
            "sequence-ets-forecast")
_batch_decl("n/a:whole-series (in-sample predictions, the fitted components, the SSE and the parameters are "
            "functions of the whole fitted series, and a forecast's horizon invariance is the batch part of the "
            "fitted forecaster lanes such as holtwinters and arima; no input rows)",
            "m-autoarima-predict",
            "m-prophet-add-seasonality",
            "holtwinters-methods")
_batch_decl("n/a:scalar-reduction (score, aic, bic, perplexity, reconstruction_error and profile_jitter each "
            "return one float over every row or over the fitted model; no per-row output)",
            "gmm-methods",
            "x-decomp-lda-perplexity",
            "m-isomap-reconstruction-error",
            "cholesky-profile-jitter")


def _m_predict_batch_latent(ml, e, Xh):
    """GaussianProcessClassifier.latent_mean_and_variance on the gpc lane's held-out query (64 rows, four
    columns): the latent mean and variance of a row are read against the fitted training rows only."""
    R = np.ascontiguousarray(Xh[:64, :4])
    return [_BatchRows("latent_mean_and_variance", R,
                       lambda r: tuple(np.asarray(a) for a in e.latent_mean_and_variance(r)))]


# The row-wise methods each lane's probe reads, on the same held-out rows. Derived inputs go through a lambda so
# the prep.py helpers are looked up when the part runs (prep.py loads after this file).
_batch_decl(_m_predict_batch_latent, "gpc-methods")
_batch_decl(_rows_calls("predict_joint_log_proba", "predict_log_proba", sl=slice(0, 256)),
            "x-prep-lda-methods", "x-prep-qda-methods")
_batch_decl(_rows_calls("predict_joint_log_proba", sl=slice(0, 256)),
            "x-prep-gaussian-nb-methods", "x-prep-bernoulli-nb-methods")
_batch_decl(_rows_calls("predict_joint_log_proba", sl=slice(0, 256), prep=lambda Xh: _prep_abs(Xh)),
            "x-prep-multinomial-nb-methods", "x-prep-complement-nb-methods")
_batch_decl(_rows_calls("predict_joint_log_proba", sl=slice(0, 256), prep=lambda Xh: _prep_cat_codes(Xh)),
            "x-prep-categorical-nb-methods")
_batch_decl(_rows_calls("decision_function", "predict_proba", sl=slice(0, 256)), "x-prep-score-edges-rfe-methods")
_batch_decl(_rows_calls("decision_function", sl=slice(0, 512)), "iforest-decision-function")
_batch_decl(_rows_calls("decision_function", sl=slice(0, 256)),
            "x-neighbors-lof-decision-function", "trees-adaboost-clf-methods")
_batch_decl(_rows_calls("score_samples", sl=slice(0, 256)),
            "x-neighbors-ocsvm-score-samples", "x-sgd-ocsvm-score-samples")
_batch_decl(_rows_calls("score_samples", sl=(slice(0, 256), slice(0, 6))), "m-robust-cov-methods")
_batch_decl(_rows_calls("transform", sl=slice(0, 256), prep=lambda Xh: np.ascontiguousarray(Xh[:, 3]) * np.float32(1.5)),
            "x-isotonic-methods")
_batch_decl(_rows_calls("transform", sl=slice(0, 256)), "trees-voting-reg-methods")
_batch_decl(_rows_calls("transform", sl=slice(0, 128)), "m-gridsearch-methods")
# apply: one leaf id per row and tree, the native walk of each row alone; it is the read-back the lanes'
# held-out probe asks (predict_with_leaves and staged_predict are hashed in the train part only)
_batch_decl(_rows_calls("apply", sl=slice(0, 256)),
            "et-clf-methods", "trees-dt-clf-methods", "m-dt-reg-methods", "m-et-reg-methods", "m-rf-methods")
_batch_decl(_rows_calls("predict_classes", sl=slice(0, 256)), "m-gbdt-readbacks")
