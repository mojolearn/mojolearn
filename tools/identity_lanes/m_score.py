# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# ESTIMATOR `score` LANES (lane vdv-cov-est, 2026-11-24; plan docs/plans/VERIFY_THROUGH_DEVICE.md item C).
#
# One lane per estimator `score` that the inventory (docs/plans/verify_through_device/endpoints.tsv) left
# unmapped "OWN LANE". A lane named `<base>-score` calls the base lane the table names, checks that the base
# lane's fitted estimator IS the endpoint's class, and digests `score` as float64 bytes on rows and targets of
# the base lane's own fit. A lane named `m-<class>-score` is a full lane: the table's base lane does not hold a
# fitted estimator of that class (it returns no estimator, or another class), so the lane fits its own small
# model with the base lane's constructor arguments.
#
# The lanes are device-agnostic (the contract of vd-verify): the harness calls each lane once per side with a
# device-bound `ml`; a lane never passes `device=` and never reads the vendor. Every lane is an independent
# unit: it builds its own fixture slice and model and shares nothing mutable with another lane (an extension
# lane re-runs its base lane in full). Every reference is OWED to the release's one verification.
#
# tools/identity_break.py executes this file in ITS OWN namespace (`_load_lane_fragments`): lane names are
# literal, helpers carry the `_m_score_` prefix, nothing existing is rebound.


def _m_score_est(fit, cls):
    """The base lane's fitted estimator, refused unless it is a `cls`."""
    est = fit.est
    if type(est).__name__ != cls:
        raise AssertionError(f"m_score: the base lane's estimator is {type(est).__name__}, not {cls}")
    return est


def _m_score_fit(est, X, y, Xh, cols=slice(None)):
    """`_fit` of one score: the train part on (X, y); the held-out probe is the same call on the first
    len(X) held-out rows (columns `cols`, the base fit's) against the same targets (the contract's spelling).
    The held-out rows are read only by the probe."""
    n = len(X)
    return _fit(dict(score=_h(np.float64(est.score(X, y)))), est,
                lambda e: (np.float64(e.score(np.ascontiguousarray(Xh[:n, cols]), y)),))


def _m_score_fit_x(est, X, Xh, cols=slice(None)):
    """`_fit` of an unsupervised score(X) (a mixture, a density, k-means, factor analysis)."""
    n = len(X)
    return _fit(dict(score=_h(np.float64(est.score(X)))), est,
                lambda e: (np.float64(e.score(np.ascontiguousarray(Xh[:n, cols]))),))


# ---------------------------------------------------------------- linear regressors: the base fits (X[:N], yr[:N])

@lane("x-ard-score")
def _(ml, X, yc, yr, Xh=None):
    est = _m_score_est(LANES["x-ard"](ml, X, yc, yr, Xh), "ARDRegression")
    return _m_score_fit(est, X[:256], yr[:256], Xh)


@lane("x-bayes-ridge-score")
def _(ml, X, yc, yr, Xh=None):
    est = _m_score_est(LANES["x-bayes-ridge"](ml, X, yc, yr, Xh), "BayesianRidge")
    return _m_score_fit(est, X[:256], yr[:256], Xh)


@lane("x-lars-score")
def _(ml, X, yc, yr, Xh=None):
    est = _m_score_est(LANES["x-lars"](ml, X, yc, yr, Xh), "Lars")
    return _m_score_fit(est, X[:256], yr[:256], Xh)


@lane("x-lasso-lars-score")
def _(ml, X, yc, yr, Xh=None):
    est = _m_score_est(LANES["x-lasso-lars"](ml, X, yc, yr, Xh), "LassoLars")
    return _m_score_fit(est, X[:256], yr[:256], Xh)


@lane("x-quantile-score")
def _(ml, X, yc, yr, Xh=None):
    est = _m_score_est(LANES["x-quantile"](ml, X, yc, yr, Xh), "QuantileRegressor")
    return _m_score_fit(est, X[:256], yr[:256], Xh)


@lane("x-pa-reg-score")
def _(ml, X, yc, yr, Xh=None):
    est = _m_score_est(LANES["x-pa-reg"](ml, X, yc, yr, Xh), "PassiveAggressiveRegressor")
    return _m_score_fit(est, X[:256], yr[:256], Xh)


@lane("x-sgd-reg-score")
def _(ml, X, yc, yr, Xh=None):
    est = _m_score_est(LANES["x-sgd-reg"](ml, X, yc, yr, Xh), "SGDRegressor")
    return _m_score_fit(est, X[:256], yr[:256], Xh)


@lane("x-ridge-cv-score")
def _(ml, X, yc, yr, Xh=None):
    est = _m_score_est(LANES["x-ridge-cv"](ml, X, yc, yr, Xh), "RidgeCV")
    return _m_score_fit(est, X[:256], yr[:256], Xh)


@lane("x-lasso-cv-score")
def _(ml, X, yc, yr, Xh=None):
    est = _m_score_est(LANES["x-lasso-cv"](ml, X, yc, yr, Xh), "LassoCV")
    return _m_score_fit(est, X[:256], yr[:256], Xh)


@lane("x-enet-cv-score")
def _(ml, X, yc, yr, Xh=None):
    est = _m_score_est(LANES["x-enet-cv"](ml, X, yc, yr, Xh), "ElasticNetCV")
    return _m_score_fit(est, X[:256], yr[:256], Xh)


@lane("lasso-score")
def _(ml, X, yc, yr, Xh=None):
    est = _m_score_est(LANES["lasso"](ml, X, yc, yr, Xh), "Lasso")
    return _m_score_fit(est, X[:256], yr[:256], Xh)


@lane("elasticnet-score")
def _(ml, X, yc, yr, Xh=None):
    est = _m_score_est(LANES["elasticnet"](ml, X, yc, yr, Xh), "ElasticNet")
    return _m_score_fit(est, X[:256], yr[:256], Xh)


@lane("linear-svr-score")
def _(ml, X, yc, yr, Xh=None):
    est = _m_score_est(LANES["linear-svr"](ml, X, yc, yr, Xh), "LinearSVR")
    return _m_score_fit(est, X[:256], yr[:256], Xh)


@lane("qn-absolute-score")
def _(ml, X, yc, yr, Xh=None):
    est = _m_score_est(LANES["qn-absolute"](ml, X, yc, yr, Xh), "QNRegressor")
    return _m_score_fit(est, X[:256], yr[:256], Xh)


# the GLM lanes fit the positive target `_linear_pos_target(X[:2000], yr[:2000])`; score its first 256 rows

@lane("x-glm-poisson-score")
def _(ml, X, yc, yr, Xh=None):
    est = _m_score_est(LANES["x-glm-poisson"](ml, X, yc, yr, Xh), "PoissonRegressor")
    return _m_score_fit(est, X[:256], _linear_pos_target(X[:2000], yr[:2000])[:256], Xh)


@lane("x-glm-gamma-score")
def _(ml, X, yc, yr, Xh=None):
    est = _m_score_est(LANES["x-glm-gamma"](ml, X, yc, yr, Xh), "GammaRegressor")
    return _m_score_fit(est, X[:256], _linear_pos_target(X[:2000], yr[:2000])[:256], Xh)


@lane("x-glm-tweedie-score")
def _(ml, X, yc, yr, Xh=None):
    est = _m_score_est(LANES["x-glm-tweedie"](ml, X, yc, yr, Xh), "TweedieRegressor")
    return _m_score_fit(est, X[:256], _linear_pos_target(X[:2000], yr[:2000])[:256], Xh)


# ---------------------------------------------------------------- linear classifiers

@lane("x-sgd-clf-score")
def _(ml, X, yc, yr, Xh=None):
    est = _m_score_est(LANES["x-sgd-clf"](ml, X, yc, yr, Xh), "SGDClassifier")
    return _m_score_fit(est, X[:256], yc[:256], Xh)


@lane("x-pa-clf-score")
def _(ml, X, yc, yr, Xh=None):
    est = _m_score_est(LANES["x-pa-clf"](ml, X, yc, yr, Xh), "PassiveAggressiveClassifier")
    return _m_score_fit(est, X[:256], yc[:256], Xh)


@lane("linear-svc-score")
def _(ml, X, yc, yr, Xh=None):
    est = _m_score_est(LANES["linear-svc"](ml, X, yc, yr, Xh), "LinearSVC")
    return _m_score_fit(est, X[:256], yc[:256], Xh)


# the three-class lanes fit `_linear_y3` of their own rows; score the first 256 of those labels

@lane("x-perceptron-score")
def _(ml, X, yc, yr, Xh=None):
    est = _m_score_est(LANES["x-perceptron"](ml, X, yc, yr, Xh), "Perceptron")
    return _m_score_fit(est, X[:256], _linear_y3(X[:2000])[:256], Xh)


@lane("x-ridge-clf-score")
def _(ml, X, yc, yr, Xh=None):
    est = _m_score_est(LANES["x-ridge-clf"](ml, X, yc, yr, Xh), "RidgeClassifier")
    return _m_score_fit(est, X[:256], _linear_y3(X[:2000])[:256], Xh)


@lane("x-logistic-cv-score")
def _(ml, X, yc, yr, Xh=None):
    est = _m_score_est(LANES["x-logistic-cv"](ml, X, yc, yr, Xh), "LogisticRegressionCV")
    return _m_score_fit(est, X[:256], _linear_y3(X[:1200])[:256], Xh)


# ---------------------------------------------------------------- kernels, Gaussian processes, neighbors

@lane("gp-score")
def _(ml, X, yc, yr, Xh=None):
    est = _m_score_est(LANES["gp"](ml, X, yc, yr, Xh), "GaussianProcessRegressor")
    return _m_score_fit(est, X[256:320, :4], yr[256:320], Xh, cols=slice(0, 4))


@lane("kernel-ridge-score")
def _(ml, X, yc, yr, Xh=None):
    est = _m_score_est(LANES["kernel-ridge"](ml, X, yc, yr, Xh), "KernelRidge")
    return _m_score_fit(est, X[256:320, :4], yr[256:320], Xh, cols=slice(0, 4))


@lane("knn-clf-score")
def _(ml, X, yc, yr, Xh=None):
    est = _m_score_est(LANES["knn-clf"](ml, X, yc, yr, Xh), "KNeighborsClassifier")
    return _m_score_fit(est, X[4096:4160], yc[4096:4160], Xh)


@lane("x-neighbors-label-propagation-score")
def _(ml, X, yc, yr, Xh=None):
    est = _m_score_est(LANES["x-neighbors-label-propagation"](ml, X, yc, yr, Xh), "LabelPropagation")
    return _m_score_fit(est, X[:128], yc[:128], Xh)


@lane("x-neighbors-label-spreading-score")
def _(ml, X, yc, yr, Xh=None):
    est = _m_score_est(LANES["x-neighbors-label-spreading"](ml, X, yc, yr, Xh), "LabelSpreading")
    return _m_score_fit(est, X[:128], yc[:128], Xh)


@lane("x-neighbors-nearest-centroid-score")
def _(ml, X, yc, yr, Xh=None):
    # the base lane's estimator is its shrunk fit on the three-class labels below
    est = _m_score_est(LANES["x-neighbors-nearest-centroid"](ml, X, yc, yr, Xh), "NearestCentroid")
    y3 = (yc[:256] + (X[:256, 5] > 0).astype(np.int32)).astype(np.int32)
    return _m_score_fit(est, X[:256], y3, Xh)


@lane("x-neighbors-svc-multiclass-score")
def _(ml, X, yc, yr, Xh=None):
    # the base lane's estimator is its 'ovr' SVC over four classes; its own query rows 384:640
    est = _m_score_est(LANES["x-neighbors-svc-multiclass"](ml, X, yc, yr, Xh), "SVC")
    y4 = (yc[384:640] + 2 * (X[384:640, 5] > 0).astype(np.int32)).astype(np.int32)
    return _m_score_fit(est, X[384:640], y4, Xh)


@lane("x-neighbors-svgp-score")
def _(ml, X, yc, yr, Xh=None):
    est = _m_score_est(LANES["x-neighbors-svgp"](ml, X, yc, yr, Xh), "SVGP")
    return _m_score_fit(est, X[:256], yr[:256], Xh)


# ---------------------------------------------------------------- unsupervised score(X)

@lane("x-decomp-factor-analysis-score")
def _(ml, X, yc, yr, Xh=None):
    est = _m_score_est(LANES["x-decomp-factor-analysis"](ml, X, yc, yr, Xh), "FactorAnalysis")
    return _m_score_fit_x(est, X[:256], Xh)


@lane("kde-score")
def _(ml, X, yc, yr, Xh=None):
    est = _m_score_est(LANES["kde"](ml, X, yc, yr, Xh), "KernelDensity")
    return _m_score_fit_x(est, X[4096:4352, :4], Xh, cols=slice(0, 4))


@lane("x-cluster-bgmm-score")
def _(ml, X, yc, yr, Xh=None):
    est = _m_score_est(LANES["x-cluster-bgmm"](ml, X, yc, yr, Xh), "BayesianGaussianMixture")
    return _m_score_fit_x(est, X[:256, 1:5], Xh, cols=slice(1, 5))


@lane("x-cluster-bisecting-kmeans-score")
def _(ml, X, yc, yr, Xh=None):
    est = _m_score_est(LANES["x-cluster-bisecting-kmeans"](ml, X, yc, yr, Xh), "BisectingKMeans")
    return _m_score_fit_x(est, X[:256, :6], Xh, cols=slice(0, 6))


@lane("x-cluster-minibatch-kmeans-score")
def _(ml, X, yc, yr, Xh=None):
    est = _m_score_est(LANES["x-cluster-minibatch-kmeans"](ml, X, yc, yr, Xh), "MiniBatchKMeans")
    return _m_score_fit_x(est, X[:256, :8], Xh, cols=slice(0, 8))


# ---------------------------------------------------------------- ensembles (accuracy or R^2 of the base fit)

@lane("trees-adaboost-reg-score")
def _(ml, X, yc, yr, Xh=None):
    est = _m_score_est(LANES["trees-adaboost-reg"](ml, X, yc, yr, Xh), "AdaBoostRegressor")
    return _m_score_fit(est, X[:256], yr[:256], Xh)


@lane("trees-bagging-clf-score")
def _(ml, X, yc, yr, Xh=None):
    est = _m_score_est(LANES["trees-bagging-clf"](ml, X, yc, yr, Xh), "BaggingClassifier")
    return _m_score_fit(est, X[:256], yc[:256], Xh)


@lane("trees-bagging-reg-score")
def _(ml, X, yc, yr, Xh=None):
    est = _m_score_est(LANES["trees-bagging-reg"](ml, X, yc, yr, Xh), "BaggingRegressor")
    return _m_score_fit(est, X[:256], yr[:256], Xh)


@lane("trees-calibrated-score")
def _(ml, X, yc, yr, Xh=None):
    est = _m_score_est(LANES["trees-calibrated"](ml, X, yc, yr, Xh), "CalibratedClassifierCV")
    return _m_score_fit(est, X[:256], yc[:256], Xh)


@lane("trees-onevsrest-score")
def _(ml, X, yc, yr, Xh=None):
    est = _m_score_est(LANES["trees-onevsrest"](ml, X, yc, yr, Xh), "OneVsRestClassifier")
    return _m_score_fit(est, X[:256], yc[:256], Xh)


@lane("trees-stacking-clf-score")
def _(ml, X, yc, yr, Xh=None):
    est = _m_score_est(LANES["trees-stacking-clf"](ml, X, yc, yr, Xh), "StackingClassifier")
    return _m_score_fit(est, X[:256], yc[:256], Xh)


# ---------------------------------------------------------------- full lanes: the table's base lane holds no
# fitted estimator of the endpoint's class

@lane("m-linear-ridge-score")
def _(ml, X, yc, yr, Xh=None):
    """LinearRegression.score and Ridge.score (base classical-float16-sequences returns no estimator): the
    base lane's two models on its 512 x 8 rows, as float32."""
    x = np.ascontiguousarray(X[:512, :8], dtype=np.float32)
    y = np.ascontiguousarray(yr[:512], dtype=np.float32)
    ols = ml.LinearRegression().fit(x, y)
    ridge = ml.Ridge(alpha=0.5).fit(x, y)
    return _fit(dict(ols=_h(np.float64(ols.score(x[:256], y[:256]))),
                     ridge=_h(np.float64(ridge.score(x[:256], y[:256])))),
                ridge, lambda e: (np.float64(e.score(np.ascontiguousarray(Xh[:256, :8]), y[:256])),))


@lane("m-huber-score")
def _(ml, X, yc, yr, Xh=None):
    """HuberRegressor.score (base public-set-params returns no estimator): its constructor arguments on its
    256 x 6 rows."""
    rows = np.ascontiguousarray(X[:256, :6], dtype=np.float32)
    target = np.ascontiguousarray(yr[:256], dtype=np.float32)
    m = ml.HuberRegressor(epsilon=1.5, max_iter=40, alpha=1e-3).fit(rows, target)
    return _m_score_fit(m, rows, target, Xh, cols=slice(0, 6))


@lane("m-kmeans-score")
def _(ml, X, yc, yr, Xh=None):
    """KMeans.score (base public-set-params returns no estimator): its k-means on its 256 x 6 rows."""
    rows = np.ascontiguousarray(X[:256, :6], dtype=np.float32)
    m = ml.KMeans(n_clusters=4, n_init=1, max_iter=20, random_state=1).fit(rows)
    return _m_score_fit_x(m, rows, Xh, cols=slice(0, 6))


@lane("m-multioutput-ridge-score")
def _(ml, X, yc, yr, Xh=None):
    """MultiOutputRegressor.score (base multioutput-ridge returns no estimator): its three targets over
    Ridge(alpha=0.5) on 1024 x 6 rows; the uniform-average R^2."""
    x = np.ascontiguousarray(X[:1024, :6], dtype=np.float32)
    Y = np.ascontiguousarray(np.stack([yr[:1024], yr[:1024][::-1].copy(), np.asarray(X[:1024, 6])], axis=1),
                             dtype=np.float32)
    m = ml.MultiOutputRegressor(ml.Ridge(alpha=0.5)).fit(x, Y)
    return _m_score_fit(m, x[:256], Y[:256], Xh, cols=slice(0, 6))


@lane("m-multioutput-clf-score")
def _(ml, X, yc, yr, Xh=None):
    """MultiOutputClassifier.score (the table's base, public-scorer-factories, fits none): two label columns
    (the fixture's labels and a sign column) over a depth-4 tree on 1024 rows; the subset accuracy."""
    Y = np.ascontiguousarray(np.stack([yc[:1024], (X[:1024, 1] > 0).astype(yc.dtype)], axis=1)).astype(np.int64)
    m = ml.MultiOutputClassifier(ml.DecisionTreeClassifier(max_depth=4)).fit(X[:1024], Y)
    return _m_score_fit(m, X[:256], Y[:256], Xh)


@lane("m-svr-gamma-scale-score")
def _(ml, X, yc, yr, Xh=None):
    """SVR.score (base svc-gamma-scale returns no estimator): its SVR, gamma='scale', on its 1000 rows,
    scored on its query rows 1000:1256."""
    m = ml.SVR(C=1.0, kernel="rbf", gamma="scale", max_iter=200).fit(X[:1000], yr[:1000])
    return _m_score_fit(m, X[1000:1256], yr[1000:1256], Xh)


@lane("m-knn-reg-score")
def _(ml, X, yc, yr, Xh=None):
    """KNeighborsRegressor.score (base x-neighbors-metrics holds its classifier): its correlation-metric,
    distance-weighted regressor on 512 rows, scored on its query rows 512:640."""
    m = ml.KNeighborsRegressor(n_neighbors=7, metric="correlation", weights="distance").fit(X[:512], yr[:512])
    return _m_score_fit(m, X[512:640], yr[512:640], Xh)


@lane("m-dart-score")
def _(ml, X, yc, yr, Xh=None):
    """DARTClassifier.score and DARTRegressor.score (base trees-dart-options holds the classifier only and
    adds a TreeExplainer): its two constructors on the first 4096 rows."""
    r = ml.DARTRegressor(n_estimators=8, num_leaves=15, max_depth=5, min_child_samples=5, drop_rate=0.5,
                         skip_drop=0.0, reg_lambda=1.0, reg_alpha=2.0, max_delta_step=40.0, colsample_bytree=0.6,
                         subsample=0.7, subsample_freq=2, random_state=7).fit(X[:4096], yr[:4096])
    c = ml.DARTClassifier(n_estimators=6, num_leaves=15, max_depth=5, min_child_samples=5, drop_rate=0.5,
                          skip_drop=0.0, reg_alpha=0.5, max_delta_step=0.7, colsample_bytree=0.8, subsample=0.8,
                          subsample_freq=1, random_state=7).fit(X[:4096], yc[:4096])
    return _fit(dict(reg=_h(np.float64(r.score(X[:256], yr[:256]))),
                     clf=_h(np.float64(c.score(X[:256], yc[:256])))),
                c, lambda e: (np.float64(e.score(Xh[:256], yc[:256])),))


# ---------------------------------------------------------------- batch declarations (lane vdv-fin, 2026-11-24)
# Every lane here digests `score`, one float over every row it is handed (R^2, accuracy, a mean log-likelihood,
# a negative inertia), so there is no per-row output to ask alone or split. The row-wise methods `score` wraps
# (predict, predict_proba, score_samples, transform) keep their batch parts on the base lanes these lanes call
# or copy.
_batch_decl("n/a:scalar-reduction (score returns one float over every row it is handed; the row-wise "
            "methods it wraps keep their batch parts on the base lanes)",
            "x-ard-score",
            "x-bayes-ridge-score",
            "x-lars-score",
            "x-lasso-lars-score",
            "x-quantile-score",
            "x-pa-reg-score",
            "x-sgd-reg-score",
            "x-ridge-cv-score",
            "x-lasso-cv-score",
            "x-enet-cv-score",
            "lasso-score",
            "elasticnet-score",
            "linear-svr-score",
            "qn-absolute-score",
            "x-glm-poisson-score",
            "x-glm-gamma-score",
            "x-glm-tweedie-score",
            "x-sgd-clf-score",
            "x-pa-clf-score",
            "linear-svc-score",
            "x-perceptron-score",
            "x-ridge-clf-score",
            "x-logistic-cv-score",
            "gp-score",
            "kernel-ridge-score",
            "knn-clf-score",
            "x-neighbors-label-propagation-score",
            "x-neighbors-label-spreading-score",
            "x-neighbors-nearest-centroid-score",
            "x-neighbors-svc-multiclass-score",
            "x-neighbors-svgp-score",
            "x-decomp-factor-analysis-score",
            "kde-score",
            "x-cluster-bgmm-score",
            "x-cluster-bisecting-kmeans-score",
            "x-cluster-minibatch-kmeans-score",
            "trees-adaboost-reg-score",
            "trees-bagging-clf-score",
            "trees-bagging-reg-score",
            "trees-calibrated-score",
            "trees-onevsrest-score",
            "trees-stacking-clf-score",
            "m-linear-ridge-score",
            "m-huber-score",
            "m-kmeans-score",
            "m-multioutput-ridge-score",
            "m-multioutput-clf-score",
            "m-svr-gamma-scale-score",
            "m-knn-reg-score",
            "m-dart-score")
