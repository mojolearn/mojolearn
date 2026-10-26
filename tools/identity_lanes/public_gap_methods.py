# SPDX-License-Identifier: Apache-2.0
# Public endpoints absent from the 2026-10-23 changed-method coverage audit.
# These lanes consume the actual public answers. They remain OWED until real
# frozen runs and the existing reference/admission process qualify them.
# Sizes below are small correctness fixtures, never product dispatch rules.


@lane("public-theta-variants")
def _(ml, X, yc, yr, Xh=None):
    y = np.ascontiguousarray(X[:96, :3].T, dtype=np.float32)
    dynamic = ml.DynamicTheta(season_length=1, alpha=0.3).fit(y)
    optimized = ml.OptimizedTheta(season_length=1, alpha=0.3).fit(y)
    return _fit(dict(dynamic=_h(dynamic.predict(9)["mean"], dynamic.info_),
                     optimized=_h(optimized.predict(9)["mean"], optimized.info_)))


@lane("public-numeric-metadata")
def _(ml, X, yc, yr, Xh=None):
    from mojolearn import ensemble, training
    rates = [ensemble.catboost_auto_learning_rate("Logloss", 10000, 1000, False, False),
             ensemble.catboost_auto_learning_rate("RMSE", 200000, 1000, False, True)]
    missing = ensemble.catboost_auto_learning_rate("MAE", 1000, 1000, False, True)
    dimensions = [ml.johnson_lindenstrauss_min_dim(n, eps=eps)
                  for n, eps in ((1, 0.2), (17, 0.25), (1000, 0.4))]
    vendor = training.vendor_used(numeric_mode="identical")
    # The vendor string must differ across devices. Check it against the
    # already selected backend and hash the consumed qualification result.
    if vendor != ml._backend.vendor():
        raise AssertionError("training.vendor_used disagrees with the selected backend")
    # An estimator's own read-backs (NumericModeMixin.numeric_mode_used / vendor_used through the native
    # mode metadata: x_ann_* and x_cluster_* on a device column, the host bindings' GPU-named exports on
    # the CPU column). Checked against the selected backend, not hashed, so the digest keeps its bits.
    for est in (ml.TSNE(random_state=5), ml.MiniBatchKMeans(n_clusters=2, random_state=5)):  # glue: two classes
        if est.numeric_mode_used() != "identical" or est.vendor_used() != ml._backend.vendor():
            raise AssertionError(f"{type(est).__name__}.numeric_mode_used/vendor_used disagree with the selected backend")
    return _fit(dict(rates=_h(np.asarray(rates, dtype=np.float64)),
                     dimensions=_h(np.asarray(dimensions, dtype=np.int64)),
                     metadata=_h(np.asarray([missing is None, vendor in ("cpu", "metal", "cuda", "hip")], dtype=np.uint8))))


@lane("public-sparse-encode")
def _(ml, X, yc, yr, Xh=None):
    rows = np.ascontiguousarray(X[:32, :6], dtype=np.float32)
    dictionary = np.eye(6, dtype=np.float32)
    threshold = ml.sparse_encode(rows, dictionary, algorithm="threshold", alpha=0.125,
                                 numeric_mode="identical")
    lasso = ml.sparse_encode(rows, dictionary, algorithm="lasso_cd", alpha=0.125,
                             max_iter=30, numeric_mode="identical")
    omp = ml.sparse_encode(rows, dictionary, algorithm="omp", n_nonzero_coefs=2,
                           numeric_mode="identical")
    return _fit(dict(threshold=_h(threshold), lasso=_h(lasso), omp=_h(omp)))


@lane("public-residual-dropout")
def _(ml, X, yc, yr, Xh=None):
    from mojolearn import training
    values = np.ascontiguousarray(X[:16, :8], dtype=np.float32)
    residual = np.ascontiguousarray(X[16:32, :8], dtype=np.float32)
    gradient = np.ascontiguousarray(X[32:48, :8], dtype=np.float32)
    forward = training.residual_dropout(values, residual, p=0.375, seed=42, stream=3, offset=17)
    dvalues, dresidual = training.residual_dropout_backward(gradient, p=0.375, seed=42, stream=3, offset=17)
    plain = training.residual_dropout(values, residual, p=0.0, seed=42)
    return _fit(dict(forward=_h(forward), dvalues=_h(dvalues), dresidual=_h(dresidual), plain=_h(plain)))


@lane("public-scorer-factories")
def _(ml, X, yc, yr, Xh=None):
    from mojolearn import model_selection as ms, metrics
    rows = np.ascontiguousarray(X[:48, :4], dtype=np.float32)
    target = np.ascontiguousarray(yr[:48], dtype=np.float32)
    splitter = ms.check_cv(3, target, classifier=False)
    folds = list(splitter.split(rows, target))
    names = ms.get_scorer_names()
    if "r2" not in names or names != sorted(set(names)):
        raise AssertionError("get_scorer_names must expose unique sorted supported names")
    model = ml.Ridge(alpha=1.0, numeric_mode="identical").fit(rows, target)
    named = ms.get_scorer("r2")
    made = ms.make_scorer(metrics.r2_score, response_method="predict", numeric_mode="identical")
    values = np.asarray([named(model, rows, target), made(model, rows, target)], dtype=np.float64)
    return _fit(dict(scores=_h(values),
                     folds=_h(*[np.asarray(indices, dtype=np.int64) for pair in folds for indices in pair]),
                     names=_h(np.frombuffer("\n".join(names).encode("utf-8"), dtype=np.uint8))))


@lane("public-validation-learning-curves")
def _(ml, X, yc, yr, Xh=None):
    from mojolearn import model_selection as ms
    rows = np.ascontiguousarray(X[:48, :4], dtype=np.float32)
    target = np.ascontiguousarray(yr[:48], dtype=np.float32)
    model = ml.Ridge(numeric_mode="identical")
    train, test = ms.validation_curve(model, rows, target, param_name="alpha", param_range=[0.5, 2.0],
                                       cv=3, scoring="neg_mean_squared_error", n_jobs=1, error_score="raise")
    sizes, learning_train, learning_test = ms.learning_curve(model, rows, target, train_sizes=[0.5, 1.0],
                                                             cv=3, scoring="neg_mean_squared_error", n_jobs=1,
                                                             error_score="raise")
    return _fit(dict(validation=_h(train, test), learning=_h(sizes, learning_train, learning_test)))


@lane("public-randomized-search")
def _(ml, X, yc, yr, Xh=None):
    from mojolearn import model_selection as ms, metrics
    rows = np.ascontiguousarray(X[:48, :4], dtype=np.float32)
    target = np.ascontiguousarray(yr[:48], dtype=np.float32)
    regression = ms.RandomizedSearchCV(ml.Ridge(numeric_mode="identical"), {"alpha": [0.5, 2.0]},
                                       n_iter=2, random_state=7, cv=3, n_jobs=1, error_score="raise").fit(rows, target)
    labels = np.arange(len(rows), dtype=np.int64) % 2
    classifier = ms.RandomizedSearchCV(ml.LogisticRegression(max_iter=30),
                                       {"C": [0.5, 2.0]}, n_iter=2, random_state=7, cv=3,
                                       n_jobs=1, error_score="raise").fit(rows, labels)
    reconstruction = ms.make_scorer(metrics.mean_squared_error, greater_is_better=False,
                                     response_method="transform", numeric_mode="identical")
    transformer = ms.RandomizedSearchCV(ml.StandardScaler(numeric_mode="identical"), {"with_mean": [False, True]},
                                        n_iter=2, random_state=7, scoring=reconstruction, cv=3,
                                        n_jobs=1, error_score="raise").fit(rows, rows)
    return _fit(dict(regression=_h(regression.predict(rows), np.float64(regression.score(rows, target))),
                     probability=_h(classifier.predict_proba(rows)), decision=_h(classifier.decision_function(rows)),
                     transformed=_h(transformer.transform(rows))))


_batch_decl("n/a:coverage-probe (the lane checks public call paths the release audit found no lane for; "
            "its outputs are train parts, digested whole)",
            "public-theta-variants", "public-numeric-metadata", "public-sparse-encode", "public-residual-dropout",
            "public-scorer-factories", "public-validation-learning-curves", "public-randomized-search")
