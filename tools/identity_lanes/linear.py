# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# THE LINEAR LANE'S IDENTITY LANES (docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md).
#
# Owned by the `linear` expansion lane. tools/identity_break.py executes this
# file in ITS OWN namespace after every helper and registry exists
# (`_load_lane_fragments`), so a lane here is written exactly like a lane there:
#
#     @lane("linear-example")
#     def _(ml, X, yc, yr, Xh=None):
#         m = ml.Example().fit(X)
#         return _fit(dict(out=_h(m.transform(X[:256]))), m, lambda e: (e.transform(Xh[:256]),))
#
#     _batch_decl(_rows_calls("transform", sl=slice(0, 256)), "linear-example")
#
# Rules the loader enforces: spell every lane name literally; add only your
# own lanes (to LANES and the per-lane registries); rebind no existing name;
# prefix your own helpers with `_linear_`. No imports are needed: np, _h,
# _fit, _rows_calls and the rest are this module's.


def _linear_reg_fit(m, X, yr, Xh):
    return _fit(dict(coef=_h(m.coef_), intercept=_h(m.intercept_), predict=_h(m.predict(X[:256]))),
                m, lambda e: (e.predict(Xh[:256]),))


def _linear_clf_fit(m, X, yc, Xh):
    return _fit(dict(coef=_h(m.coef_), intercept=_h(m.intercept_), decision=_h(m.decision_function(X[:256]))),
                m, lambda e: (e.decision_function(Xh[:256]), e.predict(Xh[:256])))


@lane("x-sgd-clf")
def _(ml, X, yc, yr, Xh=None):
    m = ml.SGDClassifier(loss="log_loss", penalty="elasticnet", max_iter=5, tol=None, random_state=3).fit(X[:2000], yc[:2000])
    return _linear_clf_fit(m, X, yc, Xh)


@lane("x-sgd-reg")
def _(ml, X, yc, yr, Xh=None):
    m = ml.SGDRegressor(loss="huber", penalty="l1", max_iter=5, tol=None, random_state=3).fit(X[:2000], yr[:2000])
    return _linear_reg_fit(m, X, yr, Xh)


_batch_decl(_rows_calls("predict", sl=slice(0, 256)), "x-sgd-reg")
_batch_decl(_rows_calls("decision_function", sl=slice(0, 256)), "x-sgd-clf")


def _linear_pos_target(X, yr):
    """A positive count-like target from the fixture's regression target:
    exp of a scaled copy, so the GLM lanes see y > 0 on every fixture."""
    z = (yr - yr.mean()) / (yr.std() + np.float32(1e-6))
    return np.exp(np.clip(z, -4, 4) * np.float32(0.5)).astype(np.float32)


@lane("x-glm-poisson")
def _(ml, X, yc, yr, Xh=None):
    m = ml.PoissonRegressor(alpha=0.1, max_iter=20).fit(X[:2000], _linear_pos_target(X[:2000], yr[:2000]))
    return _linear_reg_fit(m, X, yr, Xh)


@lane("x-glm-gamma")
def _(ml, X, yc, yr, Xh=None):
    m = ml.GammaRegressor(alpha=0.1, max_iter=20).fit(X[:2000], _linear_pos_target(X[:2000], yr[:2000]))
    return _linear_reg_fit(m, X, yr, Xh)


@lane("x-glm-tweedie")
def _(ml, X, yc, yr, Xh=None):
    m = ml.TweedieRegressor(power=1.5, alpha=0.1, max_iter=20).fit(X[:2000], _linear_pos_target(X[:2000], yr[:2000]))
    return _linear_reg_fit(m, X, yr, Xh)


_batch_decl(_rows_calls("predict", sl=slice(0, 256)), "x-glm-poisson", "x-glm-gamma", "x-glm-tweedie")


@lane("x-huber")
def _(ml, X, yc, yr, Xh=None):
    y = yr[:2000].copy()
    y[::17] = y[::17] + np.float32(25.0)  # planted outliers
    m = ml.HuberRegressor(max_iter=30).fit(X[:2000], y)
    f = _linear_reg_fit(m, X, yr, Xh)
    f["scale"] = _h(np.asarray([m.scale_], dtype=np.float32))
    return f


_batch_decl(_rows_calls("predict", sl=slice(0, 256)), "x-huber")


@lane("x-bayes-ridge")
def _(ml, X, yc, yr, Xh=None):
    m = ml.BayesianRidge().fit(X[:2000], yr[:2000])
    f = _linear_reg_fit(m, X, yr, Xh)
    f["hyper"] = _h(np.asarray([m.alpha_, m.lambda_], dtype=np.float32))
    return f


@lane("x-ard")
def _(ml, X, yc, yr, Xh=None):
    m = ml.ARDRegression(max_iter=50).fit(X[:2000], yr[:2000])
    f = _linear_reg_fit(m, X, yr, Xh)
    f["lambda"] = _h(m.lambda_)
    return f


_batch_decl(_rows_calls("predict", sl=slice(0, 256)), "x-bayes-ridge", "x-ard")


@lane("x-lars")
def _(ml, X, yc, yr, Xh=None):
    m = ml.Lars(n_nonzero_coefs=10).fit(X[:2000], yr[:2000])
    return _linear_reg_fit(m, X, yr, Xh)


@lane("x-lasso-lars")
def _(ml, X, yc, yr, Xh=None):
    m = ml.LassoLars(alpha=0.02).fit(X[:2000], yr[:2000])
    f = _linear_reg_fit(m, X, yr, Xh)
    f["active"] = _h(np.asarray(m.active_, dtype=np.int32))
    return f


_batch_decl(_rows_calls("predict", sl=slice(0, 256)), "x-lars", "x-lasso-lars")


@lane("x-quantile")
def _(ml, X, yc, yr, Xh=None):
    m = ml.QuantileRegressor(quantile=0.7, alpha=0.01, max_iter=300).fit(X[:2000], yr[:2000])
    return _linear_reg_fit(m, X, yr, Xh)


_batch_decl(_rows_calls("predict", sl=slice(0, 256)), "x-quantile")


def _linear_y3(X):
    """A three-class target from the fixture's own columns (3 and 4, which no
    fixture perturbs), cut at fixed thresholds."""
    s = X[:, 3] + np.float32(0.5) * X[:, 4]
    lo, hi = np.quantile(s, [1 / 3, 2 / 3])
    return ((s > lo).astype(np.int32) + (s > hi).astype(np.int32)).astype(np.int32)


@lane("x-perceptron")
def _(ml, X, yc, yr, Xh=None):
    m = ml.Perceptron(max_iter=5, tol=None, random_state=5).fit(X[:2000], _linear_y3(X[:2000]))
    return _linear_clf_fit(m, X, yc, Xh)


_batch_decl(_rows_calls("decision_function", sl=slice(0, 256)), "x-perceptron")


@lane("x-pa-clf")
def _(ml, X, yc, yr, Xh=None):
    m = ml.PassiveAggressiveClassifier(loss="squared_hinge", max_iter=5, tol=None, random_state=5).fit(X[:2000], yc[:2000])
    return _linear_clf_fit(m, X, yc, Xh)


@lane("x-pa-reg")
def _(ml, X, yc, yr, Xh=None):
    m = ml.PassiveAggressiveRegressor(max_iter=5, tol=None, random_state=5).fit(X[:2000], yr[:2000])
    return _linear_reg_fit(m, X, yr, Xh)


_batch_decl(_rows_calls("decision_function", sl=slice(0, 256)), "x-pa-clf")
_batch_decl(_rows_calls("predict", sl=slice(0, 256)), "x-pa-reg")


@lane("x-sgd-ocsvm")
def _(ml, X, yc, yr, Xh=None):
    m = ml.SGDOneClassSVM(nu=0.2, max_iter=5, tol=None, random_state=5).fit(X[:2000])
    return _fit(dict(coef=_h(m.coef_), offset=_h(m.offset_), decision=_h(m.decision_function(X[:256]))),
                m, lambda e: (e.decision_function(Xh[:256]), e.predict(Xh[:256])))


_batch_decl(_rows_calls("decision_function", sl=slice(0, 256)), "x-sgd-ocsvm")


@lane("x-ridge-clf")
def _(ml, X, yc, yr, Xh=None):
    m = ml.RidgeClassifier(alpha=3.0).fit(X[:2000], _linear_y3(X[:2000]))
    return _linear_clf_fit(m, X, yc, Xh)


_batch_decl(_rows_calls("decision_function", sl=slice(0, 256)), "x-ridge-clf")


@lane("x-ridge-cv")
def _(ml, X, yc, yr, Xh=None):
    m = ml.RidgeCV(alphas=(0.01, 0.3, 3.0, 30.0)).fit(X[:1000], yr[:1000])
    f = _linear_reg_fit(m, X, yr, Xh)
    f["choice"] = _h(np.asarray([m.alpha_, m.best_score_], dtype=np.float32))
    return f


_batch_decl(_rows_calls("predict", sl=slice(0, 256)), "x-ridge-cv")
