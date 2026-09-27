# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# THE PREP LANE'S IDENTITY LANES (docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md).
#
# Owned by the `prep` expansion lane. tools/identity_break.py executes this
# file in ITS OWN namespace after every helper and registry exists
# (`_load_lane_fragments`), so a lane here is written exactly like a lane there:
#
#     @lane("prep-example")
#     def _(ml, X, yc, yr, Xh=None):
#         m = ml.Example().fit(X)
#         return _fit(dict(out=_h(m.transform(X[:256]))), m, lambda e: (e.transform(Xh[:256]),))
#
#     _batch_decl(_rows_calls("transform", sl=slice(0, 256)), "prep-example")
#
# Rules the loader enforces: spell every lane name literally; add only your
# own lanes (to LANES and the per-lane registries); rebind no existing name;
# prefix your own helpers with `_prep_`. No imports are needed: np, _h,
# _fit, _rows_calls and the rest are this module's.


def _prep_transformer(m, X, Xh, **attrs):
    parts = {k: _h(getattr(m, k)) for k in attrs.get("attrs", ())}
    parts["transform"] = _h(m.transform(X[:256]))
    return _fit(parts, m, lambda e: (e.transform(Xh[:256]),))


@lane("x-prep-robust-scaler")
def _(ml, X, yc, yr, Xh=None):
    m = ml.RobustScaler().fit(X)
    return _prep_transformer(m, X, Xh, attrs=("center_", "scale_"))


@lane("x-prep-robust-scaler-unit-variance")
def _(ml, X, yc, yr, Xh=None):
    m = ml.RobustScaler(unit_variance=True, quantile_range=(10.0, 90.0)).fit(X)
    return _prep_transformer(m, X, Xh, attrs=("center_", "scale_"))


@lane("x-prep-maxabs-scaler")
def _(ml, X, yc, yr, Xh=None):
    m = ml.MaxAbsScaler().fit(X)
    return _prep_transformer(m, X, Xh, attrs=("scale_",))


_batch_decl(_rows_calls("transform", sl=slice(0, 256)), "x-prep-robust-scaler", "x-prep-maxabs-scaler",
            "x-prep-robust-scaler-unit-variance")


def _prep_categorical(X):
    """A few categories per column, ties everywhere, -1 from the denormal rows."""
    return np.clip(np.floor(X), -4, 4).astype(np.float32)


@lane("x-prep-ordinal-encoder")
def _(ml, X, yc, yr, Xh=None):
    Xq, Xhq = _prep_categorical(X), _prep_categorical(Xh)
    m = ml.OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(Xq)
    parts = {f"cat{j}": _h(c) for j, c in enumerate(m.categories_)}
    parts["transform"] = _h(m.transform(Xq[:256]))
    return _fit(parts, m, lambda e: (e.transform(Xhq[:256]),))


@lane("x-prep-onehot-encoder")
def _(ml, X, yc, yr, Xh=None):
    Xq, Xhq = _prep_categorical(X), _prep_categorical(Xh)
    m = ml.OneHotEncoder(handle_unknown="ignore", drop="if_binary").fit(Xq)
    parts = {f"cat{j}": _h(c) for j, c in enumerate(m.categories_)}
    parts["transform"] = _h(m.transform(Xq[:256]))
    return _fit(parts, m, lambda e: (e.transform(Xhq[:256]),))


_batch_decl(_rows_calls("transform", sl=slice(0, 256), prep=_prep_categorical),
            "x-prep-ordinal-encoder", "x-prep-onehot-encoder")


def _prep_categorical_nan(X):
    """The categorical fixture with NaN every 9th entry of column 0."""
    Xq = _prep_categorical(X)
    Xq[::9, 0] = np.nan
    return Xq


@lane("x-prep-encoder-options")
def _(ml, X, yc, yr, Xh=None):
    """OrdinalEncoder encoded_missing_value and inverse_transform,
    OneHotEncoder inverse_transform (drop, unknown blocks)."""
    Xq, Xhq = _prep_categorical_nan(X), _prep_categorical_nan(Xh)
    parts = {}
    for j, kw in enumerate((dict(), dict(encoded_missing_value=-3),
                            dict(handle_unknown="use_encoded_value", unknown_value=-1))):
        m = ml.OrdinalEncoder(**kw).fit(Xq[:1000])
        Z = m.transform(Xq[:256] if j < 2 else Xhq[:256])
        parts[f"ord{j}"] = _h(Z, m.inverse_transform(Z))
    for j, kw in enumerate((dict(handle_unknown="ignore"), dict(drop="first"),
                            dict(drop="if_binary", handle_unknown="ignore"))):
        m = ml.OneHotEncoder(**kw).fit(Xq[:1000])
        Z = m.transform(Xq[:256] if j == 1 else Xhq[:256])
        parts[f"ohe{j}"] = _h(m.inverse_transform(Z))
    m = ml.OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2).fit(Xq)
    return _fit(parts, m, lambda e: (e.transform(Xhq[:256]),))


_batch_decl(_rows_calls("transform", sl=slice(0, 256), prep=_prep_categorical_nan), "x-prep-encoder-options")


@lane("x-prep-encoder-categories")
def _(ml, X, yc, yr, Xh=None):
    """OrdinalEncoder / OneHotEncoder categories=<list>: per column the
    training values' sorted distinct set with one value dropped and one never
    seen added (NaN kept last), so both the unknown and the unused paths run."""
    Xq, Xhq = _prep_categorical_nan(X), _prep_categorical_nan(Xh)
    cats = []
    for j in range(Xq.shape[1]):
        u = np.unique(Xq[:, j])
        num = [float(v) for v in u if v == v]
        num = sorted(num[:1] + num[2:] + [float(np.float32(max(num) + 7.5))])
        cats.append(num + ([float("nan")] if np.isnan(u).any() else []))
    parts = {}
    for j, kw in enumerate((dict(handle_unknown="use_encoded_value", unknown_value=-1),
                            dict(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2))):
        m = ml.OrdinalEncoder(categories=cats, **kw).fit(Xq[:1000])
        Z = m.transform(Xhq[:256])
        parts[f"ord{j}"] = _h(*m.categories_, Z, m.inverse_transform(Z))
    for j, kw in enumerate((dict(handle_unknown="ignore"), dict(drop="first", handle_unknown="ignore"))):
        m = ml.OneHotEncoder(categories=cats, **kw).fit(Xq[:1000])
        Z = m.transform(Xhq[:256])
        parts[f"ohe{j}"] = _h(Z, m.inverse_transform(Z))
    m = ml.OneHotEncoder(categories=cats, handle_unknown="ignore").fit(Xq)
    return _fit(parts, m, lambda e: (e.transform(Xhq[:256]),))


_batch_decl(_rows_calls("transform", sl=slice(0, 256), prep=_prep_categorical_nan), "x-prep-encoder-categories")


@lane("x-prep-target-encoder")
def _(ml, X, yc, yr, Xh=None):
    Xq, Xhq = _prep_categorical(X), _prep_categorical(Xh)
    m = ml.TargetEncoder(random_state=3)
    cross = m.fit_transform(Xq, yr)
    mb = ml.TargetEncoder(random_state=3)
    cross_b = mb.fit_transform(Xq, yc)
    parts = dict(cross=_h(cross), cross_binary=_h(cross_b), mean=_h(m.target_mean_),
                 enc=_h(*m.encodings_), transform=_h(m.transform(Xq[:256])))
    return _fit(parts, m, lambda e: (e.transform(Xhq[:256]),))


_batch_decl(_rows_calls("transform", sl=slice(0, 256), prep=_prep_categorical), "x-prep-target-encoder")


def _prep_with_nan(X):
    """Every 7th entry missing (a fixed pattern), plus the tie-heavy rounding
    most_frequent and median read."""
    Xm = np.round(X * 2).astype(np.float32)
    Xm.reshape(-1)[::7] = np.nan
    return Xm


@lane("x-prep-simple-imputer")
def _(ml, X, yc, yr, Xh=None):
    Xm, Xhm = _prep_with_nan(X), _prep_with_nan(Xh)
    parts = {}
    for s in ("mean", "median", "most_frequent"):
        m = ml.SimpleImputer(strategy=s).fit(Xm)
        parts[s] = _h(m.statistics_)
        parts[s + "_t"] = _h(m.transform(Xm[:256]))
    m = ml.SimpleImputer(strategy="median").fit(Xm)
    return _fit(parts, m, lambda e: (e.transform(Xhm[:256]),))


@lane("x-prep-simple-imputer-indicator")
def _(ml, X, yc, yr, Xh=None):
    Xm, Xhm = _prep_with_nan(X), _prep_with_nan(Xh)
    m = ml.SimpleImputer(strategy="mean", add_indicator=True).fit(Xm)
    return _fit(dict(transform=_h(m.transform(Xm[:256]))), m, lambda e: (e.transform(Xhm[:256]),))


_batch_decl(_rows_calls("transform", sl=slice(0, 256), prep=_prep_with_nan), "x-prep-simple-imputer",
            "x-prep-simple-imputer-indicator")


@lane("x-prep-kbins")
def _(ml, X, yc, yr, Xh=None):
    parts = {}
    for s in ("uniform", "quantile", "kmeans"):
        m = ml.KBinsDiscretizer(n_bins=6, encode="ordinal", strategy=s).fit(X[:2000])
        parts[s] = _h(*m.bin_edges_)
        parts[s + "_t"] = _h(m.transform(X[:256]))
    lin = ml.KBinsDiscretizer(n_bins=4, strategy="quantile", quantile_method="linear").fit(X)
    parts["linear_onehot"] = _h(lin.transform(X[:256]))
    m = ml.KBinsDiscretizer(n_bins=5, encode="onehot-dense").fit(X)
    parts["default"] = _h(*m.bin_edges_)
    return _fit(parts, m, lambda e: (e.transform(Xh[:256]),))


_batch_decl(_rows_calls("transform", sl=slice(0, 256)), "x-prep-kbins")


def _prep_clf(m, X, Xh, attrs):
    parts = {k: _h(getattr(m, k)) for k in attrs}
    parts["predict"] = _h(m.predict(X[:256]))
    parts["proba"] = _h(m.predict_proba(X[:256]))
    parts["log_proba"] = _h(m.predict_log_proba(X[:256]))
    return _fit(parts, m, lambda e: (e.predict(Xh[:256]), e.predict_proba(Xh[:256])))


def _prep_abs(X):
    return np.abs(X).astype(np.float32)


@lane("x-prep-gaussian-nb")
def _(ml, X, yc, yr, Xh=None):
    m = ml.GaussianNB().fit(X, yc)
    return _prep_clf(m, X, Xh, ("theta_", "var_", "class_prior_"))


@lane("x-prep-multinomial-nb")
def _(ml, X, yc, yr, Xh=None):
    m = ml.MultinomialNB(alpha=0.5).fit(_prep_abs(X), yc)
    return _prep_clf(m, _prep_abs(X), _prep_abs(Xh), ("feature_count_", "feature_log_prob_", "class_log_prior_"))


@lane("x-prep-bernoulli-nb")
def _(ml, X, yc, yr, Xh=None):
    m = ml.BernoulliNB(binarize=0.25).fit(X, yc)
    return _prep_clf(m, X, Xh, ("feature_count_", "feature_log_prob_", "class_log_prior_"))


_batch_decl(_rows_calls("predict", "predict_proba", sl=slice(0, 256)), "x-prep-gaussian-nb", "x-prep-bernoulli-nb")
_batch_decl(_rows_calls("predict", "predict_proba", sl=slice(0, 256), prep=_prep_abs), "x-prep-multinomial-nb")


def _prep_three_class(X, yr):
    """A three-class target from the regression target's terciles (numpy's
    quantile, the same bytes on every box), so the discriminant lanes run
    their multiclass path."""
    lo, hi = np.quantile(yr, [1 / 3, 2 / 3])
    return np.where(yr < lo, 0, np.where(yr < hi, 1, 2)).astype(np.int32)


@lane("x-prep-lda")
def _(ml, X, yc, yr, Xh=None):
    y3 = _prep_three_class(X, yr)
    m = ml.LinearDiscriminantAnalysis().fit(X, y3)
    parts = dict(coef=_h(m.coef_), intercept=_h(m.intercept_), scal=_h(m.scalings_),
                 evr=_h(m.explained_variance_ratio_), transform=_h(m.transform(X[:256])),
                 predict=_h(m.predict(X[:256])), proba=_h(m.predict_proba(X[:256])))
    mb = ml.LinearDiscriminantAnalysis().fit(X, yc)
    parts["binary_decision"] = _h(mb.decision_function(X[:256]))
    return _fit(parts, m, lambda e: (e.predict_proba(Xh[:256]), e.transform(Xh[:256])))


@lane("x-prep-qda")
def _(ml, X, yc, yr, Xh=None):
    y3 = _prep_three_class(X, yr)
    m = ml.QuadraticDiscriminantAnalysis(reg_param=0.01).fit(X, y3)
    parts = dict(scal=_h(*m.scalings_), rot=_h(*m.rotations_), predict=_h(m.predict(X[:256])),
                 proba=_h(m.predict_proba(X[:256])))
    m0 = ml.QuadraticDiscriminantAnalysis(reg_param=0.05).fit(X, yc)
    parts["binary_decision"] = _h(m0.decision_function(X[:256]))
    return _fit(parts, m, lambda e: (e.predict(Xh[:256]), e.predict_proba(Xh[:256])))


_batch_decl(_rows_calls("predict", "predict_proba", sl=slice(0, 256)), "x-prep-lda", "x-prep-qda")


@lane("x-prep-da-solvers")
def _(ml, X, yc, yr, Xh=None):
    """LinearDiscriminantAnalysis solver 'lsqr' / 'eigen' with shrinkage None,
    'auto' (Ledoit-Wolf) and a constant, store_covariance; QDA solver 'eigen'
    with shrinkage and store_covariance."""
    y3 = _prep_three_class(X, yr)
    parts = {}
    for j, kw in enumerate((dict(solver="lsqr"), dict(solver="lsqr", shrinkage="auto"),
                            dict(solver="eigen", shrinkage=0.25), dict(solver="svd", store_covariance=True))):
        m = ml.LinearDiscriminantAnalysis(**kw).fit(X, y3)
        parts[f"lda{j}"] = _h(m.covariance_, m.coef_, m.intercept_, m.predict_proba(X[:256]))
    # (the eigen / QDA arms take a constant shrinkage: the dupes fixture's classes are
    # singular, and 'auto' can land within an ulp of the reference's refusal)
    me = ml.LinearDiscriminantAnalysis(solver="eigen", shrinkage=0.5).fit(X, yc)
    parts["eigen_binary"] = _h(me.coef_, me.explained_variance_ratio_, me.transform(X[:256]))
    for j, kw in enumerate((dict(solver="eigen", shrinkage=0.3, store_covariance=True),
                            dict(solver="svd", reg_param=0.05, store_covariance=True))):
        q = ml.QuadraticDiscriminantAnalysis(**kw).fit(X, y3)
        parts[f"qda{j}"] = _h(*q.covariance_, q.predict_proba(X[:256]))
    m = ml.LinearDiscriminantAnalysis(solver="eigen", shrinkage=0.25).fit(X, y3)
    return _fit(parts, m, lambda e: (e.predict_proba(Xh[:256]), e.transform(Xh[:256])))


_batch_decl(_rows_calls("predict", "predict_proba", sl=slice(0, 256)), "x-prep-da-solvers")


@lane("x-prep-nb-weights")
def _(ml, X, yc, yr, Xh=None):
    """sample_weight for every naive Bayes classifier; CategoricalNB min_categories."""
    y3 = _prep_three_class(X, yr)
    w = (np.abs(X[:, 0]) * 2 + 0.25).astype(np.float32)
    parts = {}
    g = ml.GaussianNB().fit(X, y3, sample_weight=w)
    parts["gnb"] = _h(g.theta_, g.var_, g.class_count_, g.predict_proba(X[:256]))
    Xa = _prep_abs(X)
    for nm in ("MultinomialNB", "ComplementNB", "BernoulliNB"):
        m = getattr(ml, nm)().fit(Xa if nm != "BernoulliNB" else X, y3, sample_weight=w)
        parts[nm] = _h(m.feature_count_, m.class_count_, m.feature_log_prob_)
    Xc = _prep_cat_codes(X)
    c = ml.CategoricalNB(min_categories=7).fit(Xc, y3, sample_weight=w)
    parts["cat"] = _h(*c.feature_log_prob_, c.class_count_, c.predict_proba(Xc[:256]))
    return _fit(parts, g, lambda e: (e.predict_proba(Xh[:256]),))


_batch_decl(_rows_calls("predict", "predict_proba", sl=slice(0, 256)), "x-prep-nb-weights")


@lane("x-prep-quantile-transformer")
def _(ml, X, yc, yr, Xh=None):
    m = ml.QuantileTransformer(n_quantiles=200, random_state=5).fit(X)
    mn = ml.QuantileTransformer(n_quantiles=64, output_distribution="normal", subsample=4000,
                                random_state=5).fit(X)
    parts = dict(q=_h(m.quantiles_), transform=_h(m.transform(X[:256])), qn=_h(mn.quantiles_),
                 normal=_h(mn.transform(X[:256])))
    return _fit(parts, mn, lambda e: (e.transform(Xh[:256]),))


_batch_decl(_rows_calls("transform", sl=slice(0, 256)), "x-prep-quantile-transformer")


@lane("x-prep-power-transformer")
def _(ml, X, yc, yr, Xh=None):
    m = ml.PowerTransformer().fit(X[:2000])
    Xp = np.abs(X[:2000]) + np.float32(0.5)
    mb = ml.PowerTransformer(method="box-cox", standardize=False).fit(Xp)
    parts = dict(lam=_h(m.lambdas_), transform=_h(m.transform(X[:256])), lam_bc=_h(mb.lambdas_),
                 bc=_h(mb.transform(Xp[:256])))
    return _fit(parts, m, lambda e: (e.transform(Xh[:256]),))


_batch_decl(_rows_calls("transform", sl=slice(0, 256)), "x-prep-power-transformer")


@lane("x-prep-inverse-transforms")
def _(ml, X, yc, yr, Xh=None):
    """inverse_transform of QuantileTransformer (uniform, normal) and
    PowerTransformer (yeo-johnson standardized, box-cox raw), on each one's
    own transform output and on a grid through the bounds."""
    grid = np.linspace(-6, 6, 97, dtype=np.float32)
    G = np.tile(grid[:, None], (1, X.shape[1]))
    G[::7, 0] = np.nan
    mu = ml.QuantileTransformer(n_quantiles=200, random_state=5).fit(X)
    mn = ml.QuantileTransformer(n_quantiles=64, output_distribution="normal", random_state=5).fit(X)
    my = ml.PowerTransformer().fit(X[:2000])
    Xp = np.abs(X[:2000]) + np.float32(0.5)
    mb = ml.PowerTransformer(method="box-cox", standardize=False).fit(Xp)
    Gu = np.clip(G, 0, 1) * np.float32(1.0)
    parts = dict(qu=_h(mu.inverse_transform(mu.transform(X[:256]))), qu_grid=_h(mu.inverse_transform(Gu)),
                 qn=_h(mn.inverse_transform(mn.transform(X[:256]))), qn_grid=_h(mn.inverse_transform(G)),
                 yj=_h(my.inverse_transform(my.transform(X[:256]))), yj_grid=_h(my.inverse_transform(G)),
                 bc=_h(mb.inverse_transform(mb.transform(Xp[:256]))), bc_grid=_h(mb.inverse_transform(G)))
    kb = ml.KBinsDiscretizer(n_bins=6, encode="ordinal", strategy="kmeans").fit(X[:2000])
    parts["kbins"] = _h(kb.inverse_transform(kb.transform(X[:256])))
    ko = ml.KBinsDiscretizer(n_bins=5, encode="onehot-dense").fit(X)
    parts["kbins_onehot"] = _h(ko.inverse_transform(ko.transform(X[:256])))
    y, yh = _prep_labels(X), _prep_labels(Xh)
    lb = ml.LabelBinarizer(neg_label=-1, pos_label=2).fit(y)
    K = np.asarray(lb.classes_).size
    Y = np.tile(X[:256], (1, K // X.shape[1] + 1))[:, :K]
    parts["labels"] = _h(lb.inverse_transform(lb.transform(y[:256])), lb.inverse_transform(Y))
    lb2 = ml.LabelBinarizer().fit(yc)
    parts["labels_binary"] = _h(lb2.inverse_transform(X[:256, :1]), lb2.inverse_transform(X[:256, :1], threshold=0.3))
    return _fit(parts, mn, lambda e: (e.inverse_transform(Xh[:256]),))


_batch_decl(_rows_calls("inverse_transform", sl=slice(0, 256)), "x-prep-inverse-transforms")


@lane("x-prep-normalizer")
def _(ml, X, yc, yr, Xh=None):
    parts = {nm: _h(ml.Normalizer(norm=nm).fit(X).transform(X[:256])) for nm in ("l1", "l2", "max")}
    m = ml.Normalizer().fit(X)
    return _fit(parts, m, lambda e: (e.transform(Xh[:256]),))


_batch_decl(_rows_calls("transform", sl=slice(0, 256)), "x-prep-normalizer")


@lane("x-prep-polynomial-features")
def _(ml, X, yc, yr, Xh=None):
    m = ml.PolynomialFeatures(degree=3).fit(X[:, :6])
    mi = ml.PolynomialFeatures(degree=(2, 3), interaction_only=True, include_bias=False).fit(X)
    parts = dict(transform=_h(m.transform(X[:256, :6])), powers=_h(m.powers_), inter=_h(mi.transform(X[:256])))
    return _fit(parts, m, lambda e: (e.transform(Xh[:256, :6]),))


def _prep_first_six(X):
    return X[:, :6]


_batch_decl(_rows_calls("transform", sl=slice(0, 256), prep=_prep_first_six), "x-prep-polynomial-features")


@lane("x-prep-spline-transformer")
def _(ml, X, yc, yr, Xh=None):
    m = ml.SplineTransformer().fit(X)
    mq = ml.SplineTransformer(n_knots=6, degree=2, knots="quantile", extrapolation="continue",
                              include_bias=False).fit(X)
    parts = dict(knots=_h(*m.bsplines_), transform=_h(m.transform(X[:256])), q=_h(mq.transform(X[:256])))
    return _fit(parts, m, lambda e: (e.transform(Xh[:256]),))


_batch_decl(_rows_calls("transform", sl=slice(0, 256)), "x-prep-spline-transformer")


@lane("x-prep-binarizer")
def _(ml, X, yc, yr, Xh=None):
    m = ml.Binarizer(threshold=0.3).fit(X)
    parts = dict(transform=_h(m.transform(X[:256])), zero=_h(ml.Binarizer().fit(X).transform(X[:256])))
    return _fit(parts, m, lambda e: (e.transform(Xh[:256]),))


_batch_decl(_rows_calls("transform", sl=slice(0, 256)), "x-prep-binarizer")


def _prep_labels(X):
    """Integer labels with gaps and ties, from the fixture's first column."""
    return (np.clip(np.floor(X[:, 0] * 3), -9, 9) * 7).astype(np.int64)


@lane("x-prep-label-encoder")
def _(ml, X, yc, yr, Xh=None):
    y, yh = _prep_labels(X), _prep_labels(Xh)
    m = ml.LabelEncoder().fit(y)
    parts = dict(classes=_h(m.classes_), transform=_h(m.transform(y[:256])),
                 floats=_h(ml.LabelEncoder().fit_transform(yr[:256])))
    known = np.isin(yh, np.asarray(m.classes_))
    return _fit(parts, m, lambda e: (e.transform(yh[known][:256]),))


@lane("x-prep-label-binarizer")
def _(ml, X, yc, yr, Xh=None):
    y, yh = _prep_labels(X), _prep_labels(Xh)
    m = ml.LabelBinarizer(neg_label=-1, pos_label=2).fit(y)
    parts = dict(classes=_h(m.classes_), transform=_h(m.transform(y[:256])),
                 binary=_h(ml.LabelBinarizer().fit(yc).transform(yc[:256])))
    return _fit(parts, m, lambda e: (e.transform(yh[:256]),))


def _prep_multilabel(X):
    """Rows of 0..3 labels each (columns 0-2 over a threshold give labels
    from column 5's integer part), so empty rows, repeats and unseen labels occur."""
    base = np.clip(np.floor(X[:, 5] * 2), -3, 3).astype(np.int64)
    return [[int(base[i]) + j for j in range(3) if X[i, j] > 0.2] for i in range(X.shape[0])]


@lane("x-prep-multilabel-binarizer")
def _(ml, X, yc, yr, Xh=None):
    ys, yhs = _prep_multilabel(X), _prep_multilabel(Xh)
    m = ml.MultiLabelBinarizer().fit(ys)
    parts = dict(classes=_h(m.classes_), transform=_h(m.transform(ys[:256])))
    return _fit(parts, m, lambda e: (e.transform(yhs[:256]),))


@lane("x-prep-iterative-imputer")
def _(ml, X, yc, yr, Xh=None):
    Xm, Xhm = _prep_with_nan(X[:3000, :8]), _prep_with_nan(Xh[:3000, :8])
    m = ml.IterativeImputer(max_iter=4, min_value=-5.0, max_value=5.0)
    out = m.fit_transform(Xm)
    md = ml.IterativeImputer(max_iter=2, imputation_order="descending", initial_strategy="median")
    parts = dict(fit=_h(out), n_iter=_h(np.array([m.n_iter_])), desc=_h(md.fit_transform(Xm)))
    return _fit(parts, m, lambda e: (e.transform(Xhm[:256]),))


def _prep_nan_first_eight(X):
    return _prep_with_nan(X[:, :8])


_batch_decl(_rows_calls("transform", sl=slice(0, 256), prep=_prep_nan_first_eight), "x-prep-iterative-imputer")


@lane("x-prep-variance-threshold")
def _(ml, X, yc, yr, Xh=None):
    m = ml.VarianceThreshold().fit(X)
    mt = ml.VarianceThreshold(threshold=0.3).fit(X)
    parts = dict(var=_h(m.variances_), transform=_h(m.transform(X[:256])), t=_h(np.array(mt.get_support())))
    return _fit(parts, m, lambda e: (e.transform(Xh[:256]),))


_batch_decl(_rows_calls("transform", sl=slice(0, 256)), "x-prep-variance-threshold")


@lane("x-prep-select-kbest")
def _(ml, X, yc, yr, Xh=None):
    y3 = _prep_three_class(X, yr)
    fc, pc = ml.f_classif(X, y3)
    fr, prv = ml.f_regression(X, yr)
    c2, pc2 = ml.chi2(np.abs(X), y3)
    m = ml.SelectKBest(k=5).fit(X, y3)
    parts = dict(fc=_h(fc, pc), fr=_h(fr, prv), c2=_h(c2, pc2), support=_h(np.array(m.get_support())),
                 transform=_h(m.transform(X[:256])))
    return _fit(parts, m, lambda e: (e.transform(Xh[:256]),))


_batch_decl(_rows_calls("transform", sl=slice(0, 256)), "x-prep-select-kbest")


@lane("x-prep-mutual-info")
def _(ml, X, yc, yr, Xh=None):
    Xs, y3 = X[:1500], _prep_three_class(X, yr)[:1500]
    mc = ml.mutual_info_classif(Xs, y3, random_state=4)
    mr = ml.mutual_info_regression(Xs, yr[:1500], random_state=4, n_neighbors=5)
    m = ml.SelectKBest(ml.mutual_info_regression, k=4).fit(Xs, yr[:1500])
    parts = dict(classif=_h(mc), regression=_h(mr), support=_h(np.array(m.get_support())))
    return _fit(parts, m, lambda e: (e.transform(Xh[:256]),))


@lane("x-prep-mi-discrete")
def _(ml, X, yc, yr, Xh=None):
    """mutual_info with discrete features: the contingency estimator
    (discrete feature, classes), Ross's with a feature's categories as the
    classes (discrete feature, continuous target) and the continuous columns
    of a mixed mask, by mask, by indices and all-discrete."""
    Xs, y3, ys = X[:1200, :6], _prep_three_class(X, yr)[:1200], yr[:1200]
    Xm = np.array(Xs, dtype=np.float32)
    Xm[:, :3] = _prep_categorical(Xs[:, :3])
    parts = {}
    for j, df in enumerate(([0, 1, 2], np.array([True, True, True, False, False, False]), [-6, 1])):
        parts[f"c{j}"] = _h(ml.mutual_info_classif(Xm, y3, discrete_features=df, random_state=3))
        parts[f"r{j}"] = _h(ml.mutual_info_regression(Xm, ys, discrete_features=df, random_state=3, n_neighbors=4))
    Xd = Xm[:, :3]
    parts["dd"] = _h(ml.mutual_info_classif(Xd, y3, discrete_features=True))
    parts["dc"] = _h(ml.mutual_info_regression(Xd, ys, discrete_features=True))
    m = ml.SelectKBest(lambda A, b: ml.mutual_info_classif(A, b, discrete_features=[0, 1, 2], random_state=3),
                       k=3).fit(Xm, y3)
    parts["support"] = _h(np.array(m.get_support()))
    return _fit(parts, m, lambda e: (e.transform(np.array(Xh[:256, :6], dtype=np.float32)),))


@lane("x-prep-rfe")
def _(ml, X, yc, yr, Xh=None):
    y3 = _prep_three_class(X, yr)
    m = ml.RFE(ml.LinearDiscriminantAnalysis(), n_features_to_select=5, step=3).fit(X, y3)
    parts = dict(ranking=_h(m.ranking_), support=_h(np.array(m.support_)), predict=_h(m.predict(X[:256])),
                 transform=_h(m.transform(X[:256])))
    return _fit(parts, m, lambda e: (e.predict(Xh[:256]), e.transform(Xh[:256])))


_batch_decl(_rows_calls("predict", "transform", sl=slice(0, 256)), "x-prep-rfe")


@lane("x-prep-score-edges")
def _(ml, X, yc, yr, Xh=None):
    """The reference's NaN / +inf score edges (f_classif of a constant and of
    a within-class-constant feature, chi2 of an all-zero feature),
    f_regression / r_regression with force_finite on and off, and RFE with
    importance_getter as a dotted path and as a callable."""
    y3 = _prep_three_class(X, yr)
    Xe = np.array(X[:, :6], dtype=np.float32)
    Xe[:, 1] = 2.5
    Xe[:, 2] = y3.astype(np.float32)
    Xz = np.abs(Xe)
    Xz[:, 4] = 0
    Xr = Xe.copy()
    Xr[:, 3] = yr * 2
    parts = dict(fc=_h(*ml.f_classif(Xe, y3)), c2=_h(*ml.chi2(Xz, y3)),
                 kbest=_h(np.array(ml.SelectKBest(k=3).fit(Xe, y3).get_support())))
    for ff in (True, False):
        parts[f"fr{int(ff)}"] = _h(*ml.f_regression(Xr, yr, force_finite=ff))
        parts[f"rr{int(ff)}"] = _h(ml.r_regression(Xe, yr, force_finite=ff))
    for j, g in enumerate(("coef_", lambda e: e.coef_[0])):
        m = ml.RFE(ml.LinearDiscriminantAnalysis(), n_features_to_select=5, step=3,
                   importance_getter=g).fit(X, y3)
        parts[f"rfe{j}"] = _h(m.ranking_, m.transform(X[:256]))
    return _fit(parts, m, lambda e: (e.predict(Xh[:256]), e.transform(Xh[:256])))


_batch_decl(_rows_calls("predict", "transform", sl=slice(0, 256)), "x-prep-score-edges")


@lane("x-prep-complement-nb")
def _(ml, X, yc, yr, Xh=None):
    y3 = _prep_three_class(X, yr)
    m = ml.ComplementNB(alpha=0.7).fit(_prep_abs(X), y3)
    mn = ml.ComplementNB(norm=True).fit(_prep_abs(X), yc)
    parts = dict(flp=_h(m.feature_log_prob_), norm=_h(mn.feature_log_prob_),
                 norm_proba=_h(mn.predict_proba(_prep_abs(X[:256]))))
    out = _prep_clf(m, _prep_abs(X), _prep_abs(Xh), ("feature_count_", "class_log_prior_"))
    out.update(parts)
    return out


_batch_decl(_rows_calls("predict", "predict_proba", sl=slice(0, 256), prep=_prep_abs), "x-prep-complement-nb")


def _prep_cat_codes(X):
    """Category indices 0..4 from the fixture (clip of |x| * 2)."""
    return np.clip(np.floor(np.abs(X) * 2), 0, 4).astype(np.float32)


@lane("x-prep-categorical-nb")
def _(ml, X, yc, yr, Xh=None):
    y3 = _prep_three_class(X, yr)
    Xc, Xhc = _prep_cat_codes(X), _prep_cat_codes(Xh)
    m = ml.CategoricalNB(alpha=0.5).fit(Xc, y3)
    parts = dict(flp=_h(*m.feature_log_prob_), ncat=_h(m.n_categories_))
    out = _prep_clf(m, Xc, Xhc, ("class_count_", "class_log_prior_"))
    out.update(parts)
    return out


_batch_decl(_rows_calls("predict", "predict_proba", sl=slice(0, 256), prep=_prep_cat_codes), "x-prep-categorical-nb")


@lane("x-prep-priors")
def _(ml, X, yc, yr, Xh=None):
    y3 = _prep_three_class(X, yr)
    g = ml.GaussianNB(priors=[0.2, 0.5, 0.3]).fit(X, y3)
    mn = ml.MultinomialNB(class_prior=[0.1, 0.6, 0.3]).fit(_prep_abs(X), y3)
    lda = ml.LinearDiscriminantAnalysis(priors=[1.0, 2.0, 1.0]).fit(X, y3)
    qda = ml.QuadraticDiscriminantAnalysis(priors=[0.25, 0.25, 0.5], reg_param=0.01).fit(X, y3)
    parts = dict(g=_h(g.predict_proba(X[:256])), mn=_h(mn.class_log_prior_, mn.predict_proba(_prep_abs(X[:256]))),
                 lda=_h(lda.priors_, lda.coef_, lda.predict_proba(X[:256])), qda=_h(qda.predict_proba(X[:256])))
    return _fit(parts, lda, lambda e: (e.predict_proba(Xh[:256]),))


class _prep_PyCov:
    """A covariance_estimator written in plain Python floats (IEEE double, the
    same bits on every box): the population covariance of the rows it is
    fitted on plus a 0.125 ridge, so every class's block is positive definite."""

    def fit(self, X):
        rows = [[float(v) for v in r] for r in X.tolist()]
        n, d = len(rows), len(rows[0])
        mean = [sum(r[j] for r in rows) / n for j in range(d)]
        self.covariance_ = [[sum((r[a] - mean[a]) * (r[b] - mean[b]) for r in rows) / n + (0.125 if a == b else 0.0)
                             for b in range(d)] for a in range(d)]
        return self


class _prep_Splits:
    """A cv splitter object: three interleaved test folds, the rest training."""

    def split(self, X, y=None):
        n = len(y)
        for k in range(3):
            test = [i for i in range(n) if i % 3 == k]
            yield [i for i in range(n) if i % 3 != k], test


@lane("x-prep-user-objects")
def _(ml, X, yc, yr, Xh=None):
    """User objects the reference accepts: SimpleImputer(strategy=<callable>),
    LinearDiscriminantAnalysis / QuadraticDiscriminantAnalysis
    covariance_estimator, TargetEncoder cv=<splitter> / cv=<(train, test)
    pairs> and categories=<list> (one training value left out, one unseen added)."""
    Xm, Xhm = _prep_with_nan(X), _prep_with_nan(Xh)
    parts = {}
    third = lambda v: (lambda s: float(s[len(s) // 3]) if s else float("nan"))(sorted(v.tolist()))
    si = ml.SimpleImputer(strategy=third, add_indicator=True).fit(Xm)
    parts["si"] = _h(si.statistics_, si.transform(Xm[:256]))
    y3 = _prep_three_class(X, yr)
    Xs, ys = X[:600], y3[:600]
    for s in ("lsqr", "eigen"):
        m = ml.LinearDiscriminantAnalysis(solver=s, covariance_estimator=_prep_PyCov()).fit(Xs, ys)
        parts["lda_" + s] = _h(m.covariance_, m.coef_, m.intercept_, m.predict_proba(X[:256]))
    q = ml.QuadraticDiscriminantAnalysis(solver="eigen", covariance_estimator=_prep_PyCov(),
                                         store_covariance=True).fit(Xs, ys)
    parts["qda"] = _h(*q.covariance_, q.predict_proba(X[:256]))
    Xq, Xhq = _prep_categorical(X), _prep_categorical(Xh)
    cats = []
    for j in range(Xq.shape[1]):
        num = [float(v) for v in np.unique(Xq[:, j])]
        cats.append(sorted(num[:1] + num[2:] + [float(np.float32(max(num) + 7.5))]))
    t = ml.TargetEncoder(categories=cats, cv=_prep_Splits())
    parts["te_split"] = _h(t.fit_transform(Xq, yr), *t.encodings_)
    pairs = [(np.flatnonzero(np.arange(len(yc)) % 4 != k), np.flatnonzero(np.arange(len(yc)) % 4 == k))
             for k in range(4)]
    tb = ml.TargetEncoder(cv=pairs)
    parts["te_pairs"] = _h(tb.fit_transform(Xq, yc), tb.transform(Xq[:256]))
    return _fit(parts, t, lambda e: (e.transform(Xhq[:256]),))


_batch_decl(_rows_calls("transform", sl=slice(0, 256), prep=_prep_categorical), "x-prep-user-objects")


@lane("x-prep-label-binarizer-multilabel")
def _(ml, X, yc, yr, Xh=None):
    """LabelBinarizer on a multilabel indicator y: pos / neg labels, a
    pos_label=0 switch, and the thresholded inverse."""
    Y = (X[:, :4] > 0.3).astype(np.int64)
    Yh = (Xh[:, :4] > 0.3).astype(np.int64)
    m = ml.LabelBinarizer(neg_label=-1, pos_label=2).fit(Y)
    z = ml.LabelBinarizer(neg_label=-3, pos_label=0).fit(Y)
    parts = dict(classes=_h(m.classes_), t=_h(m.transform(Y[:256])), z=_h(z.transform(Y[:256])),
                 inv=_h(m.inverse_transform(X[:256, :4] * 3), m.inverse_transform(X[:256, :4], threshold=0.1)))
    return _fit(parts, m, lambda e: (e.transform(Yh[:256]),))


def _prep_most_common(col):
    """The most frequent non-NaN value of a column (the first on a tie)."""
    u, c = np.unique(col[~np.isnan(col)], return_counts=True)
    return float(u[int(np.argmax(c))])


@lane("x-prep-infrequent")
def _(ml, X, yc, yr, Xh=None):
    """OrdinalEncoder / OneHotEncoder min_frequency and max_categories (a
    NaN category in column 0), handle_unknown 'infrequent_if_exist' / 'warn'
    / 'ignore' / 'use_encoded_value', drop 'first' / 'if_binary' / a list,
    and the inverse transforms."""
    Xq, Xhq = _prep_categorical_nan(X), _prep_categorical_nan(Xh)
    Xhq[::5, 1] = np.float32(55)   # unseen
    parts = {}
    for j, kw in enumerate((dict(min_frequency=40), dict(max_categories=3), dict(min_frequency=0.07, max_categories=4))):
        o = ml.OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, **kw).fit(Xq)
        Z = o.transform(Xhq[:256])
        parts[f"ord{j}"] = _h(*[c for c in o.infrequent_categories_ if c is not None], Z, o.inverse_transform(Z))
    import warnings
    for j, kw in enumerate((dict(min_frequency=40, handle_unknown="infrequent_if_exist"),
                            dict(max_categories=3, handle_unknown="ignore", drop="first"),
                            dict(max_categories=2, handle_unknown="warn", drop="if_binary"),
                            dict(min_frequency=25, handle_unknown="infrequent_if_exist",
                                 drop=[_prep_most_common(Xq[:, c]) for c in range(Xq.shape[1])]))):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            e = ml.OneHotEncoder(**kw).fit(Xq)
            Z = e.transform(Xhq[:256])
        parts[f"ohe{j}"] = _h(Z, e.inverse_transform(e.transform(Xq[:256])))
    m = ml.OneHotEncoder(min_frequency=40, handle_unknown="infrequent_if_exist").fit(Xq)
    return _fit(parts, m, lambda e: (e.transform(Xhq[:256]),))


_batch_decl(_rows_calls("transform", sl=slice(0, 256), prep=_prep_categorical_nan), "x-prep-infrequent")


@lane("x-prep-nb-partial")
def _(ml, X, yc, yr, Xh=None):
    """partial_fit of every naive Bayes classifier over three uneven batches
    (the first lacking a class, the last weighted), and a partial_fit after fit."""
    y3 = _prep_three_class(X, yr)
    cuts = [(0, 300), (300, 1100), (1100, min(len(y3), 1900))]
    first = np.flatnonzero(y3[:300] != 2)
    w = (np.abs(X[:, 1]) + 0.5).astype(np.float32)
    parts = {}
    Xa, Xc = _prep_abs(X), _prep_cat_codes(X)
    Xc[:300] = np.minimum(Xc[:300], 2)   # the categories widen in later batches
    for nm, data in (("GaussianNB", X), ("MultinomialNB", Xa), ("ComplementNB", Xa), ("BernoulliNB", X),
                     ("CategoricalNB", Xc)):
        m = getattr(ml, nm)()
        m.partial_fit(data[first], y3[first], classes=[0, 1, 2])
        m.partial_fit(data[300:1100], y3[300:1100])
        a, b = cuts[2]
        m.partial_fit(data[a:b], y3[a:b], sample_weight=w[a:b])
        attrs = [m.class_count_, m.predict_proba(data[:256])]
        attrs += [m.theta_, m.var_] if nm == "GaussianNB" else \
            (list(m.category_count_) if nm == "CategoricalNB" else [m.feature_count_])
        parts[nm] = _h(*attrs)
    g = ml.GaussianNB().fit(X[:500], y3[:500]).partial_fit(X[500:900], y3[500:900])
    parts["after_fit"] = _h(g.theta_, g.var_, g.predict_proba(X[:256]))
    return _fit(parts, g, lambda e: (e.predict_proba(Xh[:256]),))


_batch_decl(_rows_calls("predict", "predict_proba", sl=slice(0, 256)), "x-prep-nb-partial")


@lane("x-prep-kbins-methods")
def _(ml, X, yc, yr, Xh=None):
    """KBinsDiscretizer quantile_method: every numpy method besides the two
    the x-prep-kbins lane runs, on a tie-heavy (quarter-rounded) fixture."""
    Xr = np.round(X[:2001] * 4).astype(np.float32) / np.float32(4)
    parts = {}
    for meth in ("inverted_cdf", "closest_observation", "interpolated_inverted_cdf", "hazen", "weibull",
                 "median_unbiased", "normal_unbiased"):
        m = ml.KBinsDiscretizer(n_bins=7, encode="ordinal", quantile_method=meth).fit(Xr)
        parts[meth] = _h(*m.bin_edges_, m.transform(X[:256]))
    m = ml.KBinsDiscretizer(n_bins=6, quantile_method="hazen").fit(X)
    return _fit(parts, m, lambda e: (e.transform(Xh[:256]),))


_batch_decl(_rows_calls("transform", sl=slice(0, 256)), "x-prep-kbins-methods")
