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


@lane("x-prep-maxabs-scaler")
def _(ml, X, yc, yr, Xh=None):
    m = ml.MaxAbsScaler().fit(X)
    return _prep_transformer(m, X, Xh, attrs=("scale_",))


_batch_decl(_rows_calls("transform", sl=slice(0, 256)), "x-prep-robust-scaler", "x-prep-maxabs-scaler")


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


_batch_decl(_rows_calls("transform", sl=slice(0, 256), prep=_prep_with_nan), "x-prep-simple-imputer")


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


@lane("x-prep-rfe")
def _(ml, X, yc, yr, Xh=None):
    y3 = _prep_three_class(X, yr)
    m = ml.RFE(ml.LinearDiscriminantAnalysis(), n_features_to_select=5, step=3).fit(X, y3)
    parts = dict(ranking=_h(m.ranking_), support=_h(np.array(m.support_)), predict=_h(m.predict(X[:256])),
                 transform=_h(m.transform(X[:256])))
    return _fit(parts, m, lambda e: (e.predict(Xh[:256]), e.transform(Xh[:256])))


_batch_decl(_rows_calls("predict", "transform", sl=slice(0, 256)), "x-prep-rfe")


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
