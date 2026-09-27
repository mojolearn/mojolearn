# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Prep lane option parity sanity against scikit-learn: encoder infrequent
categories, LabelBinarizer multilabel, SimpleImputer callable strategy,
LDA / QDA covariance_estimator, TargetEncoder categories list and cv
splitters, naive Bayes partial_fit."""
import warnings
import numpy as np
try:
    import pytest
    skp = pytest.importorskip("sklearn.preprocessing")
except ImportError:
    import sklearn.preprocessing as skp
import sklearn.impute as ski
import sklearn.discriminant_analysis as skd
import sklearn.naive_bayes as sknb
import sklearn.covariance as skc
import sklearn.model_selection as skm
import mojolearn as ml


def _cats(seed=0, n=400, d=4):
    rng = np.random.default_rng(seed)
    X = np.floor(np.abs(rng.standard_normal((n, d))) * 3).astype(np.float32)
    X[::11, 0] = np.nan
    return X


def _obj_to_nan(a):
    a = np.array(a, dtype=object)
    out = np.empty(a.shape, dtype=np.float64)
    for idx, v in np.ndenumerate(a):
        out[idx] = np.nan if (v is None or isinstance(v, str)) else float(v)
    return out


def test_infrequent():
    X, Xh = _cats(), _cats(1)
    Xh[::7, 1] = 77.0
    for kw in (dict(min_frequency=30), dict(max_categories=3), dict(min_frequency=0.05, max_categories=4),
               dict(max_categories=1)):
        m = ml.OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, **kw).fit(X)
        r = skp.OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, **kw).fit(X)
        for a, b in zip(m.infrequent_categories_, r.infrequent_categories_):
            assert (a is None) == (b is None), kw
            if a is not None:
                np.testing.assert_array_equal(np.asarray(a), b)
        np.testing.assert_array_equal(np.asarray(m.transform(Xh)), r.transform(Xh))
        for hu in ("infrequent_if_exist", "ignore", "warn"):
            for drop in (None, "first", "if_binary"):
                if drop is not None and kw.get("max_categories") == 1:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = ml.OneHotEncoder(handle_unknown=hu, drop=drop, **kw).fit(X)
                    r = skp.OneHotEncoder(handle_unknown=hu, drop=drop, sparse_output=False, **kw).fit(X)
                    Z = np.asarray(m.transform(Xh))
                    np.testing.assert_array_equal(Z, r.transform(Xh), err_msg=str((kw, hu, drop)))
                    np.testing.assert_array_equal(np.asarray(m.inverse_transform(m.transform(X))),
                                                  _obj_to_nan(r.inverse_transform(r.transform(X))))
                assert m.drop_idx_ == (None if r.drop_idx_ is None else list(r.drop_idx_)), (kw, drop)
    m = ml.OneHotEncoder(min_frequency=30, drop=[1.0, 0.0, 0.0, 2.0]).fit(X)
    r = skp.OneHotEncoder(min_frequency=30, drop=[1.0, 0.0, 0.0, 2.0], sparse_output=False).fit(X)
    np.testing.assert_array_equal(np.asarray(m.transform(X)), r.transform(X))


def test_multilabel_binarizer():
    rng = np.random.default_rng(3)
    Y = (rng.random((50, 4)) > 0.6).astype(np.int64)
    for kw in (dict(), dict(neg_label=-1, pos_label=2), dict(neg_label=-2, pos_label=0)):
        m, r = ml.LabelBinarizer(**kw).fit(Y), skp.LabelBinarizer(**kw).fit(Y)
        np.testing.assert_array_equal(np.asarray(m.classes_), r.classes_)
        assert m.y_type_ == r.y_type_
        np.testing.assert_array_equal(np.asarray(m.transform(Y)), r.transform(Y))
        S = rng.standard_normal((50, 4)).astype(np.float32)
        np.testing.assert_array_equal(np.asarray(m.inverse_transform(S)), r.inverse_transform(S))
        np.testing.assert_array_equal(np.asarray(m.inverse_transform(S, threshold=0.3)),
                                      r.inverse_transform(S, threshold=0.3))


def test_simple_imputer_callable():
    rng = np.random.default_rng(4)
    X = np.round(rng.standard_normal((80, 3)) * 2).astype(np.float32)
    X[::5, 0] = np.nan
    X[:, 2] = np.nan
    for keep in (False, True):
        m = ml.SimpleImputer(strategy=np.median, keep_empty_features=keep).fit(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = ski.SimpleImputer(strategy=np.median, keep_empty_features=keep).fit(X)
            np.testing.assert_allclose(np.asarray(m.statistics_), r.statistics_, rtol=1e-6)
            np.testing.assert_allclose(np.asarray(m.transform(X)), r.transform(X), rtol=1e-6)


def test_covariance_estimator():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((300, 4)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int) + (X[:, 2] > 0.8)
    for est in (skc.EmpiricalCovariance, skc.LedoitWolf):
        for s in ("lsqr", "eigen"):
            m = ml.LinearDiscriminantAnalysis(solver=s, covariance_estimator=est()).fit(X, y)
            r = skd.LinearDiscriminantAnalysis(solver=s, covariance_estimator=est()).fit(X, y)
            np.testing.assert_allclose(np.asarray(m.predict_proba(X)), r.predict_proba(X), atol=2e-4)
            np.testing.assert_allclose(np.asarray(m.covariance_), r.covariance_, rtol=1e-4, atol=1e-5)
        q = ml.QuadraticDiscriminantAnalysis(solver="eigen", covariance_estimator=est()).fit(X, y)
        rq = skd.QuadraticDiscriminantAnalysis(solver="eigen", covariance_estimator=est()).fit(X, y)
        np.testing.assert_allclose(np.asarray(q.predict_proba(X)), rq.predict_proba(X), atol=2e-4)
    for cls, kw in ((ml.LinearDiscriminantAnalysis, dict(solver="svd")),
                    (ml.QuadraticDiscriminantAnalysis, dict(solver="svd")),
                    (ml.LinearDiscriminantAnalysis, dict(solver="lsqr", shrinkage=0.5))):
        try:
            cls(covariance_estimator=skc.EmpiricalCovariance(), **kw).fit(X, y)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{cls.__name__} {kw} accepted a covariance_estimator")


def test_target_encoder_cv_categories():
    rng = np.random.default_rng(6)
    X = rng.integers(0, 5, (200, 2)).astype(np.float32)
    y = rng.standard_normal(200).astype(np.float32)
    yb = (y > 0).astype(int)
    cats = [[0.0, 1.0, 2.0, 3.0, 4.0, 9.0], [0.0, 1.0, 2.0, 3.0, 4.0]]
    for target, cv in ((y, skm.KFold(4)), (yb, skm.StratifiedKFold(3)),
                       (y, list(skm.KFold(5).split(X)))):
        m = ml.TargetEncoder(categories=cats, cv=cv, smooth=2.0)
        r = skp.TargetEncoder(categories=cats, cv=cv, smooth=2.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            np.testing.assert_allclose(np.asarray(m.fit_transform(X, target)), r.fit_transform(X, target),
                                       rtol=1e-5, atol=1e-6)
        for a, b in zip(m.encodings_, r.encodings_):
            np.testing.assert_allclose(np.asarray(a), b, rtol=1e-5, atol=1e-6)
    try:
        ml.TargetEncoder(cv=[(np.arange(100), np.arange(100))]).fit_transform(X, y)
    except ValueError:
        pass
    else:
        raise AssertionError("TargetEncoder accepted folds that do not cover every row")


def test_nb_partial_fit():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((600, 5)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0.5)
    Xa = np.abs(X)
    Xc = np.clip(np.floor(Xa * 2), 0, 4).astype(np.float32)
    Xc[:200] = np.minimum(Xc[:200], 2)
    w = (Xa[:, 2] + 0.5).astype(np.float32)
    for name, data in (("MultinomialNB", Xa), ("ComplementNB", Xa), ("BernoulliNB", X),
                       ("CategoricalNB", Xc), ("GaussianNB", X)):
        m, r = getattr(ml, name)(), getattr(sknb, name)()
        sel = np.flatnonzero(y[:200] != 2)
        for est in (m, r):
            est.partial_fit(data[sel], y[sel], classes=[0, 1, 2])
            est.partial_fit(data[200:450], y[200:450])
            est.partial_fit(data[450:], y[450:], sample_weight=w[450:])
        np.testing.assert_allclose(np.asarray(m.class_count_), r.class_count_, rtol=1e-6)
        tol = 5e-3 if name == "GaussianNB" else 1e-4
        np.testing.assert_allclose(np.asarray(m.predict_proba(data)), r.predict_proba(data), atol=tol)
        if name == "CategoricalNB":
            for a, b in zip(m.category_count_, r.category_count_):
                np.testing.assert_allclose(np.asarray(a), b, rtol=1e-6)
        elif name == "GaussianNB":
            np.testing.assert_allclose(np.asarray(m.theta_), r.theta_, rtol=1e-5, atol=1e-6)
        else:
            np.testing.assert_allclose(np.asarray(m.feature_count_), r.feature_count_, rtol=1e-5)
        # one partial_fit is fit, bit for bit
        a = getattr(ml, name)().partial_fit(data, y, classes=[0, 1, 2])
        b = getattr(ml, name)().fit(data, y)
        np.testing.assert_array_equal(np.asarray(a.predict_proba(data)), np.asarray(b.predict_proba(data)))


if __name__ == "__main__":
    for t in (test_infrequent, test_multilabel_binarizer, test_simple_imputer_callable, test_covariance_estimator,
              test_target_encoder_cv_categories, test_nb_partial_fit):
        t()
        print("PASS", t.__name__)


def test_kbins_quantile_methods():
    import sklearn.preprocessing as sk
    rng = np.random.default_rng(8)
    # 299 rows: no level i / nb lands n * i / nb on an integer, where numpy's float32
    # virtual index can round off it (x_prep/NOT_IMPLEMENTED.tsv, DIFFERS BY NAME)
    X = np.round(rng.standard_normal((299, 3)) * 4).astype(np.float32) / np.float32(4)
    for meth in ("inverted_cdf", "closest_observation", "interpolated_inverted_cdf", "hazen", "weibull",
                 "median_unbiased", "normal_unbiased", "linear", "averaged_inverted_cdf"):
        for nb in (3, 5, 7):
            m = ml.KBinsDiscretizer(n_bins=nb, encode="ordinal", quantile_method=meth).fit(X)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = sk.KBinsDiscretizer(n_bins=nb, encode="ordinal", quantile_method=meth).fit(X)
            for a, b in zip(m.bin_edges_, r.bin_edges_):
                np.testing.assert_allclose(np.asarray(a), b, rtol=1e-6, atol=1e-6, err_msg=f"{meth} {nb}")


if __name__ == "__main__":
    test_kbins_quantile_methods()
    print("PASS test_kbins_quantile_methods")
