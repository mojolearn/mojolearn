# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Release gate 2026-10-22 (part 2): the preprocessing, selection and
discriminant endpoints changed since 0.8.55, on the real binding path.

Each `*_refuses_*` test names the refusal the endpoint raises on a bad call
(unfitted model, wrong width, invalid option, mismatched rows); the other
tests check the result of an ordinary call on small data against an
independent restatement (NumPy, or the same model's own `fit` + `transform`).
"""
import numpy as np
import pytest

import mojolearn as ml


def _xy(n=40, d=4, k=3, seed=0):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % k
    X = (rng.standard_normal((n, d)) + y[:, None] * 1.5).astype(np.float32)
    return X, y.astype(np.int64)


def _pos(n=30, d=3, seed=1):
    rng = np.random.default_rng(seed)
    return (rng.random((n, d)) + 0.1).astype(np.float32)


def _cats(n=24, seed=2):
    rng = np.random.default_rng(seed)
    return np.stack([rng.integers(0, 3, n), rng.integers(0, 4, n)], axis=1).astype(np.float32)


# ------------------------------------------------------------------ labels

def test_label_binarizer_fit_refuses_bad_labels_and_sparse():
    y = np.array([0, 1, 2, 1])
    with pytest.raises(ValueError):
        ml.LabelBinarizer(neg_label=1, pos_label=1).fit(y)
    with pytest.raises(NotImplementedError):
        ml.LabelBinarizer(sparse_output=True).fit(y)


def test_label_binarizer_transform_refuses_unfitted():
    with pytest.raises(RuntimeError):
        ml.LabelBinarizer().transform(np.array([0, 1]))


def test_label_binarizer_fit_transform_refuses_bad_labels():
    with pytest.raises(ValueError):
        ml.LabelBinarizer(neg_label=3, pos_label=0).fit_transform(np.array([0, 1, 1]))


def test_label_binarizer_inverse_transform_refuses_wrong_width():
    m = ml.LabelBinarizer().fit(np.array([0, 1, 2, 2]))
    with pytest.raises(ValueError):
        m.inverse_transform(np.eye(4, 2, dtype=np.float32))
    with pytest.raises(RuntimeError):
        ml.LabelBinarizer().inverse_transform(np.eye(3, dtype=np.float32))


def test_label_encoder_fit_and_fit_transform_refuse_a_string():
    with pytest.raises(ValueError):
        ml.LabelEncoder().fit("abc")
    with pytest.raises(ValueError):
        ml.LabelEncoder().fit_transform("abc")


def test_label_encoder_transform_refuses_unseen_and_unfitted():
    m = ml.LabelEncoder().fit(np.array([3, 5, 7]))
    with pytest.raises(ValueError):
        m.transform(np.array([3, 4]))
    with pytest.raises(RuntimeError):
        ml.LabelEncoder().transform(np.array([3]))


def test_label_encoder_inverse_transform_refuses_out_of_range_code():
    m = ml.LabelEncoder().fit(np.array([3, 5, 7]))
    np.testing.assert_array_equal(np.asarray(m.inverse_transform(np.array([2, 0]))), [7, 3])
    with pytest.raises(ValueError):
        m.inverse_transform(np.array([0, 9]))


def test_multilabel_binarizer_refusals():
    rows = [[1, 2], [3], [1]]
    with pytest.raises(NotImplementedError):
        ml.MultiLabelBinarizer(sparse_output=True).fit(rows)
    with pytest.raises(NotImplementedError):
        ml.MultiLabelBinarizer(sparse_output=True).fit_transform(rows)
    with pytest.raises(RuntimeError):
        ml.MultiLabelBinarizer().transform(rows)


# ------------------------------------------------------- discriminant models

def test_lda_fit_and_fit_transform_refuse_bad_options():
    X, y = _xy()
    with pytest.raises(ValueError):
        ml.LinearDiscriminantAnalysis(solver="nope").fit(X, y)
    with pytest.raises(NotImplementedError):
        ml.LinearDiscriminantAnalysis(solver="svd", shrinkage="auto").fit(X, y)
    with pytest.raises(ValueError):
        ml.LinearDiscriminantAnalysis().fit(X, np.zeros(len(X), np.int64))
    with pytest.raises(ValueError):
        ml.LinearDiscriminantAnalysis(solver="nope").fit_transform(X, y)


def test_lda_decision_function_and_transform_refusals():
    X, y = _xy()
    with pytest.raises(RuntimeError):
        ml.LinearDiscriminantAnalysis().decision_function(X)
    m = ml.LinearDiscriminantAnalysis().fit(X, y)
    with pytest.raises(ValueError):
        m.decision_function(X[:, :3])
    with pytest.raises(ValueError):
        m.transform(X[:, :3])
    with pytest.raises(NotImplementedError):
        ml.LinearDiscriminantAnalysis(solver="lsqr").fit(X, y).transform(X)


def test_lda_score_and_fit_transform():
    X, y = _xy()
    m = ml.LinearDiscriminantAnalysis()
    Z = np.asarray(m.fit_transform(X, y))
    np.testing.assert_allclose(Z, np.asarray(m.transform(X)), rtol=1e-5, atol=1e-5)
    pred = np.asarray(m.predict(X))
    assert m.score(X, y) == pytest.approx(float(np.mean(pred == y)))
    with pytest.raises(ValueError):
        m.score(X, y[:-1])
    with pytest.raises(ValueError):
        m.score(X[:, :3], y)


def test_qda_refusals_and_score():
    X, y = _xy(n=60)
    with pytest.raises(ValueError):
        ml.QuadraticDiscriminantAnalysis(solver="nope").fit(X, y)
    with pytest.raises(ValueError):
        ml.QuadraticDiscriminantAnalysis().fit(X, np.zeros(len(X), np.int64))
    with pytest.raises(RuntimeError):
        ml.QuadraticDiscriminantAnalysis().decision_function(X)
    m = ml.QuadraticDiscriminantAnalysis().fit(X, y)
    with pytest.raises(ValueError):
        m.decision_function(X[:, :2])
    pred = np.asarray(m.predict(X))
    assert m.score(X, y) == pytest.approx(float(np.mean(pred == y)))
    with pytest.raises(ValueError):
        m.score(X, y[:5])


def test_qda_has_no_fit_transform():
    # QDA is a classifier without `transform`, so `fit_transform` is not offered
    X, y = _xy(n=60)
    assert not hasattr(ml.QuadraticDiscriminantAnalysis(), "fit_transform")
    with pytest.raises(AttributeError):
        ml.QuadraticDiscriminantAnalysis().fit_transform(X, y)


def test_multinomial_nb_score():
    rng = np.random.default_rng(3)
    X = rng.integers(0, 5, (40, 6)).astype(np.float32)
    y = np.arange(40) % 3
    m = ml.MultinomialNB().fit(X, y)
    pred = np.asarray(m.predict(X))
    assert m.score(X, y) == pytest.approx(float(np.mean(pred == y)))
    with pytest.raises(ValueError):
        m.score(X, y[:-3])
    with pytest.raises(ValueError):
        m.score(X[:, :4], y)


# ---------------------------------------------------------------- encoders

def test_ordinal_encoder_refusals():
    C = _cats()
    with pytest.raises(ValueError):
        ml.OrdinalEncoder(handle_unknown="drop").fit(C)
    with pytest.raises(ValueError):
        ml.OrdinalEncoder(handle_unknown="drop").fit_transform(C)
    m = ml.OrdinalEncoder().fit(C)
    with pytest.raises(ValueError):
        m.transform(C[:, :1])
    with pytest.raises(ValueError):
        m.transform(np.array([[9.0, 0.0]], np.float32))
    with pytest.raises(ValueError):
        m.inverse_transform(np.zeros((2, 3), np.float32))
    with pytest.raises(RuntimeError):
        ml.OrdinalEncoder().transform(C)


def test_one_hot_encoder_refusals():
    C = _cats()
    with pytest.raises(ValueError):
        ml.OneHotEncoder(handle_unknown="drop").fit(C)
    with pytest.raises(ValueError):
        ml.OneHotEncoder(handle_unknown="drop").fit_transform(C)
    m = ml.OneHotEncoder(sparse_output=False).fit(C)
    with pytest.raises(ValueError):
        m.transform(C[:, :1])
    with pytest.raises(ValueError):
        m.inverse_transform(np.zeros((2, 2), np.float32))
    with pytest.raises(RuntimeError):
        ml.OneHotEncoder().transform(C)


def test_target_encoder_refusals():
    C = _cats()
    y = (np.arange(len(C)) % 2).astype(np.int64)
    with pytest.raises(ValueError):
        ml.TargetEncoder(target_type="ordinal").fit(C, y)
    with pytest.raises(ValueError):
        ml.TargetEncoder(cv=1).fit_transform(C, y)
    with pytest.raises(ValueError):
        ml.TargetEncoder().fit(C, y[:-2])
    m = ml.TargetEncoder().fit(C, y)
    with pytest.raises(ValueError):
        m.transform(C[:, :1])


def test_simple_imputer_refusals():
    X = _pos()
    with pytest.raises(ValueError):
        ml.SimpleImputer(strategy="mode").fit(X)
    with pytest.raises(ValueError):
        ml.SimpleImputer(strategy="mode").fit_transform(X)
    m = ml.SimpleImputer().fit(X)
    with pytest.raises(ValueError):
        m.transform(X[:, :2])
    with pytest.raises(RuntimeError):
        ml.SimpleImputer().transform(X)


# ------------------------------------------------------------ transformers

def test_normalizer_and_polynomial_transform_refusals():
    X = _pos()
    with pytest.raises(ValueError):
        ml.Normalizer(norm="l3").fit(X)
    with pytest.raises(ValueError):
        ml.Normalizer().fit(X).transform(X[:, :2])
    with pytest.raises(RuntimeError):
        ml.Normalizer().transform(X)
    with pytest.raises(ValueError):
        ml.PolynomialFeatures(order="Z").fit(X)
    with pytest.raises(ValueError):
        ml.PolynomialFeatures().fit(X).transform(X[:, :2])
    with pytest.raises(RuntimeError):
        ml.PolynomialFeatures().transform(X)


def test_power_transformer_fit_transform_and_refusals():
    X = _pos()
    m = ml.PowerTransformer()
    Z = np.asarray(m.fit_transform(X))
    np.testing.assert_allclose(Z, np.asarray(m.transform(X)), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(m.inverse_transform(Z)), X, rtol=1e-3, atol=1e-3)
    with pytest.raises(ValueError):
        ml.PowerTransformer(method="log").fit(X)
    with pytest.raises(ValueError):
        ml.PowerTransformer(method="log").fit_transform(X)
    with pytest.raises(ValueError):
        ml.PowerTransformer(method="box-cox").fit(X - 1.0)
    with pytest.raises(ValueError):
        m.transform(X[:, :2])
    with pytest.raises(ValueError):
        m.inverse_transform(X[:, :2])


def test_quantile_transformer_fit_transform_and_refusals():
    X = _pos(n=50)
    m = ml.QuantileTransformer(n_quantiles=20)
    Z = np.asarray(m.fit_transform(X))
    np.testing.assert_allclose(Z, np.asarray(m.transform(X)), rtol=1e-6, atol=1e-6)
    assert Z.min() >= 0.0 and Z.max() <= 1.0
    with pytest.raises(ValueError):
        ml.QuantileTransformer(output_distribution="beta").fit(X)
    with pytest.raises(ValueError):
        ml.QuantileTransformer(output_distribution="beta").fit_transform(X)
    with pytest.raises(ValueError):
        m.transform(X[:, :2])
    with pytest.raises(ValueError):
        m.inverse_transform(X[:, :2])


def test_spline_transformer_fit_transform_and_refusals():
    X = _pos(n=30, d=2)
    m = ml.SplineTransformer(n_knots=4, degree=2)
    Z = np.asarray(m.fit_transform(X))
    np.testing.assert_allclose(Z, np.asarray(m.transform(X)), rtol=1e-5, atol=1e-5)
    # B-spline bases sum to one inside the knot range (include_bias=True)
    np.testing.assert_allclose(Z.reshape(len(X), 2, -1).sum(axis=2), 1.0, atol=1e-4)
    with pytest.raises(ValueError):
        ml.SplineTransformer(extrapolation="wrap").fit(X)
    with pytest.raises(ValueError):
        ml.SplineTransformer(extrapolation="wrap").fit_transform(X)
    e = ml.SplineTransformer(n_knots=4, degree=2, extrapolation="error").fit(X)
    with pytest.raises(ValueError):
        e.transform(X + 5.0)
    with pytest.raises(ValueError):
        e.transform(X[:, :1])


# --------------------------------------------------------------- selection

def test_variance_threshold_fit_transform_and_refusals():
    X = _pos()
    X[:, 1] = 0.5
    m = ml.VarianceThreshold()
    Z = np.asarray(m.fit_transform(X))
    np.testing.assert_array_equal(Z, X[:, [0, 2]])
    with pytest.raises(ValueError):
        ml.VarianceThreshold().fit(np.full((5, 3), 2.0, np.float32))
    with pytest.raises(ValueError):
        ml.VarianceThreshold().fit_transform(np.full((5, 3), 2.0, np.float32))
    with pytest.raises(ValueError):
        m.transform(X[:, :2])


def test_select_k_best_fit_transform_and_refusals():
    X, y = _xy(d=5)
    m = ml.SelectKBest(k=2)
    Z = np.asarray(m.fit_transform(X, y))
    keep = np.asarray(m.get_support(indices=True))
    assert Z.shape == (len(X), 2)
    np.testing.assert_array_equal(Z, X[:, keep])
    with pytest.raises(ValueError):
        ml.SelectKBest(k=9).fit(X, y)
    with pytest.raises(ValueError):
        ml.SelectKBest(k=9).fit_transform(X, y)


def test_rfe_fit_transform_and_refusals():
    X, y = _xy(d=6)
    m = ml.RFE(ml.LinearDiscriminantAnalysis(), n_features_to_select=3, step=1)
    Z = np.asarray(m.fit_transform(X, y))
    keep = np.asarray(m.get_support(indices=True))
    assert Z.shape == (len(X), 3) and keep.size == 3
    np.testing.assert_array_equal(Z, X[:, keep])
    with pytest.raises(ValueError):
        ml.RFE(ml.LinearDiscriminantAnalysis(), step=0).fit(X, y)
    with pytest.raises(ValueError):
        ml.RFE(ml.LinearDiscriminantAnalysis(), importance_getter=5).fit_transform(X, y)


def test_univariate_score_functions_refuse_mismatched_rows():
    X, y = _xy()
    yr = X[:, 0].astype(np.float32)
    with pytest.raises(ValueError):
        ml.f_classif(X, y[:-1])
    with pytest.raises(ValueError):
        ml.f_regression(X, yr[:-1])
    with pytest.raises(ValueError):
        ml.r_regression(X, yr[:-1])
    with pytest.raises(ValueError):
        ml.mutual_info_classif(X, y[:-1])
    with pytest.raises(ValueError):
        ml.mutual_info_regression(X, yr[:-1])


def test_mutual_info_refuses_neighbour_count():
    X, y = _xy()
    with pytest.raises(NotImplementedError):
        ml.mutual_info_classif(X, y, n_neighbors=0)
    with pytest.raises(NotImplementedError):
        ml.mutual_info_regression(X, X[:, 0].copy(), n_neighbors=40)


def test_encoder_and_imputer_fit_transform_equal_fit_then_transform():
    y = np.array([7, 3, 7, 5, 3])
    np.testing.assert_array_equal(np.asarray(ml.LabelEncoder().fit_transform(y)), [2, 0, 2, 1, 0])
    C = _cats()
    for make in (ml.OrdinalEncoder, lambda: ml.OneHotEncoder(sparse_output=False)):
        a = np.asarray(make().fit_transform(C))
        b = np.asarray(make().fit(C).transform(C))
        np.testing.assert_array_equal(a, b)
    X = _pos()
    X[::4, 1] = np.nan
    a = np.asarray(ml.SimpleImputer().fit_transform(X))
    np.testing.assert_array_equal(a, np.asarray(ml.SimpleImputer().fit(X).transform(X)))
    assert not np.isnan(a).any()
