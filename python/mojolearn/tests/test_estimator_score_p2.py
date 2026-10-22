# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Release gate 2026-10-22 (part 2): `score` of every estimator whose score
path changed since 0.8.55, on the real binding path with small models.

A classifier's score is the fraction of its own predictions that match y; a
regressor's is R^2 of its own predictions, restated here in float64 NumPy.
Each estimator also refuses a y whose length differs from X's rows.
"""
import numpy as np
import pytest

import mojolearn as ml


def _cls_data(n=60, d=4, seed=0):
    rng = np.random.default_rng(seed)
    y = (np.arange(n) % 2).astype(np.int64)
    X = (rng.standard_normal((n, d)) + y[:, None] * 2.0).astype(np.float32)
    return X, y


def _reg_data(n=60, d=4, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (X[:, 0] * 2 - X[:, 1] + 0.1 * rng.standard_normal(n)).astype(np.float32)
    return X, y


def _seq_data(n=32, T=4, D=3, seed=2):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, T, D)).astype(np.float32)
    y = (X[:, -1, 0] + 0.5 * X[:, -2, 1]).astype(np.float32)
    return X, y


def _r2(y, p):
    y, p = np.asarray(y, np.float64), np.asarray(p, np.float64).reshape(-1)
    ss_res, ss_tot = ((y - p) ** 2).sum(), ((y - y.mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot


CLASSIFIERS = {
    "AdaBoostClassifier": lambda: ml.AdaBoostClassifier(ml.DecisionTreeClassifier(max_depth=2), n_estimators=3,
                                                        random_state=0),
    "BaggingClassifier": lambda: ml.BaggingClassifier(ml.DecisionTreeClassifier(max_depth=4), n_estimators=3,
                                                      random_state=0),
    "DARTClassifier": lambda: ml.DARTClassifier(n_estimators=4, random_state=0),
    "ExtraTreesClassifier": lambda: ml.ExtraTreesClassifier(n_estimators=4, random_state=7),
    "RandomForestClassifier": lambda: ml.RandomForestClassifier(n_estimators=4, max_depth=5, random_state=7),
    "KNeighborsClassifier": lambda: ml.KNeighborsClassifier(n_neighbors=3),
    "LinearSVC": lambda: ml.LinearSVC(max_iter=200),
    "MLPClassifier": lambda: ml.MLPClassifier(hidden_layer_sizes=(5,), max_iter=5, random_state=0),
}

REGRESSORS = {
    "AdaBoostRegressor": lambda: ml.AdaBoostRegressor(ml.DecisionTreeRegressor(max_depth=3), n_estimators=3,
                                                      random_state=0),
    "BaggingRegressor": lambda: ml.BaggingRegressor(ml.DecisionTreeRegressor(max_depth=4), n_estimators=3,
                                                    random_state=0),
    "DARTRegressor": lambda: ml.DARTRegressor(n_estimators=4, num_leaves=15, max_depth=5, min_child_samples=5,
                                              random_state=0),
    "ExtraTreesRegressor": lambda: ml.ExtraTreesRegressor(n_estimators=4, random_state=7),
    "RandomForestRegressor": lambda: ml.RandomForestRegressor(n_estimators=4, max_depth=5, random_state=7),
    "KNeighborsRegressor": lambda: ml.KNeighborsRegressor(n_neighbors=3),
    "KernelRidge": lambda: ml.KernelRidge(alpha=1.0),
    "GaussianProcessRegressor": lambda: ml.GaussianProcessRegressor(),
    "ElasticNet": lambda: ml.ElasticNet(alpha=0.01, max_iter=50),
    "Lasso": lambda: ml.Lasso(alpha=0.01, max_iter=50),
    "QNRegressor": lambda: ml.QNRegressor(l2_strength=0.5),
    "LinearSVR": lambda: ml.LinearSVR(C=4.0, epsilon=0.05),
    "MLPRegressor": lambda: ml.MLPRegressor(hidden_layer_sizes=(5,), max_iter=5, random_state=0),
}

SEQ_CLASSIFIERS = {
    "LSTMClassifier": lambda: ml.LSTMClassifier(hidden_size=4, batch_size=16, max_epochs=2),
    "GRUClassifier": lambda: ml.GRUClassifier(hidden_size=4, batch_size=16, max_epochs=2),
    "RNNClassifier": lambda: ml.RNNClassifier(hidden_size=4, batch_size=16, max_epochs=2),
}

SEQ_REGRESSORS = {
    "LSTMRegressor": lambda: ml.LSTMRegressor(hidden_size=4, batch_size=16, max_epochs=2),
    "GRURegressor": lambda: ml.GRURegressor(hidden_size=4, batch_size=16, max_epochs=2),
    "RNNRegressor": lambda: ml.RNNRegressor(hidden_size=4, batch_size=16, max_epochs=2),
}


def _check_classifier_score(model, X, y):
    pred = np.asarray(model.predict(X)).reshape(-1)
    assert model.score(X, y) == pytest.approx(float(np.mean(pred == y)), abs=1e-6)
    with pytest.raises(ValueError):
        model.score(X, y[:-1])


def _check_regressor_score(model, X, y):
    assert model.score(X, y) == pytest.approx(_r2(y, model.predict(X)), abs=1e-3)
    with pytest.raises(ValueError):
        model.score(X, y[:-1])


@pytest.mark.parametrize("name", sorted(CLASSIFIERS))
def test_classifier_score_matches_its_predictions_and_refuses_short_y(name):
    X, y = _cls_data()
    _check_classifier_score(CLASSIFIERS[name]().fit(X, y), X, y)


@pytest.mark.parametrize("name", sorted(REGRESSORS))
def test_regressor_score_is_r2_of_its_predictions_and_refuses_short_y(name):
    X, y = _reg_data()
    _check_regressor_score(REGRESSORS[name]().fit(X, y), X, y)


@pytest.mark.parametrize("name", sorted(SEQ_CLASSIFIERS))
def test_recurrent_classifier_score(name):
    X, y = _seq_data()
    yc = (y > 0).astype(np.int64)
    _check_classifier_score(SEQ_CLASSIFIERS[name]().fit(X, yc), X, yc)


@pytest.mark.parametrize("name", sorted(SEQ_REGRESSORS))
def test_recurrent_regressor_score(name):
    X, y = _seq_data()
    _check_regressor_score(SEQ_REGRESSORS[name]().fit(X, y), X, y)


def test_stacking_regressor_fit_transform_stacks_member_predictions():
    X, y = _reg_data()
    m = ml.StackingRegressor([("a", ml.DecisionTreeRegressor(max_depth=3)),
                              ("b", ml.DecisionTreeRegressor(max_depth=5))],
                             final_estimator=ml.DecisionTreeRegressor(max_depth=3), cv=3)
    Z = np.asarray(m.fit_transform(X, y))
    ref = np.column_stack([np.asarray(e.predict(X), np.float64).reshape(-1) for e in m.estimators_])
    assert Z.shape == (len(X), 2)
    np.testing.assert_allclose(Z, ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(Z, np.asarray(m.transform(X)), rtol=0, atol=0)
    with pytest.raises(ValueError):
        m.transform(X[:, :2])


def test_stacking_regressor_fit_transform_refuses_parallel_jobs():
    X, y = _reg_data()
    with pytest.raises(NotImplementedError):
        ml.StackingRegressor([("a", ml.DecisionTreeRegressor(max_depth=3))], n_jobs=2).fit_transform(X, y)
