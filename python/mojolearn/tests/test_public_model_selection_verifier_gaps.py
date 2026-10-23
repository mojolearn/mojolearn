# SPDX-License-Identifier: Apache-2.0
"""Real estimator/output coverage for the model-selection verifier audit gaps.

Controllers and NumPy calculations below are independent test oracles. These
tests do not substitute for native runtime audits or claim unexecuted coverage.
"""
import numpy as np
import pytest

import mojolearn as ml
from mojolearn import metrics, model_selection as ms


def data():
    rng = np.random.default_rng(407)
    x = rng.normal(size=(36, 3)).astype("f4")
    y = (x @ np.array([0.5, -0.25, 0.75], dtype="f4") + np.float32(0.2)).astype("f4")
    return x, y


def negative_mse(model, x, y):
    prediction = np.asarray(model.predict(x), dtype="f8")
    return -np.mean((prediction - y.astype("f8")) ** 2)


def test_check_cv_consumes_each_sample_once_and_scorer_factories_match_r2():
    x, y = data()
    splitter = ms.check_cv(3, y, classifier=False)
    folds = [(np.asarray(a), np.asarray(b)) for a, b in splitter.split(x, y)]
    assert len(folds) == 3
    np.testing.assert_array_equal(np.sort(np.concatenate([b for _, b in folds])), np.arange(len(x)))
    for train, test in folds:
        assert len(np.intersect1d(train, test)) == 0
        assert len(train) + len(test) == len(x)
    names = ms.get_scorer_names()
    assert names == sorted(set(names)) and "r2" in names
    model = ml.Ridge(alpha=0.5, numeric_mode="identical").fit(x, y)
    prediction = np.asarray(model.predict(x), dtype="f8")
    expected = 1 - np.sum((y.astype("f8") - prediction) ** 2) / np.sum((y.astype("f8") - y.astype("f8").mean()) ** 2)
    named = ms.get_scorer("r2")
    made = ms.make_scorer(metrics.r2_score, response_method="predict", numeric_mode="identical")
    assert named(model, x, y) == pytest.approx(expected, rel=2e-5, abs=2e-6)
    assert made(model, x, y) == pytest.approx(expected, rel=2e-5, abs=2e-6)
    with pytest.raises(ValueError):
        ms.get_scorer("missing-metric-name")


def test_validation_curve_matches_independently_fit_fold_models():
    x, y = data()
    alphas = [0.5, 2.0]
    folds = list(ms.KFold(3).split(x, y))
    train, test = ms.validation_curve(ml.Ridge(numeric_mode="identical"), x, y,
                                      param_name="alpha", param_range=alphas, cv=folds,
                                      scoring="neg_mean_squared_error", n_jobs=1, error_score="raise")
    expected_train, expected_test = [], []
    for alpha in alphas:
        train_row, test_row = [], []
        for tr, te in folds:
            tr, te = np.asarray(tr), np.asarray(te)
            fitted = ml.Ridge(alpha=alpha, numeric_mode="identical").fit(x[tr], y[tr])
            train_row.append(negative_mse(fitted, x[tr], y[tr]))
            test_row.append(negative_mse(fitted, x[te], y[te]))
        expected_train.append(train_row)
        expected_test.append(test_row)
    np.testing.assert_allclose(np.asarray(train), expected_train, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(np.asarray(test), expected_test, rtol=2e-5, atol=2e-6)


def test_learning_curve_matches_native_models_on_each_training_prefix():
    x, y = data()
    folds = list(ms.KFold(3).split(x, y))
    sizes, train, test = ms.learning_curve(ml.Ridge(numeric_mode="identical"), x, y,
                                           cv=folds, train_sizes=[0.5, 1.0], shuffle=False,
                                           scoring="neg_mean_squared_error", n_jobs=1, error_score="raise")
    np.testing.assert_array_equal(np.asarray(sizes), [12, 24])
    expected_train, expected_test = [], []
    for size in [12, 24]:
        train_row, test_row = [], []
        for tr, te in folds:
            tr, te = np.asarray(tr)[:size], np.asarray(te)
            fitted = ml.Ridge(numeric_mode="identical").fit(x[tr], y[tr])
            train_row.append(negative_mse(fitted, x[tr], y[tr]))
            test_row.append(negative_mse(fitted, x[te], y[te]))
        expected_train.append(train_row)
        expected_test.append(test_row)
    np.testing.assert_allclose(np.asarray(train), expected_train, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(np.asarray(test), expected_test, rtol=2e-5, atol=2e-6)


def test_randomized_search_fit_predict_score_and_refit_state():
    x, y = data()
    search = ms.RandomizedSearchCV(ml.Ridge(numeric_mode="identical"), {"alpha": [0.5, 2.0]},
                                   n_iter=2, random_state=7, cv=3, n_jobs=1, error_score="raise")
    assert search.fit(x, y) is search
    assert search.best_params_["alpha"] in (0.5, 2.0)
    assert len(search.cv_results_["params"]) == 2
    expected = ml.Ridge(alpha=search.best_params_["alpha"], numeric_mode="identical").fit(x, y)
    np.testing.assert_array_equal(np.asarray(search.predict(x)), np.asarray(expected.predict(x)))
    assert search.score(x, y) == expected.score(x, y)
    unrefitted = ms.RandomizedSearchCV(ml.Ridge(numeric_mode="identical"), {"alpha": [0.5]},
                                      n_iter=1, refit=False, cv=3, n_jobs=1, error_score="raise").fit(x, y)
    with pytest.raises(AttributeError):
        unrefitted.predict(x)


def test_randomized_search_classification_response_delegations():
    x, _ = data()
    y = (x[:, 0] > 0).astype("i8")
    search = ms.RandomizedSearchCV(ml.LogisticRegression(max_iter=30),
                                   {"C": [0.5, 2.0]}, n_iter=2, random_state=7, cv=3,
                                   n_jobs=1, error_score="raise").fit(x, y)
    probability = np.asarray(search.predict_proba(x))
    decision = np.asarray(search.decision_function(x))
    np.testing.assert_array_equal(probability, np.asarray(search.best_estimator_.predict_proba(x)))
    np.testing.assert_array_equal(decision, np.asarray(search.best_estimator_.decision_function(x)))
    np.testing.assert_allclose(probability.sum(axis=1), 1, rtol=1e-6)
    assert probability.shape == (len(x), 2)
    assert np.isfinite(decision).all()


def test_randomized_search_transform_delegates_to_refitted_native_scaler():
    x, _ = data()
    reconstruction = ms.make_scorer(metrics.mean_squared_error, greater_is_better=False,
                                     response_method="transform", numeric_mode="identical")
    search = ms.RandomizedSearchCV(ml.StandardScaler(numeric_mode="identical"), {"with_mean": [False, True]},
                                   scoring=reconstruction, n_iter=2, random_state=7, cv=3,
                                   n_jobs=1, error_score="raise").fit(x, x)
    np.testing.assert_array_equal(np.asarray(search.transform(x)), np.asarray(search.best_estimator_.transform(x)))
