# SPDX-License-Identifier: Apache-2.0
"""Shared class weights against their closed form: class_weight equals the explicit per-class sample weights.

The admission and refusal tests of the shared class_weight mapping are in test_class_weight_boundary.py.
Each parametrized case is independent and can run across pytest workers.
NumPy below prepares independent oracle inputs outside product execution.
"""
import numpy as np
import pytest

from mojolearn import ExtraTreesClassifier, LinearSVC, RandomForestClassifier


@pytest.fixture(autouse=True)
def _reference_fits():
    yield


def _model(family, weight):
    if family == "svc":
        return LinearSVC(class_weight=weight, max_iter=100, tol=1e-5)
    cls = RandomForestClassifier if family == "rf" else ExtraTreesClassifier
    return cls(class_weight=weight, n_estimators=3, max_depth=2,
               bootstrap=False, random_state=7)


def _data():
    return (np.array([[-2, 1], [-1, 2], [0, 0], [1, 0], [2, 1], [3, 0], [4, 2], [5, 1]], dtype="f4"),
            np.array([1, 1, 7, 7, 7, 7, 7, 7], dtype="i8"))


@pytest.mark.parametrize("family", ["svc", "rf", "et"])
@pytest.mark.parametrize("kind", ["balanced", "partial", "unit"])
def test_shared_class_weights_match_explicit_sample_weights(family, kind):
    x, y = _data()
    sample = np.array([0.5, 1, 2, 1, 0.5, 2, 1, 1], dtype="f4")
    if kind == "balanced":
        weight = "balanced"
        per_class = np.where(y == 1, 2.0, 2.0 / 3).astype("f4")
    elif kind == "partial":
        weight = {np.int64(7): np.float64(0.25)}
        per_class = np.where(y == 7, 0.25, 1).astype("f4")
    else:
        weight = {np.float64(1): np.float32(1)}
        per_class = np.ones(y.size, dtype="f4")
    before = sample.copy()
    actual = _model(family, weight).fit(x, y, sample_weight=sample)
    expected = _model(family, None).fit(x, y, sample_weight=sample * per_class)
    np.testing.assert_array_equal(sample, before)
    if family == "svc":
        np.testing.assert_array_equal(np.asarray(actual.coef_), np.asarray(expected.coef_))
        np.testing.assert_array_equal(np.asarray(actual.intercept_), np.asarray(expected.intercept_))
    else:
        np.testing.assert_array_equal(np.asarray(actual.predict_proba(x)), np.asarray(expected.predict_proba(x)))
    np.testing.assert_array_equal(np.asarray(actual.classes_), [1, 7])
