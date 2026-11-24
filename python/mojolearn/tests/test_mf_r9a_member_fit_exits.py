# SPDX-License-Identifier: Apache-2.0
"""The member fit exits lane mf-r9a (2026-11-23) closed (bindings/msel_member_units_door.mojo `_member_fit`,
bindings/msel_search_door.mojo `_search_member_fit`).

Before, each case answered None / False and the search or cv tool kept its Python units. Now every input is fitted
by the typed search RUN or refused natively, before any fit, in scikit-learn 1.9's words:

- an X other than C-order float32 (float64, Fortran order) is admitted once, as the estimator's own fit converts it;
- y's rows against X's: check_consistent_length's words;
- non-finite X: check_array's words naming the base estimator; with a numeric error_score the all-fits-failed words;
- a classifier's non-whole float labels: check_classification_targets' words; NaN labels `_assert_all_finite`'s;
- a one-column 2-D y of a classifier is read as its 1-D column.
Each test names its fail-proof edit.
"""
import numpy as np
import pytest

import mojolearn as ml
from mojolearn import model_selection as ms


def _classification(n=40, d=3, seed=11):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(np.int64)
    return X, y


def _regression(n=40, d=3, seed=12):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] * 2.0 - X[:, 1] + 0.1 * rng.normal(size=n)).astype(np.float32)
    return X, y


def _search(est, grid, X, y, error_score="raise", scoring=None):
    return ms.GridSearchCV(est, grid, cv=3, scoring=scoring, refit=False, error_score=error_score).fit(X, y)


@pytest.mark.parametrize("convert", [
    lambda X: X.astype(np.float64),
    lambda X: np.asfortranarray(X),
])
def test_x_not_c_float32_is_admitted_once(convert):
    # Fail-proof: in `_member_fit` replace the `admit_f32` branch by `raise Error("x")`: the search raises.
    X, y = _regression()
    got = _search(ml.Ridge(), {"alpha": [0.5, 1.0]}, convert(X), y, scoring="r2")
    want = _search(ml.Ridge(), {"alpha": [0.5, 1.0]}, X, y, scoring="r2")
    np.testing.assert_array_equal(np.asarray(got.cv_results_["mean_test_score"]),
                                  np.asarray(want.cv_results_["mean_test_score"]))


def test_cross_validate_float64_x_equals_float32():
    # Fail-proof: as above; cross_validate's carrier search raises.
    X, y = _regression()
    got = ms.cross_validate(ml.Ridge(), X.astype(np.float64), y, cv=3, scoring="r2")
    want = ms.cross_validate(ml.Ridge(), X, y, cv=3, scoring="r2")
    np.testing.assert_array_equal(np.asarray(got["test_score"]), np.asarray(want["test_score"]))


def test_y_rows_against_x_rows():
    # Fail-proof: in `_member_fit` drop the `_inconsistent_words` check: another refusal (or a fit) follows.
    X, y = _classification()
    with pytest.raises(ValueError, match=r"inconsistent numbers of samples: \[40, 39\]"):
        _search(ml.LogisticRegression(), {"C": [0.5, 1.0]}, X, y[:39])


def test_nonfinite_x_raise_names_the_base():
    # Fail-proof: in `_base_name` answer the search's own class name: the words name GridSearchCV.
    X, y = _classification()
    X[3, 1] = np.nan
    with pytest.raises(ValueError, match=r"Input X contains NaN or infinity \(LogisticRegression\)\."):
        _search(ml.LogisticRegression(), {"C": [0.5, 1.0]}, X, y)


def test_nonfinite_x_numeric_error_score_all_fits_failed():
    # Fail-proof: in `_fit_words` return `words` for a numeric error_score: the plain check_array words.
    X, y = _classification()
    X[3, 1] = np.inf
    with pytest.raises(ValueError, match="All the 6 fits failed"):
        _search(ml.LogisticRegression(), {"C": [0.5, 1.0]}, X, y, error_score=np.nan)


def test_continuous_labels_refused_in_sklearn_words():
    # Fail-proof: in `_label_kind` never answer _LABELS_CONTINUOUS: the member fits float codes.
    X, _ = _classification()
    y = np.linspace(0.0, 1.0, X.shape[0]).astype(np.float64)
    with pytest.raises(ValueError, match="Unknown label type: continuous. Maybe you are trying to fit a classifier"):
        _search(ml.LogisticRegression(), {"C": [0.5, 1.0]}, X, y)
    with pytest.raises(ValueError, match="All the 6 fits failed"):
        _search(ml.LogisticRegression(), {"C": [0.5, 1.0]}, X, y, error_score=np.nan)


def test_nan_labels_refused_before_any_fit():
    # Fail-proof: in `_label_kind` drop the NaN test: the labels read as continuous.
    X, y = _classification()
    y = y.astype(np.float64)
    y[5] = np.nan
    with pytest.raises(ValueError, match=r"Input y contains NaN\."):
        _search(ml.LogisticRegression(), {"C": [0.5, 1.0]}, X, y, error_score=np.nan)


def test_one_column_y_is_its_1d_column():
    # Fail-proof: in `_encode_outputs` treat one output as a 2-D y: the RUN fits a one-output multi-output target.
    X, y = _classification()
    got = _search(ml.LogisticRegression(), {"C": [0.5, 1.0]}, X, y.reshape(-1, 1), scoring="accuracy")
    want = _search(ml.LogisticRegression(), {"C": [0.5, 1.0]}, X, y, scoring="accuracy")
    np.testing.assert_array_equal(np.asarray(got.cv_results_["mean_test_score"]),
                                  np.asarray(want.cv_results_["mean_test_score"]))
