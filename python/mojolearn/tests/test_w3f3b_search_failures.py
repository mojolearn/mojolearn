# SPDX-License-Identifier: Apache-2.0
"""Lane w3f3-b: the warnings and errors scikit-learn raises for a search's or a cv tool's failures under a numeric
error_score (scikit-learn 1.9 model_selection/_validation.py `_score`, `_warn_or_raise_about_fit_failures`;
model_selection/_search.py `_format_results`), and member-only estimators served through another estimator's call.
"""
import warnings

import numpy as np
import pytest

import mojolearn as ml
from mojolearn import model_selection as ms


def _regression(n=60, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4)).astype(np.float32)
    y = (X @ np.array([1.0, -2.0, 0.5, 0.0], dtype=np.float32)).astype(np.float32)
    return X, y


def _messages(seen, category):
    return [str(w.message) for w in seen if issubclass(w.category, category)]


def test_some_fits_failing_warn_once_with_the_summary():
    X, y = _regression()
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        search = ms.GridSearchCV(ml.Ridge(), {"alpha": [-1.0, 1.0]}, cv=3, error_score=0.5).fit(X, y)
    words = [m for m in _messages(seen, RuntimeWarning) if "fits failed out of a total of" in m]
    assert len(words) == 1, words
    assert words[0].startswith("\n3 fits failed out of a total of 6.\n"), words[0]
    assert "will be set to 0.5." in words[0]
    assert "3 fits failed with the following error:" in words[0]
    assert [search.cv_results_["split%d_test_score" % f][0] for f in range(3)] == [0.5, 0.5, 0.5]
    assert search.best_index_ == 1


def test_every_fit_failing_raises():
    X, y = _regression()
    with pytest.raises(ValueError, match="All the 6 fits failed"):
        ms.GridSearchCV(ml.Ridge(), {"alpha": [-1.0, -2.0]}, cv=3, error_score=np.nan).fit(X, y)


def test_a_score_failing_warns_and_the_means_are_non_finite():
    X, y = _regression()
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        search = ms.GridSearchCV(ml.StandardScaler(), {"with_mean": [True, False]}, cv=3, scoring="r2",
                                 error_score=np.nan).fit(X, y)
    user = _messages(seen, UserWarning)
    failed = [m for m in user if m.startswith("Scoring failed. The score on this train-test partition for these "
                                               "parameters will be set to nan. Details: \n")]
    assert len(failed) == 6, user
    assert any(m.startswith("One or more of the test scores are non-finite: ") for m in user), user
    assert search.best_index_ == 0


def test_cross_val_score_takes_a_numeric_error_score():
    X, y = _regression()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = ms.cross_val_score(ml.StandardScaler(), X, y, cv=3, scoring="r2", error_score=-1.0)
    assert np.asarray(scores).tolist() == [-1.0, -1.0, -1.0]
