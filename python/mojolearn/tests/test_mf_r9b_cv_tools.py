# SPDX-License-Identifier: Apache-2.0
"""The cv tools lane mf-r9b (2026-11-23) routed onto the typed search RUN only (bindings/msel_cv_door.mojo
`msel_cv_by_search`, bindings/msel_search_door.mojo `_curve_by_search`, `_mean_test`).

Before, each tool kept the Python units (`msel_cv_units`) beside the RUN for what the RUN's door answered None. Now
every call is the RUN or a native refusal in scikit-learn 1.9's words:

- validation_curve with y None (an unsupervised base) is the RUN, equal to the grid search over the same values;
- validation_curve with an empty param_range splits, then raises `_aggregate_score_dicts([])`'s IndexError;
- validation_curve with several scorers raises validate_params' InvalidParameterError (was the units' KeyError);
- learning_curve's train_sizes follow `_translate_train_sizes`: empty, fraction and absolute words with the min and
  max, duplicates dropped with the RuntimeWarning, n_max the FIRST fold's train size (a shorter fold's size takes
  all its rows).
Each test names its fail-proof edit.
"""
import warnings

import numpy as np
import pytest

import mojolearn as ml
from mojolearn import model_selection as ms


def _regression(n=60, d=3, seed=21):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] * 2.0 - X[:, 1] + 0.1 * rng.normal(size=n)).astype(np.float32)
    return X, y


def _ridge():
    return ml.Ridge(alpha=1.0, numeric_mode="identical")


def test_validation_curve_without_y_is_the_grid_search():
    # Fail-proof: in `msel_validation_curve` raise for `args[2] is None` before `_curve_by_search`: this raises.
    X, _ = _regression()
    est = ml.KMeans(n_clusters=2, n_init=1, random_state=0)
    train, test = ms.validation_curve(est, X, None, param_name="n_clusters", param_range=[2, 3], cv=3)
    search = ms.GridSearchCV(ml.KMeans(n_clusters=2, n_init=1, random_state=0), {"n_clusters": [2, 3]}, cv=3,
                             refit=False, return_train_score=True).fit(X)
    for i in range(3):
        np.testing.assert_array_equal(np.asarray(train)[:, i],
                                      np.asarray(search.cv_results_["split%d_train_score" % i]))
        np.testing.assert_array_equal(np.asarray(test)[:, i],
                                      np.asarray(search.cv_results_["split%d_test_score" % i]))


def test_validation_curve_empty_range_is_sklearns_index_error():
    # Fail-proof: drop the `len(param_range) == 0` branch: the carrier's empty grid raises other words.
    X, y = _regression()
    with pytest.raises(IndexError, match="list index out of range"):
        ms.validation_curve(_ridge(), X, y, param_name="alpha", param_range=[], cv=3)


def test_validation_curve_empty_range_splits_first():
    # Fail-proof: raise the IndexError before `msel_cv_folds`: the split's words are not raised.
    X, y = _regression()
    with pytest.raises(ValueError, match="n_splits"):
        ms.validation_curve(_ridge(), X, y, param_name="alpha", param_range=[], cv=1)


@pytest.mark.parametrize("scoring", [["r2"], ("r2", "neg_mean_squared_error"), {"r2": "r2"}])
def test_validation_curve_several_scorers_is_invalid_parameter(scoring):
    # Fail-proof: remove `_one_scorer_check("validation_curve", ...)`: the carrier's multi-metric columns raise
    # KeyError instead.
    X, y = _regression()
    with pytest.raises(ValueError, match="The 'scoring' parameter of validation_curve must be a str among"):
        ms.validation_curve(_ridge(), X, y, param_name="alpha", param_range=[0.5, 1.0], cv=3, scoring=scoring)


def test_learning_curve_empty_train_sizes_is_numpys_words():
    # Fail-proof: in `_train_sizes` return an empty list for `count == 0`: the cv tool invariant raises instead.
    X, y = _regression()
    with pytest.raises(ValueError, match="zero-size array to reduction operation minimum which has no identity"):
        ms.learning_curve(_ridge(), X, y, cv=3, train_sizes=[])


def test_learning_curve_fraction_words_name_min_and_max():
    # Fail-proof: format the bounds with String(Float64) instead of `%f`: the words differ.
    X, y = _regression()
    with pytest.raises(ValueError, match=r"must be within \(0, 1\], but is within \[0\.000000, 0\.500000\]\."):
        ms.learning_curve(_ridge(), X, y, cv=3, train_sizes=[0.0, 0.5])


def test_learning_curve_absolute_words_name_min_and_max():
    # Fail-proof: check each value alone (the old per-value words [a, a]): the words differ.
    X, y = _regression()
    with pytest.raises(ValueError, match=r"must be within \(0, 40\], but is within \[5, 1000\]\."):
        ms.learning_curve(_ridge(), X, y, cv=3, train_sizes=[5, 1000])


def test_learning_curve_duplicates_warn_and_are_dropped():
    # Fail-proof: drop the `msel_warn` call in `_train_sizes`: no RuntimeWarning.
    X, y = _regression()
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        sizes, train, test = ms.learning_curve(_ridge(), X, y, cv=3, train_sizes=[0.5, 0.5, 1.0])
    np.testing.assert_array_equal(np.asarray(sizes), [20, 40])
    assert np.asarray(train).shape == (2, 3)
    assert any(issubclass(w.category, RuntimeWarning) and "2 instead of 3" in str(w.message) for w in seen)


def test_learning_curve_n_max_is_the_first_folds_train_size():
    # Fail-proof: take n_max as the minimum over folds (the old rule): 45 is refused as above (0, 40].
    X, y = _regression()
    rows = np.arange(60, dtype=np.int64)
    folds = [(rows[:50], rows[50:]), (rows[20:60], rows[:20])]  # trains of 50 and 40 rows
    sizes, train, test = ms.learning_curve(_ridge(), X, y, cv=folds, train_sizes=[45], shuffle=False)
    np.testing.assert_array_equal(np.asarray(sizes), [45])
    for f, (tr, te) in enumerate(folds):
        prefix = tr[:45]  # the second fold's 40 rows are all taken (`train[:n]`)
        fitted = _ridge().fit(X[prefix], y[prefix])
        assert np.asarray(train)[0, f] == fitted.score(X[prefix], y[prefix])
        assert np.asarray(test)[0, f] == fitted.score(X[te], y[te])
