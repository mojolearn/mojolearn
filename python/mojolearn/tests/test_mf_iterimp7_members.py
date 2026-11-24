# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IterativeImputer after the door rounds' deletion (lane mf-iterimp7): every supplied estimator runs through its
native member, so the callers the door rounds used to serve are checked here against the rounds written out by
hand with the estimator's OWN public fit and predict (scikit-learn's algorithm, sklearn/impute/_iterative.py
`_impute_one_feature`: a clone per step fitted on the observed rows of the step's feature, its predictions clipped
into the missing rows). The door rounds ran exactly these public operations, so the old door result is this oracle.

  * LabelPropagation / LabelSpreading: a target column holding -1 (the estimator's unlabeled marker), now admitted
    with deferred labels and a per-step marker word; their classes_ leave -1 out, as the estimator's own fit does.
  * a search over a classifier (GridSearchCV over LogisticRegression): the classifier route is chosen by the
    member kind's capability bit, not the class's `_estimator_type`.
  * refusals: a connected class with no predict (NearestNeighbors) in scikit-learn's parameter words; a fitted step
    whose estimator was replaced by an unfitted one in transform.
"""
import numpy as np
import pytest

import mojolearn as ml
from mojolearn.model_selection import GridSearchCV


def _labels(seed=0, n=240, d=4, low=-1, high=3):
    # integer-valued columns (classes low .. high-1, so -1 is a class when low == -1), 25% missing
    rng = np.random.default_rng(seed)
    base = rng.integers(low, high, n)
    X = np.stack([np.clip(base + ((j * 7 + rng.integers(0, 2, n)) % 2), low, high - 1) for j in range(d)],
                 axis=1).astype(np.float32)
    X[rng.random((n, d)) < 0.25] = np.nan
    return X


def _by_hand(imputer, X):
    """The fitted imputer's rounds redone with each step's own estimator class through its public fit and
    predict (one fresh estimator per step from `make`, as scikit-learn clones), in the fitted order, from the
    initial imputation. Returns the imputed (n, kept) block."""
    mask = np.isnan(np.asarray(X, dtype=np.float32))[:, imputer._keep]
    xt = np.asarray(imputer.initial_imputer_.transform(X), dtype=np.float32).copy()
    lo = np.asarray(imputer._bounds_k[0::2], dtype=np.float64)
    hi = np.asarray(imputer._bounds_k[1::2], dtype=np.float64)
    for j, nbl, fitted in imputer.imputation_sequence_:
        nbl = list(nbl)
        miss = mask[:, j]
        est = type(fitted)(**fitted.get_params(deep=False))
        est.fit(xt[~miss][:, nbl], xt[~miss, j])
        if miss.any():
            pred = np.asarray(est.predict(xt[miss][:, nbl]), dtype=np.float64)
            xt[miss, j] = np.clip(pred, lo[j], hi[j]).astype(np.float32)
    return xt


@pytest.mark.parametrize("make", [
    lambda: ml.LabelPropagation(kernel="knn", n_neighbors=5, max_iter=50),
    lambda: ml.LabelSpreading(kernel="knn", n_neighbors=5, max_iter=50),
])
def test_label_propagation_marker(make):
    X = _labels(1)
    imp = ml.IterativeImputer(estimator=make(), max_iter=1, imputation_order="ascending")
    got = np.asarray(imp.fit_transform(X))
    want = _by_hand(imp, X)
    assert np.array_equal(got[:, :want.shape[1]], want), np.argwhere(got[:, :want.shape[1]] != want)[:4]
    for _, _, est in imp.imputation_sequence_:
        assert -1 not in [float(c) for c in np.asarray(est.classes_)], est.classes_
    # transform over the held states equals the fitted pass on the same rows (one round, same order)
    assert np.array_equal(np.asarray(imp.transform(X)), got)


def test_search_over_classifier():
    X = _labels(2, low=0, high=3)
    search = GridSearchCV(ml.LogisticRegression(max_iter=200), {"C": [1.0]}, cv=2, refit=True)
    imp = ml.IterativeImputer(estimator=search, max_iter=1)
    got = np.asarray(imp.fit_transform(X))
    want = _by_hand(imp, X)
    assert np.array_equal(got, want), np.argwhere(got != want)[:4]


def test_no_predict_refused_in_sklearn_words():
    with pytest.raises(ValueError) as error:
        ml.IterativeImputer(estimator=ml.NearestNeighbors(n_neighbors=3)).fit(_labels(3))
    assert str(error.value) == ("The 'estimator' parameter of IterativeImputer must be None or an object implementing "
                                "'fit' and 'predict'. Got a NearestNeighbors instance instead."), str(error.value)


def test_replaced_step_refused_in_transform():
    X = _labels(4, low=0, high=3)
    imp = ml.IterativeImputer(estimator=ml.Ridge(alpha=1.0), max_iter=2).fit(X)
    j, nbl, _ = imp.imputation_sequence_[0]
    imp.imputation_sequence_[0] = (j, nbl, ml.Ridge(alpha=1.0))  # an unfitted estimator in place of the fitted one
    with pytest.raises(RuntimeError) as error:
        imp.transform(X)
    words = str(error.value)
    assert "IterativeImputer's imputation_sequence_ step 0 holds a Ridge" in words, words
    assert words.endswith("; fit the imputer again"), words


if __name__ == "__main__":
    for case in (lambda: ml.LabelPropagation(kernel="knn", n_neighbors=5, max_iter=50),
                 lambda: ml.LabelSpreading(kernel="knn", n_neighbors=5, max_iter=50)):
        test_label_propagation_marker(case)
    test_search_over_classifier()
    test_no_predict_refused_in_sklearn_words()
    test_replaced_step_refused_in_transform()
    print("PASS test_mf_iterimp7_members")
