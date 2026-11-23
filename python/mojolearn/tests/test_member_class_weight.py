# SPDX-License-Identifier: Apache-2.0
"""Class weights and precomputed kernels of classifier and kernel MEMBERS against the estimators' own doors (lane
mf-adm3, 2026-11-23; core/member_class_weight.mojo, bindings/member_class_weight_unpack.mojo).

The oracle is the estimator's own public fit (its door) on the rows the member saw: a CalibratedClassifierCV fold's
member must hold the same fitted words as a clone fitted by hand on that fold's train rows and labels, for
class_weight 'balanced', a dict over integer labels and a dict over string labels (scikit-learn reads the dict
against the fold's labels: so do both). A fold that lacks a class the dict names raises the door's absent-label
refusal. kernel='precomputed': AdaBoost's first SVC member equals SVC(kernel='precomputed') fitted by hand with
AdaBoost's first weights, and a calibrated fold's non-square kernel rows raise the door's square refusal.
"""
import numpy as np
import pytest

import mojolearn
from mojolearn.calibration import CalibratedClassifierCV


def _data(k, n=150, seed=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    codes = (np.arange(n) * 7 % (k * (k + 1) // 2))
    codes = np.searchsorted(np.cumsum(np.arange(1, k + 1)) - 1, codes)  # unbalanced: class c ~ (c + 1) parts
    X[:, 0] += codes
    return X, codes.astype(np.int64)


def _member(name, class_weight):
    if name == "lr":
        return mojolearn.LogisticRegression(max_iter=200, class_weight=class_weight)
    if name == "svc":
        return mojolearn.LinearSVC(C=0.5, class_weight=class_weight)
    if name == "svk":
        return mojolearn.SVC(C=1.0, gamma=0.5, class_weight=class_weight)
    if name == "sgd":
        return mojolearn.SGDClassifier(max_iter=20, random_state=3, class_weight=class_weight)
    if name == "ridgec":
        return mojolearn.RidgeClassifier(alpha=0.7, class_weight=class_weight)
    return mojolearn.LogisticRegressionCV(Cs=3, cv=3, max_iter=100, class_weight=class_weight)


def _fitted_words(est):
    words = []
    for attr in ("coef_", "intercept_", "dual_coef_"):
        if hasattr(est, attr):
            words.append(np.asarray(getattr(est, attr)))
    return words


def _folds(n):
    rows = np.arange(n)
    return [(rows[rows % 3 != f], rows[rows % 3 == f]) for f in range(3)]


@pytest.mark.parametrize("name", ["lr", "svc", "svk", "sgd", "ridgec", "lrcv"])
@pytest.mark.parametrize("weights", ["balanced", "dint", "dstr"])
def test_calibrated_member_equals_door(name, weights):
    k = 3
    X, codes = _data(k)
    labels = ["a", "b", "c"] if weights == "dstr" else [0, 1, 2]
    y = np.asarray(labels, dtype=object if weights == "dstr" else np.int64)[codes]
    cw = "balanced" if weights == "balanced" else {labels[0]: 0.4, labels[1]: 2.5}
    folds = _folds(len(y))
    m = CalibratedClassifierCV(_member(name, cw), cv=folds, ensemble=True).fit(X, y)
    for f, (train, _) in enumerate(folds):
        door = _member(name, cw).fit(X[train], y[train])
        got = _fitted_words(m.calibrated_classifiers_[f][0])
        want = _fitted_words(door)
        assert len(got) == len(want) > 0
        for a, b in zip(got, want):
            np.testing.assert_array_equal(a, b)


def test_calibrated_dict_naming_a_lacking_class_is_refused():
    k = 3
    X, y = _data(k)
    rows = np.arange(len(y))
    train0 = rows[(rows % 3 != 0) & (y != 2)]
    cv = [(train0, rows[rows % 3 == 0])] + _folds(len(y))[1:]
    cw = {0: 0.4, 2: 2.5}
    with pytest.raises(Exception) as door:
        mojolearn.LogisticRegression(class_weight=cw).fit(X[train0], y[train0])
    with pytest.raises(Exception) as member:
        CalibratedClassifierCV(mojolearn.LogisticRegression(class_weight=cw), cv=cv).fit(X, y)
    assert str(member.value) == str(door.value)


def test_precomputed_svc_member_equals_door():
    X, y = _data(2, n=90)
    K = (X @ X.T).astype(np.float32)
    ada = mojolearn.AdaBoostClassifier(mojolearn.SVC(kernel="precomputed"), n_estimators=1, random_state=0).fit(K, y)
    # AdaBoost's first weights: 1 / n as float32 (bindings/trees_public_boost_door.mojo `_trees_normalized_weights_native`)
    door = mojolearn.SVC(kernel="precomputed").fit(K, y, sample_weight=np.full(len(y), 1.0 / len(y), dtype=np.float32))
    for a, b in zip(_fitted_words(ada.estimators_[0]), _fitted_words(door)):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(ada.estimators_[0].predict(K), door.predict(K))


def test_precomputed_fold_rows_are_refused_in_the_doors_words():
    X, y = _data(2, n=90)
    K = (X @ X.T).astype(np.float32)
    train = _folds(len(y))[0][0]
    with pytest.raises(Exception) as door:
        mojolearn.SVC(kernel="precomputed").fit(K[train], y[train])
    with pytest.raises(Exception) as member:
        CalibratedClassifierCV(mojolearn.SVC(kernel="precomputed"), cv=_folds(len(y))).fit(K, y)
    assert str(member.value) == str(door.value)
