# SPDX-License-Identifier: Apache-2.0
"""Class weights and precomputed kernels of classifier and kernel MEMBERS against the estimators' own doors (lane
mf-adm3, 2026-11-23; core/member_class_weight.mojo, bindings/member_class_weight_unpack.mojo).

The oracle is the estimator's own public fit (its door) on the rows the member saw: a CalibratedClassifierCV fold's
member must hold the same fitted words as a clone fitted by hand on that fold's train rows and labels, for
class_weight 'balanced', a dict over integer labels and a dict over string labels (scikit-learn reads the dict
against the fold's labels: so do both). The equality test's folds are contiguous blocks, so every fold's train rows hold
every class (the labels repeat with period 6; folds by row % 3 put every 'a' in fold 0's test rows, and fold 0's door
then refuses the dict as scikit-learn 1.9.1's compute_class_weight does, utils/class_weight.py:101-120). A fold that
lacks a class the dict names raises the door's absent-label refusal. kernel='precomputed': a VotingClassifier's SVC
member, fitted on the whole square kernel, equals SVC(kernel='precomputed') fitted by hand (AdaBoost fits only tree
members and refuses an SVC base by name), and a calibrated fold's non-square kernel rows raise the door's square
refusal.
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


def _block_folds(n):
    # Contiguous thirds: each block holds every class of `_data` (period 6), so every train fold does too.
    block = np.arange(n) * 3 // n
    rows = np.arange(n)
    return [(rows[block != f], rows[block == f]) for f in range(3)]


@pytest.mark.parametrize("name", ["lr", "svc", "svk", "sgd", "ridgec", "lrcv"])
@pytest.mark.parametrize("weights", ["balanced", "dint", "dstr"])
def test_calibrated_member_equals_door(name, weights):
    k = 3
    X, codes = _data(k)
    labels = ["a", "b", "c"] if weights == "dstr" else [0, 1, 2]
    y = np.asarray(labels, dtype=object if weights == "dstr" else np.int64)[codes]
    cw = "balanced" if weights == "balanced" else {labels[0]: 0.4, labels[1]: 2.5}
    folds = _block_folds(len(y))
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


_FAMILIES = ["lr", "svc", "svk", "sgd", "ridgec", "lrcv"]


@pytest.mark.parametrize("name", _FAMILIES)
def test_dict_extra_key_is_ignored_when_every_class_is_weighted(name):
    # scikit-learn 1.9.1 compute_class_weight (utils/class_weight.py:101-120) refuses a key absent from y only when
    # some class of y is left unweighted; a dict weighting every class may name extra labels, which change nothing.
    X, y = _data(3)
    full = {0: 0.4, 1: 2.5, 2: 1.5}
    got = _fitted_words(_member(name, {**full, 9: 3.0}).fit(X, y))
    want = _fitted_words(_member(name, full).fit(X, y))
    assert len(got) == len(want) > 0
    for a, b in zip(got, want):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("name", _FAMILIES)
def test_dict_extra_key_is_refused_when_a_class_is_unweighted(name):
    X, y = _data(3)
    with pytest.raises(ValueError) as refused:
        _member(name, {0: 0.4, 1: 2.5, 9: 3.0}).fit(X, y)
    assert "np." not in str(refused.value)
    if name in ("sgd", "ridgec"):  # x_linear: scikit-learn's own words, labels as plain Python scalars
        assert str(refused.value) == "The classes, [2], are not in class_weight"


@pytest.mark.parametrize("name", _FAMILIES)
def test_calibrated_fold_lacking_a_named_class_is_accepted_when_every_class_is_weighted(name):
    # `_folds` puts every class-0 row in fold 0's test rows: fold 0 trains on classes {1, 2}, both weighted, so the
    # key 0 is an extra key there and the fold fits as the door does on the same rows (scikit-learn's rule above).
    X, y = _data(3)
    cw = {0: 0.4, 1: 2.5, 2: 1.5, 9: 3.0}
    folds = _folds(len(y))
    m = CalibratedClassifierCV(_member(name, cw), cv=folds, ensemble=True).fit(X, y)
    for f, (train, _) in enumerate(folds):
        door = _member(name, cw).fit(X[train], y[train])
        got = _fitted_words(m.calibrated_classifiers_[f][0])
        want = _fitted_words(door)
        assert len(got) == len(want) > 0
        for a, b in zip(got, want):
            np.testing.assert_array_equal(a, b)


def test_extra_key_rule_is_scikit_learns():
    utils = pytest.importorskip("sklearn.utils")
    classes = np.array([0, 1, 2])
    y = np.array([0, 1, 2, 2])
    np.testing.assert_array_equal(utils.compute_class_weight({0: 0.4, 1: 2.5, 2: 1.5, 9: 3.0}, classes=classes, y=y),
                                  [0.4, 2.5, 1.5])
    with pytest.raises(ValueError):
        utils.compute_class_weight({0: 0.4, 1: 2.5, 9: 3.0}, classes=classes, y=y)


def test_precomputed_svc_member_equals_door():
    X, y = _data(2, n=90)
    K = (X @ X.T).astype(np.float32)
    # A Voting member is fitted on every row, so its kernel stays square. AdaBoostClassifier fits only tree members
    # and refuses an SVC base by name (bindings/trees_public_boost_door.mojo), so it cannot carry this member.
    vote = mojolearn.VotingClassifier([("s", mojolearn.SVC(kernel="precomputed")),
                                       ("t", mojolearn.SVC(kernel="precomputed", C=0.5))], voting="hard").fit(K, y)
    door = mojolearn.SVC(kernel="precomputed").fit(K, y)
    got = _fitted_words(vote.estimators_[0])
    want = _fitted_words(door)
    assert len(got) == len(want) > 0
    for a, b in zip(got, want):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(vote.estimators_[0].predict(K), door.predict(K))


def test_precomputed_fold_rows_are_refused_in_the_doors_words():
    X, y = _data(2, n=90)
    K = (X @ X.T).astype(np.float32)
    train = _folds(len(y))[0][0]
    with pytest.raises(Exception) as door:
        mojolearn.SVC(kernel="precomputed").fit(K[train], y[train])
    with pytest.raises(Exception) as member:
        CalibratedClassifierCV(mojolearn.SVC(kernel="precomputed"), cv=_folds(len(y))).fit(K, y)
    assert str(member.value) == str(door.value)
