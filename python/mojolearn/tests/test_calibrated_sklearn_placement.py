# SPDX-License-Identifier: Apache-2.0
"""CalibratedClassifierCV over a cv whose fold 0 lacks a class, against scikit-learn's own (lane mf-cal3b,
2026-11-23; xtrees/trees_calibrated_fit_run.mojo `cal_fold_place`). scikit-learn is the independent oracle of the
STRUCTURE (calibrators per pair, the refusal words, the warning): the bits of two libraries' members differ.
Also the Host* members (lane mf-cal3b): HostSVC and HostKNeighborsClassifier fit as native members on the host
image, where they were refused before any fold ("requires a mojolearn estimator").
"""
import warnings

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn.calibration import CalibratedClassifierCV as SkCalibrated  # noqa: E402
from sklearn.linear_model import LogisticRegression as SkLogistic  # noqa: E402
from sklearn.naive_bayes import GaussianNB as SkGaussianNB  # noqa: E402

import mojolearn  # noqa: E402
from mojolearn.calibration import CalibratedClassifierCV  # noqa: E402


def _data(k, n=120, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    y = (np.arange(n) % k).astype(np.int64)
    X[:, 0] += y
    return X, y


def _gap_cv(y, k, held):
    """Three contiguous folds; fold 0's train rows hold exactly the classes in `held`.

    The folds are row blocks, not i % 3: `_data`'s class is i % k, so with k = 3 the rows of i % 3 != 0
    never hold class 0, and held (0, 1) or (0, 2) left ONE class in fold 0 (scikit-learn's own
    LogisticRegression then refuses "only one class"), never the two-class fold these cases name."""
    rows = np.arange(len(y))
    fold = rows * 3 // len(y)
    train0 = rows[(fold != 0) & np.isin(y, held)]
    return [(train0, rows[fold == 0])] + [(rows[fold != f], rows[fold == f]) for f in (1, 2)]


def _words(fit):
    try:
        fit()
    except Exception as error:  # noqa: BLE001 - the words are compared
        return type(error).__name__, str(error)
    return None


@pytest.mark.parametrize("held, counts", [((0, 1), [1, 3, 3]), ((0, 2), [1, 3, 3])])
def test_two_class_fold_has_one_calibrator(held, counts):
    X, y = _data(3)
    cv = _gap_cv(y, 3, held)
    ours = CalibratedClassifierCV(mojolearn.LogisticRegression(), cv=cv).fit(X, y)
    theirs = SkCalibrated(SkLogistic(), cv=cv).fit(X, y)
    assert [len(cals) for _, cals in ours.calibrated_classifiers_] == counts
    assert [len(c.calibrators) for c in theirs.calibrated_classifiers_] == counts
    proba = ours.predict_proba(X)
    assert proba.shape == (len(y), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6)


def test_three_class_fold_of_four_has_three_calibrators():
    X, y = _data(4)
    cv = _gap_cv(y, 4, (0, 1, 3))
    ours = CalibratedClassifierCV(mojolearn.LogisticRegression(), cv=cv).fit(X, y)
    theirs = SkCalibrated(SkLogistic(), cv=cv).fit(X, y)
    assert [len(cals) for _, cals in ours.calibrated_classifiers_] == [3, 4, 4]
    assert [len(c.calibrators) for c in theirs.calibrated_classifiers_] == [3, 4, 4]


def test_imbalanced_decision_fold_refused_as_sklearn():
    X, y = _data(3)
    cv = _gap_cv(y, 3, (0, 1))
    ours = _words(lambda: CalibratedClassifierCV(mojolearn.LogisticRegression(), cv=cv, ensemble=False).fit(X, y))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        theirs = _words(lambda: SkCalibrated(SkLogistic(), cv=cv, ensemble=False).fit(X, y))
    assert theirs is not None and ours == theirs


def test_lacking_fold_warns_as_sklearn():
    X, y = _data(4)
    cv = _gap_cv(y, 4, (0, 1, 3))
    with pytest.warns(RuntimeWarning, match=r"Number of classes in training fold \(3\) does not match total number "
                                            r"of classes \(4\)"):
        CalibratedClassifierCV(mojolearn.LogisticRegression(), cv=cv, ensemble=False).fit(X, y)


def test_one_class_proba_fold_refused_as_sklearn():
    X, y = _data(2)
    cv = _gap_cv(y, 2, (0,))
    ours = _words(lambda: CalibratedClassifierCV(mojolearn.GaussianNB(), cv=cv).fit(X, y))
    theirs = _words(lambda: SkCalibrated(SkGaussianNB(), cv=cv).fit(X, y))
    assert theirs is not None and ours == theirs


@pytest.mark.parametrize("make", [lambda host: host.HostSVC(C=1.0, gamma=0.5),
                                  lambda host: host.HostKNeighborsClassifier(n_neighbors=5)])
def test_host_member_fits_natively(make):
    from mojolearn import _classical_host as host
    X, y = _data(3)
    model = CalibratedClassifierCV(make(host), cv=3).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(y), 3)
    assert all(type(member) is type(make(host)) for member, _ in model.calibrated_classifiers_)
