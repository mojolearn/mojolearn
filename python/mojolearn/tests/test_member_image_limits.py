# SPDX-License-Identifier: Apache-2.0
"""Member image limits closed by lane mf-img3 (2026-11-23), against the estimators' own doors.

* The host QN image answers the class-code query: a three-class LogisticRegression / LinearSVC on the CPU image
  predicts the first-max argmax of its own decision scores (the doors' former host route), and a LinearSVC
  CalibratedClassifierCV fold member on the host column holds the words of a clone fitted by hand on the fold.
* GradientBoostingClassifier(numeric_mode='fast') is a member: a calibrated fold's member holds the model text of a
  clone fitted by hand on that fold, and its codes are the clone's (the learner's class kernels).
* GradientBoosting with numeric cat_features / one_hot_features: a calibrated fold's GradientBoostingClassifier is
  not the case (the adapter has no cat_features); the booster members are covered by tools/mf_img3_digests.py.
"""
import numpy as np
import pytest

import mojolearn
from mojolearn.calibration import CalibratedClassifierCV


def _data(k, n=150, seed=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    codes = np.arange(n) % k
    X[:, 0] += codes
    return X, codes.astype(np.int64)


def _folds(n):
    rows = np.arange(n)
    return [(rows[rows % 3 != f], rows[rows % 3 == f]) for f in range(3)]


def _host(name):
    if name == "lr":
        return mojolearn.LogisticRegression(max_iter=200, device="cpu")
    return mojolearn.LinearSVC(C=0.5, device="cpu")


@pytest.mark.parametrize("name", ["lr", "svc"])
def test_host_three_class_codes_are_the_decision_argmax(name):
    X, y = _data(3)
    m = _host(name).fit(X, y)
    scores = np.asarray(m.decision_function(X))
    np.testing.assert_array_equal(np.asarray(m.predict(X)), np.asarray(m.classes_)[np.argmax(scores, axis=1)])


def test_host_linear_svc_calibrated_member_equals_door():
    X, y = _data(3)
    folds = _folds(len(y))
    m = CalibratedClassifierCV(_host("svc"), cv=folds, ensemble=True).fit(X, y)
    for f, (train, _) in enumerate(folds):
        door = _host("svc").fit(X[train], y[train])
        member = m.calibrated_classifiers_[f][0]
        np.testing.assert_array_equal(np.asarray(member.coef_), np.asarray(door.coef_))
        np.testing.assert_array_equal(np.asarray(member.intercept_), np.asarray(door.intercept_))
        np.testing.assert_array_equal(np.asarray(member.predict(X)), np.asarray(door.predict(X)))


def test_fast_gradient_boosting_classifier_member_equals_door():
    X, y = _data(2)
    folds = _folds(len(y))
    try:
        mojolearn.GradientBoostingClassifier(n_estimators=10, numeric_mode="fast").fit(X, y)
    except RuntimeError as error:  # no fast gbdt binding in this installation
        pytest.skip(str(error))
    m = CalibratedClassifierCV(mojolearn.GradientBoostingClassifier(n_estimators=10, numeric_mode="fast"),
                               cv=folds, ensemble=True).fit(X, y)
    for f, (train, _) in enumerate(folds):
        door = mojolearn.GradientBoostingClassifier(n_estimators=10, numeric_mode="fast").fit(X[train], y[train])
        member = m.calibrated_classifiers_[f][0]
        assert member.model_ == door.model_
        np.testing.assert_array_equal(np.asarray(member.predict(X)), np.asarray(door.predict(X)))
