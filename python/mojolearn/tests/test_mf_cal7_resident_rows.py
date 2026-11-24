# SPDX-License-Identifier: Apache-2.0
"""CalibratedClassifierCV over the DeviceRows bases (GaussianNB, LDA, QDA, the other naive Bayes) through the typed
fit RUN (lane mf-cal7, 2026-11-23; xtrees/trees_calibrated_fit_run.mojo THE RESIDENT ROWS). Before, n_jobs=1 took
the door's DeviceRows fold units and n_jobs>1 the member set; both now take the one RUN, which on an image that
gathers on the device reads every fold by row index out of one resident X. The answers must not depend on n_jobs
(the same rows in the same order), and scikit-learn's own CalibratedClassifierCV is the oracle of the structure
(one pair per fold, its calibrator count) and, loosely, of the probabilities (two libraries' members differ in
their last bits). On a GPU install the same test exercises the indexed route; on the host the gathered rows.
"""
import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn.calibration import CalibratedClassifierCV as SkCalibrated  # noqa: E402
from sklearn import discriminant_analysis as sk_da  # noqa: E402
from sklearn import naive_bayes as sk_nb  # noqa: E402

import mojolearn  # noqa: E402
from mojolearn.calibration import CalibratedClassifierCV  # noqa: E402


def _data(k, n=150, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5)).astype(np.float32)
    y = (np.arange(n) % k).astype(np.int64)
    X[:, 0] += 1.5 * y
    X[:, 1] -= 0.5 * y
    return X, y


BASES = [
    ("GaussianNB", lambda: mojolearn.GaussianNB(), lambda: sk_nb.GaussianNB()),
    ("LinearDiscriminantAnalysis", lambda: mojolearn.LinearDiscriminantAnalysis(),
     lambda: sk_da.LinearDiscriminantAnalysis()),
    ("QuadraticDiscriminantAnalysis", lambda: mojolearn.QuadraticDiscriminantAnalysis(),
     lambda: sk_da.QuadraticDiscriminantAnalysis()),
]


@pytest.mark.parametrize("name, ours, theirs", BASES, ids=[b[0] for b in BASES])
@pytest.mark.parametrize("k", [2, 3])
@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
@pytest.mark.parametrize("ensemble", [True, False])
def test_resident_rows_route(name, ours, theirs, k, method, ensemble):
    X, y = _data(k)
    one = CalibratedClassifierCV(ours(), method=method, cv=3, ensemble=ensemble, n_jobs=1).fit(X, y)
    many = CalibratedClassifierCV(ours(), method=method, cv=3, ensemble=ensemble, n_jobs=4).fit(X, y)
    sk = SkCalibrated(theirs(), method=method, cv=3, ensemble=ensemble).fit(X, y)
    # the same RUN on every n_jobs: the same bits
    np.testing.assert_array_equal(one.predict_proba(X), many.predict_proba(X))
    np.testing.assert_array_equal(one.predict(X), many.predict(X))
    # scikit-learn's structure: one pair per fold (ensemble) or one pair, and one calibrator per score column
    assert len(one.calibrated_classifiers_) == len(sk.calibrated_classifiers_)
    for (_, cals), theirs_pair in zip(one.calibrated_classifiers_, sk.calibrated_classifiers_):
        assert len(cals) == len(theirs_pair.calibrators)
    proba = one.predict_proba(X)
    assert proba.shape == (len(y), k)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)
    # loosely scikit-learn's probabilities (members in float32 against float64)
    np.testing.assert_allclose(proba, sk.predict_proba(X), atol=0.05)


def test_resident_rows_nonfinite_x_words():
    """A non-finite X is refused once, before any fit, in the admission's words (the door units' resident route
    tested X once the same way, `_cal_admit_x_once`, deleted)."""
    X, y = _data(2)
    X[3, 2] = np.nan
    with pytest.raises(ValueError) as one:
        CalibratedClassifierCV(mojolearn.GaussianNB(), cv=3, n_jobs=1).fit(X, y)
    with pytest.raises(ValueError) as many:
        CalibratedClassifierCV(mojolearn.GaussianNB(), cv=3, n_jobs=4).fit(X, y)
    assert str(one.value) == str(many.value)
    assert "GaussianNB" in str(one.value)
