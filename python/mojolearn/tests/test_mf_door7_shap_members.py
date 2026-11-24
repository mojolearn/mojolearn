# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""KernelExplainer and PermutationExplainer over the last models the door evaluator served (lane mf-door7, 2026-11-23).

The door evaluator (`_DoorModel`: the model's compiled door per chunk, serially with the GIL held) is deleted. The
models that still reached it were meta-estimators with a member whose META admission answered False for settings
only a FIT reads: a CalibratedClassifierCV over a DeviceRows base (GaussianNB, LDA, QDA) as a member, a fitted search
as a member (its scorers, cv, splitter and candidates), a Stacking or Calibrated member whose cv is a splitter
object. A fitted member is now admitted for predict only (providers/python/member_ensemble_boundary_door.mojo, a
SERVING admission), so each is explained natively; its SHAP values are additive against the model's own public
output (Kernel SHAP's efficiency constraint; Permutation SHAP's telescoping sum), which is what the door's values
were additive against. A model nothing native serves is refused in words (`_unserved`); no public model reaches
that refusal today.
"""
import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn.datasets import make_classification, make_regression  # noqa: E402

import mojolearn as ml  # noqa: E402
from mojolearn.model_selection import GridSearchCV, ShuffleSplit, KFold  # noqa: E402


def _clf(n_classes=2, seed=0):
    X, y = make_classification(n_samples=300, n_features=4, n_informative=3, n_redundant=0, n_classes=n_classes,
                               random_state=seed)
    return X.astype(np.float32), y


def _reg(seed=0):
    X, y = make_regression(n_samples=300, n_features=4, n_informative=3, noise=1.0, random_state=seed)
    return X.astype(np.float32), y.astype(np.float32)


def _search_lr(cv=3):
    return GridSearchCV(ml.LogisticRegression(), {"C": [0.1, 1.0]}, cv=cv)


_CLASSIFIERS = {
    # a Calibrated member over a DeviceRows base (its admission answered False: the door)
    "vote_calibrated_nb": lambda: ml.VotingClassifier(
        [("lr", ml.LogisticRegression()), ("cal", ml.CalibratedClassifierCV(ml.GaussianNB(), cv=3))], voting="soft"),
    "vote_calibrated_lda": lambda: ml.VotingClassifier(
        [("cal", ml.CalibratedClassifierCV(ml.LinearDiscriminantAnalysis(), cv=3)), ("nb", ml.GaussianNB())],
        voting="soft"),
    # a fitted search as a member (now its serving admission: its best estimator)
    "vote_search": lambda: ml.VotingClassifier([("search", _search_lr()), ("nb", ml.GaussianNB())], voting="soft"),
    "vote_search_shuffle": lambda: ml.VotingClassifier(
        [("search", _search_lr(ShuffleSplit(n_splits=3, test_size=0.25, random_state=0))), ("nb", ml.GaussianNB())],
        voting="soft"),
    "stack_search": lambda: ml.StackingClassifier([("search", _search_lr()), ("nb", ml.GaussianNB())], cv=3),
    # a Stacking member whose cv is one of our splitter objects (read only by its fit)
    "vote_stack_kfold": lambda: ml.VotingClassifier(
        [("stack", ml.StackingClassifier([("lr", ml.LogisticRegression()), ("nb", ml.GaussianNB())],
                                         cv=KFold(n_splits=3))), ("nb", ml.GaussianNB())], voting="soft"),
}


@pytest.mark.parametrize("name", sorted(_CLASSIFIERS))
@pytest.mark.parametrize("n_classes", [2, 3])
def test_member_served_classifier_is_additive(name, n_classes):
    X, y = _clf(n_classes)
    m = _CLASSIFIERS[name]().fit(X, y)
    bg, rows = X[:12], X[200:205]
    want = np.asarray(m.predict_proba(rows), dtype=np.float64)
    ke = ml.KernelExplainer(m, bg)
    phi = np.asarray(ke.shap_values(rows))
    assert phi.shape == (5, 4, n_classes)
    np.testing.assert_allclose(phi.sum(1) + np.asarray(ke.expected_value), want, atol=1e-5)
    pe = ml.PermutationExplainer(m, bg, random_state=0)
    pp = np.asarray(pe.shap_values(rows, npermutations=10))
    np.testing.assert_allclose(pp.sum(1) + np.asarray(pe.expected_value), want, atol=1e-5)


def test_member_served_regressor_search_is_additive():
    X, y = _reg()
    m = ml.VotingRegressor([("search", GridSearchCV(ml.Ridge(), {"alpha": [0.1, 1.0]}, cv=3)),
                            ("lin", ml.LinearRegression())]).fit(X, y)
    bg, rows = X[:12], X[200:205]
    ke = ml.KernelExplainer(m, bg)
    phi = np.asarray(ke.shap_values(rows))
    want = np.asarray(m.predict(rows), dtype=np.float64).reshape(-1)
    np.testing.assert_allclose(phi.sum(1) + ke.expected_value, want, rtol=1e-6, atol=1e-4)


def test_member_served_expected_value_is_the_background_mean():
    """The construction's background call runs on the same native handle (no door at construction either)."""
    X, y = _clf(3)
    m = _CLASSIFIERS["vote_calibrated_nb"]().fit(X, y)
    bg = X[:12]
    ke = ml.KernelExplainer(m, bg)
    want = np.asarray(m.predict_proba(bg), dtype=np.float64).mean(0)
    np.testing.assert_allclose(np.asarray(ke.expected_value), want, atol=1e-6)
