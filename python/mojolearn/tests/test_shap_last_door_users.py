# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""KernelExplainer and PermutationExplainer over the last classes of the door's list (lane mf-shap3b, 2026-11-23).

The public boundary of the native routes: IsolationForest (its member's integer predict), LabelPropagation and
LabelSpreading (their typed predict_proba query), RFE (its fitted estimator's member over the support columns) and
StackingClassifier(stack_method='predict') (its classifier members' codes mapped to their labels) are explained
natively, so their SHAP values are additive against the public output; a forecaster is refused by name. The
numerics of each handle kind are proven in checks/native_tests/trees_shap.mojo (`test_agnostic_int_values`,
`test_agnostic_select_member`, `test_agnostic_label_propagation`, `test_agnostic_meta_stacking_labels`).
"""
import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn.datasets import make_classification, make_regression  # noqa: E402

import mojolearn as ml  # noqa: E402


def _clf(n_classes=2, seed=0):
    X, y = make_classification(n_samples=300, n_features=4, n_informative=3, n_redundant=0, n_classes=n_classes,
                               random_state=seed)
    return X.astype(np.float32), y


def _reg(seed=0):
    X, y = make_regression(n_samples=300, n_features=4, n_informative=3, noise=1.0, random_state=seed)
    return X.astype(np.float32), y.astype(np.float32)


def _additive(model, X, output):
    explainer = ml.KernelExplainer(model, X[:20])
    rows = X[200:206]
    phi = np.asarray(explainer.shap_values(rows), dtype=np.float64)
    want = np.asarray(output(rows), dtype=np.float64)
    ev = np.asarray(explainer.expected_value, dtype=np.float64)
    if want.ndim == 1:
        got = phi.sum(axis=1) + ev
    else:
        got = phi.sum(axis=1) + ev[None, :]
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)
    perm = ml.PermutationExplainer(model, X[:20], random_state=0)
    pphi = np.asarray(perm.shap_values(rows, npermutations=10), dtype=np.float64)
    assert pphi.shape == phi.shape


def test_isolation_forest_predict():
    X, _ = _clf()
    model = ml.IsolationForest(n_estimators=10, random_state=0).fit(X)
    _additive(model, X, lambda r: np.asarray(model.predict(r), dtype=np.float32))


@pytest.mark.parametrize("cls,kw", [("LabelPropagation", dict(kernel="rbf", gamma=2.0)),
                                    ("LabelSpreading", dict(kernel="knn", n_neighbors=5))])
def test_label_propagation_predict_proba(cls, kw):
    X, y = _clf(3)
    y = y.copy()
    y[60:] = -1
    model = getattr(ml, cls)(**kw).fit(X, y)
    _additive(model, X, lambda r: np.asarray(model.predict_proba(r), dtype=np.float32))


def test_label_propagation_subclass_refused():
    class Mine(ml.LabelPropagation):
        pass

    X, y = _clf()
    y = y.copy()
    y[60:] = -1
    model = Mine().fit(X, y)
    with pytest.raises(TypeError, match="requires a mojolearn estimator"):
        ml.KernelExplainer(model, X[:20])


def test_rfe_classifier_and_regressor():
    X, y = _clf(3)
    clf = ml.RFE(ml.LogisticRegression(), n_features_to_select=2).fit(X, y)
    _additive(clf, X, lambda r: np.asarray(clf.predict_proba(r), dtype=np.float32))
    Xr, yr = _reg()
    reg = ml.RFE(ml.Ridge(), n_features_to_select=2).fit(Xr, yr)
    _additive(reg, Xr, lambda r: np.asarray(reg.predict(r), dtype=np.float32))


def test_rfe_unfitted_estimator_words():
    X, _ = _clf()
    with pytest.raises(AttributeError, match="estimator_"):
        ml.KernelExplainer(ml.RFE(ml.LogisticRegression()), X[:20])


def test_stacking_predict_members():
    X, y = _clf(3)
    model = ml.StackingClassifier([("lr", ml.LogisticRegression()), ("nb", ml.GaussianNB())],
                                  stack_method="predict", cv=3).fit(X, y.astype(np.float64) * 2.5 - 1.0)
    _additive(model, X, lambda r: np.asarray(model.predict_proba(r), dtype=np.float32))


@pytest.mark.parametrize("cls", ["CrostonClassic", "ProphetForecaster", "Theta"])
def test_forecaster_refused(cls):
    X, _ = _reg()
    with pytest.raises(TypeError, match="is a forecaster"):
        ml.KernelExplainer(getattr(ml, cls)(), X[:20])
    with pytest.raises(TypeError, match="is a forecaster"):
        ml.PermutationExplainer(getattr(ml, cls)(), X[:20])
