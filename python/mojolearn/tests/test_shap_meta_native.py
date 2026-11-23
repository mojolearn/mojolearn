# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""KernelExplainer and PermutationExplainer over meta-estimators and searches (lane mf-shap3, 2026-11-23).

The public boundary of the native model handles: every meta-estimator is explained through the RUN of its own
public operation over its members' native states, so its SHAP values are additive against that public output; a
fitted search is explained through its best estimator; MultiOutputClassifier, non-finite rows and non-numeric
labels are refused in words. The numerics of each handle kind are proven in
checks/native_tests/trees_shap.mojo (`test_agnostic_meta_*`).
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


def _reg(seed=0, outputs=1):
    X, y = make_regression(n_samples=300, n_features=4, n_informative=3, n_targets=outputs, noise=1.0,
                           random_state=seed)
    return X.astype(np.float32), y.astype(np.float32)


def _lr():
    return ml.LogisticRegression()


def _tree_clf():
    return ml.DecisionTreeClassifier(max_depth=3, random_state=0)


def _tree_reg():
    return ml.DecisionTreeRegressor(max_depth=3, random_state=0)


_CLASSIFIERS = {
    "bagging": lambda: ml.BaggingClassifier(_tree_clf(), n_estimators=4, max_features=0.75, random_state=0),
    "stacking": lambda: ml.StackingClassifier([("lr", _lr()), ("nb", ml.GaussianNB())], cv=3),
    "stacking_passthrough": lambda: ml.StackingClassifier([("lr", _lr()), ("dt", _tree_clf())], passthrough=True,
                                                          cv=3),
    "one_vs_rest": lambda: ml.OneVsRestClassifier(_lr()),
    "calibrated": lambda: ml.CalibratedClassifierCV(_lr(), method="sigmoid", cv=3),
    "adaboost": lambda: ml.AdaBoostClassifier(n_estimators=5, random_state=0),
    "vote_soft_tree": lambda: ml.VotingClassifier([("lr", _lr()), ("dt", _tree_clf())], voting="soft"),
}

_REGRESSORS = {
    "bagging": lambda: ml.BaggingRegressor(_tree_reg(), n_estimators=4, max_features=0.75, random_state=0),
    "stacking": lambda: ml.StackingRegressor([("lin", ml.LinearRegression()), ("ridge", ml.Ridge())], cv=3),
    "adaboost": lambda: ml.AdaBoostRegressor(n_estimators=5, random_state=0),
}


@pytest.mark.parametrize("name", sorted(_CLASSIFIERS))
@pytest.mark.parametrize("n_classes", [2, 3])
def test_meta_classifier_shap_is_additive(name, n_classes):
    X, y = _clf(n_classes)
    m = _CLASSIFIERS[name]().fit(X, y)
    bg, rows = X[:12], X[200:205]
    ke = ml.KernelExplainer(m, bg)
    phi = np.asarray(ke.shap_values(rows))
    assert phi.shape == (5, 4, n_classes)
    want = np.asarray(m.predict_proba(rows), dtype=np.float64)
    np.testing.assert_allclose(phi.sum(1) + np.asarray(ke.expected_value), want, atol=1e-5)
    pe = ml.PermutationExplainer(m, bg, random_state=0)
    pp = np.asarray(pe.shap_values(rows, npermutations=10))
    np.testing.assert_allclose(pp.sum(1) + np.asarray(pe.expected_value), want, atol=1e-5)


@pytest.mark.parametrize("name", sorted(_REGRESSORS))
def test_meta_regressor_shap_is_additive(name):
    X, y = _reg()
    m = _REGRESSORS[name]().fit(X, y)
    bg, rows = X[:12], X[200:205]
    ke = ml.KernelExplainer(m, bg)
    phi = np.asarray(ke.shap_values(rows))
    want = np.asarray(m.predict(rows), dtype=np.float64).reshape(-1)
    np.testing.assert_allclose(phi.sum(1) + ke.expected_value, want, rtol=1e-6, atol=1e-4)


@pytest.mark.parametrize("member", ["ridge", "tree"])
def test_multioutput_regressor_shap_per_output(member):
    X, y = _reg(outputs=2)
    est = ml.Ridge() if member == "ridge" else _tree_reg()
    m = ml.MultiOutputRegressor(est).fit(X, y)
    ke = ml.KernelExplainer(m, X[:12])
    phi = np.asarray(ke.shap_values(X[200:205]))
    assert phi.shape == (5, 4, 2)
    want = np.asarray(m.predict(X[200:205]), dtype=np.float64)
    np.testing.assert_allclose(phi.sum(1) + np.asarray(ke.expected_value), want, rtol=1e-6, atol=1e-4)


def test_hard_vote_is_explained_through_its_labels():
    X, y = _clf(2)
    labels = np.where(y == 1, 4.0, -1.5)
    m = ml.VotingClassifier([("lr", _lr()), ("nb", ml.GaussianNB()), ("dt", _tree_clf())], voting="hard").fit(X,
                                                                                                          labels)
    ke = ml.KernelExplainer(m, X[:12])
    phi = np.asarray(ke.shap_values(X[200:205]))
    want = np.asarray(m.predict(X[200:205]), dtype=np.float64)
    np.testing.assert_allclose(phi.sum(1) + ke.expected_value, want, atol=1e-5)


def test_search_is_explained_through_its_best_estimator():
    from mojolearn.model_selection import GridSearchCV
    X, y = _clf(2)
    search = GridSearchCV(_lr(), {"C": [0.1, 1.0]}, cv=3).fit(X, y)
    via_search = np.asarray(ml.KernelExplainer(search, X[:12]).shap_values(X[200:205]))
    direct = np.asarray(ml.KernelExplainer(search.best_estimator_, X[:12]).shap_values(X[200:205]))
    assert via_search.tobytes() == direct.tobytes()
    unfitted = GridSearchCV(_lr(), {"C": [0.1, 1.0]}, cv=3, refit=False).fit(X, y)
    with pytest.raises(AttributeError, match="refit=False or is not fitted"):
        ml.KernelExplainer(unfitted, X[:12])


def test_multioutput_classifier_is_refused():
    X, _ = _clf(2)
    Y = np.stack([(X[:, 0] > 0), (X[:, 1] > 0)], axis=1).astype(np.int64)
    m = ml.MultiOutputClassifier(_lr()).fit(X, Y)
    with pytest.raises(ValueError, match="1-D or 2-D"):
        ml.KernelExplainer(m, X[:12])


def test_non_finite_rows_are_refused_in_words():
    X, y = _clf(2)
    m = ml.BaggingClassifier(_tree_clf(), n_estimators=3, random_state=0).fit(X, y)
    ke = ml.KernelExplainer(m, X[:12])
    rows = X[200:204].copy()
    rows[1, 2] = np.nan
    with pytest.raises(Exception, match="NaN or infinity"):
        ke.shap_values(rows)


def test_non_numeric_labels_are_refused_in_words():
    X, y = _clf(2)
    # mojolearn's LinearSVC is cuML's deterministic quasi-Newton machine (svm.py), which has no random_state.
    m = ml.LinearSVC().fit(X, np.array(["no", "yes"])[y])
    with pytest.raises(TypeError, match="not numeric"):
        ml.KernelExplainer(m, X[:12])
