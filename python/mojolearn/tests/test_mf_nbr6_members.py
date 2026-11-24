# SPDX-License-Identifier: Apache-2.0
"""Settings lane mf-nbr6 (2026-11-23) moved from the classes' own doors into their member routes, against the
estimators' own doors.

Each estimator is fitted twice: by its own fit (the door) and as a GridSearchCV member (one given split, train =
every row, a one-point grid, refit=True: the refit runs the member). The refit estimator's fitted attributes and
its outputs equal the door's. A seed given as a RandomState is a fresh RandomState(0) on both sides (the search
clones it, so its member draws the first word, as the door's fit does). Refusals keep the door's words.
"""
import numpy as np
import pytest

import mojolearn
from mojolearn.model_selection import GridSearchCV


def _data(n=40, d=3, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1]).astype(np.float32)
    return X, y


def _member(est, X, y=None, scoring=None):
    """The refit estimator of a one-split, one-point search (fitted as a member)."""
    every = np.arange(len(X))
    params = est.get_params()
    name = sorted(params)[0]  # never random_state: each fit takes its own clone of a RandomState
    # scoring=None needs the class's own `score` (scikit-learn 1.9 metrics/_scorer.py `check_scoring`, a TypeError
    # before any fit; providers/python/member_ensemble_boundary_door.mojo `_no_score_refusal`). A class without one
    # is searched by a scorer whose response it lacks: every score is error_score (nan) and the best is refitted
    # through the member, as scikit-learn does (lane w3f2-cv).
    if scoring is None and getattr(est, "score", None) is None:
        scoring = "r2"
    search = GridSearchCV(est, {name: [params[name]]}, cv=[(every, every)], scoring=scoring, refit=True)
    search.fit(X, y)
    return search.best_estimator_


def _same(a, b, names):
    for name in names:
        np.testing.assert_array_equal(np.asarray(getattr(a, name)), np.asarray(getattr(b, name)), err_msg=name)


def test_ocsvm_precomputed_member_equals_door():
    X, _ = _data()
    K = (X @ X.T).astype(np.float32)
    ones = np.ones(len(X), dtype=np.int64)
    door = mojolearn.OneClassSVM(kernel="precomputed", nu=0.3).fit(K)
    member = _member(mojolearn.OneClassSVM(kernel="precomputed", nu=0.3), K, ones, scoring="accuracy")
    _same(member, door, ["support_", "dual_coef_", "offset_", "n_features_in_"])
    assert np.asarray(member.support_vectors_).shape == (len(np.asarray(door.support_)), 0)
    np.testing.assert_array_equal(np.asarray(member.decision_function(K)), np.asarray(door.decision_function(K)))


def test_kpca_precomputed_member_equals_door():
    X, _ = _data()
    K = (X @ X.T).astype(np.float32)
    door = mojolearn.KernelPCA(n_components=2, kernel="precomputed").fit(K)
    member = _member(mojolearn.KernelPCA(n_components=2, kernel="precomputed"), K)
    _same(member, door, ["eigenvalues_", "eigenvectors_"])
    np.testing.assert_array_equal(np.asarray(member.transform(K[:5])), np.asarray(door.transform(K[:5])))


@pytest.mark.parametrize("cls", ["PolynomialCountSketch", "SkewedChi2Sampler"])
def test_randomstate_seed_member_equals_door(cls):
    X, _ = _data()
    X = np.abs(X)
    est = getattr(mojolearn, cls)
    door = est(n_components=8, random_state=np.random.RandomState(0)).fit(X)
    member = _member(est(n_components=8, random_state=np.random.RandomState(0)), X)
    np.testing.assert_array_equal(np.asarray(member.transform(X)), np.asarray(door.transform(X)))


def test_svgp_inducing_points_member_equals_door():
    X, y = _data()
    Z = X[::5].copy()
    door = mojolearn.SVGP(inducing_points=Z).fit(X, y)
    member = _member(mojolearn.SVGP(inducing_points=Z), X, y, scoring="r2")
    _same(member, door, ["Z_", "_alpha", "_C", "elbo_"])
    np.testing.assert_array_equal(np.asarray(member.predict(X)), np.asarray(door.predict(X)))


def test_fastica_w_init_member_equals_door():
    X, _ = _data(d=3)
    w = np.eye(2, dtype=np.float32) + 0.1
    door = mojolearn.FastICA(n_components=2, w_init=w).fit(X)
    member = _member(mojolearn.FastICA(n_components=2, w_init=w), X)
    _same(member, door, ["components_", "mixing_", "n_iter_"])


def test_dictionary_learning_inits_member_equal_door():
    X, _ = _data(d=4)
    code = np.ones((len(X), 3), dtype=np.float32)
    atoms = np.eye(3, 4, dtype=np.float32)
    door = mojolearn.DictionaryLearning(n_components=3, code_init=code, dict_init=atoms, max_iter=5,
                                        random_state=0).fit(X)
    member = _member(mojolearn.DictionaryLearning(n_components=3, code_init=code, dict_init=atoms, max_iter=5,
                                                  random_state=0), X)
    _same(member, door, ["components_", "n_iter_"])
    door = mojolearn.MiniBatchDictionaryLearning(n_components=3, dict_init=atoms, max_iter=2, random_state=0).fit(X)
    member = _member(mojolearn.MiniBatchDictionaryLearning(n_components=3, dict_init=atoms, max_iter=2,
                                                           random_state=0), X)
    _same(member, door, ["components_"])


def test_sparse_pca_inits_member_equal_door():
    X, _ = _data(d=4)
    V = np.eye(2, 4, dtype=np.float32)
    U = np.ones((len(X), 2), dtype=np.float32)
    door = mojolearn.SparsePCA(n_components=2, V_init=V, U_init=U, max_iter=5, random_state=0).fit(X)
    member = _member(mojolearn.SparsePCA(n_components=2, V_init=V, U_init=U, max_iter=5, random_state=0), X)
    _same(member, door, ["components_", "mean_"])


def test_isomap_radius_member_equals_door():
    X, _ = _data()
    door = mojolearn.Isomap(n_neighbors=None, radius=3.0, n_components=2).fit(X)
    member = _member(mojolearn.Isomap(n_neighbors=None, radius=3.0, n_components=2), X)
    _same(member, door, ["embedding_", "dist_matrix_"])
    assert member._knn is None
    np.testing.assert_array_equal(np.asarray(member.transform(X[:4])), np.asarray(door.transform(X[:4])))


@pytest.mark.parametrize("init", ["random", "pca", "array"])
def test_umap_init_member_equals_door(init):
    X, _ = _data(n=60)
    start = (np.arange(120, dtype=np.float32).reshape(60, 2) / 60.0) if init == "array" else init
    door = mojolearn.UMAP(n_neighbors=8, n_epochs=20, init=start, random_state=3).fit(X)
    member = _member(mojolearn.UMAP(n_neighbors=8, n_epochs=20, init=start, random_state=3), X)
    _same(member, door, ["embedding_"])


def test_umap_seed_beyond_float64_member_equals_door():
    X, _ = _data(n=60)
    seed = 2 ** 60 + 1
    door = mojolearn.UMAP(n_neighbors=8, n_epochs=20, random_state=seed).fit(X)
    member = _member(mojolearn.UMAP(n_neighbors=8, n_epochs=20, random_state=seed), X)
    _same(member, door, ["embedding_"])
    assert member._transform_config == door._transform_config
