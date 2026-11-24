# SPDX-License-Identifier: Apache-2.0
"""RFE rows lane mf-member8 (2026-11-23): the last estimators RFE refused for lack of a native member.
  * RFE with no elimination round (n_features_to_select >= n_features: scikit-learn fits the estimator once on
    every column and reads no importances) over NearestNeighbors, RadiusNeighbors, OPTICS, MDS and
    SpectralEmbedding runs the member's fit (the same entries as the estimator's own fit): estimator_'s fitted
    attributes equal the estimator's own fit's, and equal scikit-learn's RFE over the same estimator;
  * with a round, scikit-learn fails (none has coef_ or feature_importances_): refused in RFE's words;
  * Cholesky: `fit(A)` takes no y, so scikit-learn's RFE fails with Python's TypeError; refused in those words;
  * AffinityPropagation's cluster_centers_ as importance_getter: ranking_ equals the elimination written out
    over the estimator's own cluster_centers_; affinity='precomputed' publishes none (attrgetter's words);
  * IncrementalPCA mean_ / var_ as importance_getter.

Runs with the built package (the python/mojolearn/tests suite on a box, never on the laptop). Written, not run.
"""
import numpy as np
import pytest

import mojolearn


def _data(n=60, d=5, seed=8):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)).astype(np.float32)
    y = (x[:, 0] - 0.5 * x[:, 3]).astype(np.float32)
    return x, y


def _same(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    assert a.tobytes() == b.tobytes()


_NO_ROUND = [
    ("NearestNeighbors", lambda: mojolearn.NearestNeighbors(n_neighbors=3), ["_index", "n_samples_fit_",
                                                                              "n_features_in_"]),
    ("RadiusNeighbors", lambda: mojolearn.RadiusNeighbors(radius=1.5), ["_index", "n_samples_fit_",
                                                                         "n_features_in_"]),
    ("OPTICS", lambda: mojolearn.OPTICS(min_samples=4), ["labels_", "reachability_", "ordering_",
                                                          "core_distances_", "predecessor_",
                                                          "cluster_hierarchy_", "n_features_in_"]),
    ("MDS", lambda: mojolearn.MDS(n_components=2, random_state=0, n_init=1, max_iter=50),
     ["embedding_", "dissimilarity_matrix_", "stress_", "n_iter_", "n_features_in_"]),
    ("SpectralEmbedding", lambda: mojolearn.SpectralEmbedding(n_components=2, random_state=0),
     ["embedding_", "n_neighbors_", "n_features_in_"]),
    ("SpectralEmbedding-rbf", lambda: mojolearn.SpectralEmbedding(n_components=2, affinity="rbf", random_state=0),
     ["embedding_", "affinity_matrix_", "gamma_", "n_features_in_"]),
]


@pytest.mark.parametrize("name,make,attributes", _NO_ROUND, ids=[row[0] for row in _NO_ROUND])
def test_no_round_fit_is_the_estimators_own(name, make, attributes):
    x, y = _data()
    rfe = mojolearn.RFE(make(), n_features_to_select=x.shape[1]).fit(x, y)
    # support_ is a list of bool (RFE's documented attribute, python/mojolearn/_expansion_prep.py RFE docstring).
    assert np.asarray(rfe.support_, dtype=bool).all()
    assert (np.asarray(rfe.ranking_) == 1).all()
    own = make().fit(x)
    for attribute in attributes:
        mine = getattr(rfe.estimator_, attribute)
        theirs = getattr(own, attribute)
        if np.isscalar(theirs):
            assert mine == theirs, attribute
        else:
            _same(mine, theirs)


@pytest.mark.parametrize("name,make,attributes", _NO_ROUND, ids=[row[0] for row in _NO_ROUND])
def test_no_round_fit_equals_scikit_learn_rfe(name, make, attributes):
    sklearn_rfe = pytest.importorskip("sklearn.feature_selection").RFE
    x, y = _data()
    mine = mojolearn.RFE(make(), n_features_to_select=x.shape[1]).fit(x, y)
    theirs = sklearn_rfe(make(), n_features_to_select=x.shape[1]).fit(x, y)
    _same(np.asarray(mine.support_, dtype=bool), np.asarray(theirs.support_, dtype=bool))
    _same(np.asarray(mine.ranking_, dtype=np.int64), np.asarray(theirs.ranking_, dtype=np.int64))
    for attribute in attributes:
        a = getattr(mine.estimator_, attribute)
        b = getattr(theirs.estimator_, attribute)
        if np.isscalar(b):
            assert a == b, attribute
        else:
            _same(a, b)


@pytest.mark.parametrize("name,make,attributes", _NO_ROUND, ids=[row[0] for row in _NO_ROUND])
def test_a_round_is_refused_in_rfe_words(name, make, attributes):
    x, y = _data()
    with pytest.raises(Exception, match="RFE needs an estimator with coef_ or feature_importances_"):
        mojolearn.RFE(make(), n_features_to_select=2).fit(x, y)


def test_cholesky_refused_in_python_words():
    rng = np.random.default_rng(2)
    m = rng.random((6, 6)).astype(np.float32)
    a = (m @ m.T + 6 * np.eye(6, dtype=np.float32)).astype(np.float32)
    with pytest.raises(TypeError, match=r"Cholesky.fit\(\) takes 2 positional arguments but 3 were given"):
        mojolearn.RFE(mojolearn.Cholesky(jitter=0.0), n_features_to_select=6).fit(a, np.zeros(6, np.float32))


def _written_out(make, attribute, x, y, n_select):
    d = x.shape[1]
    support = np.ones(d, dtype=bool)
    ranking = np.ones(d, dtype=np.int64)
    while support.sum() > n_select:
        cols = np.flatnonzero(support)
        est = make().fit(x[:, cols], y)
        values = np.asarray(getattr(est, attribute), dtype=np.float64)
        keys = values ** 2 if values.ndim == 1 else (values ** 2).sum(axis=0)
        gone = cols[np.argsort(keys, kind="stable")[:1]]
        support[gone] = False
        ranking[~support] += 1
    return support, ranking


def test_affinity_propagation_cluster_centers_getter():
    x, y = _data(n=40, d=6)
    make = lambda: mojolearn.AffinityPropagation(random_state=0)
    rfe = mojolearn.RFE(make(), n_features_to_select=3, importance_getter="cluster_centers_").fit(x, y)
    support, ranking = _written_out(make, "cluster_centers_", x, y, 3)
    _same(np.asarray(rfe.support_, dtype=bool), support)
    _same(np.asarray(rfe.ranking_, dtype=np.int64), ranking)


def test_affinity_propagation_precomputed_has_no_centers():
    x, y = _data(n=12, d=12)
    s = -np.abs(x - x.T).astype(np.float32)
    with pytest.raises(AttributeError, match="has no attribute 'cluster_centers_'"):
        mojolearn.RFE(mojolearn.AffinityPropagation(affinity="precomputed", random_state=0),
                      n_features_to_select=6, importance_getter="cluster_centers_").fit(s, y)


@pytest.mark.parametrize("attribute", ["mean_", "var_"])
def test_incremental_pca_named_getters(attribute):
    x, y = _data(n=50, d=6)
    make = lambda: mojolearn.IncrementalPCA(n_components=2)
    rfe = mojolearn.RFE(make(), n_features_to_select=3, importance_getter=attribute).fit(x, y)
    support, ranking = _written_out(make, attribute, x, y, 3)
    _same(np.asarray(rfe.support_, dtype=bool), support)
    _same(np.asarray(rfe.ranking_, dtype=np.int64), ranking)


# ---------------------------------------------------------------- ROUND b (lane mf-member8)
@pytest.mark.parametrize("name,make,attributes", _NO_ROUND, ids=[row[0] for row in _NO_ROUND])
@pytest.mark.parametrize("target", ["two_d", "labels"])
def test_no_round_accepts_what_check_X_y_accepts(name, make, attributes, target):
    x, y = _data()
    yy = np.stack([y, -y], axis=1) if target == "two_d" else np.where(y > 0, "pos", "neg")
    rfe = mojolearn.RFE(make(), n_features_to_select=x.shape[1]).fit(x, yy)
    own = make().fit(x)
    _same(getattr(rfe.estimator_, attributes[0]), getattr(own, attributes[0]))


def test_no_round_y_refusals_in_scikit_learn_words():
    x, y = _data()
    with pytest.raises(ValueError, match="RFE requires y to be passed, but the target y is None"):
        mojolearn.RFE(mojolearn.NearestNeighbors(), n_features_to_select=x.shape[1]).fit(x, None)
    bad = y.copy()
    bad[3] = np.nan
    with pytest.raises(ValueError, match="Input y contains NaN"):
        mojolearn.RFE(mojolearn.OPTICS(min_samples=4), n_features_to_select=x.shape[1]).fit(x, bad)
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        mojolearn.RFE(mojolearn.MDS(random_state=0), n_features_to_select=x.shape[1]).fit(x, y[:-1])


def test_meta_fits_a_target_free_member_and_predict_fails_as_scikit_learn():
    x, y = _data()
    labels = (y > 0).astype(np.int64)
    vote = mojolearn.VotingClassifier([("nn", mojolearn.NearestNeighbors(n_neighbors=3))]).fit(x, labels)
    _same(vote.estimators_[0]._index, mojolearn.NearestNeighbors(n_neighbors=3).fit(x)._index)
    with pytest.raises(AttributeError, match="has no attribute 'predict'"):
        vote.predict(x)


def test_lda_covariance_getter_published_or_refused():
    x, y = _data(n=80, d=6)
    labels = (y > 0).astype(np.int64)
    make = lambda: mojolearn.LinearDiscriminantAnalysis(solver="eigen")
    rfe = mojolearn.RFE(make(), n_features_to_select=3, importance_getter="covariance_").fit(x, labels)
    support, ranking = _written_out(make, "covariance_", x, labels, 3)
    _same(np.asarray(rfe.ranking_, dtype=np.int64), ranking)
    with pytest.raises(AttributeError, match="has no attribute 'covariance_'"):
        mojolearn.RFE(mojolearn.LinearDiscriminantAnalysis(solver="svd"), n_features_to_select=3,
                      importance_getter="covariance_").fit(x, labels)


def test_mincovdet_precision_getter_refused_without_store_precision():
    x, y = _data(n=80, d=6)
    with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'ndim'"):
        mojolearn.RFE(mojolearn.MinCovDet(store_precision=False, random_state=0), n_features_to_select=3,
                      importance_getter="precision_").fit(x, y)


@pytest.mark.parametrize("make,attribute,classify", [
    (lambda: mojolearn.BayesianGaussianMixture(n_components=2, random_state=0), "means_", False),
    (lambda: mojolearn.SimpleImputer(), "statistics_", False),
    (lambda: mojolearn.NearestCentroid(), "centroids_", True),
])
def test_sweep_named_getters(make, attribute, classify):
    x, y = _data(n=80, d=6)
    target = (y > 0).astype(np.int64) if classify else y
    rfe = mojolearn.RFE(make(), n_features_to_select=3, importance_getter=attribute).fit(x, target)
    support, ranking = _written_out(make, attribute, x, target, 3)
    _same(np.asarray(rfe.ranking_, dtype=np.int64), ranking)
