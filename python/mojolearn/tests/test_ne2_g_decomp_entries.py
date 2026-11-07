"""Fused and once-per-state steps of the decomposition and mixture doors
(plan NATIVE_ENTRY_POINTS_AFTER_0866 section 2G, lane ne2-g-decomp).

Each case checks that the shortened path returns the bits of the long one
and that a change of the state it stands on is still seen:

- GaussianMixture.fit_predict returns the labels of fit(X).predict(X)
  (the GPU fit writes them from its own device buffers, N401);
- the dictionary-learning and SparsePCA transforms keep the Gram of the
  fitted components between queries (N339, N333): repeated queries agree,
  and a refit or a changed ridge_alpha is honoured;
- the PCA query record proven by the model's generation word (N315, N316)
  still sees a replaced fitted array and a refit;
- the vendor and mode check of TruncatedSVD, KernelPCA and IncrementalPCA
  runs once per connected table (N313, N317, N340): repeated queries agree;
- the fused cast and finiteness scan of the decomposition inputs (N352,
  N353) casts as before and still refuses NaN and infinity;
- MinCovDet / EllipticEnvelope resolve their kind once per call (N345).
"""
import numpy as np
import pytest

import mojolearn as ml
from mojolearn import _expansion_decomp as decomp


def _data(rows=64, cols=6, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((rows, cols)).astype(np.float32)


def test_gmm_fit_predict_equals_fit_then_predict():
    x = np.concatenate([_data(40, 3, 1) - 3.0, _data(40, 3, 2) + 3.0]).astype(np.float32)
    fused = ml.GaussianMixture(n_components=2, random_state=0).fit_predict(x)
    model = ml.GaussianMixture(n_components=2, random_state=0).fit(x)
    np.testing.assert_array_equal(np.asarray(fused), np.asarray(model.predict(x)))


def test_dictionary_transform_gram_kept_and_refit_seen():
    x = _data(48, 5, 3)
    model = decomp.DictionaryLearning(n_components=4, transform_algorithm="omp",
                                      transform_n_nonzero_coefs=2, random_state=0, max_iter=5).fit(x)
    first = np.asarray(model.transform(x))
    second = np.asarray(model.transform(x))
    np.testing.assert_array_equal(first, second)
    other = _data(48, 5, 4)
    model.fit(other)
    fresh = decomp.DictionaryLearning(n_components=4, transform_algorithm="omp",
                                      transform_n_nonzero_coefs=2, random_state=0, max_iter=5).fit(other)
    np.testing.assert_array_equal(np.asarray(model.transform(x)), np.asarray(fresh.transform(x)))


def test_sparse_pca_transform_gram_kept_and_ridge_alpha_seen():
    x = _data(40, 6, 5)
    model = decomp.SparsePCA(n_components=3, random_state=0, max_iter=5).fit(x)
    first = np.asarray(model.transform(x))
    np.testing.assert_array_equal(first, np.asarray(model.transform(x)))
    model.set_params(ridge_alpha=0.5)
    fresh = decomp.SparsePCA(n_components=3, random_state=0, max_iter=5, ridge_alpha=0.5).fit(x)
    np.testing.assert_array_equal(np.asarray(model.transform(x)), np.asarray(fresh.transform(x)))


def test_pca_query_record_sees_replaced_state_and_refit():
    x = _data(50, 6, 6)
    model = ml.PCA(n_components=3).fit(x)
    first = np.asarray(model.transform(x))
    np.testing.assert_array_equal(first, np.asarray(model.transform(x)))
    # A replaced fitted array is a new state (the generation word moves).
    model.mean_ = np.asarray(model.mean_) + np.float32(1.0)
    shifted = np.asarray(model.transform(x))
    assert not np.array_equal(first, shifted)
    model.fit(_data(50, 6, 7))
    fresh = ml.PCA(n_components=3).fit(_data(50, 6, 7))
    np.testing.assert_array_equal(np.asarray(model.transform(x)), np.asarray(fresh.transform(x)))


def test_pca_inverse_after_transform_record():
    x = _data(30, 4, 8)
    model = ml.PCA(n_components=2).fit(x)
    z = np.asarray(model.transform(x))
    np.testing.assert_array_equal(np.asarray(model.inverse_transform(z)),
                                  np.asarray(model.inverse_transform(z)))


@pytest.mark.parametrize("make", [
    lambda: ml.TruncatedSVD(n_components=2),
    lambda: ml.KernelPCA(n_components=2),
    lambda: decomp.IncrementalPCA(n_components=2),
])
def test_repeated_queries_after_once_per_table_check(make):
    x = _data(32, 5, 9)
    model = make().fit(x)
    first = np.asarray(model.transform(x))
    for _ in range(3):
        np.testing.assert_array_equal(first, np.asarray(model.transform(x)))


def test_decomp_input_cast_and_scan_in_one_pass():
    x = _data(20, 4, 10)
    model = ml.TruncatedSVD(n_components=2).fit(x)
    as_f64 = np.asarray(model.transform(x.astype(np.float64)))
    np.testing.assert_array_equal(as_f64, np.asarray(model.transform(x)))
    ints = np.arange(80, dtype=np.int64).reshape(20, 4)
    np.testing.assert_array_equal(np.asarray(model.transform(ints)),
                                  np.asarray(model.transform(ints.astype(np.float32))))
    bad = x.astype(np.float64)
    bad[3, 1] = np.nan
    with pytest.raises(ValueError):
        model.transform(bad)
    bad[3, 1] = np.inf
    with pytest.raises(ValueError):
        model.transform(bad)


def test_pca_input_cast_and_scan_in_one_pass():
    x = _data(20, 4, 11)
    model = ml.PCA(n_components=2).fit(x)
    np.testing.assert_array_equal(np.asarray(model.transform(x.astype(np.float64))),
                                  np.asarray(model.transform(x)))
    bad = x.astype(np.float64)
    bad[0, 0] = np.inf
    with pytest.raises(ValueError, match="NaN or infinity"):
        model.transform(bad)


def test_elliptic_envelope_score_and_predict():
    x = _data(60, 3, 12)
    model = decomp.EllipticEnvelope(random_state=0).fit(x)
    labels = np.asarray(model.predict(x))
    assert set(np.unique(labels)) <= {-1, 1}
    assert model.score(x, labels) == 1.0
    mcd = decomp.MinCovDet(random_state=0).fit(x)
    np.testing.assert_array_equal(np.asarray(mcd.mahalanobis(x)), np.asarray(mcd.mahalanobis(x)))
