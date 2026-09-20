# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`SpectralEmbedding` and `manifold.spectral_embedding`: the refusals, the
shape, and the two relations that tie the embedding to the path
`SpectralClustering` already runs (same graph, same Laplacian, same Lanczos).
"""
import numpy as np
import pytest

import mojolearn
from mojolearn import SpectralClustering, SpectralEmbedding, manifold


def _blobs():
    rng = np.random.default_rng(20260920)
    centers = np.asarray([[0.0, 0.0], [8.0, 0.0], [0.0, 8.0]])
    rows = [centers[i % 3] + rng.normal(0.0, 0.7, 2) for i in range(150)]
    return np.ascontiguousarray(np.asarray(rows, dtype=np.float32))


def _affinity(P):
    d2 = ((P[:, None, :].astype(np.float64) - P[None, :, :]) ** 2).sum(-1)
    return np.where(d2 < np.median(d2), 1.0 / (1.0 + d2), 0.0).astype(np.float32)


def _bytes(a):
    return np.ascontiguousarray(np.asarray(a)).tobytes()


@pytest.mark.parametrize("kwargs, exc", [
    (dict(affinity="rbf"), ValueError),
    (dict(gamma=1.0), NotImplementedError),
    (dict(eigen_solver="arpack"), NotImplementedError),
    (dict(eigen_tol="auto"), NotImplementedError),
    (dict(n_jobs=2), NotImplementedError),
    (dict(verbose=True), NotImplementedError),
    (dict(n_components=0), ValueError),
    (dict(n_neighbors=0), ValueError),
    (dict(random_state=-1), ValueError),
])
def test_unsupported_options_are_refused_by_name(kwargs, exc):
    with pytest.raises(exc, match="mojolearn SpectralEmbedding"):
        SpectralEmbedding(**kwargs)


def test_public_names():
    assert mojolearn.SpectralEmbedding is manifold.SpectralEmbedding
    assert "SpectralEmbedding" in mojolearn.__all__ and "manifold" in mojolearn.__all__
    assert callable(manifold.spectral_embedding)


def test_fit_transform_shape_and_repeatability():
    X = _blobs()
    m = SpectralEmbedding(n_components=2, random_state=5)
    out = m.fit_transform(X)
    assert out is m.embedding_
    assert np.asarray(out).shape == (150, 2) and np.asarray(out).dtype == np.float32
    assert np.isfinite(np.asarray(out)).all()
    assert m.n_neighbors_ == 15
    again = SpectralEmbedding(n_components=2, random_state=5).fit_transform(X)
    assert _bytes(again) == _bytes(out)


def test_too_many_components_is_refused():
    with pytest.raises(ValueError, match="n_samples"):
        SpectralEmbedding(n_components=150).fit(_blobs())


def test_embedding_is_the_clustering_embedding_without_its_trivial_column():
    X = _blobs()
    sc = SpectralClustering(n_clusters=3, n_components=4, n_neighbors=10, random_state=3).fit(X)
    se = SpectralEmbedding(n_components=3, n_neighbors=10, random_state=3).fit(X)
    assert _bytes(se.embedding_) == _bytes(np.asarray(sc.embedding_)[:, 1:])
    kept = manifold.spectral_embedding(X, n_components=4, n_neighbors=10, random_state=3,
                                       drop_first=False)
    assert _bytes(kept) == _bytes(sc.embedding_)


def test_precomputed_drops_the_diagonal():
    A = _affinity(_blobs())
    assert A.diagonal().min() > 0
    hollow = A.copy()
    np.fill_diagonal(hollow, 0.0)
    a = SpectralEmbedding(n_components=2, affinity="precomputed", random_state=3).fit_transform(A)
    b = SpectralEmbedding(n_components=2, affinity="precomputed", random_state=3).fit_transform(hollow)
    assert np.asarray(a).shape == (150, 2)
    assert _bytes(a) == _bytes(b)


def test_unnormalized_laplacian_is_a_different_embedding():
    X = _blobs()
    norm = manifold.spectral_embedding(X, n_components=2, n_neighbors=10, random_state=3)
    raw = manifold.spectral_embedding(X, n_components=2, n_neighbors=10, random_state=3,
                                      norm_laplacian=False)
    assert np.isfinite(np.asarray(raw)).all()
    assert _bytes(raw) != _bytes(norm)
