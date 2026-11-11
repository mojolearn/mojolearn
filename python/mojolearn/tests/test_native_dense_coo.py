"""Spectral sparse and dense-distance Python boundary refusals.

Original dense COO scans and nearest-neighbor affinity oracles moved to
checks/native_tests/spectral_dense_{cpu,gpu}_suite.mojo after native acceptance.
"""
import pytest
from mojolearn._spectral_impl import SpectralEmbedding


def test_refusals():
    """A sparse graph is refused by name; a negative or NaN distance in a
    dense matrix is refused."""
    sparse = pytest.importorskip("scipy.sparse")
    import numpy as np
    n = 6
    se = SpectralEmbedding(affinity="precomputed_nearest_neighbors", n_neighbors=2)
    se._validate_params()  # the fit-time parameter checks derive the private settings
    graph = sparse.coo_matrix((np.ones(3, "<f4"), (np.arange(3), np.arange(1, 4))), shape=(n, n))
    with pytest.raises(TypeError, match="SpectralEmbedding takes dense X only"):
        se.fit(graph)
    for bad in (-1.0, float("nan")):
        D = np.ones((n, n), "<f4")
        D[2, 4] = bad
        with pytest.raises(ValueError, match="negative or NaN"):
            se._precomputed_knn_affinity(D)
