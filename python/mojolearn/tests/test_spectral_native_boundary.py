# SPDX-License-Identifier: Apache-2.0
"""Public descriptor admission seams; numerical properties live in native suites."""
import numpy as np
import pytest

from mojolearn import SpectralClustering, SpectralEmbedding
from mojolearn import _spectral_impl


@pytest.mark.parametrize("clustering", [False, True])
def test_spectral_refuses_python_metrics_factory_without_calling_it(monkeypatch, clustering):
    context = _spectral_impl._NATIVE_CONTEXT
    mode = context["backend_fields"]["_DEFAULT_MODE"]
    binding = context["family_spectral_umap_bindings"]["metrics"][mode]
    calls = []

    def hook():
        calls.append("factory")
        raise AssertionError("Python descriptor factory must not execute")

    monkeypatch.setattr(binding, "spectral_native_api_v1", hook)
    model = (SpectralClustering(n_clusters=2, n_components=2, n_neighbors=2, random_state=7)
             if clustering else SpectralEmbedding(n_components=2, n_neighbors=2, random_state=7))
    values = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1],
                         [4, 4], [4, 5], [5, 4], [5, 5]], dtype=np.float32)
    with pytest.raises(TypeError, match="spectral_native_api_v1 must be a compiled entry"):
        model.fit(values)
    assert calls == []


@pytest.mark.parametrize("operation", ["coo", "neighbors"])
def test_spectral_refuses_python_preparation_factory_without_calling_it(monkeypatch, operation):
    context = _spectral_impl._NATIVE_CONTEXT
    binding = context["family_spectral_umap_bindings"]["base"]["identical"]
    calls = []

    def hook():
        calls.append("factory")
        raise AssertionError("Python preparation factory must not execute")

    monkeypatch.setattr(binding, "spectral_preparation_native_api_v1", hook)
    with pytest.raises(TypeError, match="spectral_preparation_native_api_v1 must be a compiled entry"):
        if operation == "coo":
            _spectral_impl._coo_triples(np.eye(4, dtype=np.float64))
        else:
            SpectralEmbedding(n_components=2, n_neighbors=2,
                              affinity="precomputed_nearest_neighbors").fit(
                                  np.asarray([[0, 1, 2, 3], [1, 0, 1, 2],
                                              [2, 1, 0, 1], [3, 2, 1, 0]], dtype=np.float32))
    assert calls == []
