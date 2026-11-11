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


@pytest.mark.parametrize("parameter", ["n_components", "n_neighbors", "random_state", "affinity"])
def test_standalone_embedding_admits_scalars_before_conversion(parameter):
    from mojolearn import manifold

    calls = []

    class ConversionHook:
        @property
        def __class__(self):
            calls.append("class")
            return int

        def __int__(self):
            calls.append("int")
            return 2

        def __float__(self):
            calls.append("float")
            return 2.0

    options = {parameter: ConversionHook()}
    with pytest.raises(TypeError):
        manifold.spectral_embedding(np.eye(4, dtype=np.float32), **options)
    assert calls == []


def test_standalone_embedding_refuses_input_shape_hook():
    from mojolearn import manifold

    calls = []

    class ShapeHook:
        @property
        def shape(self):
            calls.append("shape")
            return (4, 4)

        def __array__(self, *args, **kwargs):
            calls.append("array")
            return np.eye(4, dtype=np.float32)

    with pytest.raises(TypeError):
        manifold.spectral_embedding(ShapeHook(), n_components=2, affinity="precomputed")
    assert calls == []


@pytest.mark.parametrize("flag", ["norm_laplacian", "drop_first"])
def test_standalone_flag_refusal_does_not_read_custom_metaclass_name(flag):
    from mojolearn import manifold

    calls = []

    class NamingMeta(type):
        @property
        def __name__(cls):
            calls.append("name")
            return "spoofed"

    class BadBool(metaclass=NamingMeta):
        def __bool__(self):
            calls.append("bool")
            return True

    with pytest.raises(TypeError, match="BadBool"):
        manifold.spectral_embedding(np.eye(4, dtype=np.float32), **{flag: BadBool()})
    assert calls == []


@pytest.mark.parametrize("clustering,parameter", [
    (False, "random_state"), (False, "eigen_solver"), (False, "eigen_tol"),
    (True, "random_state"), (True, "eigen_solver"),
])
@pytest.mark.parametrize("printing", ["formatter", "override_repr"])
def test_spectral_scalar_refusal_does_not_invoke_numpy_printing(clustering, parameter, printing):
    calls = []

    def forbidden(value):
        calls.append("printing")
        raise AssertionError("NumPy Python printing callback executed")

    if printing == "override_repr" and printing not in np.get_printoptions():
        pytest.skip("This NumPy version has no override_repr option; formatter coverage remains active")
    options = {parameter: np.array(-1, dtype=np.int64)}
    model = (SpectralClustering(n_clusters=2, n_components=2, n_neighbors=2, **options)
             if clustering else SpectralEmbedding(n_components=2, n_neighbors=2, **options))
    option = {"all": forbidden} if printing == "formatter" else forbidden
    error = NotImplementedError if parameter == "eigen_solver" else ValueError
    with np.printoptions(**{printing: option}):
        with pytest.raises(error, match="<native scalar array>"):
            model.fit(np.eye(4, dtype=np.float32))
    assert calls == []
