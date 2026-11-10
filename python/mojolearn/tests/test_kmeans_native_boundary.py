# SPDX-License-Identifier: Apache-2.0
"""Installed KMeans provider, Python storage and serialization boundary seams."""
import copy
import gc
import weakref

import numpy as np
import pytest
import mojolearn as ml


def _data():
    return np.array([[-1, 0], [0, 0], [1, 0], [9, 0], [10, 0], [11, 0]], dtype=np.float32)


def _model():
    return ml.KMeans(
        n_clusters=2, init="array",
        init_centroids=np.array([[0, 0], [10, 0]], dtype=np.float32),
        n_init=1, max_iter=10, random_state=7, numeric_mode="identical",
    )


def _binding():
    from mojolearn._native_family_bridge import shared_context

    connected = shared_context()["family_binding_connections"]["_mojolearn"]["identical"]
    if len(connected) == 1:
        raise RuntimeError(connected[0])
    return connected[0]


@pytest.mark.parametrize("method", ["fit", "fit_predict", "fit_transform"])
def test_public_kmeans_uses_native_provider_without_legacy_call_envelopes(method, monkeypatch):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("legacy Python KMeans dispatch executed")

    binding = _binding()
    for name in (
        "kmeans_fit", "kmeans_fit_transform", "kmeans_predict", "kmeans_transform",
        "kmeans_centers_prepare", "kmeans_centers_release",
        "kmeans_predict_resident", "kmeans_transform_resident",
        "kmeans_fit_fold", "kmeans_predict_fold", "kmeans_transform_fold",
    ):
        if hasattr(binding, name):
            monkeypatch.setattr(binding, name, forbidden)
    model = _model()
    result = getattr(model, method)(_data())
    if method == "fit":
        assert result is model
    else:
        assert np.asarray(result).shape == ((6,) if method == "fit_predict" else (6, 2))
    np.testing.assert_array_equal(np.asarray(model.predict(_data())), [0, 0, 0, 1, 1, 1])
    np.testing.assert_allclose(np.asarray(model.transform(_data()))[0], [1, 121], rtol=0, atol=1e-5)
    assert calls == []
    # Native cleanup also uses the retained descriptor, not a Python callable.
    reference = weakref.ref(model)
    if result is model:
        result = None
    del model
    gc.collect()
    assert reference() is None
    assert calls == []


def test_native_owner_cache_and_writable_rebound_centers():
    model = _model().fit(_data())
    query = np.array([[1, 0]], dtype=np.float32)
    np.testing.assert_array_equal(np.asarray(model.predict(query)), [0])
    owner = vars(model).get("_resident_holder")
    if owner is not None:
        assert not isinstance(owner, (dict, tuple))
        model.transform(query)
        assert vars(model)["_resident_holder"] is owner
    replacement = np.array([[100, 0], [10, 0]], dtype=np.float32)
    model.cluster_centers_ = replacement
    np.testing.assert_array_equal(np.asarray(model.predict(query)), [1])
    replacement[0, 0] = 0
    np.testing.assert_array_equal(np.asarray(model.predict(query)), [0])


@pytest.mark.parametrize("copier", [copy.copy, copy.deepcopy])
def test_native_resident_owner_is_not_serialized(copier):
    model = _model().fit(_data())
    model.predict(_data())
    assert "_resident_holder" not in model.__getstate__()
    restored = copier(model)
    assert vars(restored).get("_resident_holder") is None
    np.testing.assert_array_equal(np.asarray(restored.predict(_data())), [0, 0, 0, 1, 1, 1])
    if vars(model).get("_resident_holder") is not None:
        assert vars(restored)["_resident_holder"] is not vars(model)["_resident_holder"]


def test_serialized_state_protocol_is_refused_before_python_callback():
    called = []

    class Mapping:
        def keys(self):
            called.append(True)
            return []

    model = _model()
    with pytest.raises(TypeError, match="native dictionary"):
        model.__setstate__(Mapping())
    assert called == []
