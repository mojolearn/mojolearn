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


@pytest.mark.parametrize("method", ["predict", "transform"])
def test_nonfinite_query_is_refused_before_public_output(method):
    model = _model().fit(_data())
    before = np.asarray(model.cluster_centers_).copy()
    query = _data()
    query[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinity"):
        getattr(model, method)(query)
    np.testing.assert_array_equal(np.asarray(model.cluster_centers_), before)


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


def _hooked_scalar(kind, calls):
    def forbidden(self, *args, **kwargs):
        calls.append(kind)
        raise AssertionError("Python scalar hook executed")

    hooks = dict.fromkeys(
        ("__int__", "__index__", "__float__", "__gt__", "__eq__", "__hash__", "__repr__"),
        forbidden,
    )
    hooks["__class__"] = property(forbidden)
    if kind == "metaclass":
        meta = type("HookedMeta", (type,), {"__name__": property(forbidden)})
        return meta("HookedObject", (), hooks)()
    if kind == "object_array":
        value = np.empty((), dtype=object)
        value[()] = type("HookedObject", (), hooks)()
        return value
    base, value = {
        "object": (object, None), "int": (int, 2), "float": (float, 0.01),
        "str": (str, "array"), "np_str": (np.str_, "array"),
    }[kind]
    cls = type("HookedScalar", (base,), hooks)
    return cls() if base is object else cls(value)


@pytest.mark.parametrize("field,kind", [
    ("n_clusters", "object"), ("n_clusters", "int"),
    ("metric", "object"), ("metric", "str"), ("metric", "np_str"), ("metric", "int"),
    ("init", "str"), ("init", "np_str"), ("init", "int"), ("init", "object"),
    ("oversampling_factor", "object"), ("oversampling_factor", "int"),
    ("oversampling_factor", "float"), ("oversampling_factor", "metaclass"),
    ("tol", "object"), ("tol", "float"), ("tol", "object_array"),
    ("random_state", "object"), ("random_state", "int"), ("random_state", "object_array"),
    ("max_iter", "object"), ("n_init", "object"),
])
def test_rebound_scalar_is_refused_before_python_hooks(field, kind):
    calls = []
    model = _model()
    vars(model)[field] = _hooked_scalar(kind, calls)
    with pytest.raises((TypeError, ValueError)):
        model.fit(_data())
    assert calls == []


@pytest.mark.parametrize("method", ["predict", "transform"])
def test_query_metric_is_refused_before_python_hashing(method):
    model = _model().fit(_data())
    calls = []
    vars(model)["metric"] = _hooked_scalar("str", calls)
    with pytest.raises(TypeError, match="without Python conversion hooks"):
        getattr(model, method)(_data())
    assert calls == []


def test_callable_init_keeps_explicit_refusal_without_calling_it():
    calls = []

    def init(*args, **kwargs):
        calls.append(True)
        raise AssertionError("Python init executed")

    model = _model()
    model.init = init
    with pytest.raises(TypeError, match="Python init callbacks are not accepted"):
        model.fit(_data())
    assert calls == []


@pytest.mark.parametrize("stage,late_field,message", [
    ("init", "metric", "init must be one of"),
    ("metric", "oversampling_factor", "metric must be one of"),
    ("factor", "tol", "oversampling_factor must be a float"),
    ("X", "n_clusters", "2-D"),
    ("clusters", "tol", "exceeds n_samples"),
    ("centers", "tol", "init_centroids must be"),
    ("weights", "tol", "sample_weight"),
    ("tol", "random_state", "could not convert string to float"),
    ("tol", "max_iter", "could not convert string to float"),
])
def test_scalar_admission_preserves_earlier_refusal(stage, late_field, message):
    calls = []
    model = _model()
    source, weights = _data(), None
    if stage in ("init", "metric"):
        vars(model)[stage] = "unknown"
    elif stage == "factor":
        model.oversampling_factor = "invalid"
    elif stage == "X":
        source = source[:, 0]
    elif stage == "clusters":
        model.n_clusters = 7
    elif stage == "centers":
        model.init_centroids = np.zeros((1, 2), dtype=np.float32)
    elif stage == "weights":
        weights = np.ones(5, dtype=np.float32)
    elif stage == "tol":
        model.tol = "invalid"
    vars(model)[late_field] = _hooked_scalar("object", calls)
    with pytest.raises((TypeError, ValueError), match=message):
        model.fit(source, sample_weight=weights)
    assert calls == []


@pytest.mark.parametrize("variant", ["numeric", "text", "bytes", "extended_real", "zero_dimensional", "init_code", "centers"])
def test_exact_native_scalar_forms_and_explicit_centers_remain_supported(variant):
    model = _model()
    if variant == "numeric":
        model.n_clusters, model.max_iter, model.n_init = np.int64(2), np.int32(10), np.int64(1)
        model.tol, model.random_state, model.oversampling_factor = np.float32(0.0001), np.uint32(7), np.float64(2)
    elif variant == "text":
        model.init, model.metric = np.str_("array"), np.str_("euclidean")
        model.tol, model.random_state = np.str_("0.0001"), np.str_("7")
    elif variant == "bytes":
        model.tol, model.random_state = np.bytes_("0.0001"), np.bytes_("7")
    elif variant == "extended_real":
        model.tol, model.random_state = np.longdouble(0.0001), np.longlong(7)
    elif variant == "zero_dimensional":
        model.tol, model.random_state = np.array(0.0001), np.array(7)
    elif variant == "init_code":
        model.init = 2
    elif variant == "centers":
        model.init, model.init_centroids = model.init_centroids, None
    model.fit(_data())
    np.testing.assert_array_equal(np.asarray(model.predict(_data())), [0, 0, 0, 1, 1, 1])


@pytest.mark.parametrize("field,value,message", [
    ("init", np.int64(2), "2-D"),
    ("oversampling_factor", np.float32(2), "oversampling_factor must be a float"),
    ("oversampling_factor", True, "oversampling_factor must be a float"),
])
def test_scalar_admission_does_not_widen_existing_parameter_routes(field, value, message):
    model = _model()
    vars(model)[field] = value
    with pytest.raises((TypeError, ValueError), match=message):
        model.fit(_data())
