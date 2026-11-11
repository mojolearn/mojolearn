# SPDX-License-Identifier: Apache-2.0
"""IncrementalPCA public admission, publication and lifetime boundary seams."""
import gc
import weakref

import numpy as np
import pytest
import mojolearn as ml


def _data():
    return np.array([[1, 2, 1], [3, 1, 2], [1, 1, 4], [2, 3, 1], [1, 2, 2], [4, 1, 1]], dtype=np.float32)


def _model(**options):
    settings = dict(n_components=2, batch_size=3, numeric_mode="identical")
    settings.update(options)
    return ml.IncrementalPCA(**settings)


def _hook(calls, kind="object"):
    class Hooks:
        def __int__(self):
            calls.append("int")
            return 2

        def __float__(self):
            calls.append("float")
            return 2.0

        def __bool__(self):
            calls.append("bool")
            return True

        @property
        def __class__(self):
            calls.append("class")
            return int

    class Integer(int):
        __int__ = Hooks.__int__
        __float__ = Hooks.__float__
        __bool__ = Hooks.__bool__

    return Hooks() if kind == "object" else Integer(2)


@pytest.mark.parametrize("whiten", [False, True])
def test_ipca_public_methods_do_not_call_legacy_python_entries(whiten, monkeypatch):
    from mojolearn import _backend, _expansion_decomp, _native_ipca_bridge

    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("legacy IncrementalPCA dispatch executed")

    binding = _backend.binding("_mojolearn_x_decomp", "identical")
    for name in ("x_decomp_ipca_fit", "x_decomp_ipca_partial", "x_decomp_ipca_project",
                 "x_decomp_dev_alloc", "x_decomp_dev_upload", "x_decomp_dev_free"):
        if hasattr(binding, name):
            monkeypatch.setattr(binding, name, forbidden)
    # Reconnect after replacing exports, so an old import-time function map
    # cannot silently keep its original compiled entries and pass this seam.
    _, context = _native_ipca_bridge.connect(ml.IncrementalPCA)
    monkeypatch.setattr(_expansion_decomp, "_IPCA_CONTEXT", context)
    model = _model(whiten=whiten)
    scores = model.fit_transform(_data())
    assert np.asarray(scores).shape == (6, 2)
    assert np.asarray(model.transform(_data())).shape == (6, 2)
    assert np.asarray(model.inverse_transform(scores)).shape == (6, 3)
    assert model.partial_fit(_data()) is model
    assert model.n_samples_seen_ == 12
    assert calls == []


@pytest.mark.parametrize("operation", ["fit", "partial_fit", "fit_transform", "transform", "inverse_transform"])
def test_ipca_replaced_descriptor_is_refused_without_python_call(operation, monkeypatch):
    from mojolearn._expansion_decomp import _IPCA_CONTEXT

    calls = []

    def forbidden():
        calls.append(True)
        raise AssertionError("Python IncrementalPCA descriptor called")

    model = _model().fit(_data())
    scores = model.transform(_data())
    monkeypatch.setitem(_IPCA_CONTEXT["ipca_bindings"], "identical", forbidden)
    with pytest.raises(TypeError, match="compiled extension entry"):
        getattr(model, operation)(scores if operation == "inverse_transform" else _data())
    assert calls == []


@pytest.mark.parametrize("field,operation", [
    ("batch_size", "fit"), ("n_components", "fit"),
    ("n_components", "partial_fit"), ("n_samples_seen_", "partial_fit"),
    ("n_features_in_", "transform"), ("whiten", "transform"), ("whiten", "inverse_transform"),
])
@pytest.mark.parametrize("kind", ["object", "int_subclass"])
def test_ipca_rebound_scalar_hooks_are_refused_before_conversion(field, operation, kind):
    calls = []
    model = _model().fit(_data())
    scores = model.transform(_data())
    vars(model)[field] = _hook(calls, kind)
    with pytest.raises(TypeError, match="native values"):
        getattr(model, operation)(scores if operation == "inverse_transform" else _data())
    assert calls == []


@pytest.mark.parametrize("field", ["batch_size", "n_components", "n_samples_seen_", "n_features_in_", "whiten"])
def test_ipca_object_array_scalar_hooks_are_refused(field):
    calls = []
    model = _model().fit(_data())
    vars(model)[field] = np.array(_hook(calls), dtype=object)
    operation = "fit" if field in ("batch_size", "n_components") else "partial_fit" if field == "n_samples_seen_" else "transform"
    with pytest.raises(TypeError, match="native values"):
        getattr(model, operation)(_data())
    assert calls == []


@pytest.mark.parametrize("dtype", ["U", "S"])
@pytest.mark.parametrize("field,value,operation", [
    ("batch_size", "3", "fit"), ("n_components", "2", "fit"),
    ("n_samples_seen_", "6", "partial_fit"), ("n_features_in_", "3", "transform"),
])
def test_ipca_native_zero_dimensional_text_scalar_conversion(dtype, field, value, operation):
    model = _model().fit(_data())
    vars(model)[field] = np.array(value, dtype=dtype)
    result = getattr(model, operation)(_data())
    if operation == "transform":
        assert np.asarray(result).shape == (6, 2)
    else:
        assert result is model
        assert model.n_components_ == 2
        assert model.n_samples_seen_ == (12 if operation == "partial_fit" else 6)


def test_ipca_fit_reset_and_batch_publication_precede_requested_count_failure():
    model = _model().fit(_data())
    old_mean = model.mean_
    model.batch_size = None
    model.n_components = 4
    with pytest.raises(ValueError, match="n_components"):
        model.fit(_data())
    assert model.batch_size_ == 15
    assert "components_" not in vars(model)
    assert "n_samples_seen_" not in vars(model)
    assert model.mean_ is old_mean


def test_ipca_input_and_query_width_refusals_precede_later_scalar_hooks():
    calls = []
    model = _model()
    vars(model)["batch_size"] = _hook(calls)
    with pytest.raises(ValueError):
        model.fit(np.ones(3, dtype=np.float32))
    model = _model().fit(_data())
    vars(model)["whiten"] = _hook(calls)
    with pytest.raises(ValueError, match="features"):
        model.transform(np.ones((2, 4), dtype=np.float32))
    assert calls == []


def test_ipca_partial_fit_ignores_fit_batch_size_and_keeps_running_count():
    calls = []
    model = _model(n_components=None)
    vars(model)["batch_size"] = _hook(calls)
    assert model.partial_fit(_data()[:3]) is model
    assert model.n_components_ == 3
    assert model.n_samples_seen_ == 3
    assert model.partial_fit(_data()[3:]) is model
    assert model.n_samples_seen_ == 6
    assert calls == []


@pytest.mark.parametrize("field", ["components_", "mean_", "explained_variance_"])
@pytest.mark.parametrize("rebind", [False, True])
def test_ipca_queries_observe_writable_and_rebound_public_arrays(field, rebind):
    model = _model(whiten=True).fit(_data())
    mutable = np.asarray(getattr(model, field)).copy()
    setattr(model, field, mutable)
    before = np.asarray(model.transform(_data())).copy()
    if rebind:
        mutable = mutable.copy()
        setattr(model, field, mutable)
    mutable *= np.float32(2) if field != "mean_" else np.float32(0)
    after = np.asarray(model.transform(_data()))
    assert not np.array_equal(before, after)
    assert getattr(model, field) is mutable
    assert mutable.flags.writeable


def test_ipca_resident_records_do_not_keep_model_alive():
    model = _model().fit(_data())
    model.transform(_data())
    reference = weakref.ref(model)
    del model
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize("value", [False, True, [], [object()], (), (object(),), {}, {"x": object()},
                                   0j, 1j, np.complex64(1j), np.array(["yes"], dtype="U"),
                                   np.array([[b"yes"]], dtype="S")])
def test_ipca_whiten_preserves_native_truth_without_element_protocols(value):
    model = _model().fit(_data())
    vars(model)["whiten"] = value
    assert np.asarray(model.transform(_data())).shape == (6, 2)
