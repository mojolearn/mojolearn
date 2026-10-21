# SPDX-License-Identifier: Apache-2.0
"""Temporary activation of the real public classes, never a substitute model."""
import inspect
import io
import dataclasses

import numpy as np
import pytest

from mojolearn import LinearRegression, Ridge
from mojolearn._classical_host import HostLinearRegression, HostRidge
from mojolearn._cpu_reference import reference_training
from mojolearn._native_classical_public import _install, _restore
from mojolearn._verify_native_runtime import audit_native_call

CLASSES = (Ridge, LinearRegression, HostRidge, HostLinearRegression)
_EXTERNAL = {"__name__": "native_classical_public_user"}
exec("def predict(model, x):\n    return model.predict(x)\n", _EXTERNAL)


def clean(operation, *args, error=None, **kwargs):
    audit = audit_native_call(operation, *args, **kwargs)
    report = audit.report()
    assert not any(report[k] for k in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads")), report
    if error is not None:
        assert isinstance(audit.exception, error), report
        return audit.exception
    return audit.assert_native()


@pytest.fixture
def activate():
    active = []
    def install():
        handle = _install(*CLASSES)
        active.append(handle)
        return handle
    yield install
    for handle in reversed(active):
        _restore(handle["original"])


def data():
    rng = np.random.default_rng(2059)
    x = rng.normal(size=(19, 3)).astype("f4")
    y = (x @ np.array([.75, -.5, 1.25], "f4") + .375).astype("f4")
    return x, y


def forbidden(*args, **kwargs):
    raise AssertionError("Python preparation/binding/output helper reached")


@pytest.mark.parametrize("cls", CLASSES)
def test_actual_public_constructor_first_metadata_and_signature(cls, activate, monkeypatch):
    expected = cls(numeric_mode="identical")
    expected_params = expected.get_params()
    expected_names = cls._get_param_names()
    try:
        expected_tags = dataclasses.asdict(expected.__sklearn_tags__())
    except ImportError:
        expected_tags = None
    methods = ("__init__", "fit", "predict", "score", "save", "load", "get_params", "set_params", "_get_param_names")
    signatures = {name: inspect.signature(getattr(cls, name)) for name in methods}
    docs = {name: getattr(cls, name).__doc__ for name in methods}
    activate()
    assert {name: inspect.signature(getattr(cls, name)) for name in methods} == signatures
    assert {name: getattr(cls, name).__doc__ for name in methods} == docs
    from mojolearn import _backend
    from mojolearn import _native_classical_bridge as bridge
    monkeypatch.setattr(bridge, "connect", forbidden)
    monkeypatch.setattr(bridge, "connect_host", forbidden)
    monkeypatch.setattr(_backend, "binding", forbidden)
    model = clean(cls, numeric_mode="identical")
    assert clean(model.get_params) == expected_params
    assert clean(model.get_params, deep=False) == expected_params
    assert clean(cls._get_param_names) == expected_names
    assert type(clean(cls._get_param_names)) is list
    assert clean(model.__sklearn_is_fitted__) is False
    if expected_tags is None:
        clean(model.__sklearn_tags__, error=ImportError)
    else:
        assert dataclasses.asdict(clean(model.__sklearn_tags__)) == expected_tags
    assert clean(getattr, model, "__class__") is cls
    assert clean(getattr, model, "fit").__self__ is model
    assert clean(getattr, model, "numeric_mode") == "identical"
    assert clean(model.numeric_mode_used) == "identical"
    if cls in (HostRidge, HostLinearRegression):
        assert clean(model.vendor_used) == "cpu"
    clean(getattr, model, "missing_native_attribute", error=AttributeError)
    clean(model.predict, np.ones((1, 3), "f4"), error=ValueError)


@pytest.mark.parametrize("cls", (Ridge, LinearRegression))
@pytest.mark.parametrize("layout", ("float32", "strided_float64"))
def test_actual_public_fit_predict_score_and_checkpoint_are_native(cls, layout, activate, monkeypatch, tmp_path):
    x, y = data()
    if layout == "strided_float64":
        wide_x, wide_y = np.empty((len(x), 6), "f8"), np.empty(len(y) * 2, "f8")
        wide_x[:, ::2], wide_y[::2] = x, y
        x, y = wide_x[::-1, ::2], wide_y[-2::-2]
        # Keep genuine negative/non-unit strides through both API boundaries.
        assert not x.flags.c_contiguous and not y.flags.c_contiguous
    with reference_training():
        old = cls(numeric_mode="identical").fit(x, y)
    expected_prediction = _EXTERNAL["predict"](old, x)
    expected_coef = np.asarray(old.coef_).copy()
    expected_intercept = np.float64(old.intercept_)
    expected_score = old.score(x, y)
    old_archive = io.BytesIO()
    old.save(old_archive)
    activate()
    from mojolearn import _backend, _buffer, _mode, _serialize
    from mojolearn import _native_classical_bridge as bridge
    for owner, name in ((_backend, "binding"), (_buffer, "as_f32_c"),
                        (_mode, "_public_result"), (_mode, "_ndarray"),
                        (_serialize, "write_npz"), (_serialize, "read_npz"),
                        (bridge, "connect"), (bridge, "connect_host")):
        monkeypatch.setattr(owner, name, forbidden)
    model = clean(cls, numeric_mode="identical")
    with reference_training():
        assert clean(model.fit, x, y) is model
    assert clean(model.__sklearn_is_fitted__) is True
    coef = clean(getattr, model, "coef_")
    assert type(coef) is np.ndarray
    np.testing.assert_array_equal(coef.view("u4"), expected_coef.view("u4"))
    assert np.float64(clean(getattr, model, "intercept_")).view("u8") == expected_intercept.view("u8")
    prediction = clean(model.predict, x)
    assert type(prediction) is np.ndarray and prediction.dtype == np.dtype("f8")
    np.testing.assert_array_equal(prediction.view("u8"), expected_prediction.view("u8"))
    assert np.float64(clean(model.score, x, y)).view("u8") == np.float64(expected_score).view("u8")
    checkpoint = tmp_path / "public-native.npz"
    clean(model.save, checkpoint)
    assert checkpoint.read_bytes() == old_archive.getvalue()
    restored = clean(cls.load, checkpoint)
    np.testing.assert_array_equal(clean(restored.predict, x).view("u8"), expected_prediction.view("u8"))
    assert clean(restored.set_params, fit_intercept=False) is restored
    assert not clean(restored.__sklearn_is_fitted__)
    clean(getattr, restored, "coef_", error=AttributeError)


@pytest.mark.parametrize("ordinary,host", ((Ridge, HostRidge), (LinearRegression, HostLinearRegression)))
def test_actual_host_load_predict_digest_and_roundtrip_keep_cpu_route(ordinary, host, activate, monkeypatch):
    original = ordinary(numeric_mode="identical")
    original.coef_ = np.array([.75, -.5, .125], "f4")
    original.intercept_ = .25
    original.n_features_in_ = 3
    archive = io.BytesIO()
    original.save(archive)
    old = host.load(io.BytesIO(archive.getvalue()))
    x, y = data()
    expected = _EXTERNAL["predict"](old, x)
    score = old.score(x, y)
    digest = old.model_sha256()
    host_archive = io.BytesIO()
    old.save(host_archive)
    activate()
    from mojolearn import _backend
    from mojolearn._classical_host import _HostBound
    monkeypatch.setattr(_backend, "binding", forbidden)
    monkeypatch.setattr(_HostBound, "_bind", forbidden)
    model = clean(host.load, io.BytesIO(archive.getvalue()))
    assert type(model) is host
    assert clean(model.vendor_used) == "cpu"
    assert clean(model.numeric_mode_used) == "identical"
    np.testing.assert_array_equal(clean(model.predict, x).view("u8"), expected.view("u8"))
    assert np.float64(clean(model.score, x, y)).view("u8") == np.float64(score).view("u8")
    assert clean(model.model_sha256) == digest
    saved = io.BytesIO()
    clean(model.save, saved)
    assert saved.getvalue() == host_archive.getvalue()


@pytest.mark.parametrize("cls", CLASSES)
def test_native_fitted_attribute_array_view_keeps_alias_and_storage(cls, activate, monkeypatch):
    from mojolearn._array import Array
    value = Array((3,))
    storage = np.frombuffer(value._mv, dtype="f4")
    storage[:] = [.25, -.5, .75]
    activate()
    model = clean(cls)
    object.__getattribute__(model, "__dict__")["coef_"] = value
    monkeypatch.setattr(Array, "__array__", forbidden)
    actual = clean(getattr, model, "coef_")
    assert type(actual) is np.ndarray
    assert np.shares_memory(actual, storage)
    actual[1] = np.float32(.375)
    assert storage[1] == np.float32(.375)
    assert clean(getattr, model, "__dict__")["coef_"] is value


@pytest.mark.parametrize("where", ("property", "descriptor", "dictionary", "getattr", "key"))
def test_native_attribute_boundary_refuses_python_hooks_before_execution(where, activate, monkeypatch):
    seen = []
    class Descriptor:
        def __get__(self, instance, owner):
            seen.append("get")
            raise AssertionError("descriptor callback")
        def __set__(self, instance, value):
            seen.append("set")
            raise AssertionError("descriptor callback")
    def hook(*args):
        seen.append("property")
        raise AssertionError("property callback")
    class Key:
        def __hash__(self):
            return hash("coef_")
        def __eq__(self, other):
            seen.append("key")
            raise AssertionError("dictionary key callback")
    activate()
    model = clean(Ridge)
    if where == "property":
        monkeypatch.setattr(Ridge, "coef_", property(hook), raising=False)
    elif where == "descriptor":
        monkeypatch.setattr(Ridge, "coef_", Descriptor(), raising=False)
    elif where == "dictionary":
        # CPython forbids replacing an existing type's __dict__ descriptor;
        # the supported way to declare such a Python hook is a subclass.
        class DictionaryModel(Ridge):
            __dict__ = property(hook)
        model = object.__new__(DictionaryModel)
    elif where == "getattr":
        monkeypatch.setattr(Ridge, "__getattr__", hook, raising=False)
    else:
        object.__getattribute__(model, "__dict__")[Key()] = 1
    clean(getattr, model, "coef_", error=TypeError)
    assert not seen


def test_native_constructor_refuses_python_scalar_without_running_it(activate):
    seen = []
    class Scalar:
        def __float__(self):
            seen.append("float")
            raise AssertionError("scalar callback")
    activate()
    clean(Ridge, alpha=Scalar(), error=TypeError)
    assert not seen
