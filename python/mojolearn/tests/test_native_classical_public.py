# SPDX-License-Identifier: Apache-2.0
"""LinearRegression, Ridge and their explicit host classes: every public call
is one native call, from the first constructor on, and releases what it took."""
import dataclasses
import gc
import inspect
import io
import sys
import weakref

import numpy as np
import pytest

import mojolearn
from mojolearn import LinearRegression, Ridge
from mojolearn._classical_host import HostLinearRegression, HostRidge
from mojolearn._verify_native_runtime import audit_native_call

CLASSES = (Ridge, LinearRegression, HostRidge, HostLinearRegression)


def clean(operation, *args, error=None, **kwargs):
    audit = audit_native_call(operation, *args, **kwargs)
    report = audit.report()
    assert not any(report[k] for k in ("python_frames", "forbidden_native_calls", "wrapper_work",
                                       "unobserved_existing_threads")), report
    if error is not None:
        assert isinstance(audit.exception, error), report
        return audit.exception
    return audit.assert_native()


def data():
    rng = np.random.default_rng(2059)
    x = rng.normal(size=(19, 3)).astype("f4")
    y = (x @ np.array([.75, -.5, 1.25], "f4") + .375).astype("f4")
    return x, y


def forbidden(*args, **kwargs):
    raise AssertionError("Python preparation/binding/output helper reached")


@pytest.mark.parametrize("cls", CLASSES)
def test_constructor_metadata_and_signature_are_native(cls, monkeypatch):
    from mojolearn import _backend
    monkeypatch.setattr(_backend, "binding", forbidden)
    for name in ("__init__", "fit", "predict", "score", "save", "load", "get_params", "set_params"):
        assert getattr(cls, name).__doc__ or name == "__init__"
    assert "numeric_mode" in inspect.signature(cls.__init__).parameters
    model = clean(cls, numeric_mode="identical")
    params = clean(model.get_params)
    assert params["numeric_mode"] == "identical" and params["fit_intercept"] is True
    assert clean(model.get_params, deep=False) == params
    assert clean(cls._get_param_names) == sorted(params)
    assert type(clean(cls._get_param_names)) is list
    assert clean(model.__sklearn_is_fitted__) is False
    if "sklearn" in sys.modules or _sklearn_available():
        import sklearn.utils  # noqa: F401  (sklearn asks for tags after importing itself)
        tags = clean(model.__sklearn_tags__)
        assert tags.estimator_type == "regressor" and tags.target_tags.required
        assert dataclasses.asdict(tags)["input_tags"]["two_d_array"] is True
    assert clean(getattr, model, "__class__") is cls
    assert clean(getattr, model, "fit").__self__ is model
    assert clean(getattr, model, "numeric_mode") == "identical"
    assert clean(model.numeric_mode_used) == "identical"
    if cls in (HostRidge, HostLinearRegression):
        assert clean(model.vendor_used) == "cpu"
    clean(getattr, model, "missing_native_attribute", error=AttributeError)
    clean(model.predict, np.ones((1, 3), "f4"), error=ValueError)


def _sklearn_available():
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.parametrize("cls", (Ridge, LinearRegression))
@pytest.mark.parametrize("layout", ("float32", "strided_float64"))
def test_fit_predict_score_and_checkpoint_are_native(cls, layout, monkeypatch, tmp_path):
    x, y = data()
    if layout == "strided_float64":
        wide_x, wide_y = np.empty((len(x), 6), "f8"), np.empty(len(y) * 2, "f8")
        wide_x[:, ::2], wide_y[::2] = x, y
        x, y = wide_x[::-1, ::2], wide_y[-2::-2]
        assert not x.flags.c_contiguous and not y.flags.c_contiguous
    from mojolearn import _backend, _buffer, _mode, _serialize
    for owner, name in ((_backend, "binding"), (_buffer, "as_f32_c"), (_mode, "_public_result"),
                        (_mode, "_ndarray"), (_serialize, "write_npz"), (_serialize, "read_npz")):
        monkeypatch.setattr(owner, name, forbidden)
    model = clean(cls, numeric_mode="identical")
    assert clean(model.fit, x, y) is model
    assert clean(model.__sklearn_is_fitted__) is True
    coef = clean(getattr, model, "coef_")
    assert type(coef) is np.ndarray and coef.dtype == np.float32 and coef.shape == (3,)
    xc = np.asarray(x, "f8") - np.asarray(x, "f8").mean(0)
    yc = np.asarray(y, "f8") - np.asarray(y, "f8").mean()
    alpha = 1.0 if cls is Ridge else 0.0
    reference = np.linalg.solve(xc.T @ xc + alpha * np.eye(3), xc.T @ yc)
    np.testing.assert_allclose(coef, reference, rtol=0, atol=1e-4)
    assert type(clean(getattr, model, "intercept_")) is float
    assert clean(getattr, model, "n_features_in_") == 3
    prediction = clean(model.predict, x)
    assert type(prediction) is np.ndarray and prediction.dtype == np.dtype("f8")
    # mojolearn's own callers keep the float32 words the kernels produced
    internal = model.predict.__func__  # the one-call shell
    narrow = eval("predict(model, x)", {"__name__": "mojolearn._probe", "predict": internal, "model": model, "x": x})
    assert narrow.dtype == np.float32
    np.testing.assert_array_equal(prediction, narrow.astype("f8"))
    score = clean(model.score, x, y)
    assert type(score) is float and score > .99
    first = tmp_path / "public-native.npz"
    clean(model.save, first)
    restored = clean(cls.load, first)
    np.testing.assert_array_equal(clean(restored.predict, x).view("u8"), prediction.view("u8"))
    again = io.BytesIO()
    clean(restored.save, again)
    assert again.getvalue() == first.read_bytes()
    assert clean(restored.set_params, fit_intercept=False) is restored
    assert not clean(restored.__sklearn_is_fitted__)
    clean(getattr, restored, "coef_", error=AttributeError)


@pytest.mark.parametrize("ordinary,host", ((Ridge, HostRidge), (LinearRegression, HostLinearRegression)))
def test_host_model_load_predict_digest_and_roundtrip_keep_cpu_route(ordinary, host, tmp_path):
    x, y = data()
    model = ordinary(numeric_mode="identical").fit(x, y)
    path = tmp_path / "linear.npz"
    model.save(path)
    loaded = clean(host.load, path)
    assert type(loaded) is host
    assert clean(loaded.vendor_used) == "cpu"
    assert clean(loaded.numeric_mode_used) == "identical"
    routed = mojolearn.host_model(path)
    assert type(routed) is host and routed.estimator == ordinary.__name__
    np.testing.assert_array_equal(clean(routed.predict, x).view("u8"), clean(loaded.predict, x).view("u8"))
    import hashlib
    assert clean(loaded.model_sha256) == hashlib.sha256(np.asarray(model.coef_).tobytes()).hexdigest()
    saved = io.BytesIO()
    clean(loaded.save, saved)
    assert saved.getvalue()


def test_fitted_array_attribute_view_keeps_alias_and_storage(monkeypatch):
    from mojolearn._array import Array
    value = Array((3,))
    storage = np.frombuffer(value._mv, dtype="f4")
    storage[:] = [.25, -.5, .75]
    model = clean(Ridge)
    object.__getattribute__(model, "__dict__")["coef_"] = value
    monkeypatch.setattr(Array, "__array__", forbidden)
    actual = clean(getattr, model, "coef_")
    assert type(actual) is np.ndarray
    assert np.shares_memory(actual, storage)
    assert clean(getattr, model, "__dict__")["coef_"] is value


@pytest.mark.parametrize("where", ("property", "descriptor", "getattr"))
def test_attribute_boundary_refuses_python_hooks_before_execution(where, monkeypatch):
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

    model = clean(Ridge)
    if where == "property":
        monkeypatch.setattr(Ridge, "coef_", property(hook), raising=False)
    elif where == "descriptor":
        monkeypatch.setattr(Ridge, "coef_", Descriptor(), raising=False)
    else:
        monkeypatch.setattr(Ridge, "__getattr__", hook, raising=False)
    clean(getattr, model, "coef_", error=TypeError)
    assert not seen


def test_constructor_refuses_python_scalar_without_running_it():
    seen = []

    class Scalar:
        def __float__(self):
            seen.append("float")
            raise AssertionError("scalar callback")

    clean(Ridge, alpha=Scalar(), error=TypeError)
    assert not seen


@pytest.mark.parametrize("cls", (Ridge, LinearRegression))
def test_calls_release_the_model_and_its_inputs(cls, tmp_path):
    """No reference is kept on the caller's objects, model or outputs."""
    x, y = data()
    weights = np.linspace(.5, 1.5, len(y))

    def run():
        model = cls(numeric_mode="identical")
        counts = [sys.getrefcount(v) for v in (x, y, weights)]
        model.fit(x, y)
        model.fit(x, y, sample_weight=weights)
        out = model.predict(x)
        model.score(x, y)
        model.get_params()
        model.coef_, model.intercept_, model.n_features_in_
        model.save(tmp_path / "m.npz")
        loaded = cls.load(tmp_path / "m.npz")
        assert [sys.getrefcount(v) for v in (x, y, weights)] == counts
        return [weakref.ref(model), weakref.ref(out), weakref.ref(loaded),
                weakref.ref(object.__getattribute__(model, "coef_"))]

    references = run()
    gc.collect()
    assert [r for r in references if r() is not None] == []


@pytest.mark.parametrize("cls", (Ridge, LinearRegression))
def test_steady_state_calls_allocate_nothing_lasting(cls):
    x, y = data()
    model = cls(numeric_mode="identical")

    def cycle():
        for _ in range(200):  # repeated whole public operations
            model.fit(x, y).predict(x)
            model.score(x, y)
            cls().get_params()

    cycle()
    cycle()
    gc.collect()
    before = sys.getallocatedblocks()
    cycle()
    gc.collect()
    # One retained object per call would add at least 800 blocks.
    assert sys.getallocatedblocks() - before < 200
