# SPDX-License-Identifier: Apache-2.0
"""Ridge Python admission, callback and fitted-output publication contracts."""
import numpy as np
import pytest

from mojolearn import Ridge
from mojolearn._native_classical_bridge import connect
from mojolearn._verify_native_runtime import audit_native_call

ENTRY, CONTEXT = connect(Ridge)


def native_fit(model, x, y, weights=None):
    return ENTRY(model, "ridge_fit", (x, y, weights), CONTEXT)


def native_predict(model, x):
    return ENTRY(model, "predict", (x,), CONTEXT)


def clean(audit, error=None):
    report = audit.report()
    assert not report["python_frames"]
    assert not report["forbidden_native_calls"]
    assert not report["wrapper_work"]
    assert not report["unobserved_existing_threads"]
    if error is None:
        return audit.assert_native()
    assert isinstance(audit.exception, error)


def bits(value):
    return np.asarray(value, dtype=np.float32).view(np.uint32)


@pytest.mark.parametrize("layout", ["ndarray", "strided", "memoryview", "list"])
def test_integer_prediction_preserves_binary64_intermediate(layout):
    value = 2**62 + 2**38 + 1
    original = np.array([value, -value, value - 2, -value + 2], dtype="i8")[:, None]
    source = {"ndarray": lambda: original, "strided": lambda: original[::-1],
              "memoryview": lambda: memoryview(original),
              "list": lambda: original.tolist()}[layout]()
    canonical = np.asarray(source).astype("f8").astype("f4")[:, 0]
    model = Ridge(fit_intercept=False)
    model.coef_ = np.ones(1, dtype="f4")
    model.intercept_ = 0.0
    model.n_features_in_ = 1
    old = np.asarray(model.predict(source), dtype="f4")
    np.testing.assert_array_equal(old.view("u4"), canonical.view("u4"))
    actual = clean(audit_native_call(native_predict, model, source))
    np.testing.assert_array_equal(bits(actual), old.view("u4"))
    direct = np.asarray(source).astype("f4")[:, 0]
    assert np.any(direct.view("u4") != canonical.view("u4"))


@pytest.mark.parametrize("cols,intercept,weights", [
    (3, True, None), (3, False, None), (1, True, None), (1, False, None),
    (3, True, "vector"), (3, False, "vector"), (3, True, "scalar"), (1, True, "vector"),
])
def test_ridge_native_owner_publishes_fitted_outputs(cols, intercept, weights):
    rng = np.random.default_rng(1013)
    x = rng.normal(size=(24, cols)).astype("f4")
    truth = np.arange(1, cols + 1, dtype="f4") / 2
    y = (x @ truth + (1.25 if intercept else 0)).astype("f4")
    w = (np.linspace(0, 2, len(x), dtype="f4") if weights == "vector" else
         2.0 if weights == "scalar" else None)
    # Numerical solve/predict expectations run directly against the typed
    # native descriptors in checks/native_tests/linear_owner_{cpu,gpu}.mojo.
    before_x, before_y = x.copy(), y.copy()
    out = Ridge(alpha=0.25, fit_intercept=intercept)
    assert clean(audit_native_call(native_fit, out, x, y, w)) is out
    prediction = clean(audit_native_call(native_predict, out, x))
    assert np.asarray(out.coef_).shape == (cols,)
    assert np.asarray(out.coef_).dtype == np.dtype("f4")
    assert np.asarray(out.intercept_).shape == ()
    assert out.n_features_in_ == cols
    assert np.asarray(prediction).shape == (len(x),)
    assert np.asarray(prediction).dtype.kind == "f"
    np.testing.assert_array_equal(x, before_x)
    np.testing.assert_array_equal(y, before_y)


@pytest.mark.parametrize("layout", ["fortran", "strided", "list", "memoryview", "float64", "float16"])
def test_ridge_native_owner_input_preparation_and_poisoned_python_helpers(monkeypatch, layout):
    base = np.arange(72, dtype="f4").reshape(24, 3) / 8
    base[:, 1] = np.sin(base[:, 1])
    base[:, 2] = np.cos(base[:, 2])
    x = {"fortran": lambda: np.asfortranarray(base), "strided": lambda: base[::-1],
         "list": lambda: base.tolist(), "memoryview": lambda: memoryview(base),
         "float64": lambda: base.astype("f8"), "float16": lambda: base.astype("f2")}[layout]()
    y = np.linspace(-1, 2, len(base), dtype="f4")
    canonical = np.asarray(x, dtype="f4")
    old = Ridge(alpha=0.5).fit(canonical, y)
    expected = np.asarray(old.predict(canonical)).copy()
    out = Ridge(alpha=0.5)

    def forbidden(*args, **kwargs):
        raise AssertionError("Python preparation callback reached")

    from mojolearn import _backend, _buffer
    monkeypatch.setattr(_backend, "binding", forbidden)
    monkeypatch.setattr(_buffer, "as_f32_c", forbidden)
    monkeypatch.setattr(np, "empty", forbidden)
    clean(audit_native_call(native_fit, out, x, y))
    got = clean(audit_native_call(native_predict, out, x))
    np.testing.assert_array_equal(bits(got), bits(expected))


@pytest.mark.parametrize("weights", [np.zeros(8, "f4"), -np.ones(8, "f4"),
                                     np.full(8, np.nan, "f4"), np.ones(7, "f4")])
def test_ridge_native_owner_weight_refusals_are_native(weights):
    model = Ridge()
    x = np.arange(16, dtype="f4").reshape(8, 2)
    y = np.arange(8, dtype="f4")
    clean(audit_native_call(native_fit, model, x, y, weights), ValueError)


def test_ridge_native_owner_refuses_python_array_subclass_before_properties():
    calls = []

    class Poison(np.ndarray):
        @property
        def dtype(self):
            calls.append("dtype")
            raise AssertionError("subclass dtype executed")

    x = np.arange(16, dtype="f4").reshape(8, 2).view(Poison)
    clean(audit_native_call(native_fit, Ridge(), x, np.arange(8, dtype="f4")), TypeError)
    assert not calls
