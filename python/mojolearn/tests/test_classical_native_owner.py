# SPDX-License-Identifier: Apache-2.0
"""Private owner qualification; the public classical migration remains open."""
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


@pytest.mark.parametrize("cols,intercept,weights", [
    (3, True, None), (3, False, None), (1, True, None), (1, False, None),
    (3, True, "vector"), (3, False, "vector"), (3, True, "scalar"), (1, True, "vector"),
])
def test_ridge_native_owner_preserves_existing_driver_and_independent_solution(cols, intercept, weights):
    rng = np.random.default_rng(1013)
    x = rng.normal(size=(24, cols)).astype("f4")
    truth = np.arange(1, cols + 1, dtype="f4") / 2
    y = (x @ truth + (1.25 if intercept else 0)).astype("f4")
    w = (np.linspace(0, 2, len(x), dtype="f4") if weights == "vector" else
         2.0 if weights == "scalar" else None)
    old = Ridge(alpha=0.25, fit_intercept=intercept).fit(x, y, sample_weight=w)
    expected_coef = np.asarray(old.coef_).copy()
    expected_intercept = np.float64(old.intercept_)
    expected_prediction = np.asarray(old.predict(x)).copy()
    out = Ridge(alpha=0.25, fit_intercept=intercept)
    assert clean(audit_native_call(native_fit, out, x, y, w)) is out
    prediction = clean(audit_native_call(native_predict, out, x))
    np.testing.assert_array_equal(bits(out.coef_), bits(expected_coef))
    assert np.float64(out.intercept_).view("u8") == expected_intercept.view("u8")
    np.testing.assert_array_equal(bits(prediction), bits(expected_prediction))
    # Independent float64 ridge equations, including weighted centering.
    weight = np.ones(len(x)) if w is None else np.broadcast_to(w, (len(x),)).astype("f8")
    xd, yd = x.astype("f8"), y.astype("f8")
    mx = np.average(xd, axis=0, weights=weight) if intercept else np.zeros(cols)
    my = np.average(yd, weights=weight) if intercept else 0.0
    centered = xd - mx
    wanted = np.linalg.solve(centered.T @ (weight[:, None] * centered) + 0.25 * np.eye(cols),
                             centered.T @ (weight * (yd - my)))
    np.testing.assert_allclose(out.coef_, wanted, rtol=3e-4, atol=3e-5)
    np.testing.assert_allclose(out.intercept_, my - mx @ wanted, rtol=3e-4, atol=3e-5)


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
