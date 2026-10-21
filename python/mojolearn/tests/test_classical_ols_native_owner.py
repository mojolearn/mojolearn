# SPDX-License-Identifier: Apache-2.0
"""Private OLS owner qualification against the retained driver and algebra."""
import numpy as np
import pytest

from mojolearn import LinearRegression, Ridge
from mojolearn._native_classical_bridge import connect
from mojolearn._verify_native_runtime import audit_native_call

ENTRY, CONTEXT = connect(Ridge, LinearRegression)


def native_fit(model, x, y, weights=None):
    return ENTRY(model, "ols_fit", (x, y, weights), CONTEXT)


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
    return np.asarray(value, dtype="f4").view("u4")


def fixture(case):
    rng = np.random.default_rng(2913)
    x = rng.normal(size=(29, 4)).astype("f4")
    intercept = True
    weight = None
    if case == "no_intercept":
        intercept = False
    elif case == "one_feature":
        x = x[:, :1].copy()
    elif case == "weighted":
        weight = np.linspace(0, 2, len(x), dtype="f4")
    elif case == "weighted_no_intercept":
        weight = np.linspace(0, 2, len(x), dtype="f4")
        intercept = False
    elif case == "scalar_weight":
        weight = 2.0
    elif case == "rank_deficient":
        x[:, 3] = x[:, 1]
    elif case == "constant_column":
        x[:, 3] = 3.0
    elif case == "zero_rank":
        x[:] = 0
    elif case == "column_units":
        x *= np.array([1 / 64, 1, 8, 128], dtype="f4")
    elif case == "normal_equations_shape":
        # TSQR's documented rows >= cols+1 condition is false here.
        x = np.array([[1, 0, 0], [0, 1, 0]], dtype="f4")
        intercept = False
    truth = np.arange(1, x.shape[1] + 1, dtype="f4") / 4
    y = (x @ truth + (1.25 if intercept else 0)).astype("f4")
    return x, y, intercept, weight


@pytest.mark.parametrize("case", ["full_rank", "no_intercept", "one_feature", "weighted",
                                  "weighted_no_intercept", "scalar_weight", "rank_deficient",
                                  "constant_column", "zero_rank", "column_units", "normal_equations_shape"])
def test_ols_native_preserves_driver_bits_and_independent_predictions(case):
    x, y, intercept, weight = fixture(case)
    legacy = LinearRegression(fit_intercept=intercept).fit(x, y, sample_weight=weight)
    expected_coef = np.asarray(legacy.coef_).copy()
    expected_bias = np.float64(legacy.intercept_)
    expected_pred = np.asarray(legacy.predict(x)).copy()
    native = LinearRegression(fit_intercept=intercept)
    assert clean(audit_native_call(native_fit, native, x, y, weight)) is native
    prediction = clean(audit_native_call(native_predict, native, x))
    np.testing.assert_array_equal(bits(native.coef_), bits(expected_coef))
    assert np.float64(native.intercept_).view("u8") == expected_bias.view("u8")
    np.testing.assert_array_equal(bits(prediction), bits(expected_pred))
    # Independent weighted float64 least squares; compare consumed predictions
    # since rank-deficient models can have several equivalent coefficients.
    wd = np.ones(len(x)) if weight is None else np.broadcast_to(weight, (len(x),)).astype("f8")
    xd, yd = x.astype("f8"), y.astype("f8")
    mx = np.average(xd, axis=0, weights=wd) if intercept else np.zeros(x.shape[1])
    my = np.average(yd, weights=wd) if intercept else 0.0
    sw = np.sqrt(wd)
    answer = np.linalg.lstsq((xd - mx) * sw[:, None], (yd - my) * sw, rcond=None)[0]
    independent = (xd - mx) @ answer + my
    np.testing.assert_allclose(prediction, independent, rtol=5e-4, atol=3e-4)


@pytest.mark.parametrize("layout", ["fortran", "negative_stride", "nested", "memoryview", "float64", "float16"])
def test_ols_native_prepares_native_buffers_without_python_drivers(monkeypatch, layout):
    x, y, _, _ = fixture("full_rank")
    x = {"fortran": lambda: np.asfortranarray(x), "negative_stride": lambda: x[::-1],
         "nested": lambda: x.tolist(), "memoryview": lambda: memoryview(x),
         "float64": lambda: x.astype("f8"), "float16": lambda: x.astype("f2")}[layout]()
    canonical = np.asarray(x, dtype="f4")
    legacy = LinearRegression().fit(canonical, y)
    expected = np.asarray(legacy.predict(canonical)).copy()
    native = LinearRegression()

    def forbidden(*args, **kwargs):
        raise AssertionError("Python classical driver reached")

    from mojolearn import _backend, _buffer, _expansion_decomp
    with monkeypatch.context() as patch:
        patch.setattr(_backend, "binding", forbidden)
        patch.setattr(_buffer, "as_f32_c", forbidden)
        patch.setattr(_expansion_decomp._Kit, "mm", forbidden)
        patch.setattr(_expansion_decomp._M, "s", property(forbidden))
        patch.setattr(np, "empty", forbidden)
        clean(audit_native_call(native_fit, native, x, y))
        got = clean(audit_native_call(native_predict, native, x))
    np.testing.assert_array_equal(bits(got), bits(expected))


@pytest.mark.parametrize("case", ["one_row", "target_rank", "target_length", "zero_weights", "negative_weights"])
def test_ols_native_admission_errors_do_not_enter_python(case):
    x, y, _, _ = fixture("full_rank")
    weight = None
    if case == "one_row":
        x, y = x[:1], y[:1]
    elif case == "target_rank":
        y = y[:, None]
    elif case == "target_length":
        y = y[:-1]
    elif case == "zero_weights":
        weight = np.zeros(len(x), dtype="f4")
    elif case == "negative_weights":
        weight = -np.ones(len(x), dtype="f4")
    clean(audit_native_call(native_fit, LinearRegression(), x, y, weight), ValueError)
