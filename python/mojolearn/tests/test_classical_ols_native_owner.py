# SPDX-License-Identifier: Apache-2.0
"""OLS Python admission, callback and fitted-output publication contracts."""
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
        # The square route is supported by both the CPU and GPU OLS kernels.
        x = np.array([[1, 0], [0, 1]], dtype="f4")
        intercept = False
    elif case == "wide_minimum_norm":
        # Keep the original wide fixture: GPU kernels support minimum norm,
        # while the existing CPU kernel explicitly refuses this operation.
        x = np.array([[1, 0, 0], [0, 1, 0]], dtype="f4")
        intercept = False
    truth = np.arange(1, x.shape[1] + 1, dtype="f4") / 4
    y = (x @ truth + (1.25 if intercept else 0)).astype("f4")
    return x, y, intercept, weight


@pytest.mark.parametrize("case", ["full_rank", "no_intercept", "one_feature", "weighted",
                                  "weighted_no_intercept", "scalar_weight", "rank_deficient",
                                  "constant_column", "zero_rank", "column_units", "normal_equations_shape"])
def test_ols_native_publishes_fitted_outputs(case):
    x, y, intercept, weight = fixture(case)
    # Numerical full/rank-deficient/weighted solves and consumed predictions
    # are independent native fixtures in linear_owner_{cpu,gpu}.mojo.
    before_x, before_y = x.copy(), y.copy()
    native = LinearRegression(fit_intercept=intercept)
    assert clean(audit_native_call(native_fit, native, x, y, weight)) is native
    prediction = clean(audit_native_call(native_predict, native, x))
    assert np.asarray(native.coef_).shape == (x.shape[1],)
    assert np.asarray(native.coef_).dtype == np.dtype("f4")
    assert np.asarray(native.intercept_).shape == ()
    assert native.n_features_in_ == x.shape[1]
    assert np.asarray(prediction).shape == (len(x),)
    assert np.asarray(prediction).dtype.kind == "f"
    np.testing.assert_array_equal(x, before_x)
    np.testing.assert_array_equal(y, before_y)


def test_ols_native_preserves_wide_minimum_norm_support_boundary():
    x, y, intercept, weight = fixture("wide_minimum_norm")
    native = LinearRegression(fit_intercept=intercept)
    audit = audit_native_call(native_fit, native, x, y, weight)
    if CONTEXT["vendor"] == "cpu":
        # This is a verified refusal, not a skipped correctness fixture: the
        # owner must reach the same unsupported native kernel without Python.
        clean(audit, ValueError)
        assert "lstsq_min_norm" in str(audit.exception)
        assert "no CPU restatement" in str(audit.exception)
        assert "coef_" not in vars(native)
        with pytest.raises(Exception, match="lstsq_min_norm.*no CPU restatement"):
            LinearRegression(fit_intercept=intercept).fit(x, y, sample_weight=weight)
        return
    assert clean(audit) is native
    prediction = clean(audit_native_call(native_predict, native, x))
    # Direct native fixtures check the independent [0.25, 0.5, 0] minimum
    # norm coefficients. Keep this public result/error publication boundary.
    assert np.asarray(native.coef_).shape == (3,)
    assert native.n_features_in_ == 3
    assert np.asarray(prediction).shape == (2,)


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
        # Ordinary column targets are supported and have warning coverage.
        # A rank-three target remains invalid.
        y = y[:, None, None]
    elif case == "target_length":
        y = y[:-1]
    elif case == "zero_weights":
        weight = np.zeros(len(x), dtype="f4")
    elif case == "negative_weights":
        weight = -np.ones(len(x), dtype="f4")
    clean(audit_native_call(native_fit, LinearRegression(), x, y, weight), ValueError)


class _ArrayOnly:
    """scikit-learn's _NotAnArray shape: only __array__, no buffer or sequence."""

    def __init__(self, data):
        self.data = data

    def __array__(self, dtype=None, copy=None):
        raise AssertionError("__array__ must not run in a native fit")


@pytest.mark.parametrize("estimator", [LinearRegression, Ridge])
def test_array_only_sample_weight_refusal_names_accepted_types(estimator):
    x, y, _, _ = fixture("full_rank")
    with pytest.raises(TypeError) as caught:
        estimator().fit(x, y, sample_weight=_ArrayOnly(np.ones(len(x), dtype="f4")))
    message = str(caught.value)
    assert "sample_weight must be None, a number, a NumPy array, a buffer" in message
    assert "list or tuple of numbers" in message
    assert "_ArrayOnly" in message and "__array__" in message
