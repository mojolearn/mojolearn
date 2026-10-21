# SPDX-License-Identifier: Apache-2.0
"""Native metadata and full predict-to-score private owner boundaries."""
import numpy as np
import pytest

from mojolearn import LinearRegression, Ridge
from mojolearn._native_classical_bridge import connect
from mojolearn._verify_native_runtime import audit_native_call

ENTRY, CONTEXT = connect(Ridge, LinearRegression)


def ridge_initialize(model, alpha=1.0, solver="auto", fit_intercept=True, normalize=False, numeric_mode=None):
    return ENTRY(model, "initialize", (alpha, solver, fit_intercept, normalize, numeric_mode), CONTEXT)


def linear_initialize(model, fit_intercept=True, numeric_mode=None):
    return ENTRY(model, "initialize", (fit_intercept, numeric_mode), CONTEXT)


def native_get(model, deep=True):
    return ENTRY(model, "get_params", (deep,), CONTEXT)


def native_set(model, **params):
    return ENTRY(model, "set_params", (params,), CONTEXT)


def native_fitted(model):
    return ENTRY(model, "is_fitted", (), CONTEXT)


def native_mode(model):
    return ENTRY(model, "numeric_mode_used", (), CONTEXT)


def native_vendor(model):
    return ENTRY(model, "vendor_used", (), CONTEXT)


def native_score(model, x, y):
    return ENTRY(model, "score", (x, y), CONTEXT)


def clean(audit, error=None):
    report = audit.report()
    assert not report["python_frames"]
    assert not report["forbidden_native_calls"]
    assert not report["wrapper_work"]
    assert not report["unobserved_existing_threads"]
    if error is None:
        return audit.assert_native()
    assert isinstance(audit.exception, error)


@pytest.mark.parametrize("cls,initialize", [(Ridge, ridge_initialize), (LinearRegression, linear_initialize)])
def test_native_constructor_parameters_fitted_reset_and_binding_metadata(cls, initialize):
    expected = cls()
    # Allocation is the caller's Python object ABI, matching __init__ entry.
    model = object.__new__(cls)
    assert clean(audit_native_call(initialize, model)) is None
    assert clean(audit_native_call(native_get, model)) == expected.get_params()
    assert clean(audit_native_call(native_get, model, False)) == expected.get_params(False)
    assert clean(audit_native_call(native_fitted, model)) is False
    assert clean(audit_native_call(native_mode, model)) == expected.numeric_mode_used()
    assert clean(audit_native_call(native_vendor, model)) == expected.vendor_used()
    model.coef_ = np.array([1], dtype="f4")
    assert clean(audit_native_call(native_fitted, model)) is True
    assert clean(audit_native_call(native_set, model)) is model
    assert "coef_" in vars(model)
    assert clean(audit_native_call(native_set, model, fit_intercept=False)) is model
    assert clean(audit_native_call(native_get, model))["fit_intercept"] is False
    assert clean(audit_native_call(native_fitted, model)) is False
    assert "coef_" not in vars(model)


@pytest.mark.parametrize("params,error", [({"nonsense": 1}, ValueError),
                                         ({"alpha__unknown": 1}, AttributeError)])
def test_invalid_native_parameter_update_preserves_fitted_state(params, error):
    model = Ridge()
    model.coef_ = np.array([3], dtype="f4")
    original = vars(model).copy()
    clean(audit_native_call(native_set, model, **params), error)
    assert vars(model).keys() == original.keys()
    assert model.coef_ is original["coef_"]


def test_native_metadata_refuses_python_parameter_protocol_before_hook():
    calls = []

    class Poison:
        def __bool__(self):
            calls.append("bool")
            raise AssertionError("Python bool hook reached")

        def get_params(self, **kwargs):
            calls.append("get_params")
            raise AssertionError("Python nested parameter reached")

    clean(audit_native_call(ridge_initialize, object.__new__(Ridge), alpha=Poison()), TypeError)
    model = Ridge()
    model._mojolearn_fitted = Poison()
    clean(audit_native_call(native_fitted, model), TypeError)
    assert not calls


@pytest.mark.parametrize("cls", [Ridge, LinearRegression])
@pytest.mark.parametrize("case", ["ordinary", "constant", "single_row", "strided", "float64"])
@pytest.mark.parametrize("ranges", ["0", "1"])
def test_native_score_preserves_existing_grouped_program_and_independent_value(monkeypatch, cls, case, ranges):
    monkeypatch.setenv("MOJOLEARN_ARENA_RANGES", ranges)
    rng = np.random.default_rng(319)
    x = rng.normal(size=(35, 3)).astype("f4")
    y = (x @ np.array([0.25, -0.75, 0.5], dtype="f4") + 1.0).astype("f4")
    model = cls().fit(x, y)
    if case == "constant":
        y[:] = 2
    elif case == "single_row":
        x, y = x[:1], y[:1]
    elif case == "strided":
        x, y = x[::-1], y[::-1]
    elif case == "float64":
        x, y = x.astype("f8"), y.astype("f8")
    expected = np.float64(model.score(x, y))
    prediction = np.asarray(model.predict(x), dtype="f4")

    def forbidden(*args, **kwargs):
        raise AssertionError("Python score program reached")

    from mojolearn import _expansion_metrics, _backend
    with monkeypatch.context() as patch:
        patch.setattr(_expansion_metrics, "_execute", forbidden)
        patch.setattr(_backend, "binding", forbidden)
        actual = np.float64(clean(audit_native_call(native_score, model, x, y)))
    assert actual.view("u8") == expected.view("u8")
    yd = np.asarray(y, dtype="f4").astype("f8")
    numerator = np.sum((yd - prediction.astype("f8")) ** 2)
    denominator = np.sum((yd - yd.mean()) ** 2)
    independent = 0.0 if denominator == 0 else 1 - numerator / denominator
    np.testing.assert_allclose(actual, independent, rtol=3e-5, atol=3e-6)
