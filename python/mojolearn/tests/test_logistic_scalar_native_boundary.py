# SPDX-License-Identifier: Apache-2.0
"""Actual NumPy policy, public configuration and refusal-order seams.

Arithmetic/promotion words are checked in native logistic_config_suite.mojo.
These tests retain the Python boundary behavior that a pure scalar cannot model:
active NumPy promotion/error policy, warning recording, callback refusal and
public parameter owners. Reference expressions never call Mojolearn helpers.
"""
import warnings

import numpy as np
import pytest

from mojolearn.linear_model import LogisticRegression
from mojolearn._verify_native_runtime import audit_native_call


def _inputs():
    x = np.array([[-2., 1.], [-1., 1.], [-.5, 0.],
                  [.5, 0.], [1., -1.], [2., -1.]], dtype="f4")
    return x, np.array([0, 0, 0, 1, 1, 1], dtype="i8")


def _tiny(dtype):
    # Exact smallest subnormal, without causing a setup-time floating warning.
    return np.array([1], dtype=f"u{np.dtype(dtype).itemsize}").view(dtype)[0]


def _clean(audit):
    report = audit.report()
    for name in ("python_frames", "forbidden_native_calls", "wrapper_work",
                 "unobserved_existing_threads"):
        assert not report[name], report


def _reference_event(parameter, value, policy):
    mode = "warn" if policy == "warning_error" else policy
    error = None
    with warnings.catch_warnings(record=True) as caught, np.errstate(all=mode):
        warnings.simplefilter("error" if policy == "warning_error" else "always", RuntimeWarning)
        try:
            if parameter == "C":
                1.0 / value
            else:
                float(value)  # The public tolerance converts before multiplying.
                value * .01
        except Exception as exc:
            error = exc
    return error, [(item.category, str(item.message)) for item in caught]


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("parameter", ["C", "tol"])
@pytest.mark.parametrize("policy", ["ignore", "warn", "raise", "warning_error"])
def test_native_scalar_preserves_active_numpy_promotion_and_error_policy(dtype, parameter, policy):
    value = _tiny(dtype)
    reference_error, reference_warnings = _reference_event(parameter, value, policy)
    x, y = _inputs()
    original = x.tobytes()
    # This intentional later error stops before QN RUN. It distinguishes scalar
    # warning/refusal behavior from any downstream solver response to infinity.
    model = LogisticRegression(**{parameter: value}, max_iter="invalid")
    mode = "warn" if policy == "warning_error" else policy
    with warnings.catch_warnings(record=True) as caught, np.errstate(all=mode):
        warnings.simplefilter("error" if policy == "warning_error" else "always", RuntimeWarning)
        audit = audit_native_call(model.fit, x, y)
    _clean(audit)
    assert [(item.category, str(item.message)) for item in caught] == reference_warnings
    if reference_error is None:
        assert type(audit.exception) is ValueError, audit.report()
        assert "invalid literal" in str(audit.exception)
    else:
        assert type(audit.exception) is type(reference_error), audit.report()
        assert str(audit.exception) == str(reference_error)
    assert x.tobytes() == original
    assert model.get_params()[parameter] is value
    assert "_w" not in vars(model) and "coef_" not in vars(model)


@pytest.mark.parametrize("parameter", ["C", "tol"])
def test_native_scalar_refuses_numpy_error_callback_without_invocation(parameter):
    calls = []

    def callback(*args):
        calls.append(args)

    model = LogisticRegression(**{parameter: _tiny(np.float64)}, max_iter="invalid")
    x, y = _inputs()
    previous = np.seterrcall(callback)
    try:
        with np.errstate(over="call", under="call", invalid="call"):
            audit = audit_native_call(model.fit, x, y)
    finally:
        np.seterrcall(previous)
    _clean(audit)
    assert type(audit.exception) is TypeError, audit.report()
    assert "Python NumPy error callbacks" in str(audit.exception)
    assert not calls
    assert "_w" not in vars(model) and "coef_" not in vars(model)


@pytest.mark.parametrize("tolerance", ["0.01", "not-a-number", None])
def test_native_tolerance_preserves_float_before_multiplication_refusal(tolerance):
    try:
        float(tolerance)
        tolerance * .01
    except Exception as exc:
        expected_type, expected_message = type(exc), str(exc)
    else:
        raise AssertionError("fixture must refuse in the original tolerance expression")
    x, y = _inputs()
    model = LogisticRegression(tol=tolerance, max_iter=1)
    audit = audit_native_call(model.fit, x, y)
    _clean(audit)
    assert type(audit.exception) is expected_type, audit.report()
    assert str(audit.exception) == expected_message
    assert "_w" not in vars(model) and "coef_" not in vars(model)


@pytest.mark.parametrize("configuration,match", [
    ({"penalty": "bad", "solver": "bad", "C": -1, "l1_ratio": None}, "`penalty`"),
    ({"penalty": "elasticnet", "solver": "bad", "C": -1, "l1_ratio": None}, "Only quasi-newton"),
    ({"penalty": "elasticnet", "solver": "qn", "C": 0, "l1_ratio": None}, "C must be positive"),
    ({"penalty": "elasticnet", "solver": "qn", "C": 1, "l1_ratio": None}, "l1_ratio has to be specified"),
    ({"penalty": "elasticnet", "solver": "qn", "C": 1, "l1_ratio": 2}, "between 0.0 and 1.0"),
])
def test_native_logistic_configuration_refuses_in_original_order(configuration, match):
    x, y = _inputs()
    model = LogisticRegression(**configuration, tol="not-a-number", max_iter=1)
    parameters = model.get_params()
    audit = audit_native_call(model.fit, x, y)
    _clean(audit)
    assert type(audit.exception) is ValueError, audit.report()
    assert match in str(audit.exception)
    after = model.get_params()
    assert all(after[name] is value for name, value in parameters.items())
    assert "_w" not in vars(model) and "coef_" not in vars(model)


def test_no_penalty_preserves_huge_positive_integer_owner_without_strength_conversion():
    x, y = _inputs()
    huge = 1 << 20000
    model = LogisticRegression(penalty=None, C=huge, max_iter=1)
    audit = audit_native_call(model.fit, x, y)
    _clean(audit)
    assert audit.assert_native() is model
    assert model.C is huge
    assert model.coef_.shape == (1, 2) and model.intercept_.shape == (1,)
    # The same owner is converted when an actual penalty needs 1/C.
    penalized = LogisticRegression(penalty="l2", C=huge, max_iter=1)
    refused = audit_native_call(penalized.fit, x, y)
    _clean(refused)
    assert type(refused.exception) is OverflowError, refused.report()
    assert "too large" in str(refused.exception)
    assert penalized.C is huge
    assert "_w" not in vars(penalized) and "coef_" not in vars(penalized)
