# SPDX-License-Identifier: Apache-2.0
"""Public feature-field UNPACK, policy and refusal-order seams.

Exact equality and scalar rounding live in logistic_query_width_suite.mojo.
These tests keep actual public owners, native buffer admission, live NumPy
warning policy and callback refusal at the Python boundary.
"""
import warnings

import numpy as np
import pytest

from mojolearn._array import Array
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.linear_model import LogisticRegression


def _model(width=2):
    model = LogisticRegression(fit_intercept=False)
    weights = np.zeros(width, dtype="f4")
    weights[0] = 1
    model._w = Array.from_buffer(weights)
    model.classes_ = [-7, 11]
    model.n_features_in_ = width
    x = np.zeros((2, width), dtype="f4")
    x[:, 0] = [-1, 1]
    return model, x


def _clean(audit):
    report = audit.report()
    for name in ("python_frames", "forbidden_native_calls", "wrapper_work",
                 "unobserved_existing_threads"):
        assert not report[name], report


@pytest.mark.parametrize("metadata", [
    2, 2.0, 2 + 0j, np.int64(2), np.uint64(2), np.float16(2),
    np.float32(2), np.float64(2), np.complex64(2),
    np.array(2, dtype=">i8"), np.array([[2]], dtype=">f8"),
    np.array([2 + 0j], dtype=">c16"), np.array([2], dtype=object),
])
def test_feature_metadata_uses_native_storage_and_preserves_public_owner(metadata):
    model, x = _model()
    weights = model._w
    model.n_features_in_ = metadata
    before = x.tobytes()
    audit = audit_native_call(model.decision_function, x)
    _clean(audit)
    np.testing.assert_array_equal(np.asarray(audit.assert_native()), [-1., 1.])
    assert model.n_features_in_ is metadata and model._w is weights
    assert x.tobytes() == before


@pytest.mark.parametrize("metadata", [
    3, 2.5, 2 + 1j, 1 << 200, np.nan, np.inf,
    np.array([2, 2]), np.array([], dtype="f8"),
    np.array([(2,)], dtype=[("width", "i8")]),
    np.array(["2"]), np.array([None], dtype=object),
])
def test_feature_refusal_matches_original_comparison_and_live_numpy_policy(metadata):
    model, x = _model()
    model.n_features_in_ = metadata
    reference_error = None
    with warnings.catch_warnings(record=True) as reference_warnings:
        warnings.simplefilter("always")
        try:
            if 2 != metadata:
                raise ValueError("mojolearn LogisticRegression feature count differs from fit")
        except Exception as exc:
            reference_error = exc
    with warnings.catch_warnings(record=True) as native_warnings:
        warnings.simplefilter("always")
        audit = audit_native_call(model.decision_function, x)
    _clean(audit)
    assert [(type(w.message), str(w.message)) for w in native_warnings] == [
        (type(w.message), str(w.message)) for w in reference_warnings]
    if reference_error is None:
        # NumPy before2.2 admitted legacy empty-array truth with a warning.
        np.testing.assert_array_equal(np.asarray(audit.assert_native()), [-1., 1.])
    else:
        assert type(audit.exception) is type(reference_error), audit.report()
        assert str(audit.exception) == str(reference_error)
    assert model.n_features_in_ is metadata


def test_native_array_feature_metadata_uses_its_exact_scalar_storage():
    model, x = _model()
    metadata = Array.from_buffer(np.array([2.0], dtype="f4"))
    model.n_features_in_ = metadata
    audit = audit_native_call(model.decision_function, x)
    _clean(audit)
    np.testing.assert_array_equal(np.asarray(audit.assert_native()), [-1., 1.])
    assert model.n_features_in_ is metadata
    model.n_features_in_ = Array.from_buffer(np.array([2.0, 2.0], dtype="f4"))
    refused = audit_native_call(model.decision_function, x)
    _clean(refused)
    assert type(refused.exception) is ValueError, refused.report()
    assert "truth value" in str(refused.exception) and "ambiguous" in str(refused.exception)


def test_binary_predict_retains_integer_conversion_before_width_check():
    model, x = _model()
    model.n_features_in_ = 2.75
    predict = audit_native_call(model.predict, x)
    _clean(predict)
    np.testing.assert_array_equal(np.asarray(predict.assert_native()), [-7, 11])
    decision = audit_native_call(model.decision_function, x)
    _clean(decision)
    assert type(decision.exception) is ValueError, decision.report()
    assert "feature count differs from fit" in str(decision.exception)
    assert model.n_features_in_ == 2.75


def test_object_array_metadata_refuses_user_equality_and_conversion_hooks():
    calls = []

    class Hook:
        def __eq__(self, other):
            calls.append("eq")
            return True

        def __ne__(self, other):
            calls.append("ne")
            return False

        def __int__(self):
            calls.append("int")
            return 2

        def __float__(self):
            calls.append("float")
            return 2.0

    metadata = np.empty(1, dtype=object)
    metadata[0] = Hook()
    model, x = _model()
    model.n_features_in_ = metadata
    audit = audit_native_call(model.decision_function, x)
    _clean(audit)
    assert type(audit.exception) is TypeError, audit.report()
    assert not calls
    assert model.n_features_in_ is metadata


def test_invalid_input_is_refused_before_ambiguous_feature_metadata():
    model, _ = _model()
    invalid_x = np.zeros((1, 1, 2), dtype="f4")
    baseline = audit_native_call(model.decision_function, invalid_x)
    _clean(baseline)
    assert baseline.exception is not None
    metadata = np.array([2, 2])
    model.n_features_in_ = metadata
    actual = audit_native_call(model.decision_function, invalid_x)
    _clean(actual)
    assert type(actual.exception) is type(baseline.exception)
    assert str(actual.exception) == str(baseline.exception)
    assert model.n_features_in_ is metadata


@pytest.mark.parametrize("policy", ["ignore", "warn", "raise", "warning_error"])
def test_half_feature_metadata_reuses_numpy_overflow_policy(policy):
    # One bounded metadata shape at the half-precision overflow edge. This
    # checks the scalar event, not numerical accuracy or performance.
    width = 65520
    metadata = np.float16(np.inf)
    model, x = _model(width)
    model.n_features_in_ = metadata
    mode = "warn" if policy == "warning_error" else policy
    reference_error = None
    with warnings.catch_warnings(record=True) as reference_warnings, np.errstate(over=mode):
        warnings.simplefilter("error" if policy == "warning_error" else "always", RuntimeWarning)
        try:
            if width != metadata:
                raise ValueError("mojolearn LogisticRegression feature count differs from fit")
        except Exception as exc:
            reference_error = exc
    with warnings.catch_warnings(record=True) as native_warnings, np.errstate(over=mode):
        warnings.simplefilter("error" if policy == "warning_error" else "always", RuntimeWarning)
        audit = audit_native_call(model.decision_function, x)
    _clean(audit)
    assert [(type(w.message), str(w.message)) for w in native_warnings] == [
        (type(w.message), str(w.message)) for w in reference_warnings]
    if reference_error is None:
        np.testing.assert_array_equal(np.asarray(audit.assert_native()), [-1., 1.])
    else:
        assert type(audit.exception) is type(reference_error), audit.report()
        assert str(audit.exception) == str(reference_error)
    assert model.n_features_in_ is metadata


def test_numpy_feature_overflow_callback_is_refused_without_invocation():
    calls = []

    def callback(*args):
        calls.append(args)

    model, x = _model(65520)
    model.n_features_in_ = np.float16(np.inf)
    previous = np.seterrcall(callback)
    try:
        with np.errstate(over="call"):
            audit = audit_native_call(model.decision_function, x)
    finally:
        np.seterrcall(previous)
    _clean(audit)
    # On legacy NumPy promotion no half cast occurs and the ordinary width
    # mismatch wins; under weak promotion the error callback is refused.
    if type(audit.exception) is TypeError:
        assert "Python NumPy error callbacks" in str(audit.exception)
    else:
        assert type(audit.exception) is ValueError, audit.report()
        assert "feature count differs from fit" in str(audit.exception)
    assert not calls


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float32", "float64", "int8", "uint8"])
def test_exact_torch_feature_scalar_is_unpacked_without_tensor_comparison(dtype_name):
    torch = pytest.importorskip("torch")
    metadata = torch.tensor([2], dtype=getattr(torch, dtype_name))
    model, x = _model()
    model.n_features_in_ = metadata
    audit = audit_native_call(model.decision_function, x)
    _clean(audit)
    np.testing.assert_array_equal(np.asarray(audit.assert_native()), [-1., 1.])
    assert model.n_features_in_ is metadata


@pytest.mark.parametrize("depth", [3, 66])
def test_object_metadata_unboxing_is_bounded_and_keeps_owners(depth):
    metadata = 2
    for _ in range(depth):
        outer = np.empty((), dtype=object)
        outer[()] = metadata
        metadata = outer
    model, x = _model()
    model.n_features_in_ = metadata
    audit = audit_native_call(model.decision_function, x)
    _clean(audit)
    if depth == 3:
        np.testing.assert_array_equal(np.asarray(audit.assert_native()), [-1., 1.])
    else:
        assert type(audit.exception) is TypeError, audit.report()
        assert "native nesting bounds" in str(audit.exception)
    assert model.n_features_in_ is metadata
