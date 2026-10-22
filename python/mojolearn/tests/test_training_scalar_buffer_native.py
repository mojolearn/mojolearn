"""Public training options consume primitive scalar bytes without NumPy casts."""
import warnings

import numpy as np
import pytest

from mojolearn import _backend, _training_impl as T
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests.test_training_rms_public_native import (
    inputs, arguments, kernel_oracle, assert_outputs, assert_refused,
)
from mojolearn.tests.test_training_embedding_public_native import (
    reference as embedding_reference, assert_output as embedding_output,
)
from mojolearn.tests.test_training_cross_entropy_public_native import (
    reference as loss_reference, compare_output,
)


def _warnings(caught):
    return [(type(item.message), str(item.message)) for item in caught]


def _number(code, number, form):
    value = np.array(number, dtype=code)
    if form == "scalar":
        return value[()]
    if form == "swapped":
        return value.astype(value.dtype.newbyteorder(">"))
    if form == "singleton":
        # A readonly singleton with a nonunit/negative stride exercises the
        # public raw buffer copy while retaining NumPy's versioned rank policy.
        value = np.array([number, 0, 0], dtype=code)[:: -2][1:]
        value.flags.writeable = False
    return value


FLOAT_CASES = [
    ("?", False, "scalar"), ("i1", 0, "scalar"), ("u2", 1, "scalar"),
    ("q", 0, "scalar"), ("Q", 1, "array"),
    ("f2", .125, "scalar"), ("f4", .125, "array"),
    ("f8", .125, "swapped"), ("g", .125, "scalar"),
    ("g", .125, "swapped"), ("f4", -0., "scalar"),
    ("f8", -0., "array"), ("f4", np.nextafter(np.float32(0), np.float32(1)), "array"),
    ("f4", .125, "singleton"), ("g", .125, "singleton"),
    ("c8", .125 + 1j, "scalar"), ("c16", .125 + 1j, "scalar"),
    ("G", .125 + 1j, "scalar"), ("c8", .125 + 1j, "array"),
]


@pytest.mark.parametrize("operation", ["rms_norm_forward", "rms_norm_backward"])
@pytest.mark.parametrize("code,number,form", FLOAT_CASES)
def test_rms_consumes_native_scalar_and_endian_options(operation, code, number, form):
    x, weight, gradient = inputs()
    value = _number(code, number, form)
    before = value.tobytes()
    failure = None
    with warnings.catch_warnings(record=True) as old:
        warnings.simplefilter("always")
        try:
            scalar = float(value)
        except Exception as error:
            failure = (type(error), str(error))
    expected = None if failure else kernel_oracle(operation, x, weight, gradient, scalar)
    with warnings.catch_warnings(record=True) as new:
        warnings.simplefilter("always")
        result = audit_native_call(getattr(T, operation),
                                   *arguments(operation, x, weight, gradient, value),
                                   numeric_mode="identical")
    if failure:
        assert_refused(result, failure[0])
        assert str(result.exception) == failure[1]
    else:
        assert_outputs(result.assert_native(), expected)
    assert _warnings(new) == _warnings(old)
    assert value.tobytes() == before


INTEGER_CASES = [
    ("?", True, "scalar"), ("i1", 4, "scalar"), ("u2", 4, "array"),
    ("i8", 4, "swapped"), ("q", 4, "scalar"), ("Q", 4, "scalar"),
    ("f2", 4.75, "scalar"), ("f4", 4.75, "array"),
    ("f8", 4.75, "swapped"), ("g", 4.75, "scalar"),
    ("g", 4.75, "swapped"), ("f4", 4.75, "singleton"),
    ("c8", 4.75 + 1j, "scalar"), ("G", 4.75 + 1j, "scalar"),
    ("g", np.nextafter(np.longdouble(4), np.longdouble(0)), "scalar"),
    ("g", np.nextafter(np.longdouble(4), np.longdouble(0)), "swapped"),
]


@pytest.mark.parametrize("code,number,form", INTEGER_CASES)
def test_embedding_consumes_native_integer_options_and_repeated_warnings(code, number, form):
    value = _number(code, number, form)
    before = value.tobytes()
    source, indices = np.array([[.25, -.75, 1.5]], dtype="f4"), [0]
    failure = None
    with warnings.catch_warnings(record=True) as old:
        warnings.simplefilter("always")
        try:
            count = int(value)
            # Both native preparation sites retain their previous warnings.
            count = int(value)
        except Exception as error:
            failure = (type(error), str(error))
    expected = None if failure else embedding_reference("embedding_backward", source, indices, count)
    with warnings.catch_warnings(record=True) as new:
        warnings.simplefilter("always")
        result = audit_native_call(T.embedding_backward, source, indices, value,
                                   numeric_mode="identical")
    if failure:
        assert_refused(result, failure[0])
        assert str(result.exception) == failure[1]
    else:
        embedding_output(result, expected)
    assert _warnings(new) == _warnings(old)
    assert value.tobytes() == before


@pytest.mark.parametrize("operation", ["rms_norm_forward", "rms_norm_backward", "embedding"])
@pytest.mark.parametrize("form", ["array", "swapped", "singleton"])
def test_extended_complex_array_options_preserve_consumed_values_and_warning_order(operation, form):
    if operation == "embedding":
        test_embedding_consumes_native_integer_options_and_repeated_warnings("G", 4.75 + 1j, form)
    else:
        test_rms_consumes_native_scalar_and_endian_options(operation, "G", .125 + 1j, form)


@pytest.mark.parametrize("option", ["label_smoothing", "ignore_index", "num_items"])
@pytest.mark.parametrize("code,form", [("f4", "scalar"), ("g", "swapped"),
                                        ("f8", "singleton"), ("G", "scalar")])
def test_loss_consumes_each_native_scalar_option(option, code, form):
    number = .125 if option == "label_smoothing" else -100 if option == "ignore_index" else 3.75
    value = _number(code, number, form)
    before = value.tobytes()
    x = np.array([[.2, .7, -.3], [.9, -.2, .5]], dtype="f4")
    y = np.array([2, 0], dtype="i4")
    conversion = float if option == "label_smoothing" else int
    failure = None
    with warnings.catch_warnings(record=True) as old:
        warnings.simplefilter("always")
        try:
            scalar = conversion(value)
        except Exception as error:
            failure = (type(error), str(error))
    options = dict(return_grad=True, reduction="sum")
    expected = None if failure else loss_reference(x, y, {**options, option: scalar})
    with warnings.catch_warnings(record=True) as new:
        warnings.simplefilter("always")
        result = audit_native_call(T.cross_entropy, x, y, **options,
                                   **{option: value}, numeric_mode="identical")
    if failure:
        assert_refused(result, failure[0])
        assert str(result.exception) == failure[1]
    else:
        compare_output(result.assert_native(), expected)
    assert _warnings(new) == _warnings(old)
    assert value.tobytes() == before


@pytest.mark.parametrize("operation", ["embedding", "loss"])
@pytest.mark.parametrize("code,number,form", [
    ("f4", float("nan"), "scalar"), ("f8", float("inf"), "array"),
    ("g", float("nan"), "scalar"), ("g", -float("inf"), "swapped"),
    ("c8", complex(float("nan"), 1), "scalar"),
    ("G", complex(float("inf"), 1), "scalar"),
])
def test_integer_nonfinite_errors_preserve_text_and_warning_order(operation, code, number, form):
    value = _number(code, number, form)
    before = value.tobytes()
    with warnings.catch_warnings(record=True) as old:
        warnings.simplefilter("always")
        try:
            int(value)
        except Exception as error:
            failure = (type(error), str(error))
        else:
            raise AssertionError("fixture must fail integer conversion")
    with warnings.catch_warnings(record=True) as new:
        warnings.simplefilter("always")
        if operation == "embedding":
            result = audit_native_call(T.embedding_backward, np.ones((1, 2), dtype="f4"),
                                       [0], value, numeric_mode="identical")
        else:
            result = audit_native_call(T.cross_entropy, np.ones((1, 2), dtype="f4"),
                                       [0], num_items=value, numeric_mode="identical")
    assert_refused(result, failure[0])
    assert str(result.exception) == failure[1]
    assert _warnings(new) == _warnings(old)
    assert value.tobytes() == before


@pytest.mark.parametrize("operation", ["embedding", "loss"])
@pytest.mark.parametrize("number", [complex(float("nan"), 1), complex(float("inf"), 1), complex(-float("inf"), 1)])
def test_extended_complex_array_integer_failure_preserves_warning_order(operation, number):
    test_integer_nonfinite_errors_preserve_text_and_warning_order(operation, "G", number, "swapped")


@pytest.mark.parametrize("code,number,form", [
    ("u8", 2**64 - 1, "scalar"), ("Q", 2**64 - 1, "array"),
    ("i8", -(2**63), "swapped"), ("f8", 2**100, "scalar"),
    ("g", -(2**100), "swapped"),
])
def test_loss_integer_range_preserves_downstream_error_or_consumed_result(code, number, form):
    # Reference conversion is outside the product boundary. The unchanged
    # public builtin-int route then establishes downstream width/refusal rules.
    value = _number(code, number, form)
    before = value.tobytes()
    integer = int(value)
    x, y = np.array([[.2, -.3]], dtype="f4"), [0]
    expected = audit_native_call(T.cross_entropy, x, y, ignore_index=integer,
                                 numeric_mode="identical")
    actual = audit_native_call(T.cross_entropy, x, y, ignore_index=value,
                               numeric_mode="identical")
    if expected.exception is not None:
        assert_refused(expected, type(expected.exception))
        assert_refused(actual, type(expected.exception))
        assert str(actual.exception) == str(expected.exception)
    else:
        compare_output(actual.assert_native(), expected.assert_native())
    assert value.tobytes() == before


@pytest.mark.parametrize("dtype", ["f4", "i8"])
@pytest.mark.parametrize("form", ["scalar-subclass", "array-subclass", "object-element"])
def test_training_scalar_admission_precedes_buffer_or_conversion_hooks(dtype, form):
    calls = []
    base = np.dtype(dtype).type if form == "scalar-subclass" else np.ndarray
    class Foreign(base):
        def __float__(self):
            calls.append("float")
            return .125
        def __int__(self):
            calls.append("int")
            return 1
        @property
        def dtype(self):
            calls.append("dtype")
            return np.dtype(dtype)
    value = Foreign(.125) if form == "scalar-subclass" else np.array(.125, dtype=dtype).view(Foreign)
    if form == "object-element":
        wrapped = np.empty((), dtype=object)
        wrapped[()] = value
        value = wrapped
    x, weight, _ = inputs()
    calls.clear()
    result = audit_native_call(T.rms_norm_forward, x, weight, value, numeric_mode="identical")
    assert_refused(result, TypeError)
    assert not calls


@pytest.mark.parametrize("option,code,number,form", [
    ("label_smoothing", "g", .125, "swapped"),
    ("ignore_index", "c8", -100 + 1j, "scalar"),
    ("num_items", "g", 3.75, "swapped"),
    ("num_items", "g", float("nan"), "scalar"),
])
def test_device_loss_scalar_conversion_and_failure_remain_native(option, code, number, form):
    if _backend.vendor() == "cpu":
        pytest.skip("training device pool requires GPU")
    x = np.array([[.2, .7, -.3], [.9, -.2, .5]], dtype="f4")
    y = np.array([2, 0], dtype="i4")
    value = _number(code, number, form)
    before = value.tobytes()
    failure = None
    with warnings.catch_warnings(record=True) as old:
        warnings.simplefilter("always")
        try:
            scalar = (float if option == "label_smoothing" else int)(value)
        except Exception as error:
            failure = (type(error), str(error))
    options = dict(return_grad=True, reduction="sum")
    expected = None if failure else loss_reference(x, y, {**options, option: scalar})
    device = audit_native_call(T.train_to_device, x, numeric_mode="identical").assert_native()
    try:
        with warnings.catch_warnings(record=True) as new:
            warnings.simplefilter("always")
            result = audit_native_call(T.cross_entropy, device, y, **options,
                                       **{option: value}, numeric_mode="identical")
        if failure:
            assert_refused(result, failure[0])
            assert str(result.exception) == failure[1]
        else:
            loss, gradient = result.assert_native()
            try:
                host = audit_native_call(gradient.to_host).assert_native()
                compare_output((loss, host), expected)
            finally:
                audit_native_call(T.TrainDeviceArray.__del__, gradient).assert_native()
        assert _warnings(new) == _warnings(old)
        assert value.tobytes() == before
    finally:
        audit_native_call(T.TrainDeviceArray.__del__, device).assert_native()
