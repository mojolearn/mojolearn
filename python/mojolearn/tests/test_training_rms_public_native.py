"""RMS preparation, scalar options, results and refusals stay native."""
import warnings

import numpy as np
import pytest

from mojolearn import _backend, _training_impl as T
from mojolearn._array import Array
from mojolearn._verify_native_runtime import audit_native_call


def inputs(case="matrix"):
    shape = (2, 2, 3) if case == "batch" else ((3,) if case == "vector" else (4, 3))
    x = (np.arange(np.prod(shape), dtype="f4").reshape(shape) - 3) / 8
    weight = np.array([0.5, 1.25, -0.75], dtype="f4")
    gradient = (x + 0.125).copy()
    if case in ("strided", "array"):
        x, gradient = [np.asfortranarray(value) for value in (x, gradient)]
    elif case == "negative-stride":
        x, gradient, weight = x[..., ::-1], gradient[..., ::-1], weight[::-1]
    elif case == "readonly":
        for value in (x, weight, gradient):
            value.flags.writeable = False
    if case == "array":
        x, weight, gradient = [Array.from_buffer(value) for value in (x, weight, gradient)]
    elif case in ("numpy-0d", "numpy-scalar"):
        x, gradient = np.array(1.25, dtype="f4"), np.array(0.5, dtype="f4")
        weight = np.ones(1, dtype="f4")
        if case == "numpy-scalar":
            x, gradient = x[()], gradient[()]
    return x, weight, gradient


def kernel_oracle(operation, x, weight, gradient, epsilon):
    # Controller preparation and the unchanged raw compiled entry give a
    # before/after layout/routing reference, not an independent arithmetic oracle.
    x, weight, gradient = [np.ascontiguousarray(v, dtype="f4") for v in (x, weight, gradient)]
    width, rows = x.shape[-1], x.size // x.shape[-1]
    binding = _backend.binding("_mojolearn_training", "identical")
    result = np.empty(x.shape, dtype="f4")
    if operation == "rms_norm_forward":
        binding.rms_norm_forward([result.ctypes.data, x.ctypes.data, weight.ctypes.data], [rows, width, epsilon])
        return (result,)
    dw = np.empty(width, dtype="f4")
    binding.rms_norm_backward([result.ctypes.data, dw.ctypes.data, gradient.ctypes.data, x.ctypes.data, weight.ctypes.data], [rows, width, epsilon])
    return result, dw


def arguments(operation, x, weight, gradient, epsilon):
    return (x, weight, epsilon) if operation == "rms_norm_forward" else (gradient, x, weight, epsilon)


def assert_outputs(actual, expected):
    outputs = actual if type(actual) is tuple else (actual,)
    assert len(outputs) == len(expected)
    for output, reference in zip(outputs, expected):
        assert type(output) is Array
        assert output.shape == reference.shape
        assert np.asarray(output).tobytes() == reference.tobytes()


def assert_refused(result, category):
    assert type(result.exception) is category, result.report()
    for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads"):
        assert not result.report()[key], result.report()


@pytest.mark.parametrize("operation", ["rms_norm_forward", "rms_norm_backward"])
@pytest.mark.parametrize("case", ["matrix", "batch", "vector", "strided", "negative-stride", "readonly", "array", "numpy-0d", "numpy-scalar"])
def test_public_rms_preparation_and_output_match_retained_kernel(operation, case):
    x, weight, gradient = inputs(case)
    expected = kernel_oracle(operation, x, weight, gradient, 1e-5)
    result = audit_native_call(getattr(T, operation), *arguments(operation, x, weight, gradient, 1e-5), numeric_mode="identical").assert_native()
    assert_outputs(result, expected)


EPSILON_CASES = ["float", "int", "bool", "str", "bytes", "bytearray", "memoryview", "f16", "f32", "f64", "longdouble", "int64", "longlong", "complex64", "complex128", "clongdouble", "array-f32", "array-int", "array-longlong", "array-longdouble", "array-object", "array-str", "array-bytes", "array-complex", "array-datetime-ns", "array-datetime-D", "array-timedelta-ns", "array-timedelta-D", "array-1d"]
if hasattr(np.dtypes, "StringDType"):
    EPSILON_CASES.append("array-string")


def epsilon_value(case):
    if case == "array-string":
        return np.array("1e-5", dtype=np.dtypes.StringDType())
    return {"float": 1e-5, "int": 0, "bool": False, "str": "1e-5", "bytes": b"1e-5",
            "bytearray": bytearray(b"1e-5"), "memoryview": memoryview(b"1e-5"),
            "f16": np.float16(1e-5), "f32": np.float32(1e-5), "f64": np.float64(1e-5),
            "longdouble": np.longdouble("1e-5"), "int64": np.int64(0), "longlong": np.longlong(0),
            "complex64": np.complex64(1e-5 + 0.25j), "complex128": np.complex128(1e-5 + 0.25j),
            "clongdouble": np.clongdouble(1e-5 + 0.25j), "array-f32": np.array(1e-5, dtype="f4"),
            "array-int": np.array(0, dtype="i8"), "array-object": np.array(1e-5, dtype=object),
            "array-longlong": np.array(0, dtype="q"), "array-longdouble": np.array(1e-5, dtype="g"),
            "array-complex": np.array(1e-5 + 0.25j, dtype="c8"),
            "array-datetime-ns": np.array(1, dtype="datetime64[ns]"), "array-datetime-D": np.array(1, dtype="datetime64[D]"),
            "array-timedelta-ns": np.array(1, dtype="timedelta64[ns]"), "array-timedelta-D": np.array(1, dtype="timedelta64[D]"),
            "array-str": np.array("1e-5"), "array-bytes": np.array(b"1e-5"),
            "array-1d": np.array([1e-5], dtype="f4")}[case]


@pytest.mark.parametrize("operation", ["rms_norm_forward", "rms_norm_backward"])
@pytest.mark.parametrize("case", EPSILON_CASES)
def test_public_rms_native_scalar_options_preserve_values_warnings_and_errors(operation, case):
    x, weight, gradient = inputs()
    epsilon = epsilon_value(case)
    expected_error = None
    with warnings.catch_warnings(record=True) as old:
        warnings.simplefilter("always")
        try:
            scalar = float(epsilon)
        except Exception as error:
            expected_error = type(error)
    expected = None if expected_error else kernel_oracle(operation, x, weight, gradient, scalar)
    with warnings.catch_warnings(record=True) as new:
        warnings.simplefilter("always")
        result = audit_native_call(getattr(T, operation), *arguments(operation, x, weight, gradient, epsilon), numeric_mode="identical")
    if expected_error:
        assert_refused(result, expected_error)
    else:
        assert_outputs(result.assert_native(), expected)
    assert [(type(w.message), str(w.message)) for w in new] == [(type(w.message), str(w.message)) for w in old]


@pytest.mark.parametrize("operation", ["rms_norm_forward", "rms_norm_backward"])
@pytest.mark.parametrize("case", ["float64", "big-endian", "empty", "zero-feature", "array-0d", "bad-eps", "negative-eps", "none-eps"])
def test_public_rms_preserves_retained_refusal_categories(operation, case):
    x, weight, gradient = inputs()
    epsilon, category = 1e-5, ValueError
    if case == "float64":
        x, category = x.astype("f8"), TypeError
    elif case == "big-endian":
        x, category = x.astype(">f4"), TypeError
    elif case == "empty":
        x, gradient = x[:0], gradient[:0]
    elif case == "zero-feature":
        x, gradient, weight = x[:, :0], gradient[:, :0], weight[:0]
    elif case == "array-0d":
        x, gradient = [Array.from_buffer(np.array(1., dtype="f4")) for _ in range(2)]
        weight, category = np.ones(1, dtype="f4"), IndexError
    elif case == "bad-eps":
        epsilon = "bad"
    elif case == "negative-eps":
        epsilon, category = -1., Exception
    elif case == "none-eps":
        epsilon, category = None, TypeError
    assert_refused(audit_native_call(getattr(T, operation), *arguments(operation, x, weight, gradient, epsilon), numeric_mode="identical"), category)


@pytest.mark.parametrize("shape", [(1, 3), (5,), (1, 1, 3)])
def test_public_rms_backward_preserves_safe_weight_prefix(shape):
    x, _, gradient = inputs()
    weight = np.arange(1, np.prod(shape) + 1, dtype="f4").reshape(shape)
    expected = kernel_oracle("rms_norm_backward", x, weight, gradient, 1e-5)
    result = audit_native_call(T.rms_norm_backward, gradient, x, weight, 1e-5, numeric_mode="identical").assert_native()
    assert_outputs(result, expected)


@pytest.mark.parametrize("case", ["forward-weight", "backward-short-weight", "backward-gradient"])
def test_public_rms_native_shape_guards(case):
    x, weight, gradient = inputs()
    operation = "rms_norm_backward"
    if case == "forward-weight":
        operation, weight = "rms_norm_forward", weight.reshape(1, 3)
    elif case == "backward-short-weight":
        weight = weight[:1]
    else:
        gradient = gradient.T
    assert_refused(audit_native_call(getattr(T, operation), *arguments(operation, x, weight, gradient, 1e-5), numeric_mode="identical"), ValueError)


@pytest.mark.parametrize("slot", ["input", "weight", "gradient", "epsilon", "object-element"] + (["string-missing"] if hasattr(np.dtypes, "StringDType") else []))
def test_public_rms_refuses_python_protocols_before_hooks(slot):
    calls = []
    class Foreign:
        def __float__(self):
            calls.append("float")
            return 1e-5
        def __array__(self, *args, **kwargs):
            calls.append("array")
        def __buffer__(self, *args):
            calls.append("buffer")
    x, weight, gradient = inputs()
    epsilon = 1e-5
    if slot == "input":
        x = Foreign()
    elif slot == "weight":
        weight = Foreign()
    elif slot == "gradient":
        gradient = Foreign()
    elif slot == "epsilon":
        epsilon = Foreign()
    elif slot == "object-element":
        epsilon = np.empty((), dtype=object)
        epsilon[()] = Foreign()
    else:
        sentinel = Foreign()
        epsilon = np.empty((), dtype=np.dtypes.StringDType(na_object=sentinel))
        epsilon[()] = sentinel
    calls.clear()
    assert_refused(audit_native_call(T.rms_norm_backward, gradient, x, weight, epsilon, numeric_mode="identical"), TypeError)
    assert not calls


@pytest.mark.parametrize("action", ["ignore", "error"])
def test_public_rms_complex_scalar_warning_filters_are_native(action):
    x, weight, gradient = inputs()
    epsilon = np.complex64(1e-5 + 0.5j)
    with warnings.catch_warnings():
        warnings.simplefilter(action, np.exceptions.ComplexWarning)
        result = audit_native_call(T.rms_norm_forward, x, weight, epsilon, numeric_mode="identical")
    if action == "error":
        assert_refused(result, np.exceptions.ComplexWarning)
    else:
        expected = kernel_oracle("rms_norm_forward", x, weight, gradient, float(epsilon.real))
        assert_outputs(result.assert_native(), expected)


def test_public_rms_warning_hook_is_refused_without_execution(monkeypatch):
    calls = []
    def hook(*args, **kwargs):
        calls.append("warning")
    x, weight, _ = inputs()
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        monkeypatch.setattr(warnings, "showwarning", hook)
        result = audit_native_call(T.rms_norm_forward, x, weight, np.complex64(1e-5), numeric_mode="identical")
    assert_refused(result, TypeError)
    assert not calls


def test_public_rms_warning_category_hook_is_refused_without_execution(monkeypatch):
    calls = []
    def hook(*args, **kwargs):
        calls.append("warning-category")
        raise AssertionError("Python warning constructor called")
    x, weight, _ = inputs()
    with monkeypatch.context() as patch, warnings.catch_warnings():
        warnings.simplefilter("always")
        patch.setattr(np.exceptions.ComplexWarning, "__new__", staticmethod(hook))
        result = audit_native_call(T.rms_norm_forward, x, weight, np.complex64(1e-5), numeric_mode="identical")
    assert_refused(result, TypeError)
    assert not calls


def test_public_rms_uses_no_python_preparation_or_loader(monkeypatch):
    x, weight, gradient = inputs("strided")
    expected = kernel_oracle("rms_norm_backward", x, weight, gradient, 1e-5)
    def forbidden(*args, **kwargs):
        raise AssertionError("Python RMS helper entered")
    monkeypatch.setattr(T, "_load", forbidden)
    monkeypatch.setattr(T, "_c32", forbidden)
    result = audit_native_call(T.rms_norm_backward, gradient, x, weight, 1e-5, numeric_mode="identical").assert_native()
    assert_outputs(result, expected)


@pytest.mark.parametrize("operation", ["rms_norm_forward", "rms_norm_backward"])
def test_public_rms_refuses_replaced_native_entry(monkeypatch, operation):
    x, weight, gradient = inputs()
    calls = []
    def callback(*args):
        calls.append("entry")
    monkeypatch.setitem(T._OPT_CONTEXT["bindings"]["identical"], operation, callback)
    assert_refused(audit_native_call(getattr(T, operation), *arguments(operation, x, weight, gradient, 1e-5), numeric_mode="identical"), TypeError)
    assert not calls


def test_existing_rms_primitive_contract():
    from mojolearn.tests import test_training_primitives_surface as existing
    report = existing.Report("native_rms_existing_primitive")
    existing.arm_rms_norm(report)
    assert report.rows and not report.failures, report.failures
