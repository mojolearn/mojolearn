"""Public linear-layer preparation, execution, outputs and rollback stay native."""
import numpy as np
import pytest

from mojolearn import _backend, _training_impl as T
from mojolearn._array import Array
from mojolearn._verify_native_runtime import audit_native_call


def inputs(case):
    shape = (2, 2, 3) if case == "batch" else ((3,) if case == "vector" else (4, 3))
    a = (np.arange(np.prod(shape), dtype="f4").reshape(shape) - 3) / 8
    weight = (np.arange(6, dtype="f4").reshape(2, 3) - 2) / 7
    gradient = np.arange(np.prod(shape[:-1]) * 2, dtype="f4").reshape(shape[:-1] + (2,)) / 5
    if case in ("strided", "array"):
        a, weight, gradient = [np.asfortranarray(x) for x in (a, weight, gradient)]
    if case == "negative-stride":
        a, weight, gradient = a[..., ::-1], weight[:, ::-1], gradient[..., ::-1]
    if case == "readonly":
        for x in (a, weight, gradient):
            x.flags.writeable = False
    if case == "array":
        a, weight, gradient = [Array.from_buffer(x) for x in (a, weight, gradient)]
    return a, weight, gradient


def kernel_oracle(operation, a, weight, gradient, numeric_mode):
    # Test-controller preparation deliberately sits outside the audited
    # product operation. The unchanged compiled entry is a before/after
    # reference for routing/layout, not an independent arithmetic algorithm.
    a, weight, gradient = [np.ascontiguousarray(x, dtype="f4") for x in (a, weight, gradient)]
    k, n = a.shape[-1], weight.shape[0]
    m = a.size // k
    binding = _backend.binding("_mojolearn_training", numeric_mode)
    if operation == "linear_forward":
        result = np.empty(a.shape[:-1] + (n,), dtype="f4")
        binding.linear_forward([result.ctypes.data, a.ctypes.data, weight.ctypes.data], [m, n, k])
        return (result,)
    da, dw = np.empty(a.shape, dtype="f4"), np.empty(weight.shape, dtype="f4")
    binding.linear_backward([da.ctypes.data, dw.ctypes.data, gradient.ctypes.data, a.ctypes.data, weight.ctypes.data], [m, n, k])
    return da, dw


def assert_outputs(actual, expected, device=False):
    outputs = actual if type(actual) is tuple else (actual,)
    assert len(outputs) == len(expected)
    for output, reference in zip(outputs, expected):
        if device:
            output = audit_native_call(output.to_host).assert_native()
        assert type(output) is Array
        assert output.shape == reference.shape
        assert np.asarray(output).tobytes() == reference.tobytes()


@pytest.mark.parametrize("case", ["matrix", "batch", "vector", "strided", "negative-stride", "readonly", "array"])
@pytest.mark.parametrize("operation", ["linear_forward", "linear_backward"])
def test_public_linear_native_preparation_matches_existing_kernel(case, operation):
    a, weight, gradient = inputs(case)
    expected = kernel_oracle(operation, a, weight, gradient, "identical")
    args = (a, weight) if operation == "linear_forward" else (gradient, a, weight)
    actual = audit_native_call(getattr(T, operation), *args, numeric_mode="identical").assert_native()
    assert_outputs(actual, expected)


@pytest.mark.parametrize("scalar", ["numpy-0d", "numpy-scalar"])
@pytest.mark.parametrize("operation", ["linear_forward", "linear_backward"])
def test_public_linear_preserves_scalar_buffer_shape(scalar, operation):
    a = np.array(1.25, dtype="f4") if scalar == "numpy-0d" else np.float32(1.25)
    weight, gradient = np.array([[2.], [-3.]], dtype="f4"), np.array([0.5, -0.25], dtype="f4")
    expected = kernel_oracle(operation, a, weight, gradient, "identical")
    args = (a, weight) if operation == "linear_forward" else (gradient, a, weight)
    result = audit_native_call(getattr(T, operation), *args, numeric_mode="identical").assert_native()
    assert_outputs(result, expected)


def assert_refused(result, expected):
    assert isinstance(result.exception, expected), result.report()
    for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads"):
        assert not result.report()[key], result.report()


@pytest.mark.parametrize("operation", ["linear_forward", "linear_backward"])
def test_public_linear_preserves_zero_rank_array_refusal(operation):
    a = Array.from_buffer(np.array(1.25, dtype="f4"))
    weight, gradient = np.ones((2, 1), dtype="f4"), np.ones(2, dtype="f4")
    args = (a, weight) if operation == "linear_forward" else (gradient, a, weight)
    assert_refused(audit_native_call(getattr(T, operation), *args, numeric_mode="identical"), IndexError)


@pytest.mark.parametrize("operation,slot", [("linear_forward", "input"), ("linear_forward", "weight"),
                                           ("linear_backward", "input"), ("linear_backward", "weight"),
                                           ("linear_backward", "gradient")])
@pytest.mark.parametrize("scalar", ["numpy-0d", "numpy-scalar", "array-0d"])
def test_public_linear_preserves_each_scalar_slot_contract(operation, slot, scalar):
    # The retained old-wrapper probe uses K=1, so its permissive backward
    # weight handling never reads beyond a scalar buffer.
    a, weight, gradient = np.array([[1.25]], dtype="f4"), np.array([[2.]], dtype="f4"), np.array([[0.5]], dtype="f4")
    value = np.array({"input": 1.25, "weight": 2., "gradient": 0.5}[slot], dtype="f4")
    if scalar == "numpy-scalar":
        value = value[()]
    elif scalar == "array-0d":
        value = Array.from_buffer(value)
    if slot == "input":
        a = value
    elif slot == "weight":
        weight = value
    else:
        gradient = value
    args = (a, weight) if operation == "linear_forward" else (gradient, a, weight)
    result = audit_native_call(getattr(T, operation), *args, numeric_mode="identical")
    if operation == "linear_forward" and slot == "weight":
        assert_refused(result, ValueError)
    elif scalar == "array-0d":
        assert_refused(result, IndexError)
    else:
        shape = (1,) if slot == "input" else (1, 1)
        expected = (np.full(shape, 2.5, dtype="f4"),) if operation == "linear_forward" else (
            np.full(shape, 1., dtype="f4"), np.full((1, 1), 0.625, dtype="f4"))
        assert_outputs(result.assert_native(), expected)


@pytest.mark.parametrize("shape", [(2,), (2, 3), (2, 1, 2)])
@pytest.mark.parametrize("kind", ["numpy", "array"])
def test_public_linear_backward_preserves_safe_weight_prefix(shape, kind):
    a = np.array([[1.25], [-0.5]], dtype="f4")
    gradient = np.array([[0.5, 0.25], [-0.25, 1.]], dtype="f4")
    weight = np.arange(1, np.prod(shape) + 1, dtype="f4").reshape(shape)
    expected = kernel_oracle("linear_backward", a, weight.ravel()[:2].reshape(2, 1), gradient, "identical")
    if kind == "array":
        weight = Array.from_buffer(weight)
    result = audit_native_call(T.linear_backward, gradient, a, weight, numeric_mode="identical").assert_native()
    assert_outputs(result, expected)


@pytest.mark.parametrize("case", ["float64", "big-endian", "empty", "zero-feature", "weight-rank", "weight-features"])
@pytest.mark.parametrize("operation", ["linear_forward", "linear_backward"])
def test_public_linear_refuses_invalid_buffers_before_kernel(case, operation):
    a, weight, gradient = inputs("matrix")
    error = ValueError
    if case == "float64":
        a, error = a.astype("f8"), TypeError
    elif case == "big-endian":
        a, error = a.astype(">f4"), TypeError
    elif case == "empty":
        a = a[:0]
    elif case == "zero-feature":
        a, weight = a[:, :0], weight[:, :0]
    elif case == "weight-rank":
        weight = weight.ravel()
    elif case == "weight-features":
        weight = weight[:, :2]
    args = (a, weight) if operation == "linear_forward" else (gradient, a, weight)
    assert_refused(audit_native_call(getattr(T, operation), *args, numeric_mode="identical"), error)


@pytest.mark.parametrize("shape", [(4, 3), (3, 2), (2, 4)])
def test_public_linear_backward_refuses_incompatible_gradient(shape):
    a, weight, _ = inputs("matrix")
    assert_refused(audit_native_call(T.linear_backward, np.ones(shape, dtype="f4"), a, weight, numeric_mode="identical"), ValueError)


@pytest.mark.parametrize("slot", ["input", "weight", "gradient"])
def test_public_linear_refuses_python_input_protocols_before_hooks(slot):
    calls = []
    class Foreign:
        def __getattribute__(self, name):
            calls.append(name)
            raise AssertionError("Python input hook entered")
        def __array__(self, *args, **kwargs):
            calls.append("array")
        def __buffer__(self, *args):
            calls.append("buffer")
    a, weight, gradient = inputs("matrix")
    if slot == "input":
        a = Foreign()
    elif slot == "weight":
        weight = Foreign()
    else:
        gradient = Foreign()
    assert_refused(audit_native_call(T.linear_backward, gradient, a, weight, numeric_mode="identical"), TypeError)
    assert not calls


def test_public_linear_uses_no_python_preparation_or_binding_helpers(monkeypatch):
    a, weight, gradient = inputs("strided")
    expected = kernel_oracle("linear_backward", a, weight, gradient, "identical")
    def forbidden(*args, **kwargs):
        raise AssertionError("Python linear preparation helper entered")
    for name in ("_load", "_c32", "_linear_forward_dev", "_linear_backward_dev"):
        monkeypatch.setattr(T, name, forbidden)
    actual = audit_native_call(T.linear_backward, gradient, a, weight, numeric_mode="identical").assert_native()
    assert_outputs(actual, expected)


@pytest.mark.parametrize("operation", ["linear_forward", "linear_backward"])
def test_public_linear_refuses_replaced_native_entry(monkeypatch, operation):
    a, weight, gradient = inputs("matrix")
    calls = []
    def callback(*args):
        calls.append("entry")
    entries = T._OPT_CONTEXT["bindings"]["identical"]
    monkeypatch.setitem(entries, operation, callback)
    args = (a, weight) if operation == "linear_forward" else (gradient, a, weight)
    assert_refused(audit_native_call(getattr(T, operation), *args, numeric_mode="identical"), TypeError)
    assert not calls


def test_existing_linear_primitive_exact_integer_contract():
    from mojolearn.tests import test_training_primitives_surface as existing
    report = existing.Report("native_linear_existing_primitive")
    existing.arm_linear(report)
    assert report.rows and not report.failures, report.failures


def test_existing_linear_backward_aligned_accumulation_contract():
    from mojolearn.tests import test_samba_surface as existing
    report = existing.Report()
    existing.arm_accumulation(report)
    assert report.rows and not report.failures, report.failures


@pytest.fixture
def device_binding():
    if _backend.vendor() == "cpu":
        pytest.skip("resident linear operations require the GPU training binding")
    return T._OPT_CONTEXT["bindings"]["identical"]["module"]


@pytest.mark.parametrize("case", ["matrix", "batch", "vector", "strided"])
@pytest.mark.parametrize("operation", ["linear_forward", "linear_backward"])
def test_public_device_linear_consumed_outputs_are_native(device_binding, case, operation):
    a, weight, gradient = inputs(case)
    expected = kernel_oracle(operation, a, weight, gradient, "identical")
    a, weight, gradient = [T.train_to_device(v, numeric_mode="identical") for v in (a, weight, gradient)]
    args = (a, weight) if operation == "linear_forward" else (gradient, a, weight)
    actual = audit_native_call(getattr(T, operation), *args, numeric_mode="identical").assert_native()
    assert_outputs(actual, expected, device=True)


@pytest.mark.parametrize("operation", ["linear_forward", "linear_backward"])
def test_public_device_linear_refuses_mixed_owners(device_binding, operation):
    a, weight, gradient = inputs("matrix")
    a = T.train_to_device(a, numeric_mode="identical")
    args = (a, weight) if operation == "linear_forward" else (gradient, a, weight)
    assert_refused(audit_native_call(getattr(T, operation), *args, numeric_mode="identical"), TypeError)


@pytest.mark.parametrize("operation,slot", [("linear_forward", "input"), ("linear_forward", "weight"),
                                           ("linear_backward", "input"), ("linear_backward", "weight"),
                                           ("linear_backward", "gradient")])
def test_public_device_linear_nonfinite_refusal_releases_outputs(device_binding, operation, slot):
    a, weight, gradient = inputs("matrix")
    selected = {"input": a, "weight": weight, "gradient": gradient}[slot]
    selected.flat[0] = np.nan
    a, weight, gradient = [T.train_to_device(v, numeric_mode="identical") for v in (a, weight, gradient)]
    count = 1 if operation == "linear_forward" else 2
    handles = [device_binding.train_dev_alloc(4) for _ in range(count)]
    for handle in handles:
        device_binding.train_dev_free(handle)
    args = (a, weight) if operation == "linear_forward" else (gradient, a, weight)
    result = audit_native_call(getattr(T, operation), *args, numeric_mode="identical")
    assert_refused(result, Exception)
    assert "non-finite" in str(result.exception)
    output = np.empty(4, dtype="f4")
    for handle in handles:
        with pytest.raises(Exception, match="not open"):
            device_binding.train_dev_get(handle, output.ctypes.data, output.size)
    # Failed outputs cannot release or corrupt the caller's input owners.
    for value in (a, weight, gradient):
        audit_native_call(value.to_host).assert_native()
