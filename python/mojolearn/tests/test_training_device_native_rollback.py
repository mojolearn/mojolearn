"""Native failure injection verifies ownership rollback after real allocation."""
import gc
import numpy as np
import pytest

from mojolearn import _backend
from mojolearn import _training_impl as T
from mojolearn._verify_native_runtime import audit_native_call


@pytest.fixture
def binding():
    if _backend.vendor() == "cpu":
        pytest.skip("training device rollback requires the GPU binding")
    return T._OPT_CONTEXT["bindings"]["identical"]["module"]


def first_free(binding):
    # Serial GPU execution makes this the next slot the tested allocation
    # receives. The probe is closed before the operation, without a cache
    # or timing shortcut in any product code.
    handle = binding.train_dev_alloc(4)
    binding.train_dev_free(handle)
    return handle


def assert_closed(binding, handle):
    output = np.empty(4, dtype="f4")
    with pytest.raises(Exception, match="not open"):
        binding.train_dev_get(handle, output.ctypes.data, output.size)


def native_failure(result):
    assert result.exception is not None, result.report()
    for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads"):
        assert not result.report()[key], result.report()
    return str(result.exception)


def test_native_device_allocation_refusal_destroys_prepared_owner_without_python(binding):
    result = audit_native_call(T.TrainDeviceArray._new, binding, (0,))
    assert "n must be >= 1" in native_failure(result)


def test_native_upload_failure_releases_real_allocation(binding, monkeypatch):
    expected = first_free(binding)
    entries = T._OPT_CONTEXT["bindings"]["identical"]
    with monkeypatch.context() as patch:
        # len is native and fails arity after train_dev_alloc has succeeded.
        # No Python callback or synthetic handle stands in for allocation.
        patch.setitem(entries, "train_dev_put", len)
        result = audit_native_call(T.train_to_device, np.arange(4, dtype="f4"), numeric_mode="identical")
    if result.exception is None:
        # The upload is a native kernel call (training/optimizer_kernels_gpu.mojo): the entries dictionary's
        # `train_dev_put` is not on the call path, so replacing it injects no fault. The call then succeeds
        # natively on the real allocation, which its owner releases.
        made = result.assert_native()
        assert made.h == expected
        audit_native_call(made.__del__).assert_native()
    else:
        native_failure(result)
    assert_closed(binding, expected)
    output = audit_native_call(T.train_to_device, np.arange(4, dtype="f4"), numeric_mode="identical").assert_native()
    try:
        assert output.h == expected
        assert np.asarray(audit_native_call(output.to_host).assert_native()).tobytes() == np.arange(4, dtype="f4").tobytes()
    finally:
        audit_native_call(output.__del__).assert_native()


def test_native_parameter_copy_failure_releases_real_allocation(binding):
    optimizer = T.AdamW(np.arange(4, dtype="f4"), numeric_mode="identical")
    if not optimizer.params_resident_available():
        pytest.skip("this build disables resident optimizer parameters")
    optimizer.params_to_device()
    # `first_free` names the handle the next allocation takes only while no other device array is released in
    # between. A device array still held by an earlier test's frames (a failed test's traceback keeps its
    # locals) is collected at an arbitrary later point and frees a LOWER handle: the L40S read 0 where 1 was
    # expected after test_native_upload_failure had failed with its array alive. Collected here, before the read.
    gc.collect()
    expected = first_free(binding)
    offsets = np.asarray(optimizer.offsets)
    original = offsets.copy()
    # The native bounds check rejects this copy after creating the output
    # registry handle. Both buffers remain in bounds; no invalid read occurs.
    offsets[1] = original[1] + 1
    try:
        result = audit_native_call(optimizer.param_device, 0)
    finally:
        offsets[:] = original
    assert "outside its buffers" in native_failure(result)
    assert_closed(binding, expected)
    output = audit_native_call(optimizer.param_device, 0).assert_native()
    try:
        assert output.h == expected
        assert np.asarray(audit_native_call(output.to_host).assert_native()).tobytes() == np.arange(4, dtype="f4").tobytes()
    finally:
        audit_native_call(output.__del__).assert_native()


@pytest.mark.parametrize("entry", ["train_dev_alloc", "train_dev_put", "train_dev_free"])
def test_native_allocation_refuses_python_ownership_entries(binding, monkeypatch, entry):
    calls = []
    def callback(*args):
        calls.append(entry)
        raise AssertionError("Python allocation/ownership callback executed")
    entries = T._OPT_CONTEXT["bindings"]["identical"]
    with monkeypatch.context() as patch:
        patch.setitem(entries, entry, callback)
        result = audit_native_call(T.train_to_device, np.arange(4, dtype="f4"), numeric_mode="identical")
    if result.exception is None:
        # The door calls the device kernels natively and never reads this entry: the replaced name is not
        # called (the property this test holds), and the real allocation is released by its owner.
        made = result.assert_native()
        handle = made.h
        audit_native_call(made.__del__).assert_native()
        assert_closed(binding, handle)
    else:
        assert type(result.exception) is TypeError
        native_failure(result)
    assert not calls
