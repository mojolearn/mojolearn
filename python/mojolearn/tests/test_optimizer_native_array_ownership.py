"""Native Array owners acquire storage without a mutable constructor callback."""
import builtins

import numpy as np
import pytest

from mojolearn import _training_impl as T
from mojolearn._array import Array
from mojolearn._verify_native_runtime import audit_native_call


@pytest.mark.parametrize("operation", ["rms", "linear", "construct", "accumulate"])
def test_native_output_storage_does_not_call_mutable_memoryview(monkeypatch, operation):
    _native_output_storage_audit(monkeypatch, operation, "memoryview")


@pytest.mark.parametrize("operation", ["rms", "linear", "construct", "accumulate"])
def test_native_output_storage_does_not_call_mutable_object(monkeypatch, operation):
    _native_output_storage_audit(monkeypatch, operation, "object")


def _native_output_storage_audit(monkeypatch, operation, constructor):
    source = np.arange(12, dtype="f4").reshape(4, 3) / 8
    weight = np.ones(3, dtype="f4")
    params = np.ones(3, dtype="f4")
    if operation == "rms":
        method, args = T.rms_norm_forward, (source, weight, 1e-5)
    elif operation == "linear":
        method, args = T.linear_forward, (source, np.ones((2, 3), dtype="f4"))
    elif operation == "construct":
        method, args = T.AdamW, ([params.copy()],)
    else:
        method, args = T.accumulate_grads, ([np.ones(3, dtype="f4"), np.ones(3, dtype="f4")],)
    kwargs = {"numeric_mode": "identical"} if operation in ("rms", "linear", "construct") else {}
    if operation == "accumulate":
        kwargs["tokens"] = 512
    calls = []
    def hook(*args, **kwargs):
        calls.append(args)
        raise AssertionError("mutable output constructor entered")
    class ReplacedObject:
        __new__ = staticmethod(hook)
    with monkeypatch.context() as scope:
        scope.setattr(builtins, constructor, ReplacedObject if constructor == "object" else hook)
        audit = audit_native_call(method, *args, **kwargs)
    result = audit.assert_native()
    assert not calls
    if operation == "construct":
        outputs = [result.offsets, result.buf_initialized]
    else:
        outputs = [result]
    for value in outputs:
        assert type(value) is Array
        # Storage remains live and writable after native output publication.
        actual = np.asarray(value)
        assert actual.size == value.size
        assert actual.flags.writeable
        assert value._mv.obj is value._store
    if operation == "accumulate":
        assert np.array_equal(np.asarray(result), np.full(3, 2, dtype="f4"))
    elif operation == "construct":
        assert np.asarray(result.offsets).tolist() == [0, 3]
        assert np.asarray(result.buf_initialized).tolist() == [0]


def test_native_checkpoint_arrays_own_their_storage_after_publication():
    owner = T.AdamW([np.ones(3, dtype="f4")], numeric_mode="identical", resident=False)
    audit_native_call(owner.step, [np.ones(3, dtype="f4")]).assert_native()
    result = audit_native_call(owner.state_dict).assert_native()
    del owner
    for name in ("exp_avg", "exp_avg_sq", "buf_initialized"):
        value = result[name]
        assert type(value) is Array
        assert value._mv.obj is value._store
        assert np.asarray(value).flags.writeable
        assert np.asarray(value).size == value.size


@pytest.mark.parametrize("strided", [False, True])
def test_native_device_download_does_not_call_mutable_memoryview(monkeypatch, strided):
    _native_device_download_audit(monkeypatch, strided, "memoryview")


@pytest.mark.parametrize("strided", [False, True])
def test_native_device_download_does_not_call_mutable_object(monkeypatch, strided):
    _native_device_download_audit(monkeypatch, strided, "object")


def _native_device_download_audit(monkeypatch, strided, constructor):
    from mojolearn import _backend
    if _backend.vendor() == "cpu":
        pytest.skip("device downloads require the GPU training pool")
    source = np.arange(12, dtype="f4").reshape(3, 4)
    if strided:
        source = source[:, ::2]
    device = T.train_to_device(source, numeric_mode="identical")
    calls = []
    def hook(*args, **kwargs):
        calls.append(args)
        raise AssertionError("mutable output constructor entered")
    class ReplacedObject:
        __new__ = staticmethod(hook)
    with monkeypatch.context() as scope:
        scope.setattr(builtins, constructor, ReplacedObject if constructor == "object" else hook)
        result = audit_native_call(device.to_host)
    value = result.assert_native()
    assert not calls
    assert value._mv.obj is value._store
    assert np.asarray(value).tobytes() == source.tobytes()
    audit_native_call(T.TrainDeviceArray.__del__, device).assert_native()
