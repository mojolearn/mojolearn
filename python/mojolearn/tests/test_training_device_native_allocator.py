"""Training device owners use native allocation and retain layout admission."""
import builtins
import numpy as np
import pytest
from mojolearn import _backend, _training_impl as T
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests._connected_exports import assert_connected_export_refused


@pytest.mark.parametrize("operation", ["upload", "reshape"])
def test_training_device_owner_refuses_replaced_object_constructor(monkeypatch, operation):
    if _backend.vendor() == "cpu":
        pytest.skip("training device allocation requires the GPU pool")
    source = np.arange(12, dtype="f4").reshape(4, 3)
    owner = T.train_to_device(source, numeric_mode="identical") if operation == "reshape" else None
    method, args = (owner.reshape, ((2, 6),)) if owner is not None else (T.train_to_device, (source,))
    kwargs = {} if owner is not None else {"numeric_mode": "identical"}
    calls = []
    class Replacement:
        def __new__(cls, *args, **kwargs):
            calls.append(cls)
            raise AssertionError("mutable device object constructor entered")
    with monkeypatch.context() as scope:
        scope.setattr(builtins, "object", Replacement)
        refused = audit_native_call(method, *args, **kwargs)
    # Since 31fc85981 a replaced connected builtins export is refused before
    # any callback runs (tests/_connected_exports.py); then the restored
    # export allocates natively.
    assert_connected_export_refused(refused, "builtins", "object")
    assert not calls
    result = audit_native_call(method, *args, **kwargs)
    device = result.assert_native()
    expected = source.reshape(2, 6) if owner is not None else source
    host = audit_native_call(device.to_host).assert_native()
    assert np.asarray(host).tobytes() == expected.tobytes()
    assert host.shape == expected.shape
    audit_native_call(T.TrainDeviceArray.__del__, device).assert_native()
    if owner is not None:
        audit_native_call(T.TrainDeviceArray.__del__, owner).assert_native()


def test_training_device_layout_hook_is_refused_before_allocation(monkeypatch):
    source = np.ones((2, 3), dtype="f4")
    calls = []
    def hook(*args):
        calls.append(args)
        raise AssertionError("device layout setter entered")
    monkeypatch.setattr(T.TrainDeviceArray, "__setattr__", hook)
    result = audit_native_call(T.train_to_device, source, numeric_mode="identical")
    assert type(result.exception) is TypeError, result.report()
    assert "replaced descriptor" in str(result.exception)
    assert not calls
    for field in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads"):
        assert not result.report()[field], result.report()
