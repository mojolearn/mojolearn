"""Missing owner boundaries: independent release checks and resident state writes."""
import numpy as np
import pytest

from mojolearn import _training_impl as T
from mojolearn._verify_native_runtime import audit_native_call


def _refused_native(operation, *args, **kwargs):
    audit = audit_native_call(operation, *args, **kwargs)
    assert audit.exception is not None, audit.report()
    for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads"):
        assert not audit.report()[key], audit.report()
    return audit.exception


@pytest.mark.parametrize("cls", [T.SGD, T.Adam, T.AdamW])
@pytest.mark.parametrize("resident", [False, None])
def test_replaced_moments_survive_native_checkpoint_and_resume(cls, resident):
    options = dict(lr=.003, resident=resident, numeric_mode="identical")
    if cls is T.SGD:
        options.update(momentum=.75, dampening=.25)
    params = np.arange(8, dtype="f4") / 16
    owner = cls(params, **options)
    grads = np.arange(8, dtype="f4") / 32 - .125
    owner.step(grads)
    mean = np.linspace(-.125, .25, 8, dtype="f4")
    variance = np.linspace(.125, .5, 8, dtype="f4")
    mean_bytes, variance_bytes = mean.tobytes(), variance.tobytes()
    audit_native_call(T._Optimizer.exp_avg.fset, owner, mean).assert_native()
    audit_native_call(T._Optimizer.exp_avg_sq.fset, owner, variance).assert_native()
    checkpoint = audit_native_call(owner.state_dict).assert_native()
    assert np.asarray(checkpoint["exp_avg"]).tobytes() == mean_bytes
    assert np.asarray(checkpoint["exp_avg_sq"]).tobytes() == variance_bytes
    resumed_params = params.copy()
    resumed = cls(resumed_params, **options)
    audit_native_call(resumed.load_state_dict, checkpoint).assert_native()
    audit_native_call(owner.step, grads.copy()).assert_native()
    audit_native_call(resumed.step, grads.copy()).assert_native()
    assert params.tobytes() == resumed_params.tobytes()
    # Caller checkpoint data remains a reusable snapshot after both steps.
    assert np.asarray(checkpoint["exp_avg"]).tobytes() == mean_bytes
    assert np.asarray(checkpoint["exp_avg_sq"]).tobytes() == variance_bytes


def test_resident_disabled_io_refuses_natively_without_modifying_buffers():
    params = np.arange(6, dtype="f4").reshape(3, 2)
    owner = T.AdamW(params, resident=False, numeric_mode="identical")
    assert not audit_native_call(owner.params_resident_available).assert_native()
    # Already-host parameters are returned unchanged; this is not a transfer.
    assert audit_native_call(owner.params_to_host).assert_native() is owner.params
    gradients = np.full_like(params, -.25)
    before = params.tobytes(), gradients.tobytes()
    for operation, args in ((owner.params_to_device, ()),
                            (owner.grads_to_device, ([gradients],)),
                            (owner.grads_to_host, ([gradients],))):
        error = _refused_native(operation, *args)
        assert "device-resident parameter" in str(error)
    assert (params.tobytes(), gradients.tobytes()) == before


@pytest.mark.parametrize("entry", ["seq_tensor_upload", "seq_tensor_copy"])
def test_failed_native_sequence_transfer_releases_allocated_handle(monkeypatch, entry):
    from mojolearn._x_sequence_device import SequenceDeviceTensor, _SEQ_CONTEXT, to_device
    source = np.arange(15, dtype="f4").reshape(5, 3)
    owner = to_device(source, numeric_mode="identical")
    if type(owner) is not SequenceDeviceTensor:
        pytest.skip("the CPU binding has no resident tensor pool")
    binding = owner.b
    # A freed slot is reused first. Keep the source owner alive to ensure its
    # handle is distinct from the destination allocation under test.
    probe = binding.seq_tensor_alloc(source.size, 0)
    binding.seq_tensor_free(probe)
    entries = _SEQ_CONTEXT["sequence_bindings"]["identical"]
    with monkeypatch.context() as patch:
        # Native arity failure before input access, not a Python callback.
        patch.setitem(entries, entry, len)
        error = (_refused_native(to_device, source, numeric_mode="identical")
                 if entry == "seq_tensor_upload" else _refused_native(owner.copy))
        assert "len() takes exactly one argument" in str(error)
    output = np.empty(source.size, dtype="f4")
    with pytest.raises(Exception, match="not open"):
        binding.seq_tensor_download(probe, output.ctypes.data, source.size)
    # The failed destination operation never closes or corrupts its source.
    assert audit_native_call(owner.numpy).assert_native().tobytes() == source.tobytes()
    handle = owner.h
    audit_native_call(owner.__del__).assert_native()
    audit_native_call(owner.__del__).assert_native()
    with pytest.raises(Exception, match="not open"):
        binding.seq_tensor_download(handle, output.ctypes.data, source.size)
