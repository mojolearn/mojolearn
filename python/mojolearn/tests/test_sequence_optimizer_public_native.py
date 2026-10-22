"""Audit the actual sequence optimizer lifecycle, including state outputs."""
import numpy as np
import pytest

from mojolearn import _x_sequence_optim as O
from mojolearn import _x_sequence_sched as S
from mojolearn import training
from mojolearn._verify_native_runtime import audit_native_call

CLASSES = (O.RMSprop, O.Adagrad, O.Lion, O.Adamax, O.NAdam, O.Adafactor, O.LAMB)


def refused(operation, *args, **kwargs):
    audit = audit_native_call(operation, *args, **kwargs)
    report = audit.report()
    assert audit.exception is not None, report
    for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads"):
        assert not report[key], report
    return audit.exception


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("schedule", [None, "step", "warmup"])
def test_sequence_constructor_step_state_resume_are_native(cls, schedule):
    params = [np.array([[1., -.5, .25], [-2., .75, .125]], dtype="f4"),
              np.array([.2, -.8], dtype="f4")]
    opt = audit_native_call(cls, params, numeric_mode="identical").assert_native()
    if schedule == "step":
        opt.lr_schedule = S.StepLR(.003, 2, .5)
    elif schedule == "warmup":
        opt.lr_schedule = training.WarmupCosineLR(.003, 2, 9)
    grad = [np.full_like(p, .125) for p in params]
    audit_native_call(opt.step, grad).assert_native()
    assert audit_native_call(O._ResidentState.t.fget, opt).assert_native() == 1
    state = audit_native_call(O._ResidentState.state.fget, opt).assert_native()
    audit_native_call(O._ResidentState.state.fset, opt, state).assert_native()
    assert audit_native_call(O._ResidentState._host_owned.fget, opt).assert_native()
    audit_native_call(O._ResidentState.resident_.fget, opt).assert_native()
    if cls is not O.Adafactor:
        saved = audit_native_call(opt.state_dict).assert_native()
        clone_params = [p.copy() for p in params]
        clone = audit_native_call(cls, clone_params, numeric_mode="identical").assert_native()
        if schedule:
            clone.lr_schedule = opt.lr_schedule
        audit_native_call(clone.load_state_dict, saved).assert_native()
        audit_native_call(clone.step, grad).assert_native()
        audit_native_call(opt.step, grad).assert_native()
        for actual, expected in zip(params, clone_params):
            assert actual.tobytes() == expected.tobytes()
        audit_native_call(clone.__del__).assert_native()
    audit_native_call(opt.__del__).assert_native()


@pytest.mark.parametrize("cls", CLASSES)
def test_sequence_native_admission_refuses_python_hooks(cls):
    calls = []
    class Foreign:
        def __getattribute__(self, name):
            calls.append(name)
            raise AssertionError("Python input hook invoked")
        def __float__(self):
            calls.append("float")
            raise AssertionError("Python conversion invoked")
        def __array__(self, *args, **kwargs):
            calls.append("array")
            raise AssertionError("Python input protocol invoked")
    params = np.ones(3, dtype="f4")
    refused(cls, [Foreign()], numeric_mode="identical")
    refused(cls, params, lr=Foreign(), numeric_mode="identical")
    opt = cls(params, numeric_mode="identical")
    refused(opt.step, [Foreign()])
    opt.lr_schedule = Foreign()
    error = refused(opt.step, np.ones_like(params))
    assert "native schedule" in str(error)
    del opt.lr_schedule
    assert not calls


@pytest.mark.parametrize("cls", CLASSES)
def test_sequence_strided_and_integer_gradients_match_contiguous(cls):
    original = np.arange(12, dtype="f4").reshape(3, 4) / 16
    first, second = original.copy(), original.copy()
    one, two = cls([first], numeric_mode="identical"), cls([second], numeric_mode="identical")
    for dtype in ("f4", "f2", "i4"):
        grad = np.arange(24, dtype=dtype).reshape(3, 8)[:, ::2]
        audit_native_call(one.step, [grad]).assert_native()
        audit_native_call(two.step, [grad.astype("f4", order="C")]).assert_native()
        assert first.tobytes() == second.tobytes()
    refused(one.step, [np.ones_like(first, dtype="f8")])


@pytest.mark.parametrize("cls", [O.RMSprop, O.Adagrad, O.Lion, O.Adamax, O.NAdam, O.LAMB])
def test_sequence_checkpoint_broadcast_and_legacy_scalars_are_native(cls):
    opt = cls(np.ones(3, dtype="f4"), numeric_mode="identical")
    saved = opt.state_dict()
    saved["state"] = [np.array([.125], dtype="f8") for _ in saved["state"]]
    saved.pop("scalars")
    audit_native_call(opt.load_state_dict, saved).assert_native()
    for state in audit_native_call(opt.state_dict).assert_native()["state"]:
        assert state.tolist() == [.125] * 3
    saved["state"] = [[1., 2.]]
    refused(opt.load_state_dict, saved)
    saved["state"] = [object()]
    refused(opt.load_state_dict, saved)


def test_sequence_public_state_properties_and_empty_finalizer_are_native():
    for cls, name in ((O.RMSprop, "square_avg"), (O.Adagrad, "sum"), (O.Lion, "exp_avg")):
        opt = cls(np.ones(3, dtype="f4"), numeric_mode="identical")
        value = audit_native_call(vars(cls)[name].fget, opt).assert_native()
        assert value.shape == (3,)
    empty = object.__new__(O.Adamax)
    audit_native_call(empty.__del__).assert_native()


def test_sequence_to_device_cpu_fallback_is_native():
    from mojolearn._x_sequence_device import to_device, SequenceDeviceTensor
    source = np.arange(12, dtype="f4").reshape(3, 4)[:, ::2]
    result = audit_native_call(to_device, source, numeric_mode="identical").assert_native()
    if type(result) is SequenceDeviceTensor:
        result = audit_native_call(result.numpy).assert_native()
    assert result.tobytes() == source.copy().tobytes()
    refused(to_device, source.astype("f8"), numeric_mode="identical")


@pytest.mark.parametrize("cls", CLASSES)
def test_sequence_device_owner_step_consumed_outputs_and_finalizers_are_native(cls):
    from mojolearn._x_sequence_device import to_device, SequenceDeviceTensor
    import weakref
    source = np.arange(12, dtype="f4").reshape(3, 4) / 16
    device = audit_native_call(to_device, source, numeric_mode="identical").assert_native()
    if type(device) is not SequenceDeviceTensor:
        pytest.skip("native device tensor ABI is not present in this CPU build")
    assert audit_native_call(to_device, device).assert_native() is device
    assert audit_native_call(SequenceDeviceTensor.size.fget, device).assert_native() == 12
    assert audit_native_call(SequenceDeviceTensor.ndim.fget, device).assert_native() == 2
    copy = audit_native_call(device.copy).assert_native()
    view = audit_native_call(copy.reshape, (2, -1)).assert_native()
    weak = weakref.ref(copy)
    del copy
    assert weak() is not None
    host = audit_native_call(view.numpy).assert_native()
    assert host.shape == (2, 6) and host.tobytes() == source.tobytes()
    opt = audit_native_call(cls, [device], numeric_mode="identical").assert_native()
    host_param = source.copy()
    reference = cls([host_param], numeric_mode="identical")
    gradient = np.arange(24, dtype="f4").reshape(3, 8)[:, ::2] / 32
    audit_native_call(opt.step, [gradient]).assert_native()
    reference.step([gradient.copy()])
    assert audit_native_call(device.numpy).assert_native().tobytes() == host_param.tobytes()
    audit_native_call(device.upload, gradient).assert_native()
    assert audit_native_call(device.numpy).assert_native().tobytes() == gradient.copy().tobytes()
    # Explicitly finalize only independent owners; a view keeps its base alive.
    root = weak()
    audit_native_call(view.__del__).assert_native()
    del view
    audit_native_call(root.__del__).assert_native()
    audit_native_call(opt.__del__).assert_native()
    audit_native_call(device.__del__).assert_native()
    audit_native_call(device.__del__).assert_native()


def test_sequence_device_partial_owner_finalizer_is_native():
    from mojolearn._x_sequence_device import SequenceDeviceTensor
    empty = object.__new__(SequenceDeviceTensor)
    audit_native_call(empty.__del__).assert_native()


@pytest.mark.parametrize("sign", [1, -1])
def test_sequence_int64_state_conversion_preserves_direct_float32_rounding(sign):
    # Just above a float32 midpoint at an int64 magnitude where float64 loses
    # the last bit: double-rounding would change this established NumPy cast.
    value = np.int64(sign * ((1 << 62) + (1 << 38) + 1))
    expected = np.array([value], dtype="i8").astype("f4").tobytes()
    opt = audit_native_call(O.Adagrad, np.ones(1, dtype="f4"),
                            initial_accumulator_value=value, numeric_mode="identical").assert_native()
    assert audit_native_call(O.Adagrad.sum.fget, opt).assert_native().tobytes() == expected
    saved = opt.state_dict()
    saved["state"] = [np.array([value], dtype="i8") for _ in saved["state"]]
    audit_native_call(opt.load_state_dict, saved).assert_native()
    for actual in audit_native_call(opt.state_dict).assert_native()["state"]:
        assert actual.tobytes() == expected


@pytest.mark.parametrize("cls", CLASSES)
def test_sequence_state_setter_preserves_native_tuple_containers(cls):
    opt = cls(np.ones(3, dtype="f4"), numeric_mode="identical")
    state = tuple(opt.state)
    audit_native_call(O._ResidentState.state.fset, opt, state).assert_native()
    assert audit_native_call(O._ResidentState.state.fget, opt).assert_native() is state
    audit_native_call(opt.step, np.ones(3, dtype="f4")).assert_native()
