"""Whole optimizer operations, including preparation/state, never call Python."""
import array

import numpy as np
import pytest

from mojolearn import _training_impl as T
from mojolearn._array import Array
from mojolearn._verify_native_runtime import audit_native_call


def refused_without_python(operation, *args, **kwargs):
    audit = audit_native_call(operation, *args, **kwargs)
    assert audit.exception is not None, audit.report()
    for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads"):
        assert not audit.report()[key], audit.report()
    return audit.exception


@pytest.mark.parametrize("cls", [T.SGD, T.Adam, T.AdamW])
@pytest.mark.parametrize("schedule_cls", [None, T.ConstantLR, T.WarmupLinearLR, T.WarmupCosineLR])
def test_constructor_step_checkpoint_resume_are_native(cls, schedule_cls):
    schedule = None if schedule_cls is None else schedule_cls(0.003, warmup_steps=2, total_steps=9)
    params = [np.array([1.0, -2.0, 0.25], dtype="f4"), array.array("f", [0.1, -0.3])]
    optimizer = audit_native_call(cls, params, lr=0.003, lr_schedule=schedule,
                                 numeric_mode="identical", resident=False).assert_native()
    grads = [np.array([0.3, -0.5, 0.2], dtype="f4"), array.array("f", [0.2, 0.4])]
    norm = audit_native_call(optimizer.step, grads, max_norm=1.0).assert_native()
    assert norm > 0 and optimizer.t == 1
    saved = audit_native_call(optimizer.state_dict).assert_native()
    assert set(saved) == {"t", "exp_avg", "exp_avg_sq", "buf_initialized"}
    clone_params = [np.array(p, dtype="f4", copy=True) for p in params]
    clone = audit_native_call(cls, clone_params, lr=optimizer.lr, lr_schedule=schedule,
                             numeric_mode="identical", resident=False).assert_native()
    audit_native_call(clone.load_state_dict, saved).assert_native()
    audit_native_call(optimizer.step, grads).assert_native()
    audit_native_call(clone.step, grads).assert_native()
    for actual, expected in zip(clone_params, params):
        assert actual.tobytes() == np.array(expected, dtype="f4").tobytes()
    audit_native_call(optimizer.zero_grad, grads).assert_native()
    assert all(np.count_nonzero(g) == 0 for g in grads)


@pytest.mark.parametrize("kind", ["numpy", "array", "mojolearn"])
def test_clip_and_accumulation_are_native(kind):
    raw = np.array([0.25, -0.5, 0.75, 1.0], dtype="f4")
    if kind == "numpy":
        first, second = raw.copy(), raw[::-1]
    elif kind == "array":
        first, second = array.array("f", raw), array.array("f", raw[::-1])
    else:
        first, second = Array.from_buffer(raw), Array.from_buffer(raw[::-1].copy())
    combined = audit_native_call(T.accumulate_grads, [first, second], tokens=4,
                                numeric_mode="identical").assert_native()
    assert isinstance(combined, Array)
    assert combined.shape == (4,)
    norm = audit_native_call(T.clip_grad_norm_, [first], 0.5,
                            numeric_mode="identical").assert_native()
    assert norm > 0.5
    assert audit_native_call(T.accumulation_is_aligned, 4, 2,
                             numeric_mode="identical").assert_native()


def test_accumulated_step_and_nested_checkpoint_input_are_native():
    params = np.array([0.5, -0.5], dtype="f4")
    optimizer = audit_native_call(T.AdamW, params, accumulation_steps=2,
                                 numeric_mode="identical", resident=False).assert_native()
    gradient = np.array([0.2, -0.1], dtype="f4")
    audit_native_call(optimizer.step_accumulated, [gradient, gradient], 4).assert_native()
    state = {"t": 1, "exp_avg": [0.01, -0.01], "exp_avg_sq": [0.001, 0.001],
             "buf_initialized": [1]}
    audit_native_call(optimizer.load_state_dict, state).assert_native()
    restored = audit_native_call(optimizer.state_dict).assert_native()
    assert restored["t"] == 1
    assert np.asarray(restored["buf_initialized"]).tolist() == [1]


def test_step_never_uses_python_array_or_schedule_methods(monkeypatch):
    schedule = T.WarmupCosineLR(0.003, 2, 9)
    optimizer = T.AdamW(np.ones(4, dtype="f4"), resident=False,
                        lr_schedule=schedule, numeric_mode="identical")
    def forbidden(*args, **kwargs):
        raise AssertionError("Python helper called from native optimizer")
    monkeypatch.setattr(Array, "__init__", forbidden)
    monkeypatch.setattr(Array, "copy", forbidden)
    monkeypatch.setattr(Array, "flags", property(forbidden))
    monkeypatch.setattr(T._Schedule, "_lr_spec", forbidden)
    monkeypatch.setattr(T._LrTable, "lr_at", forbidden)
    audit_native_call(optimizer.step, np.ones(4, dtype="f4")).assert_native()
    audit_native_call(optimizer.state_dict).assert_native()


def test_python_schedule_is_refused_before_getters_or_callbacks():
    calls = []
    class Callback:
        def __getattribute__(self, name):
            calls.append(name)
            raise AssertionError("callback metadata consulted")
        def lr_at(self, step):
            calls.append(step)
            return 1.0
    error = refused_without_python(T.SGD, np.ones(2, dtype="f4"), lr_schedule=Callback())
    assert "ConstantLR" in str(error) and not calls
    optimizer = T.SGD(np.ones(2, dtype="f4"), numeric_mode="identical")
    optimizer.lr_schedule = Callback()
    error = refused_without_python(optimizer.step, np.ones(2, dtype="f4"))
    assert "callback" in str(error) and not calls and optimizer.t == 0


@pytest.mark.parametrize("operation", ["construct", "step", "clip", "accumulate", "load"])
def test_python_buffer_protocol_is_refused_before_hooks(operation):
    calls = []
    class Foreign:
        def __getattribute__(self, name):
            calls.append(name)
            raise AssertionError("foreign protocol invoked")
        def __iter__(self):
            calls.append("iter")
            raise AssertionError("foreign iterable invoked")
    optimizer = T.SGD(np.ones(2, dtype="f4"), numeric_mode="identical")
    foreign = Foreign()
    if operation == "construct":
        refused_without_python(T.SGD, foreign)
    elif operation == "step":
        refused_without_python(optimizer.step, foreign)
    elif operation == "clip":
        refused_without_python(T.clip_grad_norm_, foreign, 1.0, numeric_mode="identical")
    elif operation == "accumulate":
        refused_without_python(T.accumulate_grads, [foreign, foreign], 4, numeric_mode="identical")
    else:
        refused_without_python(optimizer.load_state_dict, {"t": 0, "exp_avg": foreign,
                                                        "exp_avg_sq": [0., 0.], "buf_initialized": [0]})
    assert not calls


def test_scalar_callback_is_refused_without_float_or_bool():
    calls = []
    class Foreign:
        def __float__(self):
            calls.append("float")
            return 1.0
        def __bool__(self):
            calls.append("bool")
            return True
    optimizer = T.SGD(np.ones(2, dtype="f4"), numeric_mode="identical")
    refused_without_python(optimizer.step, np.ones(2, dtype="f4"), max_norm=Foreign())
    refused_without_python(T.SGD, np.ones(2, dtype="f4"), lr=Foreign())
    assert not calls


@pytest.mark.parametrize("schedule_name,args", [
    ("ConstantLR", (0.003, 2)), ("WarmupLinearLR", (0.003, 2, 9, 0.0001)),
    ("WarmupCosineLR", (0.003, 2, 9, 0.0001)), ("StepLR", (0.003, 3, 0.8)),
    ("ExponentialLR", (0.003, 0.8)), ("OneCycleLR", (0.003, 9)),
])
def test_builtin_schedule_public_operations_and_optimizer_use_are_native(schedule_name, args):
    import mojolearn
    cls = getattr(mojolearn, schedule_name)
    schedule = audit_native_call(cls, *args).assert_native()
    values = audit_native_call(schedule.lr_values, 7).assert_native()
    for step in (1, 3, 7):
        value = audit_native_call(schedule.lr_at, step).assert_native()
        bits = audit_native_call(schedule.bits_at, step).assert_native()
        assert array.array("f", [value]).tobytes() == values[step - 1:step].tobytes()
        assert bits == np.asarray([value], dtype="f4").view("u4")[0]
    optimizer = audit_native_call(T.SGD, np.ones(3, dtype="f4"), lr_schedule=schedule,
                                 numeric_mode="identical", resident=False).assert_native()
    audit_native_call(optimizer.step, np.ones(3, dtype="f4")).assert_native()
    assert array.array("f", [optimizer.lr]).tobytes() == values[:1].tobytes()
    if schedule_name in ("ConstantLR", "WarmupLinearLR", "WarmupCosineLR"):
        config = audit_native_call(schedule.config).assert_native()
        clone = audit_native_call(schedule.from_config, config).assert_native()
        assert audit_native_call(clone.lr_values, 7).assert_native().tobytes() == values.tobytes()
    else:
        assert audit_native_call(schedule.lrs, 7).assert_native() == values.tolist()


def test_checkpoint_initialized_count_refused_before_modifying_state():
    optimizer = T.SGD(np.ones(2, dtype="f4"), numeric_mode="identical", resident=False)
    state = {"t": 12, "exp_avg": [0.1, 0.2], "exp_avg_sq": [0.3, 0.4],
             "buf_initialized": [0, 0]}
    error = refused_without_python(optimizer.load_state_dict, state)
    assert "buf_initialized" in str(error) and optimizer.t == 0


@pytest.mark.parametrize("value,error_type", [(float("nan"), ValueError),
                                             (float("inf"), OverflowError),
                                             (2147483648, OverflowError)])
def test_checkpoint_int32_refusal_preserves_error_class(value, error_type):
    optimizer = T.SGD(np.ones(2, dtype="f4"), numeric_mode="identical", resident=False)
    state = {"t": 0, "exp_avg": [0., 0.], "exp_avg_sq": [0., 0.], "buf_initialized": [value]}
    assert isinstance(refused_without_python(optimizer.load_state_dict, state), error_type)


def test_partial_device_array_finalization_is_native_and_idempotent():
    owner = object.__new__(T.TrainDeviceArray)
    audit_native_call(T.TrainDeviceArray.__del__, owner).assert_native()
    audit_native_call(T.TrainDeviceArray.__del__, owner).assert_native()
    assert vars(owner) == {}


@pytest.mark.parametrize("strided", [False, True])
def test_consumed_device_outputs_and_release_are_native(strided):
    from mojolearn import _backend
    import gc
    import weakref
    if _backend.vendor() == "cpu":
        pytest.skip("device arrays require the GPU training pool")
    host = np.arange(12, dtype="f4").reshape(3, 4)
    source = host[:, ::2] if strided else host
    device = audit_native_call(T.train_to_device, source, numeric_mode="identical").assert_native()
    assert audit_native_call(device.to_host).assert_native().shape == source.shape
    assert np.asarray(audit_native_call(device.to_host).assert_native()).tobytes() == source.tobytes()
    view = audit_native_call(device.reshape, (source.size,)).assert_native()
    retained = weakref.ref(device)
    handle, binding = device.h, vars(device)["_b"]
    del device
    gc.collect()
    assert retained() is not None, "view must retain its root device allocation"
    assert np.asarray(audit_native_call(view.to_host).assert_native()).tobytes() == source.tobytes()
    # A view's finalizer never closes the root allocation.
    audit_native_call(T.TrainDeviceArray.__del__, view).assert_native()
    root = retained()
    del view
    gc.collect()
    audit_native_call(T.TrainDeviceArray.__del__, root).assert_native()
    audit_native_call(T.TrainDeviceArray.__del__, root).assert_native()
    assert vars(root)["_b"] is None
    output = np.empty(source.size, dtype="f4")
    with pytest.raises(Exception, match="handle|closed|not open|unknown|invalid"):
        binding.train_dev_get(handle, output.ctypes.data, source.size)


def test_optimizer_parameter_device_consumed_output_is_native():
    from mojolearn import _backend
    if _backend.vendor() == "cpu":
        pytest.skip("resident optimizer parameters require the GPU pool")
    params = np.arange(6, dtype="f4").reshape(2, 3)
    optimizer = audit_native_call(T.AdamW, params, numeric_mode="identical").assert_native()
    if not audit_native_call(optimizer.params_resident_available).assert_native():
        pytest.skip("this build disables resident optimizer parameters")
    audit_native_call(optimizer.params_to_device).assert_native()
    output = audit_native_call(optimizer.param_device, 0).assert_native()
    downloaded = audit_native_call(output.to_host).assert_native()
    assert np.asarray(downloaded).tobytes() == params.tobytes()
    audit_native_call(T.TrainDeviceArray.__del__, output).assert_native()
    audit_native_call(optimizer.params_to_host).assert_native()
