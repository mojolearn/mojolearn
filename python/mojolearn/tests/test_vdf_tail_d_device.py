# SPDX-License-Identifier: Apache-2.0
"""Boundary words of lane vdf-tail-d: `Generator.dropout` / `dropout_backward`
on a TrainDeviceArray, the sequence optimizers' `lr_schedule` device
agreement, and `config()` / `from_config` carrying a schedule's `device`.
The dropout bits on device arrays are the native check
`checks/native_tests/generator_dropout_device_array.mojo`; nothing numerical
is asserted here."""
import numpy as np
import pytest

import mojolearn
from mojolearn import training
from mojolearn import _training_impl as T

BAD_WORDS = "device must be 'gpu', 'cpu' or None"
GPU_ABSENT = "needs a GPU build of mojolearn"
MISMATCH = "was constructed with device='cpu'; give both the same device, or leave one of them None"
SIDE = "cpu" if mojolearn.vendor() == "cpu" else "gpu"


def _w():
    return np.zeros(8, dtype=np.float32)


# (public name, a construction over one tiny parameter that forwards `device`)
SEQUENCE_OPTIMIZERS = [
    ("RMSprop", lambda d: mojolearn.RMSprop([_w()], device=d)),
    ("Adagrad", lambda d: mojolearn.Adagrad([_w()], device=d)),
    ("Lion", lambda d: mojolearn.Lion([_w()], device=d)),
    ("Adamax", lambda d: mojolearn.Adamax([_w()], device=d)),
    ("NAdam", lambda d: mojolearn.NAdam([_w()], device=d)),
    ("LAMB", lambda d: mojolearn.LAMB([_w()], device=d)),
    ("Adafactor", lambda d: mojolearn.Adafactor([_w()], device=d)),
]
IDS = [name for name, _ in SEQUENCE_OPTIMIZERS]


@pytest.mark.parametrize("name,make", SEQUENCE_OPTIMIZERS, ids=IDS)
def test_sequence_optimizer_refuses_other_side_schedule_where_attached(name, make):
    # Holds on a GPU installation (a 'gpu' optimizer beside a 'cpu' schedule);
    # a CPU-only installation has no 'gpu' optimizer and refuses it first.
    if mojolearn.vendor() == "cpu":
        with pytest.raises(ValueError, match=GPU_ABSENT):
            make("gpu")
        return
    opt = make("gpu")
    sched = mojolearn.StepLR(0.1, step_size=3, device="cpu")
    with pytest.raises(ValueError, match="mojolearn." + name + " was given device='gpu' but its lr_schedule StepLR "
                       + MISMATCH):
        opt.lr_schedule = sched
    assert opt.lr_schedule is None


@pytest.mark.parametrize("name,make", SEQUENCE_OPTIMIZERS, ids=IDS)
def test_sequence_optimizer_takes_same_or_unnamed_side_schedule(name, make):
    opt = make(SIDE)
    for sched in (mojolearn.ExponentialLR(0.1, gamma=0.9, device=SIDE), mojolearn.ExponentialLR(0.1, gamma=0.9)):
        opt.lr_schedule = sched
        assert opt.lr_schedule is sched
    unnamed = make(None)
    unnamed.lr_schedule = mojolearn.StepLR(0.1, step_size=3, device=SIDE)
    del unnamed.lr_schedule
    assert unnamed.lr_schedule is None


@pytest.mark.parametrize("name,make", SEQUENCE_OPTIMIZERS, ids=IDS)
def test_sequence_optimizer_refuses_a_python_schedule(name, make):
    opt = make(None)
    with pytest.raises(TypeError, match="built-in native schedule"):
        opt.lr_schedule = lambda t: 0.1


SCHEDULES = [
    lambda d: training.ConstantLR(1e-3, warmup_steps=2, device=d),
    lambda d: training.WarmupLinearLR(1e-3, warmup_steps=2, total_steps=10, device=d),
    lambda d: training.WarmupCosineLR(1e-3, warmup_steps=2, total_steps=10, device=d),
]


@pytest.mark.parametrize("make", SCHEDULES, ids=["ConstantLR", "WarmupLinearLR", "WarmupCosineLR"])
@pytest.mark.parametrize("device", [None, SIDE])
def test_schedule_config_carries_device(make, device):
    cfg = make(device).config()
    assert cfg["device"] == device
    assert T._Schedule.from_config(cfg).device == device


def test_schedule_from_config_without_device_and_with_a_bad_word():
    cfg = training.WarmupCosineLR(1e-3, warmup_steps=2, total_steps=10).config()
    del cfg["device"]
    assert T._Schedule.from_config(cfg).device is None
    cfg["device"] = "tpu"
    with pytest.raises(ValueError, match=BAD_WORDS):
        T._Schedule.from_config(cfg)


def _device_array():
    return T.train_to_device(np.linspace(-1.0, 1.0, 16, dtype=np.float32), numeric_mode="identical")


def test_generator_dropout_returns_device_arrays_for_device_input():
    # Holds on a GPU installation with training device arrays (identical GPU
    # builds); a CPU-only installation has no TrainDeviceArray to pass.
    if mojolearn.vendor() == "cpu":
        return
    x = _device_array()
    gen = training.Generator(0, numeric_mode="identical")
    y, key = gen.dropout(x, 0.25)
    assert type(y) is T.TrainDeviceArray
    assert type(key) is dict and set(key) == {"seed", "stream", "p", "scale", "offset"}
    dx = gen.dropout_backward(_device_array(), key)
    assert type(dx) is T.TrainDeviceArray


def test_generator_dropout_device_input_refused_with_cpu_generator():
    # Holds on a GPU installation: a Generator constructed with device='cpu'
    # beside the GPU set refuses an input that lives only on the device.
    if mojolearn.vendor() == "cpu":
        return
    gen = training.Generator(0, numeric_mode="identical", device="cpu")
    with pytest.raises(ValueError, match="x is a training device array, but this Generator was constructed with "
                       + "device='cpu'"):
        gen.dropout(_device_array(), 0.25)
    key = training.Generator(0, numeric_mode="identical").dropout(np.ones(16, dtype=np.float32), 0.25)[1]
    with pytest.raises(ValueError, match="dy is a training device array, but this Generator was constructed with "
                       + "device='cpu'"):
        gen.dropout_backward(_device_array(), key)
