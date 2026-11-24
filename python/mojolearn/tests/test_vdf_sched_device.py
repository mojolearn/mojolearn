# SPDX-License-Identifier: Apache-2.0
"""`device=` on `training.Generator` and the learning-rate schedules: the
boundary words only (lane vdf-sched). The draws' bits on both sides are the
native check `checks/native_tests/generator_device_side.mojo` and verify's;
nothing numerical is asserted here."""
import inspect

import numpy as np
import pytest

import mojolearn
from mojolearn import training

BAD_WORDS = "device must be 'gpu', 'cpu' or None"
GPU_ABSENT = "needs a GPU build of mojolearn"
MISMATCH = "was constructed with device='cpu'; give both the same device, or leave one of them None"


# (public class, a construction with tiny settings that forwards `device`)
CONSTRUCTIONS = [
    (training.ConstantLR, lambda d: training.ConstantLR(1e-3, warmup_steps=2, device=d)),
    (training.WarmupLinearLR, lambda d: training.WarmupLinearLR(1e-3, warmup_steps=2, total_steps=10, device=d)),
    (training.WarmupCosineLR, lambda d: training.WarmupCosineLR(1e-3, warmup_steps=2, total_steps=10, device=d)),
    (mojolearn.StepLR, lambda d: mojolearn.StepLR(0.1, step_size=3, device=d)),
    (mojolearn.ExponentialLR, lambda d: mojolearn.ExponentialLR(0.1, gamma=0.9, device=d)),
    (mojolearn.OneCycleLR, lambda d: mojolearn.OneCycleLR(max_lr=0.01, total_steps=10, device=d)),
    (training.Generator, lambda d: training.Generator(0, device=d)),
]


@pytest.mark.parametrize("cls", [c for c, _ in CONSTRUCTIONS], ids=lambda c: c.__name__)
def test_signature_has_keyword_only_device(cls):
    parameter = inspect.signature(cls).parameters["device"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


@pytest.mark.parametrize("bad", ["tpu", 0, "GPU"])
@pytest.mark.parametrize("make", [m for _, m in CONSTRUCTIONS], ids=[c.__name__ for c, _ in CONSTRUCTIONS])
def test_bad_word_is_refused(make, bad):
    with pytest.raises(ValueError, match=BAD_WORDS):
        make(bad)


@pytest.mark.parametrize("make", [m for _, m in CONSTRUCTIONS], ids=[c.__name__ for c, _ in CONSTRUCTIONS])
def test_none_and_the_installation_side_construct(make):
    make(None)
    # 'cpu' on a CPU-only installation and 'gpu' on a GPU one name the installation's own side
    make("cpu" if mojolearn.vendor() == "cpu" else "gpu")


@pytest.mark.parametrize("make", [m for _, m in CONSTRUCTIONS], ids=[c.__name__ for c, _ in CONSTRUCTIONS])
def test_gpu_on_cpu_only_installation(make):
    # Holds on either kind of installation: a CPU-only one refuses 'gpu' in the
    # estimators' words; a GPU one constructs it.
    if mojolearn.vendor() == "cpu":
        with pytest.raises(ValueError, match=GPU_ABSENT):
            make("gpu")
    else:
        make("gpu")


def test_schedule_keeps_its_device():
    sched = training.WarmupCosineLR(1e-3, warmup_steps=2, total_steps=10, device="cpu")
    assert sched.device == "cpu"
    assert training.ConstantLR(1e-3).device is None


def _params():
    return [np.ones(4, dtype=np.float32)]


def test_optimizer_with_other_side_schedule():
    # A schedule built for the CPU side, an SGD for the GPU side. On a GPU
    # installation the optimizer refuses the schedule in words naming both; on
    # a CPU-only one the optimizer's own 'gpu' is refused first, in the
    # estimators' words.
    sched = training.WarmupCosineLR(1e-3, warmup_steps=2, total_steps=10, device="cpu")
    if mojolearn.vendor() == "cpu":
        with pytest.raises(ValueError, match=GPU_ABSENT):
            training.SGD(_params(), lr_schedule=sched, device="gpu")
    else:
        with pytest.raises(ValueError, match="mojolearn.SGD was given device='gpu' but its lr_schedule WarmupCosineLR " + MISMATCH):
            training.SGD(_params(), lr_schedule=sched, device="gpu")


@pytest.mark.parametrize("cls", [training.SGD, training.Adam, training.AdamW], ids=lambda c: c.__name__)
def test_set_params_with_other_side_schedule(cls):
    # Holds on a GPU installation only (a CPU-only one has no 'gpu' optimizer);
    # there, set_params refuses the schedule as the constructor does.
    sched = mojolearn.StepLR(0.1, step_size=3, device="cpu")
    if mojolearn.vendor() == "cpu":
        with pytest.raises(ValueError, match=GPU_ABSENT):
            cls(_params(), device="gpu")
        return
    opt = cls(_params(), device="gpu")
    with pytest.raises(ValueError, match="its lr_schedule StepLR " + MISMATCH):
        opt.set_params(lr_schedule=sched)


def test_optimizer_with_same_or_unnamed_side_schedule():
    side = "cpu" if mojolearn.vendor() == "cpu" else "gpu"
    training.SGD(_params(), lr_schedule=training.ConstantLR(1e-3, device=side), device=side)
    training.SGD(_params(), lr_schedule=training.ConstantLR(1e-3), device=side)
    training.SGD(_params(), lr_schedule=training.ConstantLR(1e-3, device=side))
