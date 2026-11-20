# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`lr_at` / `bits_at` served by the schedule's native rate block
(bindings/lr_schedule_block_door.mojo): forward, backward and random steps,
block edges, refusals, and pickle/copy rebinding, against `lr_values`."""
import copy
import pickle
import random

import pytest

import mojolearn as ml
from mojolearn import training


def _schedules():
    return [
        training.ConstantLR(1e-3, warmup_steps=300),
        training.WarmupLinearLR(1e-3, warmup_steps=100, total_steps=700, min_lr=1e-5),
        training.WarmupCosineLR(3e-4, warmup_steps=50, total_steps=900, min_lr=3e-5),
        ml.StepLR(0.1, 70, 0.5),
        ml.ExponentialLR(0.1, 0.999),
        ml.OneCycleLR(0.1, 600),
    ]


@pytest.mark.parametrize("index", range(6))
def test_block_matches_series_in_every_order(index):
    schedule = _schedules()[index]
    n = 601  # OneCycleLR answers through total_steps + 1
    expected = list(schedule.lr_values(n))
    steps = list(range(1, n + 1))
    orders = [steps, steps[::-1], random.Random(7).sample(steps, len(steps)),
              [1, 256, 257, 512, 513, 255, 258, 1]]  # block edges
    for order in orders:
        for t in order:
            assert schedule.lr_at(t) == expected[t - 1]
    import struct
    assert schedule.bits_at(257) == struct.unpack("<I", struct.pack("<f", expected[256]))[0]


def test_refusals_unchanged():
    schedule = ml.OneCycleLR(0.1, 10)
    with pytest.raises(ValueError, match="ONE-BASED"):
        schedule.lr_at(0)
    with pytest.raises(ValueError, match="beyond total_steps"):
        schedule.lr_at(12)
    assert schedule.lr_at(11) == list(schedule.lr_values(11))[10]
    with pytest.raises(TypeError):
        schedule.lr_at()
    with pytest.raises(TypeError):
        schedule.lr_at(1, 2)


def test_pickle_and_copy_rebind_the_block():
    for schedule in _schedules():
        expected = [schedule.lr_at(t) for t in (1, 2, 300, 600)]
        for clone in (pickle.loads(pickle.dumps(schedule)), copy.copy(schedule), copy.deepcopy(schedule)):
            assert type(clone) is type(schedule)
            assert [clone.lr_at(t) for t in (1, 2, 300, 600)] == expected
            assert clone.lr_at is not schedule.lr_at


def test_keyword_and_arity_forms_match_the_former_method():
    schedule = training.WarmupCosineLR(3e-4, warmup_steps=50, total_steps=900, min_lr=3e-5)
    assert schedule.lr_at(t=5) == schedule.lr_at(5)
    assert schedule.bits_at(t=700) == schedule.bits_at(700)
    with pytest.raises(TypeError, match="unexpected keyword argument 'step'"):
        schedule.lr_at(step=5)
    with pytest.raises(TypeError, match="multiple values for argument 't'"):
        schedule.lr_at(5, t=5)
    with pytest.raises(TypeError, match="missing 1 required positional argument: 't'"):
        schedule.bits_at()
    with pytest.raises(TypeError, match="takes 2 positional arguments but 3 were given"):
        schedule.lr_at(1, 2)


def test_editing_a_setting_changes_lr_at():
    schedule = training.ConstantLR(1e-3, warmup_steps=10)
    before = schedule.lr_at(300)               # fills the block holding step 300
    schedule.peak_lr = 2e-3
    after = schedule.lr_at(300)
    assert after != before
    assert after == training.ConstantLR(2e-3, warmup_steps=10).lr_at(300)
    assert list(schedule.lr_values(300)) == list(training.ConstantLR(2e-3, warmup_steps=10).lr_values(300))
    step = ml.StepLR(0.1, 70, 0.5)
    _ = step.lr_at(100)
    step.gamma = 0.25
    assert step.lr_at(100) == ml.StepLR(0.1, 70, 0.25).lr_at(100)


def test_an_edit_the_admission_refuses_raises_at_the_call():
    schedule = training.ConstantLR(1e-3, warmup_steps=10)
    schedule.peak_lr = "not a number"          # stored, as before
    with pytest.raises((TypeError, ValueError)):
        schedule.lr_at(1)
    schedule.peak_lr = 1e-3                    # a valid edit rebinds the block
    assert schedule.lr_at(1) == training.ConstantLR(1e-3, warmup_steps=10).lr_at(1)


def test_deleting_a_setting_leaves_no_stale_block():
    schedule = training.ConstantLR(1e-3, warmup_steps=10)
    _ = schedule.lr_at(5)                      # the block is filled
    del schedule.peak_lr
    with pytest.raises(Exception):
        schedule.lr_at(5)                      # no stale answer from the old block
    with pytest.raises(AttributeError):
        del schedule.no_such_setting
    schedule.peak_lr = 2e-3                    # a valid edit rebinds the block
    assert schedule.lr_at(5) == training.ConstantLR(2e-3, warmup_steps=10).lr_at(5)
