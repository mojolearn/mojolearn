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


def _schedules():
    return [
        ml.ConstantLR(1e-3, warmup_steps=300),
        ml.WarmupLinearLR(1e-3, warmup_steps=100, total_steps=700, min_lr=1e-5),
        ml.WarmupCosineLR(3e-4, warmup_steps=50, total_steps=900, min_lr=3e-5),
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
