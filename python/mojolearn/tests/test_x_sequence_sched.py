# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Public schedule construction, returned metadata and optimizer connection.

Independent closed forms moved to checks/native_tests/lr_schedule_check.mojo.
These calls preserve Python constructor and scalar/array publication behavior,
plus actual schedule assignment on the public Adamax optimizer.
"""
import numpy as np

import mojolearn as ml


def test_schedule_public_scalar_and_array_results():
    # Closed-form numerical checks moved to the direct native schedule check.
    # These are the original public constructors/options and Python results.
    schedules = [ml.StepLR(0.1, 7, 0.5), ml.ExponentialLR(0.05, 0.9), ml.OneCycleLR(0.2, 40)]
    for schedule in schedules:
        value = schedule.lr_at(3)
        assert type(value) is float
        assert np.float32(value) == value
        values = schedule.lr_values(4)
        assert values.shape == (4,) and values.dtype == np.dtype("float32")


def test_schedule_drives_an_optimizer():
    p = np.ones(4, dtype=np.float32)
    opt = ml.Adamax([p])
    opt.lr_schedule = ml.ExponentialLR(1e-2, 0.5)
    for _ in range(3):
        opt.step([np.ones(4, dtype=np.float32)])
    assert opt.lr == ml.ExponentialLR(1e-2, 0.5).lr_at(3)
