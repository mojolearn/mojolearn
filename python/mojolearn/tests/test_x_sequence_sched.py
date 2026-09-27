# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The lane's learning-rate schedules against float64 restatements of
torch.optim.lr_scheduler's closed forms (agreement with torch itself, within
float32 rounding, was checked on the lane's pod)."""
import math

import numpy as np

import mojolearn as ml


def _onecycle(step, max_lr, total, pct=0.3, div=25.0, fdiv=1e4):
    init = max_lr / div
    low = init / fdiv
    p1 = pct * total - 1
    if step <= p1:
        a, b, s, e = init, max_lr, 0.0, p1
    else:
        a, b, s, e = max_lr, low, p1, total - 1
    x = (step - s) / (e - s)
    return b + (a - b) / 2 * (math.cos(math.pi * x) + 1)


def test_closed_forms():
    st, ex, oc = ml.StepLR(0.1, 7, 0.5), ml.ExponentialLR(0.05, 0.9), ml.OneCycleLR(0.2, 40)
    for t in range(1, 41):
        e = t - 1
        assert math.isclose(st.lr_at(t), 0.1 * 0.5 ** (e // 7), rel_tol=1e-7)
        assert math.isclose(ex.lr_at(t), 0.05 * 0.9 ** e, rel_tol=1e-7)
        assert math.isclose(oc.lr_at(t), _onecycle(e, 0.2, 40), rel_tol=1e-6)
    assert np.float32(st.lr_at(3)) == st.lr_at(3)


def test_schedule_drives_an_optimizer():
    p = np.ones(4, dtype=np.float32)
    opt = ml.Adamax([p])
    opt.lr_schedule = ml.ExponentialLR(1e-2, 0.5)
    for _ in range(3):
        opt.step([np.ones(4, dtype=np.float32)])
    assert opt.lr == ml.ExponentialLR(1e-2, 0.5).lr_at(3)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
