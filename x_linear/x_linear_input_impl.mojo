# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The GPU binding's input check (`bindings/_mojolearn_x_linear.mojo`), moved
here with no Python import (lane thin-core, 2026-10-21: a pure move)."""
from x_linear.device import linear_ctx as _p2m_ctx
from x_linear.ops import FP
from core.input_device import device_all_finite, IND_MAX_N


def x_linear_finite_device(p: FP, count: Int, name: String) raises:
    """The input check both columns run before a fit: NaN or infinity is
    refused by name. cpu3-bindings: the routes whose grids do not test the
    uploaded words themselves (FAST, or the IDENTICAL defines off) test the
    caller's words with one device scan (`core/input_device`, one status
    word back per launch) instead of a host walk over every value."""
    if count <= 0:
        return
    var ctx = _p2m_ctx()
    var off = 0
    while off < count:
        var m = min(IND_MAX_N, count - off)
        if not device_all_finite(ctx, Int(p) + off * 4, m, False):
            raise Error(String("mojolearn: ", name, " contains NaN or infinity"))
        off += m
