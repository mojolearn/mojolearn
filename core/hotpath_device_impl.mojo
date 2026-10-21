# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed launch loops of `bindings/hotpath_device.mojo`, moved here with no
Python import (lane thin-core, 2026-10-21: a pure move, no bit change)."""
from max.gpu.host import DeviceContext
from core.input_device import device_all_finite, IND_MAX_N


def device_all_finite_span(ctx: DeviceContext, addr: Int, n: Int, is_f64: Bool) raises -> Bool:
    """True when all n words at addr are finite. cpu3-bindings: spans past the
    twin's Int32 range go through in IND_MAX_N-element launches instead of
    the host loop."""
    var ok = True
    var width = 8 if is_f64 else 4
    var off = 0
    while off < n and ok:
        var m = min(IND_MAX_N, n - off)
        ok = device_all_finite(ctx, addr + off * width, m, is_f64)
        off += m
    return ok
