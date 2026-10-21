# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed helpers of the GPU metrics binding, Python free.

Moved word for word from `bindings/_mojolearn_metrics.mojo` (the
thin-shell rule: the shell unpacks arguments and calls typed code).
"""
from max.gpu.host import DeviceContext

from core.input_device import device_all_finite, IND_MAX_N


def device_finite_f32_list(ctx: DeviceContext, values: List[Float32]) raises -> Bool:
    """The finiteness refusal of an embedding (or a caller's initial
    embedding) as device scans of at most `IND_MAX_N` values each
    (`core/input_device`), one status word back per scan, instead of a host
    walk over every value."""
    var n = len(values)
    if n == 0:
        return True
    var ok = True
    var off = 0
    while off < n and ok:
        var m = min(IND_MAX_N, n - off)
        ok = device_all_finite(ctx, Int(values.unsafe_ptr()) + off * 4, m, False)
        off += m
    _ = len(values)
    return ok
