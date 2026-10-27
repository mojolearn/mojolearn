# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The ann lane's typed address contract, shared by `x_ann/resident.mojo` and,
through `providers/python/x_ann_abi_pyapi.mojo`, by `bindings/_mojolearn_x_ann.mojo` and
`bindings/_mojolearn_x_ann_host.mojo`: plain copies, no arithmetic. No
Python here; the PythonObject unpacking is `providers/python/x_ann_abi_pyapi.mojo`."""

from std.memory import memcpy
from core.hostptr import f32_ptr, i32_ptr
from x_ann.switches import ANN3_HOST_PASSES


def out_f32_to(values: List[Float32], addr: Int) raises:
    """Into the caller's array at `addr`. Under `ANN3_HOST_PASSES` (lane
    ann-apple3) one memcpy; one store per word otherwise (the IVF code
    arrays are n x dim words)."""
    var p = f32_ptr(addr)
    comptime if ANN3_HOST_PASSES:
        if len(values) > 0:
            memcpy(dest=p, src=values.unsafe_ptr(), count=len(values))
    else:
        for e in range(len(values)):
            p.unsafe_store(e, values[e])


def out_i32_to(values: List[Int32], addr: Int) raises:
    var p = i32_ptr(addr)
    comptime if ANN3_HOST_PASSES:
        if len(values) > 0:
            memcpy(dest=p, src=values.unsafe_ptr(), count=len(values))
    else:
        for e in range(len(values)):
            p.unsafe_store(e, values[e])


def check_search(n_lists: Int, m: Int, k: Int, n_probes: Int) raises:
    if m <= 0 or k <= 0:
        raise Error("x_ann: search needs at least one query and k >= 1")
    if n_probes <= 0 or n_probes > n_lists:
        raise Error("x_ann: n_probes must be in [1, n_lists]; it is not clamped")
