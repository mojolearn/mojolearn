# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel.
"""Native output copies for the GIL-released CPU ANN binding."""
from std.memory import memcpy
from bindings.hostptr import f32_ptr, i32_ptr
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


