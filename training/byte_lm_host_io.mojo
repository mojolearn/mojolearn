# SPDX-License-Identifier: Apache-2.0
"""Typed element work behind the CPU byte-LM binding (`bindings/_mojolearn_byte_lm_host.mojo`).

HOST ONLY: no DeviceContext, no kernel. The binding unpacks its arguments,
calls one of these, and returns the result."""
from std.math import isfinite


def byte_host_store_f32(out: MutPointer[Float32, MutUntrackedOrigin], values: List[Float32]):
    """Write every element of `values` to `out`, in order."""
    for i in range(len(values)):
        out.unsafe_store(i, values[i])


def byte_host_store_i32(out: MutPointer[Int32, MutUntrackedOrigin], values: List[Int32], count: Int):
    """Write the first `count` elements of `values` to `out`, in order."""
    for i in range(count):
        out.unsafe_store(i, values[i])


def all_finite_f32_scan(p: MutPointer[Float32, MutUntrackedOrigin], count: Int) -> Int:
    """1 if every one of the `count` values at `p` is finite, else 0 (first failure exits)."""
    var ok: Int = 1
    for i in range(count):
        if not isfinite(p.unsafe_load(i)):
            ok = 0
            break
    return ok


def all_finite_f64_scan(p: MutPointer[Float64, MutUntrackedOrigin], count: Int) -> Int:
    """`all_finite_f32_scan` over float64 values."""
    var ok: Int = 1
    for i in range(count):
        if not isfinite(p.unsafe_load(i)):
            ok = 0
            break
    return ok


def cast_f64_to_f32_rows(sp: MutPointer[Float64, MutUntrackedOrigin],
                         dp: MutPointer[Float32, MutUntrackedOrigin], count: Int):
    """Write `Float32(sp[i])` to `dp[i]` for `i` in `[0, count)`, eight at a time then the tail."""
    var i = 0
    var body = count - (count % 8)
    while i < body:
        var v = sp.unsafe_load[width=8](i)
        dp.unsafe_store[width=8](i, v.cast[DType.float32]())
        i += 8
    while i < count:
        dp.unsafe_store(i, sp.unsafe_load(i).cast[DType.float32]())
        i += 1
