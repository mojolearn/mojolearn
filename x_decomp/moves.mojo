# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Data movement and orders the decomp lane's Python did one element at a
time (lane/py-decomp-nbrs, 2026-09-28): gathers and scatters by an int32
index list, the nonzero upper triangle of a square matrix, and the
(value, index) and (x, y, index) orders of Python's stable `sorted`.
Exact copies and compares only: no float arithmetic, so nothing here can
move a bit. Host code, compiled into both x_decomp bindings."""
from std.memory import bitcast
from std.builtin.sort import sort

from x_decomp.cells import F32Ptr, I32Ptr


def f32_key(v: Float32) raises -> UInt32:
    """v's monotone uint32 image in Python's float order (-0.0 is +0.0,
    as Python's `==` makes them). A NaN has no place in that order and is
    refused."""
    if v != v:
        raise Error("x_decomp: a NaN has no order (refused)")
    var w = v
    if w == Float32(0):
        w = Float32(0)
    var ub = bitcast[DType.uint32](w)
    return ub ^ UInt32(0xFFFFFFFF) if (ub >> 31) == 1 else ub | UInt32(0x80000000)


def gather(src: F32Ptr, idx: I32Ptr, m: Int, dst: F32Ptr):
    for a in range(m):
        dst.unsafe_store(a, src.unsafe_load(Int(idx.unsafe_load(a))))


def scatter(dst: F32Ptr, idx: I32Ptr, m: Int, src: F32Ptr):
    for a in range(m):
        dst.unsafe_store(Int(idx.unsafe_load(a)), src.unsafe_load(a))


def triu_nonzero(dis: F32Ptr, n: Int, pos: I32Ptr, mir: I32Ptr) -> Int:
    """`[i * n + j for i in range(n) for j in range(i + 1, n) if dis[i * n + j] != 0]`
    into `pos` (and each one's mirror `j * n + i` into `mir`); returns the count.
    `!= 0` is Python's: NaN is kept, -0.0 is dropped."""
    var m = 0
    for i in range(n):
        for j in range(i + 1, n):
            var v = dis.unsafe_load(i * n + j)
            if v != Float32(0):
                pos.unsafe_store(m, Int32(i * n + j))
                mir.unsafe_store(m, Int32(j * n + i))
                m += 1
    return m


def argsort_f32(x: F32Ptr, m: Int, out: I32Ptr) raises:
    """`sorted(range(m), key=lambda i: x[i])` (stable: ties in index order)."""
    var keys = List[UInt64](capacity=m)
    for i in range(m):
        keys.append((UInt64(f32_key(x.unsafe_load(i))) << 32) | UInt64(i))
    sort(keys)
    for a in range(m):
        out.unsafe_store(a, Int32(Int(keys[a] & UInt64(0xFFFFFFFF))))


def iso_order(x: F32Ptr, y: F32Ptr, xorder: I32Ptr, m: Int, out: I32Ptr) raises:
    """`sorted(range(m), key=lambda i: (x[i], y[i]))` given `xorder`, the
    stable order by x alone: each run of equal x is re-sorted by (y, index)."""
    var a = 0
    var keys = List[UInt64]()
    while a < m:
        var xa = x.unsafe_load(Int(xorder.unsafe_load(a)))
        var b = a + 1
        while b < m and x.unsafe_load(Int(xorder.unsafe_load(b))) == xa:
            b += 1
        if b - a == 1:
            out.unsafe_store(a, xorder.unsafe_load(a))
        else:
            keys.clear()
            for t in range(a, b):
                var i = Int(xorder.unsafe_load(t))
                keys.append((UInt64(f32_key(y.unsafe_load(i))) << 32) | UInt64(i))
            sort(keys)
            for t in range(b - a):
                out.unsafe_store(a + t, Int32(Int(keys[t] & UInt64(0xFFFFFFFF))))
        a = b
    for t in range(m):
        _ = f32_key(y.unsafe_load(t))          # a NaN y is refused as the full sort would meet it
