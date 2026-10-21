# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Exact select reductions and a small stable order (lane cpu2-l8-decomp,
2026-10-04, re-audit L8): the scalar decisions python/mojolearn/_expansion_decomp.py
took over downloaded vectors (max |w| of a spectrum, the first live column,
the count of kept singular values, the CD shuffle order) are made here, and on
the device by x_decomp/select_dev.mojo with the SAME per-element functions.

A max or a min is exact and, with ties to the lower index, does not depend
on the order it is taken in: the device's two-level fold (slices of
SEL_SLICE values, then the slice results in slice order) gives the word the
host's single pass gives, signed zeros and NaN payloads included (a NaN
anywhere yields the first NaN by index). The order is a rank count, an exact
integer per element. Nothing here rounds, so the host column and every
vendor agree by construction.

This module imports no device code: bindings/_mojolearn_x_decomp_host.mojo
registers `reduce_py` and `order_small_py` from this module's sibling
shell, the GPU binding registers them too (host
buffers) beside the device entries."""

from x_decomp.cells import F32Ptr

comptime SEL_MAXABS = 0
comptime SEL_MAX = 1
comptime SEL_MIN = 2
comptime SEL_LAST = SEL_MIN
#: values one device thread folds per slice
comptime SEL_SLICE = 256
#: the largest vector `order_small` takes (its rank count is O(n^2))
comptime SEL_ORDER_MAX = 1 << 16


@always_inline
def sel_val(op: Int, x: Float32) -> Float32:
    if op == SEL_MAXABS:
        return abs(x)
    return x


@always_inline
def sel_fold(op: Int, src: F32Ptr, q0: Int, q1: Int) -> Float32:
    """The op over src[q0 .. q1) (q1 > q0): the first NaN by index when one
    is there, else the first extreme by index."""
    var best = sel_val(op, src.unsafe_load(q0))
    if best != best:
        return best
    for q in range(q0 + 1, q1):
        var v = sel_val(op, src.unsafe_load(q))
        if v != v:
            return v
        if op == SEL_MIN:
            if v < best:
                best = v
        elif v > best:
            best = v
    return best


@always_inline
def order_less(a: Float32, i: Int, b: Float32, j: Int) -> Bool:
    """(a, i) before (b, j): by value ascending, NaN after every number,
    ties (and two NaN) to the lower index; -0 and +0 tie, as Python's sort."""
    var an = a != a
    var bn = b != b
    if an or bn:
        if an and bn:
            return i < j
        return bn
    if a < b:
        return True
    if a == b:
        return i < j
    return False


@always_inline
def order_rank(src: F32Ptr, n: Int, i: Int) -> Int:
    """The position of element i in the stable ascending order."""
    var a = src.unsafe_load(i)
    var r = 0
    for j in range(n):
        if j != i and order_less(src.unsafe_load(j), j, a, i):
            r += 1
    return r


def reduce_run(src: F32Ptr, dst: F32Ptr, count: Int, op: Int) raises:
    """`reduce_py`'s typed body: dst[0] = the op over src[0 .. count)."""
    if count < 1:
        raise Error("x_decomp: reduce needs at least one value")
    if op < 0 or op > SEL_LAST:
        raise Error("x_decomp: unknown reduce op")
    dst.unsafe_store(0, sel_fold(op, src, 0, count))


def order_small_run(src: F32Ptr, dst: F32Ptr, n: Int) raises:
    """`order_small_py`'s typed body: out = the stable ascending order of src."""
    if n > SEL_ORDER_MAX:
        raise Error("x_decomp: order_small exceeds its bound")
    for i in range(n):
        dst.unsafe_store(order_rank(src, n, i), Float32(i))
