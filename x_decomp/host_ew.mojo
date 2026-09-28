# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The CPU column's spelling of the elementwise cell `ew_cell`
(x_decomp/cells.mojo, DEVIATIONS 5303/5304) for the op codes built from
add, sub, the pinned product, the fused multiply-add, the flushed division
and compares (lane decomp-cpu, 2026-09-28). Host only: compiled into the
CPU host binding.

THE SAME WORDS. Each vector lane is one element, computed by the cell's own
statements lane by lane: operands flushed (`ftz`), `add`/`sub` flushed
after the IEEE operation, `mul` the pinned product (the host arm of
`pinned_mul_f32`: an arithmetic fence around `a * b`, so no neighbor add
fuses with it) flushed, `div0` the flushed IEEE division with 0 for a zero
divisor (`portable_divf`; the host binding builds IDENTICAL only), a select
for every compare, the result flushed. The op code is a compile-time
parameter, so the switch is outside the loop. A row / column / scalar
broadcast is read in place (a vector never crosses a row when an operand
is a row or column vector); the leftover elements go through `ew_cell`
itself. The transcendental ops (sqrt, exp, log, tanh, digamma, lgamma
and the Gaussian contrasts) stay on `ew_cell`.

Proof: x_decomp/checks/fold_ew_check.mojo holds HostExec.ew to the oracle
(every op code) and to `ew_cell` on every broadcast mode and a row length
that leaves a tail; x_decomp/checks/sabotage/host_ew_onemsq_fused.patch
must make it fail.
"""
from std.sys.intrinsics import llvm_intrinsic

from checks.numerics import ftz
from x_decomp.cells import (
    F32Ptr,
    OP_ABS,
    OP_ADD,
    OP_ADDS,
    OP_AXPY,
    OP_COPYB,
    OP_CUBE,
    OP_CUBEP,
    OP_DIV,
    OP_FMA,
    OP_GTS,
    OP_LE,
    OP_MAX,
    OP_MAXS,
    OP_MIN,
    OP_MINS,
    OP_MU,
    OP_MUL,
    OP_MUZ,
    OP_ONEMSQ,
    OP_RECIP,
    OP_SCALE,
    OP_SELECT,
    OP_SIGN,
    OP_SOFT,
    OP_SQ,
    OP_SQDIFF,
    OP_SUB,
    OP_SUBMUL,
    bidx,
    ew_cell,
)
from x_decomp.host_simd import W, V, ftz_v, mul_add_v

#: the fence is spelled for 8- and 16-lane vectors (x86 AVX2 / AVX-512);
#: any other width keeps every op on ew_cell
comptime EW_VEC = W == 8 or W == 16


@always_inline
def _mul(a: V, b: V) -> V:
    """`mul` (inputs already flushed): the fenced product, flushed."""
    comptime if W == 8:
        return ftz_v[W](rebind[V](llvm_intrinsic[
            "llvm.arithmetic.fence.v8f32", SIMD[DType.float32, 8], has_side_effect=False
        ](rebind[SIMD[DType.float32, 8]](a * b))))
    else:
        return ftz_v[W](rebind[V](llvm_intrinsic[
            "llvm.arithmetic.fence.v16f32", SIMD[DType.float32, 16], has_side_effect=False
        ](rebind[SIMD[DType.float32, 16]](a * b))))


@always_inline
def _add(a: V, b: V) -> V:
    return ftz_v[W](a + b)


@always_inline
def _sub(a: V, b: V) -> V:
    return ftz_v[W](a - b)


@always_inline
def _div0(a: V, b: V) -> V:
    """`div0` (inputs already flushed): 0 where b is 0, else the flushed
    IEEE quotient."""
    return b.eq(0).select(V(0), ftz_v[W](a / b))


def ew_vec_ok(op: Int) -> Bool:
    """Whether op has the vector spelling on this build."""
    comptime if not EW_VEC:
        return False
    return (
        op == OP_ADD or op == OP_SUB or op == OP_MUL or op == OP_DIV or op == OP_AXPY
        or op == OP_MAXS or op == OP_MU or op == OP_SQ or op == OP_ONEMSQ or op == OP_ABS
        or op == OP_SCALE or op == OP_FMA or op == OP_RECIP or op == OP_SOFT or op == OP_SUBMUL
        or op == OP_MINS or op == OP_COPYB or op == OP_SQDIFF or op == OP_ADDS or op == OP_GTS
        or op == OP_CUBE or op == OP_CUBEP or op == OP_MAX or op == OP_MIN or op == OP_SIGN
        or op == OP_LE or op == OP_SELECT or op == OP_MUZ
    )


@always_inline
def ew_vec[op: Int](x: V, y: V, z: V, s: V) -> V:
    """`ew_cell`'s statements for op, lane by lane (x, y, z, s flushed)."""
    var r = V(0)
    comptime if op == OP_ADD:
        r = _add(x, y)
    elif op == OP_SUB:
        r = _sub(x, y)
    elif op == OP_MUL:
        r = _mul(x, y)
    elif op == OP_DIV:
        r = _div0(x, y)
    elif op == OP_AXPY:
        r = ftz_v[W](mul_add_v[W](s, y, x))
    elif op == OP_MAXS:
        r = x.gt(s).select(x, s)
    elif op == OP_MU:
        r = _div0(_mul(x, y), _add(z, s))
    elif op == OP_SQ:
        r = _mul(x, x)
    elif op == OP_ONEMSQ:
        r = _sub(V(1), _mul(x, x))
    elif op == OP_ABS:
        r = abs(x)
    elif op == OP_SCALE:
        r = _mul(x, s)
    elif op == OP_FMA:
        r = ftz_v[W](mul_add_v[W](x, y, z))
    elif op == OP_RECIP:
        r = _div0(V(1), x)
    elif op == OP_SOFT:
        var m = _sub(abs(x), s)
        r = m.gt(0).select(x.gt(0).select(m, -m), V(0))
    elif op == OP_SUBMUL:
        r = _mul(_sub(x, y), z)
    elif op == OP_MINS:
        r = x.lt(s).select(x, s)
    elif op == OP_COPYB:
        r = y
    elif op == OP_SQDIFF:
        var t = _sub(x, y)
        r = _mul(t, t)
    elif op == OP_ADDS:
        r = _add(x, s)
    elif op == OP_GTS:
        r = x.gt(s).select(V(1), V(0))
    elif op == OP_CUBE:
        r = _mul(_mul(x, x), x)
    elif op == OP_CUBEP:
        r = _mul(V(3), _mul(x, x))
    elif op == OP_MAX:
        r = x.gt(y).select(x, y)
    elif op == OP_MIN:
        r = x.lt(y).select(x, y)
    elif op == OP_SIGN:
        r = x.gt(0).select(V(1), x.lt(0).select(V(-1), V(0)))
    elif op == OP_LE:
        r = x.le(y).select(V(1), V(0))
    elif op == OP_SELECT:
        r = x.gt(s).select(y, z)
    elif op == OP_MUZ:
        r = _mul(x, _div0(y, z.ne(0).select(z, s)))
    return ftz_v[W](r)


@always_inline
def _operand(p: F32Ptr, mode: Int, i: Int, j: Int, r: Int) -> V:
    """W consecutive elements' operand under broadcast mode (0 full, 1 row
    vector, 2 column vector, 3 scalar), flushed."""
    if mode == 0:
        return ftz_v[W](p.unsafe_load[width=W](i))
    if mode == 1:
        return ftz_v[W](p.unsafe_load[width=W](j))
    if mode == 2:
        return V(ftz(p.unsafe_load(r)))
    return V(ftz(p.unsafe_load(0)))


def ew_range_v[op: Int](
    a: F32Ptr, b: F32Ptr, bm: Int, c: F32Ptr, cm: Int, dst: F32Ptr, i0: Int, i1: Int, d: Int, s: Float32
):
    """Elements i0..i1-1 of op (row length d)."""
    var sv = V(ftz(s))
    var rowwise = bm == 1 or bm == 2 or cm == 1 or cm == 2
    var dd = d if d > 0 else max(i1, 1)  # only a row / column broadcast reads d
    var r = i0 // dd
    var j = i0 - r * dd
    var i = i0
    while i < i1:
        if i + W <= i1 and (not rowwise or j + W <= dd):
            var x = ftz_v[W](a.unsafe_load[width=W](i))
            dst.unsafe_store(i, ew_vec[op](x, _operand(b, bm, i, j, r), _operand(c, cm, i, j, r), sv))
            i += W
            j += W
        else:
            dst.unsafe_store(i, ew_cell(op, a.unsafe_load(i), b.unsafe_load(bidx(bm, i, d)), c.unsafe_load(bidx(cm, i, d)), s))
            i += 1
            j += 1
        while j >= dd:
            j -= dd
            r += 1


def ew_range(
    op: Int, a: F32Ptr, b: F32Ptr, bm: Int, c: F32Ptr, cm: Int, dst: F32Ptr, i0: Int, i1: Int, d: Int, s: Float32
):
    """Elements i0..i1-1 of `ew_cell(op, ...)`: the vector spelling when op
    has one, else the cell in a loop."""
    if not ew_vec_ok(op):
        for i in range(i0, i1):
            dst.unsafe_store(i, ew_cell(op, a.unsafe_load(i), b.unsafe_load(bidx(bm, i, d)), c.unsafe_load(bidx(cm, i, d)), s))
        return
    comptime if EW_VEC:
        if op == OP_ADD:
            ew_range_v[OP_ADD](a, b, bm, c, cm, dst, i0, i1, d, s)
        elif op == OP_SUB:
            ew_range_v[OP_SUB](a, b, bm, c, cm, dst, i0, i1, d, s)
        elif op == OP_MUL:
            ew_range_v[OP_MUL](a, b, bm, c, cm, dst, i0, i1, d, s)
        elif op == OP_DIV:
            ew_range_v[OP_DIV](a, b, bm, c, cm, dst, i0, i1, d, s)
        elif op == OP_AXPY:
            ew_range_v[OP_AXPY](a, b, bm, c, cm, dst, i0, i1, d, s)
        elif op == OP_MAXS:
            ew_range_v[OP_MAXS](a, b, bm, c, cm, dst, i0, i1, d, s)
        elif op == OP_MU:
            ew_range_v[OP_MU](a, b, bm, c, cm, dst, i0, i1, d, s)
        elif op == OP_SQ:
            ew_range_v[OP_SQ](a, b, bm, c, cm, dst, i0, i1, d, s)
        elif op == OP_ONEMSQ:
            ew_range_v[OP_ONEMSQ](a, b, bm, c, cm, dst, i0, i1, d, s)
        elif op == OP_ABS:
            ew_range_v[OP_ABS](a, b, bm, c, cm, dst, i0, i1, d, s)
        elif op == OP_SCALE:
            ew_range_v[OP_SCALE](a, b, bm, c, cm, dst, i0, i1, d, s)
        elif op == OP_FMA:
            ew_range_v[OP_FMA](a, b, bm, c, cm, dst, i0, i1, d, s)
        elif op == OP_RECIP:
            ew_range_v[OP_RECIP](a, b, bm, c, cm, dst, i0, i1, d, s)
        elif op == OP_SOFT:
            ew_range_v[OP_SOFT](a, b, bm, c, cm, dst, i0, i1, d, s)
        elif op == OP_SUBMUL:
            ew_range_v[OP_SUBMUL](a, b, bm, c, cm, dst, i0, i1, d, s)
        elif op == OP_MINS:
            ew_range_v[OP_MINS](a, b, bm, c, cm, dst, i0, i1, d, s)
        elif op == OP_COPYB:
            ew_range_v[OP_COPYB](a, b, bm, c, cm, dst, i0, i1, d, s)
        elif op == OP_SQDIFF:
            ew_range_v[OP_SQDIFF](a, b, bm, c, cm, dst, i0, i1, d, s)
        elif op == OP_ADDS:
            ew_range_v[OP_ADDS](a, b, bm, c, cm, dst, i0, i1, d, s)
        elif op == OP_GTS:
            ew_range_v[OP_GTS](a, b, bm, c, cm, dst, i0, i1, d, s)
        elif op == OP_CUBE:
            ew_range_v[OP_CUBE](a, b, bm, c, cm, dst, i0, i1, d, s)
        elif op == OP_CUBEP:
            ew_range_v[OP_CUBEP](a, b, bm, c, cm, dst, i0, i1, d, s)
        elif op == OP_MAX:
            ew_range_v[OP_MAX](a, b, bm, c, cm, dst, i0, i1, d, s)
        elif op == OP_MIN:
            ew_range_v[OP_MIN](a, b, bm, c, cm, dst, i0, i1, d, s)
        elif op == OP_SIGN:
            ew_range_v[OP_SIGN](a, b, bm, c, cm, dst, i0, i1, d, s)
        elif op == OP_LE:
            ew_range_v[OP_LE](a, b, bm, c, cm, dst, i0, i1, d, s)
        elif op == OP_SELECT:
            ew_range_v[OP_SELECT](a, b, bm, c, cm, dst, i0, i1, d, s)
        else:
            ew_range_v[OP_MUZ](a, b, bm, c, cm, dst, i0, i1, d, s)
