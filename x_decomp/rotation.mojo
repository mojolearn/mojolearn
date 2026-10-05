# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FactorAnalysis's varimax / quartimax rotation in Mojo (lane py-runtime-b,
2026-10-05): `_expansion_decomp._ortho_rotation` and `_polar`, statement for
statement on `Kit[E]`: the same cells, the power-of-two prescale (Python's
frexp / ldexp, exact) and the float64 stopping test in the same order, so the
IDENTICAL words are the Python driver's on every column.
x_decomp/rotation_dev.mojo runs the same statements on resident matrices."""
from std.math import isfinite
from std.memory import bitcast
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from x_decomp.api import _f, _n
from x_decomp.exec_trait import Exec
from x_decomp.kit import (
    Kit, Mat, OP_CUBE, OP_DIV, OP_MUL, OP_SCALE, OP_SQ, OP_SQRT, OP_SUB, mat_eye, mat_from, mat_t,
)
from x_decomp.select_ops import SEL_MAXABS


def frexp_exp(m: Float64) -> Int:
    """`math.frexp(m)[1]` of a positive finite normal binary64 m (every
    float32 value is one): m = f 2^e with 0.5 <= f < 1."""
    var bits = bitcast[DType.uint64](m)
    return Int((bits >> 52) & 0x7FF) - 1022


def pow2(e: Int) -> Float64:
    """`math.ldexp(1.0, e)` for a normal binary64 power of two."""
    return bitcast[DType.float64](UInt64(1023 + e) << 52)


def polar_exp(m: Float64) -> Int:
    """`_polar`'s scale exponent: frexp(m)[1] clamped to [-120, 120], 0 when
    m is not a positive finite value."""
    if m > 0.0 and isfinite(m):
        return max(-120, min(120, frexp_exp(m)))
    return 0


def polar[E: Exec](k: Kit[E], A0: Mat, mut sum_sv: Float64) raises -> Mat:
    """`_polar`: U V^T of A's SVD through the eigh of A^T A, and the sum of
    the singular values, scaled back by 2^e."""
    var A = A0.copy()
    var m = k.word(k.reduce(A, SEL_MAXABS)) if A.n() > 0 else 0.0
    var e = polar_exp(m)
    if e != 0:
        A = k.ew1(OP_SCALE, A, pow2(-e))
    var w = Mat(1, A.c)
    var V = Mat(A.c, A.c)
    k.eigh(k.mm(A, A, True, False), w, V)
    var sv = k.ew1(OP_SQRT, w, 0.0)
    var AV = k.ew2(OP_DIV, k.mm(A, V, False, False), sv)
    sum_sv = k.word(k.total(sv)) * pow2(e)
    return k.mm(AV, V, False, True)


def ortho_rotation[E: Exec](C: Mat, varimax: Bool, tol: Float64, max_iter: Int) raises -> Mat:
    """`_ortho_rotation(k, C, method, tol, max_iter)`: (C R)^T."""
    var k = Kit[E]()
    var nrow = C.r
    var R = mat_eye(C.c)
    var var_ = 0.0
    for _ in range(max_iter):
        var cr = k.mm(C, R, False, False)
        var target: Mat
        if varimax:
            var tmp = k.ew2(OP_MUL, cr, k.ew1(OP_SCALE, k.colsum(k.ew1(OP_SQ, cr, 0.0)), 1.0 / Float64(nrow)))
            target = k.ew2(OP_SUB, k.ew1(OP_CUBE, cr, 0.0), tmp)
        else:
            target = k.ew1(OP_CUBE, cr, 0.0)
        var var_new = 0.0
        R = polar(k, k.mm(C, target, True, False), var_new)
        if var_ != 0 and var_new < var_ * (1 + tol):
            break
        var_ = var_new
    return mat_t(k.mm(C, R, False, False))


def ortho_rotation_py[E: Exec](c: PythonObject, out_t: PythonObject, p: PythonObject, f: PythonObject) raises -> PythonObject:
    """c (nrow x ncol); out_t (ncol x nrow) = (C R)^T. p = [nrow, ncol,
    varimax, max_iter]; f = [tol]."""
    var nrow = _n(p, 0)
    var ncol = _n(p, 1)
    if nrow * ncol > 2147483647 or ncol < 1:
        raise Error("x_decomp: rotation shape out of range")
    var vm = Int(py=p[2]) != 0
    var max_iter = Int(py=p[3])
    var tol = Float64(py=f[0])
    var pc = _f(c)
    var po = _f(out_t)
    with GILReleased(Python()):
        var T = ortho_rotation[E](mat_from(pc, nrow, ncol), vm, tol, max_iter)
        for i in range(T.n()):
            po.unsafe_store(i, T.d[i])
    return PythonObject(ncol)
