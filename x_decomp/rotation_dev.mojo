# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""x_decomp/rotation.mojo's varimax / quartimax rotation on resident device
matrices (lane py-runtime-b): the same statements on `DKit`; each scalar
Python read (max |A|, the singular-value sum) one word home. GPU binding only."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from x_decomp.api import _f, _n
from x_decomp.kit import Mat, OP_CUBE, OP_DIV, OP_MUL, OP_SCALE, OP_SQ, OP_SQRT, OP_SUB, mat_eye, mat_from
from x_decomp.kit_device import DKit, DMat
from x_decomp.rotation import polar_exp, pow2
from x_decomp.select_ops import SEL_MAXABS


def polar_dev(mut k: DKit, A0: DMat, mut sum_sv: Float64) raises -> DMat:
    var A = k.copy(A0)
    var m = k.word(k.reduce(A, SEL_MAXABS)) if A.n() > 0 else 0.0
    var e = polar_exp(m)
    if e != 0:
        A = k.ew1(OP_SCALE, A, pow2(-e))
    var eg = k.eigh(k.mm(A, A, True, False))
    var sv = k.ew1(OP_SQRT, eg.wd, 0.0)
    var AV = k.ew2(OP_DIV, k.mm(A, eg.vd, False, False), sv)
    sum_sv = k.word(k.total(sv)) * pow2(e)
    return k.mm(AV, eg.vd, False, True)


def ortho_rotation_dev_py(c: PythonObject, out_t: PythonObject, p: PythonObject, f: PythonObject) raises -> PythonObject:
    """`ortho_rotation_py` on the resident kit."""
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
        var k = DKit()
        var C = k.upload(mat_from(pc, nrow, ncol))
        var R = k.upload(mat_eye(ncol))
        var var_ = 0.0
        for _ in range(max_iter):
            var cr = k.mm(C, R, False, False)
            var target: DMat
            if vm:
                var tmp = k.ew2(OP_MUL, cr, k.ew1(OP_SCALE, k.colsum(k.ew1(OP_SQ, cr, 0.0)), 1.0 / Float64(nrow)))
                target = k.ew2(OP_SUB, k.ew1(OP_CUBE, cr, 0.0), tmp)
            else:
                target = k.ew1(OP_CUBE, cr, 0.0)
            var var_new = 0.0
            R = polar_dev(k, k.mm(C, target, True, False), var_new)
            if var_ != 0 and var_new < var_ * (1 + tol):
                break
            var_ = var_new
        var T = k.t(k.mm(C, R, False, False))
        var h = k.get(T)
        k.sync()
        for i in range(h.n()):
            po.unsafe_store(i, h.d[i])
    return PythonObject(ncol)
