# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of LocallyLinearEmbedding's device iteration and ordered
cells (x_decomp/lle_iter_dev.mojo): the doors `bindings/_mojolearn_x_decomp.mojo`
registers. Each door unpacks addresses and integers, releases the GIL and
calls one typed entry in x_decomp/lle_iter_dev.mojo; no data work happens here."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from x_decomp.api_pyapi import _f, _i, _n
from x_decomp.lle_iter_dev import lle_gemm_ordered_run, lle_iterate_dev_run, lle_lu_ordered_run
from x_decomp.lle_iter_pyapi import _lle_args


def lle_iterate_dev_py(
    f0: PythonObject, lu: PythonObject, pm: PythonObject, im: PythonObject, z: PythonObject, outs: PythonObject,
    p: PythonObject, f: PythonObject,
) raises -> PythonObject:
    """`lle_iterate_py` on the resident kit."""
    var a = _lle_args(p, f)
    var pf = _f(f0)
    var pl = _f(lu)
    var ppm = _f(pm)
    var pim = _f(im)
    var pz = _f(z)
    var px = _f(outs[0])
    var py_ = _f(outs[1])
    var ps = _f(outs[2])
    var st = 0
    var e = 0.0
    with GILReleased(Python()):
        st = lle_iterate_dev_run(pf, pl, ppm, pim, pz, px, py_, ps, a, e)
    return Python.tuple(st, e)


def lle_gemm_ordered_py(a: PythonObject, b: PythonObject, output: PythonObject, p: PythonObject) raises -> PythonObject:
    """LLE's near-null construction/products through the existing ordered GPU cells."""
    var m = _n(p, 0)
    var kk = _n(p, 1)
    var n = _n(p, 2)
    var ta = _n(p, 3) != 0
    var tb = _n(p, 4) != 0
    var pa = _f(a)
    var pb = _f(b)
    var po = _f(output)
    with GILReleased(Python()):
        lle_gemm_ordered_run(pa, pb, po, m, kk, n, ta, tb)
    return PythonObject(0)


def lle_lu_ordered_py(a: PythonObject, piv: PythonObject, info: PythonObject, p: PythonObject) raises -> PythonObject:
    """LLE's ill-conditioned spectral transform, scalar trailing updates on GPU."""
    var n = _n(p, 0)
    var pa = _f(a)
    var pp = _i(piv)
    var pi = _f(info)
    with GILReleased(Python()):
        lle_lu_ordered_run(pa, pp, pi, n)
    return PythonObject(0)
