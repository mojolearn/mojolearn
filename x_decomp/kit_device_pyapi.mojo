# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of x_decomp/kit_device.mojo: the PythonObject <-> typed-value
doors that `bindings/_mojolearn_x_decomp.mojo` registers under the same
names (GPU binding only; the arguments of api.mojo's). Each door unpacks
addresses and integers, releases the GIL and calls one typed entry in
x_decomp/kit_device.mojo; no data work happens here. Every address is a
host buffer the caller owns."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from x_decomp.api_pyapi import _f, _i, _n
from x_decomp.kit_device import lda_online_dev_check, lda_online_dev_run, mcd_dev_check, mcd_dev_run


def mcd_dev_py(
    x: PythonObject, loc: PythonObject, cov: PythonObject, sup: PythonObject, dist: PythonObject,
    p: PythonObject,
) raises -> PythonObject:
    """`mcd_py` on the resident kit."""
    var q = List[Int]()
    for i in range(11):  # small-loop(the 11 plan words of p)
        q.append(Int(py=p[i]))
    mcd_dev_check(q)
    var px = _f(x)
    var pl = _f(loc)
    var pc = _f(cov)
    var ps = _i(sup)
    var pd = _f(dist)
    var n = 0
    with GILReleased(Python()):
        n = mcd_dev_run(px, pl, pc, ps, pd, q)
    return PythonObject(n)


def lda_online_dev_py(
    x: PythonObject, comps: PythonObject, exp_dir: PythonObject, p: PythonObject, f: PythonObject
) raises -> PythonObject:
    """`lda_online_py` on the resident kit."""
    var n = _n(p, 0)
    var v = _n(p, 1)
    var nc = _n(p, 2)
    var bs = _n(p, 3)
    var mdi = _n(p, 4)
    var seed = Int(py=p[5])
    var draw = Int(py=p[6])
    var nbi = Int(py=p[7])
    lda_online_dev_check(n, v, nc, bs)
    var fv = List[Float64]()
    for i in range(6):  # small-loop(the six float words of f)
        fv.append(Float64(py=f[i]))
    var px = _f(x)
    var pc = _f(comps)
    var pe = _f(exp_dir)
    with GILReleased(Python()):
        lda_online_dev_run(px, pc, pe, n, v, nc, bs, mdi, seed, draw, nbi, fv)
    return Python.tuple(draw, nbi)
