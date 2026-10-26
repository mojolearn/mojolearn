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
from x_decomp.kit_device import mcd_dev_check, mcd_dev_run


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


# TOMBSTONE: the dead x_decomp exports (registrations only where the body stays live in the decomp owners; see docs/TOMBSTONES.md) deleted 2026-10-26 by 25b5daca2; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_x_decomp.patch

