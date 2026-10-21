# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of x_decomp/lanczos_dev.mojo: the device Lanczos door and
the route flags the GPU binding registers. The door unpacks pool ids and
integers and calls `dev_lanczos_run`; no data work happens here."""
from std.python import PythonObject

from x_decomp.lanczos_dev import IPCA_FAST_DEV, KPCA_FAST_LANCZOS_DEV, dev_lanczos_run
from x_decomp.resident_pyapi import _id, _n


def kpca_lanczos_dev_on_py() raises -> PythonObject:
    """1 when this binding carries the device Lanczos route (every GPU build
    since lane cpu2-l8-decomp)."""
    comptime if KPCA_FAST_LANCZOS_DEV:
        return PythonObject(1)
    return PythonObject(0)


def dev_lanczos_py(a: PythonObject, q: PythonObject, ab: PythonObject, p: PythonObject) raises -> PythonObject:
    """Lanczos steps j0 .. j1-1 (see x_decomp/lanczos_dev.mojo), enqueued.
    a: n x n; q: the basis, rows 0 .. j0 hold q_0 .. q_j0 on entry, room
    for cap + 1 rows; ab: 2 cap floats. p = [n, j0, j1, cap]."""
    var n = _n(p, 0)
    var j0 = _n(p, 1)
    var j1 = _n(p, 2)
    var cap = _n(p, 3)
    var ia = _id(a)
    var iq = _id(q)
    var iab = _id(ab)
    var steps = dev_lanczos_run(ia, iq, iab, n, j0, j1, cap)
    return PythonObject(steps)


def ipca_dev_on_py() raises -> PythonObject:
    comptime if IPCA_FAST_DEV:
        return PythonObject(1)
    return PythonObject(0)
