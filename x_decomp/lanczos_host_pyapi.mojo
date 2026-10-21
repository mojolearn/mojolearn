# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of x_decomp/lanczos_host.mojo: the `lanczos_py` door the
host binding registers as `x_decomp_lanczos`. It unpacks addresses and
integers, releases the GIL and calls `lanczos_run`; no data work happens
here."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from x_decomp.cells import F32Ptr
from x_decomp.lanczos_host import lanczos_run


def lanczos_py(a: PythonObject, q: PythonObject, ab: PythonObject, p: PythonObject) raises -> PythonObject:
    """`x_decomp_dev_lanczos` on host addresses; p = [n, j0, j1, cap]."""
    var n = Int(py=p[0])
    var j0 = Int(py=p[1])
    var j1 = Int(py=p[2])
    var cap = Int(py=p[3])
    var pa = F32Ptr(unsafe_from_address=Int(py=a))
    var pq = F32Ptr(unsafe_from_address=Int(py=q))
    var pab = F32Ptr(unsafe_from_address=Int(py=ab))
    var steps = 0
    with GILReleased(Python()):
        steps = lanczos_run(pa, pq, pab, n, j0, j1, cap)
    return PythonObject(steps)
