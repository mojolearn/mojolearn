# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of x_decomp/mds_dev.mojo: the PythonObject <-> typed-value
door that `bindings/_mojolearn_x_decomp.mojo` registers under the same name.
It unpacks addresses and integers, releases the GIL and calls one typed
entry in x_decomp/mds_dev.mojo; no data work happens here."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from providers.python.x_decomp_api_pyapi import _f, _n
from x_decomp.mds import mds_fit_check
from x_decomp.mds_dev import mds_fit_dev_run
from providers.python.x_decomp_mds_pyapi import _mds_args


def mds_fit_dev_py(
    dis: PythonObject, starts: PythonObject, y: PythonObject, p: PythonObject, f: PythonObject
) raises -> PythonObject:
    """`mds_fit_py` on the resident kit."""
    var n = _n(p, 0)
    var nc = _n(p, 1)
    var ns = _n(p, 2)
    mds_fit_check(n, nc, ns)
    var a = _mds_args(p, f)
    var pd = _f(dis)
    var ps = _f(starts)
    var py_ = _f(y)
    var best_it = 0
    var best_s = 0.0
    with GILReleased(Python()):
        best_it = mds_fit_dev_run(pd, ps, py_, n, nc, ns, a, best_s)
    return Python.tuple(best_s, best_it)
