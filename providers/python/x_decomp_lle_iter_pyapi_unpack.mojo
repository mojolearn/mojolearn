# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of LocallyLinearEmbedding's subspace iteration
(x_decomp/lle_iter.mojo): the door `bindings/_mojolearn_x_decomp_host.mojo`
registers (`lle_iterate_py[HostExec]`) and the argument unpacker the device
door shares. Each door unpacks addresses and integers, releases the GIL and
calls one typed entry in x_decomp/lle_iter.mojo; no data work happens here."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from providers.python.x_decomp_api_pyapi_unpack import _f, _n
from x_decomp.exec_trait import Exec
from x_decomp.lle_iter import LleArgs, lle_iterate_run


def _lle_args(p: PythonObject, f: PythonObject) raises -> LleArgs:
    """p = [n, n1, nc, p, max_iter, seed]; f = [floor, sub_tol, stall_tol]."""
    return LleArgs(_n(p, 0), _n(p, 1), _n(p, 2), _n(p, 3), Int(py=p[4]), Int(py=p[5]), Float64(py=f[0]),
                   Float64(py=f[1]), Float64(py=f[2]))
