# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of x_decomp/mds.mojo: the PythonObject <-> typed-value
door that `bindings/_mojolearn_x_decomp_host.mojo` registers under the same
name. It unpacks addresses and integers, releases the GIL and calls one
typed entry in x_decomp/mds.mojo; no data work happens here."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from providers.python.x_decomp_api_pyapi_unpack import _f, _n
from x_decomp.exec_trait import Exec
from x_decomp.mds import MdsArgs, mds_fit_check, mds_fit_run


def _mds_args(p: PythonObject, f: PythonObject) raises -> MdsArgs:
    """p = [n, nc, n_starts, max_iter, metric, norm]; f = [eps]."""
    return MdsArgs(Int(py=p[3]), Int(py=p[4]) != 0, Int(py=p[5]) != 0, Float64(py=f[0]))
