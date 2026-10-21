# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of x_decomp/fa_em_dev.mojo: the PythonObject <-> typed-value
door that `bindings/_mojolearn_x_decomp.mojo` registers under the same name.
It unpacks addresses and integers, releases the GIL and calls one typed
entry in x_decomp/fa_em_dev.mojo; no data work happens here."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from x_decomp.api_pyapi import _f, _n
from x_decomp.fa_em import fa_em_main_check
from x_decomp.fa_em_dev import fa_em_main_dev_run
from x_decomp.fa_em_pyapi import _fa_args


def fa_em_main_dev_py(
    a: PythonObject, psi: PythonObject, w: PythonObject, ll: PythonObject, p: PythonObject, f: PythonObject
) raises -> PythonObject:
    """`fa_em_main_py` on the resident kit."""
    var args = _fa_args(p, f)
    var ar = _n(p, 4)
    fa_em_main_check(args, ar)
    var pa = _f(a)
    var pp = _f(psi)
    var pw = _f(w)
    var pl = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=ll))
    var it = 0
    with GILReleased(Python()):
        it = fa_em_main_dev_run(pa, pp, pw, pl, ar, args)
    return PythonObject(it)
