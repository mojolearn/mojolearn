# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of x_decomp/fa_em.mojo: the PythonObject <-> typed-value
doors that `bindings/_mojolearn_x_decomp_host.mojo` and
`bindings/_mojolearn_x_decomp.mojo` register under the same names. Each door
unpacks addresses and integers, releases the GIL and calls one typed entry
in x_decomp/fa_em.mojo; no data work happens here. Every address is a host
buffer the caller owns."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from x_decomp.api_pyapi import _f, _n
from x_decomp.exec_trait import Exec
from x_decomp.fa_em import FaArgs, dsum_f32_run, fa_em_main_check, fa_em_main_run


def _fa_args(p: PythonObject, f: PythonObject) raises -> FaArgs:
    """p = [n, d, nc, max_iter, ar] (ar: A's rows); f = [tol, llconst]."""
    return FaArgs(_n(p, 0), _n(p, 1), _n(p, 2), Int(py=p[3]), Float64(py=f[0]), Float64(py=f[1]))


def fa_em_main_py[E: Exec](
    a: PythonObject, psi: PythonObject, w: PythonObject, ll: PythonObject, p: PythonObject, f: PythonObject
) raises -> PythonObject:
    """The EM loop: a (ar x d), psi (d, in and out), w (nc x d, out), ll
    (max_iter float64, out). Returns it (the first it log-likelihoods set)."""
    var args = _fa_args(p, f)
    var ar = _n(p, 4)
    fa_em_main_check(args, ar)
    var pa = _f(a)
    var pp = _f(psi)
    var pw = _f(w)
    var pl = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=ll))
    var it = 0
    with GILReleased(Python()):
        it = fa_em_main_run[E](pa, pp, pw, pl, ar, args)
    return PythonObject(it)


def dsum_f32_py(a: PythonObject, n: PythonObject) raises -> PythonObject:
    """`_dsum` of the n float32 words at a (sequential binary64 adds)."""
    var m = Int(py=n)
    var addr = Int(py=a)
    return PythonObject(dsum_f32_run(addr, m))
