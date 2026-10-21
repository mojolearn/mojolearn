# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of x_decomp/lda_fit.mojo: the PythonObject <-> typed-value
door that the host binding registers under the same name, and the `LdaArgs`
unpacker x_decomp/lda_fit_dev_pyapi.mojo shares. Each door unpacks addresses,
integers and scalars, releases the GIL and calls one typed entry in
x_decomp/lda_fit.mojo; no data work happens here."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from x_decomp.api_pyapi import _f, _n
from x_decomp.exec_trait import Exec
from x_decomp.lda_fit import LdaArgs, lda_fit_run


def _lda_args(p: PythonObject, f: PythonObject) raises -> LdaArgs:
    """p = [n, v, nc, max_iter, online, bs, mdi, seed, draw, nbi,
    evaluate_every]; f = [doc_prior, topic_prior, offset, decay,
    mean_change_tol, total_samples, perp_tol]."""
    var max_iter = Int(py=p[3])
    var online = Int(py=p[4]) != 0
    var bs = Int(py=p[5])
    var mdi = Int(py=p[6])
    var seed = Int(py=p[7])
    var evaluate_every = Int(py=p[10])
    var doc_prior = Float64(py=f[0])
    var topic_prior = Float64(py=f[1])
    var offset = Float64(py=f[2])
    var decay = Float64(py=f[3])
    var tol = Float64(py=f[4])
    var total = Float64(py=f[5])
    var perp_tol = Float64(py=f[6])
    return LdaArgs(
        max_iter, online, bs, mdi, seed, evaluate_every, doc_prior, topic_prior, offset, decay, tol, total, perp_tol
    )


def lda_fit_py[E: Exec](
    x: PythonObject, comps: PythonObject, exp_dir: PythonObject, p: PythonObject, f: PythonObject
) raises -> PythonObject:
    """x (n x v); comps and exp_dir (nc x v) replaced. Returns (n_iter,
    draw, n_batch_iter)."""
    var n = _n(p, 0)
    var v = _n(p, 1)
    var nc = _n(p, 2)
    var draw = Int(py=p[8])
    var nbi = Int(py=p[9])
    var a = _lda_args(p, f)
    var px = _f(x)
    var pc = _f(comps)
    var pe = _f(exp_dir)
    var it = 0
    with GILReleased(Python()):
        it = lda_fit_run[E](px, pc, pe, n, v, nc, a, draw, nbi)
    return Python.tuple(it, draw, nbi)
