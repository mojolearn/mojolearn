# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of the dictionary learning loops (x_decomp/dictl.mojo):
the doors `bindings/_mojolearn_x_decomp_host.mojo` registers
(`*_py[HostExec]`) and the argument unpacker the device doors share. Each
door unpacks addresses and integers, releases the GIL and calls one typed
entry in x_decomp/dictl.mojo; no data work happens here."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from providers.python.x_decomp_api_pyapi import _f, _n
from x_decomp.dictl import DlArgs, dict_learning_run, minibatch_run
from x_decomp.exec_trait import Exec


def _dl_args(p: PythonObject, f: PythonObject) raises -> DlArgs:
    """p = [n, m, nc, algo, max_iter, enc_iter, pos_dict, pos_code, seed,
    nnz (-1: None), max_no_improvement (-1: None), extra]; f = [alpha, tol]."""
    var algo = Int(py=p[3])
    var max_iter = Int(py=p[4])
    var enc_iter = Int(py=p[5])
    var pos_dict = Int(py=p[6]) != 0
    var pos_code = Int(py=p[7]) != 0
    var seed = Int(py=p[8])
    var nnz = Int(py=p[9])
    var max_no_imp = Int(py=p[10])
    var alpha = Float64(py=f[0])
    var tol = Float64(py=f[1])
    return DlArgs(algo, alpha, max_iter, tol, enc_iter, pos_dict, pos_code, seed, nnz, max_no_imp)


def dict_learning_py[E: Exec](
    x: PythonObject, code: PythonObject, d: PythonObject, errs: PythonObject, p: PythonObject, f: PythonObject
) raises -> PythonObject:
    """x (n x m); code (n x nc) and d (nc x m) the starts, replaced; errs
    (max_iter float64) the costs. Returns (ii, errors written)."""
    var n = _n(p, 0)
    var m = _n(p, 1)
    var nc = _n(p, 2)
    var a = _dl_args(p, f)
    var px = _f(x)
    var pc = _f(code)
    var pd = _f(d)
    var pe = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=errs))
    var ii = 0
    var ne = 0
    with GILReleased(Python()):
        ii = dict_learning_run[E](px, pc, pd, pe, n, m, nc, a, ne)
    return Python.tuple(ii, ne)


def minibatch_py[E: Exec](x: PythonObject, d: PythonObject, p: PythonObject, f: PythonObject) raises -> PythonObject:
    """x (n x m) the (shuffled) rows; d (nc x m) the start, replaced. p's
    extra entries: [.., bs, n_steps]. Returns the steps taken."""
    var n = _n(p, 0)
    var m = _n(p, 1)
    var nc = _n(p, 2)
    var bs = _n(p, 11)
    var n_steps = _n(p, 12)
    var a = _dl_args(p, f)
    var px = _f(x)
    var pd = _f(d)
    var steps = 0
    with GILReleased(Python()):
        steps = minibatch_run[E](px, pd, n, m, nc, bs, n_steps, a)
    return PythonObject(steps)
