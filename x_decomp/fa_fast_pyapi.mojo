# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of x_decomp/fa_fast.mojo: the PythonObject <-> typed-value
doors that `bindings/_mojolearn_x_decomp.mojo` registers under the same
names. Each door unpacks resident matrix ids, host addresses and integers
and calls one typed entry in x_decomp/fa_fast.mojo; no data work happens
here."""
from std.python import PythonObject

from x_decomp.cells import F32Ptr
from x_decomp.fa_fast import (
    fa_defines_run, fa_em_run, fa_gram_check, fa_gram_run, fa_transform_check, fa_transform_run,
)
from x_decomp.resident_pyapi import _id, _n


def fa_gram_py(x: PythonObject, mean: PythonObject, g: PythonObject, var_: PythonObject, p: PythonObject) raises -> PythonObject:
    """p = [n, d]: G (d x d, device id g) and var (d, device id var_) of the
    resident X (n x d) centered at mean (d); enqueued, no sync. Partials from
    the pool, freed at once (the context runs in order). FA_GRAM_DF: G in
    double-float, g holds 2 d x d floats (hi words, then lo words)."""
    fa_gram_check()
    var n = _n(p, 0)
    var d = _n(p, 1)
    return PythonObject(fa_gram_run(_id(x), _id(mean), _id(g), _id(var_), n, d))


def fa_em_py(
    g: PythonObject, psi0: PythonObject, w_out: PythonObject, psi_out: PythonObject,
    ll_out: PythonObject, p: PythonObject, tol: PythonObject,
) raises -> PythonObject:
    """FactorAnalysis.fit's EM loop on the resident Gram: p = [d, nc, n,
    max_iter]; g the device id of G (d x d), psi0 host floats (d),
    w_out host floats (nc x d), psi_out host floats (d), ll_out host float64
    (max_iter). Returns the iterations run (len(loglike_))."""
    var d = _n(p, 0)
    var nc = _n(p, 1)
    var n = _n(p, 2)
    var max_iter = _n(p, 3)
    var gid = _id(g)
    var psi_src = F32Ptr(unsafe_from_address=Int(py=psi0))
    var w_dst = F32Ptr(unsafe_from_address=Int(py=w_out))
    var psi_dst = F32Ptr(unsafe_from_address=Int(py=psi_out))
    var ll_dst = MutPointer[Float64, MutAnyOrigin](unsafe_from_address=Int(py=ll_out))
    var tol64 = Float64(py=tol)
    return PythonObject(fa_em_run(gid, psi_src, w_dst, psi_dst, ll_dst, d, nc, n, max_iter, tol64))


def fa_transform_py(x: PythonObject, mean: PythonObject, pm: PythonObject, dst: PythonObject, p: PythonObject) raises -> PythonObject:
    """p = [n, d, nc]: dst (n x nc, device id) = (X - mean) P for the
    resident X (n x d), mean (d) and P (d x nc); enqueued, no sync."""
    fa_transform_check()
    var n = _n(p, 0)
    var d = _n(p, 1)
    var nc = _n(p, 2)
    return PythonObject(fa_transform_run(_id(x), _id(mean), _id(pm), _id(dst), n, d, nc))


def fa_defines_py() raises -> PythonObject:
    """The FactorAnalysis switches this FAST Apple binding was built with
    (`-D MOJOLEARN_FA_...`), comma-joined, so python/mojolearn/
    _expansion_decomp.py `_fa_fast_define` picks a route without an env read.
    Registered only under FA_FAST_APPLE (bindings/_mojolearn_x_decomp.mojo)."""
    return PythonObject(fa_defines_run())
