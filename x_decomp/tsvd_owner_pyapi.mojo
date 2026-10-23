# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The doors of TruncatedSVD's decomp-kit routes (x_decomp/tsvd_owner.mojo):
each unpacks addresses and integers, releases the GIL and calls one typed
driver; no data work happens here. The native TruncatedSVD owner
(bindings/tsvd_native.mojo) is their only caller."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from x_decomp.cells import F32Ptr
from x_decomp.exec_trait import Exec
from x_decomp.pca_randomized import RDev
from x_decomp.tsvd_owner import tsvd_randomized_fit_run, tsvd_tsqr_components_run


def _ptr(address: PythonObject) raises -> F32Ptr:
    var a = Int(py=address)
    if a == 0:
        raise Error("x_decomp: null float32 buffer address")
    return F32Ptr(unsafe_from_address=a)


def _dim(p: PythonObject, i: Int) raises -> Int:
    var v = Int(py=p[i])
    if v < 0 or v > 2147483647:
        raise Error("x_decomp: dimension out of range")
    return v


def tsvd_randomized_py[E: Exec, D: RDev, RES: Bool](
    x: PythonObject, components: PythonObject, singular: PythonObject, variance: PythonObject,
    ratio: PythonObject, p: PythonObject,
) raises -> PythonObject:
    """TruncatedSVD(algorithm='randomized').fit's arrays. p = [rows, cols,
    n_components, n_oversamples, n_iter, auto (1: n_iter='auto'), the
    Philox seed mod 2^32]."""
    if len(p) != 7:
        raise Error("x_decomp_tsvd_randomized: params must contain rows, cols, components, oversamples, iterations, auto, seed")
    var rows = _dim(p, 0)
    var cols = _dim(p, 1)
    var nc = _dim(p, 2)
    var oversamples = Int(py=p[3])
    var iterations = Int(py=p[4])
    var auto = Int(py=p[5]) != 0
    var seed = Int(py=p[6])
    var px = _ptr(x)
    var pc = _ptr(components)
    var ps = _ptr(singular)
    var pv = _ptr(variance)
    var pr = _ptr(ratio)
    with GILReleased(Python()):
        tsvd_randomized_fit_run[E, D, RES](px, pc, ps, pv, pr, rows, cols, nc, oversamples, iterations, auto, seed)
    return PythonObject(nc)


def tsvd_tsqr_py[E: Exec, D: RDev, RES: Bool](
    x: PythonObject, components: PythonObject, singular: PythonObject, p: PythonObject
) raises -> PythonObject:
    """TruncatedSVD's TSVD_QFIX components and singular values of a tall x.
    p = [rows, cols, n_components]."""
    if len(p) != 3:
        raise Error("x_decomp_tsvd_tsqr: params must contain rows, cols, components")
    var rows = _dim(p, 0)
    var cols = _dim(p, 1)
    var nc = _dim(p, 2)
    var px = _ptr(x)
    var pc = _ptr(components)
    var ps = _ptr(singular)
    with GILReleased(Python()):
        tsvd_tsqr_components_run[E, D, RES](px, pc, ps, rows, cols, nc)
    return PythonObject(nc)
