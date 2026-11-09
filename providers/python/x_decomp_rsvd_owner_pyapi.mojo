# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The door of the public randomized_svd (x_decomp/randomized_svd_owner.mojo):
it unpacks addresses and integers, releases the GIL and calls one typed
driver; no data work happens here. bindings/linalg_decomp_native.mojo is
its only caller."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from x_decomp.cells import F32Ptr
from x_decomp.exec_trait import Exec
from x_decomp.pca_randomized import RDev
from x_decomp.randomized_svd_owner import randomized_svd_run


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


def randomized_svd_py[E: Exec, D: RDev, RES: Bool](
    x: PythonObject, u: PythonObject, s: PythonObject, vt: PythonObject, p: PythonObject,
) raises -> PythonObject:
    """randomized_svd's (U, S, Vt). p = [rows, cols, n_components,
    n_oversamples, n_iter, auto (1: n_iter='auto'), transpose (1: the
    sketch runs on M^T), flip_sign, the Philox seed mod 2^32] and, optional,
    a tenth word: 1 when the caller deferred M's finite test to this call
    (RSVD_FAST_DEVSCAN, `x_decomp_s_flags` bit 1; restored 2026-11-09):
    the test runs on the device copy at M's one upload."""
    if len(p) != 9 and len(p) != 10:
        raise Error("x_decomp_randomized_svd: params must contain rows, cols, components, oversamples, iterations, auto, transpose, flip_sign, seed")
    var rows = _dim(p, 0)
    var cols = _dim(p, 1)
    var nc = _dim(p, 2)
    var oversamples = Int(py=p[3])
    var iterations = Int(py=p[4])
    var auto = Int(py=p[5]) != 0
    var transpose = Int(py=p[6]) != 0
    var flip = Int(py=p[7]) != 0
    var seed = Int(py=p[8])
    var px = _ptr(x)
    var pu = _ptr(u)
    var ps = _ptr(s)
    var pv = _ptr(vt)
    var deferred = len(p) == 10 and Int(py=p[9]) != 0
    var status = List[Int32](length=1, fill=Int32(0))
    var finite_word = 0
    comptime if RES:
        if deferred:
            finite_word = Int(status.unsafe_ptr())
    else:
        if deferred:
            raise Error("x_decomp_randomized_svd: a deferred finite test needs a resident binding")
    with GILReleased(Python()):
        randomized_svd_run[E, D, RES](px, pu, ps, pv, rows, cols, nc, oversamples, iterations, auto, transpose,
                                      flip, seed, finite_word)
    if status[0] != 0:
        raise Error("M: input must be finite; NaN/inf are unsupported")
    _ = status^
    return PythonObject(nc)
