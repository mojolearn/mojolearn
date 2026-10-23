# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The doors of the decomposition owners' drivers (x_decomp/ica_owner.mojo,
...): each unpacks one list of addresses and integers (and one of floats),
releases the GIL and calls one typed driver; no data work happens here. The
native owners in bindings/ (ica_native.mojo, ...) are their only callers."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from x_decomp.cells import F32Ptr
from x_decomp.exec_trait import Exec
from x_decomp.pca_randomized import RDev
from x_decomp.decomp_owner_kit import DecompDev
from x_decomp.ica_owner import ica_fit_run, ica_project_run


def door_ptr(p: PythonObject, i: Int) raises -> F32Ptr:
    var a = Int(py=p[i])
    if a == 0:
        raise Error("x_decomp: null float32 buffer address")
    return F32Ptr(unsafe_from_address=a)


def door_dim(p: PythonObject, i: Int) raises -> Int:
    var v = Int(py=p[i])
    if v < 0 or v > 2147483647:
        raise Error("x_decomp: dimension out of range")
    return v


def ica_fit_py[E: Exec, D: RDev, RES: Bool, X: DecompDev](p: PythonObject, f: PythonObject) raises -> PythonObject:
    """FastICA fit / fit_transform. p = [x, n, d, nc, whiten (0, 1 unit,
    2 arbitrary), w_init (0: drawn), seed, algorithm, fun, max_iter,
    project, components, mixing, whitening, mean, sources]; f = [alpha,
    tol]. Returns n_iter_."""
    if len(p) != 16 or len(f) != 2:
        raise Error("x_decomp_ica_fit: params must hold the input, the options and five outputs")
    var x = door_ptr(p, 0)
    var n = door_dim(p, 1)
    var d = door_dim(p, 2)
    var nc = door_dim(p, 3)
    var whiten = Int(py=p[4])
    var winit_address = Int(py=p[5])
    var winit = F32Ptr(unsafe_from_address=winit_address)  # read only when nonzero
    var seed = Int(py=p[6])
    var algo = Int(py=p[7])
    var fun = Int(py=p[8])
    var max_iter = Int(py=p[9])
    var project = Int(py=p[10]) != 0
    var pc = door_ptr(p, 11)
    var pm = door_ptr(p, 12)
    var pw = door_ptr(p, 13)
    var pu = door_ptr(p, 14)
    var ps = door_ptr(p, 15)
    var alpha = Float64(py=f[0])
    var tol = Float64(py=f[1])
    var it = 0
    with GILReleased(Python()):
        it = ica_fit_run[E, D, RES, X](x, n, d, nc, whiten, winit, winit_address != 0, seed, algo, fun, max_iter,
                                       alpha, tol, project, pc, pm, pw, pu, ps)
    return PythonObject(it)


def ica_project_py[E: Exec, D: RDev, RES: Bool](p: PythonObject) raises -> PythonObject:
    """FastICA transform (p[0] 0) or inverse_transform (1). p = [op, x, n,
    width, matrix, rows, mean, whiten, out]."""
    if len(p) != 9:
        raise Error("x_decomp_ica_project: params must hold op, x, n, width, the matrix, rows, mean, whiten and out")
    var inverse = Int(py=p[0]) != 0
    var x = door_ptr(p, 1)
    var n = door_dim(p, 2)
    var width = door_dim(p, 3)
    var matrix = door_ptr(p, 4)
    var rows = door_dim(p, 5)
    var mean = door_ptr(p, 6)
    var whiten = Int(py=p[7]) != 0
    var out = door_ptr(p, 8)
    with GILReleased(Python()):
        ica_project_run[E, D, RES](x, n, width, inverse, matrix, rows, mean, whiten, out)
    return PythonObject(n)
