# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python door of `x_decomp/pca_randomized.mojo`: unpack, one typed call."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from x_decomp.cells import F32Ptr
from x_decomp.exec_trait import Exec
from x_decomp.pca_randomized import RDev, pca_randomized_fit_run


def _out(addr: PythonObject) raises -> F32Ptr:
    var a = Int(py=addr)
    if a == 0:
        raise Error("x_decomp: null float32 buffer address")
    return F32Ptr(unsafe_from_address=a)


def pca_randomized_py[E: Exec, D: RDev, RES: Bool](
    x: PythonObject, components: PythonObject, mean: PythonObject, variance: PythonObject,
    ratio: PythonObject, singular: PythonObject, p: PythonObject, seed: PythonObject,
) raises -> PythonObject:
    """`PCA(svd_solver='randomized').fit` in one call. p = [n_rows,
    n_features, n_components, n_oversamples, power iterations, auto (1:
    iterated_power='auto')]; seed is the Philox seed mod 2^32. Writes
    components_ (n_components x n_features), mean_ (n_features),
    explained_variance_, explained_variance_ratio_ and singular_values_
    (n_components each); returns the noise variance."""
    if len(p) != 6:
        raise Error("x_decomp_pca_randomized: p must contain rows, features, components, oversamples, iterations, auto")
    var px = _out(x)
    var pc = _out(components)
    var pm = _out(mean)
    var pv = _out(variance)
    var pr = _out(ratio)
    var ps = _out(singular)
    var rows = Int(py=p[0])
    var cols = Int(py=p[1])
    var nc = Int(py=p[2])
    var oversamples = Int(py=p[3])
    var iterations = Int(py=p[4])
    var auto = Int(py=p[5]) != 0
    var s = Int(py=seed)
    var noise = Float64(0)
    with GILReleased(Python()):
        noise = pca_randomized_fit_run[E, D, RES](px, pc, pm, pv, pr, ps, rows, cols, nc, oversamples, iterations, auto, s)
    return PythonObject(noise)
