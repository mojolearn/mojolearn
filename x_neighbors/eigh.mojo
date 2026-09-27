# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The lane's symmetric eigensolver: spectral/checks/symmetric_eig_host.mojo
(host cyclic Jacobi, float32 through the pinned spellings, eigenvalues
ascending by (value, index), every column's sign pinned by DEVIATION 770),
called through its entry point. THE HOST IS PART OF THE NUMERICAL PLAN: both
of the lane's bindings run this same host routine, so the eigenproblem is a
pure function of the input bits on every column."""
from spectral.checks.symmetric_eig_host import symmetric_eig_host
from x_neighbors.items import FP


def op_eigh(a: Int, n: Int, evals: Int, evecs: Int) raises -> Int:
    var src = FP(unsafe_from_address=a)
    var m = List[Float32](capacity=n * n)
    for i in range(n * n):
        m.append(src.unsafe_load(i))
    var w = List[Float32](length=n, fill=Float32(0))
    var v = List[Float32](length=n * n, fill=Float32(0))
    var sweeps = symmetric_eig_host[DType.float32](m, n, w, v)
    var ow = FP(unsafe_from_address=evals)
    var ov = FP(unsafe_from_address=evecs)
    for i in range(n):
        ow.unsafe_store(i, w[i])
    for i in range(n * n):
        ov.unsafe_store(i, v[i])
    return sweeps
