# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The lane's symmetric eigensolver: spectral/checks/symmetric_eig_host.mojo
(host cyclic Jacobi, float32 through the pinned spellings, eigenvalues
ascending by (value, index), every column's sign pinned by DEVIATION 770),
called through its entry point. THE HOST IS PART OF THE NUMERICAL PLAN: both
of the lane's bindings run this same host routine, so the eigenproblem is a
pure function of the input bits on every column."""
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from spectral.checks.symmetric_eig_host import symmetric_eig_host
from x_neighbors.fast_eigh import symmetric_eig_ql, symmetric_eig_rows
from x_neighbors.items import FP

#: lane neighbors-apple3 (2026-09-28): the same Jacobi with its rotations as
#: vectors over contiguous rows (x_neighbors/fast_eigh.mojo). OPT-IN until
#: its A/B and quality check pass: `-D MOJOLEARN_XN_EIGH_ROWS` (any mode: the
#: IDENTICAL arm shows whether its words are the scalar routine's).
comptime XN_EIGH_ROWS = is_defined["MOJOLEARN_XN_EIGH_ROWS"]()
#: FAST on Apple, OPT-IN until its A/B and quality check pass
#: (`-D MOJOLEARN_XN_EIGH_QL`): tridiagonal reduction and the QL iteration in
#: binary64 (x_neighbors/fast_eigh.mojo `symmetric_eig_ql`). FAST's words
#: move.
comptime XN_EIGH_QL = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_XN_EIGH_QL"]()
)


def op_eigh(a: Int, n: Int, evals: Int, evecs: Int) raises -> Int:
    comptime if XN_EIGH_QL:
        return symmetric_eig_ql(
            FP(unsafe_from_address=a), n, FP(unsafe_from_address=evals), FP(unsafe_from_address=evecs)
        )
    comptime if XN_EIGH_ROWS:
        return symmetric_eig_rows(
            FP(unsafe_from_address=a), n, FP(unsafe_from_address=evals), FP(unsafe_from_address=evecs)
        )
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
