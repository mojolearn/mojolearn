# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Re-export. The spectral host oracle SHIPS in the metrics CPU host binding
(python/mojolearn/host_surface.py names it), so it lives in
`spectral/host/spectral_oracle.mojo` since workstream E batch 2,
2026-09-14. spectral/checks/spectral_check.mojo keeps resolving through
this file; new code imports `spectral.host.spectral_oracle` directly. The
dense Float64 cross-check stays here: it imports
decomposition/checks/jacobi_eigh.mojo, which the host binding does not
carry."""

from decomposition.checks.jacobi_eigh import jacobi_eigh
from spectral.host.spectral_oracle import (
    HostLaplacian,
    LANCZOS_ALPHA_CLAMP,
    LANCZOS_BETA_CLAMP,
    LANCZOS_U_CLAMP,
    OracleResult,
    host_dot,
    host_fold_tree,
    host_laplacian,
    host_lanczos_smallest,
    host_leaf_partial,
    host_spmv,
    lanczos_v0,
    oracle_embedding,
)
from spectral.impl.sparse.coo import CooGraph


# ---------------------------------------------------------------------------
# The dense Float64 cross-check
# ---------------------------------------------------------------------------


def dense_laplacian_eigenvalues_f64(g: CooGraph, norm_laplacian: Bool) raises -> List[Float64]:
    """The Float64 Laplacian of `g` (NOT negated: these are the eigenvalues
    of `L`, ascending) densified and handed to `decomposition`'s Float64
    Jacobi. `n <= 64` only; a third, independent opinion on the spectrum."""
    var n = g.n
    if n > 64:
        raise Error("dense_laplacian_eigenvalues_f64: n <= 64 only")
    var L = host_laplacian[DType.float64](g, norm_laplacian)
    var a = List[Float64]()
    var vecs = List[Float64]()
    for _ in range(n * n):
        a.append(0.0)
        vecs.append(0.0)
    for i in range(len(L.vals)):
        a[Int(L.rows[i]) * n + Int(L.cols[i])] = -L.vals[i]
    jacobi_eigh(a, vecs, n)
    var d = List[Float64]()
    for i in range(n):
        d.append(a[i * n + i])
    for i in range(1, n):
        var key = d[i]
        var j = i - 1
        while j >= 0 and d[j] > key:
            d[j + 1] = d[j]
            j -= 1
        d[j + 1] = key
    return d^
