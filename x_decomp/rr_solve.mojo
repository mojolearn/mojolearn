# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE host eigh of the round-robin order (cgr-decomp, 2026-10-03): the
rounds of x_decomp/rr.mojo (`host_eigh_rr`), then `host_sign_flip` and
`eigh_ascending`. It computes the words of `DevExec.eigh` and of each
problem of `rr_batch_kernel`; the x_decomp host column and the spectral
oracle's projected solve call it. No GPU import."""
from decomposition.checks.jacobi_eigh_device import JACOBI_TOL
from decomposition.host.linalg_public import eigh_ascending
from decomposition.host.pca_oracle import host_sign_flip
from x_decomp.rr import RR_EIGH_SWEEPS, host_eigh_rr


def host_eigh_rr_sorted(mut m: List[Float32], n: Int, mut w: List[Float32], mut v: List[Float32]) raises -> Int:
    """`m` (n x n row major, consumed): w = n values ascending, v = n x n row
    major, vector c in COLUMN c. Not converged in RR_EIGH_SWEEPS raises (no
    cyclic fallback). Returns the sweeps run."""
    var vr = List[Float32](length=n * n, fill=Float32(0.0))
    var rr = host_eigh_rr(m, vr, n, RR_EIGH_SWEEPS, Float32(JACOBI_TOL))
    if not rr[0]:
        raise Error(
            "eigh: the round-robin Jacobi did not converge in " + String(RR_EIGH_SWEEPS)
            + " sweeps at n = " + String(n) + ". An unconverged decomposition is not returned"
            " as if it were one (DEVIATION 590)."
        )
    host_sign_flip(vr, n)
    var diag = List[Float32]()
    for i in range(n):
        diag.append(m[i * n + i])
    var got = eigh_ascending(diag, vr, n, True, rr[1])
    w = got.w.copy()
    v = got.v.copy()
    return rr[1]
