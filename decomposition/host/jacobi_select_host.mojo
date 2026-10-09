# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into the CPU host bindings that link glm/host/glm_oracle.mojo.
"""The host column's small symmetric eigensolver for the linear models'
Gram routes (glm/host/glm_oracle.mojo `host_lstsq_eig`, `host_svd_eig`),
switched by lane fg-linear's L1 flag together with the device sites
(glm/impl/linalg/detail/svd.mojo, lstsq.mojo, lstsq_min_norm.mojo).

Off (-D MOJOLEARN_IDN_JACOBI_ROUND_ROBIN_OFF): `host_jacobi_eigh`, the cyclic replay of
`jacobi_eigh_kernel`, unchanged. On (the IDENTICAL default since 2026-10-09):
`host_eigh_rr` (x_decomp/rr.mojo), the host replay of the device
round-robin driver `_eig_rr_device`: the same rounds, the same test, the
same budget (RR_EIGH_SWEEPS). `rel` then carries the last off-diagonal sum
(the device's info[1] under the same flag); it only feeds refusal text.
No GPU import."""
from experiments.classical_identical_ideas.fg_linear_controls import IDN_JACOBI_ROUND_ROBIN
from decomposition.host.pca_oracle import JacobiHostResult, host_jacobi_eigh
from x_decomp.cells import F32Ptr
from x_decomp.rr import RR_EIGH_SWEEPS, host_eigh_rr, rr_off_fold


def host_symmetric_eigh(mut a: List[Float32], n: Int, max_sweeps: Int, tol: Float32) -> JacobiHostResult:
    """`a` consumed in place (its diagonal the eigenvalues); the vectors in
    the result's columns (row major), as `host_jacobi_eigh` returns them."""
    comptime if IDN_JACOBI_ROUND_ROBIN:
        var v = List[Float32](length=n * n, fill=Float32(0.0))
        var got = host_eigh_rr(a, v, n, RR_EIGH_SWEEPS, tol)
        var sums = rr_off_fold(F32Ptr(unsafe_from_address=Int(a.unsafe_ptr())), n)
        return JacobiHostResult(v^, got[0], sums[0], got[1])
    else:
        return host_jacobi_eigh(a, n, max_sweeps, tol)
