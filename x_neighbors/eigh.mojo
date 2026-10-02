# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`xn_eigh` on the CPU column (the host binding only): x_decomp's eigh
(`HostExec.eigh`, the pinned round-robin Jacobi of x_decomp/rr.mojo with the
cyclic fallback), the same rounds as the GPU binding's
`x_neighbors/eigh_device.mojo` (`DevExec.eigh`), so the eigenproblem is the
same words on every column (cpu-gpu-cleanup c-xneighbors, 2026-10-02; the
GPU binding ran the host cyclic Jacobi of
spectral/checks/symmetric_eig_host.mojo before). Returns 0 (the sweep count
is not reported by x_decomp's eigh)."""
from x_decomp.host import HostExec
from x_neighbors.items import FP


def op_eigh(a: Int, n: Int, evals: Int, evecs: Int) raises -> Int:
    HostExec.eigh(FP(unsafe_from_address=a), FP(unsafe_from_address=evals), FP(unsafe_from_address=evecs), n)
    return 0
