# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`xn_eigh` on the GPU binding: x_decomp's device eigh (`DevExec.eigh`,
the pinned round-robin Jacobi of x_decomp/rr.mojo on the device). The CPU
column runs the same rounds (x_neighbors/eigh.mojo, `HostExec.eigh`).
cpu-gpu-cleanup c-xneighbors (2026-10-02): before, the GPU binding ran the
host cyclic Jacobi. Returns 0, as the host column does."""
from x_decomp.device import DevExec
from x_neighbors.items import FP


def op_eigh(a: Int, n: Int, evals: Int, evecs: Int) raises -> Int:
    DevExec.eigh(FP(unsafe_from_address=a), FP(unsafe_from_address=evals), FP(unsafe_from_address=evecs), n)
    return 0
