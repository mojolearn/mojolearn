# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host binding's `DecompDev` (x_decomp/decomp_owner_kit.mojo): no
resident matrices; each solver is the host body the host binding registers
under that name (`ica_solve_py[HostExec]`, ...)."""
from x_decomp.cells import F32Ptr, I32Ptr
from x_decomp.decomp_owner_kit import DecompDev
from x_decomp.host import HostExec
from x_decomp.ica import ica_solve_run
from x_decomp.nmf import NmfArgs, nmf_nndsvd_run, nmf_solve_run


struct HostDecomp(DecompDev):
    @staticmethod
    def has_eigh() -> Bool:
        return False

    @staticmethod
    def eigh(a: Int, w: F32Ptr, v: F32Ptr, n: Int, uplo: Int) raises:
        raise Error("x_decomp: no resident matrices on the host binding")

    @staticmethod
    def has_lu() -> Bool:
        return False

    @staticmethod
    def lu(a: Int, lu: Int, piv: I32Ptr, info: F32Ptr, n: Int) raises:
        raise Error("x_decomp: no resident matrices on the host binding")

    @staticmethod
    def ica_solve(
        px: F32Ptr, pw: F32Ptr, nc: Int, xr: Int, xc: Int, algo: Int, fun: Int, max_iter: Int, alpha: Float64,
        tol: Float64,
    ) raises -> Int:
        return ica_solve_run[HostExec](px, pw, nc, xr, xc, algo, fun, max_iter, alpha, tol)

    @staticmethod
    def nmf_solve(px: F32Ptr, pw: F32Ptr, ph: F32Ptr, n: Int, d: Int, nc: Int, solver: Int, a: NmfArgs) raises -> Int:
        return nmf_solve_run[HostExec](px, pw, ph, n, d, nc, solver, a)

    @staticmethod
    def nmf_nndsvd(pu: F32Ptr, ps: F32Ptr, pv: F32Ptr, pw: F32Ptr, ph: F32Ptr, n: Int, d: Int, r: Int, nc: Int) raises -> Int:
        return nmf_nndsvd_run[HostExec](pu, ps, pv, pw, ph, n, d, r, nc)
