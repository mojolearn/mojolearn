# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The GPU binding's `DecompDev` (x_decomp/decomp_owner_kit.mojo): the
resident eigh / LU where the build registers `x_decomp_dev_eigh` /
`x_decomp_dev_lu` (IDN_EIGH_RESIDENT, IDN_LU_RESIDENT), and each solver
the GPU body the binding registers under that name (`ica_solve_dev_py`, ...)."""
from x_decomp.cells import F32Ptr, I32Ptr
from x_decomp.decomp_owner_kit import DecompDev
from x_decomp.resident import IDN_EIGH_RESIDENT, IDN_LU_RESIDENT, dev_eigh_run, dev_lu_run
from x_decomp.ica_dev import ica_solve_dev_run


struct DevDecomp(DecompDev):
    @staticmethod
    def has_eigh() -> Bool:
        return IDN_EIGH_RESIDENT

    @staticmethod
    def eigh(a: Int, w: F32Ptr, v: F32Ptr, n: Int, uplo: Int) raises:
        _ = dev_eigh_run(a, w, v, n, uplo)

    @staticmethod
    def has_lu() -> Bool:
        return IDN_LU_RESIDENT

    @staticmethod
    def lu(a: Int, lu: Int, piv: I32Ptr, info: F32Ptr, n: Int) raises:
        _ = dev_lu_run(a, lu, piv, info, n)

    @staticmethod
    def ica_solve(
        px: F32Ptr, pw: F32Ptr, nc: Int, xr: Int, xc: Int, algo: Int, fun: Int, max_iter: Int, alpha: Float64,
        tol: Float64,
    ) raises -> Int:
        return ica_solve_dev_run(px, pw, nc, xr, xc, algo, fun, max_iter, alpha, tol)
