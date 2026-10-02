# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The two executors of the decomp lane's cells: `x_decomp/device.mojo`
(`DevExec`, one thread per output on the GPU) and `x_decomp/host.mojo`
(`HostExec`, the same cell in a host loop). Every pointer is a HOST address
the Python caller owns; the device executor uploads, launches and downloads."""
from x_decomp.cells import F32Ptr, I32Ptr


trait Exec:
    @staticmethod
    def gemm(a: F32Ptr, b: F32Ptr, c: F32Ptr, m: Int, k: Int, n: Int, ta: Bool, tb: Bool) raises:
        ...

    @staticmethod
    def ew(
        op: Int, a: F32Ptr, b: F32Ptr, lb: Int, bm: Int, c: F32Ptr, lc: Int, cm: Int,
        dst: F32Ptr, count: Int, d: Int, s: Float32,
    ) raises:
        ...

    @staticmethod
    def colsum(a: F32Ptr, dst: F32Ptr, n: Int, d: Int) raises:
        ...

    @staticmethod
    def rowsum(a: F32Ptr, dst: F32Ptr, n: Int, d: Int) raises:
        ...

    @staticmethod
    def sqdist(a: F32Ptr, b: F32Ptr, dst: F32Ptr, na: Int, nb: Int, d: Int, kind: Int = 0, pw: Float32 = Float32(2)) raises:
        ...

    @staticmethod
    def rand(dst: F32Ptr, count: Int, seed: UInt32, stream: UInt32, kind: Int) raises:
        ...

    @staticmethod
    def lu(a: F32Ptr, piv: I32Ptr, info: F32Ptr, n: Int) raises:
        ...

    @staticmethod
    def lu_solve(lu: F32Ptr, piv: I32Ptr, b: F32Ptr, n: Int, nrhs: Int, trans: Int = 0) raises:
        ...

    @staticmethod
    def chol(a: F32Ptr, info: F32Ptr, n: Int) raises:
        ...

    @staticmethod
    def eigh(a: F32Ptr, w: F32Ptr, v: F32Ptr, n: Int) raises:
        ...

    @staticmethod
    def cd_rows(w: F32Ptr, hht: F32Ptr, xht: F32Ptr, perm: I32Ptr, viol: F32Ptr, n: Int, k: Int) raises:
        ...

    @staticmethod
    def orth(a: F32Ptr, m: Int, l: Int) raises:
        ...

    @staticmethod
    def orth_diag(a: F32Ptr, m: Int, l: Int, diag: F32Ptr) raises:
        ...

    @staticmethod
    def svd(a: F32Ptr, m: Int, n: Int, s: F32Ptr, v: F32Ptr) raises:
        ...

    @staticmethod
    def lasso_rows(
        g: F32Ptr, q: F32Ptr, w: F32Ptr, h: F32Ptr, its: F32Ptr, n: Int, k: Int, alpha: Float32,
        max_iter: Int, tol: Float32, positive: Bool,
    ) raises:
        ...

    @staticmethod
    def omp_rows(g: F32Ptr, q: F32Ptr, w: F32Ptr, s: F32Ptr, na: F32Ptr, n: Int, k: Int, nnz: Int) raises:
        ...

    @staticmethod
    def rand_gamma(dst: F32Ptr, count: Int, seed: UInt32, stream: UInt32, shape: Float32) raises:
        ...

    @staticmethod
    def lda_rows(
        x: F32Ptr, ew: F32Ptr, d: F32Ptr, e: F32Ptr, s: F32Ptr, its: F32Ptr, n: Int, k: Int, v: Int,
        prior: Float32, max_iter: Int, tol: Float32,
    ) raises:
        ...

    @staticmethod
    def dijkstra_rows(w: F32Ptr, dist: F32Ptr, reached: F32Ptr, n: Int) raises:
        ...

    @staticmethod
    def barycenter_rows(
        x: F32Ptr, y: F32Ptr, nbr: F32Ptr, wt: F32Ptr, flags: F32Ptr, n: Int, ny: Int, d: Int, k: Int, reg: Float32
    ) raises:
        ...

    @staticmethod
    def als_rows(c: F32Ptr, y: F32Ptr, yty: F32Ptr, x: F32Ptr, flags: F32Ptr, n: Int, m: Int, f: Int, reg: Float32) raises:
        ...

    @staticmethod
    def absmax_sign(a: F32Ptr, dst: F32Ptr, n: Int, d: Int, by_col: Bool) raises:
        ...

    @staticmethod
    def qr_r(a: F32Ptr, m: Int, n: Int, r: F32Ptr) raises:
        ...

    @staticmethod
    def geqrf(a: F32Ptr, tau: F32Ptr, m: Int, n: Int) raises:
        ...

    @staticmethod
    def orgqr(h: F32Ptr, tau: F32Ptr, q: F32Ptr, m: Int, n: Int, kk: Int, qc: Int) raises:
        ...

    @staticmethod
    def als_cg_rows(c: F32Ptr, y: F32Ptr, yty: F32Ptr, x: F32Ptr, steps: F32Ptr, n: Int, m: Int, f: Int, reg: Float32, cg: Int) raises:
        ...

    @staticmethod
    def tsqr_factor(a: F32Ptr, b: F32Ptr, r: F32Ptr, m: Int, d: Int, nrhs: Int, keep: Bool) raises:
        """R (n x n, n = d + nrhs) of [a | b] by the blocked TSQR
        (x_decomp/tsqr_core.mojo); `keep` holds the factorization for
        `tsqr_apply`."""
        ...

    @staticmethod
    def tsqr_apply(c: F32Ptr, q: F32Ptr, m: Int, n: Int, k: Int) raises:
        """q (m x k) = Q c for the kept factorization (k == 0: release it)."""
        ...

    @staticmethod
    def vendor() -> String:
        ...
