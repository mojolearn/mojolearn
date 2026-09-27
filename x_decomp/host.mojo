# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""HostExec: the decomp lane's cells in host loops, one index at a time, in
the order the device assigns them to threads. No GPU import: this file is
compiled into the CPU host binding."""
from std.memory import bitcast
from std.sys.compile import is_defined

from decomposition.checks.jacobi_eigh_device import JACOBI_SWEEPS, JACOBI_TOL
from decomposition.host.linalg_public import host_eigh, host_qr_r
from decomposition.host.pca_full_oracle import host_one_sided_jacobi_svd, host_qr_factor
from x_decomp.cells import (
    F32Ptr,
    absmax_sign_cell,
    FOLD_BLOCK,
    colsum_part_cell,
    rowsum_part_cell,
    fold_cell,
    gemm_part_cell,
    X_DECOMP_SVD_SWEEPS,
    X_DECOMP_SVD_TOL,
    I32Ptr,
    bidx,
    cd_row,
    chol_serial,
    colsum_cell,
    ew_cell,
    gemm_cell,
    als_row,
    barycenter_row,
    dijkstra_row,
    gamma_cell,
    lasso_row,
    lda_doc_row,
    lu_serial,
    omp_row,
    lu_solve_serial,
    orth_rank_guard,
    trsm_row,
    rand_cell,
    rowsum_cell,
    sqdist_cell,
)
from x_decomp.exec_trait import Exec

comptime X_DECOMP_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()


@fieldwise_init
struct HostExec(Exec):
    @staticmethod
    def gemm(a: F32Ptr, b: F32Ptr, c: F32Ptr, m: Int, k: Int, n: Int, ta: Bool, tb: Bool) raises:
        var nb = (k + FOLD_BLOCK - 1) // FOLD_BLOCK
        var part = List[Float32](length=nb * m * n if nb > 1 else 1, fill=Float32(0))
        var pp = F32Ptr(unsafe_from_address=Int(part.unsafe_ptr()))
        if nb > 1:
            for bl in range(nb):
                for t in range(m * n):
                    pp.unsafe_store(bl * m * n + t, gemm_part_cell(
                        a, b, t // n, t % n, m, k, n, ta, tb, bl * FOLD_BLOCK, min(k, (bl + 1) * FOLD_BLOCK)))
        for i in range(m):
            for j in range(n):
                var v = fold_cell(pp, i * n + j, nb, m * n) if nb > 1 else gemm_cell(a, b, i, j, m, k, n, ta, tb)
                comptime if X_DECOMP_HOST_SABOTAGE:
                    # the gate's negative control (-D MOJOLEARN_HOST_SABOTAGE=1):
                    # the host column's every product moves by one unit in the
                    # last place; the GPU binding never defines it
                    v = bitcast[DType.float32](bitcast[DType.uint32](v) ^ UInt32(1))
                c.unsafe_store(i * n + j, v)

    @staticmethod
    def ew(
        op: Int, a: F32Ptr, b: F32Ptr, lb: Int, bm: Int, c: F32Ptr, lc: Int, cm: Int,
        dst: F32Ptr, count: Int, d: Int, s: Float32,
    ) raises:
        for i in range(count):
            dst.unsafe_store(
                i,
                ew_cell(op, a.unsafe_load(i), b.unsafe_load(bidx(bm, i, d)), c.unsafe_load(bidx(cm, i, d)), s),
            )

    @staticmethod
    def colsum(a: F32Ptr, dst: F32Ptr, n: Int, d: Int) raises:
        var nb = (n + FOLD_BLOCK - 1) // FOLD_BLOCK
        if nb <= 1:
            for j in range(d):
                dst.unsafe_store(j, colsum_cell(a, j, n, d))
            return
        var part = List[Float32](length=nb * d, fill=Float32(0))
        var pp = F32Ptr(unsafe_from_address=Int(part.unsafe_ptr()))
        for bl in range(nb):
            for j in range(d):
                pp.unsafe_store(bl * d + j, colsum_part_cell(a, j, n, d, bl * FOLD_BLOCK, min(n, (bl + 1) * FOLD_BLOCK)))
        for j in range(d):
            dst.unsafe_store(j, fold_cell(pp, j, nb, d))

    @staticmethod
    def rowsum(a: F32Ptr, dst: F32Ptr, n: Int, d: Int) raises:
        var nb = (d + FOLD_BLOCK - 1) // FOLD_BLOCK
        if nb <= 1:
            for i in range(n):
                dst.unsafe_store(i, rowsum_cell(a, i, d))
            return
        var part = List[Float32](length=nb * n, fill=Float32(0))
        var pp = F32Ptr(unsafe_from_address=Int(part.unsafe_ptr()))
        for bl in range(nb):
            for i in range(n):
                pp.unsafe_store(bl * n + i, rowsum_part_cell(a, i, d, bl * FOLD_BLOCK, min(d, (bl + 1) * FOLD_BLOCK)))
        for i in range(n):
            dst.unsafe_store(i, fold_cell(pp, i, nb, n))

    @staticmethod
    def sqdist(a: F32Ptr, b: F32Ptr, dst: F32Ptr, na: Int, nb: Int, d: Int) raises:
        for i in range(na):
            for j in range(nb):
                dst.unsafe_store(i * nb + j, sqdist_cell(a, b, i, j, d))

    @staticmethod
    def rand(dst: F32Ptr, count: Int, seed: UInt32, stream: UInt32, kind: Int) raises:
        for i in range(count):
            dst.unsafe_store(i, rand_cell(i, seed, stream, kind))

    @staticmethod
    def lu(a: F32Ptr, piv: I32Ptr, info: F32Ptr, n: Int) raises:
        lu_serial(a, piv, n, info)

    @staticmethod
    def lu_solve(lu: F32Ptr, piv: I32Ptr, b: F32Ptr, n: Int, nrhs: Int, trans: Int = 0) raises:
        lu_solve_serial(lu, piv, b, n, nrhs, trans)

    @staticmethod
    def chol(a: F32Ptr, info: F32Ptr, n: Int) raises:
        chol_serial(a, n, info)

    @staticmethod
    def eigh(a: F32Ptr, w: F32Ptr, v: F32Ptr, n: Int) raises:
        var m = List[Float32](capacity=n * n)
        for i in range(n * n):
            m.append(a.unsafe_load(i))
        var got = host_eigh(m, n)
        for i in range(n):
            w.unsafe_store(i, got.w[i])
        for i in range(n * n):
            v.unsafe_store(i, got.v[i])

    @staticmethod
    def cd_rows(w: F32Ptr, hht: F32Ptr, xht: F32Ptr, perm: I32Ptr, viol: F32Ptr, n: Int, k: Int) raises:
        for i in range(n):
            viol.unsafe_store(i, cd_row(w, hht, xht, perm, i, k))

    @staticmethod
    def orth(a: F32Ptr, m: Int, l: Int) raises:
        for _ in range(2):
            var w = List[Float32](capacity=m * l)
            for t in range(m * l):
                w.append(a.unsafe_load(t))
            var r = host_qr_r(w, m, l)
            var pr = F32Ptr(unsafe_from_address=Int(r.unsafe_ptr()))
            orth_rank_guard(pr, l)
            for i in range(m):
                trsm_row(F32Ptr(unsafe_from_address=Int(w.unsafe_ptr())), pr, a, i, l)
            _ = r^
            _ = w^

    @staticmethod
    def svd(a: F32Ptr, m: Int, n: Int, s: F32Ptr, v: F32Ptr) raises:
        """PCA(svd_solver='full')'s tall route: Householder QR of a (m x n,
        m >= n), then the one-sided Jacobi SVD of R. Unordered values, V in
        columns: the host replay of DevExec.svd."""
        var work = List[Float32](capacity=m * n)
        for i in range(m * n):
            work.append(a.unsafe_load(i))
        var r = host_qr_factor(work, m, n)
        var got = host_one_sided_jacobi_svd(r, n, X_DECOMP_SVD_SWEEPS, X_DECOMP_SVD_TOL)
        if not got.converged:
            raise Error("x_decomp svd: the one-sided Jacobi SVD did not converge")
        for i in range(n):
            s.unsafe_store(i, got.s[i])
        for i in range(n * n):
            v.unsafe_store(i, got.v[i])

    @staticmethod
    def lasso_rows(
        g: F32Ptr, q: F32Ptr, w: F32Ptr, h: F32Ptr, its: F32Ptr, n: Int, k: Int, alpha: Float32,
        max_iter: Int, tol: Float32, positive: Bool,
    ) raises:
        for i in range(n):
            its.unsafe_store(i, lasso_row(g, q, w, h, i, k, alpha, max_iter, tol, positive))

    @staticmethod
    def omp_rows(g: F32Ptr, q: F32Ptr, w: F32Ptr, s: F32Ptr, na: F32Ptr, n: Int, k: Int, nnz: Int) raises:
        for i in range(n):
            na.unsafe_store(i, omp_row(g, q, w, s, i, k, nnz))

    @staticmethod
    def rand_gamma(dst: F32Ptr, count: Int, seed: UInt32, stream: UInt32, shape: Float32) raises:
        for i in range(count):
            dst.unsafe_store(i, gamma_cell(i, seed, stream, shape))

    @staticmethod
    def lda_rows(
        x: F32Ptr, ew: F32Ptr, d: F32Ptr, e: F32Ptr, s: F32Ptr, its: F32Ptr, n: Int, k: Int, v: Int,
        prior: Float32, max_iter: Int, tol: Float32,
    ) raises:
        for i in range(n):
            its.unsafe_store(i, lda_doc_row(x, ew, d, e, s, i, k, v, prior, max_iter, tol))

    @staticmethod
    def dijkstra_rows(w: F32Ptr, dist: F32Ptr, reached: F32Ptr, n: Int) raises:
        var done = List[Float32](length=n * n, fill=Float32(0))
        var pd = F32Ptr(unsafe_from_address=Int(done.unsafe_ptr()))
        for i in range(n):
            reached.unsafe_store(i, dijkstra_row(w, dist, pd, i, n))
        _ = done^

    @staticmethod
    def barycenter_rows(
        x: F32Ptr, y: F32Ptr, nbr: F32Ptr, wt: F32Ptr, flags: F32Ptr, n: Int, ny: Int, d: Int, k: Int, reg: Float32
    ) raises:
        var s = List[Float32](length=n * (k * k + k * d) if n > 0 else 1, fill=Float32(0))
        var ps = F32Ptr(unsafe_from_address=Int(s.unsafe_ptr()))
        for i in range(n):
            flags.unsafe_store(i, barycenter_row(x, y, nbr, wt, ps, i, d, k, reg))
        _ = s^

    @staticmethod
    def als_rows(c: F32Ptr, y: F32Ptr, yty: F32Ptr, x: F32Ptr, flags: F32Ptr, n: Int, m: Int, f: Int, reg: Float32) raises:
        var s = List[Float32](length=n * (f * f + f) if n > 0 else 1, fill=Float32(0))
        var ps = F32Ptr(unsafe_from_address=Int(s.unsafe_ptr()))
        for u in range(n):
            flags.unsafe_store(u, als_row(c, y, yty, x, ps, u, m, f, reg))
        _ = s^

    @staticmethod
    def absmax_sign(a: F32Ptr, dst: F32Ptr, n: Int, d: Int, by_col: Bool) raises:
        for t in range(d if by_col else n):
            dst.unsafe_store(t, absmax_sign_cell(a, t, n, d, by_col))

    @staticmethod
    def qr_r(a: F32Ptr, m: Int, n: Int, r: F32Ptr) raises:
        var w = List[Float32](capacity=m * n)
        for t in range(m * n):
            w.append(a.unsafe_load(t))
        var got = host_qr_r(w, m, n)
        for t in range(n * n):
            r.unsafe_store(t, got[t])

    @staticmethod
    def vendor() -> String:
        return String("cpu")
