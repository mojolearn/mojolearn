# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""HostExec: the decomp lane's cells in host loops, one index at a time, in
the order the device assigns them to threads. No GPU import: this file is
compiled into the CPU host binding."""
from std.sys.compile import is_defined

from decomposition.checks.jacobi_eigh_device import JACOBI_SWEEPS, JACOBI_TOL
from decomposition.host.linalg_public import host_eigh
from decomposition.host.pca_full_oracle import host_one_sided_jacobi_svd, host_qr_factor
from x_decomp.cells import (
    F32Ptr,
    I32Ptr,
    bidx,
    cd_row,
    chol_serial,
    colsum_cell,
    ew_cell,
    gemm_cell,
    lu_serial,
    lu_solve_serial,
    orth_serial,
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
        for i in range(m):
            for j in range(n):
                c.unsafe_store(i * n + j, gemm_cell(a, b, i, j, m, k, n, ta, tb))

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
        for j in range(d):
            dst.unsafe_store(j, colsum_cell(a, j, n, d))

    @staticmethod
    def rowsum(a: F32Ptr, dst: F32Ptr, n: Int, d: Int) raises:
        for i in range(n):
            dst.unsafe_store(i, rowsum_cell(a, i, d))

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
    def lu_solve(lu: F32Ptr, piv: I32Ptr, b: F32Ptr, n: Int, nrhs: Int) raises:
        lu_solve_serial(lu, piv, b, n, nrhs)

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
        orth_serial(a, m, l)

    @staticmethod
    def svd(a: F32Ptr, m: Int, n: Int, s: F32Ptr, v: F32Ptr) raises:
        """PCA(svd_solver='full')'s tall route: Householder QR of a (m x n,
        m >= n), then the one-sided Jacobi SVD of R. Unordered values, V in
        columns: the host replay of DevExec.svd."""
        var work = List[Float32](capacity=m * n)
        for i in range(m * n):
            work.append(a.unsafe_load(i))
        var r = host_qr_factor(work, m, n)
        var got = host_one_sided_jacobi_svd(r, n, JACOBI_SWEEPS, Float32(JACOBI_TOL))
        if not got.converged:
            raise Error("x_decomp svd: the one-sided Jacobi SVD did not converge")
        for i in range(n):
            s.unsafe_store(i, got.s[i])
        for i in range(n * n):
            v.unsafe_store(i, got.v[i])

    @staticmethod
    def vendor() -> String:
        return String("cpu")
