# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""HostExec: the decomp lane's cells in host loops, one index at a time, in
the order the device assigns them to threads. No GPU import: this file is
compiled into the CPU host binding.

CPU speed (lane decomp-cpu, 2026-09-28): every loop over independent
outputs (an element, a row, a fold block, a slice, a source node) is a
task of `xd_parallel`, cut by a function of the SHAPE only, and each task
writes outputs no other task writes; the O(n^3) cells (gemm, sqdist), the
QR slices and the shortest-path rows have the host spellings of
x_decomp/host_simd.mojo, host_qr.mojo and host_graph.mojo (same words).
So the bits are the same at every thread count."""
from std.memory import bitcast
from std.sys.compile import is_defined

from decomposition.checks.jacobi_eigh_device import JACOBI_SWEEPS, JACOBI_TOL
from decomposition.host.linalg_public import host_eigh, host_qr_r
from decomposition.host.pca_full_oracle import host_one_sided_jacobi_svd
from core.host_predict_threads import host_predict_chunk, host_predict_task_count
from x_decomp.cells import (
    F32Ptr,
    absmax_sign_cell,
    FOLD_BLOCK,
    colsum_part_cell,
    rowsum_part_cell,
    fold_cell,
    X_DECOMP_SVD_SWEEPS,
    X_DECOMP_SVD_TOL,
    I32Ptr,
    bidx,
    cd_row,
    chol_serial,
    ew_cell,
    als_row,
    barycenter_row,
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
    pdist_cell,
)
from x_decomp.exec_trait import Exec
from x_decomp.host_qr import fast_qr_finish, qr_slice, qr_slices
from x_decomp.host_graph import EdgeList, dijkstra_heap_row
from x_decomp.host_simd import (
    gemm_fold_rows,
    gemm_prepare,
    gemm_rowdot,
    rowdot_task,
    rowdot_task_count,
    gemm_swapped,
    gemm_task,
    gemm_task_count,
    sqdist_prepare,
    sqdist_task,
    sqdist_task_count,
    colsum_rows,
)

comptime X_DECOMP_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()

#: elements per elementwise task, rows per row-fold task: shape-only cuts
comptime EW_CHUNK = 32768
comptime ROW_CHUNK = 1024


def xd_parallel[FuncType: def(Int) -> None](ref func: FuncType, n: Int):
    """THE ONE THREAD SPLIT of the decomp host column: tasks 0..n-1, each
    writing only its own outputs, run in `host_predict_task_count(n)`
    contiguous groups (MOJOLEARN_CPU_THREADS is the ceiling; 1 is serial).
    Which thread runs a task never changes what it computes. The groups
    run through core/host_parallel.mojo once it is on main (lane cpu:
    `sync_parallelize` in the caller's floating-point environment,
    DEVIATION 5900); until then they run on the calling thread."""
    var groups = host_predict_task_count(n)
    var chunk = host_predict_chunk(n, groups)

    def group(g: Int) {imm func, imm n, imm chunk}:
        for i in range(g * chunk, min(n, (g + 1) * chunk)):
            func(i)

    for g in range(groups):
        group(g)


@fieldwise_init
struct HostExec(Exec):
    @staticmethod
    def gemm(a: F32Ptr, b: F32Ptr, c: F32Ptr, m: Int, k: Int, n: Int, ta: Bool, tb: Bool) raises:
        # the cell's arithmetic and order, SIMD across outputs (x_decomp/host_simd.mojo)
        if gemm_rowdot(m, n, ta):
            def rd(t: Int) {imm a, imm b, imm c, imm m, imm k}:
                rowdot_task(t, a, b, c, m, k)

            xd_parallel(rd, rowdot_task_count(m))
        elif gemm_swapped(m, n):
            # a narrow C (a matrix-vector product): C^T = op(B)^T op(A)^T fills
            # the vector lanes; each output's chain is the same (fma(x, y, acc)
            # == fma(y, x, acc) exactly), then C^T is transposed back
            var ct = List[Float32](length=m * n, fill=Float32(0))
            var pct = F32Ptr(unsafe_from_address=Int(ct.unsafe_ptr()))
            HostExec._gemm_simd(b, a, pct, n, k, m, not tb, not ta)
            def tr(i: Int) {imm c, imm pct, imm m, imm n}:
                for j in range(n):
                    c.unsafe_store(i * n + j, pct.unsafe_load(j * m + i))

            xd_parallel(tr, m)
            _ = ct^
        else:
            HostExec._gemm_simd(a, b, c, m, k, n, ta, tb)
        comptime if X_DECOMP_HOST_SABOTAGE:
            # the gate's negative control (-D MOJOLEARN_HOST_SABOTAGE=1):
            # the host column's every product moves by one unit in the
            # last place; the GPU binding never defines it
            for t in range(m * n):
                c.unsafe_store(t, bitcast[DType.float32](bitcast[DType.uint32](c.unsafe_load(t)) ^ UInt32(1)))

    @staticmethod
    def _gemm_simd(a: F32Ptr, b: F32Ptr, c: F32Ptr, m: Int, k: Int, n: Int, ta: Bool, tb: Bool):
        var nb = (k + FOLD_BLOCK - 1) // FOLD_BLOCK
        var part = gemm_prepare(c, m, k, n)
        var pp = F32Ptr(unsafe_from_address=Int(part.unsafe_ptr()))
        def task(t: Int) {imm a, imm b, imm c, imm pp, imm m, imm k, imm n, imm ta, imm tb}:
            gemm_task(t, a, b, c, pp, m, k, n, ta, tb)

        xd_parallel(task, gemm_task_count(m, k, n))
        if nb > 1:
            def fold(i: Int) {imm c, imm pp, imm m, imm n, imm nb}:
                gemm_fold_rows(i, c, pp, m, n, nb)

            xd_parallel(fold, m)
        _ = part^

    @staticmethod
    def _qr_r(a: F32Ptr, m: Int, n: Int) raises -> List[Float32]:
        """`host_qr_r(a, m, n)`'s R (a is not modified): the slices of
        x_decomp/host_qr.mojo as tasks, then the stacked R's once more."""
        var work = List[Float32](capacity=m * n)
        for t in range(m * n):
            work.append(a.unsafe_load(t))
        if m < 1 or n < 1 or n > 46340 or m < n:
            return host_qr_r(work, m, n)  # refuses the shape, by name
        var pw = F32Ptr(unsafe_from_address=Int(work.unsafe_ptr()))
        var ns = qr_slices(m, n)
        var r = List[Float32](length=n * n, fill=Float32(0))
        var pr = F32Ptr(unsafe_from_address=Int(r.unsafe_ptr()))
        if ns == 1:
            qr_slice(pw, pr, m, n, 1, 0)
        else:
            var scratch = List[Float32](length=ns * n * n, fill=Float32(0))
            var ps = F32Ptr(unsafe_from_address=Int(scratch.unsafe_ptr()))
            def sl(b: Int) {imm pw, imm ps, imm m, imm n, imm ns}:
                qr_slice(pw, ps, m, n, ns, b)

            xd_parallel(sl, ns)
            fast_qr_finish(ps, pr, ns, n)
            _ = scratch^
        _ = work^
        return r^

    @staticmethod
    def ew(
        op: Int, a: F32Ptr, b: F32Ptr, lb: Int, bm: Int, c: F32Ptr, lc: Int, cm: Int,
        dst: F32Ptr, count: Int, d: Int, s: Float32,
    ) raises:
        def chunk(t: Int) {imm op, imm a, imm b, imm bm, imm c, imm cm, imm dst, imm count, imm d, imm s}:
            for i in range(t * EW_CHUNK, min(count, (t + 1) * EW_CHUNK)):
                dst.unsafe_store(
                    i,
                    ew_cell(op, a.unsafe_load(i), b.unsafe_load(bidx(bm, i, d)), c.unsafe_load(bidx(cm, i, d)), s),
                )

        xd_parallel(chunk, (count + EW_CHUNK - 1) // EW_CHUNK)

    @staticmethod
    def colsum(a: F32Ptr, dst: F32Ptr, n: Int, d: Int) raises:
        var nb = (n + FOLD_BLOCK - 1) // FOLD_BLOCK
        # the cells' chains, SIMD across columns (x_decomp/host_simd.mojo)
        if nb <= 1:
            colsum_rows(a, dst, d, 0, n)
            return
        var part = List[Float32](length=nb * d, fill=Float32(0))
        var pp = F32Ptr(unsafe_from_address=Int(part.unsafe_ptr()))

        def blk(bl: Int) {imm a, imm pp, imm n, imm d}:
            colsum_rows(a, pp.unsafe_offset(bl * d), d, bl * FOLD_BLOCK, min(n, (bl + 1) * FOLD_BLOCK))

        xd_parallel(blk, nb)
        for j in range(d):
            dst.unsafe_store(j, fold_cell(pp, j, nb, d))
        _ = part^

    @staticmethod
    def rowsum(a: F32Ptr, dst: F32Ptr, n: Int, d: Int) raises:
        var nb = (d + FOLD_BLOCK - 1) // FOLD_BLOCK
        if nb <= 1:
            def rows(t: Int) {imm a, imm dst, imm n, imm d}:
                for i in range(t * ROW_CHUNK, min(n, (t + 1) * ROW_CHUNK)):
                    dst.unsafe_store(i, rowsum_cell(a, i, d))

            xd_parallel(rows, (n + ROW_CHUNK - 1) // ROW_CHUNK)
            return
        var part = List[Float32](length=nb * n, fill=Float32(0))
        var pp = F32Ptr(unsafe_from_address=Int(part.unsafe_ptr()))

        def rowp(i: Int) {imm a, imm dst, imm pp, imm n, imm d, imm nb}:
            for bl in range(nb):
                pp.unsafe_store(bl * n + i, rowsum_part_cell(a, i, d, bl * FOLD_BLOCK, min(d, (bl + 1) * FOLD_BLOCK)))
            dst.unsafe_store(i, fold_cell(pp, i, nb, n))

        xd_parallel(rowp, n)
        _ = part^

    @staticmethod
    def sqdist(a: F32Ptr, b: F32Ptr, dst: F32Ptr, na: Int, nb: Int, d: Int, kind: Int = 0, pw: Float32 = Float32(2)) raises:
        if kind == 0:
            # the cell's arithmetic and order, SIMD across outputs (x_decomp/host_simd.mojo)
            var bt = sqdist_prepare(b, nb, d)
            var pbt = F32Ptr(unsafe_from_address=Int(bt.unsafe_ptr()))
            def sq(t: Int) {imm a, imm pbt, imm dst, imm na, imm nb, imm d}:
                sqdist_task(t, a, pbt, dst, na, nb, d)

            xd_parallel(sq, sqdist_task_count(na))
            _ = bt^
            return

        def pd(i: Int) {imm a, imm b, imm dst, imm nb, imm d, imm kind, imm pw}:
            for j in range(nb):
                dst.unsafe_store(i * nb + j, pdist_cell(a, b, i, j, d, kind, pw))

        xd_parallel(pd, na)

    @staticmethod
    def rand(dst: F32Ptr, count: Int, seed: UInt32, stream: UInt32, kind: Int) raises:
        def chunk(t: Int) {imm dst, imm count, imm seed, imm stream, imm kind}:
            for i in range(t * EW_CHUNK, min(count, (t + 1) * EW_CHUNK)):
                dst.unsafe_store(i, rand_cell(i, seed, stream, kind))

        xd_parallel(chunk, (count + EW_CHUNK - 1) // EW_CHUNK)

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
        def row(i: Int) {imm w, imm hht, imm xht, imm perm, imm viol, imm k}:
            viol.unsafe_store(i, cd_row(w, hht, xht, perm, i, k))

        xd_parallel(row, n)

    @staticmethod
    def orth(a: F32Ptr, m: Int, l: Int) raises:
        for _ in range(2):
            var w = List[Float32](capacity=m * l)
            for t in range(m * l):
                w.append(a.unsafe_load(t))
            var r = HostExec._qr_r(F32Ptr(unsafe_from_address=Int(w.unsafe_ptr())), m, l)
            var pr = F32Ptr(unsafe_from_address=Int(r.unsafe_ptr()))
            orth_rank_guard(pr, l)
            var pw = F32Ptr(unsafe_from_address=Int(w.unsafe_ptr()))

            def row(i: Int) {imm pw, imm pr, imm a, imm l}:
                trsm_row(pw, pr, a, i, l)

            xd_parallel(row, m)
            _ = r^
            _ = w^

    @staticmethod
    def svd(a: F32Ptr, m: Int, n: Int, s: F32Ptr, v: F32Ptr) raises:
        """PCA(svd_solver='full')'s tall route: Householder QR of a (m x n,
        m >= n), then the one-sided Jacobi SVD of R. Unordered values, V in
        columns: the host replay of DevExec.svd."""
        var r = HostExec._qr_r(a, m, n)
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
        def row(i: Int) {imm g, imm q, imm w, imm h, imm its, imm k, imm alpha, imm max_iter, imm tol, imm positive}:
            its.unsafe_store(i, lasso_row(g, q, w, h, i, k, alpha, max_iter, tol, positive))

        xd_parallel(row, n)

    @staticmethod
    def omp_rows(g: F32Ptr, q: F32Ptr, w: F32Ptr, s: F32Ptr, na: F32Ptr, n: Int, k: Int, nnz: Int) raises:
        def row(i: Int) {imm g, imm q, imm w, imm s, imm na, imm k, imm nnz}:
            na.unsafe_store(i, omp_row(g, q, w, s, i, k, nnz))

        xd_parallel(row, n)

    @staticmethod
    def rand_gamma(dst: F32Ptr, count: Int, seed: UInt32, stream: UInt32, shape: Float32) raises:
        def chunk(t: Int) {imm dst, imm count, imm seed, imm stream, imm shape}:
            for i in range(t * EW_CHUNK, min(count, (t + 1) * EW_CHUNK)):
                dst.unsafe_store(i, gamma_cell(i, seed, stream, shape))

        xd_parallel(chunk, (count + EW_CHUNK - 1) // EW_CHUNK)

    @staticmethod
    def lda_rows(
        x: F32Ptr, ew: F32Ptr, d: F32Ptr, e: F32Ptr, s: F32Ptr, its: F32Ptr, n: Int, k: Int, v: Int,
        prior: Float32, max_iter: Int, tol: Float32,
    ) raises:
        def row(i: Int) {imm x, imm ew, imm d, imm e, imm s, imm its, imm k, imm v, imm prior, imm max_iter, imm tol}:
            its.unsafe_store(i, lda_doc_row(x, ew, d, e, s, i, k, v, prior, max_iter, tol))

        xd_parallel(row, n)

    @staticmethod
    def dijkstra_rows(w: F32Ptr, dist: F32Ptr, reached: F32Ptr, n: Int) raises:
        # the same distances on a heap over the listed edges (x_decomp/host_graph.mojo)
        var g = EdgeList(w, n)
        def row(i: Int) {imm g, imm dist, imm reached, imm n}:
            reached.unsafe_store(i, dijkstra_heap_row(g, dist, i, n))

        xd_parallel(row, n)
        _ = g^

    @staticmethod
    def barycenter_rows(
        x: F32Ptr, y: F32Ptr, nbr: F32Ptr, wt: F32Ptr, flags: F32Ptr, n: Int, ny: Int, d: Int, k: Int, reg: Float32
    ) raises:
        var s = List[Float32](length=n * (k * k + k * d) if n > 0 else 1, fill=Float32(0))
        var ps = F32Ptr(unsafe_from_address=Int(s.unsafe_ptr()))
        def row(i: Int) {imm x, imm y, imm nbr, imm wt, imm flags, imm ps, imm d, imm k, imm reg}:
            flags.unsafe_store(i, barycenter_row(x, y, nbr, wt, ps, i, d, k, reg))

        xd_parallel(row, n)
        _ = s^

    @staticmethod
    def als_rows(c: F32Ptr, y: F32Ptr, yty: F32Ptr, x: F32Ptr, flags: F32Ptr, n: Int, m: Int, f: Int, reg: Float32) raises:
        var s = List[Float32](length=n * (f * f + f) if n > 0 else 1, fill=Float32(0))
        var ps = F32Ptr(unsafe_from_address=Int(s.unsafe_ptr()))
        def row(u: Int) {imm c, imm y, imm yty, imm x, imm flags, imm ps, imm m, imm f, imm reg}:
            flags.unsafe_store(u, als_row(c, y, yty, x, ps, u, m, f, reg))

        xd_parallel(row, n)
        _ = s^

    @staticmethod
    def absmax_sign(a: F32Ptr, dst: F32Ptr, n: Int, d: Int, by_col: Bool) raises:
        def one(t: Int) {imm a, imm dst, imm n, imm d, imm by_col}:
            dst.unsafe_store(t, absmax_sign_cell(a, t, n, d, by_col))

        xd_parallel(one, d if by_col else n)

    @staticmethod
    def qr_r(a: F32Ptr, m: Int, n: Int, r: F32Ptr) raises:
        var got = HostExec._qr_r(a, m, n)
        for t in range(n * n):
            r.unsafe_store(t, got[t])

    @staticmethod
    def vendor() -> String:
        return String("cpu")
