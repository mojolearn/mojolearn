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
from x_decomp.rr_solve import host_eigh_rr_sorted
from std.memory import bitcast
from std.builtin.sort import sort
from x_decomp.lle_local import (
    hessian_cell,
    hessian_comp_cell,
    hessian_ncy,
    hessian_q_cell,
    lle_apply_cell,
    lle_gram_cell,
    lle_mean_cell,
    ltsa_cell,
    mlle_eta,
    mlle_key,
    mlle_rows_cell,
    mlle_unkey,
    mlle_weights_cell,
)
from std.sys.compile import is_defined

from decomposition.host.linalg_public import host_eigh, host_qr_r
from checks.numerics import ftz
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_chunk, host_predict_task_count
from x_decomp.cells import (
    lu_perm_src,
    lu_aux_clamp,
    lu_aux_join,
    lu_aux_val,
    trs_tri_cols,
    trisolve_serial,
    knn_select_row,
    F32Ptr,
    absmax_sign_cell,
    FOLD_BLOCK,
    colsum_part_cell,
    rowsum_part_cell,
    fold_cell,
    X_DECOMP_SVD_SWEEPS,
    X_DECOMP_SVD_TOL,
    I32Ptr,
    cd_row,
    chol_serial,
    als_row,
    als_cg_row,
    barycenter_row,
    gamma_cell,
    lasso_row,
    omp_row,
    lars_row,
    LARS_ROW_EXTRA,
    orth_diag_cell,
    orth_rank_guard,
    trsm_row,
    rand_cell,
    pdist_cell,
)
from core.host_lanes import host_row_tasks
from core.host_parallel import host_parallelize
from x_decomp.exec_trait import Exec
from x_decomp.tsqr_host import ts_apply_host, ts_factor_host, ts_free_host
from x_decomp.qr_sliced_host import qs_geqrf_host, qs_orgqr_host
from x_decomp.host_jacobi import fast_one_sided_jacobi_svd
from x_decomp.host_qr import fast_qr_finish, qr_slice, qr_slices
from x_decomp.host_ew import ew_range
from x_decomp.host_lda import lda_doc_row_host, lda_pack_t
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
    rowsum_rows,
    lu_rows,
)

comptime X_DECOMP_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()

#: columns of B per host task of the right-looking lu_solve (a schedule cut)
comptime LU_RL_COLS = 16


def lu_solve_rl(lu: F32Ptr, piv: I32Ptr, b: F32Ptr, n: Int, nrhs: Int, trans: Int):
    """lu_solve in the right-looking order (DEVIATION 5308), the device's
    chains: B gathered by the swaps' row order (`lu_perm_src`), then
    `trs_tri_cols` on slices of LU_RL_COLS columns over host tasks (each
    column's chains are its own); 'T' solves first and scatters the rows
    back through the same order."""
    if n <= 0 or nrhs <= 0:
        return
    var src = List[Int](length=n, fill=0)
    for i in range(n):
        src[i] = lu_perm_src(piv, i)
    var tmp = List[Float32](unsafe_uninit_length=n * nrhs)
    var tp = F32Ptr(unsafe_from_address=Int(tmp.unsafe_ptr()))
    for i in range(n):
        var r = src[i] if trans == 0 else i
        for c in range(nrhs):
            tp.unsafe_store(i * nrhs + c, b.unsafe_load(r * nrhs + c))
    var t0 = 0 if trans == 0 else 2
    var slices = (nrhs + LU_RL_COLS - 1) // LU_RL_COLS
    var tasks = host_row_tasks(slices, n * n * LU_RL_COLS)
    var per = (slices + tasks - 1) // tasks

    def run(t: Int) {imm lu, imm tp, imm n, imm nrhs, imm t0, imm per}:
        var c0 = t * per * LU_RL_COLS
        var c1 = min(nrhs, c0 + per * LU_RL_COLS)
        if c1 > c0:
            trs_tri_cols(lu, tp, n, nrhs, t0, c0, c1)
            trs_tri_cols(lu, tp, n, nrhs, t0 + 1, c0, c1)

    var used = (slices + per - 1) // per
    if used <= 1:
        run(0)
    else:
        host_parallelize(run, used)
    for i in range(n):
        var r = i if trans == 0 else src[i]
        for c in range(nrhs):
            b.unsafe_store(r * nrhs + c, tp.unsafe_load(i * nrhs + c))
    _ = tmp^
    _ = src^


#: elements per elementwise task, rows per row-fold task: shape-only cuts
comptime EW_CHUNK = 32768
comptime ROW_CHUNK = 1024
comptime LU_ROWS = 32


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

    # lane/neural-pass79 (2026-10-01): the groups now run on the host pool
    # (core/host_parallel.mojo, in the caller's floating-point environment);
    # until this lane they ran one after another on the calling thread, so
    # every decomp host op was single threaded. Every task body writes only
    # its own outputs and per-task scratch (rows, blocks, slices; audited),
    # so the bits are the serial loop's.
    # `-D MOJOLEARN_XDECOMP_HOST_SERIAL=1` restores the serial loop.
    comptime if is_defined["MOJOLEARN_XDECOMP_HOST_SERIAL"]():
        for g in range(groups):
            group(g)
        return
    if groups <= 1:
        group(0)
        return
    host_parallelize(group, groups)


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
        # ew_cell's statements, lanes across elements (x_decomp/host_ew.mojo)
        def chunk(t: Int) {imm op, imm a, imm b, imm bm, imm c, imm cm, imm dst, imm count, imm d, imm s}:
            ew_range(op, a, b, bm, c, cm, dst, t * EW_CHUNK, min(count, (t + 1) * EW_CHUNK), d, s)

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
            # rowsum_cell's chains, lanes across rows (host_simd.rowsum_rows)
            def rows(t: Int) {imm a, imm dst, imm n, imm d}:
                rowsum_rows(a, dst, d, t * ROW_CHUNK, min(n, (t + 1) * ROW_CHUNK))

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
        # lu_serial's statements; each step's row eliminations are tasks of
        # LU_ROWS rows, SIMD across the row (x_decomp/host_simd.mojo lu_rows)
        info.unsafe_store(0, Float32(0))
        for k in range(n):
            var p = k
            var best = abs(ftz(a.unsafe_load(k * n + k)))
            for i in range(k + 1, n):
                var v = abs(ftz(a.unsafe_load(i * n + k)))
                if v > best:
                    best = v
                    p = i
            piv.unsafe_store(k, Int32(p))
            if p != k:
                for j in range(n):
                    var t = a.unsafe_load(k * n + j)
                    a.unsafe_store(k * n + j, a.unsafe_load(p * n + j))
                    a.unsafe_store(p * n + j, t)
            var d = ftz(a.unsafe_load(k * n + k))
            if d == Float32(0):
                if info.unsafe_load(0) == Float32(0):
                    info.unsafe_store(0, Float32(k + 1))
                continue
            var rows = n - k - 1

            def elim(t: Int) {imm a, imm n, imm k, imm d, imm rows}:
                lu_rows(a, n, k, d, k + 1 + t * LU_ROWS, k + 1 + min(rows, (t + 1) * LU_ROWS))

            xd_parallel(elim, (rows + LU_ROWS - 1) // LU_ROWS)

    @staticmethod
    def trisolve(lu: F32Ptr, idx: F32Ptr, src: F32Ptr, dst: F32Ptr, n: Int, nrhs: Int, trans: Int) raises:
        var tmp = List[Float32](unsafe_uninit_length=max(n * nrhs, 1))
        trisolve_serial(lu, idx, src, dst, F32Ptr(unsafe_from_address=Int(tmp.unsafe_ptr())), n, nrhs, trans)
        _ = tmp^

    @staticmethod
    def knn_select(dmat: F32Ptr, dist: F32Ptr, idx: F32Ptr, n: Int, m: Int, k: Int, exclude_self: Int) raises:
        def row(t: Int) {imm dmat, imm dist, imm idx, imm m, imm k, imm exclude_self}:
            knn_select_row(dmat, dist, idx, t, m, k, exclude_self)

        xd_parallel(row, n)

    @staticmethod
    def lu_solve(lu: F32Ptr, piv: I32Ptr, b: F32Ptr, n: Int, nrhs: Int, trans: Int = 0) raises:
        lu_solve_rl(lu, piv, b, n, nrhs, trans)

    @staticmethod
    def chol(a: F32Ptr, info: F32Ptr, n: Int) raises:
        chol_serial(a, n, info)

    @staticmethod
    def eigh(a: F32Ptr, w: F32Ptr, v: F32Ptr, n: Int, uplo: Int) raises:
        if n < 1 or n > 46340:
            _ = host_eigh(List[Float32](), n)  # refuses the size, by name
        var m = List[Float32](capacity=n * n)
        for i in range(n * n):
            m.append(a.unsafe_load(i))
        # numpy's UPLO (the device's `sym_from_triangle_kernel`)
        for i in range(n):
            for j in range(n):
                if (uplo == 1 and i < j) or (uplo == 2 and i > j):
                    m[i * n + j] = m[j * n + i]
        _host_eigh_rr_one(m, w, v, n)

    @staticmethod
    def eigh_batch(a: F32Ptr, w: F32Ptr, v: F32Ptr, batch: Int, n: Int) raises:
        """x_decomp/rr_batch.mojo's problems one after another: each is
        `eigh`'s solve (the same words)."""
        for b in range(batch):
            var m = List[Float32](capacity=n * n)
            for i in range(n * n):
                m.append(a.unsafe_load(b * n * n + i))
            _host_eigh_rr_one(m, w + b * n, v + b * n * n, n)

    @staticmethod
    def lle_apply(wb: F32Ptr, idx: F32Ptr, emb: F32Ptr, dst: F32Ptr, nq: Int, nf: Int, nn: Int, nc: Int) raises:
        def row(i: Int) {imm wb, imm idx, imm emb, imm dst, imm nn, imm nc}:
            for c in range(nc):
                lle_apply_cell(wb, idx, emb, dst, i, c, nn, nc)

        xd_parallel(row, nq)

    @staticmethod
    def lle_local(
        x: F32Ptr, idx: F32Ptr, bmat: F32Ptr, method: Int, n: Int, d: Int, nn: Int, nc: Int, tol: Float32
    ) raises:
        """DevExec.lle_local's cells, one index at a time (the same words);
        `bmat` arrives zeroed (the caller's zeros)."""
        var nn2 = n * nn * nn
        var g = List[Float32](length=max(nn2, 1), fill=Float32(0.0))
        var pg = F32Ptr(unsafe_from_address=Int(g.unsafe_ptr()))
        var mu = List[Float32](length=max(n * d, 1), fill=Float32(0.0))
        var pmu = F32Ptr(unsafe_from_address=Int(mu.unsafe_ptr()))
        var cen = x if method == 2 else pmu
        if method != 2:
            def mean(t: Int) {imm x, imm idx, imm pmu, imm d, imm nn}:
                lle_mean_cell(x, idx, pmu, t // d, t % d, d, nn)

            xd_parallel(mean, n * d)

        def gram(i: Int) {imm x, imm idx, imm cen, imm pg, imm d, imm nn}:
            for a in range(nn):
                for b in range(nn):
                    lle_gram_cell(x, idx, cen, pg, i, a, b, d, nn)

        xd_parallel(gram, n)
        var w = List[Float32](length=max(n * nn, 1), fill=Float32(0.0))
        var v = List[Float32](length=max(nn2, 1), fill=Float32(0.0))
        var pw = F32Ptr(unsafe_from_address=Int(w.unsafe_ptr()))
        var pv = F32Ptr(unsafe_from_address=Int(v.unsafe_ptr()))
        HostExec.eigh_batch(pg, pw, pv, n, nn)
        if method == 0:
            def ltsa(i: Int) {imm pv, imm idx, imm bmat, imm n, imm nn, imm nc}:
                for a in range(nn):
                    for b in range(nn):
                        ltsa_cell(pv, idx, bmat, i, a, b, n, nn, nc)

            xd_parallel(ltsa, n)
        elif method == 1:
            var ncy = hessian_ncy(nc)
            var ncol = nn - 1 - nc
            var extra = ncol - nc * (nc + 1) // 2
            var q = List[Float32](length=max(n * nn * ncy, 1), fill=Float32(0.0))
            var pq = F32Ptr(unsafe_from_address=Int(q.unsafe_ptr()))

            def hq(i: Int) {imm pv, imm pq, imm nn, imm nc}:
                hessian_q_cell(pv, pq, i, nn, nc)

            xd_parallel(hq, n)
            var vc = List[Float32](length=max(nn2, 1), fill=Float32(0.0))
            var pvc = F32Ptr(unsafe_from_address=Int(vc.unsafe_ptr()))
            if extra > 0:
                var cm = List[Float32](length=max(nn2, 1), fill=Float32(0.0))
                var pc = F32Ptr(unsafe_from_address=Int(cm.unsafe_ptr()))

                def comp(i: Int) {imm pq, imm pc, imm nn, imm nc}:
                    for a in range(nn):
                        for b in range(nn):
                            hessian_comp_cell(pq, pc, i, a, b, nn, nc)

                xd_parallel(comp, n)
                var w2 = List[Float32](length=max(n * nn, 1), fill=Float32(0.0))
                HostExec.eigh_batch(pc, F32Ptr(unsafe_from_address=Int(w2.unsafe_ptr())), pvc, n, nn)
                _ = cm^
                _ = w2^

            def hs(i: Int) {imm pq, imm pvc, imm idx, imm bmat, imm n, imm nn, imm nc, imm tol, imm ncol}:
                for c in range(ncol):
                    hessian_cell(pq, pvc, idx, bmat, i, c, n, nn, nc, tol)

            xd_parallel(hs, n)
            _ = q^
            _ = vc^
        else:
            var nev = min(d, nn)
            var wr = List[Float32](length=max(n * nn, 1), fill=Float32(0.0))
            var rho = List[Float32](length=max(n, 1), fill=Float32(0.0))
            var scr = List[Float32](length=max(3 * n * nn, 1), fill=Float32(0.0))
            var pwr = F32Ptr(unsafe_from_address=Int(wr.unsafe_ptr()))
            var prho = F32Ptr(unsafe_from_address=Int(rho.unsafe_ptr()))
            var pscr = F32Ptr(unsafe_from_address=Int(scr.unsafe_ptr()))

            def mw(i: Int) {imm pw, imm pv, imm pwr, imm prho, imm pscr, imm nn, imm nev, imm nc}:
                mlle_weights_cell(pw, pv, pwr, prho, pscr, i, nn, nev, nc)

            xd_parallel(mw, n)
            # the device's radix-sort keys, sorted (exact: the same order)
            var keys = List[UInt64](capacity=n)
            for i in range(n):
                keys.append((UInt64(mlle_key(rho[i])) << 32) | UInt64(i))
            sort(keys)
            var srt = List[Float32](length=max(n, 1), fill=Float32(0.0))
            for i in range(n):
                srt[i] = mlle_unkey(UInt32(keys[i] >> 32))
            var eta = mlle_eta(F32Ptr(unsafe_from_address=Int(srt.unsafe_ptr())), n)

            def mr(i: Int) {imm pw, imm pv, imm pwr, imm idx, imm bmat, imm pscr, imm eta, imm n, imm nn, imm nev, imm tol}:
                mlle_rows_cell(pw, pv, pwr, idx, bmat, pscr, eta, i, n, nn, nev, tol)

            xd_parallel(mr, n)
            _ = wr^
            _ = rho^
            _ = scr^
            _ = srt^
        _ = g^
        _ = mu^
        _ = w^
        _ = v^

    @staticmethod
    def cd_rows(w: F32Ptr, hht: F32Ptr, xht: F32Ptr, perm: I32Ptr, viol: F32Ptr, n: Int, k: Int) raises:
        def row(i: Int) {imm w, imm hht, imm xht, imm perm, imm viol, imm k}:
            viol.unsafe_store(i, cd_row(w, hht, xht, perm, i, k))

        xd_parallel(row, n)

    @staticmethod
    def orth(a: F32Ptr, m: Int, l: Int) raises:
        var none = List[Float32](length=1, fill=Float32(1))
        HostExec._orth_passes(a, m, l, F32Ptr(unsafe_from_address=Int(none.unsafe_ptr())), False)
        _ = none^

    @staticmethod
    def orth_diag(a: F32Ptr, m: Int, l: Int, diag: F32Ptr) raises:
        """`orth`, and `diag` (l floats) the product of the two passes'
        guarded R diagonals (`orth_diag_cell`; the device's
        `orth_on_device_diag`)."""
        for j in range(l):
            diag.unsafe_store(j, Float32(1))
        HostExec._orth_passes(a, m, l, diag, True)

    @staticmethod
    def _orth_passes(a: F32Ptr, m: Int, l: Int, diag: F32Ptr, with_diag: Bool) raises:
        for _ in range(2):
            var w = List[Float32](capacity=m * l)
            for t in range(m * l):
                w.append(a.unsafe_load(t))
            var r = HostExec._qr_r(F32Ptr(unsafe_from_address=Int(w.unsafe_ptr())), m, l)
            var pr = F32Ptr(unsafe_from_address=Int(r.unsafe_ptr()))
            orth_rank_guard(pr, l)
            if with_diag:
                for j in range(l):
                    orth_diag_cell(pr, diag, j, l)
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
        var got = fast_one_sided_jacobi_svd(r, n, X_DECOMP_SVD_SWEEPS, X_DECOMP_SVD_TOL)
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
    def lu_aux(
        lu: F32Ptr, piv: I32Ptr, pm: F32Ptr, im: F32Ptr, diag: F32Ptr, stats: F32Ptr, n: Int, clamp: Int
    ) raises:
        var v = SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0)
        for i in range(n):
            var p = lu_perm_src(piv, i)
            pm.unsafe_store(i, Float32(p))
            im.unsafe_store(p, Float32(i))
            v = lu_aux_join(v, lu_aux_val(lu, piv, i, n))
        for c in range(4):
            stats.unsafe_store(c, v[c])
        for i in range(n):
            lu_aux_clamp(lu, diag, v[0], i, n, clamp != 0)

    @staticmethod
    def lars_rows(g: F32Ptr, q: F32Ptr, w: F32Ptr, na: F32Ptr, n: Int, k: Int, m: Int, nnz: Int) raises:
        var sl = List[Float32](length=max(n * (k * k + LARS_ROW_EXTRA * k), 1), fill=Float32(0))
        var s = F32Ptr(unsafe_from_address=Int(sl.unsafe_ptr()))

        def row(i: Int) {imm g, imm q, imm w, imm s, imm na, imm k, imm m, imm nnz}:
            na.unsafe_store(i, lars_row(g, q, w, s, i, k, m, nnz))

        xd_parallel(row, n)
        _ = sl^

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
        # lda_doc_row's statements, the folds' lanes across words then topics
        # (x_decomp/host_lda.mojo); s (the cell's scratch) is not needed
        var ewt = lda_pack_t(ew, k, v)
        var pt = F32Ptr(unsafe_from_address=Int(ewt.unsafe_ptr()))

        def row(i: Int) {imm x, imm ew, imm pt, imm d, imm e, imm its, imm k, imm v, imm prior, imm max_iter, imm tol}:
            its.unsafe_store(i, lda_doc_row_host(x, ew, pt, d, e, i, k, v, prior, max_iter, tol))

        xd_parallel(row, n)
        _ = ewt^

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
    def geqrf(a: F32Ptr, tau: F32Ptr, m: Int, n: Int) raises:
        """The sliced order's host replay (x_decomp/qr_sliced_host.mojo)."""
        qs_geqrf_host(a, tau, m, n)

    @staticmethod
    def orgqr(h: F32Ptr, tau: F32Ptr, q: F32Ptr, m: Int, n: Int, kk: Int, qc: Int) raises:
        qs_orgqr_host(h, tau, q, m, n, kk, qc)

    @staticmethod
    def als_cg_rows(c: F32Ptr, y: F32Ptr, yty: F32Ptr, x: F32Ptr, steps: F32Ptr, n: Int, m: Int, f: Int, reg: Float32, cg: Int) raises:
        var s = List[Float32](length=n * 3 * f if n * f > 0 else 1, fill=Float32(0))
        var ps = F32Ptr(unsafe_from_address=Int(s.unsafe_ptr()))
        for u in range(n):
            steps.unsafe_store(u, als_cg_row(c, y, yty, x, ps, u, m, f, reg, cg))
        _ = s^

    @staticmethod
    def qr_r(a: F32Ptr, m: Int, n: Int, r: F32Ptr) raises:
        var got = HostExec._qr_r(a, m, n)
        for t in range(n * n):
            r.unsafe_store(t, got[t])

    @staticmethod
    def tsqr_factor(a: F32Ptr, b: F32Ptr, r: F32Ptr, m: Int, d: Int, nrhs: Int, keep: Bool) raises:
        """The blocked TSQR's host replay (x_decomp/tsqr_host.mojo)."""
        ts_factor_host(a, b, r, m, d, nrhs, keep)

    @staticmethod
    def tsqr_apply(c: F32Ptr, q: F32Ptr, m: Int, n: Int, k: Int) raises:
        if k == 0:
            ts_free_host()
            return
        ts_apply_host(c, q, m, n, k)

    @staticmethod
    def vendor() -> String:
        return String("cpu")


def _host_eigh_rr_one(mut m: List[Float32], w: F32Ptr, v: F32Ptr, n: Int) raises:
    """`host_eigh_rr_sorted` (x_decomp/rr_solve.mojo) into w and v:
    DevExec.eigh's and rr_batch_kernel's words."""
    var wl = List[Float32]()
    var vl = List[Float32]()
    _ = host_eigh_rr_sorted(m, n, wl, vl)
    for i in range(n):
        w.unsafe_store(i, wl[i])
    for i in range(n * n):
        v.unsafe_store(i, vl[i])
