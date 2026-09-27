# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DevExec: the decomp lane's cells on the GPU. One thread per output, the
cell of `x_decomp/cells.mojo` verbatim; the serial routines run on ONE
device thread. Host in, host dst: upload, launch, download."""
from std.gpu import block_dim, block_idx, thread_idx
from std.ffi import _Global
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.vendor import COMPILED_VENDOR
from core.householder_qr import qr_factor, qr_slice_count
from decomposition.impl.linalg.detail.svd_full import svd_of_r
from decomposition.linalg_public_device import device_eigh, device_qr_r
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
    als_cg_row,
    geqrf_serial,
    orgqr_col,
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
    pdist_cell,
    sqdist_cell,
)
from x_decomp.exec_trait import Exec


struct _XdContext(Defaultable, Movable):
    """ONE process-lifetime DeviceContext for every x_decomp entry (the x_cnn
    `_Global` pattern, CURRENT DIRECTIVES 2026-09-27): a context per call
    exhausts Metal's per-process command queues within one fit, and a
    fit here is hundreds of calls. Every entry still frees and drains its
    own buffers before returning; the context outlives them all."""
    var ctx: Optional[DeviceContext]

    def __init__(out self):
        self.ctx = Optional[DeviceContext]()


comptime X_DECOMP_CONTEXT = _Global[StorageType=_XdContext, name="MojoXDecompContextIdentical", init_fn=_XdContext.__init__]


def xd_ctx() raises -> DeviceContext:
    """The shared context, created on first use."""
    var slot = X_DECOMP_CONTEXT.get_or_create_ptr()
    if not slot[].ctx:
        slot[].ctx = DeviceContext()
    return slot[].ctx.value().copy()

comptime TPB = 128


def gemm_kernel(a: F32Ptr, b: F32Ptr, c: F32Ptr, m: Int32, k: Int32, n: Int32, ta: Int32, tb: Int32):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(m) * Int(n):
        var i = t // Int(n)
        var j = t % Int(n)
        c.unsafe_store(t, gemm_cell(a, b, i, j, Int(m), Int(k), Int(n), ta != 0, tb != 0))


def gemm_part_kernel(
    a: F32Ptr, b: F32Ptr, p: F32Ptr, m: Int32, k: Int32, n: Int32, ta: Int32, tb: Int32, nb: Int32
):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var mn = Int(m) * Int(n)
    if t < mn * Int(nb):
        var bl = t // mn
        var c = t % mn
        p.unsafe_store(t, gemm_part_cell(
            a, b, c // Int(n), c % Int(n), Int(m), Int(k), Int(n), ta != 0, tb != 0,
            bl * FOLD_BLOCK, min(Int(k), (bl + 1) * FOLD_BLOCK)))


def fold_kernel(p: F32Ptr, dst: F32Ptr, count: Int32, nb: Int32):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(count):
        dst.unsafe_store(t, fold_cell(p, t, Int(nb), Int(count)))


def colsum_part_kernel(a: F32Ptr, p: F32Ptr, n: Int32, d: Int32, nb: Int32):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(d) * Int(nb):
        var bl = t // Int(d)
        var j = t % Int(d)
        p.unsafe_store(t, colsum_part_cell(a, j, Int(n), Int(d), bl * FOLD_BLOCK, min(Int(n), (bl + 1) * FOLD_BLOCK)))


def rowsum_part_kernel(a: F32Ptr, p: F32Ptr, n: Int32, d: Int32, nb: Int32):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(n) * Int(nb):
        var bl = t // Int(n)
        var i = t % Int(n)
        p.unsafe_store(t, rowsum_part_cell(a, i, Int(d), bl * FOLD_BLOCK, min(Int(d), (bl + 1) * FOLD_BLOCK)))


def absmax_kernel(a: F32Ptr, dst: F32Ptr, n: Int32, d: Int32, by_col: Int32):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var cnt = Int(d) if by_col != 0 else Int(n)
    if t < cnt:
        dst.unsafe_store(t, absmax_sign_cell(a, t, Int(n), Int(d), by_col != 0))


def ew_kernel(
    op: Int32, a: F32Ptr, b: F32Ptr, bm: Int32, c: F32Ptr, cm: Int32, dst: F32Ptr,
    count: Int32, d: Int32, s: Float32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(count):
        dst.unsafe_store(
            i,
            ew_cell(
                Int(op), a.unsafe_load(i), b.unsafe_load(bidx(Int(bm), i, Int(d))),
                c.unsafe_load(bidx(Int(cm), i, Int(d))), s,
            ),
        )


def colsum_kernel(a: F32Ptr, dst: F32Ptr, n: Int32, d: Int32):
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if j < Int(d):
        dst.unsafe_store(j, colsum_cell(a, j, Int(n), Int(d)))


def rowsum_kernel(a: F32Ptr, dst: F32Ptr, n: Int32, d: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        dst.unsafe_store(i, rowsum_cell(a, i, Int(d)))


def sqdist_kernel(a: F32Ptr, b: F32Ptr, dst: F32Ptr, na: Int32, nb: Int32, d: Int32, kind: Int32, pw: Float32):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(na) * Int(nb):
        if kind == 0:
            dst.unsafe_store(t, sqdist_cell(a, b, t // Int(nb), t % Int(nb), Int(d)))
        else:
            dst.unsafe_store(t, pdist_cell(a, b, t // Int(nb), t % Int(nb), Int(d), Int(kind), pw))


def rand_kernel(dst: F32Ptr, count: Int32, seed: UInt32, stream: UInt32, kind: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(count):
        dst.unsafe_store(i, rand_cell(i, seed, stream, Int(kind)))


def lu_kernel(a: F32Ptr, piv: I32Ptr, info: F32Ptr, n: Int32):
    if block_idx.x == 0 and thread_idx.x == 0:
        lu_serial(a, piv, Int(n), info)


def lu_solve_kernel(lu: F32Ptr, piv: I32Ptr, b: F32Ptr, n: Int32, nrhs: Int32, trans: Int32):
    if block_idx.x == 0 and thread_idx.x == 0:
        lu_solve_serial(lu, piv, b, Int(n), Int(nrhs), Int(trans))


def chol_kernel(a: F32Ptr, info: F32Ptr, n: Int32):
    if block_idx.x == 0 and thread_idx.x == 0:
        chol_serial(a, Int(n), info)


def cd_rows_kernel(w: F32Ptr, hht: F32Ptr, xht: F32Ptr, perm: I32Ptr, viol: F32Ptr, n: Int32, k: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        viol.unsafe_store(i, cd_row(w, hht, xht, perm, i, Int(k)))


def trsm_kernel(a: F32Ptr, r: F32Ptr, q: F32Ptr, m: Int32, l: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(m):
        trsm_row(a, r, q, i, Int(l))


def lasso_rows_kernel(
    g: F32Ptr, q: F32Ptr, w: F32Ptr, h: F32Ptr, its: F32Ptr, n: Int32, k: Int32, alpha: Float32,
    max_iter: Int32, tol: Float32, positive: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        its.unsafe_store(i, lasso_row(g, q, w, h, i, Int(k), alpha, Int(max_iter), tol, positive != 0))


def omp_rows_kernel(g: F32Ptr, q: F32Ptr, w: F32Ptr, s: F32Ptr, na: F32Ptr, n: Int32, k: Int32, nnz: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        na.unsafe_store(i, omp_row(g, q, w, s, i, Int(k), Int(nnz)))


def gamma_kernel(dst: F32Ptr, count: Int32, seed: UInt32, stream: UInt32, shape: Float32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(count):
        dst.unsafe_store(i, gamma_cell(i, seed, stream, shape))


def lda_rows_kernel(
    x: F32Ptr, ew: F32Ptr, d: F32Ptr, e: F32Ptr, s: F32Ptr, its: F32Ptr, n: Int32, k: Int32, v: Int32,
    prior: Float32, max_iter: Int32, tol: Float32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        its.unsafe_store(i, lda_doc_row(x, ew, d, e, s, i, Int(k), Int(v), prior, Int(max_iter), tol))


def dijkstra_kernel(w: F32Ptr, dist: F32Ptr, done: F32Ptr, reached: F32Ptr, n: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        reached.unsafe_store(i, dijkstra_row(w, dist, done, i, Int(n)))


def barycenter_kernel(
    x: F32Ptr, y: F32Ptr, nbr: F32Ptr, wt: F32Ptr, s: F32Ptr, flags: F32Ptr, n: Int32, d: Int32, k: Int32, reg: Float32
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        flags.unsafe_store(i, barycenter_row(x, y, nbr, wt, s, i, Int(d), Int(k), reg))


def als_kernel(
    c: F32Ptr, y: F32Ptr, yty: F32Ptr, x: F32Ptr, s: F32Ptr, flags: F32Ptr, n: Int32, m: Int32, f: Int32, reg: Float32
):
    var u = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if u < Int(n):
        flags.unsafe_store(u, als_row(c, y, yty, x, s, u, Int(m), Int(f), reg))


def als_cg_kernel(
    c: F32Ptr, y: F32Ptr, yty: F32Ptr, x: F32Ptr, s: F32Ptr, steps: F32Ptr, n: Int32, m: Int32, f: Int32, reg: Float32,
    cg: Int32,
):
    var u = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if u < Int(n):
        steps.unsafe_store(u, als_cg_row(c, y, yty, x, s, u, Int(m), Int(f), reg, Int(cg)))


def geqrf_kernel(a: F32Ptr, tau: F32Ptr, m: Int32, n: Int32):
    if block_idx.x == 0 and thread_idx.x == 0:
        geqrf_serial(a, tau, Int(m), Int(n))


def orgqr_kernel(h: F32Ptr, tau: F32Ptr, q: F32Ptr, m: Int32, n: Int32, kk: Int32, qc: Int32):
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if j < Int(qc):
        orgqr_col(h, tau, q, j, Int(m), Int(n), Int(kk), Int(qc))


def _blocks(count: Int) -> Int:
    return (count + TPB - 1) // TPB if count > 0 else 1


def _up(ctx: DeviceContext, p: F32Ptr, n: Int) raises -> DeviceBuffer[DType.float32]:
    var buf = ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
    if n > 0:
        ctx.enqueue_copy(dst_buf=buf.create_sub_buffer[DType.float32](0, n), src_ptr=p)
    return buf^


def _up_i(ctx: DeviceContext, p: I32Ptr, n: Int) raises -> DeviceBuffer[DType.int32]:
    var buf = ctx.enqueue_create_buffer[DType.int32](n if n > 0 else 1)
    if n > 0:
        ctx.enqueue_copy(dst_buf=buf.create_sub_buffer[DType.int32](0, n), src_ptr=p)
    return buf^


def _down(ctx: DeviceContext, buf: DeviceBuffer[DType.float32], p: F32Ptr, n: Int) raises:
    if n > 0:
        ctx.enqueue_copy(dst_ptr=p, src_buf=buf.create_sub_buffer[DType.float32](0, n))


def _down_i(ctx: DeviceContext, buf: DeviceBuffer[DType.int32], p: I32Ptr, n: Int) raises:
    if n > 0:
        ctx.enqueue_copy(dst_ptr=p, src_buf=buf.create_sub_buffer[DType.int32](0, n))


@fieldwise_init
struct DevExec(Exec):
    @staticmethod
    def gemm(a: F32Ptr, b: F32Ptr, c: F32Ptr, m: Int, k: Int, n: Int, ta: Bool, tb: Bool) raises:
        var ctx = xd_ctx()
        var da = _up(ctx, a, m * k)
        var db = _up(ctx, b, k * n)
        var dc = ctx.enqueue_create_buffer[DType.float32](m * n if m * n > 0 else 1)
        var nb = (k + FOLD_BLOCK - 1) // FOLD_BLOCK
        var dp = ctx.enqueue_create_buffer[DType.float32](nb * m * n if nb > 1 else 1)
        if nb > 1:
            ctx.enqueue_function[gemm_part_kernel](
                da.unsafe_ptr(), db.unsafe_ptr(), dp.unsafe_ptr(), Int32(m), Int32(k), Int32(n),
                Int32(1 if ta else 0), Int32(1 if tb else 0), Int32(nb), grid_dim=_blocks(nb * m * n), block_dim=TPB,
            )
            ctx.enqueue_function[fold_kernel](
                dp.unsafe_ptr(), dc.unsafe_ptr(), Int32(m * n), Int32(nb), grid_dim=_blocks(m * n), block_dim=TPB
            )
        else:
            ctx.enqueue_function[gemm_kernel](
                da.unsafe_ptr(), db.unsafe_ptr(), dc.unsafe_ptr(), Int32(m), Int32(k), Int32(n),
                Int32(1 if ta else 0), Int32(1 if tb else 0), grid_dim=_blocks(m * n), block_dim=TPB,
            )
        _down(ctx, dc, c, m * n)
        ctx.synchronize()
        _ = da^
        _ = db^
        _ = dc^
        _ = dp^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def ew(
        op: Int, a: F32Ptr, b: F32Ptr, lb: Int, bm: Int, c: F32Ptr, lc: Int, cm: Int,
        dst: F32Ptr, count: Int, d: Int, s: Float32,
    ) raises:
        var ctx = xd_ctx()
        var da = _up(ctx, a, count)
        var db = _up(ctx, b, lb)
        var dc = _up(ctx, c, lc)
        var dout = ctx.enqueue_create_buffer[DType.float32](count if count > 0 else 1)
        ctx.enqueue_function[ew_kernel](
            Int32(op), da.unsafe_ptr(), db.unsafe_ptr(), Int32(bm), dc.unsafe_ptr(), Int32(cm),
            dout.unsafe_ptr(), Int32(count), Int32(d), s, grid_dim=_blocks(count), block_dim=TPB,
        )
        _down(ctx, dout, dst, count)
        ctx.synchronize()
        _ = da^
        _ = db^
        _ = dc^
        _ = dout^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def colsum(a: F32Ptr, dst: F32Ptr, n: Int, d: Int) raises:
        var ctx = xd_ctx()
        var da = _up(ctx, a, n * d)
        var dout = ctx.enqueue_create_buffer[DType.float32](d if d > 0 else 1)
        var nb = (n + FOLD_BLOCK - 1) // FOLD_BLOCK
        var dp = ctx.enqueue_create_buffer[DType.float32](nb * d if nb > 1 else 1)
        if nb > 1:
            ctx.enqueue_function[colsum_part_kernel](
                da.unsafe_ptr(), dp.unsafe_ptr(), Int32(n), Int32(d), Int32(nb), grid_dim=_blocks(nb * d), block_dim=TPB
            )
            ctx.enqueue_function[fold_kernel](dp.unsafe_ptr(), dout.unsafe_ptr(), Int32(d), Int32(nb), grid_dim=_blocks(d), block_dim=TPB)
        else:
            ctx.enqueue_function[colsum_kernel](
                da.unsafe_ptr(), dout.unsafe_ptr(), Int32(n), Int32(d), grid_dim=_blocks(d), block_dim=TPB
            )
        _down(ctx, dout, dst, d)
        ctx.synchronize()
        _ = da^
        _ = dout^
        _ = dp^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def rowsum(a: F32Ptr, dst: F32Ptr, n: Int, d: Int) raises:
        var ctx = xd_ctx()
        var da = _up(ctx, a, n * d)
        var dout = ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
        var nb = (d + FOLD_BLOCK - 1) // FOLD_BLOCK
        var dp = ctx.enqueue_create_buffer[DType.float32](nb * n if nb > 1 else 1)
        if nb > 1:
            ctx.enqueue_function[rowsum_part_kernel](
                da.unsafe_ptr(), dp.unsafe_ptr(), Int32(n), Int32(d), Int32(nb), grid_dim=_blocks(nb * n), block_dim=TPB
            )
            ctx.enqueue_function[fold_kernel](dp.unsafe_ptr(), dout.unsafe_ptr(), Int32(n), Int32(nb), grid_dim=_blocks(n), block_dim=TPB)
        else:
            ctx.enqueue_function[rowsum_kernel](
                da.unsafe_ptr(), dout.unsafe_ptr(), Int32(n), Int32(d), grid_dim=_blocks(n), block_dim=TPB
            )
        _down(ctx, dout, dst, n)
        ctx.synchronize()
        _ = da^
        _ = dout^
        _ = dp^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def sqdist(a: F32Ptr, b: F32Ptr, dst: F32Ptr, na: Int, nb: Int, d: Int, kind: Int = 0, pw: Float32 = Float32(2)) raises:
        var ctx = xd_ctx()
        var da = _up(ctx, a, na * d)
        var db = _up(ctx, b, nb * d)
        var dout = ctx.enqueue_create_buffer[DType.float32](na * nb if na * nb > 0 else 1)
        ctx.enqueue_function[sqdist_kernel](
            da.unsafe_ptr(), db.unsafe_ptr(), dout.unsafe_ptr(), Int32(na), Int32(nb), Int32(d), Int32(kind), pw,
            grid_dim=_blocks(na * nb), block_dim=TPB,
        )
        _down(ctx, dout, dst, na * nb)
        ctx.synchronize()
        _ = da^
        _ = db^
        _ = dout^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def rand(dst: F32Ptr, count: Int, seed: UInt32, stream: UInt32, kind: Int) raises:
        var ctx = xd_ctx()
        var dout = ctx.enqueue_create_buffer[DType.float32](count if count > 0 else 1)
        ctx.enqueue_function[rand_kernel](
            dout.unsafe_ptr(), Int32(count), seed, stream, Int32(kind), grid_dim=_blocks(count), block_dim=TPB
        )
        _down(ctx, dout, dst, count)
        ctx.synchronize()
        _ = dout^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def lu(a: F32Ptr, piv: I32Ptr, info: F32Ptr, n: Int) raises:
        var ctx = xd_ctx()
        var da = _up(ctx, a, n * n)
        var dp = ctx.enqueue_create_buffer[DType.int32](n if n > 0 else 1)
        var di = ctx.enqueue_create_buffer[DType.float32](1)
        ctx.enqueue_function[lu_kernel](
            da.unsafe_ptr(), dp.unsafe_ptr(), di.unsafe_ptr(), Int32(n), grid_dim=1, block_dim=1
        )
        _down(ctx, da, a, n * n)
        _down_i(ctx, dp, piv, n)
        _down(ctx, di, info, 1)
        ctx.synchronize()
        _ = da^
        _ = dp^
        _ = di^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def lu_solve(lu: F32Ptr, piv: I32Ptr, b: F32Ptr, n: Int, nrhs: Int, trans: Int = 0) raises:
        var ctx = xd_ctx()
        var dl = _up(ctx, lu, n * n)
        var dp = _up_i(ctx, piv, n)
        var db = _up(ctx, b, n * nrhs)
        ctx.enqueue_function[lu_solve_kernel](
            dl.unsafe_ptr(), dp.unsafe_ptr(), db.unsafe_ptr(), Int32(n), Int32(nrhs), Int32(trans), grid_dim=1, block_dim=1
        )
        _down(ctx, db, b, n * nrhs)
        ctx.synchronize()
        _ = dl^
        _ = dp^
        _ = db^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def chol(a: F32Ptr, info: F32Ptr, n: Int) raises:
        var ctx = xd_ctx()
        var da = _up(ctx, a, n * n)
        var di = ctx.enqueue_create_buffer[DType.float32](1)
        ctx.enqueue_function[chol_kernel](da.unsafe_ptr(), di.unsafe_ptr(), Int32(n), grid_dim=1, block_dim=1)
        _down(ctx, da, a, n * n)
        _down(ctx, di, info, 1)
        ctx.synchronize()
        _ = da^
        _ = di^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def eigh(a: F32Ptr, w: F32Ptr, v: F32Ptr, n: Int) raises:
        var m = List[Float32](capacity=n * n)
        for i in range(n * n):
            m.append(a.unsafe_load(i))
        var got = device_eigh(xd_ctx(), m, n)
        for i in range(n):
            w.unsafe_store(i, got.w[i])
        for i in range(n * n):
            v.unsafe_store(i, got.v[i])

    @staticmethod
    def cd_rows(w: F32Ptr, hht: F32Ptr, xht: F32Ptr, perm: I32Ptr, viol: F32Ptr, n: Int, k: Int) raises:
        var ctx = xd_ctx()
        var dw = _up(ctx, w, n * k)
        var dh = _up(ctx, hht, k * k)
        var dx = _up(ctx, xht, n * k)
        var dp = _up_i(ctx, perm, k)
        var dv = ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
        ctx.enqueue_function[cd_rows_kernel](
            dw.unsafe_ptr(), dh.unsafe_ptr(), dx.unsafe_ptr(), dp.unsafe_ptr(), dv.unsafe_ptr(), Int32(n), Int32(k),
            grid_dim=_blocks(n), block_dim=TPB,
        )
        _down(ctx, dw, w, n * k)
        _down(ctx, dv, viol, n)
        ctx.synchronize()
        _ = dw^
        _ = dh^
        _ = dx^
        _ = dp^
        _ = dv^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def orth(a: F32Ptr, m: Int, l: Int) raises:
        for _ in range(2):
            var w = List[Float32](capacity=m * l)
            for t in range(m * l):
                w.append(a.unsafe_load(t))
            var r = device_qr_r(xd_ctx(), w, m, l)
            orth_rank_guard(F32Ptr(unsafe_from_address=Int(r.unsafe_ptr())), l)
            var ctx = xd_ctx()
            var da = _up(ctx, F32Ptr(unsafe_from_address=Int(w.unsafe_ptr())), m * l)
            var dr = _up(ctx, F32Ptr(unsafe_from_address=Int(r.unsafe_ptr())), l * l)
            var dq = ctx.enqueue_create_buffer[DType.float32](m * l if m * l > 0 else 1)
            ctx.enqueue_function[trsm_kernel](
                da.unsafe_ptr(), dr.unsafe_ptr(), dq.unsafe_ptr(), Int32(m), Int32(l), grid_dim=_blocks(m), block_dim=TPB
            )
            _down(ctx, dq, a, m * l)
            ctx.synchronize()
            _ = da^
            _ = dr^
            _ = dq^
            ctx.synchronize()
            _ = ctx^
            _ = r^
            _ = w^

    @staticmethod
    def svd(a: F32Ptr, m: Int, n: Int, s: F32Ptr, v: F32Ptr) raises:
        """`device_svdvals`'s route (qr_factor, then svd_of_r) keeping V."""
        var ctx = xd_ctx()
        var da = _up(ctx, a, m * n)
        var scratch = ctx.enqueue_create_buffer[DType.float32](qr_slice_count(m, n) * n * n)
        var r_buf = ctx.enqueue_create_buffer[DType.float32](n * n)
        var v_buf = ctx.enqueue_create_buffer[DType.float32](n * n)
        var s_buf = ctx.enqueue_create_buffer[DType.float32](n)
        ctx.synchronize()
        _ = qr_factor(ctx, da, scratch, r_buf, m, n)
        svd_of_r(ctx, r_buf, v_buf, s_buf, n, X_DECOMP_SVD_SWEEPS, X_DECOMP_SVD_TOL)
        _down(ctx, s_buf, s, n)
        _down(ctx, v_buf, v, n * n)
        ctx.synchronize()
        _ = da^
        _ = scratch^
        _ = r_buf^
        _ = v_buf^
        _ = s_buf^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def lasso_rows(
        g: F32Ptr, q: F32Ptr, w: F32Ptr, h: F32Ptr, its: F32Ptr, n: Int, k: Int, alpha: Float32,
        max_iter: Int, tol: Float32, positive: Bool,
    ) raises:
        var ctx = xd_ctx()
        var dg = _up(ctx, g, k * k)
        var dq = _up(ctx, q, n * k)
        var dw = _up(ctx, w, n * k)
        var dh = ctx.enqueue_create_buffer[DType.float32](n * k if n * k > 0 else 1)
        var di = ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
        ctx.enqueue_function[lasso_rows_kernel](
            dg.unsafe_ptr(), dq.unsafe_ptr(), dw.unsafe_ptr(), dh.unsafe_ptr(), di.unsafe_ptr(), Int32(n), Int32(k),
            alpha, Int32(max_iter), tol, Int32(1 if positive else 0), grid_dim=_blocks(n), block_dim=TPB,
        )
        _down(ctx, dw, w, n * k)
        _down(ctx, di, its, n)
        ctx.synchronize()
        _ = dg^
        _ = dq^
        _ = dw^
        _ = dh^
        _ = di^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def omp_rows(g: F32Ptr, q: F32Ptr, w: F32Ptr, s: F32Ptr, na: F32Ptr, n: Int, k: Int, nnz: Int) raises:
        var ctx = xd_ctx()
        var per = k * k + 3 * k
        var dg = _up(ctx, g, k * k)
        var dq = _up(ctx, q, n * k)
        var dw = ctx.enqueue_create_buffer[DType.float32](n * k if n * k > 0 else 1)
        var ds = ctx.enqueue_create_buffer[DType.float32](n * per if n * per > 0 else 1)
        var dn = ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
        ctx.enqueue_function[omp_rows_kernel](
            dg.unsafe_ptr(), dq.unsafe_ptr(), dw.unsafe_ptr(), ds.unsafe_ptr(), dn.unsafe_ptr(), Int32(n), Int32(k),
            Int32(nnz), grid_dim=_blocks(n), block_dim=TPB,
        )
        _down(ctx, dw, w, n * k)
        _down(ctx, dn, na, n)
        ctx.synchronize()
        _ = dg^
        _ = dq^
        _ = dw^
        _ = ds^
        _ = dn^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def rand_gamma(dst: F32Ptr, count: Int, seed: UInt32, stream: UInt32, shape: Float32) raises:
        var ctx = xd_ctx()
        var dout = ctx.enqueue_create_buffer[DType.float32](count if count > 0 else 1)
        ctx.enqueue_function[gamma_kernel](
            dout.unsafe_ptr(), Int32(count), seed, stream, shape, grid_dim=_blocks(count), block_dim=TPB
        )
        _down(ctx, dout, dst, count)
        ctx.synchronize()
        _ = dout^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def lda_rows(
        x: F32Ptr, ew: F32Ptr, d: F32Ptr, e: F32Ptr, s: F32Ptr, its: F32Ptr, n: Int, k: Int, v: Int,
        prior: Float32, max_iter: Int, tol: Float32,
    ) raises:
        var ctx = xd_ctx()
        var dx = _up(ctx, x, n * v)
        var dw = _up(ctx, ew, k * v)
        var dd = _up(ctx, d, n * k)
        var de = _up(ctx, e, n * k)
        var ds = ctx.enqueue_create_buffer[DType.float32](n * (v + k) if n > 0 else 1)
        var di = ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
        ctx.enqueue_function[lda_rows_kernel](
            dx.unsafe_ptr(), dw.unsafe_ptr(), dd.unsafe_ptr(), de.unsafe_ptr(), ds.unsafe_ptr(), di.unsafe_ptr(),
            Int32(n), Int32(k), Int32(v), prior, Int32(max_iter), tol, grid_dim=_blocks(n), block_dim=TPB,
        )
        _down(ctx, dd, d, n * k)
        _down(ctx, de, e, n * k)
        _down(ctx, di, its, n)
        ctx.synchronize()
        _ = dx^
        _ = dw^
        _ = dd^
        _ = de^
        _ = ds^
        _ = di^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def dijkstra_rows(w: F32Ptr, dist: F32Ptr, reached: F32Ptr, n: Int) raises:
        var ctx = xd_ctx()
        var dw = _up(ctx, w, n * n)
        var dd = ctx.enqueue_create_buffer[DType.float32](n * n if n > 0 else 1)
        var dn = ctx.enqueue_create_buffer[DType.float32](n * n if n > 0 else 1)
        var dr = ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
        ctx.enqueue_function[dijkstra_kernel](
            dw.unsafe_ptr(), dd.unsafe_ptr(), dn.unsafe_ptr(), dr.unsafe_ptr(), Int32(n), grid_dim=_blocks(n), block_dim=TPB
        )
        _down(ctx, dd, dist, n * n)
        _down(ctx, dr, reached, n)
        ctx.synchronize()
        _ = dw^
        _ = dd^
        _ = dn^
        _ = dr^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def barycenter_rows(
        x: F32Ptr, y: F32Ptr, nbr: F32Ptr, wt: F32Ptr, flags: F32Ptr, n: Int, ny: Int, d: Int, k: Int, reg: Float32
    ) raises:
        var ctx = xd_ctx()
        var dx = _up(ctx, x, n * d)
        var dy = _up(ctx, y, ny * d)
        var dnb = _up(ctx, nbr, n * k)
        var dwt = ctx.enqueue_create_buffer[DType.float32](n * k if n * k > 0 else 1)
        var ds = ctx.enqueue_create_buffer[DType.float32](n * (k * k + k * d) if n > 0 else 1)
        var df = ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
        ctx.enqueue_function[barycenter_kernel](
            dx.unsafe_ptr(), dy.unsafe_ptr(), dnb.unsafe_ptr(), dwt.unsafe_ptr(), ds.unsafe_ptr(), df.unsafe_ptr(),
            Int32(n), Int32(d), Int32(k), reg, grid_dim=_blocks(n), block_dim=TPB,
        )
        _down(ctx, dwt, wt, n * k)
        _down(ctx, df, flags, n)
        ctx.synchronize()
        _ = dx^
        _ = dy^
        _ = dnb^
        _ = dwt^
        _ = ds^
        _ = df^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def als_rows(c: F32Ptr, y: F32Ptr, yty: F32Ptr, x: F32Ptr, flags: F32Ptr, n: Int, m: Int, f: Int, reg: Float32) raises:
        var ctx = xd_ctx()
        var dc = _up(ctx, c, n * m)
        var dy = _up(ctx, y, m * f)
        var dg = _up(ctx, yty, f * f)
        var dx = ctx.enqueue_create_buffer[DType.float32](n * f if n * f > 0 else 1)
        var ds = ctx.enqueue_create_buffer[DType.float32](n * (f * f + f) if n > 0 else 1)
        var df = ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
        ctx.enqueue_function[als_kernel](
            dc.unsafe_ptr(), dy.unsafe_ptr(), dg.unsafe_ptr(), dx.unsafe_ptr(), ds.unsafe_ptr(), df.unsafe_ptr(),
            Int32(n), Int32(m), Int32(f), reg, grid_dim=_blocks(n), block_dim=TPB,
        )
        _down(ctx, dx, x, n * f)
        _down(ctx, df, flags, n)
        ctx.synchronize()
        _ = dc^
        _ = dy^
        _ = dg^
        _ = dx^
        _ = ds^
        _ = df^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def absmax_sign(a: F32Ptr, dst: F32Ptr, n: Int, d: Int, by_col: Bool) raises:
        var ctx = xd_ctx()
        var da = _up(ctx, a, n * d)
        var cnt = d if by_col else n
        var dout = ctx.enqueue_create_buffer[DType.float32](cnt if cnt > 0 else 1)
        ctx.enqueue_function[absmax_kernel](
            da.unsafe_ptr(), dout.unsafe_ptr(), Int32(n), Int32(d), Int32(1 if by_col else 0),
            grid_dim=_blocks(cnt), block_dim=TPB,
        )
        _down(ctx, dout, dst, cnt)
        ctx.synchronize()
        _ = da^
        _ = dout^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def geqrf(a: F32Ptr, tau: F32Ptr, m: Int, n: Int) raises:
        var ctx = xd_ctx()
        var kk = m if m < n else n
        var da = _up(ctx, a, m * n)
        var dt = ctx.enqueue_create_buffer[DType.float32](kk if kk > 0 else 1)
        ctx.enqueue_function[geqrf_kernel](da.unsafe_ptr(), dt.unsafe_ptr(), Int32(m), Int32(n), grid_dim=1, block_dim=1)
        _down(ctx, da, a, m * n)
        _down(ctx, dt, tau, kk)
        ctx.synchronize()
        _ = da^
        _ = dt^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def orgqr(h: F32Ptr, tau: F32Ptr, q: F32Ptr, m: Int, n: Int, kk: Int, qc: Int) raises:
        var ctx = xd_ctx()
        var dh = _up(ctx, h, m * n)
        var dt = _up(ctx, tau, kk if kk > 0 else 1)
        var dq = ctx.enqueue_create_buffer[DType.float32](m * qc if m * qc > 0 else 1)
        ctx.enqueue_function[orgqr_kernel](
            dh.unsafe_ptr(), dt.unsafe_ptr(), dq.unsafe_ptr(), Int32(m), Int32(n), Int32(kk), Int32(qc),
            grid_dim=_blocks(qc), block_dim=TPB,
        )
        _down(ctx, dq, q, m * qc)
        ctx.synchronize()
        _ = dh^
        _ = dt^
        _ = dq^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def als_cg_rows(c: F32Ptr, y: F32Ptr, yty: F32Ptr, x: F32Ptr, steps: F32Ptr, n: Int, m: Int, f: Int, reg: Float32, cg: Int) raises:
        var ctx = xd_ctx()
        var dc = _up(ctx, c, n * m)
        var dy = _up(ctx, y, m * f)
        var dg = _up(ctx, yty, f * f)
        var dx = _up(ctx, x, n * f)
        var ds = ctx.enqueue_create_buffer[DType.float32](n * 3 * f if n * f > 0 else 1)
        var dstp = ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
        ctx.enqueue_function[als_cg_kernel](
            dc.unsafe_ptr(), dy.unsafe_ptr(), dg.unsafe_ptr(), dx.unsafe_ptr(), ds.unsafe_ptr(), dstp.unsafe_ptr(),
            Int32(n), Int32(m), Int32(f), reg, Int32(cg), grid_dim=_blocks(n), block_dim=TPB,
        )
        _down(ctx, dx, x, n * f)
        _down(ctx, dstp, steps, n)
        ctx.synchronize()
        _ = dc^
        _ = dy^
        _ = dg^
        _ = dx^
        _ = ds^
        _ = dstp^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def qr_r(a: F32Ptr, m: Int, n: Int, r: F32Ptr) raises:
        var w = List[Float32](capacity=m * n)
        for t in range(m * n):
            w.append(a.unsafe_load(t))
        var got = device_qr_r(xd_ctx(), w, m, n)
        for t in range(n * n):
            r.unsafe_store(t, got[t])

    @staticmethod
    def vendor() -> String:
        return String(COMPILED_VENDOR)
