# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Resident GPU drivers of the lane's iterations (lane neighbors-apple2).

`op_lp_iterate`: LabelPropagation / LabelSpreading's fit loop
(`_LabelPropagationBase.fit` in python/mojolearn/_expansion_neighbors.py) on
the device with the graph uploaded ONCE. The Python loop called three ops per
iteration, and each op uploaded its inputs: the n x n graph crossed to the
device on every iteration. Here the same kernels (`absdiff_sum_kernel`,
`matmul_kernel`, `lp_clamp_kernel` / `ls_clamp_kernel`, the generated
one-item-per-thread drivers of the same items) run in the same order on the
same values; only the stopping sum crosses back, one float per iteration,
compared in double exactly as Python compared it. The CPU column runs the
same loop over the items (`x_neighbors/iter_host.mojo`).
"""
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.numerics import ftz, identical_mul_add

from std.sys.compile import is_defined
from x_neighbors.items import FP, IP, absdiff_sum_item, _sub, knn_sq_item
from std.memory import bitcast as _bc
from x_neighbors.device_ops import (
    xn_ctx, _buf, _buf_i, _down, _down_i, _grid, BLOCK,
    absdiff_sum_kernel, matmul_kernel, lp_clamp_kernel, ls_clamp_kernel,
    pagerank_step_kernel, cc_step_kernel,
    pcs_sketch_kernel, pcs_conv_kernel, pcs_copy0_kernel, op_knn_sq,
)


def lp_spmm_kernel(
    indptr: IP, cols: IP, vals: FP, x: FP, res: FP, n_: Int64, c_: Int64,
):
    """`matmul_item` (G x, cell t = i*c + j, p ascending) over G's NONZERO
    entries only, columns ascending. Exact when every x is finite: a
    skipped term is fma(+-0, x, acc) with x finite, whose product is a zero
    and whose sum is acc unchanged, because acc starts at +0.0 and an fma
    returns -0.0 only from (-0) + (-0), so acc is never -0.0. The caller
    checks x's finiteness each iteration and runs the dense kernel when it
    fails."""
    var n = Int(n_)
    var c = Int(c_)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= n * c:
        return
    var i = t // c
    var j = t - i * c
    var acc = Float32(0)
    for e in range(Int(indptr.unsafe_load(i)), Int(indptr.unsafe_load(i + 1))):
        acc = ftz(identical_mul_add(ftz(vals.unsafe_load(e)), ftz(x.unsafe_load(Int(cols.unsafe_load(e)) * c + j)), acc))
    res.unsafe_store(t, acc)


def op_lp_iterate(
    g: Int, ld: Int, ystatic: Int, unlabeled: Int, info: Int,
    n: Int, c: Int, max_iter: Int, variant: Int, tol_hi: Int, tol_lo: Int,
    alpha: Float32,
) raises:
    """variant 0 propagation (lp_clamp), 1 spreading (ls_clamp). `ld` in:
    the initial label distributions, out: the last. info (int32 x 2): the
    fit's n_iter_ and converged."""
    var tol = bitcast[DType.float64]((UInt64(tol_hi) << UInt64(32)) | UInt64(tol_lo))
    var nc = n * c
    var ctx = xn_ctx()
    var d_g = _buf(ctx, g, n * n, True)
    var d_a = _buf(ctx, ld, nc, True)
    var d_b = _buf(ctx, 0, nc, False)
    ctx.enqueue_memset(d_b, Float32(0))
    var d_nxt = _buf(ctx, 0, nc, False)
    var d_ys = _buf(ctx, ystatic, nc, True)
    var d_unl = _buf_i(ctx, unlabeled, n, True)
    var d_s = _buf(ctx, 0, 1, False)
    var hs = List[Float32](length=1, fill=Float32(0))
    var cur: FP = d_a.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var prev: FP = d_b.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var cur_is_a = True
    var n_iter = 0
    var converged = False
    # The stopping sum's ONE-ITEM fold over n * c values ran on ONE GPU
    # thread every iteration (1.5 ms at n = 5,000 with many classes). By
    # default the host folds it: the current distributions come back each
    # iteration (they are what the step just wrote), the previous ones are
    # the host copy from the iteration before, and `absdiff_sum_item` runs
    # its same statements on them. `-D MOJOLEARN_XN_LP_DEVICE_FOLD` keeps
    # the device fold.
    var hc = List[Float32](length=nc if nc > 0 else 1, fill=Float32(0))
    var hp = List[Float32](length=nc if nc > 0 else 1, fill=Float32(0))
    # G's nonzero entries (a knn graph holds ~k per row), row-major CSR,
    # built once on the host from the caller's array. Sparse only when it
    # pays (under an eighth nonzero) and only in the host-fold path, which
    # has each iteration's x on the host for the finiteness test.
    # `-D MOJOLEARN_XN_LP_DENSE` keeps the dense product.
    var pg = FP(unsafe_from_address=g)
    var h_ip = List[Int32](length=n + 1, fill=Int32(0))
    var h_cols = List[Int32]()
    var h_vals = List[Float32]()
    var use_sparse = False
    comptime if not is_defined["MOJOLEARN_XN_LP_DENSE"]() and not is_defined["MOJOLEARN_XN_LP_DEVICE_FOLD"]():
        var nnz = 0
        for q in range(n * n):
            if pg.unsafe_load(q) != Float32(0):
                nnz += 1
        use_sparse = nnz > 0 and nnz * 8 < n * n
        if use_sparse:
            for i in range(n):
                for jj in range(n):
                    var v = pg.unsafe_load(i * n + jj)
                    if v != Float32(0):
                        h_cols.append(Int32(jj))
                        h_vals.append(v)
                h_ip[i + 1] = Int32(len(h_cols))
    var nnzb = len(h_cols) if len(h_cols) > 0 else 1
    var d_ip = ctx.enqueue_create_buffer[DType.int32](n + 1)
    var d_cols = ctx.enqueue_create_buffer[DType.int32](nnzb)
    var d_vals = ctx.enqueue_create_buffer[DType.float32](nnzb)
    if use_sparse:
        ctx.enqueue_copy(dst_buf=d_ip, src_ptr=h_ip.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=d_cols, src_ptr=h_cols.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=d_vals, src_ptr=h_vals.unsafe_ptr())
    for it in range(max_iter):
        n_iter = it
        comptime if is_defined["MOJOLEARN_XN_LP_DEVICE_FOLD"]():
            ctx.enqueue_function[absdiff_sum_kernel](
                cur, prev, d_s.unsafe_ptr(), Int64(nc), grid_dim=1, block_dim=1,
            )
            ctx.enqueue_copy(dst_ptr=hs.unsafe_ptr(), src_buf=d_s)
            ctx.synchronize()
        else:
            if cur_is_a:
                ctx.enqueue_copy(dst_ptr=hc.unsafe_ptr(), src_buf=d_a)
            else:
                ctx.enqueue_copy(dst_ptr=hc.unsafe_ptr(), src_buf=d_b)
            ctx.synchronize()
            absdiff_sum_item(
                0, FP(unsafe_from_address=Int(hc.unsafe_ptr())),
                FP(unsafe_from_address=Int(hp.unsafe_ptr())),
                FP(unsafe_from_address=Int(hs.unsafe_ptr())), nc,
            )
            for q in range(nc):
                hp[q] = hc[q]
        if Float64(hs[0]) < tol:
            converged = True
            break
        var sparse_now = use_sparse
        if sparse_now:
            for q in range(nc):
                var bits = bitcast[DType.uint32](hc[q]) & UInt32(0x7F800000)
                if bits == UInt32(0x7F800000):
                    sparse_now = False
                    break
        if sparse_now:
            ctx.enqueue_function[lp_spmm_kernel](
                d_ip.unsafe_ptr(), d_cols.unsafe_ptr(), d_vals.unsafe_ptr(), cur, d_nxt.unsafe_ptr(),
                Int64(n), Int64(c), grid_dim=_grid(nc), block_dim=(BLOCK if nc > 1 else 1),
            )
        else:
            ctx.enqueue_function[matmul_kernel](
                d_g.unsafe_ptr(), cur, d_nxt.unsafe_ptr(), Int64(n), Int64(n), Int64(c),
                grid_dim=_grid(nc), block_dim=(BLOCK if nc > 1 else 1),
            )
        # prev = ld; ld = clamp(nxt): the clamp writes over the buffer the
        # old prev held, then the two names swap.
        if variant == 0:
            ctx.enqueue_function[lp_clamp_kernel](
                d_nxt.unsafe_ptr(), d_ys.unsafe_ptr(), d_unl.unsafe_ptr(), prev, Int64(n), Int64(c),
                grid_dim=_grid(n), block_dim=(BLOCK if n > 1 else 1),
            )
        else:
            ctx.enqueue_function[ls_clamp_kernel](
                d_nxt.unsafe_ptr(), d_ys.unsafe_ptr(), prev, Int64(nc), alpha,
                grid_dim=_grid(nc), block_dim=(BLOCK if nc > 1 else 1),
            )
        var t = cur
        cur = prev
        prev = t
        cur_is_a = not cur_is_a
    if not converged:
        n_iter += 1
    if cur_is_a:
        _down(ctx, d_a, ld, nc)
    else:
        _down(ctx, d_b, ld, nc)
    ctx.synchronize()
    var inf = IP(unsafe_from_address=info)
    inf.unsafe_store(0, Int32(n_iter))
    inf.unsafe_store(1, Int32(1 if converged else 0))
    _ = hs^
    _ = hc^
    _ = hp^
    _ = h_ip^
    _ = h_cols^
    _ = h_vals^
    _ = d_ip^
    _ = d_cols^
    _ = d_vals^
    _ = d_g^
    _ = d_a^
    _ = d_b^
    _ = d_nxt^
    _ = d_ys^
    _ = d_unl^
    _ = d_s^
    _ = ctx^


def op_pr_iterate(
    q: Int, x: Int, p: Int, dw: Int, dangling: Int, info: Int,
    n: Int, max_iter: Int, thr_hi: Int, thr_lo: Int, alpha: Float32,
) raises:
    """PageRank.fit's power iteration (`pagerank_step` then `absdiff_sum`
    per iteration, x <- the step) with Q resident. `thr` is Python's
    n * tol as float64 bits. `x` in: the start, out: the last iterate.
    info (int32 x 2): iterations run, converged."""
    var thr = bitcast[DType.float64]((UInt64(thr_hi) << UInt64(32)) | UInt64(thr_lo))
    var ctx = xn_ctx()
    var d_q = _buf(ctx, q, n * n, True)
    var d_a = _buf(ctx, x, n, True)
    var d_b = _buf(ctx, 0, n, False)
    var d_p = _buf(ctx, p, n, True)
    var d_dw = _buf(ctx, dw, n, True)
    var d_dg = _buf_i(ctx, dangling, n, True)
    var d_s = _buf(ctx, 0, 1, False)
    var hs = List[Float32](length=1, fill=Float32(0))
    var cur: FP = d_a.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var nxt: FP = d_b.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var cur_is_a = True
    var n_iter = 0
    var converged = False
    for it in range(max_iter):
        ctx.enqueue_function[pagerank_step_kernel](
            d_q.unsafe_ptr(), cur, d_p.unsafe_ptr(), d_dw.unsafe_ptr(), d_dg.unsafe_ptr(), nxt,
            Int64(n), alpha, grid_dim=_grid(n), block_dim=(BLOCK if n > 1 else 1),
        )
        ctx.enqueue_function[absdiff_sum_kernel](
            nxt, cur, d_s.unsafe_ptr(), Int64(n), grid_dim=1, block_dim=1,
        )
        ctx.enqueue_copy(dst_ptr=hs.unsafe_ptr(), src_buf=d_s)
        ctx.synchronize()
        var t = cur
        cur = nxt
        nxt = t
        cur_is_a = not cur_is_a
        n_iter = it + 1
        if Float64(hs[0]) < thr:
            converged = True
            break
    if cur_is_a:
        _down(ctx, d_a, x, n)
    else:
        _down(ctx, d_b, x, n)
    ctx.synchronize()
    var inf = IP(unsafe_from_address=info)
    inf.unsafe_store(0, Int32(n_iter))
    inf.unsafe_store(1, Int32(1 if converged else 0))
    _ = hs^
    _ = d_q^
    _ = d_a^
    _ = d_b^
    _ = d_p^
    _ = d_dw^
    _ = d_dg^
    _ = d_s^
    _ = ctx^


def op_cc_iterate(a: Int, lab: Int, info: Int, n: Int) raises:
    """connected_components' min-label iteration (`cc_step` until the labels
    stop changing) with A resident; the labels come back each step (n
    integers) for the host's equality test, as Python compared them.
    `lab` in: 0..n-1, out: the fixed point. info (int32 x 1): steps."""
    var ctx = xn_ctx()
    var d_a = _buf(ctx, a, n * n, True)
    var d_l0 = _buf_i(ctx, lab, n, True)
    var d_l1 = _buf_i(ctx, 0, n, False)
    var cur_host = List[Int32](length=n if n > 0 else 1, fill=Int32(0))
    var nxt_host = List[Int32](length=n if n > 0 else 1, fill=Int32(0))
    var pl = IP(unsafe_from_address=lab)
    for i in range(n):
        cur_host[i] = pl.unsafe_load(i)
    var cur: IP = d_l0.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var nxt: IP = d_l1.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var nxt_is_1 = True
    var steps = 0
    while True:
        ctx.enqueue_function[cc_step_kernel](
            d_a.unsafe_ptr(), cur, nxt, Int64(n),
            grid_dim=_grid(n), block_dim=(BLOCK if n > 1 else 1),
        )
        if nxt_is_1:
            ctx.enqueue_copy(dst_ptr=nxt_host.unsafe_ptr(), src_buf=d_l1)
        else:
            ctx.enqueue_copy(dst_ptr=nxt_host.unsafe_ptr(), src_buf=d_l0)
        ctx.synchronize()
        steps += 1
        var same = True
        for i in range(n):
            if nxt_host[i] != cur_host[i]:
                same = False
                break
        if same:
            break
        for i in range(n):
            cur_host[i] = nxt_host[i]
        var t = cur
        cur = nxt
        nxt = t
        nxt_is_1 = not nxt_is_1
    for i in range(n):
        pl.unsafe_store(i, cur_host[i])
    IP(unsafe_from_address=info).unsafe_store(0, Int32(steps))
    _ = cur_host^
    _ = nxt_host^
    _ = d_a^
    _ = d_l0^
    _ = d_l1^
    _ = ctx^


comptime PCS_ROW_TPB = 256
comptime PCS_ROW_MAX_NC = 2048


def pcs_conv_row_kernel(acc: FP, sk: FP, res: FP, n_: Int64, nc_: Int64, degree_: Int64, p_: Int64):
    """`pcs_conv_item` for one ROW per block: the row's running product and
    its degree-p sketch staged in threadgroup memory (read-only after one
    barrier), then each thread folds its components' convolutions with the
    item's statements, a ascending. nc <= PCS_ROW_MAX_NC."""
    var nc = Int(nc_)
    var degree = Int(degree_)
    var p = Int(p_)
    var r = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var ar = stack_allocation[PCS_ROW_MAX_NC, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var sr = stack_allocation[PCS_ROW_MAX_NC, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var q = tid
    while q < nc:
        ar[q] = acc.unsafe_load(r * nc + q)
        sr[q] = sk.unsafe_load((r * degree + p) * nc + q)
        q += PCS_ROW_TPB
    barrier()
    var h = tid
    while h < nc:
        var s = Float32(0)
        for a in range(nc):
            var b = h - a
            if b < 0:
                b += nc
            s = ftz(identical_mul_add(ar[a], sr[b], s))
        res.unsafe_store(r * nc + h, s)
        h += PCS_ROW_TPB


def op_pcs_resident(
    x: Int, hidx: Int, hbit: Int, res: Int,
    n: Int, d_in: Int, nf: Int, nc: Int, degree: Int, gamma: Float32, coef0: Float32,
) raises:
    """PolynomialCountSketch.transform: `pcs_item` (one thread per ROW, the
    convolution O(nc^2) per row) as three kernels over the same statements:
    one sketch per (row, degree), then per degree p >= 1 one thread per
    output cell folding the convolution in the same ascending order, the
    running product ping-ponging on the device."""
    var ctx = xn_ctx()
    var d_x = _buf(ctx, x, n * d_in, True)
    var d_hi = _buf_i(ctx, hidx, degree * nf, True)
    var d_hb = _buf_i(ctx, hbit, degree * nf, True)
    var d_sk = _buf(ctx, 0, n * degree * nc, False)
    var d_a = _buf(ctx, 0, n * nc, False)
    var d_b = _buf(ctx, 0, n * nc, False)
    var nd = n * degree
    var cells = n * nc
    ctx.enqueue_function[pcs_sketch_kernel](
        d_x.unsafe_ptr(), d_hi.unsafe_ptr(), d_hb.unsafe_ptr(), d_sk.unsafe_ptr(),
        Int64(n), Int64(d_in), Int64(nf), Int64(nc), Int64(degree), gamma, coef0,
        grid_dim=_grid(nd), block_dim=(BLOCK if nd > 1 else 1),
    )
    ctx.enqueue_function[pcs_copy0_kernel](
        d_sk.unsafe_ptr(), d_a.unsafe_ptr(), Int64(n), Int64(nc), Int64(degree),
        grid_dim=_grid(cells), block_dim=(BLOCK if cells > 1 else 1),
    )
    var cur: FP = d_a.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var nxt: FP = d_b.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var cur_is_a = True
    for p in range(1, degree):
        var row_kernel = nc <= PCS_ROW_MAX_NC
        comptime if is_defined["MOJOLEARN_XN_PCS_CELL"]():
            row_kernel = False
        if row_kernel:
            ctx.enqueue_function[pcs_conv_row_kernel](
                cur, d_sk.unsafe_ptr(), nxt, Int64(n), Int64(nc), Int64(degree), Int64(p),
                grid_dim=n, block_dim=PCS_ROW_TPB,
            )
        else:
            ctx.enqueue_function[pcs_conv_kernel](
                cur, d_sk.unsafe_ptr(), nxt, Int64(n), Int64(nc), Int64(degree), Int64(p),
                grid_dim=_grid(cells), block_dim=(BLOCK if cells > 1 else 1),
            )
        var t = cur
        cur = nxt
        nxt = t
        cur_is_a = not cur_is_a
    if cur_is_a:
        _down(ctx, d_a, res, cells)
    else:
        _down(ctx, d_b, res, cells)
    ctx.synchronize()
    _ = d_x^
    _ = d_hi^
    _ = d_hb^
    _ = d_sk^
    _ = d_a^
    _ = d_b^
    _ = ctx^


comptime KNN_TILE_TPB = 128
comptime KNN_TILE_ROWS = 64
comptime KNN_TILE_MAX_D = 64


def knn_sq_tiled_kernel(
    x: FP, y: FP, dist: FP, idx: IP, n_: Int64, m_: Int64, d_: Int64, k_: Int64, ex_: Int64,
):
    """`knn_sq_item` for x row t = this thread, with the y rows staged
    KNN_TILE_ROWS at a time in threadgroup memory (read-only between two
    barriers) instead of read by every thread from device memory. Candidate
    columns are offered in the same ascending order to the same strict-<
    insertion, each value by the item's statements, so the lists are the
    item's. d <= KNN_TILE_MAX_D."""
    var n = Int(n_)
    var m = Int(m_)
    var d = Int(d_)
    var k = Int(k_)
    var ex = Int(ex_)
    var tid = Int(thread_idx.x)
    var t = Int(block_idx.x) * KNN_TILE_TPB + tid
    var ys = stack_allocation[KNN_TILE_ROWS * KNN_TILE_MAX_D, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var inf = _bc[DType.float32](UInt32(0x7F800000))
    var live = t < n
    if live:
        for s in range(k):
            dist.unsafe_store(t * k + s, inf)
            idx.unsafe_store(t * k + s, Int32(-1))
    var worst = inf
    var j0 = 0
    while j0 < m:
        var rows = min(KNN_TILE_ROWS, m - j0)
        var q = tid
        while q < rows * d:
            ys[q] = y.unsafe_load(j0 * d + q)
            q += KNN_TILE_TPB
        barrier()
        if live:
            for jj in range(rows):
                var j = j0 + jj
                if ex != 0 and j == t:
                    continue
                var acc = Float32(0)
                for f in range(d):
                    var df = _sub(x.unsafe_load(t * d + f), ys[jj * d + f])
                    acc = ftz(identical_mul_add(df, df, acc))
                var v = acc
                if not (v < worst):
                    continue
                var s = k - 1
                while s > 0 and v < dist.unsafe_load(t * k + s - 1):
                    dist.unsafe_store(t * k + s, dist.unsafe_load(t * k + s - 1))
                    idx.unsafe_store(t * k + s, idx.unsafe_load(t * k + s - 1))
                    s -= 1
                dist.unsafe_store(t * k + s, v)
                idx.unsafe_store(t * k + s, Int32(j))
                worst = dist.unsafe_load(t * k + k - 1)
        barrier()
        j0 += rows


def op_knn_sq_tiled(
    x: Int, y: Int, dist: Int, idx: Int, n: Int, m: Int, d: Int, k: Int, exclude_self: Int,
) raises:
    """The fused k-NN (`knn_sq`) with y staged per block; d above
    KNN_TILE_MAX_D takes the one-thread-per-row item kernel."""
    if d > KNN_TILE_MAX_D:
        op_knn_sq(x, y, dist, idx, n, m, d, k, exclude_self)
        return
    var ctx = xn_ctx()
    var d_x = _buf(ctx, x, n * d, True)
    var d_y = _buf(ctx, y, m * d, True)
    var d_dist = _buf(ctx, 0, n * k, False)
    var d_idx = _buf_i(ctx, 0, n * k, False)
    ctx.enqueue_function[knn_sq_tiled_kernel](
        d_x.unsafe_ptr(), d_y.unsafe_ptr(), d_dist.unsafe_ptr(), d_idx.unsafe_ptr(),
        Int64(n), Int64(m), Int64(d), Int64(k), Int64(exclude_self),
        grid_dim=(n + KNN_TILE_TPB - 1) // KNN_TILE_TPB, block_dim=KNN_TILE_TPB,
    )
    _down(ctx, d_dist, dist, n * k)
    _down_i(ctx, d_idx, idx, n * k)
    ctx.synchronize()
    _ = d_x^
    _ = d_y^
    _ = d_dist^
    _ = d_idx^
    _ = ctx^
