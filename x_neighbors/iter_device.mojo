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

from x_neighbors.items import FP, IP
from x_neighbors.device_ops import (
    xn_ctx, _buf, _buf_i, _down, _down_i, _grid, BLOCK,
    absdiff_sum_kernel, matmul_kernel, lp_clamp_kernel, ls_clamp_kernel,
    pagerank_step_kernel, cc_step_kernel,
)


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
    for it in range(max_iter):
        n_iter = it
        ctx.enqueue_function[absdiff_sum_kernel](
            cur, prev, d_s.unsafe_ptr(), Int64(nc), grid_dim=1, block_dim=1,
        )
        ctx.enqueue_copy(dst_ptr=hs.unsafe_ptr(), src_buf=d_s)
        ctx.synchronize()
        if Float64(hs[0]) < tol:
            converged = True
            break
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
