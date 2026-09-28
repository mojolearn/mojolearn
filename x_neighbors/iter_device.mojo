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
