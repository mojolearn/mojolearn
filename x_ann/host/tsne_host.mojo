# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""t-SNE on the host: `x_ann/tsne_device.mojo`'s launches as loops over the
SAME cells (`x_ann/tsne_core.mojo`), in the same order.

Lane ann-cpu (2026-09-28): every per-row / per-coordinate launch is split
over tasks (`ann_host_cells.ann_rows`), the exact k-NN is
`cagra_host.knn_rows_host` (the cell's chain, eight candidates per vector
step), and the exact repulsion is `ts_repulse_cell` for sixteen rows at once:
lane r carries row i0 + r's three folds (Z, rep_x, rep_y) over j ascending,
each step the cell's statement on the same values (`ts_q`'s flushed
differences, one fused step each, `portable_divf`; the pinned products; a
flush whose input is provably never subnormal is left out, see
`_repulse_span`), and
the j == i step leaves the lane's folds as they were, as the cell's
`continue` does. So every row's sums are the cell's."""

from checks.numerics import ftz, identical_log
from core.device_fold import host_sum_f32_fixed
from x_ann.tsne_core import (
    F32P, I32P, ts_kl_cell, ts_perplexity_cell, ts_step_cell,
    ts_sum_cell, tsne_nn, tsne_symmetrize, tsne_validate,
)
from x_ann.tsne_core import TS_LANE_FOLD, TS_LANES
from x_ann.host.ann_host_cells import ann_span, ann_task_count, ann_tasks, ftz_v, mul_add_v, mul_v
from x_ann.host.cagra_host import knn_rows_host
from x_ann.host.ivf_pq_host import fp, ip

#: Rows whose repulsion folds run side by side in one vector.
comptime REP_W = 16


@always_inline
def _repulse_span[masked: Bool](
    yf0: F32P, yf1: F32P, j0: Int, j1: Int, ids: SIMD[DType.int32, REP_W], yi0: SIMD[DType.float32, REP_W],
    yi1: SIMD[DType.float32, REP_W], mut z: SIMD[DType.float32, REP_W], mut r0: SIMD[DType.float32, REP_W],
    mut r1: SIMD[DType.float32, REP_W],
):
    """`ts_repulse_cell`'s j loop over [j0, j1) for REP_W rows at once. Only
    the span that holds the block's own rows is `masked` (the j == i step
    leaves that lane's folds as they were)."""
    comptime W = REP_W
    var one = SIMD[DType.float32, W](1.0)
    var zero = SIMD[DType.float32, W](0.0)
    for j in range(j0, j1):
        var bj0 = SIMD[DType.float32, W](yf0.unsafe_load(j))
        var bj1 = SIMD[DType.float32, W](yf1.unsafe_load(j))
        var d0 = ftz_v[W](yi0 - bj0)
        var d1 = ftz_v[W](yi1 - bj1)
        var acc = ftz_v[W](mul_add_v[W](d0, d0, zero))
        acc = ftz_v[W](mul_add_v[W](d1, d1, acc))
        # `ftz(identical_div(1, ftz(1 + acc)))`: acc >= 0, so 1 + acc >= 1 is
        # never subnormal and both operand flushes (and the flush of the sum)
        # return their input; the result keeps its flush.
        var q = ftz_v[W](one / (one + acc))
        var qq = ftz_v[W](mul_v[W](q, q))
        # `ftz(z + q)`: z and q are >= 0 and each zero or normal, so the sum
        # is zero or >= the larger normal one: never subnormal, the flush
        # returns its input.
        var nz = z + q
        var n0 = ftz_v[W](r0 + ftz_v[W](mul_v[W](qq, d0)))
        var n1 = ftz_v[W](r1 + ftz_v[W](mul_v[W](qq, d1)))
        comptime if masked:
            var keep = ids.ne(SIMD[DType.int32, W](Int32(j)))
            z = keep.select(nz, z)
            r0 = keep.select(n0, r0)
            r1 = keep.select(n1, r1)
        else:
            z = nz
            r0 = n0
            r1 = n1


@always_inline
def _repulse_lane(
    yf0: F32P, yf1: F32P, s: Int, n: Int, ids: SIMD[DType.int32, REP_W], yi0: SIMD[DType.float32, REP_W],
    yi1: SIMD[DType.float32, REP_W], mut z: SIMD[DType.float32, REP_W], mut r0: SIMD[DType.float32, REP_W],
    mut r1: SIMD[DType.float32, REP_W],
):
    """TS_LANE_FOLD (x_ann/tsne_core.mojo): lane s of the repulsion for REP_W
    rows at once, j = s, s + TS_LANES, ... below n, ascending, `_repulse_span`'s
    statements per j, the j == i step leaving that row's folds as they were."""
    comptime W = REP_W
    var one = SIMD[DType.float32, W](1.0)
    var zero = SIMD[DType.float32, W](0.0)
    var j = s
    while j < n:
        var bj0 = SIMD[DType.float32, W](yf0.unsafe_load(j))
        var bj1 = SIMD[DType.float32, W](yf1.unsafe_load(j))
        var d0 = ftz_v[W](yi0 - bj0)
        var d1 = ftz_v[W](yi1 - bj1)
        var acc = ftz_v[W](mul_add_v[W](d0, d0, zero))
        acc = ftz_v[W](mul_add_v[W](d1, d1, acc))
        var q = ftz_v[W](one / (one + acc))
        var qq = ftz_v[W](mul_v[W](q, q))
        var nz = z + q
        var n0 = ftz_v[W](r0 + ftz_v[W](mul_v[W](qq, d0)))
        var n1 = ftz_v[W](r1 + ftz_v[W](mul_v[W](qq, d1)))
        var keep = ids.ne(SIMD[DType.int32, W](Int32(j)))
        z = keep.select(nz, z)
        r0 = keep.select(n0, r0)
        r1 = keep.select(n1, r1)
        j += TS_LANES


def _repulse_host(y: F32P, n: Int, row_z: F32P, rep: F32P, yf0: F32P, yf1: F32P):
    """`ts_repulse_cell` for every row (module docstring). `yf0` / `yf1`
    hold `ftz(y[2 j])` / `ftz(y[2 j + 1])` for j < n, padded to a multiple of
    REP_W (the padded lanes are computed and never stored)."""
    comptime W = REP_W
    var n_pad = ((n + W - 1) // W) * W
    for j in range(n):
        yf0.unsafe_store(j, ftz(y.unsafe_load(2 * j)))
        yf1.unsafe_store(j, ftz(y.unsafe_load(2 * j + 1)))
    for j in range(n, n_pad):
        yf0.unsafe_store(j, Float32(0.0))
        yf1.unsafe_store(j, Float32(0.0))
    var n_blocks = n_pad // W
    var tasks = ann_task_count(n_blocks, W * n * 16)

    def task(t: Int) {imm}:
        var span = ann_span(t, tasks, n_blocks)
        var lane = SIMD[DType.int32, W]()
        comptime for r in range(W):
            lane[r] = Int32(r)
        for blk in range(span[0], span[1]):
            var i0 = blk * W
            var ids = lane + Int32(i0)
            var yi0 = yf0.load[width=W](i0)
            var yi1 = yf1.load[width=W](i0)
            var z = SIMD[DType.float32, W](0.0)
            var r0 = SIMD[DType.float32, W](0.0)
            var r1 = SIMD[DType.float32, W](0.0)
            comptime if TS_LANE_FOLD:
                # the device's lane fold: each lane from +0, then the lane
                # partials added in ascending s, each add flushed
                for s in range(TS_LANES):
                    var lz = SIMD[DType.float32, W](0.0)
                    var l0 = SIMD[DType.float32, W](0.0)
                    var l1 = SIMD[DType.float32, W](0.0)
                    _repulse_lane(yf0, yf1, s, n, ids, yi0, yi1, lz, l0, l1)
                    z = ftz_v[W](z + lz)
                    r0 = ftz_v[W](r0 + l0)
                    r1 = ftz_v[W](r1 + l1)
            else:
                var mid = min(i0 + W, n)
                _repulse_span[False](yf0, yf1, 0, i0, ids, yi0, yi1, z, r0, r1)
                _repulse_span[True](yf0, yf1, i0, mid, ids, yi0, yi1, z, r0, r1)
                _repulse_span[False](yf0, yf1, mid, n, ids, yi0, yi1, z, r0, r1)
            var rows = min(W, n - i0)
            for r in range(rows):
                row_z.unsafe_store(i0 + r, z[r])
                rep.unsafe_store(2 * (i0 + r), r0[r])
                rep.unsafe_store(2 * (i0 + r) + 1, r1[r])

    ann_tasks(task, tasks)


def tsne_fit_host(
    x_in: List[Float32], n: Int, d: Int, y0: List[Float32], perplexity: Float32, exaggeration: Float32,
    learning_rate: Float32, max_iter: Int, exploration: Int, mut y_out: List[Float32], mut kl_out: Float32,
) raises:
    tsne_validate(n, d, perplexity, max_iter, exploration)
    var nn = tsne_nn(n, perplexity)
    var x = x_in.copy()
    var nn_d = List[Float32](length=n * nn, fill=Float32(0.0))
    var nn_i = List[Int32](length=n * nn, fill=Int32(0))
    var p_cond = List[Float32](length=n * nn, fill=Float32(0.0))
    var log_perp = identical_log(perplexity)
    knn_rows_host(fp(x), n, d, nn, fp(nn_d), ip(nn_i))
    var ndp = fp(nn_d)
    var pcp = fp(p_cond)
    var ptasks = ann_task_count(n, nn * 400)

    def perplexity_rows(t: Int) {imm}:
        var span = ann_span(t, ptasks, n)
        for i in range(span[0], span[1]):
            ts_perplexity_cell(i, ndp, nn, log_perp, pcp)

    ann_tasks(perplexity_rows, ptasks)
    var indptr = List[Int32]()
    var indices = List[Int32]()
    var values = List[Float32]()
    tsne_symmetrize(n, nn, nn_i, p_cond, indptr, indices, values)

    var ya = y0.copy()
    var yb = y0.copy()
    var upd = List[Float32](length=2 * n, fill=Float32(0.0))
    var gains = List[Float32](length=2 * n, fill=Float32(1.0))
    var rz = List[Float32](length=n, fill=Float32(0.0))
    var rep = List[Float32](length=2 * n, fill=Float32(0.0))
    var z = List[Float32](length=1, fill=Float32(0.0))
    var kl = List[Float32](length=n, fill=Float32(0.0))
    var n_pad = ((n + REP_W - 1) // REP_W) * REP_W
    var yf0 = List[Float32](length=n_pad, fill=Float32(0.0))
    var yf1 = List[Float32](length=n_pad, fill=Float32(0.0))
    var ipp = ip(indptr)
    var idp = ip(indices)
    var vp = fp(values)
    var repp = fp(rep)
    var zp = fp(z)
    var up = fp(upd)
    var gp = fp(gains)
    var nnz = len(indices)
    var stasks = ann_task_count(2 * n, 8 * (nnz // n + 2))
    for it in range(max_iter):
        var ex = exaggeration if it < exploration else Float32(1.0)
        var mom = Float32(0.5) if it < exploration else Float32(0.8)
        var ycur = fp(ya) if it % 2 == 0 else fp(yb)
        var ynext = fp(yb) if it % 2 == 0 else fp(ya)
        _repulse_host(ycur, n, fp(rz), repp, fp(yf0), fp(yf1))
        ts_sum_cell(fp(rz), n, zp)

        def step_coords(t: Int) {imm}:
            var span = ann_span(t, stasks, 2 * n)
            for e in range(span[0], span[1]):
                ts_step_cell(e, ycur, ynext, ipp, idp, vp, repp, zp, up, gp, ex, mom, learning_rate)

        ann_tasks(step_coords, stasks)
    var yfin = fp(ya) if max_iter % 2 == 0 else fp(yb)
    _repulse_host(yfin, n, fp(rz), repp, fp(yf0), fp(yf1))
    ts_sum_cell(fp(rz), n, zp)
    var klp = fp(kl)
    var ktasks = ann_task_count(n, 40 * (nnz // n + 1))

    def kl_rows(t: Int) {imm}:
        var span = ann_span(t, ktasks, n)
        for i in range(span[0], span[1]):
            ts_kl_cell(i, yfin, ipp, idp, vp, zp, klp)

    ann_tasks(kl_rows, ktasks)
    if max_iter % 2 == 0:
        y_out = ya.copy()
    else:
        y_out = yb.copy()
    # the device's fixed fold order (core/device_fold.mojo)
    kl_out = host_sum_f32_fixed(kl, n)
    _ = x^
    _ = nn_d^
    _ = nn_i^
    _ = p_cond^
    _ = indptr^
    _ = indices^
    _ = values^
    _ = upd^
    _ = gains^
    _ = rz^
    _ = rep^
    _ = z^
    _ = kl^
    _ = yf0^
    _ = yf1^
