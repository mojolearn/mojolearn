# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The trees lane's boosting and calibration steps on the device (lane
cpu2-l5-trees, 2026-10-04).

`xtrees/api.mojo` ran AdaBoost's weight steps (`samme_step`, `r2_step`),
its weighted median, DART's host-loop gradients and leaf Newton values, and
CalibratedClassifierCV's Platt fit / Platt and isotonic applies through the
serial host loops of `xtrees/ops.mojo` on every GPU build. A GPU install now
runs them here; the CPU column (`-D MOJOLEARN_COLUMN_CPU`) keeps ops.mojo's
loops, which were rewritten to the SAME order.

BITS. Every binary64 operation is `checks/soft_f64.mojo`'s integer spelling
(the Apple GPU has no float64; every float64 buffer travels as UInt64
words): sf64 add / sub / mul / div are the correctly rounded IEEE operations
the host's hardware double performs (the host's `identical_mul64` is an
unfused product); sf64_exp / sf64_log are `portable_exp64` /
`portable_log64` statement for statement, the host's `identical_exp64` /
`identical_log64` under IDENTICAL. Each kernel performs the host loop's
per-element operations in the host's order, and every n-sized sum folds in
xtrees/fold_order.mojo's fixed order (FOLD_CHUNK chunk partials in row
order, then the pairwise tree) on both columns, so the words agree on every
vendor and the host column under IDENTICAL. Under FAST the host's exp / log
are the platform libm and these kernels keep the portable forms (bits may
differ from the FAST host column; the same quality).

SCALARS. The step decisions (AdaBoost's status, alpha and boost, R2's beta,
Platt's 2x2 Newton solve and line search) run on the host from the few
folded words read back per step: the same host statements as ops.mojo's
(`samme_alpha`, `platt_drive`). No data-sized array is read back mid-step.

REFUSALS. The index checks the host loops made are device flags read back
before the result is written; the messages are the host loops'. When both a
bad row and a bad node occur, the row message wins (the host loop raises the
first bad position's)."""
from std.atomic import Atomic
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.math import ceildiv
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import identical_exp64, identical_log64, identical_mul64
from checks.soft_f64 import (
    SF64_ONE, SF64_SIGN, SF64_ZERO, sf64_add, sf64_div, sf64_exp, sf64_from_f32, sf64_from_int, sf64_is_nan, sf64_log, sf64_lt,
    sf64_mul, sf64_neg, sf64_sub, sf64_to_f32,
)
from xtrees.fold_order import FOLD_CHUNK, fold_chunk_size, fold_chunks, fold_tree_device
from xtrees.ops import PlattSums, platt_drive, platt_targets, samme_alpha, median_total
from xtrees.ops_device import OPS_TPB, _blocks, _ctx
from xtrees.ops_device_elem import _flags, _gt64, _read_i32, _up_f32, _up_f64, _up_i32


@always_inline
def _w(x: Float64) -> UInt64:
    return bitcast[DType.uint64](x)


@always_inline
def _f(w: UInt64) -> Float64:
    return bitcast[DType.float64](w)


@always_inline
def _ge64(a: UInt64, b: UInt64) -> Bool:
    """IEEE `a >= b` on binary64 words (a NaN compares false)."""
    if sf64_is_nan(a) or sf64_is_nan(b):
        return False
    return not sf64_lt(a, b)


@always_inline
def _abs64(a: UInt64) -> UInt64:
    return a & ~SF64_SIGN


def _head_u64(ctx: DeviceContext, d: DeviceBuffer[DType.uint64], k: Int) raises -> List[UInt64]:
    """The first k words of `d` back on the host, synchronized."""
    var h = ctx.enqueue_create_host_buffer[DType.uint64](k)
    var s = d.create_sub_buffer[DType.uint64](0, k)
    ctx.enqueue_copy(dst_buf=h, src_buf=s)
    ctx.synchronize()
    var res = List[UInt64](capacity=k)
    for i in range(k):  # small-loop(k: head words): every caller reads at most 5 scalar words
        res.append(h.unsafe_ptr().unsafe_load(i))
    _ = s^
    _ = h^
    return res^


# ------------------------------------------------------------- SAMME --


def samme_sums_kernel(
    w: MutPointer[UInt64, MutAnyOrigin], pred: MutPointer[Int32, MutAnyOrigin], y: MutPointer[Int32, MutAnyOrigin],
    n: Int64, m: Int64, p: MutPointer[UInt64, MutAnyOrigin],
):
    """One thread per chunk: p[2c] = the misclassified weight, p[2c + 1] =
    the weight, each from +0.0 in row order (ops.samme_step's chunk loop)."""
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while c < Int(m):
        var e = SF64_ZERO
        var t = SF64_ZERO
        for i in range(c * FOLD_CHUNK, min((c + 1) * FOLD_CHUNK, Int(n))):
            var wi = w.unsafe_load(i)
            t = sf64_add(t, wi)
            if pred.unsafe_load(i) != y.unsafe_load(i):
                e = sf64_add(e, wi)
        p.unsafe_store(2 * c, e)
        p.unsafe_store(2 * c + 1, t)
        c += stride


def samme_update_kernel(
    w: MutPointer[UInt64, MutAnyOrigin], pred: MutPointer[Int32, MutAnyOrigin], y: MutPointer[Int32, MutAnyOrigin],
    n: Int64, m: Int64, boost: UInt64, q: MutPointer[UInt64, MutAnyOrigin],
):
    """One thread per chunk: a misclassified positive weight times `boost`,
    then the chunk's new sum from +0.0 in row order."""
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while c < Int(m):
        var run = SF64_ZERO
        for i in range(c * FOLD_CHUNK, min((c + 1) * FOLD_CHUNK, Int(n))):
            var wi = w.unsafe_load(i)
            if pred.unsafe_load(i) != y.unsafe_load(i) and _gt64(wi, SF64_ZERO):
                wi = sf64_mul(wi, boost)
                w.unsafe_store(i, wi)
            run = sf64_add(run, wi)
        q.unsafe_store(c, run)
        c += stride


def samme_step_device(
    w: MutPointer[Float64, MutUntrackedOrigin], pred: MutPointer[Int32, MutUntrackedOrigin],
    y: MutPointer[Int32, MutUntrackedOrigin], n: Int, n_classes: Int,
    learning_rate: Float64, last: Bool, stats: MutPointer[Float64, MutUntrackedOrigin],
) raises:
    """`ops.samme_step` on the device: w, pred and y go up once; the error
    and total fold on the device and come back as two words; the reweighting
    and its sum run on the device; w comes back once (only when it moved)."""
    var ctx = _ctx()
    var m = fold_chunks(n)
    var d_w = _up_f64(ctx, w, n)
    var d_p = _up_i32(ctx, pred, n)
    var d_y = _up_i32(ctx, y, n)
    var d_s = ctx.enqueue_create_buffer[DType.uint64](2 * m)
    ctx.enqueue_function[samme_sums_kernel](
        d_w.unsafe_ptr(), d_p.unsafe_ptr(), d_y.unsafe_ptr(), Int64(n), Int64(m), d_s.unsafe_ptr(),
        grid_dim=_blocks(m), block_dim=OPS_TPB,
    )
    fold_tree_device(ctx, d_s, m, 2)
    var et = _head_u64(ctx, d_s, 2)
    var tot = _f(et[1])
    var err = _f(et[0]) / tot
    var status: Float64 = 0.0
    var alpha: Float64 = 0.0
    var s = tot
    if err <= 0.0:
        status = 1.0
        alpha = 1.0
        err = 0.0
    else:
        var k = Float64(n_classes)
        if err >= 1.0 - 1.0 / k:
            status = 2.0
        else:
            alpha = samme_alpha(err, k, learning_rate)
            if not last:
                # DEVIATION 5603: the pinned exp (a host scalar).
                var boost = identical_exp64(alpha)
                var d_q = ctx.enqueue_create_buffer[DType.uint64](m)
                ctx.enqueue_function[samme_update_kernel](
                    d_w.unsafe_ptr(), d_p.unsafe_ptr(), d_y.unsafe_ptr(), Int64(n), Int64(m), _w(boost),
                    d_q.unsafe_ptr(), grid_dim=_blocks(m), block_dim=OPS_TPB,
                )
                fold_tree_device(ctx, d_q, m, 1)
                s = _f(_head_u64(ctx, d_q, 1)[0])
                ctx.enqueue_copy(dst_ptr=w.bitcast[UInt64](), src_buf=d_w)
                ctx.synchronize()
                _ = d_q^
    stats[unsafe_offset=0] = status
    stats[unsafe_offset=1] = alpha
    stats[unsafe_offset=2] = err
    stats[unsafe_offset=3] = s
    _ = d_w^
    _ = d_p^
    _ = d_y^
    _ = d_s^


# ---------------------------------------------------------- AdaBoost.R2 --


@always_inline
def _r2_error(p: Float32, t: Float32, emax: UInt64, loss: Int) -> UInt64:
    """`ops.r2_error` over binary64 words."""
    var e = _abs64(sf64_sub(sf64_from_f32(p), sf64_from_f32(t)))
    if (emax & ~SF64_SIGN) != 0:
        e = sf64_div(e, emax)
    if loss == 1:
        e = sf64_mul(e, e)
    elif loss == 2:
        e = sf64_sub(SF64_ONE, sf64_exp(sf64_neg(e)))
    return e


def r2_emax_kernel(
    w: MutPointer[UInt64, MutAnyOrigin], pred: MutPointer[Float32, MutAnyOrigin], y: MutPointer[Float32, MutAnyOrigin],
    n: Int64, m: Int64, p: MutPointer[UInt64, MutAnyOrigin],
):
    """One thread per chunk: the chunk's largest |pred - y| over positive
    weights (from +0.0; a NaN never wins, as `e > emax` in the host loop).
    A maximum is exact in any order."""
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while c < Int(m):
        var mx = SF64_ZERO
        for i in range(c * FOLD_CHUNK, min((c + 1) * FOLD_CHUNK, Int(n))):
            if _gt64(w.unsafe_load(i), SF64_ZERO):
                var e = _abs64(sf64_sub(sf64_from_f32(pred.unsafe_load(i)), sf64_from_f32(y.unsafe_load(i))))
                if _gt64(e, mx):
                    mx = e
        p.unsafe_store(c, mx)
        c += stride


def max_tree_pass_kernel(p: MutPointer[UInt64, MutAnyOrigin], m: Int64, s: Int64):
    """One pass of the pairwise tree with `>` in place of `+`."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var pairs = (Int(m) + 2 * Int(s) - 1) // (2 * Int(s))
    while t < pairs:
        var j = t * 2 * Int(s)
        if j + Int(s) < Int(m):
            var b = p.unsafe_load(j + Int(s))
            if _gt64(b, p.unsafe_load(j)):
                p.unsafe_store(j, b)
        t += stride


def r2_err_kernel(
    w: MutPointer[UInt64, MutAnyOrigin], pred: MutPointer[Float32, MutAnyOrigin], y: MutPointer[Float32, MutAnyOrigin],
    n: Int64, m: Int64, emax: UInt64, loss: Int64, p: MutPointer[UInt64, MutAnyOrigin],
):
    """One thread per chunk: sum of w * e over positive weights, from +0.0
    in row order (ops.r2_step's first chunk loop)."""
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while c < Int(m):
        var run = SF64_ZERO
        for i in range(c * FOLD_CHUNK, min((c + 1) * FOLD_CHUNK, Int(n))):
            var wi = w.unsafe_load(i)
            if _gt64(wi, SF64_ZERO):
                run = sf64_add(run, sf64_mul(wi, _r2_error(pred.unsafe_load(i), y.unsafe_load(i), emax, Int(loss))))
        p.unsafe_store(c, run)
        c += stride


def r2_update_kernel(
    w: MutPointer[UInt64, MutAnyOrigin], pred: MutPointer[Float32, MutAnyOrigin], y: MutPointer[Float32, MutAnyOrigin],
    n: Int64, m: Int64, emax: UInt64, loss: Int64, lr: UInt64, lb: UInt64, update: Int64,
    q: MutPointer[UInt64, MutAnyOrigin],
):
    """One thread per chunk: w *= exp(((1 - e) lr) log(beta)) for positive
    weights (unless the last step), then the chunk's sum from +0.0."""
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while c < Int(m):
        var run = SF64_ZERO
        for i in range(c * FOLD_CHUNK, min((c + 1) * FOLD_CHUNK, Int(n))):
            var wi = w.unsafe_load(i)
            if update != 0 and _gt64(wi, SF64_ZERO):
                var e = _r2_error(pred.unsafe_load(i), y.unsafe_load(i), emax, Int(loss))
                var t = sf64_mul(sf64_sub(SF64_ONE, e), lr)
                wi = sf64_mul(wi, sf64_exp(sf64_mul(t, lb)))
                w.unsafe_store(i, wi)
            run = sf64_add(run, wi)
        q.unsafe_store(c, run)
        c += stride


def r2_step_device(
    w: MutPointer[Float64, MutUntrackedOrigin], pred: MutPointer[Float32, MutUntrackedOrigin],
    y: MutPointer[Float32, MutUntrackedOrigin], n: Int, loss: Int,
    learning_rate: Float64, last: Bool, stats: MutPointer[Float64, MutUntrackedOrigin],
) raises:
    """`ops.r2_step` on the device: the largest error, the weighted error and
    the reweighted sum on the device; status / beta / alpha on the host from
    the folded words; w comes back once."""
    var ctx = _ctx()
    var m = fold_chunks(n)
    var d_w = _up_f64(ctx, w, n)
    var d_p = _up_f32(ctx, pred, n)
    var d_y = _up_f32(ctx, y, n)
    var d_s = ctx.enqueue_create_buffer[DType.uint64](m)
    ctx.enqueue_function[r2_emax_kernel](
        d_w.unsafe_ptr(), d_p.unsafe_ptr(), d_y.unsafe_ptr(), Int64(n), Int64(m), d_s.unsafe_ptr(),
        grid_dim=_blocks(m), block_dim=OPS_TPB,
    )
    var s = 1
    while s < m:
        ctx.enqueue_function[max_tree_pass_kernel](
            d_s.unsafe_ptr(), Int64(m), Int64(s), grid_dim=_blocks(ceildiv(m, 2 * s)), block_dim=OPS_TPB,
        )
        s *= 2
    var emax = _head_u64(ctx, d_s, 1)[0]
    ctx.enqueue_function[r2_err_kernel](
        d_w.unsafe_ptr(), d_p.unsafe_ptr(), d_y.unsafe_ptr(), Int64(n), Int64(m), emax, Int64(loss),
        d_s.unsafe_ptr(), grid_dim=_blocks(m), block_dim=OPS_TPB,
    )
    fold_tree_device(ctx, d_s, m, 1)
    var err = _f(_head_u64(ctx, d_s, 1)[0])
    if err <= 0.0:
        stats[unsafe_offset=0] = 1.0
        stats[unsafe_offset=1] = 1.0
        stats[unsafe_offset=2] = 0.0
    elif err >= 0.5:
        stats[unsafe_offset=0] = 2.0
        stats[unsafe_offset=1] = 0.0
        stats[unsafe_offset=2] = err
    else:
        var beta = err / (1.0 - err)
        var alpha = identical_mul64(learning_rate, identical_log64(1.0 / beta))
        var lb = identical_log64(beta)
        ctx.enqueue_function[r2_update_kernel](
            d_w.unsafe_ptr(), d_p.unsafe_ptr(), d_y.unsafe_ptr(), Int64(n), Int64(m), emax, Int64(loss),
            _w(learning_rate), _w(lb), Int64(0 if last else 1), d_s.unsafe_ptr(),
            grid_dim=_blocks(m), block_dim=OPS_TPB,
        )
        fold_tree_device(ctx, d_s, m, 1)
        var tot = _f(_head_u64(ctx, d_s, 1)[0])
        if not last:
            ctx.enqueue_copy(dst_ptr=w.bitcast[UInt64](), src_buf=d_w)
            ctx.synchronize()
        stats[unsafe_offset=0] = 0.0
        stats[unsafe_offset=1] = alpha
        stats[unsafe_offset=2] = err
        stats[unsafe_offset=3] = tot
    _ = d_w^
    _ = d_p^
    _ = d_y^
    _ = d_s^


# ------------------------------------------------------ weighted median --


def median_kernel(
    preds: MutPointer[Float32, MutAnyOrigin], weights: MutPointer[UInt64, MutAnyOrigin], n: Int64, m: Int64,
    half: UInt64, res: MutPointer[Float32, MutAnyOrigin],
):
    """One thread per row: `ops.median_pick`'s law (the cumulative weight of
    each key is the weights at or below it, added in estimator order; the
    smallest key reaching `half` wins; none: the largest key), then the
    picked estimator's prediction (a copy)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var nn = Int(n)
    var mm = Int(m)
    while i < nn:
        var pick = -1
        var pick_v = Float32(0)
        var last = 0
        var last_v = preds.unsafe_load(i)
        for j in range(mm):
            var v = preds.unsafe_load(j * nn + i)
            if j > 0 and not (v < last_v):
                last = j
                last_v = v
            var c = SF64_ZERO
            for l in range(mm):
                var u = preds.unsafe_load(l * nn + i)
                if u < v or (u == v and l <= j):
                    c = sf64_add(c, weights.unsafe_load(l))
            if _ge64(c, half) and (pick < 0 or v < pick_v or (v == pick_v and j < pick)):
                pick = j
                pick_v = v
        var at = pick if pick >= 0 else last
        res.unsafe_store(i, preds.unsafe_load(at * nn + i))
        i += stride


def weighted_median_device(
    preds: MutPointer[Float32, MutUntrackedOrigin], weights: MutPointer[Float64, MutUntrackedOrigin],
    n: Int, m: Int, res: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    """`ops.weighted_median` on the device: the m x n predictions and the m
    weights go up once, one thread per row, the medians come back once. The
    half-total is the m-sized estimator-order sum (`ops.median_total`), one
    host scalar for every row."""
    var half = identical_mul64(0.5, median_total(weights, m))
    var ctx = _ctx()
    var d_p = ctx.enqueue_create_buffer[DType.uint32](m * n)
    ctx.enqueue_copy(dst_buf=d_p, src_ptr=preds.bitcast[UInt32]())
    var d_wt = _up_f64(ctx, weights, m)
    var d_r = ctx.enqueue_create_buffer[DType.uint32](n)
    ctx.enqueue_function[median_kernel](
        d_p.unsafe_ptr().bitcast[Float32](), d_wt.unsafe_ptr(), Int64(n), Int64(m), _w(half),
        d_r.unsafe_ptr().bitcast[Float32](), grid_dim=_blocks(n), block_dim=OPS_TPB,
    )
    ctx.enqueue_copy(dst_ptr=res.bitcast[UInt32](), src_buf=d_r)
    ctx.synchronize()
    _ = d_p^
    _ = d_wt^
    _ = d_r^


# ------------------------------------------------------------ gradients --


def gradients_kernel(
    score: MutPointer[UInt64, MutAnyOrigin], y: MutPointer[Float32, MutAnyOrigin], n: Int64, kind: Int64, k: Int64,
    factor: UInt64, g: MutPointer[UInt64, MutAnyOrigin], h: MutPointer[UInt64, MutAnyOrigin],
    target: MutPointer[Float32, MutAnyOrigin],
):
    """One thread per row: `ops.gradients`' row body (class-major for kind 2:
    the max, the exp sum in class order, then each class's p, g, h, -g)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var nn = Int(n)
    while i < nn:
        if kind == 2:
            var kk = Int(k)
            var mx = score.unsafe_load(i)
            for c in range(1, kk):
                var v = score.unsafe_load(c * nn + i)
                if _gt64(v, mx):
                    mx = v
            var s = SF64_ZERO
            for c in range(kk):
                s = sf64_add(s, sf64_exp(sf64_sub(score.unsafe_load(c * nn + i), mx)))
            var yi = Int(y.unsafe_load(i))
            for c in range(kk):
                var p = sf64_div(sf64_exp(sf64_sub(score.unsafe_load(c * nn + i), mx)), s)
                var gi = sf64_sub(p, SF64_ONE) if yi == c else p
                g.unsafe_store(c * nn + i, gi)
                h.unsafe_store(c * nn + i, sf64_mul(sf64_mul(factor, p), sf64_sub(SF64_ONE, p)))
                target.unsafe_store(c * nn + i, sf64_to_f32(sf64_neg(gi)))
        else:
            var s = score.unsafe_load(i)
            var yv = sf64_from_f32(y.unsafe_load(i))
            var gi: UInt64
            var hi: UInt64
            if kind == 0:
                gi = sf64_sub(s, yv)
                hi = SF64_ONE
            else:
                var p = sf64_div(SF64_ONE, sf64_add(SF64_ONE, sf64_exp(sf64_neg(s))))
                gi = sf64_sub(p, yv)
                hi = sf64_mul(p, sf64_sub(SF64_ONE, p))
            g.unsafe_store(i, gi)
            h.unsafe_store(i, hi)
            target.unsafe_store(i, sf64_to_f32(sf64_neg(gi)))
        i += stride


def gradients_device(
    score: MutPointer[Float64, MutUntrackedOrigin], y: MutPointer[Float32, MutUntrackedOrigin], n: Int,
    kind: Int, g: MutPointer[Float64, MutUntrackedOrigin], h: MutPointer[Float64, MutUntrackedOrigin],
    target: MutPointer[Float32, MutUntrackedOrigin], k: Int = 1,
) raises:
    """`ops.gradients` on the device: score and y up once, one thread per
    row, g / h / target back once. The softmax factor k / (k - 1) is a host
    scalar (the host loop's expression)."""
    var kk = k if kind == 2 else 1
    var factor = Float64(k) / (Float64(k) - 1.0) if kind == 2 else 0.0
    var ctx = _ctx()
    var d_s = _up_f64(ctx, score, kk * n)
    var d_y = _up_f32(ctx, y, n)
    var d_g = ctx.enqueue_create_buffer[DType.uint64](kk * n)
    var d_h = ctx.enqueue_create_buffer[DType.uint64](kk * n)
    var d_t = ctx.enqueue_create_buffer[DType.float32](kk * n)
    ctx.enqueue_function[gradients_kernel](
        d_s.unsafe_ptr(), d_y.unsafe_ptr(), Int64(n), Int64(kind), Int64(kk), _w(factor),
        d_g.unsafe_ptr(), d_h.unsafe_ptr(), d_t.unsafe_ptr(), grid_dim=_blocks(n), block_dim=OPS_TPB,
    )
    ctx.enqueue_copy(dst_ptr=g.bitcast[UInt64](), src_buf=d_g)
    ctx.enqueue_copy(dst_ptr=h.bitcast[UInt64](), src_buf=d_h)
    ctx.enqueue_copy(dst_ptr=target, src_buf=d_t)
    ctx.synchronize()
    _ = d_s^
    _ = d_y^
    _ = d_g^
    _ = d_h^
    _ = d_t^


# --------------------------------------------------------- leaf Newton --


def leaf_check_kernel(
    nodes: MutPointer[Int32, MutAnyOrigin], rows: MutPointer[Int32, MutAnyOrigin], use_rows: Int64, m: Int64,
    n: Int64, n_nodes: Int64, flags: MutPointer[Int32, MutAnyOrigin],
):
    """flags[0] = 1 on a row outside [0, n), flags[1] = 1 on a node outside
    [0, n_nodes) (read only at an in-range row)."""
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while r < Int(m):
        var i = Int(rows.unsafe_load(r)) if use_rows != 0 else r
        if i < 0 or i >= Int(n):
            flags.unsafe_store(0, Int32(1))
        else:
            var k = Int(nodes.unsafe_load(i))
            if k < 0 or k >= Int(n_nodes):
                flags.unsafe_store(1, Int32(1))
        r += stride


def leaf_sum_kernel(
    nodes: MutPointer[Int32, MutAnyOrigin], rows: MutPointer[Int32, MutAnyOrigin], use_rows: Int64, m: Int64,
    g: MutPointer[UInt64, MutAnyOrigin], h: MutPointer[UInt64, MutAnyOrigin], n_nodes: Int64, cs: Int64,
    nc: Int64, p: MutPointer[UInt64, MutAnyOrigin],
):
    """One thread per (chunk, node): the node's g and h over the chunk's list
    positions in order, from +0.0 (`ops.leaf_sums`' chunk loop)."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var nk = Int(n_nodes)
    var w = 2 * nk
    while t < Int(nc) * nk:
        var c = t // nk
        var k = t - c * nk
        var sg = SF64_ZERO
        var sh = SF64_ZERO
        for r in range(c * Int(cs), min((c + 1) * Int(cs), Int(m))):
            var i = Int(rows.unsafe_load(r)) if use_rows != 0 else r
            if Int(nodes.unsafe_load(i)) == k:
                sg = sf64_add(sg, g.unsafe_load(i))
                sh = sf64_add(sh, h.unsafe_load(i))
        p.unsafe_store(c * w + 2 * k, sg)
        p.unsafe_store(c * w + 2 * k + 1, sh)
        t += stride


def newton_kernel(
    p: MutPointer[UInt64, MutAnyOrigin], n_nodes: Int64, lam: UInt64, l1: UInt64, mds: UInt64,
    values: MutPointer[Float32, MutAnyOrigin],
):
    """One thread per node: `ops._newton_values` over the folded sums."""
    var k = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while k < Int(n_nodes):
        var den = sf64_add(p.unsafe_load(2 * k + 1), lam)
        if not _gt64(den, SF64_ZERO):
            values.unsafe_store(k, Float32(0.0))
        else:
            var s = p.unsafe_load(2 * k)
            if _gt64(l1, SF64_ZERO):
                # FeatureHistogram::ThresholdL1: Sign(s) * max(0, |s| - l1).
                var a = sf64_sub(s if _ge64(s, SF64_ZERO) else sf64_neg(s), l1)
                var reg = a if _gt64(a, SF64_ZERO) else SF64_ZERO
                s = reg if _gt64(s, SF64_ZERO) else (sf64_neg(reg) if sf64_lt(s, SF64_ZERO) and not sf64_is_nan(s) else SF64_ZERO)
            var ret = sf64_div(sf64_neg(s), den)
            var mag = ret if _ge64(ret, SF64_ZERO) else sf64_neg(ret)
            if _gt64(mds, SF64_ZERO) and _gt64(mag, mds):
                ret = mds if _gt64(ret, SF64_ZERO) else sf64_neg(mds)
            values.unsafe_store(k, sf64_to_f32(ret))
        k += stride


def leaf_newton_device(
    nodes: MutPointer[Int32, MutUntrackedOrigin], rows: MutPointer[Int32, MutUntrackedOrigin], use_rows: Bool,
    m: Int, g: MutPointer[Float64, MutUntrackedOrigin], h: MutPointer[Float64, MutUntrackedOrigin], n: Int,
    n_nodes: Int, reg_lambda: Float64, l1: Float64, max_delta_step: Float64,
    values: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    """`ops.leaf_newton` (use_rows False: positions are the rows 0 .. n) and
    `ops.leaf_newton_rows` (the bagged rows) on the device: nodes, rows, g
    and h up once; the index check; one unit per (chunk, node) for the
    partial sums; the pairwise tree; one thread per node for the values,
    which come back once."""
    var ctx = _ctx()
    var w = 2 * n_nodes
    var cs = fold_chunk_size(m, w)
    var nc = max(1, ceildiv(m, cs))
    var d_n = _up_i32(ctx, nodes, n)
    var d_r = _up_i32(ctx, rows, m if use_rows else 0)
    var d_g = _up_f64(ctx, g, n)
    var d_h = _up_f64(ctx, h, n)
    var d_f = _flags(ctx, 2)
    ctx.enqueue_function[leaf_check_kernel](
        d_n.unsafe_ptr(), d_r.unsafe_ptr(), Int64(1 if use_rows else 0), Int64(m), Int64(n), Int64(n_nodes),
        d_f.unsafe_ptr(), grid_dim=_blocks(m), block_dim=OPS_TPB,
    )
    var fl = _read_i32(ctx, d_f, 2)
    if fl[0] != 0:
        raise Error("x_trees leaf_newton_rows: row out of range")
    if fl[1] != 0:
        raise Error("x_trees leaf_newton_rows: node out of range" if use_rows else "x_trees leaf_newton: node out of range")
    var d_p = ctx.enqueue_create_buffer[DType.uint64](nc * w)
    ctx.enqueue_function[leaf_sum_kernel](
        d_n.unsafe_ptr(), d_r.unsafe_ptr(), Int64(1 if use_rows else 0), Int64(m), d_g.unsafe_ptr(), d_h.unsafe_ptr(),
        Int64(n_nodes), Int64(cs), Int64(nc), d_p.unsafe_ptr(), grid_dim=_blocks(nc * n_nodes), block_dim=OPS_TPB,
    )
    fold_tree_device(ctx, d_p, nc, w)
    var d_v = ctx.enqueue_create_buffer[DType.float32](n_nodes)
    ctx.enqueue_function[newton_kernel](
        d_p.unsafe_ptr(), Int64(n_nodes), _w(reg_lambda), _w(l1), _w(max_delta_step), d_v.unsafe_ptr(),
        grid_dim=_blocks(n_nodes), block_dim=OPS_TPB,
    )
    ctx.enqueue_copy(dst_ptr=values, src_buf=d_v)
    ctx.synchronize()
    _ = d_n^
    _ = d_r^
    _ = d_g^
    _ = d_h^
    _ = d_f^
    _ = d_p^
    _ = d_v^


# --------------------------------------------------------------- Platt --


@always_inline
def _log1pexp(x: UInt64) -> UInt64:
    """`ops._log1pexp` over binary64 words."""
    if _ge64(x, SF64_ZERO):
        return sf64_add(x, sf64_log(sf64_add(SF64_ONE, sf64_exp(sf64_neg(x)))))
    return sf64_log(sf64_add(SF64_ONE, sf64_exp(x)))


def platt_count_kernel(y: MutPointer[Int32, MutAnyOrigin], n: Int64, m: Int64, p: MutPointer[Int32, MutAnyOrigin]):
    """One thread per chunk: the chunk's positive labels, added into the one
    word `p[0]` (zeroed by the caller) by an integer atomic, exact in any
    order (cpu3-trees: the m chunk counts no longer come back to be summed
    on the host)."""
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while c < Int(m):
        var cnt = 0
        for i in range(c * FOLD_CHUNK, min((c + 1) * FOLD_CHUNK, Int(n))):
            if y.unsafe_load(i) > 0:
                cnt += 1
        if cnt != 0:
            _ = Atomic.fetch_add(p, Int32(cnt))
        c += stride


def platt_value_kernel(
    f: MutPointer[UInt64, MutAnyOrigin], y: MutPointer[Int32, MutAnyOrigin], n: Int64, m: Int64, hi: UInt64,
    lo: UInt64, a: UInt64, b: UInt64, p: MutPointer[UInt64, MutAnyOrigin],
):
    """One thread per chunk: `ops.PlattHost.value`'s chunk loop."""
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while c < Int(m):
        var run = SF64_ZERO
        for i in range(c * FOLD_CHUNK, min((c + 1) * FOLD_CHUNK, Int(n))):
            var t = hi if y.unsafe_load(i) > 0 else lo
            var z = sf64_add(sf64_mul(f.unsafe_load(i), a), b)
            run = sf64_add(
                sf64_add(run, sf64_mul(t, _log1pexp(z))), sf64_mul(sf64_sub(SF64_ONE, t), _log1pexp(sf64_neg(z))))
        p.unsafe_store(c, run)
        c += stride


def platt_grad_kernel(
    f: MutPointer[UInt64, MutAnyOrigin], y: MutPointer[Int32, MutAnyOrigin], n: Int64, m: Int64, hi: UInt64,
    lo: UInt64, a: UInt64, b: UInt64, p: MutPointer[UInt64, MutAnyOrigin],
):
    """One thread per chunk: `ops.PlattHost.grad`'s chunk loop (5 fields)."""
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while c < Int(m):
        var s0 = SF64_ZERO
        var s1 = SF64_ZERO
        var s2 = SF64_ZERO
        var s3 = SF64_ZERO
        var s4 = SF64_ZERO
        for i in range(c * FOLD_CHUNK, min((c + 1) * FOLD_CHUNK, Int(n))):
            var fi = f.unsafe_load(i)
            var z = sf64_add(sf64_mul(fi, a), b)
            var pp: UInt64
            var q: UInt64
            if _ge64(z, SF64_ZERO):
                var e = sf64_exp(sf64_neg(z))
                pp = sf64_div(e, sf64_add(SF64_ONE, e))
                q = sf64_div(SF64_ONE, sf64_add(SF64_ONE, e))
            else:
                var e = sf64_exp(z)
                pp = sf64_div(SF64_ONE, sf64_add(SF64_ONE, e))
                q = sf64_div(e, sf64_add(SF64_ONE, e))
            var d2 = sf64_mul(pp, q)
            var t = hi if y.unsafe_load(i) > 0 else lo
            var d1 = sf64_sub(t, pp)
            s0 = sf64_add(s0, sf64_mul(sf64_mul(fi, fi), d2))
            s1 = sf64_add(s1, d2)
            s2 = sf64_add(s2, sf64_mul(fi, d2))
            s3 = sf64_add(s3, sf64_mul(fi, d1))
            s4 = sf64_add(s4, d1)
        p.unsafe_store(5 * c, s0)
        p.unsafe_store(5 * c + 1, s1)
        p.unsafe_store(5 * c + 2, s2)
        p.unsafe_store(5 * c + 3, s3)
        p.unsafe_store(5 * c + 4, s4)
        c += stride


struct PlattDevice(PlattSums):
    """Platt's sums with f and y resident on the device for the whole fit:
    each call is one chunk launch, the pairwise tree and a read of the 1 or
    5 folded words."""
    var ctx: DeviceContext
    var d_f: DeviceBuffer[DType.uint64]
    var d_y: DeviceBuffer[DType.int32]
    var d_p: DeviceBuffer[DType.uint64]
    var n: Int
    var m: Int
    var hi: UInt64
    var lo: UInt64

    def __init__(
        out self, ctx: DeviceContext, var d_f: DeviceBuffer[DType.uint64], var d_y: DeviceBuffer[DType.int32],
        var d_p: DeviceBuffer[DType.uint64], n: Int, hi: Float64, lo: Float64,
    ):
        self.ctx = ctx.copy()
        self.d_f = d_f^
        self.d_y = d_y^
        self.d_p = d_p^
        self.n = n
        self.m = fold_chunks(n)
        self.hi = _w(hi)
        self.lo = _w(lo)

    def value(mut self, a: Float64, b: Float64) raises -> Float64:
        self.ctx.enqueue_function[platt_value_kernel](
            self.d_f.unsafe_ptr(), self.d_y.unsafe_ptr(), Int64(self.n), Int64(self.m), self.hi, self.lo, _w(a), _w(b),
            self.d_p.unsafe_ptr(), grid_dim=_blocks(self.m), block_dim=OPS_TPB,
        )
        fold_tree_device(self.ctx, self.d_p, self.m, 1)
        return _f(_head_u64(self.ctx, self.d_p, 1)[0])

    def grad(mut self, a: Float64, b: Float64, mut out: List[Float64]) raises:
        self.ctx.enqueue_function[platt_grad_kernel](
            self.d_f.unsafe_ptr(), self.d_y.unsafe_ptr(), Int64(self.n), Int64(self.m), self.hi, self.lo, _w(a), _w(b),
            self.d_p.unsafe_ptr(), grid_dim=_blocks(self.m), block_dim=OPS_TPB,
        )
        fold_tree_device(self.ctx, self.d_p, self.m, 5)
        var r = _head_u64(self.ctx, self.d_p, 5)
        for q in range(5):
            out[q] = _f(r[q])


def platt_fit_device(
    f: MutPointer[Float64, MutUntrackedOrigin], y: MutPointer[Int32, MutUntrackedOrigin], n: Int,
    ab: MutPointer[Float64, MutUntrackedOrigin],
) raises:
    """`ops.platt_fit` on the device: f and y go up once and stay for the
    whole Newton fit (`PlattDevice`); the one driver `ops.platt_drive` runs
    the same scalar steps as the host column."""
    var ctx = _ctx()
    var m = fold_chunks(n)
    var d_f = _up_f64(ctx, f, n)
    var d_y = _up_i32(ctx, y, n)
    var d_c = ctx.enqueue_create_buffer[DType.int32](1)
    d_c.enqueue_fill(Int32(0))
    ctx.enqueue_function[platt_count_kernel](
        d_y.unsafe_ptr(), Int64(n), Int64(m), d_c.unsafe_ptr(), grid_dim=_blocks(m), block_dim=OPS_TPB,
    )
    var cnt = _read_i32(ctx, d_c, 1)[0]
    var pr = platt_targets(cnt, n)
    var d_p = ctx.enqueue_create_buffer[DType.uint64](5 * m)
    var sums = PlattDevice(ctx, d_f^, d_y^, d_p^, n, pr[1], pr[2])
    var res = platt_drive(sums, pr[0], n)
    ab[unsafe_offset=0] = res[0]
    ab[unsafe_offset=1] = res[1]
    _ = sums^
    _ = d_c^


def platt_apply_kernel(
    f: MutPointer[UInt64, MutAnyOrigin], fs: Int64, n: Int64, a: UInt64, b: UInt64,
    res: MutPointer[UInt64, MutAnyOrigin], rs: Int64,
):
    """One thread per element: `ops.platt_apply_strided`'s body."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < Int(n):
        var z = sf64_add(sf64_mul(f.unsafe_load(i * Int(fs)), a), b)
        var r: UInt64
        if _ge64(z, SF64_ZERO):
            var e = sf64_exp(sf64_neg(z))
            r = sf64_div(e, sf64_add(SF64_ONE, e))
        else:
            r = sf64_div(SF64_ONE, sf64_add(SF64_ONE, sf64_exp(z)))
        res.unsafe_store(i * Int(rs), r)
        i += stride


def platt_apply_strided_device(
    f: MutPointer[Float64, MutUntrackedOrigin], fs: Int, n: Int, a: Float64, b: Float64,
    res: MutPointer[Float64, MutUntrackedOrigin], rs: Int,
) raises:
    """`ops.platt_apply_strided` on the device. The strided spans go up as
    words (res too, so the words between its stride positions come back
    unchanged), one thread per element, the res span comes back once."""
    if n <= 0:
        return
    var fl = (n - 1) * fs + 1
    var rl = (n - 1) * rs + 1
    var ctx = _ctx()
    var d_f = _up_f64(ctx, f, fl)
    var d_r = _up_f64(ctx, res, rl)
    ctx.enqueue_function[platt_apply_kernel](
        d_f.unsafe_ptr(), Int64(fs), Int64(n), _w(a), _w(b), d_r.unsafe_ptr(), Int64(rs),
        grid_dim=_blocks(n), block_dim=OPS_TPB,
    )
    ctx.enqueue_copy(dst_ptr=res.bitcast[UInt64](), src_buf=d_r)
    ctx.synchronize()
    _ = d_f^
    _ = d_r^


# ------------------------------------------------------ isotonic apply --


def isotonic_predict_kernel(
    kx: MutPointer[UInt64, MutAnyOrigin], ky: MutPointer[UInt64, MutAnyOrigin], m: Int64,
    t: MutPointer[UInt64, MutAnyOrigin], ts: Int64, n: Int64, res: MutPointer[UInt64, MutAnyOrigin], rs: Int64,
):
    """One thread per element: `ops.isotonic_predict_strided`'s body (the
    clip, the binary search over the knots, the interpolation)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var mm = Int(m)
    while i < Int(n):
        var v = t.unsafe_load(i * Int(ts))
        var r: UInt64
        if mm == 1 or _ge64(kx.unsafe_load(0), v):
            r = ky.unsafe_load(0)
        elif _ge64(v, kx.unsafe_load(mm - 1)):
            r = ky.unsafe_load(mm - 1)
        else:
            var lo = 0
            var hi = mm - 1
            while hi - lo > 1:
                var mid = (lo + hi) // 2
                if _ge64(v, kx.unsafe_load(mid)):
                    lo = mid
                else:
                    hi = mid
            var x0 = kx.unsafe_load(lo)
            var y0 = ky.unsafe_load(lo)
            var slope = sf64_div(sf64_sub(ky.unsafe_load(hi), y0), sf64_sub(kx.unsafe_load(hi), x0))
            r = sf64_add(sf64_mul(slope, sf64_sub(v, x0)), y0)
        res.unsafe_store(i * Int(rs), r)
        i += stride


def isotonic_predict_strided_device(
    kx: MutPointer[Float64, MutUntrackedOrigin], ky: MutPointer[Float64, MutUntrackedOrigin], m: Int,
    t: MutPointer[Float64, MutUntrackedOrigin], ts: Int, n: Int, res: MutPointer[Float64, MutUntrackedOrigin],
    rs: Int,
) raises:
    """`ops.isotonic_predict_strided` on the device (the knots and the
    strided spans up once, one thread per element, the res span back once)."""
    if n <= 0:
        return
    var tl = (n - 1) * ts + 1
    var rl = (n - 1) * rs + 1
    var ctx = _ctx()
    var d_kx = _up_f64(ctx, kx, m)
    var d_ky = _up_f64(ctx, ky, m)
    var d_t = _up_f64(ctx, t, tl)
    var d_r = _up_f64(ctx, res, rl)
    ctx.enqueue_function[isotonic_predict_kernel](
        d_kx.unsafe_ptr(), d_ky.unsafe_ptr(), Int64(m), d_t.unsafe_ptr(), Int64(ts), Int64(n), d_r.unsafe_ptr(),
        Int64(rs), grid_dim=_blocks(n), block_dim=OPS_TPB,
    )
    ctx.enqueue_copy(dst_ptr=res.bitcast[UInt64](), src_buf=d_r)
    ctx.synchronize()
    _ = d_kx^
    _ = d_ky^
    _ = d_t^
    _ = d_r^


# -------------------------------------------------------- isotonic fit --
# `ops.isotonic_fit` on the device: the stable sort by (x, y, index) as
# bottom-up merge passes (each element finds its output slot by a binary
# search in the other run: a parallel merge, the unique stable order the host
# merge sort reaches), the unique-x sums by the segmented Hillis-Steele scan
# (`ops.iso_seg_scan`), and pool-adjacent-violators as rounds
# (`ops.iso_pav_round`): each round pools every maximal run of adjacent
# violators at once; the host reads one violation count per round.


@always_inline
def _lt64(a: UInt64, b: UInt64) -> Bool:
    if sf64_is_nan(a) or sf64_is_nan(b):
        return False
    return sf64_lt(a, b)


@always_inline
def _eq64(a: UInt64, b: UInt64) -> Bool:
    if sf64_is_nan(a) or sf64_is_nan(b):
        return False
    return not sf64_lt(a, b) and not sf64_lt(b, a)


@always_inline
def _iso_less(x: MutPointer[UInt64, MutAnyOrigin], y: MutPointer[UInt64, MutAnyOrigin], a: Int, b: Int) -> Bool:
    """Row a sorts strictly before row b: x[a] < x[b], or equal x and y[a] < y[b]."""
    var xa = x.unsafe_load(a)
    var xb = x.unsafe_load(b)
    return _lt64(xa, xb) or (_eq64(xa, xb) and _lt64(y.unsafe_load(a), y.unsafe_load(b)))


def iso_merge_kernel(
    x: MutPointer[UInt64, MutAnyOrigin], y: MutPointer[UInt64, MutAnyOrigin], src: MutPointer[Int32, MutAnyOrigin],
    dst: MutPointer[Int32, MutAnyOrigin], n: Int64, w: Int64,
):
    """One merge pass of width w: element p of the run pair [lo, lo + 2w)
    lands at lo + (its rank in its run) + (the elements of the other run
    that go before it: right ones strictly less, left ones not greater)."""
    var p = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var nn = Int(n)
    var ww = Int(w)
    while p < nn:
        var lo = (p // (2 * ww)) * 2 * ww
        var mid = min(lo + ww, nn)
        var hi = min(lo + 2 * ww, nn)
        var e = Int(src.unsafe_load(p))
        var pos: Int
        if p < mid:
            var a = mid
            var b = hi
            while a < b:  # first right element not strictly before e
                var c = (a + b) // 2
                if _iso_less(x, y, Int(src.unsafe_load(c)), e):
                    a = c + 1
                else:
                    b = c
            pos = lo + (p - lo) + (a - mid)
        else:
            var a = lo
            var b = mid
            while a < b:  # first left element that e sorts strictly before
                var c = (a + b) // 2
                if _iso_less(x, y, e, Int(src.unsafe_load(c))):
                    b = c
                else:
                    a = c + 1
            pos = lo + (p - mid) + (a - lo)
        dst.unsafe_store(pos, Int32(e))
        p += stride


def iso_gather_kernel(
    x: MutPointer[UInt64, MutAnyOrigin], y: MutPointer[UInt64, MutAnyOrigin], idx: MutPointer[Int32, MutAnyOrigin],
    n: Int64, xs: MutPointer[UInt64, MutAnyOrigin], v: MutPointer[UInt64, MutAnyOrigin],
):
    """xs[p] = x[idx[p]], v[p] = y[idx[p]] (copies)."""
    var p = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while p < Int(n):
        var i = Int(idx.unsafe_load(p))
        xs.unsafe_store(p, x.unsafe_load(i))
        v.unsafe_store(p, y.unsafe_load(i))
        p += stride


def iso_first_x_kernel(xs: MutPointer[UInt64, MutAnyOrigin], n: Int64, first: MutPointer[Int32, MutAnyOrigin]):
    """first[p] = 1 where a new x value starts (p == 0 or xs[p] != xs[p - 1])."""
    var p = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while p < Int(n):
        first.unsafe_store(p, Int32(1) if (p == 0 or not _eq64(xs.unsafe_load(p), xs.unsafe_load(p - 1))) else Int32(0))
        p += stride


def iso_start_kernel(first: MutPointer[Int32, MutAnyOrigin], n: Int64, seg: MutPointer[Int32, MutAnyOrigin],
                     gid: MutPointer[Int32, MutAnyOrigin]):
    """Scan seeds: seg[p] = p at a segment start else 0 (max-scanned into
    each entry's segment start), gid[p] = first[p] (sum-scanned into the
    1-based segment id)."""
    var p = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while p < Int(n):
        var f = first.unsafe_load(p)
        seg.unsafe_store(p, Int32(p) if f != 0 else Int32(0))
        gid.unsafe_store(p, f)
        p += stride


def iso_iscan_kernel(src: MutPointer[Int32, MutAnyOrigin], dst: MutPointer[Int32, MutAnyOrigin], n: Int64, s: Int64,
                     use_max: Int64):
    """One Hillis-Steele pass of an inclusive integer scan (sum, or max)."""
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while j < Int(n):
        var v = src.unsafe_load(j)
        if j >= Int(s):
            var u = src.unsafe_load(j - Int(s))
            v = max(u, v) if use_max != 0 else u + v
        dst.unsafe_store(j, v)
        j += stride


def iso_seg_pass_kernel(src: MutPointer[UInt64, MutAnyOrigin], dst: MutPointer[UInt64, MutAnyOrigin],
                        seg: MutPointer[Int32, MutAnyOrigin], n: Int64, s: Int64):
    """One pass of `ops.iso_seg_scan`: dst[j] = src[j - s] + src[j] when
    j - s is in j's segment, else src[j]."""
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while j < Int(n):
        var v = src.unsafe_load(j)
        if j - Int(s) >= Int(seg.unsafe_load(j)):
            v = sf64_add(src.unsafe_load(j - Int(s)), v)
        dst.unsafe_store(j, v)
        j += stride


def iso_iota_kernel(idx: MutPointer[Int32, MutAnyOrigin], n: Int64):
    var p = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while p < Int(n):
        idx.unsafe_store(p, Int32(p))
        p += stride


def _copy_head_i32(ctx: DeviceContext, mut dst: DeviceBuffer[DType.int32], src: DeviceBuffer[DType.int32],
                   n: Int) raises:
    """dst[0 .. n) = src[0 .. n) on the device (the buffers may be longer)."""
    var a = dst.create_sub_buffer[DType.int32](0, n)
    var b = src.create_sub_buffer[DType.int32](0, n)
    ctx.enqueue_copy(dst_buf=a, src_buf=b)
    _ = a^
    _ = b^


def _copy_head_u64(ctx: DeviceContext, mut dst: DeviceBuffer[DType.uint64], src: DeviceBuffer[DType.uint64],
                   n: Int) raises:
    var a = dst.create_sub_buffer[DType.uint64](0, n)
    var b = src.create_sub_buffer[DType.uint64](0, n)
    ctx.enqueue_copy(dst_buf=a, src_buf=b)
    _ = a^
    _ = b^


def _iscan(ctx: DeviceContext, mut a: DeviceBuffer[DType.int32], mut b: DeviceBuffer[DType.int32], n: Int,
           use_max: Bool) raises:
    """Inclusive integer scan of `a` (result in `a`), `b` scratch."""
    var s = 1
    while s < n:
        ctx.enqueue_function[iso_iscan_kernel](
            a.unsafe_ptr(), b.unsafe_ptr(), Int64(n), Int64(s), Int64(1 if use_max else 0),
            grid_dim=_blocks(n), block_dim=OPS_TPB,
        )
        _copy_head_i32(ctx, a, b, n)
        s *= 2


def _seg_scan(ctx: DeviceContext, mut v: DeviceBuffer[DType.uint64], mut tmp: DeviceBuffer[DType.uint64],
              seg: DeviceBuffer[DType.int32], n: Int) raises:
    """`ops.iso_seg_scan` of `v` (result in `v`), `tmp` scratch."""
    var s = 1
    while s < n:
        ctx.enqueue_function[iso_seg_pass_kernel](
            v.unsafe_ptr(), tmp.unsafe_ptr(), seg.unsafe_ptr(), Int64(n), Int64(s),
            grid_dim=_blocks(n), block_dim=OPS_TPB,
        )
        _copy_head_u64(ctx, v, tmp, n)
        s *= 2


def iso_unique_kernel(
    xs: MutPointer[UInt64, MutAnyOrigin], v: MutPointer[UInt64, MutAnyOrigin], first: MutPointer[Int32, MutAnyOrigin],
    seg: MutPointer[Int32, MutAnyOrigin], gid: MutPointer[Int32, MutAnyOrigin], n: Int64,
    ux: MutPointer[UInt64, MutAnyOrigin], bs: MutPointer[UInt64, MutAnyOrigin], bw: MutPointer[UInt64, MutAnyOrigin],
    blk: MutPointer[Int32, MutAnyOrigin],
):
    """At each x-group's last entry: knot g = its id: ux = the group's x,
    uy = sum / count, the block sums (count * uy, count), blk[g] = g."""
    var p = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while p < Int(n):
        if p == Int(n) - 1 or first.unsafe_load(p + 1) != 0:
            var g = Int(gid.unsafe_load(p)) - 1
            var st = Int(seg.unsafe_load(p))
            var c = sf64_from_int(p - st + 1)
            var uy = sf64_div(v.unsafe_load(p), c)
            ux.unsafe_store(g, xs.unsafe_load(st))
            bs.unsafe_store(g, sf64_mul(c, uy))
            bw.unsafe_store(g, c)
            blk.unsafe_store(g, Int32(g))
        p += stride


def iso_viol_kernel(bs: MutPointer[UInt64, MutAnyOrigin], bw: MutPointer[UInt64, MutAnyOrigin], nb: Int64,
                    first: MutPointer[Int32, MutAnyOrigin], cnt: MutPointer[Int32, MutAnyOrigin]):
    """first[b] = 0 where block b - 1 violates b (mean[b - 1] >= mean[b]),
    else 1; cnt[b] = 1 per violation (sum-scanned: the round's count)."""
    var b = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while b < Int(nb):
        var f = Int32(1)
        if b > 0:
            var m0 = sf64_div(bs.unsafe_load(b - 1), bw.unsafe_load(b - 1))
            var m1 = sf64_div(bs.unsafe_load(b), bw.unsafe_load(b))
            if _ge64(m0, m1):
                f = Int32(0)
        first.unsafe_store(b, f)
        cnt.unsafe_store(b, Int32(1) - f if b > 0 else Int32(0))
        b += stride


def iso_pool_kernel(
    bs: MutPointer[UInt64, MutAnyOrigin], bw: MutPointer[UInt64, MutAnyOrigin], first: MutPointer[Int32, MutAnyOrigin],
    gid: MutPointer[Int32, MutAnyOrigin], nb: Int64, nbs: MutPointer[UInt64, MutAnyOrigin],
    nbw: MutPointer[UInt64, MutAnyOrigin], nid: MutPointer[Int32, MutAnyOrigin],
):
    """After the segmented scans: each new block's sums from its last old
    block; nid[b] = the new block of old block b."""
    var b = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while b < Int(nb):
        var g = Int(gid.unsafe_load(b)) - 1
        nid.unsafe_store(b, Int32(g))
        if b == Int(nb) - 1 or first.unsafe_load(b + 1) != 0:
            nbs.unsafe_store(g, bs.unsafe_load(b))
            nbw.unsafe_store(g, bw.unsafe_load(b))
        b += stride


def iso_relabel_kernel(blk: MutPointer[Int32, MutAnyOrigin], m: Int64, nid: MutPointer[Int32, MutAnyOrigin]):
    var q = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while q < Int(m):
        blk.unsafe_store(q, nid.unsafe_load(Int(blk.unsafe_load(q))))
        q += stride


def iso_knots_kernel(
    ux: MutPointer[UInt64, MutAnyOrigin], bs: MutPointer[UInt64, MutAnyOrigin], bw: MutPointer[UInt64, MutAnyOrigin],
    blk: MutPointer[Int32, MutAnyOrigin], m: Int64, kx: MutPointer[UInt64, MutAnyOrigin],
    ky: MutPointer[UInt64, MutAnyOrigin],
):
    var q = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while q < Int(m):
        var b = Int(blk.unsafe_load(q))
        kx.unsafe_store(q, ux.unsafe_load(q))
        ky.unsafe_store(q, sf64_div(bs.unsafe_load(b), bw.unsafe_load(b)))
        q += stride


def _last_i32(ctx: DeviceContext, d: DeviceBuffer[DType.int32], n: Int) raises -> Int:
    var h = ctx.enqueue_create_host_buffer[DType.int32](1)
    var s = d.create_sub_buffer[DType.int32](n - 1, 1)
    ctx.enqueue_copy(dst_buf=h, src_buf=s)
    ctx.synchronize()
    var v = Int(h.unsafe_ptr().unsafe_load(0))
    _ = s^
    _ = h^
    return v


def isotonic_fit_device(
    x: MutPointer[Float64, MutUntrackedOrigin], y: MutPointer[Float64, MutUntrackedOrigin], n: Int,
    kx: MutPointer[Float64, MutUntrackedOrigin], ky: MutPointer[Float64, MutUntrackedOrigin],
) raises -> Int:
    """`ops.isotonic_fit` on the device (the law of its body, operation for
    operation): x and y up once, the knots back once; one integer read per
    pooling round (the violation count) and one for the knot count."""
    if n < 1:
        raise Error("x_trees isotonic_fit: no rows")
    var ctx = _ctx()
    var d_x = _up_f64(ctx, x, n)
    var d_y = _up_f64(ctx, y, n)
    var d_ia = ctx.enqueue_create_buffer[DType.int32](n)
    var d_ib = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_function[iso_iota_kernel](d_ia.unsafe_ptr(), Int64(n), grid_dim=_blocks(n), block_dim=OPS_TPB)
    var w = 1
    var in_a = True
    while w < n:
        if in_a:
            ctx.enqueue_function[iso_merge_kernel](
                d_x.unsafe_ptr(), d_y.unsafe_ptr(), d_ia.unsafe_ptr(), d_ib.unsafe_ptr(), Int64(n), Int64(w),
                grid_dim=_blocks(n), block_dim=OPS_TPB,
            )
        else:
            ctx.enqueue_function[iso_merge_kernel](
                d_x.unsafe_ptr(), d_y.unsafe_ptr(), d_ib.unsafe_ptr(), d_ia.unsafe_ptr(), Int64(n), Int64(w),
                grid_dim=_blocks(n), block_dim=OPS_TPB,
            )
        in_a = not in_a
        w *= 2
    var d_xs = ctx.enqueue_create_buffer[DType.uint64](n)
    var d_v = ctx.enqueue_create_buffer[DType.uint64](n)
    var d_tmp = ctx.enqueue_create_buffer[DType.uint64](n)
    ctx.enqueue_function[iso_gather_kernel](
        d_x.unsafe_ptr(), d_y.unsafe_ptr(), d_ia.unsafe_ptr() if in_a else d_ib.unsafe_ptr(), Int64(n),
        d_xs.unsafe_ptr(), d_v.unsafe_ptr(), grid_dim=_blocks(n), block_dim=OPS_TPB,
    )
    var d_first = ctx.enqueue_create_buffer[DType.int32](n)
    var d_seg = ctx.enqueue_create_buffer[DType.int32](n)
    var d_gid = ctx.enqueue_create_buffer[DType.int32](n)
    var d_s2 = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_function[iso_first_x_kernel](d_xs.unsafe_ptr(), Int64(n), d_first.unsafe_ptr(),
                                             grid_dim=_blocks(n), block_dim=OPS_TPB)
    ctx.enqueue_function[iso_start_kernel](d_first.unsafe_ptr(), Int64(n), d_seg.unsafe_ptr(), d_gid.unsafe_ptr(),
                                           grid_dim=_blocks(n), block_dim=OPS_TPB)
    _iscan(ctx, d_seg, d_s2, n, True)
    _iscan(ctx, d_gid, d_s2, n, False)
    _seg_scan(ctx, d_v, d_tmp, d_seg, n)
    var m = _last_i32(ctx, d_gid, n)
    var d_ux = ctx.enqueue_create_buffer[DType.uint64](m)
    var d_bs = ctx.enqueue_create_buffer[DType.uint64](m)
    var d_bw = ctx.enqueue_create_buffer[DType.uint64](m)
    var d_blk = ctx.enqueue_create_buffer[DType.int32](m)
    ctx.enqueue_function[iso_unique_kernel](
        d_xs.unsafe_ptr(), d_v.unsafe_ptr(), d_first.unsafe_ptr(), d_seg.unsafe_ptr(), d_gid.unsafe_ptr(), Int64(n),
        d_ux.unsafe_ptr(), d_bs.unsafe_ptr(), d_bw.unsafe_ptr(), d_blk.unsafe_ptr(),
        grid_dim=_blocks(n), block_dim=OPS_TPB,
    )
    # pooling rounds over nb blocks (scratch sized m, the first round's count)
    var d_nbs = ctx.enqueue_create_buffer[DType.uint64](m)
    var d_nbw = ctx.enqueue_create_buffer[DType.uint64](m)
    var d_nid = ctx.enqueue_create_buffer[DType.int32](m)
    var d_cnt = ctx.enqueue_create_buffer[DType.int32](m)
    var nb = m
    while nb > 1:
        ctx.enqueue_function[iso_viol_kernel](
            d_bs.unsafe_ptr(), d_bw.unsafe_ptr(), Int64(nb), d_first.unsafe_ptr(), d_cnt.unsafe_ptr(),
            grid_dim=_blocks(nb), block_dim=OPS_TPB,
        )
        _iscan(ctx, d_cnt, d_s2, nb, False)
        if _last_i32(ctx, d_cnt, nb) == 0:
            break
        ctx.enqueue_function[iso_start_kernel](d_first.unsafe_ptr(), Int64(nb), d_seg.unsafe_ptr(), d_gid.unsafe_ptr(),
                                               grid_dim=_blocks(nb), block_dim=OPS_TPB)
        _iscan(ctx, d_seg, d_s2, nb, True)
        _iscan(ctx, d_gid, d_s2, nb, False)
        _seg_scan(ctx, d_bs, d_tmp, d_seg, nb)
        _seg_scan(ctx, d_bw, d_tmp, d_seg, nb)
        ctx.enqueue_function[iso_pool_kernel](
            d_bs.unsafe_ptr(), d_bw.unsafe_ptr(), d_first.unsafe_ptr(), d_gid.unsafe_ptr(), Int64(nb),
            d_nbs.unsafe_ptr(), d_nbw.unsafe_ptr(), d_nid.unsafe_ptr(), grid_dim=_blocks(nb), block_dim=OPS_TPB,
        )
        ctx.enqueue_function[iso_relabel_kernel](d_blk.unsafe_ptr(), Int64(m), d_nid.unsafe_ptr(),
                                                 grid_dim=_blocks(m), block_dim=OPS_TPB)
        var nb2 = _last_i32(ctx, d_gid, nb)
        _copy_head_u64(ctx, d_bs, d_nbs, nb2)
        _copy_head_u64(ctx, d_bw, d_nbw, nb2)
        nb = nb2
    var d_kx = ctx.enqueue_create_buffer[DType.uint64](m)
    var d_ky = ctx.enqueue_create_buffer[DType.uint64](m)
    ctx.enqueue_function[iso_knots_kernel](
        d_ux.unsafe_ptr(), d_bs.unsafe_ptr(), d_bw.unsafe_ptr(), d_blk.unsafe_ptr(), Int64(m), d_kx.unsafe_ptr(),
        d_ky.unsafe_ptr(), grid_dim=_blocks(m), block_dim=OPS_TPB,
    )
    ctx.enqueue_copy(dst_ptr=kx.bitcast[UInt64](), src_buf=d_kx)
    ctx.enqueue_copy(dst_ptr=ky.bitcast[UInt64](), src_buf=d_ky)
    ctx.synchronize()
    _ = d_x^
    _ = d_y^
    _ = d_ia^
    _ = d_ib^
    _ = d_xs^
    _ = d_v^
    _ = d_tmp^
    _ = d_first^
    _ = d_seg^
    _ = d_gid^
    _ = d_s2^
    _ = d_ux^
    _ = d_bs^
    _ = d_bw^
    _ = d_blk^
    _ = d_nbs^
    _ = d_nbw^
    _ = d_nid^
    _ = d_cnt^
    _ = d_kx^
    _ = d_ky^
    return m
