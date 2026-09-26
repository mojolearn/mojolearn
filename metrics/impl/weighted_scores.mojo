# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Weighted accuracy and weighted R2. RAFT's `scores.cuh` has no weighted
arm and cuML 26.08's Python weights both in cupy, so these follow the
scikit-learn reference definitions on the repository's pinned-sum path
(`metrics/checks/pinned_sum.mojo`), the same `PINNED_SUM_W = 256` slab tree
and ascending host fold `r2_score` uses:

    accuracy_score(y_true, y_pred, sample_weight=w)
        = np.average(y_true == y_pred, weights=w)
        = sum_i [y_true_i == y_pred_i] w_i  /  sum_i w_i

    r2_score(y_true, y_pred, sample_weight=w)          (force_finite=True)
        numerator   = sum_i w_i (y_i - yhat_i)^2
        y_avg       = sum_i w_i y_i / sum_i w_i
        denominator = sum_i w_i (y_i - y_avg)^2
        denominator == 0 -> 1.0 if numerator == 0 else 0.0
        otherwise   1 - numerator / denominator

Float32 throughout, as the unweighted kernels are. Every operand and every
product is flushed (`ftz`), so a subnormal weight or target cannot become a
normal term on a device without flush-to-zero; each quotient is the
`identical_div` seam; the R2 ratio is `scores.mojo::r2_epilogue`, the
unweighted metric's own guards and NaN canonicalization. The chunk totals
are read back and folded on the host (`host_fold_partials`), so the value
is a function of `n` and the data only, never of the launch.

The Python surface validates the weights (finite, non-negative, positive
total) before any launch; a zero total never reaches a division here.
"""

from max.gpu import thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import ftz, identical_div
from metrics.checks.pinned_sum import (
    PINNED_SUM_TPB,
    PINNED_SUM_W,
    chunk_count,
    host_fold_partials,
    linear_block_id,
    physical_block_count,
    virtual_block_sum,
)
from metrics.impl.stats.detail.scores import r2_epilogue
from metrics.checks.device_io import download_f32


def weighted_equal_chunks_kernel[
    block_size: Int
](
    y_true: MutPointer[Int32, MutAnyOrigin],
    y_pred: MutPointer[Int32, MutAnyOrigin],
    w: MutPointer[Float32, MutAnyOrigin],
    n: Int32,
    num_partials: MutPointer[Float32, MutAnyOrigin],
    den_partials: MutPointer[Float32, MutAnyOrigin],
):
    """`num = tree(w where y_true == y_pred)` and `den = tree(w)`, one pass,
    two trees of the same shape."""
    comptime R = PINNED_SUM_W // block_size
    var tid = Int(thread_idx.x)
    var chunks = chunk_count(Int(n))
    var chunk = linear_block_id()
    while chunk < chunks:
        var num = SIMD[DType.float32, R](0.0)
        var den = SIMD[DType.float32, R](0.0)
        comptime for r in range(R):
            var i = chunk * PINNED_SUM_W + tid + r * block_size
            if i < Int(n):
                var wi = ftz(w.unsafe_load(i))
                den[r] = wi
                if y_true.unsafe_load(i) == y_pred.unsafe_load(i):
                    num[r] = wi
        var sn = virtual_block_sum[block_size](num)
        var sd = virtual_block_sum[block_size](den)
        if tid == 0:
            num_partials.unsafe_store(chunk, ftz(sn))
            den_partials.unsafe_store(chunk, ftz(sd))
        chunk += physical_block_count()


def weighted_mean_chunks_kernel[
    block_size: Int
](
    y: MutPointer[Float32, MutAnyOrigin],
    w: MutPointer[Float32, MutAnyOrigin],
    n: Int32,
    wy_partials: MutPointer[Float32, MutAnyOrigin],
    w_partials: MutPointer[Float32, MutAnyOrigin],
):
    """`tree(w * y)` and `tree(w)`: the two sums of `np.average(y, weights=w)`."""
    comptime R = PINNED_SUM_W // block_size
    var tid = Int(thread_idx.x)
    var chunks = chunk_count(Int(n))
    var chunk = linear_block_id()
    while chunk < chunks:
        var wy = SIMD[DType.float32, R](0.0)
        var ww = SIMD[DType.float32, R](0.0)
        comptime for r in range(R):
            var i = chunk * PINNED_SUM_W + tid + r * block_size
            if i < Int(n):
                var wi = ftz(w.unsafe_load(i))
                wy[r] = ftz(wi * ftz(y.unsafe_load(i)))
                ww[r] = wi
        var swy = virtual_block_sum[block_size](wy)
        var sw = virtual_block_sum[block_size](ww)
        if tid == 0:
            wy_partials.unsafe_store(chunk, ftz(swy))
            w_partials.unsafe_store(chunk, ftz(sw))
        chunk += physical_block_count()


def weighted_sse_ssto_chunks_kernel[
    block_size: Int
](
    y: MutPointer[Float32, MutAnyOrigin],
    y_hat: MutPointer[Float32, MutAnyOrigin],
    w: MutPointer[Float32, MutAnyOrigin],
    n: Int32,
    y_avg: Float32,
    sse_partials: MutPointer[Float32, MutAnyOrigin],
    ssto_partials: MutPointer[Float32, MutAnyOrigin],
):
    """`tree(w * (y - y_hat)^2)` and `tree(w * (y - y_avg)^2)`, scikit-learn's
    `weight * (a - b) ** 2`: the square first, then the weight."""
    comptime R = PINNED_SUM_W // block_size
    var tid = Int(thread_idx.x)
    var chunks = chunk_count(Int(n))
    var chunk = linear_block_id()
    while chunk < chunks:
        var se = SIMD[DType.float32, R](0.0)
        var st = SIMD[DType.float32, R](0.0)
        comptime for r in range(R):
            var i = chunk * PINNED_SUM_W + tid + r * block_size
            if i < Int(n):
                var wi = ftz(w.unsafe_load(i))
                var yi = ftz(y.unsafe_load(i))
                var d1 = ftz(yi - ftz(y_hat.unsafe_load(i)))
                var d2 = ftz(yi - y_avg)
                se[r] = ftz(wi * ftz(d1 * d1))
                st[r] = ftz(wi * ftz(d2 * d2))
        var sse = virtual_block_sum[block_size](se)
        var ssto = virtual_block_sum[block_size](st)
        if tid == 0:
            sse_partials.unsafe_store(chunk, ftz(sse))
            ssto_partials.unsafe_store(chunk, ftz(ssto))
        chunk += physical_block_count()


def weighted_accuracy_finalize_kernel(
    num_partials: MutPointer[Float32, MutAnyOrigin],
    den_partials: MutPointer[Float32, MutAnyOrigin],
    chunks: Int32,
    result: MutPointer[Float32, MutAnyOrigin],
):
    if Int(thread_idx.x) == 0:
        var num = Float32(0)
        var den = Float32(0)
        for c in range(Int(chunks)):
            num = ftz(num + num_partials.unsafe_load(c))
            den = ftz(den + den_partials.unsafe_load(c))
        result.unsafe_store(0,ftz(identical_div(num,den)) if den > 0 else Float32(0))
        result.unsafe_store(1,den)


def _fold(
    ctx: DeviceContext, mut partials: DeviceBuffer[DType.float32], chunks: Int
) raises -> Float32:
    var h = ctx.enqueue_create_host_buffer[DType.float32](chunks)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=partials)
    ctx.synchronize()
    var lst = List[Float32]()
    for c in range(chunks):
        lst.append(h.unsafe_ptr().unsafe_load(c))
    _ = h^
    return host_fold_partials(lst, chunks)


def weighted_accuracy_score(
    ctx: DeviceContext,
    mut y_true: DeviceBuffer[DType.int32],
    mut y_pred: DeviceBuffer[DType.int32],
    mut w: DeviceBuffer[DType.float32],
    n: Int,
) raises -> Float32:
    """`np.average(y_true == y_pred, weights=w)` in Float32."""
    if n <= 0 or n > 2147483647:
        raise Error("weighted accuracy_score: n must be in [1, 2^31 - 1], got " + String(n))
    var chunks = chunk_count(n)
    var num_p = ctx.enqueue_create_buffer[DType.float32](chunks)
    var den_p = ctx.enqueue_create_buffer[DType.float32](chunks)
    ctx.enqueue_function[weighted_equal_chunks_kernel[PINNED_SUM_TPB]](
        y_true.unsafe_ptr(),
        y_pred.unsafe_ptr(),
        w.unsafe_ptr(),
        Int32(n),
        num_p.unsafe_ptr(),
        den_p.unsafe_ptr(),
        grid_dim=(chunks, 1, 1),
        block_dim=(PINNED_SUM_TPB, 1, 1),
    )
    var result = ctx.enqueue_create_buffer[DType.float32](2)
    ctx.enqueue_function[weighted_accuracy_finalize_kernel](
        num_p.unsafe_ptr(),den_p.unsafe_ptr(),Int32(chunks),result.unsafe_ptr(),
        grid_dim=1,block_dim=32,
    )
    var host = download_f32(ctx,result,2)
    var score = host[0]
    var den = host[1]
    _ = num_p^
    _ = den_p^
    _ = result^
    if den <= Float32(0.0):
        raise Error("weighted accuracy_score: the weights must have positive total")
    return score


def weighted_r2_score(
    ctx: DeviceContext,
    mut y: DeviceBuffer[DType.float32],
    mut y_hat: DeviceBuffer[DType.float32],
    mut w: DeviceBuffer[DType.float32],
    n: Int,
) raises -> Float32:
    """scikit-learn's weighted `r2_score` with `force_finite=True`, Float32."""
    if n <= 0 or n > 2147483647:
        raise Error("weighted r2_score: n must be in [1, 2^31 - 1], got " + String(n))
    var chunks = chunk_count(n)
    var wy_p = ctx.enqueue_create_buffer[DType.float32](chunks)
    var w_p = ctx.enqueue_create_buffer[DType.float32](chunks)
    ctx.enqueue_function[weighted_mean_chunks_kernel[PINNED_SUM_TPB]](
        y.unsafe_ptr(),
        w.unsafe_ptr(),
        Int32(n),
        wy_p.unsafe_ptr(),
        w_p.unsafe_ptr(),
        grid_dim=(chunks, 1, 1),
        block_dim=(PINNED_SUM_TPB, 1, 1),
    )
    var swy = _fold(ctx, wy_p, chunks)
    var sw = _fold(ctx, w_p, chunks)
    _ = wy_p^
    _ = w_p^
    if sw <= Float32(0.0):
        raise Error("weighted r2_score: the weights must have positive total")
    var y_avg = ftz(identical_div(swy, sw))
    var sse_p = ctx.enqueue_create_buffer[DType.float32](chunks)
    var ssto_p = ctx.enqueue_create_buffer[DType.float32](chunks)
    ctx.enqueue_function[weighted_sse_ssto_chunks_kernel[PINNED_SUM_TPB]](
        y.unsafe_ptr(),
        y_hat.unsafe_ptr(),
        w.unsafe_ptr(),
        Int32(n),
        y_avg,
        sse_p.unsafe_ptr(),
        ssto_p.unsafe_ptr(),
        grid_dim=(chunks, 1, 1),
        block_dim=(PINNED_SUM_TPB, 1, 1),
    )
    var sse = _fold(ctx, sse_p, chunks)
    var ssto = _fold(ctx, ssto_p, chunks)
    _ = sse_p^
    _ = ssto_p^
    return r2_epilogue(sse, ssto)
