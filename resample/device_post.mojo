# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The resample lane's finishing statistics on the device (lane cgr5-owed,
2026-10-03): the bootstrap
standard error, the BCa jackknife moments and the two integer counts (the
bias percentile's, the p-value's). Each was a host loop over n or over
n_resamples after the replicates came back.

ORDERS.

  * The OBSERVED statistic is `resample/estimator.mojo::_perm_observed`
    (lane/apple-fast-purity2), the replicate kernels' order.
  * The STANDARD ERROR and the BCa MOMENTS are grid-wide sums: the chunk
    trees, then `metrics/checks/pinned_sum.mojo`'s level tree over the chunk
    totals (`host_grid_sum` on the host column, `resample/checks/
    intervals.mojo`). Their words changed from the chained order on every
    vendor and the host column together.
  * The COUNTS are integers: exact and order free.
"""

from std.gpu import block_idx, grid_dim, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import ftz, identical_div, identical_mul, identical_sqrt
from metrics.checks.pinned_sum import (
    canonicalize_nan,
    PINNED_SUM_TPB,
    PINNED_SUM_W,
    chunk_count,
    fold_partials_level_kernel,
    fold_partials_levels,
    fold_scratch_len,
    virtual_block_sum,
)
from resample.checks.statistics import STAT_DIFF_MEANS, STAT_MEAN, _mean_of_sum

comptime _F32P = MutPointer[Float32, MutAnyOrigin]
comptime _I32P = MutPointer[Int32, MutAnyOrigin]

comptime POST_TPB = 256
comptime POST_MAX_BLOCKS = 256


# ---------------------------------------------------------------- kernels ----


def dev_sq_kernel(dst: _F32P, src: _F32P, n_in: Int32, mean_sum: _F32P, scale_in: Int32, mode: Int32):
    """Element maps over `src[0:n]` against `m = _mean_of_sum(mean_sum[0],
    scale)`. mode 0 (standard error): `dst[i] = ftz(identical_mul(d, d))`,
    `d = ftz(src[i] - m)`. mode 1 (BCa): `u = ftz(identical_mul(n - 1,
    ftz(m - src[i])))`, `dst[i] = u^2`, `dst[n + i] = u^3`."""
    var i = Int(block_idx.x) * POST_TPB + Int(thread_idx.x)
    var n = Int(n_in)
    if i >= n:
        return
    var m = _mean_of_sum(mean_sum.unsafe_load(0), Int(scale_in))
    if mode == 0:
        var d = ftz(src.unsafe_load(i) - m)
        dst.unsafe_store(i, ftz(identical_mul(d, d)))
    else:
        var u = ftz(identical_mul(Float32(n - 1), ftz(m - src.unsafe_load(i))))
        var sq = ftz(identical_mul(u, u))
        dst.unsafe_store(i, sq)
        dst.unsafe_store(n + i, ftz(identical_mul(sq, u)))


def count_kernel(part: _I32P, v: _F32P, n_in: Int32, a: Float32, b: Float32, mode: Int32):
    """Block partial counts over `v[0:n]` (grid-stride): mode 0 the
    p-value's (`v <= a`, `v >= b`), mode 1 the bias percentile's (`v < a`,
    `v <= a`). part[2 * block + 0/1]."""
    var c0 = stack_allocation[POST_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var c1 = stack_allocation[POST_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var n = Int(n_in)
    var k0 = Int32(0)
    var k1 = Int32(0)
    var i = Int(block_idx.x) * POST_TPB + tid
    var stride = Int(grid_dim.x) * POST_TPB
    while i < n:
        var x = v.unsafe_load(i)
        if mode == 0:
            if x <= a:
                k0 += 1
            if x >= b:
                k1 += 1
        else:
            if x < a:
                k0 += 1
            if x <= a:
                k1 += 1
        i += stride
    c0[tid] = k0
    c1[tid] = k1
    barrier()
    var h = POST_TPB // 2
    while h > 0:
        if tid < h:
            c0[tid] = c0[tid] + c0[tid + h]
            c1[tid] = c1[tid] + c1[tid + h]
        barrier()
        h //= 2
    if tid == 0:
        part.unsafe_store(2 * Int(block_idx.x), c0[0])
        part.unsafe_store(2 * Int(block_idx.x) + 1, c1[0])


def count_fold_kernel(res: _I32P, part: _I32P, blocks_in: Int32):
    """The block counts added (integers: any order is exact)."""
    var c0 = stack_allocation[POST_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var c1 = stack_allocation[POST_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var k0 = Int32(0)
    var k1 = Int32(0)
    if tid < Int(blocks_in):
        k0 = part.unsafe_load(2 * tid)
        k1 = part.unsafe_load(2 * tid + 1)
    c0[tid] = k0
    c1[tid] = k1
    barrier()
    var h = POST_TPB // 2
    while h > 0:
        if tid < h:
            c0[tid] = c0[tid] + c0[tid + h]
            c1[tid] = c1[tid] + c1[tid + h]
        barrier()
        h //= 2
    if tid == 0:
        res.unsafe_store(0, c0[0])
        res.unsafe_store(1, c1[0])


def diff_map_kernel(dst: _F32P, a: _F32P, b: _F32P, n_in: Int32, c: Float32, mode: Int32):
    """Element maps of the unpaired bootstrap: mode 0 `dst[i] =
    canonicalize_nan(ftz(a[i] - b[i]))` (the replicate differences), mode 1
    `ftz(a[i] - c)`, mode 2 `ftz(c - a[i])` (the two jackknife shifts)."""
    var i = Int(block_idx.x) * POST_TPB + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    if mode == 0:
        dst.unsafe_store(i, canonicalize_nan(ftz(a.unsafe_load(i) - b.unsafe_load(i))))
    elif mode == 1:
        dst.unsafe_store(i, ftz(a.unsafe_load(i) - c))
    else:
        dst.unsafe_store(i, ftz(c - a.unsafe_load(i)))


def enqueue_diff_map(
    ctx: DeviceContext, mut dst: DeviceBuffer[DType.float32], mut a: DeviceBuffer[DType.float32],
    b: _F32P, n: Int, c: Float32, mode: Int,
) raises:
    ctx.enqueue_function[diff_map_kernel](
        dst.unsafe_ptr(), a.unsafe_ptr(), b, Int32(n), c, Int32(mode),
        grid_dim=((n + POST_TPB - 1) // POST_TPB, 1, 1), block_dim=(POST_TPB, 1, 1),
    )


def chunk_tree_kernel(part: _F32P, v: _F32P, n_in: Int32):
    """`host_tree_sum`'s first stage: block c folds chunk c of `v[0:n]`
    (`+0.0` past n) with `virtual_block_sum` into part[c]."""
    comptime R = PINNED_SUM_W // PINNED_SUM_TPB
    var tid = Int(thread_idx.x)
    var c = Int(block_idx.x)
    var n = Int(n_in)
    var vals = SIMD[DType.float32, R](0.0)
    comptime for r in range(R):
        var i = c * PINNED_SUM_W + tid + r * PINNED_SUM_TPB
        if i < n:
            vals[r] = v.unsafe_load(i)
    var t = virtual_block_sum[PINNED_SUM_TPB](vals)
    if tid == 0:
        part.unsafe_store(c, t)


def chunk_chain_kernel(res: _F32P, part: _F32P, chunks_in: Int32):
    """`host_fold_partials`: the chunk totals ascending from +0.0, flushed;
    one thread over the n / PINNED_SUM_W chunk totals (the chain every
    replicate block runs over its own chunks, so theta-hat keeps the
    replicates' words)."""
    if Int(block_idx.x) != 0 or Int(thread_idx.x) != 0:
        return
    var acc = Float32(0.0)
    for c in range(Int(chunks_in)):
        acc = ftz(acc + part.unsafe_load(c))
    res.unsafe_store(0, acc)


def pe_map_kernel(dst: _F32P, x: _F32P, n_in: Int32, nf_in: Int32, means: _F32P, mode: Int32):
    """`point_estimate_host`'s per-row maps, row i: mode 0 `dst[i] =
    ftz(x[i, 0])`, `dst[n + i] = ftz(x[i, 1])` (0.0 for one column); mode 1
    (std) `d = ftz(ftz(x[i, 0]) - m)`, `dst[i] = ftz(d * d)`; mode 2
    (pearson) `dst[i]`, `dst[n + i]`, `dst[2n + i]` = dx dy, dx dx, dy dy.
    The means are `_mean_of_sum(means[0 / 1], n)`."""
    var i = Int(block_idx.x) * POST_TPB + Int(thread_idx.x)
    var n = Int(n_in)
    var nf = Int(nf_in)
    if i >= n:
        return
    var a = ftz(x.unsafe_load(i * nf))
    var b = Float32(0.0)
    if nf > 1:
        b = ftz(x.unsafe_load(i * nf + 1))
    if mode == 0:
        dst.unsafe_store(i, a)
        dst.unsafe_store(n + i, b)
    elif mode == 1:
        var d = ftz(a - _mean_of_sum(means.unsafe_load(0), n))
        dst.unsafe_store(i, ftz(identical_mul(d, d)))
    else:
        var dx = ftz(a - _mean_of_sum(means.unsafe_load(0), n))
        var dy = ftz(b - _mean_of_sum(means.unsafe_load(1), n))
        dst.unsafe_store(i, ftz(identical_mul(dx, dy)))
        dst.unsafe_store(n + i, ftz(identical_mul(dx, dx)))
        dst.unsafe_store(2 * n + i, ftz(identical_mul(dy, dy)))


# ---------------------------------------------------------------- drivers ----


def _read_f32(ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int) raises -> List[Float32]:
    var h = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf)
    ctx.synchronize()
    var res = List[Float32]()
    for i in range(n):
        res.append(h.unsafe_ptr().unsafe_load(i))
    _ = h^
    return res^


def device_grid_sum_into(
    ctx: DeviceContext, res: _F32P, vals: _F32P, n: Int,
    mut part: DeviceBuffer[DType.float32],
    mut s0: DeviceBuffer[DType.float32],
    mut s1: DeviceBuffer[DType.float32],
) raises:
    """`out[0]` = `host_grid_sum(vals, n)`'s words: the chunk trees into
    `part` (chunk_count(n) floats), the level tree over them (`s0`, `s1`:
    fold_scratch_len(chunk_count(n)) floats each), the last level into res.
    Enqueues only."""
    var chunks = chunk_count(n)
    var pp = part.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    ctx.enqueue_function[fold_partials_level_kernel[PINNED_SUM_TPB]](
        vals, Int32(n), pp, grid_dim=chunks, block_dim=PINNED_SUM_TPB,
    )
    var lv = fold_partials_levels(ctx, pp, chunks, s0, s1)
    ctx.enqueue_function[fold_partials_level_kernel[PINNED_SUM_TPB]](
        lv[0], Int32(lv[1]), res, grid_dim=chunk_count(lv[1]), block_dim=PINNED_SUM_TPB,
    )


def device_standard_error(ctx: DeviceContext, mut dist: DeviceBuffer[DType.float32], n_resamples: Int) raises -> Float32:
    """`distribution_standard_error`'s value from the device distribution:
    two grid-wide sums (the mean, then the squared deviations), the ddof=1
    root on one scalar."""
    if n_resamples < 2:
        raise Error(
            "bootstrap: the standard error needs at least 2 resamples (it is"
            " the ddof=1 standard deviation of the bootstrap distribution,"
            " scipy.stats.bootstrap's correction=1); got n_resamples="
            + String(n_resamples)
        )
    var chunks = chunk_count(n_resamples)
    var part = ctx.enqueue_create_buffer[DType.float32](chunks)
    var s0 = ctx.enqueue_create_buffer[DType.float32](fold_scratch_len(chunks))
    var s1 = ctx.enqueue_create_buffer[DType.float32](fold_scratch_len(chunks))
    var sq = ctx.enqueue_create_buffer[DType.float32](n_resamples)
    var sums = ctx.enqueue_create_buffer[DType.float32](2)
    var sp = sums.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var dp = dist.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var qp = sq.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    device_grid_sum_into(ctx, sp, dp, n_resamples, part, s0, s1)
    ctx.enqueue_function[dev_sq_kernel](
        qp, dp, Int32(n_resamples), sp, Int32(n_resamples), Int32(0),
        grid_dim=((n_resamples + POST_TPB - 1) // POST_TPB, 1, 1), block_dim=(POST_TPB, 1, 1),
    )
    device_grid_sum_into(ctx, sp + 1, qp, n_resamples, part, s0, s1)
    var ssd = _read_f32(ctx, sums, 2)[1]
    _ = part^
    _ = s0^
    _ = s1^
    _ = sq^
    _ = sums^
    return ftz(identical_sqrt(ftz(identical_div(ssd, Float32(n_resamples - 1)))))


@fieldwise_init
struct DeviceBcaMoments(ImplicitlyCopyable, Movable):
    var num: Float32
    var den: Float32


def device_bca_moments(ctx: DeviceContext, mut jack: DeviceBuffer[DType.float32], n: Int) raises -> DeviceBcaMoments:
    """`_bca_moments` from the device jackknife: the mean, then `U^2` and
    `U^3` as grid-wide sums; the two divisions on the scalars."""
    if n < 2:
        raise Error(
            "bootstrap: the BCa acceleration needs at least 2 observations;"
            " got n=" + String(n)
        )
    var chunks = chunk_count(n)
    var part = ctx.enqueue_create_buffer[DType.float32](chunks)
    var s0 = ctx.enqueue_create_buffer[DType.float32](fold_scratch_len(chunks))
    var s1 = ctx.enqueue_create_buffer[DType.float32](fold_scratch_len(chunks))
    var u = ctx.enqueue_create_buffer[DType.float32](2 * n)
    var sums = ctx.enqueue_create_buffer[DType.float32](3)
    var sp = sums.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var jp = jack.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var up = u.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    device_grid_sum_into(ctx, sp, jp, n, part, s0, s1)
    ctx.enqueue_function[dev_sq_kernel](
        up, jp, Int32(n), sp, Int32(n), Int32(1),
        grid_dim=((n + POST_TPB - 1) // POST_TPB, 1, 1), block_dim=(POST_TPB, 1, 1),
    )
    device_grid_sum_into(ctx, sp + 1, up, n, part, s0, s1)
    device_grid_sum_into(ctx, sp + 2, up + n, n, part, s0, s1)
    var h = _read_f32(ctx, sums, 3)
    _ = part^
    _ = s0^
    _ = s1^
    _ = u^
    _ = sums^
    var nf = Float32(n)
    var n2 = ftz(identical_mul(nf, nf))
    var n3 = ftz(identical_mul(n2, nf))
    return DeviceBcaMoments(ftz(identical_div(h[2], n3)), ftz(identical_div(h[1], n2)))


def device_counts(
    ctx: DeviceContext, mut v: DeviceBuffer[DType.float32], n: Int, a: Float32, b: Float32, mode: Int,
) raises -> Tuple[Int, Int]:
    """`count_kernel`'s two counts over `v[0:n]`."""
    var blocks = (n + POST_TPB - 1) // POST_TPB
    if blocks > POST_MAX_BLOCKS:
        blocks = POST_MAX_BLOCKS
    if blocks < 1:
        blocks = 1
    var part = ctx.enqueue_create_buffer[DType.int32](2 * blocks)
    var res = ctx.enqueue_create_buffer[DType.int32](2)
    ctx.enqueue_function[count_kernel](
        part.unsafe_ptr(), v.unsafe_ptr(), Int32(n), a, b, Int32(mode),
        grid_dim=(blocks, 1, 1), block_dim=(POST_TPB, 1, 1),
    )
    ctx.enqueue_function[count_fold_kernel](
        res.unsafe_ptr(), part.unsafe_ptr(), Int32(blocks),
        grid_dim=(1, 1, 1), block_dim=(POST_TPB, 1, 1),
    )
    var h = ctx.enqueue_create_host_buffer[DType.int32](2)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=res)
    ctx.synchronize()
    var r = (Int(h.unsafe_ptr().unsafe_load(0)), Int(h.unsafe_ptr().unsafe_load(1)))
    _ = h^
    _ = part^
    _ = res^
    return r


def device_tree_sum_into(ctx: DeviceContext, res: _F32P, v: _F32P, n: Int, mut part: DeviceBuffer[DType.float32]) raises:
    """`res[0]` = `host_tree_sum(v, n)`'s words: the chunk trees in parallel
    into `part` (chunk_count(n) floats), then the chunk chain. Enqueues only."""
    var chunks = chunk_count(n)
    var pp = part.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    ctx.enqueue_function[chunk_tree_kernel](
        pp, v, Int32(n), grid_dim=(max(chunks, 1), 1, 1), block_dim=(PINNED_SUM_TPB, 1, 1),
    )
    ctx.enqueue_function[chunk_chain_kernel](res, pp, Int32(chunks), grid_dim=(1, 1, 1), block_dim=(1, 1, 1))


def device_point_moments(
    ctx: DeviceContext, mut x: DeviceBuffer[DType.float32], n: Int, nf: Int, stat_mode: Int,
) raises -> List[Float32]:
    """The sums `point_estimate_host` folds, on the device in
    `host_tree_sum`'s order, one download: stat_mode 0 [sum a, sum b];
    1 (std) [sum a, sum b, sum sq]; 2 (pearson) [sum a, sum b, sum dxdy,
    sum dxdx, sum dydy]."""
    var maps = ctx.enqueue_create_buffer[DType.float32](3 * max(n, 1))
    var part = ctx.enqueue_create_buffer[DType.float32](max(chunk_count(n), 1))
    var sums = ctx.enqueue_create_buffer[DType.float32](5)
    var mp = maps.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var sp = sums.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var xp = x.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var grid = (n + POST_TPB - 1) // POST_TPB
    ctx.enqueue_memset(sums, Float32(0.0))
    ctx.enqueue_function[pe_map_kernel](mp, xp, Int32(n), Int32(nf), sp, Int32(0), grid_dim=(grid, 1, 1), block_dim=(POST_TPB, 1, 1))
    device_tree_sum_into(ctx, sp, mp, n, part)
    device_tree_sum_into(ctx, sp + 1, mp + n, n, part)
    var k = 2
    if stat_mode == 1:
        ctx.enqueue_function[pe_map_kernel](mp, xp, Int32(n), Int32(nf), sp, Int32(1), grid_dim=(grid, 1, 1), block_dim=(POST_TPB, 1, 1))
        device_tree_sum_into(ctx, sp + 2, mp, n, part)
        k = 3
    elif stat_mode == 2:
        ctx.enqueue_function[pe_map_kernel](mp, xp, Int32(n), Int32(nf), sp, Int32(2), grid_dim=(grid, 1, 1), block_dim=(POST_TPB, 1, 1))
        device_tree_sum_into(ctx, sp + 2, mp, n, part)
        device_tree_sum_into(ctx, sp + 3, mp + n, n, part)
        device_tree_sum_into(ctx, sp + 4, mp + 2 * n, n, part)
        k = 5
    var h = _read_f32(ctx, sums, 5)
    _ = maps^
    _ = part^
    _ = sums^
    var r = List[Float32]()
    for i in range(k):
        r.append(h[i])
    return r^
