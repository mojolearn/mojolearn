# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The KPSS test of a batch as ONE LAUNCH (lane/apple-fast-tsa2,
`-D MOJOLEARN_TSA2_KPSS`; FAST + Apple only, reached from
`tsa/estimator.mojo::kpss_test_host` under that define, for d = D = 0 and
n_obs <= KPSS_FUSED_MAX_N).

`stationarity.mojo::_kpss_test` is eight launches over eight buffers with a
wait, after an upload that waits and a host scan of the input for
non-finite values that waits again, then a download that waits: four waits
and a host pass over every word for a 3.0 ms cell against statsmodels'
2.6 ms. Here one block of KPSS_FUSED_TPB threads per series holds the
centred series in threadgroup memory and runs every stage of
`_kpss_test` in turn, barrier-separated:
  1. the sum (thread t folds t, t + TPB, ... ascending, then the block
     fold: `series_sum_kernel[False]`'s chain) -> the mean, and the finite
     scan of the same read (a per-series word: -1, or an index of a
     non-finite value, raised by name after the wait);
  2. centring in place (`center_kernel`'s subtraction) and the sum of
     squares (`series_sum_kernel[True]`'s chain) -> s2A;
  3. the lagged products per cell (`s2B_accumulation_kernel`'s chain) and
     their sum (`series_sum_kernel[False]`'s chain) -> s2B;
  4. the inclusive scan of the centred series: each thread's contiguous
     chunk folded ascending, the chunk totals through the block prefix
     sum, then the chunk's running sums and their squares, block-folded
     -> eta (FAST's fold of `cumsum_by_series_kernel`'s serial scan and of
     the strided eta sum: a different association, FAST only);
  5. thread 0: the statistic, the p-value and the flag
     (`kpss_stationarity_check_kernel`).
Stages 1-3 fold the same partials in the same order as the eight-launch
path; under FAST the block fold is the library's, as there.
"""
from std.gpu import block_idx, thread_idx
from std.math import isfinite
from std.memory import stack_allocation
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from bindings.hostptr import copy_f32
from core.pinned_reduce import pinned_block_min, pinned_block_prefix_sum, pinned_block_sum
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, ftz, identical_mul_add
from tsa.impl.timeSeries.stationarity import kpss_lags, kpss_pvalue, kpss_s2B_coefficients, kpss_stat_from_sums

#: the switch (default OFF; FAST + Apple only)
comptime TSA2_KPSS = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_TSA2_KPSS"]()
)
#: lane/apple-fast-gap-tsa: the KPSS test as one launch over ONE device
#: buffer (input and packed outputs) and ONE host stage, one download
#: (`kpss_rounds`); -D MOJOLEARN_TSA_FAST_KPSS_PACK (default OFF; FAST +
#: Apple only)
comptime TSA_KPSS_PACK = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_TSA_FAST_KPSS_PACK"]()
)
#: lane/apple-fast-gap-tsa: `select_d` as one launch (every round of a
#: series inside its block, the first stationary order chosen there), the
#: same packed buffers, one download; -D MOJOLEARN_TSA_FAST_SELD_FUSED
#: (default OFF; FAST + Apple only)
comptime TSA_SELD_FUSED = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_TSA_FAST_SELD_FUSED"]()
)
#: words per series in the packed output: stat, flag, first bad index
#: (-1 none), chosen d
comptime KPSS_PACK_W = 4
#: threads of the one block per series
comptime KPSS_FUSED_TPB = 256
#: threadgroup words for the centred series (16 KB); longer series take
#: the eight-launch path
comptime KPSS_FUSED_MAX_N = 4096


def kpss_fused_kernel(
    results: MutPointer[UInt8, MutAnyOrigin],
    stat_out: MutPointer[Float32, MutAnyOrigin],
    bad: MutPointer[Int32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    n_obs_in: Int32,
    lags_in: Int32,
    ratio: Float32,
    coeff_a: Float32,
    coeff_b: Float32,
    pval_threshold: Float32,
):
    """One block per series b = block_idx.x; the five stages above.
    `ratio` is the host's `1 / n_obs_f` (`_kpss_test`'s mean scale)."""
    var n = Int(n_obs_in)
    var lags = Int(lags_in)
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var base = b * n
    var sh = stack_allocation[KPSS_FUSED_MAX_N, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var mean_w = stack_allocation[1, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var bad_w = stack_allocation[1, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    if tid == 0:
        bad_w[0] = Int32(-1)
    barrier()
    # 1. the sum and the finite scan
    var acc = Float32(0.0)
    var first_bad = -1
    var t = tid
    while t < n:
        var raw = y.unsafe_load(base + t)
        if first_bad < 0 and not isfinite(raw):
            first_bad = t
        var x = ftz(raw)
        sh[t] = x
        acc = ftz(acc + x)
        t += KPSS_FUSED_TPB
    if first_bad >= 0:
        bad_w[0] = Int32(first_bad)
    var s0 = ftz(pinned_block_sum[KPSS_FUSED_TPB](acc))
    if tid == 0:
        mean_w[0] = ftz(s0 * ratio)
    barrier()
    var mean = ftz(mean_w[0])
    # 2. centring in place and s2A
    acc = Float32(0.0)
    t = tid
    while t < n:
        var c = ftz(ftz(sh[t]) - mean)
        sh[t] = c
        acc = ftz(identical_mul_add(c, c, acc))
        t += KPSS_FUSED_TPB
    var s2A = ftz(pinned_block_sum[KPSS_FUSED_TPB](acc))
    barrier()
    # 3. the lagged products and s2B
    acc = Float32(0.0)
    t = tid
    while t < n:
        var x0 = ftz(sh[t])
        var cell = Float32(0.0)
        var k = 1
        while k <= lags and t < n - k:
            var xk = ftz(sh[t + k])
            var dp = ftz(x0 * xk)
            var coeff = ftz(identical_mul_add(coeff_a, Float32(k), coeff_b))
            cell = ftz(identical_mul_add(coeff, dp, cell))
            k += 1
        acc = ftz(acc + cell)
        t += KPSS_FUSED_TPB
    var s2B = ftz(pinned_block_sum[KPSS_FUSED_TPB](acc))
    barrier()
    # 4. the scan in contiguous chunks and eta
    var per = (n + KPSS_FUSED_TPB - 1) // KPSS_FUSED_TPB
    var begin = min(tid * per, n)
    var end = min(begin + per, n)
    var local = Float32(0.0)
    var i = begin
    while i < end:
        local = ftz(local + ftz(sh[i]))
        i += 1
    var running = ftz(pinned_block_prefix_sum[KPSS_FUSED_TPB, exclusive=True](local))
    acc = Float32(0.0)
    i = begin
    while i < end:
        running = ftz(running + ftz(sh[i]))
        acc = ftz(identical_mul_add(running, running, acc))
        i += 1
    var eta = ftz(pinned_block_sum[KPSS_FUSED_TPB](acc))
    # 5. the statistic and the flag
    if tid == 0:
        var stat = kpss_stat_from_sums(s2A, s2B, eta, Float32(n))
        var pvalue = kpss_pvalue(stat)
        stat_out.unsafe_store(b, stat)
        results.unsafe_store(b, UInt8(1) if pvalue > pval_threshold else UInt8(0))
        bad.unsafe_store(b, bad_w[0])


def kpss_fused(
    ctx: DeviceContext,
    mut y: DeviceBuffer[DType.float32],
    flags_ptr: MutPointer[Int32, MutUntrackedOrigin],
    stat_ptr: MutPointer[Float32, MutUntrackedOrigin],
    batch_size: Int,
    n_obs: Int,
    pval_threshold: Float32,
) raises:
    """`kpss_test_host`'s body under the define: the one launch over the
    queued upload `y`, three downloads sharing one wait, the non-finite
    refusal (by name, with the flat index, as `_refuse_non_finite`), and
    the flags and statistics written to the caller's arrays."""
    var results = ctx.enqueue_create_buffer[DType.uint8](batch_size)
    var stat = ctx.enqueue_create_buffer[DType.float32](batch_size)
    var bad = ctx.enqueue_create_buffer[DType.int32](batch_size)
    var lags = kpss_lags(n_obs)
    var coeffs = kpss_s2B_coefficients(n_obs, lags)
    var ratio = Float32(1.0) / Float32(n_obs)
    ctx.enqueue_function[kpss_fused_kernel](
        results.unsafe_ptr(), stat.unsafe_ptr(), bad.unsafe_ptr(), y.unsafe_ptr(),
        Int32(n_obs), Int32(lags), ratio, coeffs[0], coeffs[1], pval_threshold,
        grid_dim=(batch_size, 1, 1), block_dim=(KPSS_FUSED_TPB, 1, 1),
    )
    var hr = ctx.enqueue_create_host_buffer[DType.uint8](batch_size)
    var hs = ctx.enqueue_create_host_buffer[DType.float32](batch_size)
    var hb = ctx.enqueue_create_host_buffer[DType.int32](batch_size)
    ctx.enqueue_copy(dst_ptr=hr.unsafe_ptr(), src_buf=results)
    ctx.enqueue_copy(dst_ptr=hs.unsafe_ptr(), src_buf=stat)
    ctx.enqueue_copy(dst_ptr=hb.unsafe_ptr(), src_buf=bad)
    ctx.synchronize()
    for i in range(batch_size):
        var at = Int(hb.unsafe_ptr().unsafe_load(i))
        if at >= 0:
            raise Error(
                "kpss_test: y contains a non-finite value at index "
                + String(i * n_obs + at) + "; missing or infinite observations are refused by name"
            )
    for i in range(batch_size):
        flags_ptr.unsafe_store(i, Int32(1) if hr.unsafe_ptr().unsafe_load(i) != 0 else Int32(0))
        stat_ptr.unsafe_store(i, hs.unsafe_ptr().unsafe_load(i))
    _ = hr^
    _ = hs^
    _ = hb^
    _ = results^
    _ = stat^
    _ = bad^


def kpss_rounds_kernel(
    out_v: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    n_obs_in: Int32,
    rounds_in: Int32,
    lags0_in: Int32,
    ratio0: Float32,
    ca0: Float32,
    cb0: Float32,
    lags1_in: Int32,
    ratio1: Float32,
    ca1: Float32,
    cb1: Float32,
    pval_threshold: Float32,
):
    """One block per series b. Round r (0 .. rounds - 1) tests the series
    differenced r times (r <= 1: round 1 is `batched_diff_kernel`'s
    `y[t + 1] - y[t]`, read from device memory) with `kpss_fused_kernel`'s
    five stages; the rounds stop at the first stationary one (a block-uniform
    word). Round 0 also scans the raw input for non-finite values (the
    block's smallest index). out_v[b * 4 ..]: round 0's statistic and flag,
    the first bad index (-1 none), the chosen order (rounds when no round
    tested stationary: auto_arima's d_max fallback)."""
    var n0 = Int(n_obs_in)
    var rounds = Int(rounds_in)
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var base = b * n0
    var sh = stack_allocation[KPSS_FUSED_MAX_N, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var word = stack_allocation[2, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var chosen = rounds
    var stat0 = Float32(0.0)
    var flag0 = Float32(0.0)
    var bad = Float32(-1.0)
    for r in range(rounds):
        var n = n0 - r
        var lags = Int(lags0_in) if r == 0 else Int(lags1_in)
        var ratio = ratio0 if r == 0 else ratio1
        var coeff_a = ca0 if r == 0 else ca1
        var coeff_b = cb0 if r == 0 else cb1
        # 1. the series of this round, its sum, and (round 0) the finite scan
        var acc = Float32(0.0)
        var first_bad = Float32(1.0e9)
        var t = tid
        while t < n:
            var x: Float32
            if r == 0:
                var raw = y.unsafe_load(base + t)
                if first_bad > Float32(1.0e8) and not isfinite(raw):
                    first_bad = Float32(t)
                x = ftz(raw)
            else:
                var hi = ftz(y.unsafe_load(base + t + 1))
                var lo = ftz(y.unsafe_load(base + t))
                x = ftz(hi - lo)
            sh[t] = x
            acc = ftz(acc + x)
            t += KPSS_FUSED_TPB
        if r == 0:
            var fb = pinned_block_min[KPSS_FUSED_TPB](first_bad)
            if fb < Float32(1.0e8):
                bad = fb
        var s0 = ftz(pinned_block_sum[KPSS_FUSED_TPB](acc))
        if tid == 0:
            word[0] = ftz(s0 * ratio)
        barrier()
        var mean = ftz(word[0])
        # 2. centring in place and s2A
        acc = Float32(0.0)
        t = tid
        while t < n:
            var c = ftz(ftz(sh[t]) - mean)
            sh[t] = c
            acc = ftz(identical_mul_add(c, c, acc))
            t += KPSS_FUSED_TPB
        var s2A = ftz(pinned_block_sum[KPSS_FUSED_TPB](acc))
        barrier()
        # 3. the lagged products and s2B
        acc = Float32(0.0)
        t = tid
        while t < n:
            var x0 = ftz(sh[t])
            var cell = Float32(0.0)
            var k = 1
            while k <= lags and t < n - k:
                var xk = ftz(sh[t + k])
                var dp = ftz(x0 * xk)
                var coeff = ftz(identical_mul_add(coeff_a, Float32(k), coeff_b))
                cell = ftz(identical_mul_add(coeff, dp, cell))
                k += 1
            acc = ftz(acc + cell)
            t += KPSS_FUSED_TPB
        var s2B = ftz(pinned_block_sum[KPSS_FUSED_TPB](acc))
        barrier()
        # 4. the scan in contiguous chunks and eta
        var per = (n + KPSS_FUSED_TPB - 1) // KPSS_FUSED_TPB
        var begin = min(tid * per, n)
        var end = min(begin + per, n)
        var local = Float32(0.0)
        var i = begin
        while i < end:
            local = ftz(local + ftz(sh[i]))
            i += 1
        var running = ftz(pinned_block_prefix_sum[KPSS_FUSED_TPB, exclusive=True](local))
        acc = Float32(0.0)
        i = begin
        while i < end:
            running = ftz(running + ftz(sh[i]))
            acc = ftz(identical_mul_add(running, running, acc))
            i += 1
        var eta = ftz(pinned_block_sum[KPSS_FUSED_TPB](acc))
        # 5. the statistic and the flag, shared with the block
        if tid == 0:
            var stat = kpss_stat_from_sums(s2A, s2B, eta, Float32(n))
            var pvalue = kpss_pvalue(stat)
            var fl = Float32(1.0) if pvalue > pval_threshold else Float32(0.0)
            if r == 0:
                stat0 = stat
                flag0 = fl
            word[1] = fl
        barrier()
        var stationary = word[1] != Float32(0.0)
        barrier()
        if stationary:
            chosen = r
            break
    if tid == 0:
        out_v.unsafe_store(b * KPSS_PACK_W, stat0)
        out_v.unsafe_store(b * KPSS_PACK_W + 1, flag0)
        out_v.unsafe_store(b * KPSS_PACK_W + 2, bad)
        out_v.unsafe_store(b * KPSS_PACK_W + 3, Float32(chosen))


def kpss_rounds(
    ctx: DeviceContext,
    y_ptr: MutPointer[Float32, MutUntrackedOrigin],
    batch_size: Int,
    n_obs: Int,
    rounds: Int,
    pval_threshold: Float32,
    mut res: List[Float32],
) raises:
    """`kpss_rounds_kernel` over the caller's series (n_obs <=
    KPSS_FUSED_MAX_N, rounds <= 2, n_obs > rounds - 1, checked by the
    caller): one device buffer holds the input then the packed outputs, one
    host stage takes the input then the outputs back; the upload is queued,
    one launch, one download, ONE wait. A non-finite input is refused by
    name with the smallest flat index (`_refuse_non_finite`'s message).
    `res` receives the batch_size * KPSS_PACK_W packed words."""
    var total = batch_size * n_obs
    var words = total + KPSS_PACK_W * batch_size
    var dbuf = ctx.enqueue_create_buffer[DType.float32](words)
    var hbuf = ctx.enqueue_create_host_buffer[DType.float32](words)
    copy_f32(y_ptr, hbuf.unsafe_ptr(), total)
    var din = dbuf.create_sub_buffer[DType.float32](0, total)
    var dout = dbuf.create_sub_buffer[DType.float32](total, KPSS_PACK_W * batch_size)
    ctx.enqueue_copy(dst_buf=din, src_ptr=hbuf.unsafe_ptr())
    var lags0 = kpss_lags(n_obs)
    var c0 = kpss_s2B_coefficients(n_obs, lags0)
    var n1 = n_obs - 1 if n_obs > 1 else 1
    var lags1 = kpss_lags(n1)
    var c1 = kpss_s2B_coefficients(n1, lags1)
    # the two regions as two sub-buffers: two pointers off one buffer alias
    # in the launch's argument check
    ctx.enqueue_function[kpss_rounds_kernel](
        dout.unsafe_ptr(), din.unsafe_ptr(), Int32(n_obs), Int32(rounds),
        Int32(lags0), Float32(1.0) / Float32(n_obs), c0[0], c0[1],
        Int32(lags1), Float32(1.0) / Float32(n1), c1[0], c1[1],
        pval_threshold,
        grid_dim=(batch_size, 1, 1), block_dim=(KPSS_FUSED_TPB, 1, 1),
    )
    ctx.enqueue_copy(dst_ptr=hbuf.unsafe_ptr() + total, src_buf=dout)
    ctx.synchronize()
    var hp = hbuf.unsafe_ptr() + total
    if rounds > 0:
        for i in range(batch_size):
            var at = Int(hp.unsafe_load(i * KPSS_PACK_W + 2))
            if at >= 0:
                raise Error(
                    "kpss_test: y contains a non-finite value at index "
                    + String(i * n_obs + at) + "; missing or infinite observations are refused by name"
                )
    res.clear()
    for i in range(KPSS_PACK_W * batch_size):
        res.append(hp.unsafe_load(i))
    _ = din^
    _ = dout^
    _ = hbuf^
    _ = dbuf^
