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

from core.pinned_reduce import pinned_block_prefix_sum, pinned_block_sum
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, ftz, identical_mul_add
from tsa.impl.timeSeries.stationarity import kpss_lags, kpss_pvalue, kpss_s2B_coefficients, kpss_stat_from_sums

#: the switch (default OFF; FAST + Apple only)
comptime TSA2_KPSS = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_TSA2_KPSS"]()
)
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
    y: DeviceBuffer[DType.float32],
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
