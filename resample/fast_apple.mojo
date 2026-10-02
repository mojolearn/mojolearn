# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FAST + Apple experiments for the resampling lane (lane/apple-fast-resample,
2026-10-02). Every entry here is compiled under `RESAMPLE_FAST_APPLE` only and
reached from `resample/estimator.mojo` behind an env switch read on the host;
IDENTICAL compiles none of it and its bits never move.

Four kernels, all one launch, all parallel, no host step:

  * `rank_sort_f32_kernel`: the sorted bootstrap distribution by RANK (one
    thread per replicate, the keys staged 256 at a time in threadgroup
    memory) in the same total order `(twiddle_in(theta_r), r)` the 32-pass
    LSD radix of `core/segmented_sort.mojo` produces, so the same bits at
    every rank. Switch MOJOLEARN_RESAMPLE_FAST_RANK_SORT=1.
  * `bootstrap_mean_fast_kernel`: mean / diff_means replicates with ONE
    block fold per replicate (each thread folds its own draws in registers,
    then `block.sum`) instead of one `virtual_block_sum` per 256-draw chunk
    (79 block folds for n = 20,000). Same draws (`draw_row_index` at the
    same positions); FAST's summation order is not pinned. Switch
    MOJOLEARN_RESAMPLE_FAST_ONE_FOLD=1.
  * `perm_select_fast_kernel`: the permutation null above PERM_MAX_POOLED.
    `perm_stat_kernel` ranks every pooled position by counting (O(N^2) per
    replicate, 8 N bytes of threadgroup memory) and the host refuses N >
    1024; the board's permutation-test row (20,000 + 20,000) is REFUSED on
    every box. The membership `rank < n_x` is the set of the n_x smallest
    keys in the total order `(key, position)`, which a radix SELECT over
    the 64-bit key finds in at most 16 four-bit passes (one block, 16
    counters per thread, keys recomputed from Philox, no atomics; with
    random keys the pass that isolates the pivot ends it, 4-5 passes for N
    = 40,000). The same permutation, no bound. Switch
    MOJOLEARN_RESAMPLE_FAST_PERM_SELECT=1.
  * `gather_rows_f32_kernel`: `sklearn.utils.resample`'s row gather on the
    device (one thread per output cell) from the device-drawn indices,
    instead of a per-element index download, a per-element store and a
    numpy fancy-index gather on the host. Switch
    MOJOLEARN_RESAMPLE_FAST_GATHER=1.

Written without a Mojo toolchain; the first M3 build is the compile check.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.math import ceildiv
from std.memory import bitcast, stack_allocation
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.primitives.block import sum as block_sum
from max.gpu.sync import barrier

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, ftz
from core.segmented_sort import float_to_sortable
from metrics.checks.pinned_sum import canonicalize_nan
from resample.checks.index_map import (
    draw_permutation_key,
    draw_row_index,
    key_hi,
    key_join,
    key_lo,
)
from resample.checks.statistics import STAT_DIFF_MEANS, STAT_MEAN, _mean_of_sum


#: FAST on Apple only. IDENTICAL and DETERMINISTIC (and every other vendor)
#: compile the old code in `resample/estimator.mojo`; nothing below is
#: instantiated there. Each switch is a build-time define on top of this
#: (estimator.mojo, `RESAMPLE_FAST_*`).
comptime RESAMPLE_FAST_APPLE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
)

#: Threads per block of the map-shaped kernels here (rank sort, gather).
comptime FAST_MAP_TPB = 256

#: The radix select's digit: 4 bits, 16 bins, 16 counters per thread.
comptime PERM_DIGIT_BITS = 4
comptime PERM_BINS = 16


# ===========================================================================
# 1. The sorted distribution by rank (one launch)
# ===========================================================================


def rank_sort_f32_kernel(
    src: MutPointer[Float32, MutAnyOrigin],
    dst: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`dst[rank(i)] = src[i]` with `rank(i)` the number of positions below
    `i` in the total order `(float_to_sortable(bits), position)`: the order
    `core/segmented_sort.mojo` pins (`-0.0` below `+0.0`, equal bits in
    ascending position order), so every rank carries the same bits the
    radix sort would put there. One thread per position; the keys are
    staged `FAST_MAP_TPB` at a time in threadgroup memory. Every thread of
    the block reaches both barriers on every chunk."""
    var n = Int(n_in)
    var tid = Int(thread_idx.x)
    var i = Int(block_idx.x) * FAST_MAP_TPB + tid
    var slab = stack_allocation[
        FAST_MAP_TPB,
        Scalar[DType.uint32],
        address_space = AddressSpace.SHARED,
    ]()
    var mine = UInt32(0)
    var value = Float32(0.0)
    if i < n:
        value = src.unsafe_load(i)
        mine = float_to_sortable(bitcast[DType.uint32](value))
    var rank = 0
    var base = 0
    while base < n:
        var j = base + tid
        barrier()
        if j < n:
            slab[unsafe_offset = tid] = float_to_sortable(
                bitcast[DType.uint32](src.unsafe_load(j))
            )
        barrier()
        var limit = n - base
        if limit > FAST_MAP_TPB:
            limit = FAST_MAP_TPB
        if i < n:
            for t in range(limit):
                var kj = slab[unsafe_offset = t]
                if kj < mine or (kj == mine and base + t < i):
                    rank += 1
        base += FAST_MAP_TPB
    if i < n:
        dst.unsafe_store(rank, value)


def rank_sort_f32(
    ctx: DeviceContext,
    mut src: DeviceBuffer[DType.float32],
    mut dst: DeviceBuffer[DType.float32],
    n: Int,
) raises:
    """`dst` = `src` sorted by `rank_sort_f32_kernel`, one launch, drained."""
    comptime if RESAMPLE_FAST_APPLE:
        if n <= 0:
            return
        ctx.enqueue_function[rank_sort_f32_kernel](
            src.unsafe_ptr(),
            dst.unsafe_ptr(),
            Int32(n),
            grid_dim=(ceildiv(n, FAST_MAP_TPB), 1, 1),
            block_dim=(FAST_MAP_TPB, 1, 1),
        )
        _ = src.unsafe_ptr()
        _ = dst.unsafe_ptr()
        ctx.synchronize()
    else:
        raise Error("resample fast_apple: rank_sort_f32 is FAST + Apple only")


# ===========================================================================
# 2. mean / diff_means replicates with one block fold each
# ===========================================================================


def bootstrap_mean_fast_kernel[two: Bool, tpb: Int](
    theta: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    lo_bits: Int32,
    hi_bits: Int32,
    r_first_in: Int32,
    n_replicates_in: Int32,
    n_in: Int32,
    n_rows_in: Int32,
    n_features_in: Int32,
):
    """`theta[r] = mean(resample(x, r))` (`two`: minus the mean of column
    1, the paired diff_means). One block per replicate; thread `t` draws
    positions `t, t + tpb, ...` with `draw_row_index` (the same draws as
    `bootstrap_stat_kernel`) and folds them in a register, then ONE
    `block.sum` per column. Every thread reaches the block primitive."""
    var rr = Int(block_idx.x)
    if rr >= Int(n_replicates_in):
        return
    var r = Int(r_first_in) + rr
    var tid = Int(thread_idx.x)
    var key = key_join(lo_bits, hi_bits)
    var n = Int(n_in)
    var d = Int(n_features_in)
    var acc0 = Float32(0.0)
    var acc1 = Float32(0.0)
    var i = tid
    while i < n:
        var row = Int(draw_row_index(key, r, i, n_rows_in))
        acc0 += ftz(x.unsafe_load(row * d))
        comptime if two:
            acc1 += ftz(x.unsafe_load(row * d + 1))
        i += tpb
    var s0 = block_sum[block_size=tpb](acc0)
    var s1 = Float32(0.0)
    comptime if two:
        s1 = block_sum[block_size=tpb](acc1)
    if tid == 0:
        var value = _mean_of_sum(s0, n)
        comptime if two:
            value = ftz(value - _mean_of_sum(s1, n))
        theta.unsafe_store(rr, canonicalize_nan(value))


def bootstrap_mean_fast(
    ctx: DeviceContext,
    mut theta: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    key: UInt64,
    r_first: Int,
    n_replicates: Int,
    n: Int,
    n_features: Int,
    statistic: Int,
) raises:
    """Replicates `[r_first, r_first + n_replicates)` of mean or diff_means
    into `theta`, 256 threads per replicate, drained."""
    comptime if RESAMPLE_FAST_APPLE:
        if statistic == STAT_DIFF_MEANS:
            comptime kern2 = bootstrap_mean_fast_kernel[True, 256]
            ctx.enqueue_function[kern2](
                theta.unsafe_ptr(),
                x.unsafe_ptr(),
                key_lo(key),
                key_hi(key),
                Int32(r_first),
                Int32(n_replicates),
                Int32(n),
                Int32(n),
                Int32(n_features),
                grid_dim=(n_replicates, 1, 1),
                block_dim=(256, 1, 1),
            )
        elif statistic == STAT_MEAN:
            comptime kern1 = bootstrap_mean_fast_kernel[False, 256]
            ctx.enqueue_function[kern1](
                theta.unsafe_ptr(),
                x.unsafe_ptr(),
                key_lo(key),
                key_hi(key),
                Int32(r_first),
                Int32(n_replicates),
                Int32(n),
                Int32(n),
                Int32(n_features),
                grid_dim=(n_replicates, 1, 1),
                block_dim=(256, 1, 1),
            )
        else:
            raise Error(
                "resample fast_apple: bootstrap_mean_fast serves mean and"
                " diff_means only"
            )
        _ = theta.unsafe_ptr()
        _ = x.unsafe_ptr()
        ctx.synchronize()
    else:
        raise Error("resample fast_apple: bootstrap_mean_fast is FAST + Apple only")


# ===========================================================================
# 3. The permutation null by radix select (no PERM_MAX_POOLED)
# ===========================================================================


def perm_select_fast_kernel[stat: Int, tpb: Int](
    null_dist: MutPointer[Float32, MutAnyOrigin],
    pooled: MutPointer[Float32, MutAnyOrigin],
    lo_bits: Int32,
    hi_bits: Int32,
    r_first_in: Int32,
    n_replicates_in: Int32,
    n_pooled_in: Int32,
    n_x_in: Int32,
):
    """`null_dist[r] = statistic(x_r, y_r)`, the pooled sample split by
    replicate `r`'s permutation: position `j` is in `x_r` exactly when its
    rank in the total order `(draw_permutation_key(key, r, j), j)` is below
    `n_x` (`perm_stat_kernel`'s rule), i.e. when it is one of the `n_x`
    smallest keys.

    ONE BLOCK PER REPLICATE. A radix select over the 64-bit key, four bits
    a pass from the top: every thread counts its positions' digits among
    the keys that still match the selected prefix (16 counters in a SIMD
    register), the block totals them through threadgroup memory (no
    atomics), every thread walks the same 16 totals to the bin holding the
    `want`-th key, and the pass ends the search when every key in that bin
    is selected. Keys are recomputed from Philox on every pass (pure
    functions of `(key, r, j)`; no scratch). A full 64-bit tie (two equal
    keys, about N^2 / 2^65 of replicates) falls to the position rule by
    counting, as the total order says. The fold is FAST's: each thread
    folds its own positions, then `block.sum` per group."""
    comptime assert (
        stat == STAT_MEAN or stat == STAT_DIFF_MEANS
    ), "perm_select_fast_kernel: mean and diff_means are the arms"
    var rr = Int(block_idx.x)
    if rr >= Int(n_replicates_in):
        return
    var r = Int(r_first_in) + rr
    var tid = Int(thread_idx.x)
    var key = key_join(lo_bits, hi_bits)
    var n_pooled = Int(n_pooled_in)
    var n_x = Int(n_x_in)
    var n_y = n_pooled - n_x

    var hist = stack_allocation[
        PERM_BINS * tpb,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var tot = stack_allocation[
        PERM_BINS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()

    var prefix = UInt64(0)
    var mask = UInt64(0)
    var want = n_x
    var full = False
    var shift = 64 - PERM_DIGIT_BITS
    while shift >= 0 and not full:
        var h = SIMD[DType.int32, PERM_BINS](0)
        var j = tid
        while j < n_pooled:
            var kj = draw_permutation_key(key, r, j)
            if (kj & mask) == prefix:
                var dgt = Int((kj >> UInt64(shift)) & UInt64(PERM_BINS - 1))
                h[dgt] = h[dgt] + 1
            j += tpb
        comptime for b in range(PERM_BINS):
            hist[unsafe_offset = b * tpb + tid] = h[b]
        barrier()
        if tid < PERM_BINS:
            var s = Int32(0)
            for t in range(tpb):
                s += hist[unsafe_offset = tid * tpb + t]
            tot[unsafe_offset = tid] = s
        barrier()
        var below = 0
        var chosen = -1
        var cnt = 0
        for b in range(PERM_BINS):
            var c = Int(tot[unsafe_offset = b])
            if chosen < 0:
                if below + c >= want:
                    chosen = b
                    cnt = c
                else:
                    below += c
        barrier()
        if chosen < 0:
            # Unreachable: the keys matching the prefix number at least
            # `want`. Kept so the loop cannot shift a negative digit in.
            chosen = PERM_BINS - 1
        want -= below
        prefix = prefix | (UInt64(chosen) << UInt64(shift))
        mask = mask | (UInt64(PERM_BINS - 1) << UInt64(shift))
        if cnt == want:
            full = True
        shift -= PERM_DIGIT_BITS

    var sx = Float32(0.0)
    var sy = Float32(0.0)
    var j2 = tid
    while j2 < n_pooled:
        var kj2 = draw_permutation_key(key, r, j2)
        var p = kj2 & mask
        var inx = False
        if p < prefix:
            inx = True
        elif p == prefix:
            if full:
                inx = True
            else:
                # Equal 64-bit keys: the earlier positions rank first.
                var eq = 0
                for l in range(j2):
                    if draw_permutation_key(key, r, l) == kj2:
                        eq += 1
                inx = eq < want
        var v = ftz(pooled.unsafe_load(j2))
        if inx:
            sx += v
        else:
            sy += v
        j2 += tpb
    var tx = block_sum[block_size=tpb](sx)
    var ty = block_sum[block_size=tpb](sy)
    if tid == 0:
        var value = _mean_of_sum(tx, n_x)
        comptime if stat == STAT_DIFF_MEANS:
            value = ftz(value - _mean_of_sum(ty, n_y))
        null_dist.unsafe_store(rr, canonicalize_nan(value))


def perm_select_fast(
    ctx: DeviceContext,
    mut null_dist: DeviceBuffer[DType.float32],
    mut pooled: DeviceBuffer[DType.float32],
    key: UInt64,
    r_first: Int,
    n_replicates: Int,
    n_pooled: Int,
    n_x: Int,
    statistic: Int,
) raises:
    """The null `[r_first, r_first + n_replicates)` by `perm_select_fast_kernel`,
    256 threads per replicate, drained."""
    comptime if RESAMPLE_FAST_APPLE:
        if statistic == STAT_DIFF_MEANS:
            comptime kern2 = perm_select_fast_kernel[STAT_DIFF_MEANS, 256]
            ctx.enqueue_function[kern2](
                null_dist.unsafe_ptr(),
                pooled.unsafe_ptr(),
                key_lo(key),
                key_hi(key),
                Int32(r_first),
                Int32(n_replicates),
                Int32(n_pooled),
                Int32(n_x),
                grid_dim=(n_replicates, 1, 1),
                block_dim=(256, 1, 1),
            )
        elif statistic == STAT_MEAN:
            comptime kern1 = perm_select_fast_kernel[STAT_MEAN, 256]
            ctx.enqueue_function[kern1](
                null_dist.unsafe_ptr(),
                pooled.unsafe_ptr(),
                key_lo(key),
                key_hi(key),
                Int32(r_first),
                Int32(n_replicates),
                Int32(n_pooled),
                Int32(n_x),
                grid_dim=(n_replicates, 1, 1),
                block_dim=(256, 1, 1),
            )
        else:
            raise Error(
                "resample fast_apple: perm_select_fast serves mean and diff_means"
                " only"
            )
        _ = null_dist.unsafe_ptr()
        _ = pooled.unsafe_ptr()
        ctx.synchronize()
    else:
        raise Error("resample fast_apple: perm_select_fast is FAST + Apple only")


# ===========================================================================
# 4. sklearn.utils.resample's row gather on the device
# ===========================================================================


def gather_rows_f32_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin],
    count_in: Int32,
    d_in: Int32,
):
    """`dst[o] = src[rows[o // d] * d + o % d]`, one thread per output cell
    (consecutive threads read consecutive cells of one source row)."""
    var o = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var d = Int(d_in)
    if o >= Int(count_in) * d:
        return
    var rr = o // d
    var c = o - rr * d
    dst.unsafe_store(o, src.unsafe_load(Int(rows.unsafe_load(rr)) * d + c))


def gather_rows_f32(
    ctx: DeviceContext,
    mut dst: DeviceBuffer[DType.float32],
    mut src: DeviceBuffer[DType.float32],
    mut rows: DeviceBuffer[DType.int32],
    count: Int,
    d: Int,
) raises:
    """`count` rows of width `d` from `src` at `rows`, into `dst`; enqueued,
    not drained (the caller copies out and drains)."""
    comptime if RESAMPLE_FAST_APPLE:
        var total = count * d
        if total <= 0:
            return
        ctx.enqueue_function[gather_rows_f32_kernel](
            dst.unsafe_ptr(),
            src.unsafe_ptr(),
            rows.unsafe_ptr(),
            Int32(count),
            Int32(d),
            grid_dim=(ceildiv(total, FAST_MAP_TPB), 1, 1),
            block_dim=(FAST_MAP_TPB, 1, 1),
        )
        _ = dst.unsafe_ptr()
        _ = src.unsafe_ptr()
        _ = rows.unsafe_ptr()
    else:
        raise Error("resample fast_apple: gather_rows_f32 is FAST + Apple only")
