# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FAST + Apple experiments for the resampling family, recovered from
lane/apple-fast-resample@50b96e795 onto current main (lane
apple-fast-rec-resample, 2026-10-04). Every entry is compiled under
`RESAMPLE_FAST_APPLE` only and reached from `resample/estimator.mojo` behind a
default-OFF build-time define; IDENTICAL compiles none of it and its bits
never move.

Three kernels, all one launch, all parallel, no host step:

  * (`rank_sort_f32_kernel`, -D MOJOLEARN_RESAMPLE_FAST_RANK_SORT, was
    deleted 2026-10-09: DROPPED-slower; docs/TOMBSTONES.md.)
  * (`bootstrap_mean_fast_kernel`, -D MOJOLEARN_RESAMPLE_FAST_ONE_FOLD, was
    deleted 2026-10-09: DROPPED-slower; docs/TOMBSTONES.md.)
  * `perm_select_fast_kernel` (RESAMPLE_FAST_PERM_SELECT, default; _OFF): the
    permutation null by a 4-bit radix select (16 counters per thread in a
    SIMD register, block totals through threadgroup memory, no atomics, keys
    recomputed from Philox) instead of main's `perm_select_stat_kernel` (8
    byte passes, 256-bucket atomic histogram, the pinned fold). Same
    membership (the n_x smallest keys of the total order `(key, j)`), FAST's
    own fold.

The device row gather of `sklearn.utils.resample` is NOT here: main carries
it as `resample/gather_fast.mojo` (-D MOJOLEARN_RESAMPLE_FAST_GATHER).
"""

from std.gpu import block_idx, thread_idx
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


#: FAST on Apple only. IDENTICAL (and every other vendor) compiles the old
#: code in `resample/estimator.mojo`; nothing below is instantiated there.
#: Each switch is a build-time define on top of this (estimator.mojo,
#: `RESAMPLE_FAST_*`).
comptime RESAMPLE_FAST_APPLE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
)

#: Threads per block of every kernel here.
comptime FAST_TPB = 256

#: The radix select's digit: 4 bits, 16 bins, 16 counters per thread.
#: Threadgroup memory: PERM_BINS * FAST_TPB * 4 B = 16 KB + 64 B, inside
#: Apple's 32 KB per threadgroup.
comptime PERM_DIGIT_BITS = 4
comptime PERM_BINS = 16


# TOMBSTONE: MOJOLEARN_RESAMPLE_FAST_RANK_SORT (DROPPED-slower: bootstrap taxi 14.38 -> 13.64 ms but istella 11.59 -> 14.46)
# deleted 2026-10-09 on lane/owed-deletions-D1 (the bootstrap distribution sorted by one rank launch); code recoverable at
# b639a2bd2. Restore: git apply experiments/removed/MOJOLEARN_RESAMPLE_FAST_RANK_SORT.patch; record in docs/TOMBSTONES.md.


# TOMBSTONE: MOJOLEARN_RESAMPLE_FAST_ONE_FOLD (DROPPED-slower: bootstrap taxi 16.31 -> 13.89 ms but istella 12.84 -> 14.23)
# deleted 2026-10-09 on lane/owed-deletions-D1 (mean / diff_means replicates folded once per block); code recoverable at
# b639a2bd2. Restore: git apply experiments/removed/MOJOLEARN_RESAMPLE_FAST_ONE_FOLD.patch; record in docs/TOMBSTONES.md.


# ===========================================================================
# 3. The permutation null by a 4-bit radix select
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

    ONE BLOCK PER REPLICATE. A radix select over the 64-bit key, four bits a
    pass from the top: every thread counts its positions' digits among the
    keys that still match the selected prefix (16 counters in a SIMD
    register), the block totals them through threadgroup memory (no
    atomics), every thread walks the same 16 totals to the bin holding the
    `want`-th key, and the pass ends the search when every key in that bin
    is selected (`full` is the same in every thread, so every thread
    reaches every barrier). Keys are recomputed from Philox on every pass.
    A full 64-bit tie (about N^2 / 2^65 of replicates) falls to the
    position rule by counting, as the total order says. The fold is FAST's:
    each thread folds its own positions, then `block.sum` per group."""
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
            hist[unsafe_offset=b * tpb + tid] = h[b]
        barrier()
        if tid < PERM_BINS:
            var s = Int32(0)
            for t in range(tpb):
                s += hist[unsafe_offset=tid * tpb + t]
            tot[unsafe_offset=tid] = s
        barrier()
        var below = 0
        var chosen = -1
        var cnt = 0
        for b in range(PERM_BINS):
            var c = Int(tot[unsafe_offset=b])
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
) raises -> Bool:
    """The null `[r_first, r_first + n_replicates)` by
    `perm_select_fast_kernel`, FAST_TPB threads per replicate, drained.
    False (nothing enqueued): another statistic, the caller takes main's
    launch."""
    comptime if RESAMPLE_FAST_APPLE:
        if n_replicates <= 0:
            return True
        if statistic == STAT_DIFF_MEANS:
            comptime kern2 = perm_select_fast_kernel[STAT_DIFF_MEANS, FAST_TPB]
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
                block_dim=(FAST_TPB, 1, 1),
            )
        elif statistic == STAT_MEAN:
            comptime kern1 = perm_select_fast_kernel[STAT_MEAN, FAST_TPB]
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
                block_dim=(FAST_TPB, 1, 1),
            )
        else:
            return False
        _ = null_dist.unsafe_ptr()
        _ = pooled.unsafe_ptr()
        ctx.synchronize()
        return True
    else:
        return False
