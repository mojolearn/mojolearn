# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The combination (tensor) FeatureFreq CTR's per-row work ON THE DEVICE
(lane cpu4-gbdt, 2026-10-04).

`gbdt/models/tensor_ctr_value_table.mojo` fitted the tensor table, its learn
column and the column's bins with host loops over every row, inside the
two-level FeatureFreq fit (`estimator.gbdt_fit_two_level_feature_freq` and
the level-two regeneration the structure searcher calls). These kernels do
the same arithmetic one row per thread:

  * `tensor_keys_counts_kernel`: the mixed-radix key over the source codes
    (`key = key * card + code`, sources in canonical order), then one bit
    per canonical split from the quantized columns (`TakeBin`: bin ==,
    `TakeGreater`: bin >), then an INTEGER atomic add of 1 into the dense
    count table. Integer adds commute, so the counts are the host loop's
    exactly, in any order, on every vendor.
  * `tensor_values_kernel`: `(Float32(count) + prior_num) / (Float32(n) +
    prior_denom)`, `tensor_value_for_key`'s FeatureFreq arm, one add and one
    divide, no product to contract: the host's words.
  * `float_minmax_partials_kernel` / `float_minmax_final_kernel`: the
    column's extremes as order-preserving Int32 keys, folded with plain
    register and shared-memory integer compares in two stages, NO atomics
    (exact; `uniform_borders` reads nothing else of the column). See the
    note above `float_minmax_partials_kernel` for why the atomics went.
  * `tensor_bins_kernel`: the bin is the count of borders the value
    strictly exceeds, `materialize_tensor_candidate`'s comparison.

The host column (`gbdt/host/gbdt_oracle_feature_freq.mojo`) restates the
host loops; same integers, same single-rounding float ops, so it is
unchanged.
"""

from std.atomic import Atomic
from std.bit import log2_floor
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.memory import bitcast, stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from gbdt.models.tensor_ctr_apply import TENSOR_SPLIT_TAKE_BIN

comptime TENSOR_CTR_BLOCK = 256


def tensor_ctr_blocks(n: Int) -> Int:
    return max(1, min((n + TENSOR_CTR_BLOCK - 1) // TENSOR_CTR_BLOCK, 65535))


def tensor_keys_counts_kernel(
    codes: MutPointer[UInt32, MutAnyOrigin],
    cards: MutPointer[Int32, MutAnyOrigin],
    n_src_in: Int32,
    bins: MutPointer[UInt32, MutAnyOrigin],
    split_meta: MutPointer[Int32, MutAnyOrigin],
    n_splits_in: Int32,
    keys: MutPointer[Int32, MutAnyOrigin],
    counts: MutPointer[Int32, MutAnyOrigin],
    n_rows_in: Int32,
):
    """`codes[i * n + r]`: source `i`'s validated dense code of row `r`;
    `bins[f * n + r]`: quantized column `f`; `split_meta[3 s ..]`: split
    `s`'s (feature, bin, type). Every index was range-checked on the host
    before the launch."""
    var n = Int(n_rows_in)
    var n_src = Int(n_src_in)
    var n_splits = Int(n_splits_in)
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while r < n:
        var key = 0
        for i in range(n_src):
            var code = Int(codes.unsafe_load(i * n + r))
            key = key * Int(cards.unsafe_load(i)) + code
        for s in range(n_splits):
            var feat = Int(split_meta.unsafe_load(3 * s))
            var bin = Int(split_meta.unsafe_load(3 * s + 1))
            var typ = Int(split_meta.unsafe_load(3 * s + 2))
            var v = Int(bins.unsafe_load(feat * n + r))
            var bit = 0
            if typ == TENSOR_SPLIT_TAKE_BIN:
                if v == bin:
                    bit = 1
            elif v > bin:
                bit = 1
            key = 2 * key + bit
        keys.unsafe_store(r, Int32(key))
        _ = Atomic.fetch_add(counts.unsafe_offset(key), Int32(1))
        r += stride


def tensor_values_kernel(
    keys: MutPointer[Int32, MutAnyOrigin],
    counts: MutPointer[Int32, MutAnyOrigin],
    n_counts_in: Int32,
    prior_num: Float32,
    prior_denom: Float32,
    denominator: Float32,
    values: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
):
    var n = Int(n_rows_in)
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while r < n:
        var key = Int(keys.unsafe_load(r))
        var count = Int32(0)
        if key >= 0 and key < Int(n_counts_in):
            count = counts.unsafe_load(key)
        values.unsafe_store(
            r,
            (count.cast[DType.float32]() + prior_num)
            / (denominator + prior_denom),
        )
        r += stride


@always_inline
def float_order_key(v: Float32) -> Int32:
    """An Int32 whose signed order is the float order (finite values; -0.0
    sorts just below +0.0, which compare equal and give the same grid).
    The map is its own inverse."""
    var i = bitcast[DType.int32](v)
    return i ^ ((i >> 31) & Int32(0x7FFFFFFF))


@always_inline
def float_from_order_key(k: Int32) -> Float32:
    return bitcast[DType.float32](k ^ ((k >> 31) & Int32(0x7FFFFFFF)))


# lane apple-feature-freq (2026-10-13): the extremes were `Atomic.min` /
# `Atomic.max` on SIGNED Int32 order keys, and a negative float's key is a
# negative Int32. Every numeric column of the identity fixtures is standard
# normal, so every fixture fed negative keys to those atomics. NVIDIA, AMD
# and the host column agreed on every gbdt-feature-freq and
# gbdt-tensor-ctr-tables part; the Apple column diverged on all 72 parts
# (9 fixtures x 4 parts x 2 lanes) from the commit that moved these extremes
# onto the device (35b3c9e02, lane cpu4-gbdt) and on no other GBDT lane,
# and this is the only signed atomic min/max over possibly negative words
# in the two-level fit (the category scans fold non-negative words and row
# ids). The fold below needs no atomic min/max at all, so its answer does not
# depend on how a backend lowers a signed atomic: integer compares in
# registers, then a shared-memory halving tree, then one block over the
# fixed number of partials. A min or max over a total order is exact in any
# order, so NVIDIA, AMD and the host keep their words; Apple gets theirs.

comptime TENSOR_MINMAX_GRID = 256
"""Stage-1 blocks of the extremes fold: a fixed geometry (a min or max is
exact in any order, so it moves no bits; it only bounds the partials to one
stage-2 block)."""

comptime _KEY_MAX = Int32(2147483647)
comptime _KEY_MIN = Int32(-2147483647) - Int32(1)


@always_inline
def _block_key_minmax[block_size: Int](lo_in: Int32, hi_in: Int32) -> Tuple[Int32, Int32]:
    """The block's smallest `lo_in` and largest `hi_in`: a halving tree of
    integer compares in shared memory. Every thread of the block must call
    it (it holds barriers)."""
    comptime assert block_size > 0 and (block_size & (block_size - 1)) == 0, (
        "the halving tree needs a power-of-two block"
    )
    var t = Int(thread_idx.x)
    var s_lo = stack_allocation[
        block_size,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var s_hi = stack_allocation[
        block_size,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    s_lo[t] = lo_in
    s_hi[t] = hi_in
    barrier()
    comptime for k in range(log2_floor(block_size)):
        comptime step = block_size >> (k + 1)
        if t < step:
            var a = s_lo[t + step]
            if a < s_lo[t]:
                s_lo[t] = a
            var c = s_hi[t + step]
            if c > s_hi[t]:
                s_hi[t] = c
        barrier()
    var lo = s_lo[0]
    var hi = s_hi[0]
    barrier()
    return (lo, hi)


def float_minmax_partials_kernel(
    values: MutPointer[Float32, MutAnyOrigin],
    partials: MutPointer[Int32, MutAnyOrigin],
    n_rows_in: Int32,
):
    """Stage 1. Grid `(TENSOR_MINMAX_GRID, 1, 1)`, block
    `(TENSOR_CTR_BLOCK, 1, 1)`. Block `b` writes `partials[2 b]`, the
    smallest order key of its strided rows, and `partials[2 b + 1]`, the
    largest (Int32 max / min for a block with no rows)."""
    var n = Int(n_rows_in)
    var t = Int(thread_idx.x)
    var b = Int(block_idx.x)
    var lo = _KEY_MAX
    var hi = _KEY_MIN
    var r = b * TENSOR_CTR_BLOCK + t
    while r < n:
        var k = float_order_key(values.unsafe_load(r))
        if k < lo:
            lo = k
        if k > hi:
            hi = k
        r += TENSOR_CTR_BLOCK * TENSOR_MINMAX_GRID
    var folded = _block_key_minmax[TENSOR_CTR_BLOCK](lo, hi)
    if t == 0:
        partials.unsafe_store(2 * b, folded[0])
        partials.unsafe_store(2 * b + 1, folded[1])


def float_minmax_final_kernel(
    partials: MutPointer[Int32, MutAnyOrigin],
    n_partials_in: Int32,
    words: MutPointer[Int32, MutAnyOrigin],
):
    """Stage 2. Grid `(1, 1, 1)`, block `(TENSOR_CTR_BLOCK, 1, 1)`: the
    `n_partials_in` stage-1 pairs into `words[0]` (smallest order key) and
    `words[1]` (largest)."""
    var t = Int(thread_idx.x)
    var m = Int(n_partials_in)
    var lo = _KEY_MAX
    var hi = _KEY_MIN
    var i = t
    while i < m:
        var a = partials.unsafe_load(2 * i)
        if a < lo:
            lo = a
        var c = partials.unsafe_load(2 * i + 1)
        if c > hi:
            hi = c
        i += TENSOR_CTR_BLOCK
    var folded = _block_key_minmax[TENSOR_CTR_BLOCK](lo, hi)
    if t == 0:
        words.unsafe_store(0, folded[0])
        words.unsafe_store(1, folded[1])


def tensor_bins_kernel(
    values: MutPointer[Float32, MutAnyOrigin],
    borders: MutPointer[Float32, MutAnyOrigin],
    n_borders_in: Int32,
    bins: MutPointer[UInt32, MutAnyOrigin],
    n_rows_in: Int32,
):
    var n = Int(n_rows_in)
    var nb = Int(n_borders_in)
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while r < n:
        var v = values.unsafe_load(r)
        var b = 0
        for j in range(nb):
            if v > borders.unsafe_load(j):
                b += 1
        bins.unsafe_store(r, UInt32(b))
        r += stride


def widen_i32_kernel(
    src: MutPointer[Int32, MutAnyOrigin],
    dst: MutPointer[Int64, MutAnyOrigin],
    n_in: Int32,
):
    """The Int32 count table as the Int64 words of the host table's
    `List[Int]`, so it downloads with one copy (Metal has no 64-bit
    atomics, so the counts are accumulated in Int32: at most `n_rows`)."""
    var n = Int(n_in)
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while r < n:
        dst.unsafe_store(r, src.unsafe_load(r).cast[DType.int64]())
        r += stride
