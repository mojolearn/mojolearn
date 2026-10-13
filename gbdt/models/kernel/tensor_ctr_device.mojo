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
  * `float_minmax_kernel`: the column's extremes as order-preserving Int32
    keys under `Atomic.min` / `Atomic.max` (exact; `uniform_borders` reads
    nothing else of the column).
  * `tensor_bins_kernel`: the bin is the count of borders the value
    strictly exceeds, `materialize_tensor_candidate`'s comparison.

The host column (`gbdt/host/gbdt_oracle_feature_freq.mojo`) restates the
host loops; same integers, same single-rounding float ops, so it is
unchanged.
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.memory import bitcast

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


def float_minmax_kernel(
    values: MutPointer[Float32, MutAnyOrigin],
    words: MutPointer[Int32, MutAnyOrigin],
    n_rows_in: Int32,
):
    """`words[0]` the smallest, `words[1]` the largest order key; the host
    seeds them with Int32 max / min."""
    var n = Int(n_rows_in)
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while r < n:
        var k = float_order_key(values.unsafe_load(r))
        _ = Atomic.min(words.unsafe_offset(0), k)
        _ = Atomic.max(words.unsafe_offset(1), k)
        r += stride


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
