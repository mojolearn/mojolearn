# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Numeric per-tree feature sampling and GPU compressed-index projection.

Reference LightGBM 3d1cf301 src/treelearner/col_sampler.hpp:34-54, 77-94;
CUDA consumer cuda_single_gpu_tree_learner.cpp:153-154. Eligible features have
positive fold counts; count=max(1, round(eligible*fraction)), original ID order.
FEATURE-SAMPLE-1: reuse this repository's portable TRandom with partial Fisher
Yates, rather than LightGBM Random::Sample's LCG/density-dependent branches.
The independent stream is seeded once per fit with random_seed XOR
0x4645415455524553. Same-seed cross-library models are not promised. No draws
when all eligible features are selected. Each of K uniform(remaining) requests
uses the existing rejection sampler; raw draw counts may vary.
FEATURE-SAMPLE-2: selected numeric bins are GPU-repacked once per tree, so the
existing packed histogram kernels omit excluded features. Metadata retains all
original IDs with excluded fold counts zero. Projection adds overhead; no net
speed claim. Default fraction 1 bypasses this module's per-tree work entirely.
"""
from std.math import isfinite
from std.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from gbdt.data.permutation import TRandom
from gbdt.gpu_data.compressed_index_builder import CompressedIndexLayout, build_layout


def check_feature_fraction(fraction: Float64) raises:
    if not isfinite(fraction) or fraction <= 0 or fraction > 1:
        raise Error("feature_fraction must be finite and in (0, 1]")


def sample_tree_folds(
    folds: List[Int], fraction: Float64, mut random: TRandom,
) raises -> List[Int]:
    check_feature_fraction(fraction)
    var eligible = List[Int]()
    for f in range(len(folds)):
        if folds[f] > 0:
            eligible.append(f)
    var count = max(1, Int(Float64(len(eligible))*fraction+0.5))
    if count >= len(eligible):
        return folds.copy()
    # Partial Fisher-Yates freezes K bounded uniform requests per sampled tree. Restore
    # original ID order by writing a full-length fold vector, never compact IDs.
    var result = List[Int]()
    for _ in range(len(folds)):
        result.append(0)
    for i in range(count):
        var j = i+Int(random.uniform(UInt64(len(eligible)-i)))
        var selected = eligible[j]
        eligible[j] = eligible[i]
        eligible[i] = selected
        result[selected] = folds[selected]
    return result^


def project_tree_columns_kernel(
    source: MutPointer[UInt32, MutAnyOrigin], descriptors: MutPointer[UInt32, MutAnyOrigin],
    starts: MutPointer[UInt32, MutAnyOrigin], output: MutPointer[UInt32, MutAnyOrigin],
    n_in: Int32, count_in: Int64,
):
    var i = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if i < Int(count_in):
        var n = Int(n_in)
        var column = i//n
        var row = i%n
        var packed = UInt32(0)
        for slot in range(Int(starts.unsafe_load(column)), Int(starts.unsafe_load(column+1))):
            var source_column = Int(descriptors.unsafe_load(4*slot))
            var source_mask = descriptors.unsafe_load(4*slot+1)
            var source_shift = descriptors.unsafe_load(4*slot+2)
            var destination_shift = descriptors.unsafe_load(4*slot+3)
            var value = (source.unsafe_load(source_column*n+row)>>source_shift)&source_mask
            packed |= value<<destination_shift
        output.unsafe_store(i, packed)


def project_tree_columns(
    ctx: DeviceContext, mut source: DeviceBuffer[DType.uint32], n: Int,
    original: CompressedIndexLayout, selected: CompressedIndexLayout,
) raises -> DeviceBuffer[DType.uint32]:
    if selected.columns == 0:
        return source.copy()
    var by_column = List[List[UInt32]]()
    for _ in range(selected.columns):
        by_column.append(List[UInt32]())
    for f in range(len(selected.features)):
        ref dst = selected.features[f]
        if Int(dst.folds) == 0:
            continue
        ref src = original.features[f]
        var column = Int(dst.offset)
        by_column[column].append(src.offset)
        by_column[column].append(src.mask)
        by_column[column].append(src.shift)
        by_column[column].append(dst.shift)
    var descriptors = List[UInt32]()
    var starts = List[UInt32]()
    for column in range(selected.columns):
        starts.append(UInt32(len(descriptors)//4))
        for i in range(len(by_column[column])):
            descriptors.append(by_column[column][i])
    starts.append(UInt32(len(descriptors)//4))
    var hd = ctx.enqueue_create_host_buffer[DType.uint32](len(descriptors))
    var hs = ctx.enqueue_create_host_buffer[DType.uint32](len(starts))
    for i in range(len(descriptors)):
        hd[i] = descriptors[i]
    for i in range(len(starts)):
        hs[i] = starts[i]
    var dd = ctx.enqueue_create_buffer[DType.uint32](len(descriptors))
    var ds = ctx.enqueue_create_buffer[DType.uint32](len(starts))
    ctx.enqueue_copy(dst_buf=dd, src_buf=hd)
    ctx.enqueue_copy(dst_buf=ds, src_buf=hs)
    var output = ctx.enqueue_create_buffer[DType.uint32](n*selected.columns)
    ctx.enqueue_function[project_tree_columns_kernel](
        source.unsafe_ptr(), dd.unsafe_ptr(), ds.unsafe_ptr(), output.unsafe_ptr(),
        Int32(n), Int64(n*selected.columns),
        grid_dim=(n*selected.columns+255)//256, block_dim=256,
    )
    # Metadata staging and scratch must survive the queued projection.
    ctx.synchronize()
    _ = ds^
    _ = dd^
    _ = hs^
    _ = hd^
    return output^
