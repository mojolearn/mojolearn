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
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
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


struct FeatureProjectionWorkspace(Movable):
    """Fit-owned staging/output capacity; sampled metadata changes each tree.

    The same stream is drained before rewriting staging or output, so preceding
    tree consumers finish before memory is reused. The projection itself stays
    queued; this workspace keeps all of its operands alive until the fit ends.
    """
    var descriptors_host: HostBuffer[DType.uint32]
    var starts_host: HostBuffer[DType.uint32]
    var descriptors_device: DeviceBuffer[DType.uint32]
    var starts_device: DeviceBuffer[DType.uint32]
    var output: DeviceBuffer[DType.uint32]
    var rows: Int
    var feature_capacity: Int
    var column_capacity: Int
    var output_column_capacity: Int

    def __init__(out self, ctx: DeviceContext, n: Int, original: CompressedIndexLayout) raises:
        self.rows = n
        self.feature_capacity = len(original.features)
        self.column_capacity = original.columns
        self.output_column_capacity = 0
        self.descriptors_host = ctx.enqueue_create_host_buffer[DType.uint32](max(1,4*self.feature_capacity))
        self.starts_host = ctx.enqueue_create_host_buffer[DType.uint32](self.column_capacity+1)
        self.descriptors_device = ctx.enqueue_create_buffer[DType.uint32](max(1,4*self.feature_capacity))
        self.starts_device = ctx.enqueue_create_buffer[DType.uint32](self.column_capacity+1)
        # Reserve output only for selected columns actually observed.
        self.output = ctx.enqueue_create_buffer[DType.uint32](1)
        for i in range(max(1,4*self.feature_capacity)):
            self.descriptors_host[i] = UInt32(0)
        for i in range(self.column_capacity+1):
            self.starts_host[i] = UInt32(0)

    def project(
        mut self, ctx: DeviceContext, mut source: DeviceBuffer[DType.uint32],
        original: CompressedIndexLayout, selected: CompressedIndexLayout,
    ) raises -> DeviceBuffer[DType.uint32]:
        if selected.columns > self.column_capacity or len(selected.features) != self.feature_capacity:
            raise Error("feature projection workspace capacity mismatch")
        if selected.columns == 0:
            return source.copy()
        # The previous tree may still have queued work reading this output.
        ctx.synchronize()
        # Capacity is the maximum selected width seen so far. Growth can
        # transiently hold both old and new allocations; no strict peak-memory
        # equivalence to a fresh buffer per tree is claimed.
        if selected.columns > self.output_column_capacity:
            self.output = ctx.enqueue_create_buffer[DType.uint32](
                self.rows * selected.columns
            )
            self.output_column_capacity = selected.columns
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
        for i in range(len(descriptors)):
            self.descriptors_host[i] = descriptors[i]
        for i in range(len(starts)):
            self.starts_host[i] = starts[i]
        ctx.enqueue_copy(dst_buf=self.descriptors_device, src_buf=self.descriptors_host)
        ctx.enqueue_copy(dst_buf=self.starts_device, src_buf=self.starts_host)
        ctx.enqueue_function[project_tree_columns_kernel](
            source.unsafe_ptr(), self.descriptors_device.unsafe_ptr(),
            self.starts_device.unsafe_ptr(), self.output.unsafe_ptr(),
            Int32(self.rows), Int64(self.rows*selected.columns),
            grid_dim=(self.rows*selected.columns+255)//256, block_dim=256,
        )
        return self.output.copy()


def project_tree_columns(
    ctx: DeviceContext, mut source: DeviceBuffer[DType.uint32], n: Int,
    original: CompressedIndexLayout, selected: CompressedIndexLayout,
) raises -> DeviceBuffer[DType.uint32]:
    """Standalone projection wrapper; training retains the workspace instead."""
    var workspace = FeatureProjectionWorkspace(ctx,n,original)
    var result = workspace.project(ctx,source,original,selected)
    ctx.synchronize()
    _ = workspace^
    return result^
