# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# This lane MIRRORS CatBoost (pinned commit 54a8143a). Per-file provenance is in gbdt/DERIVATION_MAP.tsv, in this file's own docstring, and in NOTICE; files with no CatBoost counterpart say so.
"""`SegmentedReduceVector(..., Sum)`: per-segment sums over ragged offsets.

PORT OF THE CALL, not the library. Their entry is
`catboost/cuda/cuda_util/reduce.h:27` (`SegmentedReduceVector`), whose Sum
arm is `cub::DeviceSegmentedReduce::Sum` over
`[SegmentStarts[s], SegmentStarts[s+1])`. THE VENDOR CHECK WAS RUN
2026-08-21 and recorded here rather than assumed: MAX ships no
device-wide segmented reduce -- `algorithm` is the ROW-WISE reduction
library over regular rows, `graph`/`nn` reductions are Python-level graph
ops -- so per the vendor rule this is hand-written portably, like
`segmented_scan.mojo` before it and for the same reason.

One caller: `TWeightedBinFreqCalcer::VisitEqualUpToPriorFreqCtrs`
(`ctrs/ctr_calcers.h:322`), summing gathered row weights into per-category
weights. Segments are categories: counts range from 1 to the row count,
so the shape is one BLOCK per segment with a grid-stride inner loop and a
shared-memory tree -- no warp intrinsics, no queried wavefront width, no
vendor row. A one-thread segment wastes a block and does not care; a
one-segment input runs the whole sum through one block's loop, which is
the same worst case their cub call has.

FLOAT ORDER: the tree reduction sums in a different order than a host
sequential loop, and cub's own reduction is likewise unordered. On the
trivial weights their GPU learner actually builds (every weight 1.0,
`ctr_helper.h:19-21`) integer-valued float sums below 2^24 are EXACT in
any order, so the device answer is bit-identical to the host reference --
which is what lets `check-freq-ctr-device` demand equality rather than a
tolerance.
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation

comptime SEG_REDUCE_BLOCK = 256


def segmented_reduce_sum_kernel(
    src: MutPointer[Float32, MutAnyOrigin],
    offsets: MutPointer[UInt32, MutAnyOrigin],
    dst: MutPointer[Float32, MutAnyOrigin],
):
    var seg = Int(block_idx.x)
    var begin = Int(offsets.unsafe_load(seg))
    var end = Int(offsets.unsafe_load(seg + 1))
    var acc = Float32(0.0)
    var i = begin + Int(thread_idx.x)
    while i < end:
        acc += src.unsafe_load(i)
        i += Int(block_dim.x)
    var red = stack_allocation[
        SEG_REDUCE_BLOCK,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    red[unsafe_offset = Int(thread_idx.x)] = acc
    barrier()
    var step = SEG_REDUCE_BLOCK // 2
    while step > 0:
        if Int(thread_idx.x) < step:
            red[unsafe_offset = Int(thread_idx.x)] = (
                red[unsafe_offset = Int(thread_idx.x)]
                + red[unsafe_offset = Int(thread_idx.x) + step]
            )
        barrier()
        step //= 2
    if thread_idx.x == 0:
        dst.unsafe_store(seg, red[unsafe_offset=0])


def launch_segmented_reduce_sum(
    ctx: DeviceContext,
    mut src: DeviceBuffer[DType.float32],
    mut offsets: DeviceBuffer[DType.uint32],
    mut dst: DeviceBuffer[DType.float32],
    num_segments: Int,
) raises:
    """`SegmentedReduceVector(input, offsets, output, EOperatorType::Sum)`
    (`cuda_util/reduce.h:27`): `offsets` holds `num_segments + 1` entries,
    segment `s` is `[offsets[s], offsets[s+1])`, `dst` holds
    `num_segments`."""
    if num_segments <= 0:
        return
    ctx.enqueue_function[segmented_reduce_sum_kernel](
        src.unsafe_ptr(),
        offsets.unsafe_ptr(),
        dst.unsafe_ptr(),
        grid_dim=(num_segments, 1, 1),
        block_dim=(SEG_REDUCE_BLOCK, 1, 1),
    )
