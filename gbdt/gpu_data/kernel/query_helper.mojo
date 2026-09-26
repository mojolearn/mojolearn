# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Reference: `catboost/cuda/gpu_data/kernel/query_helper.cu`, the two kernels
the querywise targets reach: `ComputeGroupIdsImpl` and its launcher
(`:11-41`) and the offsets-and-sizes overload of `ComputeGroupMeansImpl` with
its launcher (`:132-189`); and `InversePermutationImpl` with its launcher
from `catboost/cuda/cuda_util/kernel/fill.cu:58-77`, which the querywise
permutation der calcer uses (`targets/permutation_der_calcer.h:176-183`).

THE LAYOUT BOTH GROUP KERNELS SHARE. A block of 128 threads carries FOUR
queries, 32 lanes each: thread `t` of block `b` works on query
`b * 4 + t / 32` and walks that query's rows `t & 31, (t & 31) + 32, ...`.
The grid is `ceil(qCount * 32 / 128)` blocks, so the thread count is fixed by
the QUERY count, never by a query's size.

THE MEAN'S FOLD SHAPE IS THEIRS AND IT IS PINNED. Each lane sums its strided
rows in float32 in row order (`sumTarget += t * w; sumWeight += w`), and the
32 lane sums of a query meet in `WarpReduce` (`cuda_util/kernel/
kernel_helpers.cuh:92-106`): `val = val + shuffle_down(val, s)` for
`s = 16, 8, 4, 2, 1`. Lane 0's result reads only lanes that step `s` did not
write (`x + s >= s`), so the tree below, `line[x] += line[x + s]` for
`x < s` with a barrier after each step, is that reduce in the same operands
and order on every vendor. The quotient is theirs:
`totalWeight != 0 ? totalSum / totalWeight : 0`.

DEVIATION (flush at derivation, IDENTITY_PATHS row 10's policy): the stored
mean passes through `ftz`, a comptime no-op outside IDENTICAL, because it is
a kernel-to-kernel seam the QueryRMSE kernel reads.
"""

from std.math import fma
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import ftz

#: `const ui64 blockSize = 128` (`query_helper.cu:36`, `:181`)
comptime QUERY_HELPER_BLOCK_SIZE = 128
#: the `/ 32` and `& 31` of both group kernels
comptime QUERY_LANES = 32
comptime QUERIES_PER_BLOCK = QUERY_HELPER_BLOCK_SIZE // QUERY_LANES
#: `const ui32 blockSize = 512` (`fill.cu:71`)
comptime INVERSE_PERMUTATION_BLOCK_SIZE = 512
#: `TArchProps::MaxBlockCount()`: the kernels grid-stride, so any large cap serves
comptime QUERY_HELPER_MAX_BLOCKS = 65535


def query_helper_blocks(q_count: Int) -> Int:
    """`CeilDivide(qCount * 32, blockSize)` (`:37`, `:182`)."""
    return (q_count * QUERY_LANES + QUERY_HELPER_BLOCK_SIZE - 1) // QUERY_HELPER_BLOCK_SIZE


def compute_group_ids_kernel(
    q_sizes: MutPointer[UInt32, MutAnyOrigin],
    q_offsets: MutPointer[UInt32, MutAnyOrigin],
    offsets_bias: UInt32,
    q_count_in: Int32,
    dst: MutPointer[UInt32, MutAnyOrigin],
):
    """`ComputeGroupIdsImpl` (`query_helper.cu:11-26`): every row of query
    `qid` is written `qid`, lane by lane."""
    var q_count = Int(q_count_in)
    var local_qid = Int(thread_idx.x) // QUERY_LANES
    var qid = Int(block_idx.x) * QUERIES_PER_BLOCK + local_qid
    var write_offset = 0
    var query_size = 0
    if qid < q_count:
        write_offset = Int(q_offsets.unsafe_load(qid) - offsets_bias)
        query_size = Int(q_sizes.unsafe_load(qid))
    var i = Int(thread_idx.x) & (QUERY_LANES - 1)
    while i < query_size:
        dst.unsafe_store(write_offset + i, UInt32(qid))
        i += QUERY_LANES


def launch_compute_group_ids(
    ctx: DeviceContext,
    mut q_sizes: DeviceBuffer[DType.uint32],
    mut q_offsets: DeviceBuffer[DType.uint32],
    offsets_bias: UInt32,
    q_count: Int,
    mut dst: DeviceBuffer[DType.uint32],
) raises:
    """`ComputeGroupIds` (`query_helper.cu:35-41`)."""
    var blocks = query_helper_blocks(q_count)
    if blocks <= 0:
        return
    ctx.enqueue_function[compute_group_ids_kernel](
        q_sizes.unsafe_ptr(), q_offsets.unsafe_ptr(), offsets_bias,
        Int32(q_count), dst.unsafe_ptr(),
        grid_dim=(blocks, 1, 1),
        block_dim=(QUERY_HELPER_BLOCK_SIZE, 1, 1),
    )


def compute_group_means_kernel(
    target: MutPointer[Float32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    has_weights: Int32,
    q_offsets: MutPointer[UInt32, MutAnyOrigin],
    offsets_bias: UInt32,
    q_sizes: MutPointer[UInt32, MutAnyOrigin],
    q_count_in: Int32,
    query_means: MutPointer[Float32, MutAnyOrigin],
):
    """`ComputeGroupMeansImpl`, the offsets-and-sizes overload
    (`query_helper.cu:132-176`). `has_weights == 0` is their
    `weights == nullptr` arm, a unit weight."""
    var q_count = Int(q_count_in)
    var tid = Int(thread_idx.x)
    var local_qid = tid // QUERY_LANES
    var qid = Int(block_idx.x) * QUERIES_PER_BLOCK + local_qid
    var lane = tid & (QUERY_LANES - 1)

    var read_offset = 0
    var query_size = 0
    if qid < q_count:
        read_offset = Int(q_offsets.unsafe_load(qid) - offsets_bias)
        query_size = Int(q_sizes.unsafe_load(qid))

    # `for (int i = x; i < querySize; i += 32) { sumTarget += t * w; sumWeight += w; }`
    var sum_target = Float32(0.0)
    var sum_weight = Float32(0.0)
    var i = lane
    while i < query_size:
        var t = target.unsafe_load(read_offset + i)
        var w = Float32(1.0)
        if has_weights != Int32(0):
            w = weights.unsafe_load(read_offset + i)
        sum_target = fma(t, w, sum_target)  # the default build's fused op (lane/pinned-mul-contract-free)
        sum_weight = sum_weight + w
        i += QUERY_LANES

    # the two `WarpReduce`s over the query's 32 lanes, in two shared slabs
    var line_target = stack_allocation[
        QUERY_HELPER_BLOCK_SIZE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var line_weight = stack_allocation[
        QUERY_HELPER_BLOCK_SIZE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    line_target[tid] = sum_target
    line_weight[tid] = sum_weight
    barrier()
    var base = local_qid * QUERY_LANES
    var step = QUERY_LANES // 2
    while step > 0:
        if lane < step:
            line_target[base + lane] = (
                line_target[base + lane] + line_target[base + lane + step]
            )
            line_weight[base + lane] = (
                line_weight[base + lane] + line_weight[base + lane + step]
            )
        barrier()
        step //= 2

    if lane == 0 and qid < q_count:
        var total_sum = line_target[base]
        var total_weight = line_weight[base]
        var mean = Float32(0.0)
        if total_weight != Float32(0.0):
            mean = total_sum / total_weight
        query_means.unsafe_store(qid, ftz(mean))


def launch_compute_group_means(
    ctx: DeviceContext,
    mut target: DeviceBuffer[DType.float32],
    mut weights: DeviceBuffer[DType.float32],
    has_weights: Bool,
    mut q_offsets: DeviceBuffer[DType.uint32],
    offsets_bias: UInt32,
    mut q_sizes: DeviceBuffer[DType.uint32],
    q_count: Int,
    mut query_means: DeviceBuffer[DType.float32],
) raises:
    """`ComputeGroupMeans` (`query_helper.cu:179-189`)."""
    var blocks = query_helper_blocks(q_count)
    if blocks <= 0:
        return
    ctx.enqueue_function[compute_group_means_kernel](
        target.unsafe_ptr(), weights.unsafe_ptr(),
        Int32(1) if has_weights else Int32(0),
        q_offsets.unsafe_ptr(), offsets_bias, q_sizes.unsafe_ptr(),
        Int32(q_count), query_means.unsafe_ptr(),
        grid_dim=(blocks, 1, 1),
        block_dim=(QUERY_HELPER_BLOCK_SIZE, 1, 1),
    )


def inverse_permutation_kernel(
    indices: MutPointer[UInt32, MutAnyOrigin],
    dst: MutPointer[UInt32, MutAnyOrigin],
    size_in: Int32,
):
    """`InversePermutationImpl` (`fill.cu:58-65`): `dst[indices[i]] = i`."""
    var size = Int(size_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    while i < size:
        dst.unsafe_store(Int(indices.unsafe_load(i)), UInt32(i))
        i += Int(grid_dim.x) * Int(block_dim.x)


def launch_inverse_permutation(
    ctx: DeviceContext,
    mut indices: DeviceBuffer[DType.uint32],
    mut dst: DeviceBuffer[DType.uint32],
    size: Int,
) raises:
    """`InversePermutation` (`fill.cu:67-77`)."""
    if size <= 0:
        return
    var blocks = (size + INVERSE_PERMUTATION_BLOCK_SIZE - 1) // INVERSE_PERMUTATION_BLOCK_SIZE
    if blocks > QUERY_HELPER_MAX_BLOCKS:
        blocks = QUERY_HELPER_MAX_BLOCKS
    ctx.enqueue_function[inverse_permutation_kernel](
        indices.unsafe_ptr(), dst.unsafe_ptr(), Int32(size),
        grid_dim=(blocks, 1, 1),
        block_dim=(INVERSE_PERMUTATION_BLOCK_SIZE, 1, 1),
    )
