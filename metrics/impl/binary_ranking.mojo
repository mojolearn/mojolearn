# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Binary ranking metrics, all score ordering and metric arithmetic on GPU.

Reference: cuML v26.08.00 python/cuml/cuml/metrics/_ranking.py
_binary_clf_curve, _group_same_scores, _calculate_area_under_curve; sklearn
metrics/_ranking.py precision_recall_curve for no-positive and terminal policy.
Deviations: existing stable 32-pass radix replaces CuPy argsort; exact integer
scan replaces float group atomics. AUC uses the equivalent tie-aware pair-count
formula with an exact Int64 tree fold instead of floating trapezoids. This fixes
the result in all modes (integer sums); ratios separately convert counts to Float32.
"""
from std.gpu import block_idx, block_dim, thread_idx
from std.memory import bitcast, stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext, DeviceBuffer
from checks.numerics import ftz, identical_div
from metrics.checks.device_io import download_f32
from gbdt.gpu_util.kernel.radix_sort import launch_radix_sort_bins, scan_key_bit_kernel
from gbdt.gpu_util.kernel.reorder_one_bit import REORDER_BLOCK, scan_block_sums_kernel, add_block_carry_kernel


def ranking_keys_kernel(
    y: MutPointer[Int32, MutAnyOrigin],
    scores: MutPointer[Float32, MutAnyOrigin],
    keys: MutPointer[UInt32, MutAnyOrigin],
    labels: MutPointer[UInt32, MutAnyOrigin],
    n: Int32,
):
    var i = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if i < Int(n):
        var score = scores.unsafe_load(i)
        # Numeric equality groups both zeros; canonicalize before radix sort.
        var bits = bitcast[DType.uint32](score)
        if (bits & UInt32(0x7fffffff)) == 0:
            bits = UInt32(0)
        var key = ~bits if (bits & UInt32(0x80000000)) != 0 else bits | UInt32(0x80000000)
        keys.unsafe_store(i,key)
        labels.unsafe_store(i,UInt32(y.unsafe_load(i)))


def ranking_group_flags_kernel(
    keys: MutPointer[UInt32, MutAnyOrigin],
    flags: MutPointer[UInt32, MutAnyOrigin],
    n: Int32,
):
    var i = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if i < Int(n):
        var start = True
        if i > 0:
            start = keys.unsafe_load(i) != keys.unsafe_load(i-1)
        flags.unsafe_store(i,UInt32(1) if start else UInt32(0))


# Exclusive flag offsets assign exactly one slot to each distinct key.
def ranking_compact_kernel(
    flags: MutPointer[UInt32, MutAnyOrigin],
    offsets: MutPointer[Int32, MutAnyOrigin],
    starts: MutPointer[Int32, MutAnyOrigin],
    n: Int32,
    size: MutPointer[Int32, MutAnyOrigin],
):
    var i = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if i < Int(n):
        var flag = Int32(flags.unsafe_load(i))
        var offset = offsets.unsafe_load(i)
        if flag != 0:
            starts.unsafe_store(Int(offset),Int32(i))
        if i == Int(n)-1:
            size.unsafe_store(0,offset+flag)


# Prefix counts exclude the current row; each compacted start begins a tie.
def ranking_groups_kernel[curve: Bool](
    keys: MutPointer[UInt32, MutAnyOrigin],
    labels: MutPointer[UInt32, MutAnyOrigin],
    prefix: MutPointer[Int32, MutAnyOrigin],
    starts: MutPointer[Int32, MutAnyOrigin],
    size: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
    output: MutPointer[Float32, MutAnyOrigin],
    contributions: MutPointer[Int64, MutAnyOrigin],
):
    var group = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    var m = Int(size.unsafe_load(0))
    var n = Int(n_in)
    if group < m:
        var i = Int(starts.unsafe_load(group))
        var j = n if group+1 == m else Int(starts.unsafe_load(group+1))
        var positives = Int64(prefix.unsafe_load(n-1))+Int64(labels.unsafe_load(n-1))
        var before = Int64(prefix.unsafe_load(i))
        comptime if curve:
            var tp = positives-before
            output.unsafe_store(group,ftz(identical_div(Float32(tp),Float32(n-i))))
            var recall = Float32(1) if positives == 0 else ftz(identical_div(Float32(tp),Float32(positives)))
            output.unsafe_store(n+1+group,recall)
            var key = keys.unsafe_load(i)
            var bits = key & UInt32(0x7fffffff) if (key & UInt32(0x80000000)) != 0 else ~key
            output.unsafe_store(2*(n+1)+group,bitcast[DType.float32](bits))
            if group == 0:
                output.unsafe_store(m,Float32(1))
                output.unsafe_store(n+1+m,Float32(0))
        else:
            var through = positives if j == n else Int64(prefix.unsafe_load(j))
            var group_positive = through-before
            var group_negative = Int64(j-i)-group_positive
            contributions.unsafe_store(group,group_positive*(2*(Int64(i)-before)+group_negative))


# AUC contributions are integers, so their grouping does not affect the exact
# result.  Collapse bounded contiguous runs in parallel, then fold the run
# totals as a multi-block tree (`ranking_auc_level_kernel`, 256 per block per
# level) whose last level carries the AUC epilogue: no lane walks the runs.
comptime AUC_TPB = 256


def ranking_auc_partial_kernel(
    size: MutPointer[Int32, MutAnyOrigin],
    contributions: MutPointer[Int64, MutAnyOrigin],
    partials: MutPointer[Int64, MutAnyOrigin],
    chunks: Int32,
):
    comptime RUN = 256
    var chunk = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if chunk >= Int(chunks):
        return
    var begin = chunk*RUN
    var end = min(begin+RUN,Int(size.unsafe_load(0)))
    var total = Int64(0)
    for group in range(begin,end):
        total += contributions.unsafe_load(group)
    partials.unsafe_store(chunk,total)


@always_inline
def _block_sum_i64(value: Int64) -> Int64:
    """Exact Int64 sum over the AUC_TPB threads of a block (shared-memory
    halving tree). Every thread calls it; thread 0's return is the sum."""
    var tid = Int(thread_idx.x)
    var slab = stack_allocation[AUC_TPB, Scalar[DType.int64], address_space = AddressSpace.SHARED]()
    slab[unsafe_offset = tid] = value
    barrier()
    var step = AUC_TPB // 2
    while step > 0:
        if tid < step:
            slab[unsafe_offset = tid] = slab[unsafe_offset = tid] + slab[unsafe_offset = tid + step]
        barrier()
        step //= 2
    var total = slab[unsafe_offset = 0]
    barrier()
    return total


def ranking_auc_level_kernel(
    src: MutPointer[Int64, MutAnyOrigin],
    m: Int32,
    dst: MutPointer[Int64, MutAnyOrigin],
):
    """One level: block `b` sums `src[b*AUC_TPB, (b+1)*AUC_TPB)` into `dst[b]`."""
    var i = Int(block_idx.x)*AUC_TPB+Int(thread_idx.x)
    var v = src.unsafe_load(i) if i < Int(m) else Int64(0)
    var total = _block_sum_i64(v)
    if Int(thread_idx.x) == 0:
        dst.unsafe_store(Int(block_idx.x),total)


def ranking_auc_partial_fold_kernel(
    labels: MutPointer[UInt32, MutAnyOrigin],
    prefix: MutPointer[Int32, MutAnyOrigin],
    partials: MutPointer[Int64, MutAnyOrigin],
    m: Int32,
    n_in: Int32,
    output: MutPointer[Float32, MutAnyOrigin],
):
    """The last level (`m <= AUC_TPB` partials) and the AUC epilogue."""
    var t = Int(thread_idx.x)
    var v = partials.unsafe_load(t) if t < Int(m) else Int64(0)
    var total = _block_sum_i64(v)
    if t == 0:
        var n = Int(n_in)
        var positives = Int64(prefix.unsafe_load(n-1))+Int64(labels.unsafe_load(n-1))
        var negatives = Int64(n)-positives
        output.unsafe_store(0,ftz(identical_div(Float32(total),Float32(2*positives*negatives))))


def binary_ranking[curve: Bool](
    ctx: DeviceContext,
    mut y: DeviceBuffer[DType.int32],
    mut scores: DeviceBuffer[DType.float32],
    n: Int,
) raises -> Tuple[List[Float32], Int]:
    if n <= 0 or n > 2147483647 or n > len(y) or n > len(scores):
        raise Error("binary ranking: invalid input length")
    var blocks = (n+REORDER_BLOCK-1)//REORDER_BLOCK
    var keys = ctx.enqueue_create_buffer[DType.uint32](n)
    var labels = ctx.enqueue_create_buffer[DType.uint32](n)
    var temp_keys = ctx.enqueue_create_buffer[DType.uint32](n)
    var temp_labels = ctx.enqueue_create_buffer[DType.uint32](n)
    var prefix = ctx.enqueue_create_buffer[DType.int32](n)
    var block_sums = ctx.enqueue_create_buffer[DType.int32](blocks)
    var output_size = 3*n+2 if curve else 1
    var output = ctx.enqueue_create_buffer[DType.float32](output_size)
    output.enqueue_fill(0)
    var size = ctx.enqueue_create_buffer[DType.int32](1)
    var starts = ctx.enqueue_create_buffer[DType.int32](n)
    var contributions = ctx.enqueue_create_buffer[DType.int64](1 if curve else n)
    var auc_chunks = (n+255)//256
    var auc_partials = ctx.enqueue_create_buffer[DType.int64](1 if curve else auc_chunks)
    var auc_scratch = (auc_chunks+AUC_TPB-1)//AUC_TPB
    var auc_s0 = ctx.enqueue_create_buffer[DType.int64](1 if curve else auc_scratch)
    var auc_s1 = ctx.enqueue_create_buffer[DType.int64](1 if curve else auc_scratch)
    ctx.enqueue_function[ranking_keys_kernel](
        y.unsafe_ptr(),
        scores.unsafe_ptr(),
        keys.unsafe_ptr(),
        labels.unsafe_ptr(),
        Int32(n),
        grid_dim=blocks,
        block_dim=REORDER_BLOCK,
    )
    launch_radix_sort_bins(ctx,n,0,32,keys,labels,temp_keys,temp_labels,prefix,block_sums)
    ctx.enqueue_function[scan_key_bit_kernel](
        labels.unsafe_ptr(),
        Int32(0),
        Int32(n),
        prefix.unsafe_ptr(),
        block_sums.unsafe_ptr(),
        grid_dim=blocks,
        block_dim=REORDER_BLOCK,
    )
    ctx.enqueue_function[scan_block_sums_kernel](block_sums.unsafe_ptr(),Int32(blocks),grid_dim=1,block_dim=1)
    ctx.enqueue_function[add_block_carry_kernel](
        prefix.unsafe_ptr(),
        block_sums.unsafe_ptr(),
        Int32(n),
        grid_dim=blocks,
        block_dim=REORDER_BLOCK,
    )
    # Radix scratch is dead after sorting: reuse it for group flags/offsets.
    ctx.enqueue_function[ranking_group_flags_kernel](
        keys.unsafe_ptr(),
        temp_keys.unsafe_ptr(),
        Int32(n),
        grid_dim=blocks,
        block_dim=REORDER_BLOCK,
    )
    ctx.enqueue_function[scan_key_bit_kernel](
        temp_keys.unsafe_ptr(),
        Int32(0),
        Int32(n),
        temp_labels.unsafe_ptr().unsafe_bitcast[Int32](),
        block_sums.unsafe_ptr(),
        grid_dim=blocks,
        block_dim=REORDER_BLOCK,
    )
    ctx.enqueue_function[scan_block_sums_kernel](block_sums.unsafe_ptr(),Int32(blocks),grid_dim=1,block_dim=1)
    ctx.enqueue_function[add_block_carry_kernel](
        temp_labels.unsafe_ptr().unsafe_bitcast[Int32](),
        block_sums.unsafe_ptr(),
        Int32(n),
        grid_dim=blocks,
        block_dim=REORDER_BLOCK,
    )
    ctx.enqueue_function[ranking_compact_kernel](
        temp_keys.unsafe_ptr(),
        temp_labels.unsafe_ptr().unsafe_bitcast[Int32](),
        starts.unsafe_ptr(),
        Int32(n),
        size.unsafe_ptr(),
        grid_dim=blocks,
        block_dim=REORDER_BLOCK,
    )
    ctx.enqueue_function[ranking_groups_kernel[curve]](
        keys.unsafe_ptr(),
        labels.unsafe_ptr(),
        prefix.unsafe_ptr(),
        starts.unsafe_ptr(),
        size.unsafe_ptr(),
        Int32(n),
        output.unsafe_ptr(),
        contributions.unsafe_ptr(),
        grid_dim=blocks,
        block_dim=REORDER_BLOCK,
    )
    comptime if not curve:
        ctx.enqueue_function[ranking_auc_partial_kernel](
            size.unsafe_ptr(),
            contributions.unsafe_ptr(),
            auc_partials.unsafe_ptr(),
            Int32(auc_chunks),
            grid_dim=(auc_chunks+255)//256,
            block_dim=256,
        )
        var src = rebind[MutPointer[Int64, MutAnyOrigin]](auc_partials.unsafe_ptr())
        var p0 = rebind[MutPointer[Int64, MutAnyOrigin]](auc_s0.unsafe_ptr())
        var p1 = rebind[MutPointer[Int64, MutAnyOrigin]](auc_s1.unsafe_ptr())
        var m = auc_chunks
        var flip = False
        while m > AUC_TPB:
            var next_m = (m+AUC_TPB-1)//AUC_TPB
            var dst = p1 if flip else p0
            ctx.enqueue_function[ranking_auc_level_kernel](
                src, Int32(m), dst, grid_dim=next_m, block_dim=AUC_TPB,
            )
            src = dst
            m = next_m
            flip = not flip
        ctx.enqueue_function[ranking_auc_partial_fold_kernel](
            labels.unsafe_ptr(),
            prefix.unsafe_ptr(),
            src,
            Int32(m),
            Int32(n),
            output.unsafe_ptr(),
            grid_dim=(m+AUC_TPB-1)//AUC_TPB,
            block_dim=AUC_TPB,
        )
    var host = download_f32(ctx,output,output_size)
    var m = 0
    with size.map_to_host() as h:
        m = Int(h[0])
    _ = contributions^
    _ = auc_partials^
    _ = auc_s0^
    _ = auc_s1^
    _ = starts^
    _ = size^
    _ = output^
    _ = block_sums^
    _ = prefix^
    _ = temp_labels^
    _ = temp_keys^
    _ = labels^
    _ = keys^
    return (host^,m)
