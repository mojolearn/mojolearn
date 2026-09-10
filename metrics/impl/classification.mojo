# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Unweighted encoded single-label classification; integer GPU counts.

PRF uses O(k) true-positive/true/predicted counts, preserving off-selection
errors. Confusion uses dense counts and optionally skips labels encoded -1.
Counts cannot overflow: n<=Int32.max, and combined denominators use Int64.
Each exact integer numerator/denominator converts separately to Float32.
Final floating arithmetic uses ascending class order and the mode's division
policy; no metric arithmetic runs on the CPU.
"""
from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx
from std.math import ceildiv
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import ftz, identical_div

comptime MAX_CONFUSION_CLASSES = 4096
comptime MAX_PRF_CLASSES = 715827882


def count_labels_kernel[matrix: Bool, count_total: Bool](
    truth: MutPointer[Int32, MutAnyOrigin], prediction: MutPointer[Int32, MutAnyOrigin],
    n: Int32, classes: Int32, counts: MutPointer[Int32, MutAnyOrigin],
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        var y = Int(truth.unsafe_load(i))
        var p = Int(prediction.unsafe_load(i))
        var k = Int(classes)
        comptime if matrix:
            if y >= 0 and p >= 0:
                _ = Atomic.fetch_add(counts.unsafe_offset(y*k+p), Int32(1))
                comptime if count_total:
                    _ = Atomic.fetch_add(counts.unsafe_offset(k*k), Int32(1))
        else:
            _ = Atomic.fetch_add(counts.unsafe_offset(k+y), Int32(1))
            _ = Atomic.fetch_add(counts.unsafe_offset(2*k+p), Int32(1))
            if y == p:
                _ = Atomic.fetch_add(counts.unsafe_offset(y), Int32(1))


@always_inline
def count_ratio(numerator: Int64, denominator: Int64, zero: Int32) -> Float32:
    if denominator == 0:
        return Float32(zero)
    return ftz(identical_div(Float32(numerator), Float32(denominator)))


def confusion_finish_kernel[dtype: DType](
    counts: MutPointer[Int32, MutAnyOrigin], k_in: Int32, normalization: Int32,
    result: MutPointer[Scalar[dtype], MutAnyOrigin],
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var k = Int(k_in)
    comptime if dtype == DType.int64:
        if i < k*k:
            result.unsafe_store(i, Scalar[dtype](Int64(counts.unsafe_load(i))))
    else:
        if normalization == 3:
            if i < k*k:
                result.unsafe_store(i, Scalar[dtype](count_ratio(Int64(counts.unsafe_load(i)), Int64(counts.unsafe_load(k*k)), 0)))
        elif i < k:
            var denominator = Int64(0)
            for j in range(k):
                var index = i*k+j if normalization == 1 else j*k+i
                denominator += Int64(counts.unsafe_load(index))
            for j in range(k):
                var index = i*k+j if normalization == 1 else j*k+i
                result.unsafe_store(index, Scalar[dtype](count_ratio(Int64(counts.unsafe_load(index)), denominator, 0)))


def prf_finish_kernel(
    counts: MutPointer[Int32, MutAnyOrigin], k_in: Int32, selected_in: Int32,
    average: Int32, positive: Int32, zero: Int32,
    result: MutPointer[Float32, MutAnyOrigin],
):
    if Int(thread_idx.x) != 0:
        return
    var k = Int(k_in)
    var selected = Int(selected_in)
    var width = selected if average == 0 else 1
    var tp_sum = Int64(0)
    var true_sum = Int64(0)
    var pred_sum = Int64(0)
    for c in range(selected):
        tp_sum += Int64(counts.unsafe_load(c))
        true_sum += Int64(counts.unsafe_load(k+c))
        pred_sum += Int64(counts.unsafe_load(2*k+c))
    comptime for metric in range(3):
        var total = Float32(0)
        var undefined = False
        if average == 2:
            var numerator = tp_sum if metric < 2 else 2*tp_sum
            var denominator = pred_sum if metric == 0 else (true_sum if metric == 1 else true_sum+pred_sum)
            total = count_ratio(numerator, denominator, zero)
            undefined = denominator == 0
        else:
            var count = 1 if average == 1 else selected
            for j in range(count):
                var c = Int(positive) if average == 1 else j
                var tp = Int64(counts.unsafe_load(c))
                var support = Int64(counts.unsafe_load(k+c))
                var predicted = Int64(counts.unsafe_load(2*k+c))
                var denominator = predicted if metric == 0 else (support if metric == 1 else support+predicted)
                var numerator = tp if metric < 2 else 2*tp
                var score = count_ratio(numerator, denominator, zero)
                undefined = undefined or denominator == 0
                if average == 0:
                    result.unsafe_store(metric*width+j, score)
                elif average == 1:
                    total = score
                else:
                    # With no selected support, sklearn's weighted average
                    # falls back to the unweighted mean of selected scores.
                    if average == 4 and true_sum > 0:
                        total = ftz(total + ftz(score * Float32(support)))
                    else:
                        total = ftz(total + score)
            if average == 3 or average == 4:
                var denominator = true_sum if average == 4 and true_sum > 0 else Int64(selected)
                total = ftz(identical_div(total, Float32(denominator)))
        if average != 0:
            result.unsafe_store(metric, total)
        result.unsafe_store(3*width+metric, Float32(1 if undefined else 0))


def classification_counts[matrix: Bool, count_total: Bool = False](
    ctx: DeviceContext, mut y: DeviceBuffer[DType.int32], mut p: DeviceBuffer[DType.int32],
    n: Int, k: Int,
) raises -> DeviceBuffer[DType.int32]:
    # Device callers supply validated dense labels; host-list entry below
    # validates their values before upload. Matrix permits the sentinel -1.
    if n <= 0 or n > 2147483647 or n > len(y) or n > len(p):
        raise Error("classification metrics: invalid n (requires1..Int32.max)")
    comptime cap = MAX_CONFUSION_CLASSES if matrix else MAX_PRF_CLASSES
    if k <= 0 or k > cap:
        raise Error("classification metrics: class allocation limit exceeded")
    var cells = k*k+1 if matrix else 3*k
    var counts = ctx.enqueue_create_buffer[DType.int32](cells)
    counts.enqueue_fill(0)
    ctx.enqueue_function[count_labels_kernel[matrix, count_total]](
        y.unsafe_ptr(), p.unsafe_ptr(), Int32(n), Int32(k), counts.unsafe_ptr(),
        grid_dim=ceildiv(n,256), block_dim=256,
    )
    return counts^


def confusion_matrix[dtype: DType](
    ctx: DeviceContext, mut y: DeviceBuffer[DType.int32], mut p: DeviceBuffer[DType.int32],
    n: Int, k: Int, normalization: Int = 0,
) raises -> DeviceBuffer[dtype]:
    comptime assert dtype == DType.int64 or dtype == DType.float32
    if normalization < 0 or normalization > 3:
        raise Error("confusion_matrix: invalid normalization")
    comptime if dtype == DType.int64:
        if normalization != 0:
            raise Error("confusion_matrix: raw counts require normalization0")
    else:
        if normalization == 0:
            raise Error("confusion_matrix: normalized output requires mode1..3")
    var counts: DeviceBuffer[DType.int32]
    if normalization == 3:
        counts = classification_counts[True, True](ctx,y,p,n,k)
    else:
        counts = classification_counts[True, False](ctx,y,p,n,k)
    var result = ctx.enqueue_create_buffer[dtype](k*k)
    var cells = k*k
    comptime if dtype == DType.float32:
        if normalization != 3:
            cells = k
    ctx.enqueue_function[confusion_finish_kernel[dtype]](
        counts.unsafe_ptr(), Int32(k), Int32(normalization), result.unsafe_ptr(),
        grid_dim=ceildiv(cells,256), block_dim=256,
    )
    # counts must stay live until its queued consumer completes.
    ctx.synchronize()
    _ = counts^
    return result^


def precision_recall_fscore(
    ctx: DeviceContext, mut y: DeviceBuffer[DType.int32], mut p: DeviceBuffer[DType.int32],
    n: Int, k: Int, average: Int, positive: Int, zero: Int, selected: Int,
) raises -> DeviceBuffer[DType.float32]:
    if average < 0 or average > 4 or zero < 0 or zero > 1:
        raise Error("precision_recall_fscore: invalid average/zero_division")
    if selected <= 0 or selected > k or (average == 1 and (positive < 0 or positive >= k)):
        raise Error("precision_recall_fscore: invalid selected/positive class")
    var counts = classification_counts[False](ctx,y,p,n,k)
    var width = selected if average == 0 else 1
    var result = ctx.enqueue_create_buffer[DType.float32](3*width+3)
    ctx.enqueue_function[prf_finish_kernel](
        counts.unsafe_ptr(), Int32(k), Int32(selected), Int32(average), Int32(positive), Int32(zero), result.unsafe_ptr(),
        grid_dim=1, block_dim=32,
    )
    ctx.synchronize()
    _ = counts^
    return result^
