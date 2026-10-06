# SPDX-License-Identifier: Apache-2.0
"""T29 V1 generated-PairLogit production objective kernel.
NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
"""
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from gbdt.targets.tree_t29_units import (
    T29_GROUP_LANES, t29_pair_row, t29_add, t29_fold_lanes,
)


def pair_logit_group_versioned_kernel[
    estimation: Bool, second_order: Bool, store_acc: Bool, work_class: Int = 0,
](
    point: MutPointer[Float32, MutAnyOrigin],
    grades: MutPointer[Float32, MutAnyOrigin],
    group_offsets: MutPointer[UInt32, MutAnyOrigin],
    acc: MutPointer[Float32, MutAnyOrigin],
    group_w_at: Int32,
    row_weights: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    write_map: MutPointer[UInt32, MutAnyOrigin],
    has_write_map: Int32,
    stats: MutPointer[Float32, MutAnyOrigin],
    function_value: MutPointer[Float32, MutAnyOrigin],
    compute_fv: Int32,
    plane_magnitudes: MutPointer[Float32, MutAnyOrigin],
    compute_magnitudes: Int32,
    der_acc_at: Int32,
    der2_acc_at: Int32,
    fv_acc_at: Int32,
):
    comptime assert not (estimation and second_order), "second_order is a search flag"
    var group = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var begin = Int(group_offsets[group])
    var end = Int(group_offsets[group + 1])
    # The independent S arm partitions launches by one logical group tile.
    # It never changes a partner chunk, lane, row fold or query membership.
    comptime if work_class == 1:
        if end - begin > T29_GROUP_LANES:
            return
    elif work_class == 2:
        if end - begin <= T29_GROUP_LANES:
            return
    var n_rows = Int(n_rows_in)
    var weight = acc[Int(group_w_at) + group]
    var local_value = Float32(0.0)
    var local_weight = Float32(0.0)
    var local_gradient = Float32(0.0)
    var row = begin + tid
    while row < end:
        var sums = t29_pair_row(point, grades, row, begin, end, weight, compute_fv != Int32(0))
        var plane0 = row_weights[row]
        comptime if second_order:
            plane0 = sums[1]
        comptime if estimation:
            var dst = row
            if has_write_map != Int32(0):
                dst = Int(write_map[row])
            stats[dst] = sums[0]
            stats[n_rows + dst] = sums[1]
        else:
            stats[row] = plane0
            stats[n_rows + row] = sums[0]
        comptime if store_acc:
            acc[Int(der_acc_at) + row] = sums[0]
            acc[Int(der2_acc_at) + row] = sums[1]
        local_value = t29_add(local_value, sums[2])
        local_weight = t29_add(local_weight, abs(plane0))
        local_gradient = t29_add(local_gradient, abs(sums[0]))
        row += T29_GROUP_LANES
    var shared = stack_allocation[
        3 * T29_GROUP_LANES, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    shared[tid] = local_value
    shared[T29_GROUP_LANES + tid] = local_weight
    shared[2 * T29_GROUP_LANES + tid] = local_gradient
    barrier()
    if tid == 0:
        var lanes = InlineArray[Float32, T29_GROUP_LANES](fill=0.0)
        if compute_fv != Int32(0):
            for lane in range(T29_GROUP_LANES):
                lanes[lane] = shared[lane]
            var value = t29_fold_lanes(lanes)
            function_value[group] = value
            comptime if store_acc:
                acc[Int(fv_acc_at) + group] = value
        if compute_magnitudes != Int32(0):
            for lane in range(T29_GROUP_LANES):
                lanes[lane] = shared[T29_GROUP_LANES + lane]
            plane_magnitudes[2 * group] = t29_fold_lanes(lanes)
            for lane in range(T29_GROUP_LANES):
                lanes[lane] = shared[2 * T29_GROUP_LANES + lane]
            plane_magnitudes[2 * group + 1] = t29_fold_lanes(lanes)
