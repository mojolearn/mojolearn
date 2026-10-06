# SPDX-License-Identifier: Apache-2.0
"""A07 actual production WorkloadInfo contract and complete weighted fits.
Reference-only host fixture verifies every row exactly once and each
histogram task's work bound. No host or Apple IDENTICAL timing.
"""
from std.math import ceildiv
from max.gpu.host import DeviceContext
from ensemble.decisiontree.batched_levelalgo.builder import IDN_RF_TASK_ROWS256, HIST_WORKLOAD_GRANULARITY, update_workload_info_host
from ensemble.decisiontree.batched_levelalgo.kernels.builder_kernels import NodeWorkItem, InstanceRange, WorkloadInfo
from ensemble.checks.sample_weight_check import arm_a_zero_weight_drop, arm_b_validation, arm_c_double_counting, arm_d_weighted_bins_train


def check_descriptors() raises:
    var counts: List[Int] = [0, 1, 255, 256, 257, 513, 16385]
    var items = List[NodeWorkItem]()
    var begin = 0
    for node in range(len(counts)):
        items.append(NodeWorkItem(node, Int32(0), InstanceRange(begin, counts[node])))
        begin += counts[node]
    var table = List[WorkloadInfo]()
    table.resize(begin // 128 + len(counts) * 2, WorkloadInfo(Int32(0), Int32(0), Int32(1)))
    var n = update_workload_info_host(items, table.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), HIST_WORKLOAD_GRANULARITY)
    var visits = List[Int](length=begin, fill=0)
    var expected_tasks = 0
    for node in range(len(counts)):
        expected_tasks += max(ceildiv(counts[node], HIST_WORKLOAD_GRANULARITY), 1)
    if n != expected_tasks:
        raise Error("A07 actual descriptor count differs")
    for task in range(n):
        var work = table[task]
        var node = Int(work.nodeid)
        var rows = 0
        for tid in range(128):
            var pos = items[node].instances.begin + tid + Int(work.offset_blockid) * 128
            while pos < items[node].instances.begin + counts[node]:
                visits[pos] += 1
                rows += 1
                pos += 128 * Int(work.num_blocks)
        if rows > HIST_WORKLOAD_GRANULARITY:
            raise Error("A07 histogram task exceeds actual row bound")
    for pos in range(begin):
        if visits[pos] != 1:
            raise Error("A07 task lost/duplicated row")
    comptime if IDN_RF_TASK_ROWS256:
        comptime assert HIST_WORKLOAD_GRANULARITY == 256
    print("A07_TASK_MAP_PASS tasks", n, "bound", HIST_WORKLOAD_GRANULARITY)


def main() raises:
    check_descriptors()
    var ctx = DeviceContext()
    var fails = arm_a_zero_weight_drop(ctx)
    fails += arm_b_validation(ctx)
    fails += arm_c_double_counting(ctx)
    fails += arm_d_weighted_bins_train(ctx)
    if fails != 0:
        raise Error("A07 complete weighted fit/failure contract failed")
    print("A07_PRODUCTION_PASS weighted_fits validation bootstrap objective_contract")
