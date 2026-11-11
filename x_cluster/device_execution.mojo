# SPDX-License-Identifier: Apache-2.0
"""One native execution owner shared by typed requests and legacy boundaries.

All promoted and optional pointer routes retain their original order and
defaults. Python admission and output packing stay outside this function.
"""
from core.hostptr import f32_ptr, read_f32
from core.device_scan import device_first_nonfinite
from x_cluster.device_ops import DeviceOps
from x_cluster.bisect_fast import BISECT_FAST_ZEROCOPY, bisect_entry_ptr
from x_cluster.entries import ENTRY_BISECT, ENTRY_MINIBATCH, run_entry
from x_cluster.minibatch_ptr import (
    MBK_DEVSCAN_ANY, MBK_IDN_RESIDENT, MBK_NONFINITE_MSG, MBK_ZEROCOPY,
    minibatch_entry_ptr, minibatch_entry_ptr_idn,
)
from x_cluster.out import ClusterOut


def cluster_device_execute(which: Int, source: Int, count: Int, auxiliary: Int,
                           auxiliary_count: Int, ints: List[Int], floats: List[Float64]) raises -> ClusterOut:
    # Same route sequence as the original x_cluster_call binding. Keep tuned
    # kernels behind their existing switches, including both rollback paths.
    comptime if MBK_ZEROCOPY:
        if which == ENTRY_MINIBATCH and count > 0:
            var a0 = read_f32(auxiliary, auxiliary_count) if auxiliary_count > 0 else List[Float32]()
            var xp = f32_ptr(source)
            var res0 = ClusterOut()
            var took = False
            var ops0 = DeviceOps()
            took = minibatch_entry_ptr(ops0, xp, count, a0, ints, floats, res0)
            _ = ops0^
            _ = a0^
            if took:
                return res0^
    comptime if BISECT_FAST_ZEROCOPY:
        if which == ENTRY_BISECT and count > 0:
            var xp1 = f32_ptr(source)
            var res1 = ClusterOut()
            var took1 = False
            var ops1 = DeviceOps()
            took1 = bisect_entry_ptr(ops1, xp1, count, auxiliary_count > 0, ints, floats, res1)
            _ = ops1^
            if took1:
                return res1^
    comptime if MBK_IDN_RESIDENT:
        if which == ENTRY_MINIBATCH and count > 0:
            var a2 = read_f32(auxiliary, auxiliary_count) if auxiliary_count > 0 else List[Float32]()
            var xp2 = f32_ptr(source)
            var res2 = ClusterOut()
            var took2 = False
            var ops2 = DeviceOps()
            took2 = minibatch_entry_ptr_idn(ops2, xp2, count, a2, ints, floats, res2)
            _ = ops2^
            _ = a2^
            if took2:
                return res2^
    var x = read_f32(source, count) if count > 0 else List[Float32]()
    var a = read_f32(auxiliary, auxiliary_count) if auxiliary_count > 0 else List[Float32]()
    var scan_dev = False
    comptime if MBK_DEVSCAN_ANY:
        scan_dev = which == ENTRY_MINIBATCH and len(ints) > 12 and ints[12] != 0 and count > 0
    var result = ClusterOut()
    var bad = False
    var ops = DeviceOps()
    if scan_dev:
        var xd = ops.ctx.enqueue_create_buffer[DType.float32](count)
        ops.ctx.enqueue_copy(dst_buf=xd, src_ptr=x.unsafe_ptr())
        bad = device_first_nonfinite(ops.ctx, xd, count) >= 0
        _ = xd^
    if not bad:
        result = run_entry(ops, which, x, a, ints, floats)
    _ = ops^
    _ = x^
    _ = a^
    if bad:
        raise Error(MBK_NONFINITE_MSG)
    return result^
