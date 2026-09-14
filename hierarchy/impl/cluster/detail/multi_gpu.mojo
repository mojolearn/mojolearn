# SPDX-License-Identifier: Apache-2.0
"""Pinned pairwise rows; root diagonal policy and Boruvka order stay intact."""
from max.gpu.host import DeviceContext, DeviceBuffer
from std.os import getenv
from std.sys.compile import is_defined
from core.multi_gpu import peer_clone
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from neighbors.checks.pinned_distance_tile import pinned_distance_tile_kernel


def hierarchy_device_count(rows: Int, sabotage: Int32) raises -> Int:
    var count = Int(getenv("MOJOLEARN_HIERARCHY_DEVICE_COUNT", "1"))
    if count < 1 or count > 64:
        raise Error("hierarchy device count must be in 1..64")
    if count > 1:
        if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
            raise Error("parallel hierarchy requires IDENTICAL mode")
        if sabotage != 0:
            raise Error("hierarchy sabotage probes require one device")
    return min(rows, count)


def pairwise_rows(ctx: DeviceContext, mut x: DeviceBuffer[DType.float32],
                  mut norms: DeviceBuffer[DType.float32],
                  mut data: DeviceBuffer[DType.float32], m: Int, d: Int,
                  is_sqrt: Int32, tpb: Int, count: Int) raises:
    ctx.synchronize()
    var devices = List[DeviceContext]()
    var queries = List[DeviceBuffer[DType.float32]]()
    var references = List[DeviceBuffer[DType.float32]]()
    var qnorms = List[DeviceBuffer[DType.float32]]()
    var rnorms = List[DeviceBuffer[DType.float32]]()
    var outputs = List[DeviceBuffer[DType.float32]]()
    for rank in range(count):
        var first = m * rank // count
        var rows = m * (rank + 1) // count - first
        devices.append(DeviceContext(device_id=rank))
        var source = first
        comptime if is_defined["MOJOLEARN_HIERARCHY_PARALLEL_SABOTAGE"]():
            # Check-only reach witness: later owners read their query rows one row early.
            if rank > 0:
                source = first - 1
        var qv = x.create_sub_buffer[DType.float32](source*d, rows*d)
        var nv = norms.create_sub_buffer[DType.float32](source, rows)
        queries.append(peer_clone(ctx, devices[rank], qv))
        references.append(peer_clone(ctx, devices[rank], x))
        qnorms.append(peer_clone(ctx, devices[rank], nv))
        rnorms.append(peer_clone(ctx, devices[rank], norms))
        outputs.append(devices[rank].enqueue_create_buffer[DType.float32](rows*m))
        devices[rank].enqueue_function[pinned_distance_tile_kernel](
            outputs[rank].unsafe_ptr(), queries[rank].unsafe_ptr(), references[rank].unsafe_ptr(),
            qnorms[rank].unsafe_ptr(), rnorms[rank].unsafe_ptr(),
            Int32(rows), Int32(m), Int32(d), is_sqrt,
            grid_dim=((rows*m+tpb-1)//tpb, 1, 1), block_dim=(tpb, 1, 1))
    for rank in range(count):
        var first = m * rank // count
        var rows = m * (rank + 1) // count - first
        devices[rank].synchronize()
        var target = data.create_sub_buffer[DType.float32](first*m, rows*m)
        outputs[rank].enqueue_copy_to(target)
        devices[rank].synchronize()
    _ = outputs^
    _ = rnorms^
    _ = qnorms^
    _ = references^
    _ = queries^
    for rank in range(count):
        devices[rank].synchronize()
    _ = devices^
    ctx.synchronize()
