# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Small contention check of the production mutex helper.

Two blocks, one payload writer per block, 128 claims each. Check every
handoff's count/checksum pair and the final count. The skip-write sabotage
must fail the same host check. This is a runtime smoke, not a memory-model
proof or a substitute for repeated forest fits on gfx942.
"""
from std.atomic import Atomic, Ordering
from std.gpu import thread_idx
from max.gpu.host import DeviceContext
from core.device_mutex import claim_device_mutex


def contend[skip_write: Bool, available: Int, held: Int](
    mutex: MutPointer[Int32, MutAnyOrigin],
    payload: MutPointer[Int32, MutAnyOrigin],
):
    if thread_idx.x != 0:
        return
    for _ in range(128):
        claim_device_mutex(mutex, Int32(available), Int32(held))
        var count = payload[unsafe_offset=0]
        if payload[unsafe_offset=1] != (count ^ Int32(0x12345678)):
            payload[unsafe_offset=2] = Int32(1)
        comptime if not skip_write:
            count += 1
            payload[unsafe_offset=0] = count
            payload[unsafe_offset=1] = count ^ Int32(0x12345678)
        Atomic.store[ordering = Ordering.RELEASE](mutex, Int32(available))


def check[skip_write: Bool, available: Int, held: Int](
    ctx: DeviceContext
) raises -> Bool:
    var mutex = ctx.enqueue_create_buffer[DType.int32](1)
    var data = ctx.enqueue_create_buffer[DType.int32](3)
    var host = ctx.enqueue_create_host_buffer[DType.int32](3)
    host[0] = Int32(0)
    host[1] = Int32(0x12345678)
    host[2] = Int32(0)
    ctx.enqueue_memset(mutex, Int32(available))
    ctx.enqueue_copy(dst_buf=data, src_buf=host)
    ctx.enqueue_function[contend[skip_write, available, held]](
        mutex.unsafe_ptr(), data.unsafe_ptr(), grid_dim=2, block_dim=32,
    )
    ctx.enqueue_copy(dst_buf=host, src_buf=data)
    ctx.synchronize()
    var ok = host[0] == Int32(256) and host[1] == (Int32(256) ^ Int32(0x12345678)) and host[2] == Int32(0)
    print("mutex", available, "->", held, "skip-write", skip_write,
          "count", host[0], "handoff-error", host[2], "accepted", ok)
    _ = mutex^
    _ = data^
    _ = host^
    return ok


def main() raises:
    var ctx = DeviceContext()
    for _ in range(8):
        if not check[False, 0, 1](ctx):
            raise Error("mutex count or payload handoff failed")
        if not check[False, -2, -1](ctx):
            raise Error("negative-state mutex handoff failed")
    if check[True, 0, 1](ctx):
        raise Error("skip-write sabotage was not detected")
    print("PASS device_mutex: both state pairs, sabotage rejected")
