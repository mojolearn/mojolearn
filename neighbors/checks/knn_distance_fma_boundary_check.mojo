# SPDX-License-Identifier: Apache-2.0
"""Require round-then-flush at the register distance tile's exact FMA seam.

The first four results round to the smallest normal magnitude. NVIDIA's
fma.rn.ftz instead produces zero, so this is a necessary gate even when all
ordinary distance fixtures agree (their epilogues can hide tiny differences).
"""
from max.gpu.host import DeviceContext
from std.gpu import thread_idx
from std.memory import bitcast
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz
from neighbors.checks.pinned_distance_tile import _rt_load, _rt_step


def boundary_kernel(words: MutPointer[UInt32, MutAnyOrigin], output: MutPointer[UInt32, MutAnyOrigin]):
    var row = Int(thread_idx.x)
    if row >= 8:
        return
    var a = bitcast[DType.float32](words.unsafe_load(row * 4))
    var b = bitcast[DType.float32](words.unsafe_load(row * 4 + 1))
    var c = bitcast[DType.float32](words.unsafe_load(row * 4 + 2))
    output.unsafe_store(row, bitcast[DType.uint32](_rt_step(_rt_load(a), _rt_load(b), ftz(c))))


def main() raises:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("FMA boundary gate requires IDENTICAL")
    var fixtures: List[UInt32] = [
        0x3f7fffff, 0x00800000, 0, 0x00800000,
        0xbf7fffff, 0x00800000, 0, 0x80800000,
        0x3f7fffff, 0x80800000, 0, 0x80800000,
        0xbf7fffff, 0x80800000, 0, 0x00800000,
        0x3f7ffffe, 0x00800000, 0, 0,
        0xbf7ffffe, 0x00800000, 0, 0x80000000,
        0x007fffff, 0x40000000, 0, 0,
        0x3f800000, 0x00800000, 0x80800000, 0,
    ]
    with DeviceContext() as ctx:
        var host = ctx.enqueue_create_host_buffer[DType.uint32](32)
        var actual = ctx.enqueue_create_host_buffer[DType.uint32](8)
        ctx.synchronize()
        for i in range(32):
            host.unsafe_ptr().unsafe_store(i, fixtures[i])
        var words = ctx.enqueue_create_buffer[DType.uint32](32)
        var output = ctx.enqueue_create_buffer[DType.uint32](8)
        ctx.enqueue_copy(dst_buf=words, src_ptr=host.unsafe_ptr())
        ctx.enqueue_function[boundary_kernel](words.unsafe_ptr(), output.unsafe_ptr(), grid_dim=(1, 1, 1), block_dim=(32, 1, 1))
        ctx.enqueue_copy(dst_ptr=actual.unsafe_ptr(), src_buf=output)
        ctx.synchronize()
        for row in range(8):
            var got = actual.unsafe_ptr().unsafe_load(row)
            if got != fixtures[row * 4 + 3]:
                print("DISTANCE_FMA_BOUNDARY_FAIL", row, got, fixtures[row * 4 + 3])
                raise Error("distance FMA violated round-then-flush contract")
        print("KNN DISTANCE FMA BOUNDARY PASS", "cases", 8)
        _ = host^
        _ = words^
