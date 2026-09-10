# SPDX-License-Identifier: Apache-2.0
"""Generate fixtures with tools/knn_zero_fma_oracle.py before running."""
from max.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from std.memory import bitcast
from checks.numerics import ftz
from neighbors.checks.pinned_distance_tile import _rt_load, _rt_step
from neighbors.checks.zero_fma_boundary import repair_zero_fma


def oracle_kernel(words: MutPointer[UInt32, MutAnyOrigin], output: MutPointer[UInt32, MutAnyOrigin], count: Int32):
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row >= Int(count):
        return
    var a = _rt_load(bitcast[DType.float32](words.unsafe_load(row * 5)))
    var b = _rt_load(bitcast[DType.float32](words.unsafe_load(row * 5 + 1)))
    var c = ftz(bitcast[DType.float32](words.unsafe_load(row * 5 + 2)))
    var v = _rt_step(a, b, c)
    output.unsafe_store(row * 2, bitcast[DType.uint32](v))
    # Simulate pre-round FTZ only where the exact oracle says the unrounded
    # result is subnormal (including zero), never for exact normal inputs.
    if words.unsafe_load(row * 5 + 4) != 0:
        var zero = bitcast[DType.float32](words.unsafe_load(row * 5 + 3) & 0x80000000)
        v = repair_zero_fma(a, b, c, zero)
    output.unsafe_store(row * 2 + 1, bitcast[DType.uint32](v))


def main() raises:
    var f = open("bench/knn_zero_fma_oracle.txt", "r")
    var text = f.read()
    f.close()
    var data = List[UInt32]()
    for line in text.splitlines():
        for word in String(line).split(" "):
            data.append(UInt32(Int(String(word))))
    var count = len(data) // 5
    with DeviceContext() as ctx:
        var host = ctx.enqueue_create_host_buffer[DType.uint32](len(data))
        var actual = ctx.enqueue_create_host_buffer[DType.uint32](count * 2)
        ctx.synchronize()
        for i in range(len(data)):
            host.unsafe_ptr().unsafe_store(i, data[i])
        var words = ctx.enqueue_create_buffer[DType.uint32](len(data))
        var output = ctx.enqueue_create_buffer[DType.uint32](count * 2)
        ctx.enqueue_copy(dst_buf=words, src_ptr=host.unsafe_ptr())
        ctx.enqueue_function[oracle_kernel](words.unsafe_ptr(), output.unsafe_ptr(), Int32(count), grid_dim=((count + 127) // 128, 1, 1), block_dim=(128, 1, 1))
        ctx.enqueue_copy(dst_ptr=actual.unsafe_ptr(), src_buf=output)
        ctx.synchronize()
        for row in range(count):
            for arm in range(2):
                var got = actual.unsafe_ptr().unsafe_load(row * 2 + arm)
                if got != data[row * 5 + 3]:
                    print("ZERO_FMA_CANDIDATE_FAIL", row, "arm", arm, data[row * 5], data[row * 5 + 1], data[row * 5 + 2], got, data[row * 5 + 3])
                    raise Error("candidate differs from independent exact integer oracle")
        print("ZERO FMA CANDIDATE PASS", "cases", count)
        _ = host^
        _ = words^
