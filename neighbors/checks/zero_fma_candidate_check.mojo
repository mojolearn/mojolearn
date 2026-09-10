# SPDX-License-Identifier: Apache-2.0
"""Generate fixtures with tools/knn_zero_fma_oracle.py before running."""
from max.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from std.memory import bitcast
from checks.numerics import ftz
from neighbors.checks.pinned_distance_tile import _rt_load, _rt_step, _rt_accumulate_tile, RT_ROWS, RT_COLS
from neighbors.checks.zero_fma_boundary import repair_zero_fma


def oracle_kernel(words: MutPointer[UInt32, MutAnyOrigin], output: MutPointer[UInt32, MutAnyOrigin], packed: MutPointer[Float32, MutAnyOrigin], count: Int32):
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row >= Int(count):
        return
    var a = _rt_load(bitcast[DType.float32](words.unsafe_load(row * 5)))
    var b = _rt_load(bitcast[DType.float32](words.unsafe_load(row * 5 + 1)))
    var c = ftz(bitcast[DType.float32](words.unsafe_load(row * 5 + 2)))
    var v = _rt_step(a, b, c)
    output.unsafe_store(row * 3, bitcast[DType.uint32](v))
    # Simulate pre-round FTZ only where the exact oracle says the unrounded
    # result is subnormal (including zero), never for exact normal inputs.
    if words.unsafe_load(row * 5 + 4) != 0:
        var zero = bitcast[DType.float32](words.unsafe_load(row * 5 + 3) & 0x80000000)
        v = repair_zero_fma(a, b, c, zero)
    output.unsafe_store(row * 3 + 1, bitcast[DType.uint32](v))
    # The production tile's complete admission/accumulation path must also
    # satisfy the oracle. Four features compute two leading zeros, c*1, then a*b+c.
    var r = SIMD[DType.int32, RT_ROWS](0)
    var col = SIMD[DType.int32, RT_COLS](0)
    var tile = _rt_accumulate_tile(packed.unsafe_offset(row * 8), packed.unsafe_offset(row * 8 + 4), r, col, 4, 1)
    output.unsafe_store(row * 3 + 2, bitcast[DType.uint32](tile[0]))


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
        var actual = ctx.enqueue_create_host_buffer[DType.uint32](count * 3)
        var packed_host = ctx.enqueue_create_host_buffer[DType.float32](count * 8)
        ctx.synchronize()
        for i in range(len(data)):
            host.unsafe_ptr().unsafe_store(i, data[i])
        for row in range(count):
            for i in range(8):
                packed_host.unsafe_ptr().unsafe_store(row * 8 + i, Float32(0.0))
            packed_host.unsafe_ptr().unsafe_store(row * 8 + 2, bitcast[DType.float32](data[row * 5 + 2]))
            packed_host.unsafe_ptr().unsafe_store(row * 8 + 3, bitcast[DType.float32](data[row * 5]))
            packed_host.unsafe_ptr().unsafe_store(row * 8 + 6, Float32(1.0))
            packed_host.unsafe_ptr().unsafe_store(row * 8 + 7, bitcast[DType.float32](data[row * 5 + 1]))
        var packed = ctx.enqueue_create_buffer[DType.float32](count * 8)
        ctx.enqueue_copy(dst_buf=packed, src_ptr=packed_host.unsafe_ptr())
        var words = ctx.enqueue_create_buffer[DType.uint32](len(data))
        var output = ctx.enqueue_create_buffer[DType.uint32](count * 3)
        ctx.enqueue_copy(dst_buf=words, src_ptr=host.unsafe_ptr())
        ctx.enqueue_function[oracle_kernel](words.unsafe_ptr(), output.unsafe_ptr(), packed.unsafe_ptr(), Int32(count), grid_dim=((count + 127) // 128, 1, 1), block_dim=(128, 1, 1))
        ctx.enqueue_copy(dst_ptr=actual.unsafe_ptr(), src_buf=output)
        ctx.synchronize()
        for row in range(count):
            for arm in range(3):
                var got = actual.unsafe_ptr().unsafe_load(row * 3 + arm)
                if got != data[row * 5 + 3]:
                    print("ZERO_FMA_CANDIDATE_FAIL", row, "arm", arm, data[row * 5], data[row * 5 + 1], data[row * 5 + 2], got, data[row * 5 + 3])
                    raise Error("candidate differs from independent exact integer oracle")
        print("ZERO FMA CANDIDATE PASS", "cases", count, "arms", 3)
        _ = host^
        _ = words^
