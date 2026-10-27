# SPDX-License-Identifier: Apache-2.0
# Testing-only probe of the production abs helper, including its FTZ boundary.
from max.gpu.host import DeviceContext
from std.gpu import thread_idx
from std.memory import bitcast
from autograd.import_ops import _abs
from checks.numerics import ftz


def kernel(src: MutPointer[UInt32, MutAnyOrigin], dst: MutPointer[UInt32, MutAnyOrigin]):
    var i = Int(thread_idx.x)
    dst[i] = bitcast[DType.uint32](ftz(_abs(ftz(bitcast[DType.float32](src[i])))))


def main() raises:
    var words: List[UInt32] = [0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff,
        0x00800000, 0x80800000, 0x7f800000, 0xff800000, 0x7fc00123, 0xffc00456,
        0x7f800123, 0xff800456, 0x3f000000, 0xbf000000]
    var ctx = DeviceContext()
    var host = ctx.enqueue_create_host_buffer[DType.uint32](len(words))
    var source = ctx.enqueue_create_buffer[DType.uint32](len(words))
    var result = ctx.enqueue_create_buffer[DType.uint32](len(words))
    ctx.synchronize()
    for i in range(len(words)):
        host.unsafe_ptr()[i] = words[i]
    ctx.enqueue_copy(dst_buf=source, src_ptr=host.unsafe_ptr())
    ctx.enqueue_function[kernel](source.unsafe_ptr(), result.unsafe_ptr(), grid_dim=1, block_dim=16)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=result)
    ctx.synchronize()
    for i in range(len(words)):
        var expected = words[i] & UInt32(0x7fffffff)
        if expected < UInt32(0x00800000):
            expected = UInt32(0)
        elif expected > UInt32(0x7f800000):
            expected = UInt32(0x7fc00000)
        var actual = host.unsafe_ptr()[i]
        print(i, hex(words[i]), hex(actual), hex(expected))
        if actual != expected:
            raise Error("abs GPU word mismatch")
        var cpu = bitcast[DType.uint32](ftz(_abs(ftz(bitcast[DType.float32](words[i])))))
        if cpu != expected:
            raise Error("abs CPU word mismatch")
    print("PASS: 16 CPU and GPU abs words")
